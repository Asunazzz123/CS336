"""Validate one unsaved FP32/AMP/CPU-AdamW update and optional vLLM sync.

This is an environment acceptance check, not an experiment run. It reads an
already downloaded local model, uses two short hand-written responses, and
never saves weights or optimizer state. Its JSON report is the only artifact.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alignment.grpo import run_get_response_log_probs, run_tokenize_prompt_and_output
from alignment import vllm_utils
from scripts.train_grpo import (
    CPUOffloadOptimizer, Question, RolloutBatch, TrainConfig, build_runtime,
    close_runtime, generate_rollouts, json_value, seed_everything, sync_generator,
    train_on_rollouts,
)


def _nvidia_smi():
    fields = "index,uuid,name,memory.used,memory.total"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=10, check=True,
        )
        return [dict(zip(fields.split(","), [field.strip() for field in row]))
                for row in csv.reader(result.stdout.splitlines())]
    except (OSError, subprocess.SubprocessError) as error:
        return {"unavailable": str(error)}


class GPUMemoryMonitor:
    """Sample device-wide usage, which includes the separate vLLM process.

    Torch allocator statistics in this process cannot measure vLLM's GPU1
    allocations. These sampled device peaks are observations, not exact peaks.
    """

    def __init__(self, interval=1.0):
        self.interval = interval
        self.peaks = {}
        self.samples = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            rows = _nvidia_smi()
            if isinstance(rows, list):
                for row in rows:
                    try:
                        used = float(row["memory.used"])
                    except ValueError:
                        continue
                    self.peaks[row["uuid"]] = max(self.peaks.get(row["uuid"], 0), used)
                self.samples += 1
            self.stop_event.wait(self.interval)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=12)


def _memory_snapshot():
    snapshot = {"nvidia_smi_devices": _nvidia_smi(), "torch_allocator": []}
    for index in range(torch.cuda.device_count()):
        snapshot["torch_allocator"].append({
            "device": index,
            "allocated_bytes": torch.cuda.memory_allocated(index),
            "reserved_bytes": torch.cuda.memory_reserved(index),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
        })
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            name, _, value = line.partition(":")
            if name in ("VmRSS", "VmHWM"):
                snapshot[name + "_bytes"] = int(value.split()[0]) * 1024
    else:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        snapshot["VmHWM_bytes"] = peak if sys.platform == "darwin" else peak * 1024
    snapshot["cgroup"] = {}
    for filename in ("memory.current", "memory.peak", "memory.max"):
        path = Path("/sys/fs/cgroup") / filename
        if path.exists():
            value = path.read_text().strip()
            snapshot["cgroup"][filename] = int(value) if value.isdigit() else value
    return snapshot


def _cpu_optimizer_summary(optimizer):
    if not isinstance(optimizer, CPUOffloadOptimizer):
        raise AssertionError("The check requires CPUOffloadOptimizer")
    for master in optimizer.master_parameters:
        assert master.device.type == "cpu" and master.dtype == torch.float32
    tensors = [value for state in optimizer.optimizer.state.values()
               for value in state.values() if isinstance(value, torch.Tensor)]
    for value in tensors:
        assert value.device.type == "cpu" and value.dtype == torch.float32
    return {
        "master_parameter_tensors": len(optimizer.master_parameters),
        "master_parameter_bytes": sum(p.numel() * p.element_size()
                                      for p in optimizer.master_parameters),
        "optimizer_state_parameter_count": len(optimizer.optimizer.state),
        "optimizer_state_tensor_count": len(tensors),
        "optimizer_state_bytes": sum(t.numel() * t.element_size() for t in tensors),
        "all_master_and_state_tensors_cpu_float32": True,
    }


def _probe_parameter(model):
    candidates = [(name, parameter) for name, parameter in model.named_parameters()
                  if parameter.requires_grad and 0 < parameter.numel() <= 16384]
    if not candidates:
        raise AssertionError("No small trainable parameter available for the update check")
    for name, parameter in reversed(candidates):
        if name.endswith(("model.norm.weight", "ln_f.weight", "final_layernorm.weight")):
            return name, parameter
    return candidates[-1]


def _check_gradients(model):
    finite = torch.ones((), dtype=torch.bool, device=next(model.parameters()).device)
    count, elements = 0, 0
    # Bound diagnostic memory even for the vocabulary embedding/output matrices.
    for parameter in model.parameters():
        if parameter.grad is not None:
            count += 1
            elements += parameter.grad.numel()
            for chunk in parameter.grad.detach().reshape(-1).split(16 * 1024 * 1024):
                finite.logical_and_(torch.isfinite(chunk).all())
    assert count > 0, "No parameter received a gradient"
    assert bool(finite.item()), "Non-finite model gradient"
    return {"gradient_parameter_count": count, "gradient_elements": elements,
            "all_gradients_finite": True}


def _score_responses(runtime, rollouts):
    tokenized = run_tokenize_prompt_and_output(
        rollouts.repeated_prompts, rollouts.responses, runtime.tokenizer,
    )
    device = next(runtime.model.parameters()).device
    previous_mode = runtime.model.training
    runtime.model.eval()
    means = []
    try:
        with torch.no_grad():
            for index in range(len(rollouts.responses)):
                scores = run_get_response_log_probs(
                    model=runtime.model,
                    input_ids=tokenized["input_ids"][index:index + 1].to(device),
                    labels=tokenized["labels"][index:index + 1].to(device),
                    return_token_entropy=True, autocast_dtype=torch.bfloat16,
                )
                assert all(bool(torch.isfinite(value).all().item()) for value in scores.values())
                mask = tokenized["response_mask"][index:index + 1].to(device)
                means.append(float(scores["log_probs"][mask].mean().item()))
    finally:
        runtime.model.train(previous_mode)
    return {"finite": True, "response_mean_log_probs": means,
            "input_shape": list(tokenized["input_ids"].shape),
            "response_tokens": int(tokenized["response_mask"].sum().item())}


def _verify_weight_transfer(runtime, config):
    """Prove receipt on GPU1 using temporary BF16-visible sentinels, then restore."""
    names = ["language_model.model.embed_tokens.weight", "language_model.lm_head.weight"]
    tensors = [runtime.model.get_input_embeddings().weight,
               runtime.model.get_output_embeddings().weight]
    unique = list({id(tensor): tensor for tensor in tensors}.values())
    original = [tensor[0, 0].detach().clone() for tensor in unique]

    def inspect():
        response = vllm_utils._http_json(
            "POST", runtime.generator.base_url + "/collective_rpc",
            {"method": "weight_sync_probe", "kwargs": {"names": json.dumps(names)}, "timeout": 30},
            timeout=40,
        )
        assert len(response["results"]) == 1
        result = response["results"][0]
        for name, tensor in zip(names, tensors):
            expected = tensor[0, :4].detach().to(torch.bfloat16).float().cpu().tolist()
            assert result[name]["values"] == expected, f"vLLM received different values for {name}"
            assert result[name]["shape"] == list(tensor.shape)
        return result

    baseline = inspect()
    try:
        with torch.no_grad():
            for i, tensor in enumerate(unique):
                tensor[0, 0].add_(0.125 * (i + 1))
        sync_generator(runtime, config)
        changed = inspect()
        assert all(changed[name]["values"][0] != baseline[name]["values"][0] for name in names)
    finally:
        with torch.no_grad():
            for tensor, value in zip(unique, original):
                tensor[0, 0].copy_(value)
    sync_generator(runtime, config)
    restored = inspect()
    assert restored == baseline
    return {"worker_samples_before": baseline, "worker_samples_sentinel": changed,
            "worker_samples_restored": restored, "restored_original_policy": True}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Existing local text-only model directory; never downloaded here")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--generation-backend", choices=["transformers", "vllm"], default="vllm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vllm-gpu", type=int, default=1)
    parser.add_argument("--vllm-port", type=int, default=8017)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--skip-update", action="store_true",
                        help="Only load, synchronize, score and generate; no backward/optimizer step")
    parser.add_argument("--verify-weight-transfer", action="store_true",
                        help="Read worker parameter samples through an optional RPC and verify restored sentinels")
    args = parser.parse_args()
    args.model_path = args.model_path.expanduser().resolve()
    args.report = args.report.expanduser().resolve()
    if not args.model_path.is_dir() or not (args.model_path / "config.json").is_file():
        parser.error("--model-path must be an existing local model directory containing config.json")
    if args.report == args.model_path or args.model_path in args.report.parents:
        parser.error("--report must be outside the model directory")
    if args.report.suffix != ".json" or args.report.exists():
        parser.error("--report must be a new .json file")
    if not 1 <= args.max_new_tokens <= 32:
        parser.error("--max-new-tokens must be between 1 and 32 for this short acceptance check")
    if args.verify_weight_transfer and args.generation_backend != "vllm":
        parser.error("--verify-weight-transfer requires the vllm backend")
    return args


def main():
    args = parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "purpose": "one unsaved environment acceptance update; not an experiment",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args), "status": "running", "stages": [], "versions": {},
        "updates_executed": 0,
        "allocator_note": "Torch allocator data covers this process; use nvidia-smi for vLLM GPU usage",
    }
    for package in ("torch", "transformers", "vllm", "triton"):
        try:
            report["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["versions"][package] = None
    started = time.monotonic()
    monitor = GPUMemoryMonitor()
    runtime = None
    exit_code = 0

    def write_report():
        temporary = args.report.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(json_value(report), indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(args.report)

    def stage(name, details=None):
        entry = {"name": name, "elapsed_seconds": time.monotonic() - started,
                 "memory": _memory_snapshot()}
        if details is not None:
            entry["details"] = details
        report["stages"].append(entry)
        write_report()
        print(json.dumps(json_value({"stage": name, "elapsed_seconds": entry["elapsed_seconds"],
                                     "details": details}), allow_nan=False), flush=True)

    try:
        if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
            raise RuntimeError("This acceptance check requires a CUDA training device")
        seed_everything(42)
        if args.verify_weight_transfer:
            os.environ["CS336_VLLM_WEIGHT_PROBE"] = "1"
        for index in range(torch.cuda.device_count()):
            # reset_peak_memory_stats needs the allocator initialized on each
            # device, not just the CUDA driver context created by is_available.
            allocation = torch.empty(1, dtype=torch.uint8, device=f"cuda:{index}")
            del allocation
            torch.cuda.reset_peak_memory_stats(index)
        monitor.start()
        stage("before_load")
        config = TrainConfig(
            model_id=str(args.model_path), train_path=Path("unused"), eval_path=Path("unused"),
            prompt_path=Path("unused"), output_dir=args.report.parent,
            device=args.device, dtype="float32", autocast="bfloat16",
            gradient_checkpointing=True, optimizer_name="adamw", optimizer_device="cpu",
            generation_backend=args.generation_backend, vllm_gpu=args.vllm_gpu,
            vllm_port=args.vllm_port, learning_rate=1e-6, weight_decay=0.0,
            prompts_per_batch=1, group_size=2, gradient_accumulation_steps=2,
            max_new_tokens=args.max_new_tokens, generation_batch_size=1,
            temperature=0.0, save_optimizer_state=False,
            advantage_normalizer="none", loss_normalization="constant",
        )
        report["runtime_config"] = vars(config)
        runtime = build_runtime(config)
        assert all(parameter.dtype == torch.float32 for parameter in runtime.model.parameters())
        assert runtime.model.is_gradient_checkpointing
        stage("loaded", {"parameter_count": sum(p.numel() for p in runtime.model.parameters()),
                         "optimizer": _cpu_optimizer_summary(runtime.optimizer)})
        rollouts = RolloutBatch(["1+1="] * 2, [r"\boxed{2}", r"\boxed{3}"], ["2"] * 2)
        questions = [Question("What is 1 + 1?", "2")]

        def synchronize_and_generate(label):
            began = time.monotonic()
            sync_generator(runtime, config)
            generated = generate_rollouts(
                runtime, questions, "Question: {question}\nAnswer:", replace(config, group_size=1),
            )
            torch.cuda.synchronize(torch.device(args.device))
            stage(label, {"seconds": time.monotonic() - began,
                          "responses": generated.responses,
                          "scores": _score_responses(runtime, rollouts)})

        synchronize_and_generate("before_update_sync_and_generation")
        if args.verify_weight_transfer:
            stage("verified_cross_gpu_weight_transfer", _verify_weight_transfer(runtime, config))
        if not args.skip_update:
            probe_name, probe = _probe_parameter(runtime.model)
            probe_before = probe.detach().cpu().clone()
            original_step = runtime.optimizer.step

            def checked_step():
                gradients = _check_gradients(runtime.model)
                stage("after_backward_before_optimizer", gradients)
                began = time.monotonic()
                result = original_step()
                report["updates_executed"] = 1
                stage("after_cpu_optimizer_before_zero_grad", {
                    "optimizer_seconds": time.monotonic() - began,
                    "optimizer": _cpu_optimizer_summary(runtime.optimizer),
                })
                return result

            runtime.optimizer.step = checked_step
            began = time.monotonic()
            loss, metadata = train_on_rollouts(runtime, rollouts, config)
            torch.cuda.synchronize(torch.device(args.device))
            update_seconds = time.monotonic() - began
            assert bool(torch.isfinite(loss).item()), "Non-finite policy loss"
            grad_norm = float(metadata["grad_norm"].item())
            assert math.isfinite(grad_norm) and grad_norm > 0, "Expected a finite nonzero gradient"
            assert float(metadata["mean_reward"]) == 0.5
            assert all(parameter.grad is None for parameter in runtime.model.parameters())
            probe_after = probe.detach().cpu()
            delta = probe_after - probe_before
            assert bool(torch.isfinite(probe_after).all().item())
            assert bool(torch.count_nonzero(delta).item()), "Expected a real FP32 parameter update"
            stage("updated", {"seconds": update_seconds, "loss": loss, "metadata": metadata,
                              "probe_parameter": probe_name, "probe_elements": probe.numel(),
                              "probe_changed_elements": int(torch.count_nonzero(delta).item()),
                              "probe_max_abs_change": float(delta.abs().max().item()),
                              "probe_dtype": str(probe.dtype),
                              "optimizer": _cpu_optimizer_summary(runtime.optimizer)})
            synchronize_and_generate("after_update_sync_and_generation")
        report["status"] = "passed"
    except BaseException as error:
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
        traceback.print_exc()
    finally:
        if runtime is not None:
            try:
                close_runtime(runtime)
                report["runtime_closed"] = True
            except BaseException as error:
                report["runtime_closed"] = False
                report["cleanup_error"] = repr(error)
                report["status"] = "failed"
                exit_code = exit_code or 1
        if monitor.thread.ident is not None:
            monitor.stop()
        report["gpu_memory_sampling"] = {
            "interval_seconds": monitor.interval, "samples": monitor.samples,
            "peak_observed_used_mib_by_gpu_uuid": monitor.peaks,
            "note": "Sampled device-wide usage includes other processes and can miss brief peaks",
        }
        report["total_seconds"] = time.monotonic() - started
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["final_memory"] = _memory_snapshot()
        write_report()
        print(f"Runtime check: {report['status']}; JSON report: {args.report}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
