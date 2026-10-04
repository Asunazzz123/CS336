"""Sequential 64-run TinyStories sweep with a constant training-token budget."""
from __future__ import annotations

import argparse
import csv
import fcntl
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
TOKEN_BUDGET = 40_960_000
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3)
BATCH_SIZES = (16, 32, 64, 128)
WARMUP_RATIOS = (0.0, 0.01, 0.03, 0.05)


def make_plan():
    runs = []
    for lr, batch, warmup in itertools.product(LEARNING_RATES, BATCH_SIZES, WARMUP_RATIOS):
        steps, remainder = divmod(TOKEN_BUDGET, batch * 256)
        assert not remainder
        warmup_steps = math.ceil(steps * warmup)
        runs.append({"name": f"lr{lr:.0e}_bs{batch}_warmup{warmup*100:02.0f}pct",
                     "lr": lr, "batch_size": batch, "warmup_ratio": warmup,
                     "warmup_iters": warmup_steps, "actual_warmup_ratio": warmup_steps / steps,
                     "max_iters": steps, "training_tokens": TOKEN_BUDGET})
    return {"version": 1, "model": {"vocab_size": 10000, "context_length": 256, "d_model": 512,
            "num_layers": 8, "num_heads": 8, "d_ff": 2048}, "seed": 42,
            "validation": {"batch_size": 16, "batches": 50, "seed": 1234, "evaluations": 10},
            "runs": runs}


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def update_summary(root, plan):
    rows = []
    for run in plan["runs"]:
        result = root / run["name"] / "result.json"
        rows.append(json.loads(result.read_text()) if result.exists() else {**run, "status": "pending"})
    fields = list(plan["runs"][0]) + ["status", "attempt", "final_val_loss", "best_val_loss", "perplexity",
              "wall_seconds", "tokens_per_second", "peak_allocated_GiB", "weights_path", "error"]
    temporary = root / "summary.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "summary.csv")
    successful = [row for row in rows if row["status"] == "complete"]
    if successful:
        atomic_json(root / "best_run.json", min(successful, key=lambda row: row["final_val_loss"]))


def summarize_run(output, run, keep_optimizer):
    summary = json.loads((output / "performance_summary.json").read_text())
    config = json.loads((output / "config.json").read_text())
    assert summary["final_iteration"] == run["max_iters"]
    assert summary["session_tokens_processed"] == TOKEN_BUDGET
    assert config["batch_size"] == run["batch_size"] and config["lr"] == run["lr"]
    events = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    losses = [event["val_loss"] for event in events if event["event"] == "validation"]
    assert len(losses) == 10 and all(math.isfinite(loss) for loss in losses)
    weights = output / "final_model.pt"
    if not keep_optimizer:
        # Keep every model for inference, without retaining 64 copies of AdamW state.
        import torch
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        assert checkpoint["iteration"] == run["max_iters"]
        target = output / "model_weights.pt"
        temporary = output / "model_weights.pt.tmp"
        torch.save({"model": checkpoint["model"], "iteration": checkpoint["iteration"]}, temporary)
        temporary.replace(target)
        weights.unlink()
        weights = target
    return {"final_val_loss": losses[-1], "best_val_loss": min(losses),
            "perplexity": math.exp(losses[-1]), "tokens_per_second": summary["tokens_per_second"],
            "peak_allocated_GiB": summary["peak_allocated_bytes"] / 2**30,
            "weights_path": str(weights.resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sweep_tinystories_10k"))
    parser.add_argument("--conda-exe", default=shutil.which("conda") or "/root/miniconda3/bin/conda")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--keep-optimizer-state", action="store_true",
                        help="Retain full resumable checkpoints (~32 GiB for 64 runs)")
    args = parser.parse_args()
    plan = make_plan()
    plan["keep_optimizer_state"] = args.keep_optimizer_state
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".sweep.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another sweep is already using this output directory")
        plan_path = output_root / "sweep_plan.json"
        if plan_path.exists():
            assert json.loads(plan_path.read_text()) == plan, "Existing sweep has a different plan"
        else:
            atomic_json(plan_path, plan)
        for filename in ("train.npy", "valid.npy"):
            if not (ROOT / "data/encoded/tinystories-10k" / filename).is_file():
                raise FileNotFoundError(filename)
        update_summary(output_root, plan)
        for index, run in enumerate(plan["runs"], 1):
            directory = output_root / run["name"]
            directory.mkdir(exist_ok=True)
            result_path = directory / "result.json"
            if result_path.exists():
                previous = json.loads(result_path.read_text())
                if previous["status"] == "complete" or (previous["status"] == "failed" and not args.retry_failed):
                    print(f"[{index}/64] Skip {run['name']}: {previous['status']}", flush=True)
                    continue
            attempt = 1
            while (directory / f"attempt_{attempt:03d}").exists():
                attempt += 1
            output = directory / f"attempt_{attempt:03d}"
            output.mkdir()
            command = [args.conda_exe, "run", "--no-capture-output", "-n", "agent", "python", "-u", "-m", "train.train",
                       "--data_path", "data/encoded/tinystories-10k/train.npy",
                       "--val_data_path", "data/encoded/tinystories-10k/valid.npy"]
            settings = {**plan["model"], "batch_size": run["batch_size"], "max_iters": run["max_iters"],
                        "lr": run["lr"], "warmup_iters": run["warmup_iters"], "weight_decay": 0.1,
                        "grad_clip": 1.0, "seed": 42, "eval_batch_size": 16, "eval_batches": 50,
                        "eval_seed": 1234, "eval_interval": run["max_iters"] // 10,
                        "log_interval": max(1, run["max_iters"] // 100),
                        "save_interval": run["max_iters"] + 1, "probe_warmup_steps": 10,
                        "output_dir": str(output)}
            for key, value in settings.items():
                command.extend((f"--{key}", str(value)))
            command.append("--fail_on_nonfinite")
            result = {**run, "status": "running", "attempt": attempt}
            atomic_json(result_path, result)
            update_summary(output_root, plan)
            print(f"[{index}/64] Start {run['name']}, steps={run['max_iters']}, warmup={run['warmup_iters']}", flush=True)
            started = time.perf_counter()
            try:
                with (output / "train.log").open("w") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                result.update(summarize_run(output, run, args.keep_optimizer_state))
                result["status"] = "complete"
            except (Exception, KeyboardInterrupt) as error:
                result.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                              error=f"{type(error).__name__}: {error}; see {output / 'train.log'}")
                if isinstance(error, KeyboardInterrupt):
                    atomic_json(result_path, result)
                    raise
            finally:
                result["wall_seconds"] = time.perf_counter() - started
                atomic_json(result_path, result)
                update_summary(output_root, plan)
            print(f"[{index}/64] {result['status']}: {run['name']}", flush=True)
        print(f"Sweep finished. Results: {output_root / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
