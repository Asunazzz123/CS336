"""CPU algorithm/transaction contracts; CUDA acceptance is explicitly gated."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import socket
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

from training.backends.fsdp_grpo_update import FSDPGRPOTrainer, batch_from_rollout, policy_log_probs
from alignment.grpo import run_compute_policy_gradient_loss
from distributed import DistributedBestCheckpoint, RolloutResult, Phase
from training.configs import UnifiedGRPOConfig
from training.trainers.grpo import train
from training.backends.fsdp_grpo import validate_fsdp_config, evaluate
from checkpointing.best import validate_hf_checkpoint
from utils.device import nvidia_cuda_available
from .test_distributed_runtime import local_engine, LocalCollectives


def tiny_model():
    torch.manual_seed(6)
    return GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=8, vocab_size=16,
                                     n_positions=32, pad_token_id=0, eos_token_id=1,
                                     resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))


def batch():
    # Left padding, distinct response lengths, actual EOS == pad still counted
    # by the explicit response mask rather than an ID equality heuristic.
    prompts = torch.tensor([[0, 2, 3], [0, 2, 3], [4, 5, 6], [4, 5, 6]])
    prompt_mask = torch.tensor([[0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1]])
    rollout = RolloutResult(torch.tensor([[7, 1, 0], [8, 9, 1], [7, 1, 0], [8, 9, 1]]),
                            torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1]]).bool(), 0, 0)
    return batch_from_rollout(prompts, prompt_mask, rollout, [1., 0., 0., 1.], group_size=2,
                              advantage_normalizer="none")


@pytest.mark.parametrize("normalization", ["sequence", "constant"])
@pytest.mark.parametrize("method", ["none", "grpo", "gspo", "noclip"])
def test_fsdp_adapter_matches_unsharded_reference_across_epochs(normalization, method):
    engine = local_engine(tiny_model())
    reference = deepcopy(engine.model)
    optimizer = torch.optim.AdamW(reference.parameters(), lr=0.003, weight_decay=0,
                                 foreach=False, fused=False)
    rollout = batch()
    trainer = FSDPGRPOTrainer(engine, num_epochs_per_rollout=3, gradient_accumulation_steps=2,
                              importance_reweighting_method=method, loss_normalization=normalization)
    reference.eval()
    old = policy_log_probs(reference, rollout).detach() if method != "none" else None
    expected_losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        per_token, _ = run_compute_policy_gradient_loss(rollout.advantages,
            policy_log_probs(reference, rollout), method, old, 0.2, rollout.response_mask)
        masked = per_token * rollout.response_mask
        loss = ((masked.sum(-1) / rollout.response_mask.sum(-1)).mean() if normalization == "sequence"
                else masked.sum() / rollout.response_mask.sum())
        expected_losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), engine.config.max_grad_norm)
        optimizer.step()
    actual = trainer.update(rollout, policy_version=0)
    assert [r.policy_version for r in actual] == [1, 2, 3]
    assert [r.loss for r in actual] == pytest.approx(expected_losses, abs=1e-6)
    for name, tensor in reference.state_dict().items():
        # AdamW amplifies roundoff in near-zero attention-key bias gradients
        # when comparing full-batch vs microbatch FP32 reductions.
        torch.testing.assert_close(engine.model.state_dict()[name], tensor, atol=1e-5, rtol=1e-5)


def test_on_policy_skips_old_pass_and_rejects_stale_rollouts(monkeypatch):
    engine = local_engine(tiny_model())
    trainer = FSDPGRPOTrainer(engine, gradient_accumulation_steps=2)
    from training.backends import fsdp_grpo_update as fsdp_grpo
    real = fsdp_grpo.policy_log_probs
    calls = []
    def score(*args):
        calls.append(torch.is_grad_enabled())
        return real(*args)
    monkeypatch.setattr(fsdp_grpo, "policy_log_probs", score)
    trainer.update(batch(), policy_version=0)
    assert calls == [True, True]
    with pytest.raises(ValueError, match="Stale policy"):
        trainer.update(batch(), policy_version=0)


def test_collective_best_skips_losers_exports_selected_weights_and_no_optimizer(tmp_path, monkeypatch):
    engine = local_engine(tiny_model())
    original_config = deepcopy(engine.model.config.to_dict())
    manager = DistributedBestCheckpoint(engine, tmp_path / "cache")
    monkeypatch.setattr(engine.optimizer, "state_dict", lambda: pytest.fail("Optimizer serialized"))
    assert manager.consider(0, 0.5)
    expected = deepcopy(engine.model.state_dict())
    def unexpected_export(*args, **kwargs):
        pytest.fail("Non-improving score gathered model weights")
    with monkeypatch.context() as context:
        context.setattr(engine, "export_weights", unexpected_export)
        assert not manager.consider(1, 0.5)
        assert not manager.consider(2, 0.4)
    with torch.no_grad():
        for parameter in engine.model.parameters():
            parameter.add_(1)
    path = manager.export(tmp_path / "best")
    restored = AutoModelForCausalLM.from_pretrained(path, dtype="auto", local_files_only=True)
    assert next(restored.parameters()).dtype == torch.bfloat16
    for name, value in restored.state_dict().items():
        assert torch.equal(value, expected[name].bfloat16())
    assert engine.model.config.to_dict() == original_config
    assert manager.selection()["step"] == 0
    assert not list(path.rglob("*.pt"))
    assert len(list((tmp_path / "cache").glob(".checkpoint-*"))) == 1
    manager.cleanup()
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("mode,scores", [("max", [0.2, 0.8]), ("min", [0.8, 0.2])])
def test_best_replacement_and_failed_candidate_keep_previous_best(tmp_path, mode, scores):
    engine = local_engine(tiny_model())
    manager = DistributedBestCheckpoint(engine, tmp_path / "cache", mode=mode)
    assert manager.consider(0, scores[0])
    assert manager.consider(1, scores[1])
    previous = manager.store.best_path.resolve()
    manager.save_assets = lambda path: (_ for _ in ()).throw(OSError("disk full"))
    with pytest.raises(OSError, match="disk full"):
        manager.consider(2, 1.0 if mode == "max" else 0.0)
    assert engine.phase == Phase.FAILED
    assert manager.store.best_path.resolve() == previous
    validate_hf_checkpoint(previous)
    assert not list((tmp_path / "cache").glob(".export-*"))
    assert manager.selection()["step"] == 1


def test_existing_dcp_can_load_then_export_hf(tmp_path):
    engine = local_engine(tiny_model())
    engine.save_checkpoint(tmp_path / "dcp", metadata={"source": "test"})
    expected = deepcopy(engine.model.state_dict())
    with torch.no_grad():
        next(engine.model.parameters()).add_(1)
    engine.load_checkpoint(tmp_path / "dcp")
    path = engine.export_model_only_hf(tmp_path / "hf", dtype=torch.float32)
    restored = AutoModelForCausalLM.from_pretrained(path, dtype="auto", local_files_only=True)
    for name, value in restored.state_dict().items():
        assert torch.equal(value, expected[name])


def test_fsdp_constraints_and_whole_group_partition():
    config = SimpleNamespace(generation_backend="transformers", optimizer_name="adamw",
        optimizer_device="model", dtype="bfloat16", prompts_per_batch=4, group_size=2,
        gradient_accumulation_steps=2)
    validate_fsdp_config(config, 2)
    for field, value in (("generation_backend", "unknown"), ("optimizer_device", "cpu"),
                         ("dtype", "float16"), ("prompts_per_batch", 3),
                         ("gradient_accumulation_steps", 3)):
        with pytest.raises(ValueError):
            validate_fsdp_config(SimpleNamespace(**{**vars(config), field: value}), 2)


def test_evaluation_excludes_duplicate_padding_on_short_rank(monkeypatch):
    from training.backends import fsdp_grpo as runtime
    engine = local_engine(tiny_model())
    engine.context.rank, engine.context.world_size = 1, 2
    # Rank 1 has no real examples when evaluation size is 1. Only remote totals count.
    def reduce(totals):
        assert totals.tolist() == [0, 0, 0, 0, 0]
        totals.add_(torch.tensor([1., 1., 1., 2., 1.]))
    engine.context.sum = reduce
    monkeypatch.setattr(runtime, "generate", lambda *a, **k: (None, None,
        SimpleNamespace(response_mask=torch.ones((1, 2))), ["answer"]))
    metrics = evaluate(engine, None, [SimpleNamespace(question="q", ground_truth="a")], "{question}",
                       None, lambda *args: {"reward": 1, "answer_reward": 1, "format_reward": 1})
    assert metrics["accuracy"] == 1 and metrics["eval_examples"] == 1


@pytest.mark.parametrize("backend,generation", [("transformers", "transformers"),
    ("fsdp2", "transformers"), ("fsdp2", "vllm")])
def test_unified_entry_runs_local_tiny_training_and_best_export(tmp_path, monkeypatch, backend, generation):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from training.backends import fsdp_grpo as runtime
    model_path = tmp_path / "model"
    model = tiny_model()
    model.generation_config.suppress_tokens = [0, 2]
    model.save_pretrained(model_path)
    words = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "q": 3, "v": 4}
    words.update({f"token{i}": i for i in range(5, 16)})
    tokens = Tokenizer(WordLevel(words, unk_token="<unk>"))
    tokens.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokens, pad_token="<pad>",
                                        eos_token="<eos>", unk_token="<unk>")
    tokenizer.save_pretrained(model_path)
    training, validation, prompt = (tmp_path / p for p in ("train.jsonl", "val.jsonl", "prompt.txt"))
    training.write_text("\n".join(json.dumps({"question": q, "answer": "x #### 1"})
                                  for q in ("q", "q q")) + "\n")
    validation.write_text(json.dumps({"question": "v", "answer": "x #### 2"}) + "\n")
    prompt.write_text("{question}")
    config = UnifiedGRPOConfig(str(model_path), training, validation, prompt, tmp_path / "out",
        dtype="float32", num_steps=1, prompts_per_batch=1, group_size=2,
        gradient_accumulation_steps=2, max_new_tokens=2, eval_examples=1, eval_every=1,
        num_epochs_per_rollout=2, best_checkpoint_dir=tmp_path / "cache", training_backend=backend,
        generation_backend=generation, rollout_checkpoint_dir=tmp_path / "rollouts")
    external_versions = []
    if backend == "fsdp2":
        class Context(LocalCollectives):
            world_size = 2 if generation == "vllm" else 1
            def gather_objects(self, value):
                return [value] * self.world_size
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        monkeypatch.setattr(runtime.NvidiaContext, "from_torchrun", lambda **kwargs: Context())
        def build(ctx, factory, fsdp_config):
            engine = local_engine(factory())
            engine.context, engine.config = ctx, fsdp_config
            if generation == "vllm":
                from distributed import Phase
                engine.park = lambda: setattr(engine, "phase", Phase.PARKED)
                engine.resume = lambda: setattr(engine, "phase", Phase.READY)
            return engine
        monkeypatch.setattr(runtime, "NvidiaFSDPEngine", build)
    if generation == "vllm":
        from generation import vllm
        config.prompts_per_batch = 2
        class Backend:
            def __init__(self, **kwargs):
                pass
            def load_policy(self, snapshot):
                validate_hf_checkpoint(snapshot.path)
                external_versions.append(snapshot.version)
                return snapshot.version
            def generate(self, prompts, params):
                return [vllm.VLLMCompletion("q", [3, 1], "stop") for _ in prompts]
            def close(self):
                pass
        monkeypatch.setattr(vllm, "VLLMBackend", Backend)
    train(config)
    assert not config.best_checkpoint_dir.exists()
    validate_hf_checkpoint(config.output_dir / "checkpoint-best")
    records = [json.loads(line) for line in (config.output_dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["split"] for r in records] == ["eval", "train", "eval"]
    if generation == "vllm":
        assert external_versions == [0, 0, 2]
        assert not list(config.rollout_checkpoint_dir.glob("policy-*"))


def test_real_two_process_dtensor_best_bridge(tmp_path):
    # Static IPv4 rendezvous avoids macOS reverse-DNS returning an ip6.arpa
    # hostname for localhost in the elastic standalone rendezvous implementation.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run",
        "--master-addr=127.0.0.1", f"--master-port={port}",
        "--nproc-per-node=2", "-m", "tests.fsdp_bridge_worker", "--output-dir", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        timeout=45, env=dict(os.environ, OMP_NUM_THREADS="1", GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo"))
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
    assert (tmp_path / "passed.json").is_file()


@pytest.mark.skipif(not nvidia_cuda_available() or torch.cuda.device_count() < 2,
                    reason="Requires two real NVIDIA CUDA GPUs")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_real_two_gpu_grpo_and_best_bridge(tmp_path, dtype):
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc-per-node=2", "-m", "tests.fsdp_bridge_worker", "--output-dir", str(tmp_path),
        "--cuda", "--compute-dtype", dtype], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=180, env=dict(os.environ, OMP_NUM_THREADS="1"))
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]
    assert (tmp_path / "passed.json").is_file()
