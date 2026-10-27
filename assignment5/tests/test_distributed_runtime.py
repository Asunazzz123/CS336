"""CPU contracts plus explicitly gated, real two-NVIDIA-GPU acceptance."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys

import pytest
import torch

from distributed import (FSDPConfig, GenerationConfig, HFCheckpointSink, LossOutput,
                        NvidiaContext, NvidiaFSDPEngine, Phase)
from distributed.engine import normalize_and_clip, validate_shared_parameters
from distributed.rollout import generate
from distributed.state import move_optimizer_state, move_tree
from utils.device import nvidia_cuda_available


class LocalCollectives:
    """Unit-test adapter only. The public constructor NEVER selects a CPU backend."""
    rank, world_size, device, closed = 0, 1, torch.device("cpu"), False

    def agree(self, name, value):
        pass

    def all_true(self, value):
        return bool(value)

    def sum(self, tensor):
        return tensor

    def root_call(self, fn):
        return fn()

    def barrier(self):
        pass


def local_engine(model=None):
    # Exercise isolated state/loss/checkpoint logic, not FSDP on CPU.
    engine = object.__new__(NvidiaFSDPEngine)
    engine.context = LocalCollectives()
    engine.model = model if model is not None else torch.nn.Linear(3, 2)
    engine.config = FSDPConfig(max_grad_norm=0.7)
    engine.optimizer = torch.optim.AdamW(engine.model.parameters(), lr=0.003,
                                         weight_decay=0, foreach=False, fused=False)
    engine.phase, engine.optimizer_on_gpu = Phase.READY, True
    engine.policy_version = engine.rollout_id = 0
    return engine


@pytest.mark.parametrize("kwargs", [{"master_dtype": torch.bfloat16}, {"compute_dtype": torch.float16},
                                    {"learning_rate": float("nan")}, {"max_grad_norm": 0},
                                    {"betas": (1.0, 0.9)}, {"weight_decay": -1}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        FSDPConfig(**kwargs)


def test_nvidia_required_without_silent_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        NvidiaContext.from_torchrun()


@pytest.mark.parametrize("kwargs", [{"pad_token_id": None}, {"max_new_tokens": 1.5},
                                    {"temperature": float("nan")}])
def test_invalid_generation_config(kwargs):
    with pytest.raises(ValueError):
        GenerationConfig(**kwargs)


def test_context_does_not_destroy_borrowed_process_group(monkeypatch):
    destroyed = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: destroyed.append(True))
    borrowed = NvidiaContext(0, 2, 0, torch.device("cuda", 0), None, False)
    borrowed.close()
    assert borrowed.closed and not destroyed
    owned = NvidiaContext(0, 2, 0, torch.device("cuda", 0), None, True)
    owned.close()
    owned.close()
    assert destroyed == [True]


def test_tied_parameters_must_share_a_wrap_unit():
    model = torch.nn.Module()
    model.embedding = torch.nn.Embedding(4, 4)
    model.head = torch.nn.Linear(4, 4, bias=False)
    model.head.weight = model.embedding.weight
    validate_shared_parameters(model, [])  # Common tied embedding + head at root.
    with pytest.raises(ValueError, match="tied parameter"):
        validate_shared_parameters(model, ["head"])


def test_root_io_failure_is_broadcast_and_raised_on_every_rank(monkeypatch):
    root = NvidiaContext(0, 2, 0, torch.device("cuda", 0), None, False)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="disk full"):
        root.root_call(lambda: (_ for _ in ()).throw(OSError("disk full")))
    peer = NvidiaContext(1, 2, 1, torch.device("cuda", 1), None, False)

    def receive(payload, src):
        payload[0] = (False, "OSError: disk full")

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", receive)
    with pytest.raises(RuntimeError, match="disk full"):
        peer.root_call(lambda: pytest.fail("Peer must not execute rank-0 I/O"))


def test_global_norm_uses_remote_shards_and_correct_token_scaling():
    parameter = torch.nn.Parameter(torch.zeros(1))
    parameter.grad = torch.tensor([6.0])  # Averaged, unnormalized distributed gradient.
    context = LocalCollectives()
    context.world_size = 2
    context.sum = lambda value: value.add_(16)  # Other shard's normalized gradient is 4.
    norm = normalize_and_clip([parameter], context, global_weight=4, max_norm=2)
    assert norm == 5
    torch.testing.assert_close(parameter.grad, torch.tensor([1.2]))


def test_accumulation_matches_token_weighted_reference_and_rejects_bad_update():
    torch.manual_seed(3)
    engine = local_engine()
    reference = deepcopy(engine.model)
    ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.003, weight_decay=0,
                                     foreach=False, fused=False)
    batches = [torch.randn(2, 3), torch.randn(5, 3)]

    def callback(model, batch):
        return LossOutput(model(batch).square().sum(), batch.shape[0])

    expected_loss = sum(callback(reference, x).loss_sum for x in batches) / 7
    expected_loss.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.7)
    ref_optimizer.step()
    result = engine.train_step(batches, callback)
    assert result.global_weight == 7 and result.policy_version == 1
    assert result.loss == pytest.approx(expected_loss.item())
    assert result.grad_norm == pytest.approx(expected_norm.item())
    for actual, expected in zip(engine.model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)
        assert actual.grad is None
    before = deepcopy(engine.model.state_dict())
    with pytest.raises(ValueError, match="finite"):
        engine.train_step([batches[0]], lambda m, b: LossOutput(m(b).sum() * float("nan"), 1))
    assert engine.phase == Phase.FAILED and engine.policy_version == 1
    for name, tensor in engine.model.state_dict().items():
        assert torch.equal(tensor, before[name])
    with pytest.raises(RuntimeError, match="failed"):
        engine.train_step(batches, callback)


def test_optimizer_movement_preserves_keys_scalars_and_update_continuity():
    parameter = torch.nn.Parameter(torch.tensor([0.2, 0.4]))
    optimizer = torch.optim.AdamW([parameter], foreach=False)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    original = deepcopy(optimizer.state_dict())
    move_optimizer_state(optimizer, "cpu")
    assert list(optimizer.state) == [parameter]
    assert optimizer.state[parameter]["step"].device.type == "cpu"
    for key, value in original["state"][0].items():
        assert torch.equal(optimizer.state[parameter][key], value)
    tree = {"a": [torch.ones(1), (torch.zeros(1), 3)], "b": "metadata"}
    moved = move_tree(tree, "cpu")
    assert isinstance(moved["a"][1], tuple) and moved["b"] == "metadata"
    engine = local_engine()
    engine.offload_optimizer()
    with pytest.raises(RuntimeError, match="Load optimizer"):
        engine.train_step([torch.ones(1, 3)], lambda m, x: LossOutput(m(x).sum(), 1))
    engine.load_optimizer()
    assert engine.train_step([torch.ones(1, 3)], lambda m, x: LossOutput(m(x).sum(), 1)).policy_version == 1


def test_rollout_keeps_forwarding_after_local_eos_until_all_ranks_finish():
    class Model:
        calls = 0

        def __call__(self, input_ids, attention_mask, position_ids, use_cache):
            self.calls += 1
            assert not use_cache
            torch.testing.assert_close(position_ids, (attention_mask.cumsum(-1) - 1).clamp_min(0))
            logits = torch.zeros((*input_ids.shape, 4))
            logits[..., 2] = 10  # Local EOS on the first token.
            return SimpleNamespace(logits=logits)

    class Context(LocalCollectives):
        world_size = 2

        def __init__(self):
            self.true_calls = 0

        def all_true(self, value):
            # Initial shapes/mask checks, then finite-logits and global-EOS pairs.
            self.true_calls += 1
            if self.true_calls in (4, 6):
                return False  # Another rank is still generating.
            return bool(value)

    engine = SimpleNamespace(context=Context(), config=FSDPConfig(), policy_version=8,
                             rollout_id=2, model=Model())
    result = generate(engine, torch.tensor([[0, 1]]), torch.tensor([[0, 1]]),
                      GenerationConfig(max_new_tokens=5, eos_token_id=2, pad_token_id=0))
    assert engine.model.calls == 3
    assert result.response_ids.tolist() == [[2, 0, 0]]
    assert result.response_mask.tolist() == [[True, False, False]]
    assert result.policy_version == 8 and result.rollout_id == 2


def test_invalid_rollout_mask_rejected_before_forward():
    engine = SimpleNamespace(context=LocalCollectives(), model=SimpleNamespace())
    with pytest.raises(ValueError, match="left padding"):
        generate(engine, torch.tensor([[1, 0]]), torch.tensor([[1, 0]]), GenerationConfig())


def test_weight_export_publishes_reloadable_hf_model(tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel, AutoModelForCausalLM
    model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=8, vocab_size=16, n_positions=16))
    engine = local_engine(model)
    engine.policy_version = 7
    path = tmp_path / "hf"
    engine.export_weights(HFCheckpointSink(path, model.config.save_pretrained), dtype=torch.bfloat16)
    restored = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, local_files_only=True)
    for name, tensor in restored.state_dict().items():
        assert torch.equal(tensor, model.state_dict()[name].bfloat16())
    assert json.loads((path / "policy_version.json").read_text())["policy_version"] == 7


def test_failed_weight_publish_aborts_and_blocks_future_use(tmp_path):
    engine = local_engine()
    sink = HFCheckpointSink(tmp_path / "hf", lambda directory: None)  # Missing config.json.
    with pytest.raises(ValueError, match="config.json"):
        engine.export_weights(sink)
    assert not (tmp_path / "hf").exists() and not list(tmp_path.glob("*.partial-*"))
    assert engine.phase == Phase.FAILED


def test_dcp_model_only_roundtrip_clears_stale_moments_and_does_not_overwrite(tmp_path):
    engine = local_engine()
    callback = lambda m, x: LossOutput(m(x).square().sum(), 1)
    engine.train_step([torch.ones(1, 3)], callback)
    expected = deepcopy(engine.model.state_dict())
    path = tmp_path / "checkpoint"
    engine.save_checkpoint(path, metadata={"data_cursor": 4})
    assert (path / "manifest.json").is_file()
    assert not list(tmp_path.glob("*.partial-*"))
    engine.train_step([torch.ones(1, 3)], callback)
    assert engine.load_checkpoint(path) == {"data_cursor": 4}
    assert engine.policy_version == 1 and not engine.optimizer.state
    for name, tensor in engine.model.state_dict().items():
        assert torch.equal(tensor, expected[name])
    with pytest.raises(FileExistsError):
        engine.save_checkpoint(path)
    with pytest.raises(ValueError, match="Model-only"):
        engine.load_checkpoint(path, restore_optimizer=True)
    with pytest.raises(ValueError, match="partial checkpoint"):
        engine.load_checkpoint(tmp_path / "interrupted.partial-1234")


def test_dcp_optimizer_roundtrip_repeats_the_same_next_update(tmp_path, monkeypatch):
    # DCP/AdamW continuation tested on CPU; only the CUDA RNG API is substituted.
    # Actual CUDA/FSDP tensor paths remain covered by the gated torchrun test.
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: torch.get_rng_state())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: None)
    engine = local_engine()
    batch = torch.tensor([[1.0, 2.0, 3.0]])
    callback = lambda m, x: LossOutput(m(x).square().sum(), 1)
    engine.train_step([batch], callback)
    path = tmp_path / "resume"
    engine.save_checkpoint(path, include_optimizer=True, metadata={"cursor": 1})
    engine.train_step([batch], callback)
    expected = deepcopy(engine.model.state_dict())
    assert engine.load_checkpoint(path, restore_optimizer=True) == {"cursor": 1}
    assert engine.policy_version == 1
    engine.train_step([batch], callback)
    for name, tensor in engine.model.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


@pytest.mark.skipif(not nvidia_cuda_available() or torch.cuda.device_count() < 2,
                    reason="Requires two real NVIDIA CUDA GPUs")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_real_two_gpu_fsdp_training_rollout_and_checkpoint(tmp_path, dtype):
    log = tmp_path / "torchrun.log"
    with log.open("w") as output:
        result = subprocess.run([
            sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
            "-m", "distributed.smoke", "--output-dir", str(tmp_path / "run"),
            "--compute-dtype", dtype,
        ], cwd=Path(__file__).resolve().parents[1], stdout=output, stderr=subprocess.STDOUT,
            timeout=240, env=dict(os.environ, OMP_NUM_THREADS="1"))
    assert result.returncode == 0, log.read_text()[-12000:]
    report = json.loads((tmp_path / "run/report.json").read_text())
    assert report["status"] == "passed" and len(report["ranks"]) == 2
