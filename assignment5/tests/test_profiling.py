"""NVTX remains optional, balanced on errors, and transparent to updates."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from profiling import nvtx as profiling
from training.optim import cpu_offload as train_grpo
from training.trainers import sft as train_sft
from training import losses


@pytest.mark.parametrize("enabled,device", [(None, "cuda:0"), ("0", "cuda:0"),
                                           ("1", "cpu"), ("1", "mps")])
def test_inactive_ranges_do_not_touch_cuda(monkeypatch, enabled, device):
    if enabled is None:
        monkeypatch.delenv("CS336_NVTX", raising=False)
    else:
        monkeypatch.setenv("CS336_NVTX", enabled)

    def forbidden(*args, **kwargs):
        raise AssertionError("Inactive annotations must not query or initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda.nvtx, "range", forbidden)
    with profiling.nvtx_range("test", device):
        pass


def test_cuda_range_balances_on_error_without_synchronizing(monkeypatch):
    events = []
    monkeypatch.setenv("CS336_NVTX", "1")
    monkeypatch.setattr(torch.version, "cuda", "12.9")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    @contextmanager
    def scope(name):
        events.append(("enter", name))
        try:
            yield
        finally:
            events.append(("exit", name))

    monkeypatch.setattr(torch.cuda.nvtx, "range", scope)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: pytest.fail("Added synchronization"))
    with pytest.raises(ValueError, match="probe"):
        with profiling.nvtx_range("outer", "cuda:0"):
            with profiling.nvtx_range("inner", "cuda:0"):
                raise ValueError("probe")
    assert events == [("enter", "outer"), ("enter", "inner"),
                      ("exit", "inner"), ("exit", "outer")]


def test_annotations_preserve_accumulated_sft_and_split_offload_phases(monkeypatch):
    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(7, 4)
            self.head = torch.nn.Linear(4, 7)

        def forward(self, input_ids, attention_mask, use_cache):
            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    torch.manual_seed(5)
    reference = TinyLM()
    annotated = deepcopy(reference)
    records = [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
               {"input_ids": [2, 4, 5, 6], "labels": [-100, -100, 5, 6]}]
    ref_optimizer = train_grpo.CPUOffloadOptimizer(reference.parameters(), "adamw", lr=0.01)
    optimizer = train_grpo.CPUOffloadOptimizer(annotated.parameters(), "adamw", lr=0.01)
    monkeypatch.delenv("CS336_NVTX", raising=False)
    ref_loss, ref_metadata = train_sft.train_sft_step(reference, ref_optimizer, records, 0, 1)
    events, stack = [], []

    @contextmanager
    def scope(name, device):
        events.append((name, tuple(stack)))
        stack.append(name)
        try:
            yield
        finally:
            assert stack.pop() == name

    monkeypatch.setattr(train_sft, "nvtx_range", scope)
    monkeypatch.setattr(losses, "nvtx_range", scope)
    monkeypatch.setattr(train_grpo, "nvtx_range", scope)
    loss, metadata = train_sft.train_sft_step(annotated, optimizer, records, 0, 1)
    assert not stack
    assert torch.equal(loss, ref_loss)
    assert metadata["supervised_tokens"] == ref_metadata["supervised_tokens"]
    assert torch.equal(metadata["grad_norm"], ref_metadata["grad_norm"])
    for actual, expected in zip(annotated.parameters(), reference.parameters()):
        assert torch.equal(actual, expected)
    phases = [name for name, _ in events]
    assert phases.count("sft.forward") == phases.count("sft.backward") == 2
    offload = [(name, parents) for name, parents in events if name.startswith("optimizer.offload.")]
    assert [name for name, _ in offload] == ["optimizer.offload.grad_to_cpu",
                                           "optimizer.offload.cpu_adamw",
                                           "optimizer.offload.param_to_gpu"]
    assert all(parents == ("sft.optimizer_step",) for _, parents in offload)
