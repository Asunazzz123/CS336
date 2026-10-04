import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from train.train import evaluate
from train.training_utils import get_batch


spec = importlib.util.spec_from_file_location(
    "train_sweep", Path(__file__).resolve().parents[1] / "scripts/train_sweep.py"
)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def test_grid_has_identical_token_budgets_and_validation_intervals():
    runs = sweep.make_plan()["runs"]
    assert len(runs) == len({run["name"] for run in runs}) == 64
    for run in runs:
        assert run["max_iters"] * run["batch_size"] * 256 == 40_960_000
        assert (run["max_iters"] // 10) * run["batch_size"] * 256 == 4_096_000
        assert run["warmup_iters"] == math.ceil(run["max_iters"] * run["warmup_ratio"])
    largest = [run for run in runs if run["batch_size"] == 128 and run["lr"] == 1e-4]
    assert [run["warmup_iters"] for run in largest] == [0, 13, 38, 63]


def test_dedicated_batch_generator_preserves_global_rng():
    data = np.arange(1000, dtype=np.uint16)
    state = torch.random.get_rng_state().clone()
    a = get_batch(data, 16, 8, "cpu", generator=torch.Generator().manual_seed(1234))
    b = get_batch(data, 16, 8, "cpu", generator=torch.Generator().manual_seed(1234))
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[0][:, 1:], a[1][:, :-1])


def test_validation_batch_and_samples_are_independent_of_training_batch():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, x):
            self.inputs.append(x.clone())
            return torch.zeros(*x.shape, 10)

    model = Model()
    args = SimpleNamespace(eval_batches=2, eval_batch_size=16, batch_size=128, context_length=8, eval_seed=1234)
    data = np.arange(1000, dtype=np.uint16) % 10
    state = torch.random.get_rng_state().clone()
    evaluate(model, data, args, "cpu")
    args.batch_size = 32
    evaluate(model, data, args, "cpu")
    assert all(x.shape == (16, 8) for x in model.inputs)
    assert torch.equal(model.inputs[0], model.inputs[2])
    assert torch.equal(model.inputs[1], model.inputs[3])
    assert torch.equal(state, torch.random.get_rng_state())
    assert model.training


def make_result_files(tmp_path, val_loss=2.0, tokens=40_960_000):
    run = sweep.make_plan()["runs"][0]
    (tmp_path / "performance_summary.json").write_text(json.dumps({
        "final_iteration": run["max_iters"], "session_tokens_processed": tokens,
        "tokens_per_second": 50000, "peak_allocated_bytes": 2**30,
    }))
    (tmp_path / "config.json").write_text(json.dumps({"batch_size": run["batch_size"], "lr": run["lr"]}))
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps({"event": "validation", "val_loss": val_loss}) for _ in range(10)))
    torch.save({"model": {"weight": torch.tensor([1.0, 2.0])}, "optimizer": {}, "iteration": run["max_iters"]}, tmp_path / "final_model.pt")
    return run


def test_completed_run_keeps_inference_weights_without_optimizer(tmp_path):
    run = make_result_files(tmp_path)
    result = sweep.summarize_run(tmp_path, run, keep_optimizer=False)
    checkpoint = torch.load(result["weights_path"], weights_only=True)
    assert torch.equal(checkpoint["model"]["weight"], torch.tensor([1.0, 2.0]))
    assert "optimizer" not in checkpoint
    assert not (tmp_path / "final_model.pt").exists()
    assert result["final_val_loss"] == 2.0


@pytest.mark.parametrize("loss,tokens", [(float("nan"), 40_960_000), (2.0, 4096)])
def test_invalid_or_incomplete_runs_are_not_ranked(tmp_path, loss, tokens):
    run = make_result_files(tmp_path, loss, tokens)
    with pytest.raises(AssertionError):
        sweep.summarize_run(tmp_path, run, keep_optimizer=False)
    assert (tmp_path / "final_model.pt").exists()
