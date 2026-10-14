import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
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


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_mixed_precision_preserves_entire_fp32_sweep(precision):
    fp32 = sweep.make_plan(precision="fp32")
    mixed = sweep.make_plan(precision=precision)

    assert fp32["precision"] == "fp32"
    assert mixed["precision"] == precision
    assert {key: value for key, value in mixed.items() if key != "precision"} == {
        key: value for key, value in fp32.items() if key != "precision"
    }
    assert len(mixed["runs"]) == 64


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_training_commands_only_change_precision_and_output(precision, tmp_path):
    fp32 = sweep.make_plan(precision="fp32")
    mixed = sweep.make_plan(precision=precision)
    conda = "/test/miniconda/bin/conda"

    def training_options(command):
        assert command[:9] == [
            conda, "run", "--no-capture-output", "-n", "agent", "python", "-u", "-m", "train.train"
        ]
        assert command[-1] == "--fail_on_nonfinite"
        return dict(zip(command[9:-1:2], command[10:-1:2]))

    for fp32_run, mixed_run in zip(fp32["runs"], mixed["runs"], strict=True):
        fp32_output = tmp_path / "fp32" / fp32_run["name"]
        mixed_output = tmp_path / precision / mixed_run["name"]
        fp32_options = training_options(sweep.make_training_command(conda, fp32, fp32_run, fp32_output))
        mixed_options = training_options(sweep.make_training_command(conda, mixed, mixed_run, mixed_output))
        assert fp32_options.pop("--precision") == "fp32"
        assert mixed_options.pop("--precision") == precision
        assert fp32_options.pop("--output_dir") == str(fp32_output)
        assert mixed_options.pop("--output_dir") == str(mixed_output)
        assert mixed_options == fp32_options
        assert int(mixed_options["--max_iters"]) * int(mixed_options["--batch_size"]) * 256 == 40_960_000
        assert int(mixed_options["--eval_interval"]) * int(mixed_options["--batch_size"]) * 256 == 4_096_000
        assert mixed_options["--seed"] == "42"
        assert mixed_options["--eval_batch_size"] == "16"
        assert mixed_options["--eval_batches"] == "50"
        assert mixed_options["--eval_seed"] == "1234"


@pytest.mark.parametrize(
    "script,arguments,precision",
    [
        ("train_sweep.py", [], "fp32"),
        ("train_mixed_precision_sweep.py", [], "bf16"),
        ("train_mixed_precision_sweep.py", ["--precision", "fp16"], "fp16"),
    ],
)
def test_plan_only_cli_preserves_original_grid(script, arguments, precision, tmp_path):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script
    output_root = tmp_path / "unused_output"
    result = subprocess.run(
        [sys.executable, str(script_path), "--plan-only", "--output-root", str(output_root), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["precision"] == precision
    assert plan["runs"] == sweep.make_plan(precision="fp32")["runs"]
    assert plan["model"] == sweep.make_plan(precision="fp32")["model"]
    assert plan["validation"] == sweep.make_plan(precision="fp32")["validation"]
    assert not output_root.exists()


@pytest.mark.parametrize(
    "precision,exception,match",
    [("fp32", FileNotFoundError, "train.npy"), ("bf16", AssertionError, "different plan")],
)
def test_legacy_fp32_plan_only_resumes_as_fp32(tmp_path, monkeypatch, precision, exception, match):
    output_root = tmp_path / "legacy_sweep"
    output_root.mkdir()
    legacy_plan = sweep.make_plan()
    legacy_plan.pop("precision")
    legacy_plan["keep_optimizer_state"] = False
    (output_root / "sweep_plan.json").write_text(json.dumps(legacy_plan))
    monkeypatch.setattr(sweep, "ROOT", tmp_path / "missing_dataset")
    monkeypatch.setattr(sys, "argv", ["train_sweep.py", "--precision", precision, "--output-root", str(output_root)])

    # The FP32 case reaches the dataset check; BF16 must fail before it.
    with pytest.raises(exception, match=match):
        sweep.main()
    assert not (output_root / "summary.csv").exists()


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_legacy_results_are_only_summarized_as_fp32(tmp_path, precision):
    plan = sweep.make_plan(precision=precision)
    run = plan["runs"][0]
    directory = tmp_path / run["name"]
    directory.mkdir()
    legacy_result = {**run, "status": "complete", "final_val_loss": 2.0}
    (directory / "result.json").write_text(json.dumps(legacy_result))

    if precision == "bf16":
        with pytest.raises(AssertionError, match="different precision"):
            sweep.update_summary(tmp_path, plan)
        assert not (tmp_path / "summary.csv").exists()
        assert not (tmp_path / "best_run.json").exists()
    else:
        sweep.update_summary(tmp_path, plan)
        with (tmp_path / "summary.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 64
        assert rows[0]["status"] == "complete"
        assert all(row["precision"] == "fp32" for row in rows)
        assert json.loads((tmp_path / "best_run.json").read_text())["precision"] == "fp32"


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


def make_result_files(tmp_path, val_loss=2.0, tokens=40_960_000, precision=None):
    run = sweep.make_plan()["runs"][0]
    (tmp_path / "performance_summary.json").write_text(json.dumps({
        "final_iteration": run["max_iters"], "session_tokens_processed": tokens,
        "tokens_per_second": 50000, "peak_allocated_bytes": 2**30,
    }))
    config = {"batch_size": run["batch_size"], "lr": run["lr"]}
    if precision is not None:
        config["precision"] = precision
    (tmp_path / "config.json").write_text(json.dumps(config))
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


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_completed_mixed_precision_run_is_summarized(tmp_path, precision):
    run = make_result_files(tmp_path, precision=precision)
    result = sweep.summarize_run(tmp_path, run, keep_optimizer=False, precision=precision)
    assert result["final_val_loss"] == 2.0
    assert Path(result["weights_path"]).is_file()
    assert not (tmp_path / "final_model.pt").exists()


@pytest.mark.parametrize(
    "stored_precision,requested_precision",
    [(None, "bf16"), ("fp32", "bf16"), ("bf16", "fp32"), ("fp16", "bf16")],
)
def test_wrong_precision_is_rejected_before_stripping_optimizer(tmp_path, stored_precision, requested_precision):
    run = make_result_files(tmp_path, precision=stored_precision)
    with pytest.raises(AssertionError):
        sweep.summarize_run(tmp_path, run, keep_optimizer=False, precision=requested_precision)
    assert (tmp_path / "final_model.pt").is_file()
    assert not (tmp_path / "model_weights.pt").exists()


@pytest.mark.parametrize("loss,tokens", [(float("nan"), 40_960_000), (2.0, 4096)])
def test_invalid_or_incomplete_runs_are_not_ranked(tmp_path, loss, tokens):
    run = make_result_files(tmp_path, loss, tokens)
    with pytest.raises(AssertionError):
        sweep.summarize_run(tmp_path, run, keep_optimizer=False)
    assert (tmp_path / "final_model.pt").exists()
