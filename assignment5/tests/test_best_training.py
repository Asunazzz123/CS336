"""Best-model retention selects validation checkpoints, not the final live policy."""
from dataclasses import replace
import json
import logging
import math
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts import train_grpo, train_grpo_full, train_sft
from scripts.train_grpo import RolloutBatch
from .test_train_grpo import config
from .test_train_sft import tiny_sft_config


def frozen_state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def assert_best_export(output_dir, expected_state, step, score):
    checkpoint = output_dir / "checkpoint-best"
    assert list(output_dir.glob("checkpoint-*")) == [checkpoint]
    assert not list(checkpoint.rglob("*.pt"))
    assert not list(checkpoint.rglob("*optimizer*"))
    selection = json.loads((checkpoint / "selection.json").read_text())
    assert selection["step"] == step
    assert selection["score"] == score
    assert selection["metadata"]["checkpoint_dtype"] == "bfloat16"
    assert selection["metadata"]["metric_model"] == "live_policy_before_export"
    assert selection["metadata"]["optimizer_saved"] is False
    assert selection["metadata"]["selection_split"] == "validation"
    assert json.loads((output_dir / "best_checkpoint.json").read_text())["step"] == step
    reloaded = AutoModelForCausalLM.from_pretrained(checkpoint, dtype="auto", local_files_only=True)
    assert next(reloaded.parameters()).dtype == torch.bfloat16
    for name, value in reloaded.state_dict().items():
        assert torch.equal(value, expected_state[name].to(torch.bfloat16))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    assert tokenizer.eos_token_id is not None
    return checkpoint


@pytest.mark.parametrize("scores,best_step", [([3.0, 1.0, 2.0], 1), ([1.0, 2.0, 3.0], 0)])
def test_sft_best_retains_the_selected_state_and_only_one_ssd_model(
    tiny_sft_config, tmp_path, monkeypatch, scores, best_step,
):
    cfg = replace(tiny_sft_config, best_checkpoint_dir=tmp_path / "ram-cache")
    original_build = train_sft.build_runtime
    captured = {}

    def build(runtime_config):
        runtime = original_build(runtime_config)
        captured["runtime"] = runtime
        monkeypatch.setattr(runtime.optimizer, "state_dict", lambda: pytest.fail("Optimizer was serialized"))
        return runtime

    evaluations = []

    def evaluate(model, records, *args):
        index = len(evaluations)
        evaluations.append(frozen_state(model))
        return {"loss": scores[index], "perplexity": math.exp(scores[index]),
                "num_examples": len(records), "supervised_tokens": 1}

    monkeypatch.setattr(train_sft, "build_runtime", build)
    monkeypatch.setattr(train_sft, "evaluate_sft", evaluate)
    checkpoint = train_sft.train(cfg)
    assert checkpoint == cfg.output_dir / "checkpoint-best"
    assert len(evaluations) == 3
    assert any(not torch.equal(evaluations[best_step][name], evaluations[-1][name])
               for name in evaluations[best_step])
    # Later updates and caller-side live mutations must not modify the frozen best.
    with torch.no_grad():
        for parameter in captured["runtime"].model.parameters():
            parameter.fill_(123)
    assert_best_export(cfg.output_dir, evaluations[best_step], best_step, scores[best_step])
    assert not cfg.best_checkpoint_dir.exists()  # Only after successful SSD export.


@pytest.mark.parametrize("full", [False, True], ids=["basic-grpo", "full-grpo"])
def test_rl_best_exports_validation_peak_instead_of_later_live_weights(config, tmp_path, monkeypatch, full):
    validation = tmp_path / "validation.jsonl"
    validation.write_text(json.dumps({"question": "Held out three ?", "answer": "reason #### 3"}) + "\n")
    cfg = replace(config, eval_path=validation, best_checkpoint_dir=tmp_path / "ram-cache",
                  num_steps=2, eval_every=1, save_every=1)
    module = train_grpo_full if full else train_grpo
    original_build = module.build_runtime
    held = {}
    snapshots = []
    scores = [0.4, 0.8, 0.5]

    def build(base):
        runtime = original_build(base)
        held["runtime"] = runtime
        monkeypatch.setattr(runtime.optimizer, "state_dict", lambda: pytest.fail("Optimizer was serialized"))
        return runtime

    def evaluate(runtime, *args):
        index = len(snapshots)
        snapshots.append(frozen_state(runtime.model))
        return {"accuracy": scores[index], "mean_reward": scores[index], "format_rate": 1.0,
                "mean_response_tokens": 1, "examples": []}

    update_count = [0]

    def update(runtime, *args):
        update_count[0] += 1
        with torch.no_grad():
            next(runtime.model.parameters()).fill_(update_count[0])
        return torch.tensor(0.1), {"mean_reward": 0.5}

    monkeypatch.setattr(module, "build_runtime", build)
    monkeypatch.setattr(module, "evaluate", evaluate)
    monkeypatch.setattr(module, "generate_rollouts", lambda *args: RolloutBatch(["q"] * 4, ["r"] * 4, ["3"] * 4))
    monkeypatch.setattr(module, "train_on_rollouts_full" if full else "train_on_rollouts", update)
    module.train(train_grpo_full.FullGRPOConfig(cfg) if full else cfg)
    assert update_count[0] == 2 and len(snapshots) == 3
    assert any(not torch.equal(snapshots[1][name], snapshots[2][name]) for name in snapshots[1])
    with torch.no_grad():
        next(held["runtime"].model.parameters()).fill_(42)
    assert_best_export(cfg.output_dir, snapshots[1], 1, 0.8)
    assert not cfg.best_checkpoint_dir.exists()


@pytest.mark.parametrize("module", [train_grpo, train_grpo_full], ids=["basic-grpo", "full-grpo"])
def test_best_cli_cannot_select_on_default_official_test(module, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["trainer", "--model-id", "unused", "--best-checkpoint-dir", str(tmp_path / "ram")])
    with pytest.raises(SystemExit) as error:
        module.parse_args()
    assert error.value.code == 2
    assert "validation data" in capsys.readouterr().err


@pytest.mark.parametrize("module", [train_grpo, train_grpo_full], ids=["basic-grpo", "full-grpo"])
def test_programmatic_best_rejects_training_questions_in_validation(config, tmp_path, monkeypatch, module):
    cfg = replace(config, best_checkpoint_dir=tmp_path / "ram-cache")  # Fixture uses the same train/eval file.
    monkeypatch.setattr(module, "build_runtime", lambda *args: pytest.fail("Model was loaded before validation rejection"))
    with pytest.raises(ValueError, match="disjoint"):
        module.train(train_grpo_full.FullGRPOConfig(cfg) if module is train_grpo_full else cfg)
    assert not cfg.best_checkpoint_dir.exists()


def test_sft_best_rejects_explicit_official_test_path(tiny_sft_config, tmp_path, monkeypatch):
    cfg = replace(tiny_sft_config, best_checkpoint_dir=tmp_path / "ram", eval_path=train_grpo.PROJECT_DIR / "data/gsm8k/test.jsonl")
    monkeypatch.setattr(train_sft, "build_runtime", lambda *args: pytest.fail("Model loaded before test-path rejection"))
    with pytest.raises(ValueError, match="official test"):
        train_sft.train(cfg)


def test_training_failure_retains_last_complete_ram_best_without_ssd_export(
    tiny_sft_config, tmp_path, monkeypatch,
):
    cfg = replace(tiny_sft_config, best_checkpoint_dir=tmp_path / "ram-cache")
    real_update = train_sft.train_sft_step
    updates, snapshots = [], []

    def evaluate(model, records, *args):
        snapshots.append(frozen_state(model))
        return {"loss": 3.0 if len(snapshots) == 1 else 1.0,
                "num_examples": len(records), "supervised_tokens": 1}

    def update(*args, **kwargs):
        updates.append(True)
        if len(updates) == 2:
            raise RuntimeError("simulated training interruption")
        return real_update(*args, **kwargs)

    monkeypatch.setattr(train_sft, "evaluate_sft", evaluate)
    monkeypatch.setattr(train_sft, "train_sft_step", update)
    with pytest.raises(RuntimeError, match="simulated training interruption"):
        train_sft.train(cfg)
    assert not list(cfg.output_dir.glob("checkpoint-*"))
    selected = cfg.best_checkpoint_dir / "best"
    assert selected.is_symlink()
    assert json.loads((selected / "selection.json").read_text())["step"] == 1
    restored = AutoModelForCausalLM.from_pretrained(selected, dtype="auto", local_files_only=True)
    for name, value in restored.state_dict().items():
        assert torch.equal(value, snapshots[1][name].bfloat16())


def test_cleanup_io_failure_does_not_invalidate_successfully_exported_best(
    config, tmp_path, monkeypatch, caplog,
):
    validation = tmp_path / "validation.jsonl"
    validation.write_text(json.dumps({"question": "Held out three ?", "answer": "reason #### 3"}) + "\n")
    cfg = replace(config, eval_path=validation, best_checkpoint_dir=tmp_path / "ram-cache")
    cfg.output_dir.mkdir()
    runtime = train_grpo.build_runtime(cfg)
    expected = frozen_state(runtime.model)
    store = train_grpo.make_best_store(cfg, "accuracy", "max")
    train_grpo.consider_best(runtime, cfg, store, 1, {"accuracy": 0.8},
                             eval_path=cfg.eval_path, eval_examples=1)
    calls = []

    def fail_cleanup():
        calls.append(True)
        raise OSError("simulated RAM-cache cleanup failure")

    monkeypatch.setattr(store, "cleanup", fail_cleanup)
    with caplog.at_level(logging.WARNING):
        destination = train_grpo.finish_best(store, cfg.output_dir)
    assert destination == cfg.output_dir / "checkpoint-best"
    assert calls == [True]
    assert store.best_path.is_symlink() and store.best_path.is_dir()
    assert any(record.levelno >= logging.WARNING for record in caplog.records)
    assert_best_export(cfg.output_dir, expected, 1, 0.8)
    cached = AutoModelForCausalLM.from_pretrained(store.best_path, dtype="auto", local_files_only=True)
    for name, tensor in cached.state_dict().items():
        assert torch.equal(tensor, expected[name].bfloat16())
