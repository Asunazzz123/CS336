"""Package boundaries and resource invariants after the scripts extraction."""
import ast
from dataclasses import fields
import json
from pathlib import Path
import warnings

import pytest

from orchestration.profile import profile_command, main as profile_main
from training.configs import EvaluationConfig
from .test_train_sft import tiny_sft_config

ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = ("training", "alignment", "distributed", "generation", "checkpointing", "profiling", "diagnostics", "utils", "orchestration")


def imports(path):
    # Existing grader docstrings contain escapes; this checks imports, not prose.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            yield from (item.name for item in node.names)


def test_libraries_never_import_script_entry_points():
    violations = [(str(path.relative_to(ROOT)), module) for package in LIBRARIES
                  for path in (ROOT / package).rglob("*.py") for module in imports(path)
                  if module == "scripts" or module.startswith("scripts.")]
    assert violations == []


def test_system_packages_do_not_import_training_applications():
    violations = [(str(path.relative_to(ROOT)), module) for package in
                  ("distributed", "generation", "checkpointing", "profiling", "utils", "alignment")
                  for path in (ROOT / package).rglob("*.py") for module in imports(path)
                  if module.split(".")[0] in ("training", "diagnostics", "orchestration")]
    assert violations == []


def test_generation_service_has_no_algorithm_or_fsdp_dependency():
    assert not [(str(path), module) for path in (ROOT / "generation").glob("*.py")
                for module in imports(path) if module.split(".")[0] in ("alignment", "distributed")]


@pytest.mark.parametrize("profile,backend,gpus,accumulation", [
    ("4b-single", "transformers", "0", "8"), ("9b-fsdp", "fsdp2", "0,1", "4")])
def test_experiment_profiles_preserve_global_batch_and_preview_only(profile, backend, gpus, accumulation, capsys):
    path = ROOT / "configs/experiments" / (profile + ".json")
    command, environment = profile_command(path)
    assert environment["TRAINING_BACKEND"] == backend
    assert environment["CUDA_VISIBLE_DEVICES"] == gpus
    assert environment["OPTIMIZER_DEVICE"] == "model"
    assert int(environment["SFT_ACCUMULATION_STEPS"]) * len(gpus.split(",")) == 8
    assert environment["RL_ACCUMULATION_STEPS"] == accumulation
    assert environment["SFT_MAX_STEPS"] == ""
    profile_main(["--profile", str(path)])
    preview = json.loads(capsys.readouterr().out)
    assert preview["execute"] is False and preview["command"] == command


def test_profile_argument_boundaries_and_overrides(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"environment": {"MODEL_PATH": "a model;$HOME"}}))
    command, values = profile_command(path, ["RL_STEPS=2"])
    assert command[0] == "bash"
    assert values == {"MODEL_PATH": "a model;$HOME", "RL_STEPS": "2"}
    with pytest.raises(ValueError):
        profile_command(path, ["invalid-key=2"])


def test_evaluation_configuration_has_no_optimizer_or_training_budget():
    names = {field.name for field in fields(EvaluationConfig)}
    assert names.isdisjoint({"optimizer_name", "learning_rate", "num_steps", "num_epochs", "gradient_accumulation_steps"})


def test_saved_policy_evaluation_never_constructs_optimizer(tiny_sft_config, tmp_path, monkeypatch):
    import torch
    from transformers import AutoModelForCausalLM
    from training.evaluation.reports import evaluate_checkpoint
    model = AutoModelForCausalLM.from_pretrained(tiny_sft_config.model_id, local_files_only=True)
    model.generation_config.suppress_tokens = [0, 2]
    model.save_pretrained(tiny_sft_config.model_id)
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("Evaluation created optimizer"))
    output = tmp_path / "evaluation.json"
    report = evaluate_checkpoint(tiny_sft_config.model_id, tiny_sft_config.train_path,
        tiny_sft_config.prompt_path, output, device="cpu", max_new_tokens=2, examples=1)
    assert report["num_examples"] == 1 and report["dtype"] == "bfloat16"
    assert 0 <= report["metrics"]["accuracy"] <= 1
    assert output.is_file()
    with pytest.raises(FileExistsError):
        evaluate_checkpoint(tiny_sft_config.model_id, tiny_sft_config.train_path,
                            tiny_sft_config.prompt_path, output, device="cpu")
