"""CPU regressions for the full runner's mixed-precision/checkpoint interface."""

from copy import deepcopy
from dataclasses import replace
import sys

import pytest
import torch

from training.trainers import grpo as full
from training.types import RolloutBatch, Runtime
from training.configs import UnifiedGRPOConfig as TrainConfig
from scripts.train_grpo_unified import parse_args


@pytest.fixture
def rollouts():
    return RolloutBatch(
        repeated_prompts=["Hello", "Hello"],
        responses=["world", "another test"],
        repeated_ground_truths=["world", "world"],
    )


def test_cli_preserves_full_defaults_and_forwards_runtime_options(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_grpo_unified.py", "--model-id", "unused", "--num-epochs-per-rollout", "3"])
    default = parse_args()
    assert default.num_epochs_per_rollout == 3
    assert default.advantage_normalizer == "std"
    assert default.loss_normalization == "sequence"
    assert default.autocast == "none"
    assert not default.gradient_checkpointing
    assert default.save_optimizer_state

    monkeypatch.setattr(sys, "argv", [
        "train_grpo_unified.py", "--model-id", "unused", "--num-epochs-per-rollout", "3", "--dtype", "float32", "--autocast", "bfloat16",
        "--gradient-checkpointing", "--no-save-optimizer-state",
        "--advantage-normalizer", "none", "--loss-normalization", "constant",
    ])
    configured = parse_args()
    assert configured.dtype == "float32"
    assert configured.autocast == "bfloat16"
    assert configured.gradient_checkpointing
    assert not configured.save_optimizer_state
    assert configured.advantage_normalizer == "none"
    assert configured.loss_normalization == "constant"


def test_old_scores_are_fixed_normal_tensors_and_restore_mode(
    tiny_train_model, tokenizer, rollouts,
):
    tiny_train_model.train()
    scores = full._compute_old_log_probs(tiny_train_model, rollouts, tokenizer)
    assert tiny_train_model.training
    assert not scores.requires_grad
    assert not scores.is_inference()
    # A fixed factor may need to be saved by an autograd operation. Inference
    # tensors cannot be saved this way, even after detach().
    current = torch.ones_like(scores, requires_grad=True)
    (current * scores).sum().backward()
    torch.testing.assert_close(current.grad, scores)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_full_update_forwards_same_precision_and_activates_checkpointing(
    tiny_train_model, tokenizer, rollouts, tmp_path, monkeypatch, checkpointing,
):
    model = tiny_train_model
    if checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    runtime = Runtime(model, tokenizer, optimizer, None)
    base = TrainConfig(
        model_id="unused", train_path=tmp_path / "train.jsonl",
        eval_path=tmp_path / "eval.jsonl", prompt_path=tmp_path / "prompt",
        output_dir=tmp_path / "output", group_size=2,
        gradient_accumulation_steps=2, autocast="bfloat16",
        gradient_checkpointing=checkpointing,
    )
    config = replace(base, num_epochs_per_rollout=3)
    monkeypatch.setattr(full, "reward_function", lambda config: lambda response, truth: {
        "reward": float(response == truth), "format_reward": 0.0,
        "answer_reward": float(response == truth),
    })
    observed = []
    real_score = full.run_get_response_log_probs
    real_train = full.run_grpo_train_step

    def old_score(**kwargs):
        observed.append(("old", kwargs["autocast_dtype"], model.training))
        return real_score(**kwargs)

    def update(**kwargs):
        observed.append(("current", kwargs["autocast_dtype"], model.training))
        assert not kwargs["old_log_probs"].is_inference()
        return real_train(**kwargs)

    monkeypatch.setattr(full, "run_get_response_log_probs", old_score)
    monkeypatch.setattr(full, "run_grpo_train_step", update)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss, metadata = full.train_rollout_batch(runtime, rollouts, config)
    assert observed == [
        ("old", torch.bfloat16, False),
        ("old", torch.bfloat16, False),
        ("current", torch.bfloat16, checkpointing),
        ("current", torch.bfloat16, checkpointing),
        ("current", torch.bfloat16, checkpointing),
    ]
    assert not model.training
    assert torch.isfinite(loss)
    assert "grad_norm" in metadata
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


@pytest.mark.parametrize("training", [False, True])
def test_microbatched_old_scores_match_whole_batch_with_unequal_lengths(
    tiny_train_model, tokenizer, training,
):
    rollouts = RolloutBatch(
        repeated_prompts=["Hello", "This is", "Hello"],
        responses=["world", "another test", "a test"],
        repeated_ground_truths=["world"] * 3,
    )
    tiny_train_model.train(training)
    whole = full._compute_old_log_probs(tiny_train_model, rollouts, tokenizer)
    microbatched = full._compute_old_log_probs(
        tiny_train_model, rollouts, tokenizer, microbatch_size=2,
    )
    assert tiny_train_model.training == training
    torch.testing.assert_close(microbatched, whole, rtol=1e-5, atol=1e-6)
    assert not microbatched.requires_grad
    assert not microbatched.is_inference()


def test_old_score_failure_restores_mode(tiny_train_model, tokenizer, rollouts, monkeypatch):
    tiny_train_model.train()

    def fail_score(**kwargs):
        raise RuntimeError("simulated scoring failure")

    monkeypatch.setattr(full, "run_get_response_log_probs", fail_score)
    with pytest.raises(RuntimeError, match="simulated scoring failure"):
        full._compute_old_log_probs(tiny_train_model, rollouts, tokenizer, microbatch_size=1)
    assert tiny_train_model.training


def test_full_update_checks_accumulation_before_scoring(
    tiny_train_model, tokenizer, rollouts, tmp_path, monkeypatch,
):
    runtime = Runtime(tiny_train_model, tokenizer, None, None)
    base = TrainConfig("unused", tmp_path, tmp_path, tmp_path, tmp_path,
                       gradient_accumulation_steps=3)

    def fail_score(*args, **kwargs):
        raise AssertionError("invalid batches must fail before old-policy scoring")

    monkeypatch.setattr(full, "_compute_old_log_probs", fail_score)
    with pytest.raises(ValueError, match="divisible"):
        full.train_rollout_batch(runtime, rollouts, replace(base, num_epochs_per_rollout=3))


def test_microbatched_old_scoring_preserves_full_update(
    tiny_train_model, tokenizer, rollouts, tmp_path, monkeypatch,
):
    whole_model = deepcopy(tiny_train_model)
    micro_model = deepcopy(tiny_train_model)
    base = TrainConfig("unused", tmp_path, tmp_path, tmp_path, tmp_path,
                       group_size=2, gradient_accumulation_steps=2)
    config = replace(base, num_epochs_per_rollout=3)
    monkeypatch.setattr(full, "reward_function", lambda config: lambda response, truth: {
        "reward": float(response == truth), "format_reward": 0.0,
        "answer_reward": float(response == truth),
    })
    real_old_scores = full._compute_old_log_probs

    def whole_batch_scores(*args, **kwargs):
        kwargs["microbatch_size"] = None
        return real_old_scores(*args, **kwargs)

    whole_runtime = Runtime(whole_model, tokenizer, torch.optim.SGD(whole_model.parameters(), lr=0.01), None)
    micro_runtime = Runtime(micro_model, tokenizer, torch.optim.SGD(micro_model.parameters(), lr=0.01), None)
    with monkeypatch.context() as whole_patch:
        whole_patch.setattr(full, "_compute_old_log_probs", whole_batch_scores)
        whole_loss, _ = full.train_rollout_batch(whole_runtime, rollouts, config)
    micro_loss, _ = full.train_rollout_batch(micro_runtime, rollouts, config)
    torch.testing.assert_close(micro_loss, whole_loss, rtol=1e-5, atol=1e-6)
    for whole, micro in zip(whole_model.parameters(), micro_model.parameters()):
        torch.testing.assert_close(micro, whole, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("method", ["noclip", "grpo", "gspo"])
def test_multi_epoch_keeps_one_fixed_old_policy(tiny_train_model, tokenizer, rollouts, tmp_path, monkeypatch, method):
    cfg = TrainConfig("unused", tmp_path, tmp_path, tmp_path, tmp_path,
        group_size=2, gradient_accumulation_steps=2, num_epochs_per_rollout=3,
        importance_reweighting_method=method)
    runtime = Runtime(tiny_train_model, tokenizer, torch.optim.SGD(tiny_train_model.parameters(), lr=0.01), None)
    monkeypatch.setattr(full, "reward_function", lambda _: lambda response, truth: {
        "reward": float(response == truth), "format_reward": 0., "answer_reward": float(response == truth)})
    references = []
    original = full.run_grpo_train_step
    def update(**kwargs):
        references.append(kwargs["old_log_probs"])
        return original(**kwargs)
    monkeypatch.setattr(full, "run_grpo_train_step", update)
    _, metadata = full.train_rollout_batch(runtime, rollouts, cfg)
    assert len(metadata["epoch_losses"]) == 3
    assert len(references) == 3 and all(tensor is references[0] for tensor in references)
    assert not references[0].requires_grad


def test_no_reweighting_skips_old_policy_forward(tiny_train_model, tokenizer, rollouts, tmp_path, monkeypatch):
    cfg = TrainConfig("unused", tmp_path, tmp_path, tmp_path, tmp_path,
        group_size=2, gradient_accumulation_steps=2, num_epochs_per_rollout=3,
        importance_reweighting_method="none")
    runtime = Runtime(tiny_train_model, tokenizer, torch.optim.SGD(tiny_train_model.parameters(), lr=0.01), None)
    monkeypatch.setattr(full, "_compute_old_log_probs", lambda *a, **k: pytest.fail("Unnecessary old-policy pass"))
    monkeypatch.setattr(full, "reward_function", lambda _: lambda response, truth: {
        "reward": float(response == truth), "format_reward": 0., "answer_reward": float(response == truth)})
    assert len(full.train_rollout_batch(runtime, rollouts, cfg)[1]["epoch_losses"]) == 3

