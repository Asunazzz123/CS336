"""Full on-policy GRPO: computes old log-probs before each update and applies
token-level PPO-style importance-weighted clip (importance_reweighting_method='grpo').

This script extends train_grpo.py by scoring the rollouts under the current,
pre-update policy so that run_grpo_train_step can apply the clip ratio.
These old policy scores are not a separate reference model or a KL penalty.
All infrastructure (data loading, generation, evaluation, checkpointing) is reused
directly from train_grpo.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alignment.grpo import (
    run_get_response_log_probs,
    run_grpo_train_step,
    run_tokenize_prompt_and_output,
)
from scripts.train_grpo import (
    RolloutBatch,
    Runtime,
    TrainConfig,
    build_runtime,
    close_runtime,
    evaluate,
    generate_rollouts,
    json_value,
    load_questions,
    load_prompt_template,
    log_metrics,
    reward_function,
    sample_questions,
    save_checkpoint,
    seed_everything,
    sync_generator,
)


@dataclass
class FullGRPOConfig:
    """Extends TrainConfig with GRPO-specific importance-weighting parameters."""

    # All base fields are inherited by embedding TrainConfig rather than
    # subclassing so that dataclass field ordering stays clean.
    base: TrainConfig

    # Importance-weighting method forwarded to run_compute_policy_gradient_loss.
    # "grpo" applies token-level PPO-style clip; "gspo" applies sequence-level clip.
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "grpo"

    # PPO clip range (epsilon).  Typical value: 0.2.
    cliprange: float = 0.2

    # Baseline and normalizer forwarded to run_compute_group_normalized_rewards.
    baseline: Literal["mean", "none"] = "mean"
    advantage_normalizer: Literal["std", "none", "mean"] = "std"

    # Loss normalisation forwarded to run_aggregate_loss_across_microbatch.
    loss_normalization: Literal["sequence", "constant"] = "sequence"


def _compute_old_log_probs(
    model: torch.nn.Module,
    rollouts: RolloutBatch,
    tokenizer,
    autocast_dtype: torch.dtype | None = None,
    microbatch_size: int | None = None,
) -> torch.Tensor:
    """Tokenize rollouts and compute per-token log-probs under the current policy.

    Tokenize/pad the whole batch once, then score microbatches to bound the
    full-vocabulary logits allocation. This pass retains no gradients.
    The returned tensor is a normal detached tensor on the model's device,
    because autograd may need to save these fixed scores during the update.
    """
    device = next(model.parameters()).device
    tokenized = run_tokenize_prompt_and_output(
        prompt_strs=rollouts.repeated_prompts,
        output_strs=rollouts.responses,
        tokenizer=tokenizer,
    )
    input_ids = tokenized["input_ids"].to(device)
    labels = tokenized["labels"].to(device)
    batch_size = input_ids.shape[0]
    if microbatch_size is None:
        microbatch_size = batch_size
    if microbatch_size <= 0:
        raise ValueError("old-policy microbatch_size must be positive")

    previous_mode = model.training
    model.eval()
    try:
        with torch.inference_mode():
            scores = []
            for start in range(0, batch_size, microbatch_size):
                result = run_get_response_log_probs(
                    model=model,
                    input_ids=input_ids[start:start + microbatch_size],
                    labels=labels[start:start + microbatch_size],
                    return_token_entropy=False,
                    autocast_dtype=autocast_dtype,
                )
                scores.append(result["log_probs"])
            old_log_probs = torch.cat(scores, dim=0)
    finally:
        model.train(previous_mode)

    # Cloning outside inference_mode removes the inference-tensor restriction.
    return old_log_probs.clone().detach()


def train_on_rollouts_full(
    runtime: Runtime,
    rollouts: RolloutBatch,
    config: FullGRPOConfig,
) -> tuple[torch.Tensor, dict]:
    """Compute old log-probs then call run_grpo_train_step with GRPO clipping."""
    autocast_dtype = (
        torch.bfloat16 if config.base.autocast == "bfloat16" else None
    )
    batch_size = len(rollouts.responses)
    accumulation_steps = config.base.gradient_accumulation_steps
    if accumulation_steps <= 0 or batch_size == 0 or batch_size % accumulation_steps:
        raise ValueError("rollout batch size must be positive and divisible by gradient accumulation steps")
    old_log_probs = _compute_old_log_probs(
        runtime.model, rollouts, runtime.tokenizer, autocast_dtype=autocast_dtype,
        microbatch_size=batch_size // accumulation_steps,
    )

    previous_mode = runtime.model.training
    # Transformers only activates gradient checkpointing in training mode.
    runtime.model.train(bool(config.base.gradient_checkpointing))
    try:
        return run_grpo_train_step(
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            optimizer=runtime.optimizer,
            gradient_accumulation_steps=config.base.gradient_accumulation_steps,
            max_grad_norm=config.base.max_grad_norm,
            reward_fn=reward_function(config.base),
            repeated_prompts=rollouts.repeated_prompts,
            rollout_responses=rollouts.responses,
            repeated_ground_truths=rollouts.repeated_ground_truths,
            group_size=config.base.group_size,
            importance_reweighting_method=config.importance_reweighting_method,
            old_log_probs=old_log_probs,
            cliprange=config.cliprange,
            baseline=config.baseline,
            advantage_normalizer=config.advantage_normalizer,
            loss_normalization=config.loss_normalization,
            autocast_dtype=autocast_dtype,
        )
    finally:
        runtime.model.train(previous_mode)


def train(config: FullGRPOConfig) -> None:
    base = config.base
    seed_everything(base.seed)

    train_questions = load_questions(base.train_path)
    eval_questions = load_questions(base.eval_path)[: base.eval_examples]
    if len(train_questions) < base.prompts_per_batch:
        raise ValueError("训练题目数小于 prompts_per_batch")

    template = load_prompt_template(base.prompt_path)
    reward_function(base)  # early dependency check before loading the model

    if base.output_dir.exists() and any(base.output_dir.iterdir()):
        raise ValueError("输出目录非空，请选择新的 --output-dir")
    base.output_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = {
        "base": json_value(dataclasses.asdict(base)),
        "importance_reweighting_method": config.importance_reweighting_method,
        "cliprange": config.cliprange,
        "baseline": config.baseline,
        "advantage_normalizer": config.advantage_normalizer,
        "loss_normalization": config.loss_normalization,
    }
    (base.output_dir / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2), encoding="utf-8"
    )

    runtime = build_runtime(base)
    try:
        sync_generator(runtime, base)
        log_metrics(
            base.output_dir, 0, "eval",
            evaluate(runtime, eval_questions, template, base),
        )

        for step in range(1, base.num_steps + 1):
            questions = sample_questions(
                train_questions, base.prompts_per_batch, step,
                base.seed, base.sampling_strategy,
            )
            started = time.monotonic()
            sync_generator(runtime, base)
            rollouts = generate_rollouts(
                runtime, questions, template,
                replace(base, seed=base.seed + step),
            )
            generation_seconds = time.monotonic() - started

            started = time.monotonic()
            loss, metadata = train_on_rollouts_full(runtime, rollouts, config)
            if runtime.model.device.type == "cuda":
                torch.cuda.synchronize(runtime.model.device)

            log_metrics(base.output_dir, step, "train", {
                **metadata,
                "loss": loss,
                "generation_seconds": generation_seconds,
                "update_seconds": time.monotonic() - started,
                "examples": [
                    {"prompt": p, "response": r, "ground_truth": g}
                    for p, r, g in zip(
                        rollouts.repeated_prompts[:3],
                        rollouts.responses[:3],
                        rollouts.repeated_ground_truths[:3],
                    )
                ],
            })

            if step % base.eval_every == 0 or step == base.num_steps:
                sync_generator(runtime, base)
                log_metrics(
                    base.output_dir, step, "eval",
                    evaluate(runtime, eval_questions, template, base),
                )
            if step % base.save_every == 0 or step == base.num_steps:
                save_checkpoint(runtime, base, step)
    finally:
        close_runtime(runtime)


def parse_args() -> FullGRPOConfig:
    parser = argparse.ArgumentParser(description=__doc__)

    # Model and data
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B-Base")
    for name, default in (
        ("train-path", PROJECT_DIR / "data/gsm8k/train.jsonl"),
        ("eval-path", PROJECT_DIR / "data/gsm8k/test.jsonl"),
        ("prompt-path", PROJECT_DIR / "alignment/prompts/r1_zero.prompt"),
        ("output-dir", PROJECT_DIR / "outputs/grpo_full"),
    ):
        parser.add_argument(f"--{name}", type=Path, default=default)

    # Integer hyper-params
    for name, default in (
        ("seed", 42),
        ("num-steps", 100),
        ("prompts-per-batch", 8),
        ("group-size", 8),
        ("gradient-accumulation-steps", 8),
        ("max-new-tokens", 512),
        ("eval-every", 10),
        ("eval-examples", 128),
        ("save-every", 20),
        ("generation-batch-size", 4),
        ("vllm-port", 8000),
        ("vllm-gpu", 1),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)

    # Float hyper-params
    for name, default in (
        ("learning-rate", 1e-5),
        ("max-grad-norm", 1.0),
        ("temperature", 1.0),
        ("weight-decay", 0.0),
        ("cliprange", 0.2),
    ):
        parser.add_argument(f"--{name}", type=float, default=default)

    # Categorical
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--autocast", choices=["none", "bfloat16"], default="none")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--no-save-optimizer-state", dest="save_optimizer_state",
        action="store_false",
        help="Save model weights and run state without the optimizer state",
    )
    parser.add_argument(
        "--generation-backend", choices=["transformers", "vllm"], default="transformers"
    )
    parser.add_argument(
        "--optimizer-name", choices=["adam", "adamw", "muon"], default="adamw"
    )
    parser.add_argument(
        "--optimizer-device", choices=["model", "cpu"], default="model",
        help="Keep Adam/AdamW FP32 master weights and optimizer state on CPU",
    )
    parser.add_argument(
        "--sampling-strategy", choices=["random", "order"], default="random"
    )
    parser.add_argument(
        "--reward-type",
        choices=["gsm8k_numeric", "r1_zero", "question_only"],
        default="gsm8k_numeric",
    )
    parser.add_argument(
        "--importance-reweighting-method",
        choices=["none", "noclip", "grpo", "gspo"],
        default="grpo",
    )
    parser.add_argument(
        "--baseline", choices=["mean", "none"], default="mean"
    )
    parser.add_argument(
        "--advantage-normalizer", choices=["std", "none", "mean"], default="std"
    )
    parser.add_argument(
        "--loss-normalization", choices=["sequence", "constant"], default="sequence"
    )
    parser.add_argument("--vllm-host", default="127.0.0.1")

    args = parser.parse_args()
    if args.optimizer_device == "cpu" and args.optimizer_name == "muon":
        parser.error("CPU optimizer offload only supports adam/adamw")

    # Basic validation mirroring train_grpo.py
    for name in (
        "num_steps", "prompts_per_batch", "group_size",
        "gradient_accumulation_steps", "max_new_tokens",
        "eval_every", "eval_examples", "save_every", "generation_batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.group_size < 2:
        parser.error("group_size must be at least 2")
    if (args.prompts_per_batch * args.group_size) % args.gradient_accumulation_steps:
        parser.error(
            "rollout batch size (prompts_per_batch * group_size) must be "
            "divisible by gradient_accumulation_steps"
        )
    if args.learning_rate <= 0 or args.max_grad_norm <= 0 or args.temperature <= 0:
        parser.error("lr / max_grad_norm / temperature must be positive")
    if args.cliprange <= 0:
        parser.error("cliprange must be positive")

    base = TrainConfig(
        model_id=args.model_id,
        train_path=args.train_path,
        eval_path=args.eval_path,
        prompt_path=args.prompt_path,
        output_dir=args.output_dir,
        device=args.device,
        generation_backend=args.generation_backend,
        seed=args.seed,
        num_steps=args.num_steps,
        prompts_per_batch=args.prompts_per_batch,
        group_size=args.group_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        eval_every=args.eval_every,
        eval_examples=args.eval_examples,
        save_every=args.save_every,
        optimizer_name=args.optimizer_name,
        optimizer_device=args.optimizer_device,
        weight_decay=args.weight_decay,
        dtype=args.dtype,
        autocast=args.autocast,
        gradient_checkpointing=args.gradient_checkpointing,
        save_optimizer_state=args.save_optimizer_state,
        advantage_normalizer=args.advantage_normalizer,
        loss_normalization=args.loss_normalization,
        sampling_strategy=args.sampling_strategy,
        reward_type=args.reward_type,
        generation_batch_size=args.generation_batch_size,
        vllm_host=args.vllm_host,
        vllm_port=args.vllm_port,
        vllm_gpu=args.vllm_gpu,
    )
    return FullGRPOConfig(
        base=base,
        importance_reweighting_method=args.importance_reweighting_method,
        baseline=args.baseline,
        advantage_normalizer=args.advantage_normalizer,
        loss_normalization=args.loss_normalization,
        cliprange=args.cliprange,
    )


if __name__ == "__main__":
    train(parse_args())
