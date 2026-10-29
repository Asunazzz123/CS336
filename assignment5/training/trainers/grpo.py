"""GRPO application workflow for single-device and FSDP execution."""
from __future__ import annotations
import json
import time
from dataclasses import asdict, replace
from typing import Any
import numpy as np
import torch
from alignment.grpo import run_get_response_log_probs, run_grpo_train_step, run_tokenize_prompt_and_output
from alignment.rewards.factory import reward_function
from checkpointing.best import BestCheckpointStore
from training.configs import UnifiedGRPOConfig
from training.types import RolloutBatch, Runtime
from training.backends.single_device import build_runtime, close_runtime
from training.data.gsm8k import load_questions, load_prompt_template, validate_best_eval_path, validate_best_data
from training.data.sampling import sample_questions
from training.checkpoint_policy import make_best_store, consider_best, finish_best, save_checkpoint
from training.rollout import generate_rollouts, sync_generator
from training.evaluation.generation import evaluate
from training.reproducibility import seed_everything
from training.validation import validate_config
from utils.serialization import json_value
from profiling.metrics import log_metrics


def _compute_old_log_probs(
    model: torch.nn.Module,
    rollouts: RolloutBatch,
    tokenizer: Any,
    autocast_dtype: torch.dtype | None = None,
    microbatch_size: int | None = None,
) -> torch.Tensor:
    """在当前策略下计算 rollouts 的 per-token log-probs，用于后续 ratio 计算。

    仅在 num_epochs_per_rollout > 1 时调用，避免单次更新模式的额外开销。
    返回 detached tensor，作为固定参考。
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
                end = start + microbatch_size
                result = run_get_response_log_probs(
                    model=model,
                    input_ids=input_ids[start:end],
                    labels=labels[start:end],
                    return_token_entropy=False,
                    autocast_dtype=autocast_dtype,
                )
                scores.append(result["log_probs"])
            old_log_probs = torch.cat(scores, dim=0)
    finally:
        model.train(previous_mode)

    return old_log_probs.clone().detach()


def train_on_rollouts_unified(
    runtime: Runtime,
    rollouts: RolloutBatch,
    config: UnifiedGRPOConfig,
    old_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """统一的 rollout 训练函数。

    根据 num_epochs_per_rollout 决定是否使用 importance weighting：
    - num_epochs=1 且 old_log_probs=None: 纯 on-policy 单次更新
    - num_epochs>=1 且 old_log_probs 提供: 应用 ratio clip
    """
    autocast_dtype = torch.bfloat16 if config.autocast == "bfloat16" else None

    previous_mode = runtime.model.training
    runtime.model.train(bool(config.gradient_checkpointing))
    try:
        return run_grpo_train_step(
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            optimizer=runtime.optimizer,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            reward_fn=reward_function_from_config(config),
            repeated_prompts=rollouts.repeated_prompts,
            rollout_responses=rollouts.responses,
            repeated_ground_truths=rollouts.repeated_ground_truths,
            group_size=config.group_size,
            importance_reweighting_method=(config.importance_reweighting_method
                                           if old_log_probs is not None else "none"),
            old_log_probs=old_log_probs,
            cliprange=config.cliprange,
            baseline=config.baseline,
            advantage_normalizer=config.advantage_normalizer,
            loss_normalization=config.loss_normalization,
            autocast_dtype=autocast_dtype,
        )
    finally:
        runtime.model.train(previous_mode)


def reward_function_from_config(config: UnifiedGRPOConfig):
    return reward_function(config)


def train_rollout_batch(runtime, rollouts, config):
    """Update a rollout once or over multiple epochs with one fixed old policy."""
    if config.num_epochs_per_rollout <= 0:
        raise ValueError("num_epochs_per_rollout must be positive")
    count = len(rollouts.responses)
    if (not count or config.gradient_accumulation_steps <= 0
            or count % config.gradient_accumulation_steps):
        raise ValueError("rollout batch must be nonempty and divisible by gradient_accumulation_steps")
    if config.group_size < 2 or count % config.group_size:
        raise ValueError("rollout batch must contain complete reward groups of size >=2")
    old = None
    if config.num_epochs_per_rollout > 1 and config.importance_reweighting_method != "none":
        old = _compute_old_log_probs(runtime.model, rollouts, runtime.tokenizer,
            autocast_dtype=torch.bfloat16 if config.autocast == "bfloat16" else None,
            microbatch_size=count // config.gradient_accumulation_steps)
    losses = []
    for _ in range(config.num_epochs_per_rollout):
        loss, metadata = train_on_rollouts_unified(runtime, rollouts, config, old_log_probs=old)
        losses.append(loss.item())
    return loss, {**metadata, "epoch_losses": losses, "mean_epoch_loss": float(np.mean(losses))}


def train(config: UnifiedGRPOConfig) -> None:
    """统一训练主循环。"""
    validate_config(config)
    if config.training_backend == "fsdp2":
        from training.backends.fsdp_grpo import train_fsdp
        return train_fsdp(config)
    if config.training_backend != "transformers":
        raise ValueError("Unknown training backend")
    seed_everything(config.seed)
    validate_best_eval_path(config.best_checkpoint_dir, config.eval_path)

    # 数据加载
    train_questions = load_questions(config.train_path)
    validation_questions = load_questions(config.eval_path)
    validate_best_data_unified(config, train_questions, validation_questions)
    eval_questions = validation_questions[: config.eval_examples]

    if len(train_questions) < config.prompts_per_batch:
        raise ValueError("训练题目数小于 prompts_per_batch")

    template = load_prompt_template(config.prompt_path)
    reward_function_from_config(config)  # 早期依赖检查

    # 输出目录准备
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise ValueError("输出目录非空，请选择新的 --output-dir")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 保存运行配置
    run_cfg = json_value(asdict(config))
    run_cfg["train_path"] = str(config.train_path)
    run_cfg["eval_path"] = str(config.eval_path)
    run_cfg["prompt_path"] = str(config.prompt_path)
    run_cfg["output_dir"] = str(config.output_dir)
    if config.best_checkpoint_dir:
        run_cfg["best_checkpoint_dir"] = str(config.best_checkpoint_dir)
    (config.output_dir / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2), encoding="utf-8"
    )

    # 构建运行时
    best = make_best_store_unified(config)
    runtime = build_runtime_unified(config)

    try:
        # 初始评估
        sync_generator(runtime, config)
        metrics = evaluate(runtime, eval_questions, template, config)
        log_metrics(config.output_dir, 0, "eval", metrics)
        consider_best(runtime, config, best, 0, metrics,
                     eval_path=config.eval_path, eval_examples=len(eval_questions))

        # 训练循环
        for step in range(1, config.num_steps + 1):
            questions = sample_questions(
                train_questions, config.prompts_per_batch, step,
                config.seed, config.sampling_strategy,
            )

            # 生成 rollouts
            started = time.monotonic()
            sync_generator(runtime, config)
            rollouts = generate_rollouts(
                runtime, questions, template,
                replace(config, seed=config.seed + step),
            )
            generation_seconds = time.monotonic() - started

            # 多 epoch 训练逻辑
            started = time.monotonic()

            final_loss, metadata = train_rollout_batch(runtime, rollouts, config)

            if runtime.model.device.type == "cuda":
                torch.cuda.synchronize(runtime.model.device)

            # 记录训练指标
            log_metrics(config.output_dir, step, "train", {
                **metadata,
                "loss": final_loss.item() if torch.is_tensor(final_loss) else final_loss,
                "generation_seconds": generation_seconds,
                "update_seconds": time.monotonic() - started,
                "num_epochs_per_rollout": config.num_epochs_per_rollout,
                "examples": [
                    {"prompt": p, "response": r, "ground_truth": g}
                    for p, r, g in zip(
                        rollouts.repeated_prompts[:3],
                        rollouts.responses[:3],
                        rollouts.repeated_ground_truths[:3],
                    )
                ],
            })

            # 评估与保存
            if step % config.eval_every == 0 or step == config.num_steps:
                sync_generator(runtime, config)
                metrics = evaluate(runtime, eval_questions, template, config)
                log_metrics(config.output_dir, step, "eval", metrics)
                consider_best(runtime, config, best, step, metrics,
                             eval_path=config.eval_path, eval_examples=len(eval_questions))

            if best is None and (step % config.save_every == 0 or step == config.num_steps):
                save_checkpoint(runtime, config, step)

        if best is not None:
            finish_best(best, config.output_dir)

    finally:
        close_runtime(runtime)


def build_runtime_unified(config: UnifiedGRPOConfig) -> Runtime:
    """构建运行时（调用 single-device backend）。"""
    return build_runtime(config)


def validate_best_data_unified(config, train_questions, validation_questions):
    """验证 best checkpoint 数据（调用 checkpoint/data 功能模块）。"""
    validate_best_data(
        config,
        train_questions,
        validation_questions,
        config.eval_path
    )


def make_best_store_unified(config) -> BestCheckpointStore | None:
    """创建 best checkpoint store（调用 checkpoint/data 功能模块）。"""
    return make_best_store(config, "accuracy", "max")
