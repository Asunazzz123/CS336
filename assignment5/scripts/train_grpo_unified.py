"""统一 GRPO 实现：支持单次更新和多 epoch PPO-style 裁剪。

架构设计：
- num_epochs_per_rollout=1: 纯 on-policy 单次更新，跳过 old_log_probs 计算（快速模式）
- num_epochs_per_rollout>1: 多 epoch 更新，固定 old_log_probs，ratio clip 生效

参数统一：
- 显式暴露所有归一化、baseline、clip 参数
- 提供合理默认值，支持 CLI 完全覆盖
- 向后兼容原有两个脚本的典型用法
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alignment.grpo import (
    run_get_response_log_probs,
    run_grpo_train_step,
    run_tokenize_prompt_and_output,
)
from scripts.train_grpo_base import (
    RolloutBatch,
    TrainConfig,
    Runtime,
    BestCheckpointStore,
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
    validate_best_eval_path,
    validate_best_data,
    make_best_store,
    consider_best,
    finish_best,
)

logger = logging.getLogger(__name__)


@dataclass
class UnifiedGRPOConfig:
    """统一 GRPO 配置：整合基础训练参数与 PPO-style 裁剪参数。"""

    # === 模型与数据 ===
    model_id: str
    train_path: Path
    eval_path: Path
    prompt_path: Path
    output_dir: Path

    # === 设备与精度 ===
    device: str = "cpu"
    dtype: str = "float32"
    autocast: str = "none"
    gradient_checkpointing: bool = False

    # === 生成后端 ===
    generation_backend: str = "transformers"
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000
    vllm_gpu: int = 1

    # === 训练超参数 ===
    seed: int = 42
    num_steps: int = 100
    prompts_per_batch: int = 8
    group_size: int = 8
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    optimizer_name: str = "adamw"
    optimizer_device: str = "model"

    # === 生成参数 ===
    temperature: float = 1.0
    max_new_tokens: int = 512
    generation_batch_size: int = 4
    sampling_strategy: str = "random"
    reward_type: str = "gsm8k_numeric"

    # === 多 epoch 与裁剪控制（核心融合点）===
    num_epochs_per_rollout: int = 1  # 1=单次更新，>1=多epoch PPO-style
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "grpo"
    cliprange: float = 0.2

    # === 奖励归一化 ===
    baseline: Literal["mean", "none"] = "mean"
    advantage_normalizer: Literal["std", "none", "mean"] = "std"
    loss_normalization: Literal["sequence", "constant"] = "sequence"

    # === 评估与保存 ===
    eval_every: int = 10
    eval_examples: int = 128
    save_every: int = 20
    save_optimizer_state: bool = True
    best_checkpoint_dir: Path | None = None
    training_backend: str = "transformers"
    rollout_checkpoint_dir: Path = Path("/dev/shm/cs336-rollout")
    vllm_memory_utilization: float = 0.8
    vllm_max_model_len: int = 2048
    vllm_startup_timeout: int = 600
    checkpoint_format: str = "dcp"


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
    """从配置创建奖励函数（复用 train_grpo.py 的 reward_function）。"""
    # 构造临时 TrainConfig 对象以复用现有 reward_function
    temp_config = TrainConfig(
        model_id=config.model_id,
        train_path=config.train_path,
        eval_path=config.eval_path,
        prompt_path=config.prompt_path,
        output_dir=config.output_dir,
        reward_type=config.reward_type,
    )
    return reward_function(temp_config)


def train(config: UnifiedGRPOConfig) -> None:
    """统一训练主循环。"""
    if config.training_backend == "fsdp2":
        from scripts.fsdp_grpo_runtime import train_fsdp
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
        sync_generator(runtime, convert_to_train_config(config))
        metrics = evaluate(runtime, eval_questions, template, convert_to_train_config(config))
        log_metrics(config.output_dir, 0, "eval", metrics)
        consider_best(runtime, convert_to_train_config(config), best, 0, metrics,
                     eval_path=config.eval_path, eval_examples=len(eval_questions))

        # 训练循环
        for step in range(1, config.num_steps + 1):
            questions = sample_questions(
                train_questions, config.prompts_per_batch, step,
                config.seed, config.sampling_strategy,
            )

            # 生成 rollouts
            started = time.monotonic()
            sync_generator(runtime, convert_to_train_config(config))
            rollouts = generate_rollouts(
                runtime, questions, template,
                replace(convert_to_train_config(config), seed=config.seed + step),
            )
            generation_seconds = time.monotonic() - started

            # 多 epoch 训练逻辑
            started = time.monotonic()

            if config.num_epochs_per_rollout > 1:
                # 固定 old_log_probs，多个 epoch 更新
                autocast_dtype = torch.bfloat16 if config.autocast == "bfloat16" else None
                batch_size = len(rollouts.responses)
                microbatch_size = batch_size // config.gradient_accumulation_steps

                old_log_probs = _compute_old_log_probs(
                    runtime.model, rollouts, runtime.tokenizer,
                    autocast_dtype=autocast_dtype,
                    microbatch_size=microbatch_size,
                )

                epoch_losses = []
                epoch_metadata = {}

                for epoch in range(config.num_epochs_per_rollout):
                    loss, metadata = train_on_rollouts_unified(
                        runtime, rollouts, config, old_log_probs=old_log_probs
                    )
                    epoch_losses.append(loss.item())
                    if epoch == config.num_epochs_per_rollout - 1:
                        epoch_metadata = metadata

                # 汇总多 epoch 指标
                final_loss = torch.tensor(epoch_losses[-1])
                metadata = {
                    **epoch_metadata,
                    "epoch_losses": epoch_losses,
                    "mean_epoch_loss": float(np.mean(epoch_losses)),
                }
            else:
                # 单次更新，不计算 old_log_probs
                final_loss, metadata = train_on_rollouts_unified(
                    runtime, rollouts, config, old_log_probs=None
                )

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
                sync_generator(runtime, convert_to_train_config(config))
                metrics = evaluate(runtime, eval_questions, template, convert_to_train_config(config))
                log_metrics(config.output_dir, step, "eval", metrics)
                consider_best(runtime, convert_to_train_config(config), best, step, metrics,
                             eval_path=config.eval_path, eval_examples=len(eval_questions))

            if best is None and (step % config.save_every == 0 or step == config.num_steps):
                save_checkpoint(runtime, convert_to_train_config(config), step)

        if best is not None:
            finish_best(best, config.output_dir)

    finally:
        close_runtime(runtime)


def convert_to_train_config(config: UnifiedGRPOConfig):
    """将 UnifiedGRPOConfig 转换为 TrainConfig 以复用现有函数。"""
    return TrainConfig(
        model_id=config.model_id,
        train_path=config.train_path,
        eval_path=config.eval_path,
        prompt_path=config.prompt_path,
        output_dir=config.output_dir,
        device=config.device,
        generation_backend=config.generation_backend,
        seed=config.seed,
        num_steps=config.num_steps,
        prompts_per_batch=config.prompts_per_batch,
        group_size=config.group_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
        eval_every=config.eval_every,
        eval_examples=config.eval_examples,
        save_every=config.save_every,
        optimizer_name=config.optimizer_name,
        optimizer_device=config.optimizer_device,
        weight_decay=config.weight_decay,
        dtype=config.dtype,
        sampling_strategy=config.sampling_strategy,
        reward_type=config.reward_type,
        generation_batch_size=config.generation_batch_size,
        vllm_host=config.vllm_host,
        vllm_port=config.vllm_port,
        vllm_gpu=config.vllm_gpu,
        advantage_normalizer=config.advantage_normalizer,
        loss_normalization=config.loss_normalization,
        autocast=config.autocast,
        gradient_checkpointing=config.gradient_checkpointing,
        save_optimizer_state=config.save_optimizer_state,
        best_checkpoint_dir=config.best_checkpoint_dir,
    )


def build_runtime_unified(config: UnifiedGRPOConfig) -> Runtime:
    """构建运行时（复用 train_grpo.py 的 build_runtime）。"""
    return build_runtime(convert_to_train_config(config))


def validate_best_data_unified(config, train_questions, validation_questions):
    """验证 best checkpoint 数据（复用 train_grpo.py）。"""
    validate_best_data(
        convert_to_train_config(config),
        train_questions,
        validation_questions,
        config.eval_path
    )


def make_best_store_unified(config) -> BestCheckpointStore | None:
    """创建 best checkpoint store（复用 train_grpo.py）。"""
    return make_best_store(convert_to_train_config(config), "accuracy", "max")


def parse_args() -> UnifiedGRPOConfig:
    """统一参数解析，整合两个脚本的所有选项。"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # === 模型与数据 ===
    parser.add_argument("--model-id", required=True, help="HuggingFace 模型 ID")
    parser.add_argument("--train-path", type=Path, default=PROJECT_DIR / "data/gsm8k/train.jsonl")
    parser.add_argument("--eval-path", type=Path, default=PROJECT_DIR / "data/gsm8k/test.jsonl")
    parser.add_argument("--prompt-path", type=Path, default=PROJECT_DIR / "alignment/prompts/r1_zero.prompt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/grpo_unified")

    # === 设备与精度 ===
    parser.add_argument("--device", default="cpu", help="训练设备（cpu/cuda/cuda:0）")
    parser.add_argument("--training-backend", choices=["transformers", "fsdp2"], default="transformers",
                       help="fsdp2 requires torchrun with >=2 NVIDIA GPUs; device comes from LOCAL_RANK")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--autocast", choices=["none", "bfloat16"], default="none",
                       help="FP32 主权重 + BF16 autocast 计算")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                       help="以重算换显存")

    # === 生成后端 ===
    parser.add_argument("--generation-backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-gpu", type=int, default=1)
    parser.add_argument("--rollout-checkpoint-dir", type=Path, default=Path("/dev/shm/cs336-rollout"))
    parser.add_argument("--vllm-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-max-model-len", type=int, default=2048)
    parser.add_argument("--vllm-startup-timeout", type=int, default=600)

    # === 训练超参数 ===
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=100, help="训练步数")
    parser.add_argument("--prompts-per-batch", type=int, default=8, help="每批采样的题目数")
    parser.add_argument("--group-size", type=int, default=8, help="每题生成的回答数")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer-name", choices=["adam", "adamw", "muon"], default="adamw")
    parser.add_argument("--optimizer-device", choices=["model", "cpu"], default="model",
                       help="cpu 将优化器状态放在 CPU")

    # === 生成参数 ===
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--sampling-strategy", choices=["random", "order"], default="random")
    parser.add_argument("--reward-type", choices=["gsm8k_numeric", "r1_zero", "question_only"],
                       default="gsm8k_numeric")

    # === 多 epoch 与裁剪控制（核心融合参数）===
    parser.add_argument("--num-epochs-per-rollout", type=int, default=1,
                       help="每批 rollout 的更新 epoch 数。1=单次更新（快速），>1=多 epoch PPO-style（clip 生效）")
    parser.add_argument("--importance-reweighting-method",
                       choices=["none", "noclip", "grpo", "gspo"], default="grpo",
                       help="重要性加权方法。grpo=token-level clip，gspo=sequence-level clip")
    parser.add_argument("--cliprange", type=float, default=0.2,
                       help="PPO clip 范围（epsilon）")

    # === 奖励归一化 ===
    parser.add_argument("--baseline", choices=["mean", "none"], default="mean",
                       help="优势基线。mean=减组均值，none=不减")
    parser.add_argument("--advantage-normalizer", choices=["std", "none", "mean"], default="std",
                       help="优势归一化。std=除以标准差，none=不归一化，mean=除以均值")
    parser.add_argument("--loss-normalization", choices=["sequence", "constant"], default="sequence",
                       help="损失归一化。sequence=先逐序列平均再批平均，constant=按 token 总数平均")

    # === 评估与保存 ===
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--checkpoint-format", choices=["dcp", "hf"], default="dcp",
                       help="FSDP periodic output format; best checkpoints always use HF")
    parser.add_argument("--no-save-optimizer-state", dest="save_optimizer_state",
                       action="store_false",
                       help="checkpoint 不保存优化器状态")
    parser.add_argument("--best-checkpoint-dir", type=Path,
                       help="滚动最佳 model-only 缓存目录（建议 /dev/shm）")

    args = parser.parse_args()

    # 验证
    try:
        validate_best_eval_path(args.best_checkpoint_dir, args.eval_path)
    except ValueError as error:
        parser.error(str(error))

    if args.best_checkpoint_dir is not None:
        args.save_optimizer_state = False

    if args.optimizer_device == "cpu" and args.optimizer_name == "muon":
        parser.error("CPU optimizer offload only supports adam/adamw")

    for name in ("num_steps", "prompts_per_batch", "group_size", "gradient_accumulation_steps",
                 "max_new_tokens", "eval_every", "eval_examples", "save_every", "generation_batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")

    if args.group_size < 2:
        parser.error("group_size must be at least 2")

    if (args.prompts_per_batch * args.group_size) % args.gradient_accumulation_steps:
        parser.error("rollout batch size (prompts_per_batch * group_size) must be divisible by gradient_accumulation_steps")

    if args.learning_rate <= 0 or args.max_grad_norm <= 0 or args.temperature <= 0:
        parser.error("lr / max_grad_norm / temperature must be positive")

    if args.weight_decay < 0:
        parser.error("weight_decay must be nonnegative")

    if args.cliprange <= 0:
        parser.error("cliprange must be positive")

    if args.num_epochs_per_rollout <= 0:
        parser.error("num_epochs_per_rollout must be positive")

    if args.training_backend == "fsdp2":
        from scripts.fsdp_grpo_runtime import validate_fsdp_config
        try:
            validate_fsdp_config(args)
        except ValueError as error:
            parser.error(str(error))

    return UnifiedGRPOConfig(**vars(args))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    train(parse_args())
