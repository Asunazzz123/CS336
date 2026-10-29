"""Configuration schemas; no model loading or process initialization."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class EvaluationConfig:
    """Generation/scoring settings without optimizer or training-loop fields."""
    device: str = "cuda:0"
    generation_backend: str = "transformers"
    seed: int = 42
    max_new_tokens: int = 512
    generation_batch_size: int = 1
    reward_type: str = "gsm8k_numeric"
    autocast: str = "none"
    group_size: int = 1
    temperature: float = 0.0


@dataclass
class RuntimeConfig:
    model_id: str
    train_path: Path
    eval_path: Path
    prompt_path: Path
    output_dir: Path
    device: str = "cpu"
    generation_backend: str = "transformers"
    seed: int = 42
    num_steps: int = 2
    prompts_per_batch: int = 2
    group_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    temperature: float = 1.0
    max_new_tokens: int = 128
    eval_every: int = 1
    eval_examples: int = 4
    save_every: int = 1
    optimizer_name: str = "adamw"
    optimizer_device: str = "model"
    weight_decay: float = 0.0
    dtype: str = "float32"
    sampling_strategy: str = "random"
    reward_type: str = "gsm8k_numeric"
    generation_batch_size: int = 2
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000
    vllm_gpu: int = 1
    # 默认优势只减组均值，loss 按本次更新的全批 response token 数平均。
    # 与旧实验的 std/sequence 不同；可用 CLI 显式恢复旧目标。
    advantage_normalizer: str = "none"
    loss_normalization: str = "constant"
    # 配合 --dtype float32 实现 FP32 主权重 + BF16 autocast 计算，
    # 使 lr~1e-6 量级的更新不被 BF16 舍入吞掉
    autocast: str = "none"
    gradient_checkpointing: bool = False
    save_optimizer_state: bool = True
    best_checkpoint_dir: Path | None = None


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


@dataclass
class SFTConfig:
    model_id: str
    train_path: Path
    output_dir: Path
    prompt_path: Path | None = None
    eval_path: Path | None = None
    validation_fraction: float = 0.1
    device: str = "cpu"
    dtype: str = "float32"
    seed: int = 42
    num_epochs: int = 1
    max_steps: int | None = None
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    optimizer_name: str = "adamw"
    optimizer_device: str = "model"
    weight_decay: float = 0.0
    max_seq_length: int = 1024
    eval_every: int = 50
    save_every: int = 100
    eval_examples: int = 64
    eval_generation_examples: int = 0
    max_new_tokens: int = 512
    generation_batch_size: int = 1
    response_format: str = "r1_zero"
    gradient_checkpointing: bool = False
    prepare_only: bool = False
    # 配合 --dtype float32 实现 FP32 主权重 + BF16 autocast 计算
    autocast: str = "none"
    save_optimizer_state: bool = True
    best_checkpoint_dir: Path | None = None
    training_backend: str = "transformers"
