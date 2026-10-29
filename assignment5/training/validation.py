"""Pure configuration checks shared by CLIs and trainers."""
import math
from training.configs import SFTConfig
from training.data.gsm8k import validate_best_eval_path


def validate_sft_config(config: SFTConfig) -> None:
    validate_best_eval_path(config.best_checkpoint_dir, config.eval_path)
    for name in (
        "num_epochs", "batch_size", "gradient_accumulation_steps", "max_seq_length",
        "eval_every", "save_every", "eval_examples", "max_new_tokens", "generation_batch_size",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.max_seq_length < 2 or (config.max_steps is not None and config.max_steps <= 0):
        raise ValueError("max_seq_length must be >= 2; max_steps must be positive when set")
    if config.eval_generation_examples < 0:
        raise ValueError("eval_generation_examples must be nonnegative")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(config.max_grad_norm) or config.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if config.response_format not in ("r1_zero", "boxed"):
        raise ValueError("response_format must be r1_zero or boxed")
    if config.dtype not in ("float32", "bfloat16", "float16"):
        raise ValueError("Unsupported dtype")
    if config.optimizer_name not in ("adam", "adamw", "muon"):
        raise ValueError("Unsupported optimizer_name")
    if config.dtype == "float16":
        raise ValueError("float16 SFT requires loss scaling, which this runner does not implement; use bfloat16 or float32")


def validate_fsdp_config(config, world_size=None):
    for name in ("num_steps", "prompts_per_batch", "group_size", "gradient_accumulation_steps",
                 "max_new_tokens", "generation_batch_size", "eval_examples", "eval_every", "save_every",
                 "num_epochs_per_rollout"):
        if hasattr(config, name) and getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.group_size < 2:
        raise ValueError("group_size must be >=2")
    for name in ("temperature", "cliprange"):
        if hasattr(config, name) and (not math.isfinite(getattr(config, name)) or getattr(config, name) <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if config.generation_backend not in ("transformers", "vllm"):
        raise ValueError("Unknown generation backend")
    if config.generation_backend == "vllm":
        utilization = getattr(config, "vllm_memory_utilization", 0.8)
        if not math.isfinite(utilization) or not 0 < utilization < 1:
            raise ValueError("vllm_memory_utilization must be between 0 and 1")
        for name in ("vllm_startup_timeout", "vllm_max_model_len"):
            if getattr(config, name, 1) <= 0:
                raise ValueError(f"{name} must be positive")
    if config.optimizer_name != "adamw" or config.optimizer_device != "model":
        raise ValueError("fsdp2 requires GPU-sharded AdamW (--optimizer-name adamw --optimizer-device model)")
    if config.dtype not in ("float32", "bfloat16"):
        raise ValueError("fsdp2 supports float32/bfloat16 compute and FP32 master shards")
    if world_size is not None:
        if config.prompts_per_batch % world_size:
            raise ValueError("prompts_per_batch must be divisible by world_size to keep groups intact")
        local_size = config.prompts_per_batch // world_size * config.group_size
        if not local_size or local_size % config.gradient_accumulation_steps:
            raise ValueError("Per-rank rollout batch must be divisible by gradient_accumulation_steps")


def validate_config(config):
    """Validate CLI and programmatic calls before loading data/models."""
    validate_best_eval_path(config.best_checkpoint_dir, config.eval_path)
    for name in ("num_steps", "prompts_per_batch", "group_size", "gradient_accumulation_steps",
                 "max_new_tokens", "eval_every", "eval_examples", "save_every", "generation_batch_size",
                 "num_epochs_per_rollout"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config.group_size < 2:
        raise ValueError("group_size must be at least 2")
    if config.prompts_per_batch * config.group_size % config.gradient_accumulation_steps:
        raise ValueError("rollout batch size must be divisible by gradient_accumulation_steps")
    for name in ("learning_rate", "max_grad_norm", "temperature", "cliprange"):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    choices = {"training_backend": ("transformers", "fsdp2"), "generation_backend": ("transformers", "vllm"),
        "dtype": ("float32", "bfloat16"), "autocast": ("none", "bfloat16"),
        "optimizer_name": ("adam", "adamw", "muon"), "optimizer_device": ("model", "cpu"),
        "sampling_strategy": ("random", "order"), "reward_type": ("gsm8k_numeric", "r1_zero", "question_only"),
        "baseline": ("mean", "none"), "advantage_normalizer": ("std", "none", "mean"),
        "loss_normalization": ("sequence", "constant"), "checkpoint_format": ("hf", "dcp"),
        "importance_reweighting_method": ("none", "noclip", "grpo", "gspo")}
    for name, allowed in choices.items():
        if getattr(config, name) not in allowed:
            raise ValueError(f"Invalid {name}: {getattr(config, name)}")
    if config.optimizer_device == "cpu" and config.optimizer_name == "muon":
        raise ValueError("CPU optimizer offload only supports adam/adamw")
    if config.training_backend == "fsdp2":
        validate_fsdp_config(config)
