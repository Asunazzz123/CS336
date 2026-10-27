"""Configuration independent of any SFT/RL entry point or dataset."""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class FSDPConfig:
    block_class_names: tuple[str, ...] = ()  # Otherwise use HF _no_split_modules.
    master_dtype: torch.dtype = torch.float32
    compute_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.float32
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    max_grad_norm: float | None = 1.0
    gradient_checkpointing: bool = False
    seed: int = 42
    timeout_seconds: int = 180

    def __post_init__(self):
        if not isinstance(self.block_class_names, tuple) or any(
            not isinstance(name, str) or not name for name in self.block_class_names
        ):
            raise ValueError("block_class_names must be a tuple of nonempty class names")
        if self.master_dtype != torch.float32:
            raise ValueError("This engine keeps FP32 master shards and AdamW moments")
        if self.compute_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("compute_dtype must be float32 or bfloat16")
        if self.reduce_dtype != torch.float32:
            raise ValueError("Gradient reduction must use float32")
        for name in ("learning_rate", "eps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if len(self.betas) != 2 or any(not 0 <= b < 1 for b in self.betas):
            raise ValueError("betas must lie in [0, 1)")
        if self.max_grad_norm is not None and (
            not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0
        ):
            raise ValueError("max_grad_norm must be positive or None")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 32
    temperature: float = 0.0
    eos_token_id: int | None = None
    pad_token_id: int = 0
    offload_optimizer: bool = True

    def __post_init__(self):
        if not isinstance(self.max_new_tokens, int) or isinstance(self.max_new_tokens, bool) or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if self.pad_token_id is None or any(not isinstance(value, int) or isinstance(value, bool) or value < 0
               for value in (self.pad_token_id, self.eos_token_id) if value is not None):
            raise ValueError("Token IDs must be nonnegative")
