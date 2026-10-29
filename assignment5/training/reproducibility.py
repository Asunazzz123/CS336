"""Random-state setup and precision scopes."""
import random
from contextlib import nullcontext
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(model: torch.nn.Module, autocast: str):
    """FP32 主权重 + BF16 autocast 前向的混合精度入口，仅在 CUDA 上生效。"""
    if autocast == "none":
        return nullcontext()
    if autocast != "bfloat16":
        raise ValueError(f"未知 autocast 类型：{autocast}")
    if next(model.parameters()).device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
