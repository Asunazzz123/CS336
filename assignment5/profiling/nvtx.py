"""Opt-in NVTX annotations; no timing, synchronization, or tensor retention."""
import os
from contextlib import nullcontext

import torch
from utils.device import nvidia_cuda_available


def nvtx_range(name: str, device: torch.device | str):
    """Return a balanced NVTX scope only for explicitly enabled CUDA runs.

    These are CPU submission ranges, not synchronized GPU elapsed times.
    Set CS336_NVTX=1 before launching under Nsight Systems.
    """
    if (os.environ.get("CS336_NVTX") == "1"
            and torch.device(device).type == "cuda"
            and nvidia_cuda_available()):
        return torch.cuda.nvtx.range(name)
    return nullcontext()
