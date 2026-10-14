"""Autocast with FP32 parameters and optional FP16 gradient scaling."""

from contextlib import nullcontext

import torch


class MixedPrecision:
    def __init__(self, device: torch.device, precision: str):
        self.device = torch.device(device)
        self.name = precision
        if precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"Unsupported precision: {precision}")
        if precision == "fp16" and self.device.type != "cuda":
            raise ValueError("FP16 training requires CUDA; use --precision bf16 or fp32 on CPU")
        if precision == "bf16" and self.device.type == "cuda":
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("This CUDA device does not support BF16; use --precision fp16 or fp32")
        self.dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
        self.scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")

    def autocast(self):
        if self.name == "fp32":
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def unscale_(self, optimizer):
        # Clip the true gradients, rather than the loss-scaled gradients.
        self.scaler.unscale_(optimizer)

    def step(self, optimizer, scheduler) -> bool:
        previous_scale = self.scaler.get_scale()
        self.scaler.step(optimizer)
        self.scaler.update()
        updated = self.scaler.get_scale() >= previous_scale
        if updated:
            scheduler.step()
        return updated
