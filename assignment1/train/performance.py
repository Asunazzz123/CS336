"""Synchronized single-device training measurements, excluding warmup steps."""

import math
import time
from contextlib import contextmanager

import torch


class PerformanceProbe:
    def __init__(self, device: torch.device, warmup_steps: int = 5):
        if warmup_steps < 0:
            raise ValueError("probe warmup steps must be nonnegative")
        self.device = torch.device(device)
        self.warmup_steps = warmup_steps
        self.warmup_steps_seen = 0
        self.steps = 0
        self.tokens = 0
        self.step_seconds = 0.0
        self.step_seconds_squared = 0.0
        self.phase_seconds = {}
        self.peak_allocated = 0 if self.device.type == "cuda" else None
        self.peak_reserved = 0 if self.device.type == "cuda" else None

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def phase(self, timings: dict, name: str):
        self.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            timings[name] = time.perf_counter() - started

    def begin_step(self, local_step: int):
        self.synchronize()
        if local_step >= self.warmup_steps and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        return time.perf_counter()

    def end_step(self, started: float, local_step: int, tokens: int, timings: dict):
        self.synchronize()
        elapsed = time.perf_counter() - started
        if local_step < self.warmup_steps:
            self.warmup_steps_seen += 1
            return
        self.steps += 1
        self.tokens += tokens
        self.step_seconds += elapsed
        self.step_seconds_squared += elapsed * elapsed
        for name, seconds in timings.items():
            self.phase_seconds[name] = self.phase_seconds.get(name, 0.0) + seconds
        if self.device.type == "cuda":
            self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated(self.device))
            self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved(self.device))

    def summary(self):
        mean = self.step_seconds / self.steps if self.steps else None
        variance = max(0.0, self.step_seconds_squared / self.steps - mean * mean) if self.steps else None
        return {
            "measured_steps": self.steps,
            "measured_tokens": self.tokens,
            "warmup_steps_excluded": self.warmup_steps_seen,
            "mean_step_seconds": mean,
            "std_step_seconds": math.sqrt(variance) if variance is not None else None,
            "training_step_seconds": self.step_seconds,
            "tokens_per_second": self.tokens / self.step_seconds if self.step_seconds > 0 else None,
            "mean_phase_seconds": {name: value / self.steps for name, value in self.phase_seconds.items()},
            "peak_allocated_bytes": self.peak_allocated,
            "peak_reserved_bytes": self.peak_reserved,
        }
