#!/usr/bin/env python3
"""Run the original 64 TinyStories configurations with BF16 AMP by default.

From assignment1:
    conda run -n agent python scripts/train_mixed_precision_sweep.py --plan-only
    conda run -n agent python scripts/train_mixed_precision_sweep.py

The model, parameter grid, token budget and fixed validation windows match
the original FP32 sweep. Use --precision fp16 for CUDA FP16 instead.
Results default to outputs/sweep_tinystories_10k_bf16 (or _fp16).
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_sweep import main


if __name__ == "__main__":
    main(default_precision="bf16")
