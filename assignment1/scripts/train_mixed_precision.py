#!/usr/bin/env python3
"""Train with BF16 autocast by default, or --precision fp16 on CUDA.

Run from assignment1 (all normal train.train arguments are supported):
    conda run -n agent python scripts/train_mixed_precision.py \
        --data_path data/encoded/tinystories-10k/train.npy \
        --val_data_path data/encoded/tinystories-10k/valid.npy --vocab_size 10000

Multi-GPU:
    conda run -n agent python -m torch.distributed.run --nproc_per_node=2 \
        scripts/train_mixed_precision.py --data_path ... --val_data_path ... \
        --vocab_size 10000 --precision bf16

Parameters, gradients and AdamW state remain FP32. FP16 uses GradScaler;
BF16 does not need loss scaling. CPU BF16 is available for smoke checks.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.train import main


if __name__ == "__main__":
    main(default_precision="bf16", default_output_dir="outputs/mixed_precision")
