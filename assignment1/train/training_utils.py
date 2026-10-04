"""Training utilities: data loading, loss, gradient clipping, lr schedule."""

import math
import os
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch import Tensor


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic ops may impact performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample a batch of (input, target) from the dataset.

    Args:
        dataset: 1D array of token ids
        batch_size: number of sequences
        context_length: sequence length
        device: where to place tensors

    Returns:
        (input_ids, target_ids), each of shape (batch_size, context_length)
    """
    max_start = len(dataset) - context_length
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([torch.from_numpy(dataset[s : s + context_length].copy()) for s in starts])
    y = torch.stack([torch.from_numpy(dataset[s + 1 : s + 1 + context_length].copy()) for s in starts])
    return x.to(device=device, dtype=torch.long), y.to(device=device, dtype=torch.long)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable softmax."""
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy loss (manual implementation matching PyTorch).

    Args:
        logits: (batch, seq_len, vocab_size) unnormalized
        targets: (batch, seq_len) token ids

    Returns:
        scalar loss
    """
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(batch_size * seq_len, vocab_size)
    targets_flat = targets.view(batch_size * seq_len)

    log_probs = torch.log_softmax(logits_flat, dim=-1)
    nll = -log_probs[torch.arange(len(targets_flat), device=logits.device), targets_flat]
    return nll.mean()


def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    """Clip gradients by global L2 norm (in-place).

    Args:
        parameters: model parameters with .grad
        max_l2_norm: clip threshold
    """
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return
    total_norm = torch.sqrt(sum(p.grad.detach().pow(2).sum() for p in params))
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            p.grad.detach().mul_(clip_coef)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing with linear warmup.

    Args:
        optimizer: optimizer to schedule
        num_warmup_steps: linear ramp from 0 to base_lr
        num_training_steps: total steps
        min_lr_ratio: final_lr / base_lr

    Returns:
        LambdaLR scheduler (call .step() after optimizer.step())
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = (current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    path: str,
) -> None:
    """Save training checkpoint."""
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
) -> int:
    """Load checkpoint and return iteration number."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
