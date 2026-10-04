#!/usr/bin/env python3
"""Training script for Transformer LM on TinyStories or OWT.

Supports single-GPU and multi-GPU DDP training (auto-detected via RANK/WORLD_SIZE).

Example usage:
    # Single GPU
    python -m train.train --data_path data/tinystories_train.npy \\
        --val_data_path data/tinystories_val.npy --vocab_size 10000

    # Multi-GPU DDP (torchrun auto-sets RANK/WORLD_SIZE/LOCAL_RANK)
    torchrun --nproc_per_node=8 -m train.train --data_path ... --batch_size 32

Training checkpoints, logs, and generated samples are saved to --output_dir.
"""

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root to path so we can import cs336_basics
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.model import TransformerLM
from train.training_utils import (
    cross_entropy,
    get_batch,
    get_cosine_schedule_with_warmup,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)
from train.optimizer import TinyAdamW
from train.performance import PerformanceProbe


def setup_ddp():
    """Initialize DDP if running in distributed mode, return (rank, world_size, local_rank)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def log_print(rank: int, *args, **kwargs):
    """Print only from rank 0."""
    if rank == 0:
        print(*args, **kwargs)


@torch.no_grad()
def evaluate(model: nn.Module, val_data: np.ndarray, args, device) -> float:
    """Compute validation loss over a fixed number of batches."""
    model.eval()
    total_loss = 0.0
    num_eval_batches = args.eval_batches

    for _ in range(num_eval_batches):
        x, y = get_batch(val_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        total_loss += loss.item()

    model.train()
    return total_loss / num_eval_batches


def generate_sample(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device,
) -> str:
    """Generate text continuation (greedy or temperature sampling)."""
    model.eval()
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits = model(input_ids)  # (1, seq_len, vocab_size)
        next_token_logits = logits[0, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        # Stop at <|endoftext|>
        if next_token.item() == tokenizer.encode("<|endoftext|>")[0]:
            break

    model.train()
    generated_ids = input_ids[0].tolist()
    return tokenizer.decode(generated_ids)


def main():
    parser = argparse.ArgumentParser(description="Train Transformer LM")

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to train .npy")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to val .npy")
    parser.add_argument("--vocab_path", type=str, help="Tokenizer vocab.json (optional for generation)")
    parser.add_argument("--merges_path", type=str, help="Tokenizer merges.txt (optional)")

    # Model
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # Evaluation & logging
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--sample_interval", type=int, default=1000)
    parser.add_argument("--sample_prompt", type=str, default="Once upon a time")
    parser.add_argument("--probe_warmup_steps", type=int, default=5,
                        help="Initial steps excluded from single-device performance statistics")
    parser.add_argument("--disable_probe", action="store_true",
                        help="Disable synchronized phase timing and performance files")

    # Misc
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--resume_from", type=str, help="Resume from checkpoint path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true", help="torch.compile model (needs PyTorch 2.0+)")

    args = parser.parse_args()
    if args.probe_warmup_steps < 0:
        parser.error("--probe_warmup_steps must be nonnegative")

    # Setup distributed
    rank, world_size, local_rank = setup_ddp()
    is_main = rank == 0

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        if world_size > 1:
            raise RuntimeError("DDP requires CUDA")

    log_print(rank, f"[Rank {rank}/{world_size}] Using device: {device}")

    # Create output directory
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        # Save config
        with open(f"{args.output_dir}/config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    # Seed
    set_seed(args.seed + rank)  # Different seed per rank for data sampling

    # Load data (memory-mapped)
    train_data = np.load(args.data_path, mmap_mode="r")
    val_data = np.load(args.val_data_path, mmap_mode="r")
    log_print(rank, f"Loaded train: {len(train_data):,} tokens, val: {len(val_data):,} tokens")

    # Build model
    log_print(rank, "Building model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )

    if args.compile and hasattr(torch, "compile"):
        log_print(rank, "Compiling model with torch.compile...")
        model = torch.compile(model)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    num_params = sum(p.numel() for p in model.parameters())
    log_print(rank, f"Model parameters: {num_params:,}")

    # Optimizer & scheduler
    optimizer = TinyAdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_iters,
        num_training_steps=args.max_iters,
        min_lr_ratio=args.min_lr_ratio,
    )

    # Resume
    start_iter = 0
    if args.resume_from:
        log_print(rank, f"Resuming from {args.resume_from}")
        raw_model = model.module if isinstance(model, DDP) else model
        start_iter = load_checkpoint(args.resume_from, raw_model, optimizer, device)
        # Fast-forward scheduler
        for _ in range(start_iter):
            scheduler.step()
        log_print(rank, f"Resumed at iteration {start_iter}")

    # Optional tokenizer for generation
    tokenizer = None
    if is_main and args.vocab_path and args.merges_path:
        try:
            from train.tokenizer import Tokenizer

            with open(args.vocab_path) as f:
                vocab = json.load(f)
            with open(args.merges_path) as f:
                merges = [tuple(line.strip().split()) for line in f if line.strip()]
            tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
            log_print(rank, "Loaded tokenizer for generation")
        except Exception as e:
            log_print(rank, f"Tokenizer load failed: {e}")

    # Training loop
    log_print(rank, "Starting training...")
    model.train()
    probe = PerformanceProbe(device, args.probe_warmup_steps) if world_size == 1 and not args.disable_probe else None
    run_id = time.time_ns()
    metrics_path = Path(args.output_dir) / "metrics.jsonl"
    run_started = time.perf_counter()
    overhead_seconds = {"validation": 0.0, "checkpoint": 0.0, "generation": 0.0}

    def write_metrics(event: str, **values):
        if probe is not None:
            record = {"event": event, "run_id": run_id,
                      "elapsed_seconds": time.perf_counter() - run_started, **values}
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")

    write_metrics("start", device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                  torch_version=torch.__version__, num_params=num_params, start_iteration=start_iter,
                  warmup_steps=args.probe_warmup_steps, tokens_per_step=args.batch_size * args.context_length,
                  compile=args.compile, timing_mode="synchronized_phases")

    for iteration in range(start_iter, args.max_iters):
        local_step = iteration - start_iter
        timings = {}
        step_started = probe.begin_step(local_step) if probe is not None else None
        phase = lambda name: probe.phase(timings, name) if probe is not None else nullcontext()
        lr_used = optimizer.param_groups[0]["lr"]
        # Get batch
        with phase("data_loading"):
            x, y = get_batch(train_data, args.batch_size, args.context_length, device)

        # Forward
        with phase("forward_loss"):
            logits = model(x)
            loss = cross_entropy(logits, y)

        # Backward
        with phase("backward"):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

        # Gradient clipping
        with phase("gradient_clipping"):
            if args.grad_clip > 0:
                gradient_clipping(model.parameters(), args.grad_clip)

        # Optimizer step
        with phase("optimizer_scheduler"):
            optimizer.step()
            scheduler.step()
        if probe is not None:
            probe.end_step(step_started, local_step, x.numel(), timings)

        # Logging
        if ((iteration + 1) % args.log_interval == 0 or iteration + 1 == args.max_iters) and is_main:
            lr = optimizer.param_groups[0]["lr"]
            stats = probe.summary() if probe is not None else {}
            speed = (f" | step {stats['mean_step_seconds']:.4f}s | {stats['tokens_per_second']:.0f} tokens/s"
                     if stats.get("measured_steps") else "")
            log_print(
                rank,
                f"iter {iteration + 1}/{args.max_iters} | "
                f"loss {loss.item():.4f} | lr {lr:.2e} | "
                f"elapsed {time.perf_counter() - run_started:.2f}s{speed}",
            )
            write_metrics("train", iteration=iteration + 1, loss=loss.item(), lr_used=lr_used, next_lr=lr,
                          total_tokens_processed=(iteration + 1) * args.batch_size * args.context_length,
                          session_tokens_processed=(local_step + 1) * args.batch_size * args.context_length,
                          **stats)

        # Evaluation
        if (iteration + 1) % args.eval_interval == 0 and is_main:
            raw_model = model.module if isinstance(model, DDP) else model
            with phase("validation"):
                val_loss = evaluate(raw_model, val_data, args, device)
            overhead_seconds["validation"] += timings.get("validation", 0.0)
            write_metrics("validation", iteration=iteration + 1, val_loss=val_loss,
                          seconds=timings.get("validation"))
            log_print(rank, f"[Eval] iter {iteration + 1} | val_loss {val_loss:.4f}")

        # Generation sample
        if (iteration + 1) % args.sample_interval == 0 and is_main and tokenizer:
            raw_model = model.module if isinstance(model, DDP) else model
            with phase("generation"):
                sample = generate_sample(
                    raw_model, tokenizer, args.sample_prompt, max_new_tokens=50, temperature=0.8, device=device
                )
            overhead_seconds["generation"] += timings.get("generation", 0.0)
            write_metrics("generation", iteration=iteration + 1, seconds=timings.get("generation"))
            log_print(rank, f"[Sample] iter {iteration + 1}:\n{sample}\n")

        # Save checkpoint
        if (iteration + 1) % args.save_interval == 0 and is_main:
            ckpt_path = f"{args.output_dir}/checkpoint_{iteration + 1}.pt"
            raw_model = model.module if isinstance(model, DDP) else model
            with phase("checkpoint"):
                save_checkpoint(raw_model, optimizer, iteration + 1, ckpt_path)
            overhead_seconds["checkpoint"] += timings.get("checkpoint", 0.0)
            write_metrics("checkpoint", iteration=iteration + 1, path=ckpt_path, seconds=timings.get("checkpoint"))
            log_print(rank, f"Saved checkpoint to {ckpt_path}")

    # Final save
    if is_main:
        final_path = f"{args.output_dir}/final_model.pt"
        raw_model = model.module if isinstance(model, DDP) else model
        final_timings = {}
        with probe.phase(final_timings, "checkpoint") if probe is not None else nullcontext():
            save_checkpoint(raw_model, optimizer, args.max_iters, final_path)
        overhead_seconds["checkpoint"] += final_timings.get("checkpoint", 0.0)
        write_metrics("checkpoint", iteration=args.max_iters, path=final_path,
                      seconds=final_timings.get("checkpoint"), final=True)
        if probe is not None:
            summary = {"run_id": run_id, "device": str(device), "start_iteration": start_iter,
                       "final_iteration": args.max_iters,
                       "total_tokens_processed": args.max_iters * args.batch_size * args.context_length,
                       "session_tokens_processed": max(0, args.max_iters - start_iter) * args.batch_size * args.context_length,
                       "session_wall_seconds": time.perf_counter() - run_started,
                       "overhead_seconds": overhead_seconds, **probe.summary()}
            with (Path(args.output_dir) / "performance_summary.json").open("w", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2)
            write_metrics("summary", **summary)
        log_print(rank, f"Training complete. Final model saved to {final_path}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
