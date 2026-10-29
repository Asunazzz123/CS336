"""Single-device SFT application; distributed execution is a backend adapter."""
from __future__ import annotations
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
import torch
from alignment.sft import collate_sft_batch, encode_sft_examples
from training.losses import sft_loss_sum as _loss_sum
from training.evaluation.sft import evaluate_sft
from training.configs import SFTConfig, RuntimeConfig as TrainConfig
from training.types import Question
from training.data.sft import prepare_examples
from training.data.artifacts import write_sft_data
from training.data.gsm8k import validate_best_data
from training.backends.single_device import build_runtime
from training.evaluation.generation import evaluate
from training.reproducibility import seed_everything
from training.checkpoint_policy import save_checkpoint, make_best_store, consider_best, finish_best
from training.validation import validate_sft_config
from profiling.nvtx import nvtx_range
from profiling.metrics import log_metrics
from utils.serialization import json_value


def _runtime_config(config: SFTConfig, prompt_path: Path) -> TrainConfig:
    return TrainConfig(
        model_id=config.model_id,
        train_path=config.output_dir / "data/train.jsonl",
        eval_path=config.output_dir / "data/validation.jsonl",
        prompt_path=prompt_path,
        output_dir=config.output_dir,
        device=config.device,
        dtype=config.dtype,
        seed=config.seed,
        generation_backend="transformers",
        optimizer_name=config.optimizer_name,
        optimizer_device=config.optimizer_device,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_new_tokens=config.max_new_tokens,
        generation_batch_size=config.generation_batch_size,
        reward_type="gsm8k_numeric",
        autocast=config.autocast,
        save_optimizer_state=config.save_optimizer_state,
        best_checkpoint_dir=config.best_checkpoint_dir,
    )


def train_sft_step(model, optimizer, records, pad_token_id, microbatch_size, max_grad_norm=1.0,
                   autocast: str = "none"):
    """One optimizer update, normalized by all response tokens in its actual batch."""
    if not records or microbatch_size <= 0:
        raise ValueError("SFT records and a positive microbatch_size are required")
    total_tokens = sum(sum(label != -100 for label in record["labels"][1:]) for record in records)
    if total_tokens == 0:
        raise ValueError("The SFT update has no supervised tokens")
    model.train()
    device = next(model.parameters()).device
    with nvtx_range("sft.zero_grad.begin", device):
        optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=device)
    for start in range(0, len(records), microbatch_size):
        with nvtx_range("sft.microbatch", device):
            with nvtx_range("sft.collate", device):
                batch = collate_sft_batch(records[start:start + microbatch_size], pad_token_id)
            loss_sum, _ = _loss_sum(model, batch, autocast)
            with nvtx_range("sft.loss_check", device):
                if not torch.isfinite(loss_sum):
                    raise ValueError("Non-finite SFT loss")
            with nvtx_range("sft.backward", device):
                (loss_sum / total_tokens).backward()
            total_loss += loss_sum.detach() / total_tokens
    metadata = {"supervised_tokens": total_tokens, "examples": len(records)}
    if max_grad_norm is not None:
        with nvtx_range("sft.clip_grad", device):
            metadata["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm, error_if_nonfinite=True,
            ).detach()
    with nvtx_range("sft.optimizer_step", device):
        optimizer.step()
    with nvtx_range("sft.zero_grad.end", device):
        optimizer.zero_grad(set_to_none=True)
    return total_loss, metadata


def train(config: SFTConfig) -> Path:
    validate_sft_config(config)
    if config.training_backend == "fsdp2":
        from training.backends.fsdp_sft import train_fsdp_sft
        return train_fsdp_sft(config)
    if config.training_backend != "transformers":
        raise ValueError("Unknown training backend")
    seed_everything(config.seed)
    train_examples, validation, template, prompt_path = prepare_examples(config)
    validate_best_data(config, train_examples, validation, config.eval_path)
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise ValueError("输出目录非空，请选择新的 output-dir")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_sft_data(config, train_examples, validation)
    (config.output_dir / "run_config.json").write_text(
        json.dumps(json_value(asdict(config)), indent=2), encoding="utf-8",
    )
    stats = {"train_examples": len(train_examples), "validation_examples": len(validation),
             "source_train_path": str(config.train_path),
             "source_eval_path": str(config.eval_path) if config.eval_path is not None else None,
             "seed": config.seed, "response_format": config.response_format,
             "prompt_path": str(prompt_path), "tokenized": False}
    stats_path = config.output_dir / "data_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if config.prepare_only:
        return config.output_dir

    runtime_config = _runtime_config(config, prompt_path)
    runtime = build_runtime(runtime_config)
    # Avoid the additional CUDA foreach parameter-sized optimizer temporaries.
    if config.optimizer_name in ("adam", "adamw"):
        for group in runtime.optimizer.param_groups:
            group["foreach"] = False
    train_records, train_stats = encode_sft_examples(train_examples, runtime.tokenizer, config.max_seq_length)
    validation_records, validation_stats = encode_sft_examples(validation, runtime.tokenizer, config.max_seq_length)
    stats.update({"tokenized": True, "train_tokens": train_stats, "validation_tokens": validation_stats})
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if not train_records or not validation_records:
        raise ValueError("No examples remain after length filtering; increase max_seq_length")
    best = make_best_store(config, "loss", "min")
    context_limit = getattr(runtime.model.config, "max_position_embeddings", None)
    longest = max(len(r["input_ids"]) for r in train_records + validation_records)
    if context_limit and longest > context_limit:
        raise ValueError("SFT sequences exceed the model context limit")
    if config.gradient_checkpointing:
        runtime.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def validate(step):
        with nvtx_range("sft.evaluate", runtime.model.device):
            metrics = evaluate_sft(runtime.model, validation_records[:config.eval_examples],
                                   runtime.tokenizer.pad_token_id, config.batch_size, config.autocast)
        log_metrics(config.output_dir, step, "eval", metrics)
        consider_best(runtime, config, best, step, metrics, eval_path=runtime_config.eval_path,
                      eval_examples=metrics["num_examples"])
        if config.eval_generation_examples:
            questions = [Question(e.question, e.ground_truth) for e in validation[:config.eval_generation_examples]]
            log_metrics(config.output_dir, step, "eval_generation",
                        evaluate(runtime, questions, template, runtime_config))

    validate(0)
    step, saved_step = 0, None
    update_size = config.batch_size * config.gradient_accumulation_steps
    for epoch in range(config.num_epochs):
        indices = list(range(len(train_records)))
        random.Random(config.seed + epoch).shuffle(indices)
        for start in range(0, len(indices), update_size):
            selected = [train_records[i] for i in indices[start:start + update_size]]
            started = time.monotonic()
            with nvtx_range("sft.update", runtime.model.device):
                loss, metadata = train_sft_step(runtime.model, runtime.optimizer, selected,
                                                runtime.tokenizer.pad_token_id, config.batch_size,
                                                config.max_grad_norm, config.autocast)
                if runtime.model.device.type == "cuda":
                    torch.cuda.synchronize(runtime.model.device)
            step += 1
            log_metrics(config.output_dir, step, "train", {
                **metadata, "loss": loss, "epoch": epoch + 1,
                "update_seconds": time.monotonic() - started,
            })
            final_step = (config.max_steps is not None and step >= config.max_steps) or (
                epoch == config.num_epochs - 1 and start + update_size >= len(indices)
            )
            if step % config.eval_every == 0 or final_step:
                validate(step)
            if best is None and (step % config.save_every == 0 or final_step):
                with nvtx_range("sft.save_checkpoint", runtime.model.device):
                    save_checkpoint(runtime, config, step)
                saved_step = step
            if final_step:
                if best is not None:
                    return finish_best(best, config.output_dir)
                return config.output_dir / f"checkpoint-{saved_step:06d}"
    raise RuntimeError("SFT finished without an optimizer update")
