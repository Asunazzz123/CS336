"""Single-device GSM8K SFT with response-only loss and Hugging Face checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alignment.sft import (
    SFTExample,
    collate_sft_batch,
    encode_sft_examples,
    load_gsm8k_examples,
    split_examples,
)
from scripts.train_grpo_base import (
    Question,
    TrainConfig,
    autocast_context,
    build_runtime,
    evaluate,
    json_value,
    log_metrics,
    save_checkpoint,
    seed_everything,
    validate_best_eval_path, validate_best_data, make_best_store, consider_best, finish_best,
)
from scripts.profiling import nvtx_range


@dataclass
class SFTConfig:
    model_id: str
    train_path: Path
    output_dir: Path
    prompt_path: Path | None = None
    eval_path: Path | None = None
    validation_fraction: float = 0.1
    device: str = "cpu"
    dtype: str = "float32"
    seed: int = 42
    num_epochs: int = 1
    max_steps: int | None = None
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    optimizer_name: str = "adamw"
    optimizer_device: str = "model"
    weight_decay: float = 0.0
    max_seq_length: int = 1024
    eval_every: int = 50
    save_every: int = 100
    eval_examples: int = 64
    eval_generation_examples: int = 0
    max_new_tokens: int = 512
    generation_batch_size: int = 1
    response_format: str = "r1_zero"
    gradient_checkpointing: bool = False
    prepare_only: bool = False
    # 配合 --dtype float32 实现 FP32 主权重 + BF16 autocast 计算
    autocast: str = "none"
    save_optimizer_state: bool = True
    best_checkpoint_dir: Path | None = None
    training_backend: str = "transformers"


def _validate_config(config: SFTConfig) -> None:
    validate_best_eval_path(config.best_checkpoint_dir, config.eval_path)
    for name in (
        "num_epochs", "batch_size", "gradient_accumulation_steps", "max_seq_length",
        "eval_every", "save_every", "eval_examples", "max_new_tokens", "generation_batch_size",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.max_seq_length < 2 or (config.max_steps is not None and config.max_steps <= 0):
        raise ValueError("max_seq_length must be >= 2; max_steps must be positive when set")
    if config.eval_generation_examples < 0:
        raise ValueError("eval_generation_examples must be nonnegative")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(config.max_grad_norm) or config.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if config.response_format not in ("r1_zero", "boxed"):
        raise ValueError("response_format must be r1_zero or boxed")
    if config.dtype not in ("float32", "bfloat16", "float16"):
        raise ValueError("Unsupported dtype")
    if config.optimizer_name not in ("adam", "adamw", "muon"):
        raise ValueError("Unsupported optimizer_name")
    if config.dtype == "float16":
        raise ValueError("float16 SFT requires loss scaling, which this runner does not implement; use bfloat16 or float32")


def _prepare_examples(config: SFTConfig) -> tuple[list[SFTExample], list[SFTExample], str, Path]:
    filename = "r1_zero.prompt" if config.response_format == "r1_zero" else "question_only.prompt"
    prompt_path = config.prompt_path or PROJECT_DIR / "alignment/prompts" / filename
    template = prompt_path.read_text(encoding="utf-8")
    if config.response_format == "boxed" and template.rstrip().endswith("<think>"):
        raise ValueError("boxed responses require a question-only prompt, not an open <think> tag")
    examples = load_gsm8k_examples(config.train_path, template, config.response_format)
    if config.eval_path is None:
        train_examples, validation = split_examples(examples, config.validation_fraction, config.seed)
    else:
        train_examples = examples
        validation = load_gsm8k_examples(config.eval_path, template, config.response_format)
        overlap = {e.question for e in train_examples} & {e.question for e in validation}
        if overlap:
            raise ValueError("Training and validation data overlap by question")
    return train_examples, validation, template, prompt_path


def _write_data(config: SFTConfig, train_examples: list[SFTExample], validation: list[SFTExample]) -> None:
    directory = config.output_dir / "data"
    directory.mkdir(parents=True)
    for split, examples in (("train", train_examples), ("validation", validation)):
        # Keep the same held-out questions when handing the run to RL.
        raw = [{"question": e.question, "answer": e.rationale + "\n#### " + e.ground_truth} for e in examples]
        formatted = [{"question": e.question, "prompt": e.prompt, "response": e.response,
                      "ground_truth": e.ground_truth} for e in examples]
        for suffix, records in (("", raw), ("_sft", formatted)):
            with (directory / f"{split}{suffix}.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def _loss_sum(model, batch: dict[str, torch.Tensor], autocast: str = "none") -> tuple[torch.Tensor, int]:
    device = next(model.parameters()).device
    with nvtx_range("sft.batch_to_device", device):
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
    targets = batch["labels"][:, 1:]
    supervised = targets.ne(-100)
    token_count = int(supervised.sum().item())
    if token_count == 0:
        raise ValueError("An SFT batch has no supervised next-token targets")
    with nvtx_range("sft.forward", device), autocast_context(model, autocast):
        logits = model(batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
    # Select response positions before the FP32 CE, keeping prompt/padding out.
    with nvtx_range("sft.loss", device):
        selected_logits = logits[:, :-1, :][supervised]
        loss = F.cross_entropy(selected_logits.float(), targets[supervised], reduction="sum")
    return loss, token_count


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


def evaluate_sft(model, records, pad_token_id, batch_size, autocast: str = "none"):
    """Token-weighted held-out teacher-forcing NLL; restore the caller's mode."""
    if not records or batch_size <= 0:
        raise ValueError("Validation records and a positive batch_size are required")
    previous_mode = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    try:
        with torch.inference_mode():
            for start in range(0, len(records), batch_size):
                batch = collate_sft_batch(records[start:start + batch_size], pad_token_id)
                loss, tokens = _loss_sum(model, batch, autocast)
                total_loss += loss.item()
                total_tokens += tokens
    finally:
        model.train(previous_mode)
    mean_loss = total_loss / total_tokens
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss) if mean_loss < 700 else None,
            "supervised_tokens": total_tokens, "num_examples": len(records)}


def train(config: SFTConfig) -> Path:
    _validate_config(config)
    if config.training_backend == "fsdp2":
        from scripts.fsdp_sft_runtime import train_fsdp_sft
        return train_fsdp_sft(config)
    if config.training_backend != "transformers":
        raise ValueError("Unknown training backend")
    seed_everything(config.seed)
    train_examples, validation, template, prompt_path = _prepare_examples(config)
    validate_best_data(config, train_examples, validation, config.eval_path)
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise ValueError("输出目录非空，请选择新的 output-dir")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_data(config, train_examples, validation)
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


def parse_args() -> SFTConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B-Base")
    parser.add_argument("--train-path", type=Path, default=PROJECT_DIR / "data/gsm8k/train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/sft")
    parser.add_argument("--eval-path", type=Path)
    parser.add_argument("--prompt-path", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--training-backend", choices=["transformers", "fsdp2"], default="transformers")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--optimizer-name", choices=["adam", "adamw", "muon"], default="adamw")
    parser.add_argument("--optimizer-device", choices=["model", "cpu"], default="model",
                        help="cpu 将 Adam/AdamW 的 FP32 主权重与优化器状态放在 CPU")
    parser.add_argument("--response-format", choices=["r1_zero", "boxed"], default="r1_zero")
    parser.add_argument("--max-steps", type=int)
    for name, default in (("seed", 42), ("num-epochs", 1), ("batch-size", 1),
                          ("gradient-accumulation-steps", 8), ("max-seq-length", 1024),
                          ("eval-every", 50), ("save-every", 100), ("eval-examples", 64),
                          ("eval-generation-examples", 0), ("max-new-tokens", 512),
                          ("generation-batch-size", 1)):
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("validation-fraction", 0.1), ("learning-rate", 1e-5),
                          ("max-grad-norm", 1.0), ("weight-decay", 0.0)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--autocast", choices=["none", "bfloat16"], default="none",
                        help="建议配合 --dtype float32：FP32 主权重 + BF16 autocast 计算")
    parser.add_argument("--no-save-optimizer-state", dest="save_optimizer_state", action="store_false",
                        help="checkpoint 不保存优化器状态（磁盘紧张时用）")
    parser.add_argument("--best-checkpoint-dir", type=Path,
                        help="滚动最佳model-only缓存目录（建议/dev/shm）；结束仅保存checkpoint-best")
    parser.add_argument("--prepare-only", action="store_true")
    config = SFTConfig(**vars(parser.parse_args()))
    if config.best_checkpoint_dir is not None:
        config.save_optimizer_state = False
    try:
        if config.optimizer_device == "cpu" and config.optimizer_name == "muon":
            raise ValueError("CPU optimizer offload only supports adam/adamw")
        _validate_config(config)
    except ValueError as error:
        parser.error(str(error))
    return config


if __name__ == "__main__":
    destination = train(parse_args())
    print(f"SFT artifacts: {destination}", flush=True)
