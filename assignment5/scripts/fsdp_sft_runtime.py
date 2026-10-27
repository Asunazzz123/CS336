"""Response-only SFT on the shared FSDP2 engine, with HF stage handoff."""
from dataclasses import asdict
import json
import math
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.sft import collate_sft_batch, encode_sft_examples
from distributed import DistributedBestCheckpoint, FSDPConfig, LossOutput, NvidiaContext, NvidiaFSDPEngine
from scripts.fsdp_grpo_runtime import local_call
from scripts.train_grpo_base import json_value, log_metrics, validate_best_data


def loss_output(model, batch):
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                   use_cache=False).logits[:, :-1]
    targets = batch["labels"][:, 1:]
    active = targets.ne(-100)
    # Empty selection still has a differentiable zero sum on padding-only ranks.
    loss = F.cross_entropy(logits[active].float(), targets[active], reduction="sum")
    return LossOutput(loss, int(active.sum().item()))


def padded_microbatches(records, indices, *, rank, world_size, batch_size, accumulation, pad_id, device):
    """Fixed microbatch counts on all ranks; tail fillers have zero loss weight."""
    dummy = {**records[0], "labels": [-100] * len(records[0]["labels"])}
    total = world_size * batch_size * accumulation
    if len(indices) > total:
        raise ValueError("Too many indices for an update")
    selected = [records[i] for i in indices] + [dummy] * (total - len(indices))
    result = []
    for micro in range(accumulation):
        start = (micro * world_size + rank) * batch_size
        batch = collate_sft_batch(selected[start:start + batch_size], pad_id)
        result.append({name: value.to(device) for name, value in batch.items()})
    return result


@torch.no_grad()
def evaluate_sft(engine, records, pad_id, batch_size):
    ctx = engine.context
    totals = torch.zeros(2, dtype=torch.float64, device=ctx.device)
    was_training = engine.model.training
    engine.model.eval()
    try:
        width = batch_size * ctx.world_size
        for start in range(0, len(records), width):
            batch = padded_microbatches(records, list(range(start, min(start + width, len(records)))),
                rank=ctx.rank, world_size=ctx.world_size, batch_size=batch_size, accumulation=1,
                pad_id=pad_id, device=ctx.device)[0]
            output = loss_output(engine.model, batch)
            totals[0] += output.loss_sum.double()
            totals[1] += output.weight
        ctx.sum(totals)
    finally:
        engine.model.train(was_training)
    if totals[1].item() <= 0:
        raise ValueError("No supervised validation tokens")
    loss = float((totals[0] / totals[1]).item())
    return {"loss": loss, "perplexity": math.exp(min(loss, 80)),
            "num_examples": len(records), "supervised_tokens": int(totals[1].item())}


def train_fsdp_sft(config):
    from scripts.train_sft import _prepare_examples, _write_data
    if config.optimizer_name != "adamw" or config.optimizer_device != "model":
        raise ValueError("FSDP SFT requires GPU-sharded AdamW")
    if config.eval_generation_examples:
        raise ValueError("FSDP SFT evaluates NLL; use the GRPO stage for generated-answer evaluation")
    with NvidiaContext.from_torchrun() as ctx:
        ctx.agree("SFT config", json_value(asdict(config)))
        def prepare():
            training, validation, _, prompt_path = _prepare_examples(config)
            validate_best_data(config, training, validation, config.eval_path)
            if config.output_dir.exists() and any(config.output_dir.iterdir()):
                raise ValueError("Output directory must be new or empty")
            config.output_dir.mkdir(parents=True, exist_ok=True)
            _write_data(config, training, validation)
            (config.output_dir / "run_config.json").write_text(json.dumps(json_value(asdict(config)), indent=2))
            return training, validation, str(prompt_path)
        training, validation, prompt_path = ctx.root_call(prepare)
        if config.prepare_only:
            return config.output_dir
        tokenizer = local_call(ctx, lambda: AutoTokenizer.from_pretrained(config.model_id))
        def encode():
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer requires EOS")
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            return (encode_sft_examples(training, tokenizer, config.max_seq_length),
                    encode_sft_examples(validation, tokenizer, config.max_seq_length))
        (records, train_stats), (eval_records, eval_stats) = local_call(ctx, encode)
        if not ctx.all_true(bool(records) and bool(eval_records)):
            raise ValueError("No SFT examples remain after length filtering")
        ctx.agree("SFT encoded counts", (len(records), len(eval_records)))
        ctx.root_call(lambda: (config.output_dir / "data_stats.json").write_text(json.dumps({
            "train_examples": len(training), "validation_examples": len(validation),
            "tokenized": True, "train_tokens": train_stats, "validation_tokens": eval_stats,
            "seed": config.seed, "prompt_path": prompt_path}, indent=2)))
        def factory():
            model = AutoModelForCausalLM.from_pretrained(config.model_id, dtype=torch.float32)
            model.config.pad_token_id = tokenizer.pad_token_id
            limit = getattr(model.config, "max_position_embeddings", None)
            if limit and max(len(r["input_ids"]) for r in records + eval_records) > limit:
                raise ValueError("SFT sequences exceed model context length")
            return model
        compute = torch.bfloat16 if config.dtype == "bfloat16" or config.autocast == "bfloat16" else torch.float32
        engine = NvidiaFSDPEngine(ctx, factory, FSDPConfig(compute_dtype=compute,
            learning_rate=config.learning_rate, weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm, gradient_checkpointing=config.gradient_checkpointing,
            seed=config.seed))
        try:
            best = (DistributedBestCheckpoint(engine, config.best_checkpoint_dir, "loss", "min", tokenizer=tokenizer)
                    if config.best_checkpoint_dir is not None else None)
            def validate(step):
                metrics = evaluate_sft(engine, eval_records[:config.eval_examples], tokenizer.pad_token_id,
                                       config.batch_size)
                ctx.root_call(lambda: log_metrics(config.output_dir, step, "eval", metrics))
                if best is not None:
                    best.consider(step, metrics["loss"], metadata={"selection_split": "validation",
                        "metric_model": "live_policy_before_export", "checkpoint_dtype": "bfloat16",
                        "optimizer_saved": False, "eval_examples": metrics["num_examples"],
                        "policy_version": engine.policy_version})
            validate(0)
            step = 0
            update_size = config.batch_size * config.gradient_accumulation_steps * ctx.world_size
            for epoch in range(config.num_epochs):
                indices = list(range(len(records)))
                random.Random(config.seed + epoch).shuffle(indices)
                for start in range(0, len(indices), update_size):
                    batches = padded_microbatches(records, indices[start:start + update_size], rank=ctx.rank,
                        world_size=ctx.world_size, batch_size=config.batch_size,
                        accumulation=config.gradient_accumulation_steps, pad_id=tokenizer.pad_token_id,
                        device=ctx.device)
                    result = engine.train_step(batches, loss_output)
                    step += 1
                    ctx.root_call(lambda: log_metrics(config.output_dir, step, "train", {
                        **asdict(result), "epoch": epoch + 1}))
                    final = ((config.max_steps is not None and step >= config.max_steps) or
                             epoch == config.num_epochs - 1 and start + update_size >= len(indices))
                    if step % config.eval_every == 0 or final:
                        validate(step)
                    if best is None and (step % config.save_every == 0 or final):
                        destination = config.output_dir / f"checkpoint-{step:06d}"
                        engine.export_model_only_hf(destination, tokenizer=tokenizer)
                        # Provenance supports Qwen text -> vLLM wrapper lookup.
                        ctx.root_call(lambda: (destination / "config.json.run").write_text(
                            json.dumps({"model_id": config.model_id}) + "\n"))
                        if config.save_optimizer_state:
                            engine.save_checkpoint(config.output_dir / f"resume-{step:06d}",
                                include_optimizer=True, metadata={"step": step, "epoch": epoch})
                    if final:
                        if best is None:
                            return destination
                        destination = best.export(config.output_dir / "checkpoint-best")
                        selected = best.selection()
                        ctx.root_call(lambda: (destination / "config.json.run").write_text(
                            json.dumps({"model_id": config.model_id}) + "\n"))
                        ctx.root_call(lambda: (config.output_dir / "best_checkpoint.json").write_text(
                            json.dumps({**selected, "path": str(destination)}, indent=2)))
                        best.cleanup()
                        return destination
            raise RuntimeError("SFT finished without updates")
        finally:
            engine.close()
