"""torchrun runtime for train_grpo_unified --training-backend fsdp2."""
from dataclasses import asdict
import json
import logging
import math

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from alignment.fsdp_grpo import FSDPGRPOTrainer, batch_from_rollout
from distributed import (DistributedBestCheckpoint, FSDPConfig, GenerationConfig,
                         NvidiaContext, NvidiaFSDPEngine, RolloutResult)
from scripts.train_grpo_base import (json_value, load_questions, load_prompt_template,
                                    log_metrics, reward_function, sample_questions,
                                    validate_best_data, validate_best_eval_path)


def validate_fsdp_config(config, world_size=None):
    for name in ("num_steps", "prompts_per_batch", "group_size", "gradient_accumulation_steps",
                 "max_new_tokens", "generation_batch_size", "eval_examples", "eval_every", "save_every",
                 "num_epochs_per_rollout"):
        if hasattr(config, name) and getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.group_size < 2:
        raise ValueError("group_size must be >=2")
    for name in ("temperature", "cliprange"):
        if hasattr(config, name) and (not math.isfinite(getattr(config, name)) or getattr(config, name) <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if config.generation_backend != "transformers":
        raise ValueError("fsdp2 currently uses collective engine rollout; vLLM is unsupported")
    if config.optimizer_name != "adamw" or config.optimizer_device != "model":
        raise ValueError("fsdp2 requires GPU-sharded AdamW (--optimizer-name adamw --optimizer-device model)")
    if config.dtype not in ("float32", "bfloat16"):
        raise ValueError("fsdp2 supports float32/bfloat16 compute and FP32 master shards")
    if world_size is not None:
        if config.prompts_per_batch % world_size:
            raise ValueError("prompts_per_batch must be divisible by world_size to keep groups intact")
        local_size = config.prompts_per_batch // world_size * config.group_size
        if not local_size or local_size % config.gradient_accumulation_steps:
            raise ValueError("Per-rank rollout batch must be divisible by gradient_accumulation_steps")


def local_call(ctx, fn):
    """Surface tokenization/reward errors on all peers before the next forward."""
    result, error = None, None
    try:
        result = fn()
    except Exception as exc:
        error = exc
    if not ctx.all_true(error is None):
        raise RuntimeError(f"Rank-local GRPO preparation failed; local error: {error}") from error
    return result


def generate(engine, tokenizer, prompts, config, *, temperature):
    ctx = engine.context
    def tokenize():
        previous = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            return tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt")
        finally:
            tokenizer.padding_side = previous
    inputs = local_call(ctx, tokenize)
    ids, mask = (inputs[key].to(ctx.device) for key in ("input_ids", "attention_mask"))
    ctx.agree("generation batches", math.ceil(len(prompts) / config.generation_batch_size))
    results = []
    for start in range(0, len(prompts), config.generation_batch_size):
        results.append(engine.rollout(ids[start:start + config.generation_batch_size],
                                      mask[start:start + config.generation_batch_size],
                                      GenerationConfig(max_new_tokens=config.max_new_tokens,
                                                       temperature=temperature,
                                                       eos_token_id=tokenizer.eos_token_id,
                                                       pad_token_id=tokenizer.pad_token_id)))
    width = max(result.response_ids.shape[1] for result in results)
    def padded(tensor, value):
        return torch.nn.functional.pad(tensor, (0, width - tensor.shape[1]), value=value)
    rollout = RolloutResult(
        torch.cat([padded(r.response_ids, tokenizer.pad_token_id) for r in results]),
        torch.cat([padded(r.response_mask, False) for r in results]),
        results[0].policy_version, results[0].rollout_id)
    responses = local_call(ctx, lambda: [tokenizer.decode(row[active].tolist(), skip_special_tokens=True)
                                        for row, active in zip(rollout.response_ids, rollout.response_mask)])
    return ids, mask, rollout, responses


def evaluate(engine, tokenizer, questions, template, config, reward_fn):
    ctx = engine.context
    # Equal local counts keep FSDP forwards matched, including short final eval
    # batches. Duplicate padding examples are excluded from ALL metric sums.
    indices = list(range(ctx.rank, len(questions), ctx.world_size))
    count = math.ceil(len(questions) / ctx.world_size)
    valid_count = len(indices)
    indices += [0] * (count - valid_count)
    local = [questions[i] for i in indices]
    _, _, rollout, responses = generate(engine, tokenizer,
        [template.format(question=q.question) for q in local], config, temperature=0.0)
    scores = local_call(ctx, lambda: [reward_fn(r, q.ground_truth) for r, q in zip(responses, local)])
    totals = torch.tensor([
        sum(s["reward"] for s in scores[:valid_count]),
        sum(s["answer_reward"] for s in scores[:valid_count]),
        sum(s["format_reward"] for s in scores[:valid_count]),
        rollout.response_mask[:valid_count].sum().item(), valid_count,
    ], dtype=torch.float64, device=ctx.device)
    ctx.sum(totals)
    return {**{name: float(totals[i].item() / totals[4].item()) for i, name in enumerate(
        ("mean_reward", "accuracy", "format_rate", "mean_response_tokens"))},
        "eval_examples": int(totals[4].item())}


def train_fsdp(config):
    validate_fsdp_config(config)
    validate_best_eval_path(config.best_checkpoint_dir, config.eval_path)
    with NvidiaContext.from_torchrun() as ctx:
        ctx.agree("GRPO config", json_value(asdict(config)))
        validate_fsdp_config(config, ctx.world_size)
        def prepare():
            train_questions = load_questions(config.train_path)
            validation = load_questions(config.eval_path)
            validate_best_data(config, train_questions, validation, config.eval_path)
            if len(train_questions) < config.prompts_per_batch or not validation:
                raise ValueError("Insufficient training questions or empty validation data")
            template = load_prompt_template(config.prompt_path)
            reward_function(config)
            if config.output_dir.exists() and any(config.output_dir.iterdir()):
                raise ValueError("Output directory must be new or empty")
            config.output_dir.mkdir(parents=True, exist_ok=True)
            (config.output_dir / "run_config.json").write_text(
                json.dumps(json_value(asdict(config)), indent=2) + "\n")
            return train_questions, validation[:config.eval_examples], template
        train_questions, validation, template = ctx.root_call(prepare)
        reward_fn = local_call(ctx, lambda: reward_function(config))
        tokenizer = local_call(ctx, lambda: AutoTokenizer.from_pretrained(config.model_id))
        def setup_tokenizer():
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define EOS")
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
        local_call(ctx, setup_tokenizer)

        def factory():
            hf_config = AutoConfig.from_pretrained(config.model_id)
            # Functional attention dropout is usually copied from config at
            # construction, so disable it before loading, including text configs.
            for cfg in (hf_config, getattr(hf_config, "text_config", hf_config)):
                for name in ("attention_dropout", "attention_probs_dropout_prob",
                             "hidden_dropout_prob", "resid_pdrop", "embd_pdrop", "attn_pdrop"):
                    if hasattr(cfg, name):
                        setattr(cfg, name, 0.0)
            hf_config.pad_token_id = tokenizer.pad_token_id
            return AutoModelForCausalLM.from_pretrained(config.model_id, config=hf_config, dtype=torch.float32)

        compute = torch.bfloat16 if config.dtype == "bfloat16" or config.autocast == "bfloat16" else torch.float32
        engine = NvidiaFSDPEngine(ctx, factory, FSDPConfig(
            compute_dtype=compute, learning_rate=config.learning_rate,
            weight_decay=config.weight_decay, max_grad_norm=config.max_grad_norm,
            gradient_checkpointing=config.gradient_checkpointing, seed=config.seed))
        try:
            trainer = FSDPGRPOTrainer(engine, num_epochs_per_rollout=config.num_epochs_per_rollout,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                importance_reweighting_method=config.importance_reweighting_method,
                cliprange=config.cliprange, loss_normalization=config.loss_normalization)
            best = (DistributedBestCheckpoint(engine, config.best_checkpoint_dir, tokenizer=tokenizer)
                    if config.best_checkpoint_dir is not None else None)

            def validation_step(step):
                metrics = evaluate(engine, tokenizer, validation, template, config, reward_fn)
                ctx.root_call(lambda: log_metrics(config.output_dir, step, "eval", metrics))
                if best is not None:
                    best.consider(step, metrics["accuracy"], metadata={
                        "policy_version": engine.policy_version, "checkpoint_dtype": "bfloat16",
                        "live_parameter_dtype": "torch.float32", "optimizer_saved": False,
                        "metric_model": "live_policy_before_export", "selection_split": "validation",
                        "eval_path": str(config.eval_path), "eval_examples": metrics["eval_examples"],
                    })
            validation_step(0)
            for step in range(1, config.num_steps + 1):
                selected = sample_questions(train_questions, config.prompts_per_batch, step,
                                             config.seed, config.sampling_strategy)
                per_rank = len(selected) // ctx.world_size
                local = selected[ctx.rank * per_rank:(ctx.rank + 1) * per_rank]
                prompts = [template.format(question=q.question) for q in local for _ in range(config.group_size)]
                truths = [q.ground_truth for q in local for _ in range(config.group_size)]
                ids, mask, rollout, responses = generate(engine, tokenizer, prompts, config,
                                                         temperature=config.temperature)
                def prepare_batch():
                    rewards = [reward_fn(r, g)["reward"] for r, g in zip(responses, truths)]
                    return batch_from_rollout(ids, mask, rollout, rewards, group_size=config.group_size,
                        baseline=config.baseline, advantage_normalizer=config.advantage_normalizer)
                batch = local_call(ctx, prepare_batch)
                updates = trainer.update(batch, policy_version=rollout.policy_version)
                ctx.root_call(lambda: log_metrics(config.output_dir, step, "train", {
                    **asdict(updates[-1]), "epoch_losses": [r.loss for r in updates]}))
                if step % config.eval_every == 0 or step == config.num_steps:
                    validation_step(step)
                if best is None and (step % config.save_every == 0 or step == config.num_steps):
                    # DCP remains the collective training-state format. Multi-node
                    # jobs need shared storage here, unlike the rank-0 best cache.
                    engine.save_checkpoint(config.output_dir / f"checkpoint-{step:06d}",
                        include_optimizer=config.save_optimizer_state, metadata={"step": step})
            if best is not None:
                destination = best.export(config.output_dir / "checkpoint-best")
                selection = best.selection()
                ctx.root_call(lambda: (config.output_dir / "best_checkpoint.json").write_text(
                    json.dumps({**selection, "path": str(destination)}, indent=2) + "\n"))
                try:
                    best.cleanup()
                except (RuntimeError, OSError):
                    logging.getLogger(__name__).warning("Best exported; RAM cleanup failed", exc_info=True)
        finally:
            engine.close()
