"""torchrun runtime for train_grpo_unified --training-backend fsdp2."""
from dataclasses import asdict
from training.validation import validate_fsdp_config
from distributed.context import local_call
import json
import logging
import math
import os

import torch
from training.model_factory import load_tokenizer, policy_factory

from training.backends.fsdp_grpo_update import FSDPGRPOTrainer, batch_from_rollout
from distributed import (DistributedBestCheckpoint, FSDPConfig, GenerationConfig,
                         NvidiaContext, NvidiaFSDPEngine, RolloutResult)
from utils.serialization import json_value
from profiling.metrics import log_metrics
from alignment.rewards.factory import reward_function
from training.data.gsm8k import load_questions, load_prompt_template, validate_best_data, validate_best_eval_path
from training.data.sampling import sample_questions


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
    if getattr(engine, "external_generation", None) is not None:
        raw_ids = [row[active.bool()].tolist() for row, active in zip(ids, mask)]
        if not ctx.all_true(all(len(row) + config.max_new_tokens <= config.vllm_max_model_len for row in raw_ids)):
            raise ValueError("Prompt plus generation exceeds vLLM context limit")
        rollout = engine.external_generation.generate(raw_ids, temperature=temperature,
            max_new_tokens=config.max_new_tokens,
            seed=config.seed + 104729 * engine.rollout_id + 1009 * engine.policy_version)
        responses = local_call(ctx, lambda: [tokenizer.decode(row[active].tolist(), skip_special_tokens=True)
                                            for row, active in zip(rollout.response_ids, rollout.response_mask)])
        return ids, mask, rollout, responses
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
    context = (NvidiaContext.from_torchrun(timeout_seconds=config.vllm_startup_timeout + 4500)
               if config.generation_backend == "vllm" else NvidiaContext.from_torchrun())
    with context as ctx:
        ctx.agree("GRPO config", json_value(asdict(config)))
        validate_fsdp_config(config, ctx.world_size)
        if config.generation_backend == "vllm":
            if int(os.environ.get("LOCAL_WORLD_SIZE", ctx.world_size)) != ctx.world_size:
                raise ValueError("Colocated vLLM generation currently requires a single-node torchrun job")
            if not 0 <= config.vllm_gpu < ctx.world_size or config.vllm_gpu == 0:
                raise ValueError("Use a nonzero local GPU for vLLM; rank 0 is the NCCL sender")
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
        tokenizer = local_call(ctx, lambda: load_tokenizer(config.model_id))
        factory = policy_factory(config.model_id, tokenizer.pad_token_id, disable_dropout=True)

        compute = torch.bfloat16 if config.dtype == "bfloat16" or config.autocast == "bfloat16" else torch.float32
        engine = NvidiaFSDPEngine(ctx, factory, FSDPConfig(
            compute_dtype=compute, learning_rate=config.learning_rate,
            weight_decay=config.weight_decay, max_grad_norm=config.max_grad_norm,
            gradient_checkpointing=config.gradient_checkpointing, seed=config.seed))
        try:
            if config.generation_backend == "vllm":
                from distributed.generation import ExternalGeneration
                from generation.vllm import VLLMBackend
                engine.external_generation = ExternalGeneration(engine, tokenizer,
                    lambda: VLLMBackend(host=config.vllm_host, port=config.vllm_port, gpu=config.vllm_gpu,
                        policy_device=str(ctx.device), gpu_memory_utilization=config.vllm_memory_utilization,
                        max_model_len=config.vllm_max_model_len, max_num_seqs=config.generation_batch_size,
                        startup_timeout=config.vllm_startup_timeout),
                    config.rollout_checkpoint_dir, source_model=config.model_id)
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
                    destination = config.output_dir / f"checkpoint-{step:06d}"
                    if config.checkpoint_format == "hf":
                        engine.export_model_only_hf(destination, tokenizer=tokenizer)
                        ctx.root_call(lambda: (destination / "config.json.run").write_text(
                            json.dumps({"model_id": config.model_id}) + "\n"))
                        if config.save_optimizer_state:
                            engine.save_checkpoint(config.output_dir / f"resume-{step:06d}",
                                include_optimizer=True, metadata={"step": step})
                    else:
                        # DCP paths require storage shared by all ranks.
                        engine.save_checkpoint(destination, include_optimizer=config.save_optimizer_state,
                                               metadata={"step": step})
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
            try:
                if getattr(engine, "external_generation", None) is not None:
                    engine.external_generation.close()
            finally:
                engine.close()
