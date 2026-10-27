"""Real two-GPU SFT -> HF -> external-vLLM -> GRPO acceptance (no downloads)."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

from alignment.fsdp_grpo import FSDPGRPOTrainer, batch_from_rollout
from alignment.sft import collate_sft_batch
from distributed import DistributedBestCheckpoint, ExternalGeneration, FSDPConfig, NvidiaContext, NvidiaFSDPEngine
from generation.vllm import VLLMBackend
from scripts.fsdp_sft_runtime import loss_output, padded_microbatches, evaluate_sft
from .test_fsdp_grpo_integration import tiny_model
from .test_fsdp_sft_integration import records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vllm-port", type=int, default=8029)
    args = parser.parse_args()
    with NvidiaContext.from_torchrun(timeout_seconds=1800) as ctx:
        ctx.root_call(lambda: args.output_dir.mkdir(parents=True, exist_ok=False))
        vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{f"t{i}": i for i in range(3, 16)}}
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
            pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
        config = FSDPConfig(block_class_names=("GPT2Block",), compute_dtype=torch.float32,
                            learning_rate=0.003, max_grad_norm=0.7)
        engine = NvidiaFSDPEngine(ctx, tiny_model, config)
        external = None
        try:
            # Unequal supervision and a padding-only final microbatch on rank 1.
            data = records()
            reference = tiny_model().to(ctx.device).eval()
            reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=config.learning_rate,
                foreach=False, fused=False, weight_decay=config.weight_decay)
            reference_loss = loss_output(reference, {key: value.to(ctx.device)
                for key, value in collate_sft_batch(data, 0).items()})
            (reference_loss.loss_sum / reference_loss.weight).backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), config.max_grad_norm)
            reference_optimizer.step()
            batches = padded_microbatches(data, [0, 1, 2], rank=ctx.rank, world_size=ctx.world_size,
                batch_size=1, accumulation=2, pad_id=0, device=ctx.device)
            step = engine.train_step(batches, loss_output)
            assert step.global_weight == reference_loss.weight
            full = get_model_state_dict(engine.model, options=StateDictOptions(full_state_dict=True))
            for name, tensor in reference.state_dict().items():
                torch.testing.assert_close(full[name], tensor, atol=1e-5, rtol=1e-5)
            del full, reference, reference_optimizer
            metrics = evaluate_sft(engine, data[:1], 0, 1)
            assert metrics["supervised_tokens"] == 2
            best = DistributedBestCheckpoint(engine, args.output_dir / "sft-cache", "loss", "min", tokenizer=tokenizer)
            assert best.consider(1, metrics["loss"])
            checkpoint = best.export(args.output_dir / "sft-best")
            best.cleanup()
            # Recreate the engine from the exported SFT artifact, like the pipeline.
            engine.close()
            engine = NvidiaFSDPEngine(ctx, lambda: AutoModelForCausalLM.from_pretrained(
                checkpoint, dtype=torch.float32, local_files_only=True), config)
            external = ExternalGeneration(engine, tokenizer, lambda: VLLMBackend(
                port=args.vllm_port, gpu=1, policy_device=str(ctx.device), max_model_len=32,
                max_num_seqs=2, gpu_memory_utilization=0.2), args.output_dir / "rollouts", source_model=checkpoint)
            prompts = torch.tensor([[3 + ctx.rank, 4]] * 2, device=ctx.device)
            prompt_mask = torch.ones_like(prompts)
            for version in (0, 1):
                rollout = external.generate(prompts.tolist(), temperature=1., max_new_tokens=3, seed=42 + version)
                assert rollout.policy_version == version
                assert engine.optimizer_on_gpu
                batch = batch_from_rollout(prompts, prompt_mask, rollout, [1., 0.], group_size=2)
                result = FSDPGRPOTrainer(engine, gradient_accumulation_steps=2).update(batch, policy_version=version)
                assert result[-1].policy_version == version + 1
            ctx.root_call(lambda: (args.output_dir / "passed.json").write_text(json.dumps({
                "sft": "passed", "vllm_rounds": 2, "grpo_policy_version": engine.policy_version})))
        finally:
            if external is not None:
                external.close()
            engine.close()


if __name__ == "__main__":
    main()
