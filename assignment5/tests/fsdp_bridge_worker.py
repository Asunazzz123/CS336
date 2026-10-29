"""Two-process CPU DTensor export acceptance; optional real CUDA FSDP/GRPO."""
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from transformers import AutoModelForCausalLM

from distributed import (DistributedBestCheckpoint, FSDPConfig, GenerationConfig,
                         NvidiaContext, NvidiaFSDPEngine)
from training.backends.fsdp_grpo_update import FSDPGRPOTrainer, batch_from_rollout
from .test_fsdp_grpo_integration import tiny_model, batch
from .test_distributed_runtime import local_engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--compute-dtype", default="float32")
    args = parser.parse_args()
    if args.cuda:
        ctx = NvidiaContext.from_torchrun()
        engine = NvidiaFSDPEngine(ctx, tiny_model, FSDPConfig(
            block_class_names=("GPT2Block",), compute_dtype=getattr(torch, args.compute_dtype)))
        prompts = torch.tensor([[2 + ctx.rank, 3]] * 4, device=ctx.device)
        mask = torch.ones_like(prompts)
        rollout = engine.rollout(prompts, mask, GenerationConfig(max_new_tokens=3,
            temperature=1.0, eos_token_id=1, pad_token_id=0))
        data = batch_from_rollout(prompts, mask, rollout, [1., 0., 0., 1.], group_size=2)
        results = FSDPGRPOTrainer(engine, num_epochs_per_rollout=3,
                                  gradient_accumulation_steps=2).update(data, policy_version=0)
        assert results[-1].policy_version == 3
    else:
        dist.init_process_group("gloo")
        mesh = init_device_mesh("cpu", (dist.get_world_size(),))
        ctx = NvidiaContext(dist.get_rank(), dist.get_world_size(), dist.get_rank(),
                            torch.device("cpu"), mesh, True)
        model = tiny_model()
        converted = {}
        for module in model.modules():
            for name, parameter in list(module.named_parameters(recurse=False, remove_duplicate=False)):
                if id(parameter) not in converted:
                    converted[id(parameter)] = torch.nn.Parameter(distribute_tensor(parameter.detach(), mesh, [Shard(0)]))
                module._parameters[name] = converted[id(parameter)]
        engine = local_engine(model)
        engine.context = ctx
    try:
        manager = DistributedBestCheckpoint(engine, args.output_dir / "cache")
        assert (manager.store is not None) == (ctx.rank == 0)
        assert manager.consider(0, 0.5)
        assert not manager.consider(1, 0.5)
        assert manager.consider(2, 0.8)
        previous = manager.selection()
        # Error raised on rank 0 must reach every peer without hanging a gather.
        manager.save_assets = lambda path: (_ for _ in ()).throw(OSError("disk full"))
        try:
            manager.consider(3, 0.9)
        except RuntimeError as error:
            assert "disk full" in str(error)
        else:
            raise AssertionError("Expected broadcast export failure")
        assert manager.selection() == previous
        path = manager.export(args.output_dir / "hf")
        def verify():
            restored = AutoModelForCausalLM.from_pretrained(path, dtype="auto", local_files_only=True)
            assert next(restored.parameters()).dtype == torch.bfloat16
            assert json.loads((path / "selection.json").read_text())["step"] == 2
            if not args.cuda:
                for name, tensor in tiny_model().state_dict().items():
                    assert torch.equal(restored.state_dict()[name], tensor.bfloat16())
            (args.output_dir / "passed.json").write_text(json.dumps({"ranks": ctx.world_size}))
        ctx.root_call(verify)
        manager.cleanup()
    finally:
        engine.close()
        ctx.close()


if __name__ == "__main__":
    main()
