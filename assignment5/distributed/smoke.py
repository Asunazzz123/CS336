"""Real two-GPU acceptance, no downloads and no existing trainer imports.

torchrun --standalone --nproc-per-node=2 -m distributed.smoke --output-dir /tmp/fsdp-smoke
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch.distributed.tensor import DTensor
import torch.nn.functional as F

from . import FSDPConfig, GenerationConfig, HFCheckpointSink, LossOutput, NvidiaContext, NvidiaFSDPEngine


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(16, 32)
        self.down = torch.nn.Linear(32, 16)

    def forward(self, x):
        return x + self.down(torch.tanh(self.up(x)))


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 16)
        self.blocks = torch.nn.ModuleList([TinyBlock(), TinyBlock()])
        self.head = torch.nn.Linear(16, 16)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x)
        if not self.training:
            # Different ranks finish at different times, without changing the
            # block call graph. Exposes local-EOS collective deadlocks.
            eos_now = input_ids.shape[1] >= input_ids[:, 0]
            forced = torch.zeros_like(logits[:, -1, :])
            forced[:, 0] = 100
            forced[:, 3] = 50
            logits = logits.clone()
            logits[:, -1, :] = torch.where(eos_now[:, None], forced,
                                           torch.roll(forced, shifts=3, dims=-1))
        return SimpleNamespace(logits=logits)


def loss_fn(model, batch):
    device = next(model.parameters()).device
    ids, labels = (tensor.to(device) for tensor in batch)
    logits = model(input_ids=ids, use_cache=False).logits
    loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, 16), labels[:, 1:].reshape(-1),
                           ignore_index=-100, reduction="sum")
    return LossOutput(loss, int(labels[:, 1:].ne(-100).sum().item()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compute-dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    config = FSDPConfig(block_class_names=("TinyBlock",), compute_dtype=getattr(torch, args.compute_dtype),
                        learning_rate=0.003, max_grad_norm=0.3)
    with NvidiaContext.from_torchrun() as ctx:
        ctx.root_call(lambda: args.output_dir.mkdir(parents=True, exist_ok=False))
        engine = NvidiaFSDPEngine(ctx, TinyLM, config)
        try:
            torch.manual_seed(config.seed)
            reference = TinyLM().to(ctx.device)
            reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=config.learning_rate,
                                                    weight_decay=config.weight_decay,
                                                    foreach=False, fused=False)
            assert all(isinstance(p, DTensor) and p.dtype == torch.float32
                       for p in engine.model.parameters())
            assert any(p.to_local().numel() < p.numel() for p in engine.model.parameters())
            batches = []
            for length in (4 + ctx.rank, 6 + ctx.rank):
                ids = torch.arange(1, length + 1).reshape(1, -1) % 15
                labels = ids.clone()
                labels[:, :ctx.rank + 1] = -100
                batches.append((ids, labels))
            gathered = [None] * ctx.world_size
            dist.all_gather_object(gathered, batches)
            all_batches = [batch for rank_batches in gathered for batch in rank_batches]
            total_weight = sum(int(labels[:, 1:].ne(-100).sum()) for _, labels in all_batches)

            def reference_update():
                reference.train()
                reference_optimizer.zero_grad(set_to_none=True)
                for batch in all_batches:
                    amp = torch.autocast("cuda", dtype=torch.bfloat16) if args.compute_dtype == "bfloat16" else nullcontext()
                    with amp:
                        output = loss_fn(reference, batch)
                    (output.loss_sum / total_weight).backward()
                torch.nn.utils.clip_grad_norm_(reference.parameters(), config.max_grad_norm)
                reference_optimizer.step()

            def assert_matches_reference():
                full = get_model_state_dict(engine.model, options=StateDictOptions(full_state_dict=True))
                tolerance = 2e-3 if args.compute_dtype == "bfloat16" else 3e-6
                for name, value in reference.state_dict().items():
                    torch.testing.assert_close(full[name], value, rtol=tolerance, atol=tolerance)

            steps = []
            for _ in range(2):
                reference_update()
                result = engine.train_step(batches, loss_fn)
                assert result.global_weight == total_weight
                assert_matches_reference()
                steps.append(asdict(result))
            for parameter, state in engine.optimizer.state.items():
                for key in ("exp_avg", "exp_avg_sq"):
                    assert isinstance(state[key], DTensor)
                    assert state[key].to_local().is_cuda and state[key].dtype == torch.float32

            rollout = engine.rollout(torch.tensor([[1 if ctx.rank == 0 else 5, 2]], device=ctx.device),
                                     torch.ones((1, 2), dtype=torch.long, device=ctx.device),
                                     GenerationConfig(max_new_tokens=5, eos_token_id=0, pad_token_id=0))
            assert rollout.response_mask.sum().item() == (1 if ctx.rank == 0 else 4)
            assert rollout.policy_version == 2
            engine.park()
            assert all(p.device.type == "cpu" for p in engine.model.parameters())
            engine.resume()
            assert all(p.device == ctx.device for p in engine.model.parameters())
            reference_update()
            engine.train_step(batches, loss_fn)
            assert_matches_reference()  # Includes moment/parameter residency roundtrip.

            engine.save_checkpoint(args.output_dir / "resume", include_optimizer=True,
                                   metadata={"cursor": 3})
            engine.train_step(batches, loss_fn)
            assert engine.load_checkpoint(args.output_dir / "resume", restore_optimizer=True) == {"cursor": 3}
            assert engine.policy_version == 3
            assert_matches_reference()
            reference_update()
            engine.train_step(batches, loss_fn)
            assert_matches_reference()  # Restored optimizer continues the same update.

            def assets(directory):
                (directory / "config.json").write_text(json.dumps({"model_type": "tiny-smoke"}))

            sink = HFCheckpointSink(args.output_dir / "weights", assets) if ctx.rank == 0 else None
            engine.export_weights(sink, dtype=torch.float32)
            local = {"rank": ctx.rank, "device": str(ctx.device), "steps": steps,
                     "response_mask": rollout.response_mask.cpu().tolist(),
                     "policy_version": engine.policy_version,
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(ctx.device)}
            reports = [None] * ctx.world_size
            dist.all_gather_object(reports, local)
            ctx.root_call(lambda: (args.output_dir / "report.json").write_text(
                json.dumps({"status": "passed", "torch": torch.__version__, "ranks": reports}, indent=2)))
            if ctx.rank == 0:
                print(f"NVIDIA FSDP smoke PASSED: {args.output_dir}", flush=True)
        finally:
            engine.close()


if __name__ == "__main__":
    main()
