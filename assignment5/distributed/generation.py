"""FSDP adapter for a versioned, independent generation backend."""
import json
from pathlib import Path
import shutil
from uuid import uuid4

import torch

from generation import PolicySnapshot
from .engine import RolloutResult
from .state import Phase


class ExternalGeneration:
    """Collective single-node time sharing. ALL ranks call generate/close.

    Publish HF snapshot -> park training -> load/serve on rank 0 -> close the
    owned receiver -> resume training. No live FSDP and vLLM model overlap.
    Backend construction and filesystem writes are rank-0-only.
    """
    def __init__(self, engine, tokenizer, backend_factory, staging_root, *, source_model):
        self.engine, self.ctx, self.tokenizer = engine, engine.context, tokenizer
        self.factory, self.root = backend_factory, Path(staging_root)
        self.source_model = str(source_model)
        self.backend = None

    def generate(self, prompt_ids, *, temperature, max_new_tokens, seed):
        engine, ctx = self.engine, self.ctx
        engine._require(Phase.READY)
        if not ctx.all_true(bool(prompt_ids) and all(bool(row) and all(
                isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in row)
                for row in prompt_ids)):
            raise ValueError("Every rank requires nonempty prompt token IDs")
        ctx.agree("external generation", (engine.policy_version, engine.rollout_id,
                                         temperature, max_new_tokens, seed))
        requests = ctx.gather_objects(prompt_ids)
        counts = [len(batch) for batch in requests]
        snapshot = Path(ctx.root_call(lambda: str(self.root / ("policy-" + uuid4().hex))))
        version, rollout_id = engine.policy_version, engine.rollout_id
        try:
            engine.export_model_only_hf(snapshot, tokenizer=self.tokenizer)
            ctx.root_call(lambda: (snapshot / "config.json.run").write_text(
                json.dumps({"model_id": self.source_model}) + "\n"))
            engine.park()

            def serve():
                self.backend = self.factory()
                try:
                    acknowledged = self.backend.load_policy(PolicySnapshot(snapshot, version))
                    if acknowledged != version:
                        raise RuntimeError("Generation receiver acknowledged a stale policy version")
                    flat = [ids for batch in requests for ids in batch]
                    return self.backend.generate(flat, {"temperature": temperature,
                        "max_tokens": max_new_tokens, "n": 1, "seed": seed})
                finally:
                    self.backend.close()
                    self.backend = None
            completions = ctx.root_call(serve)
            start = sum(counts[:ctx.rank])
            local = completions[start:start + counts[ctx.rank]]
            valid = (len(local) == len(prompt_ids) and all(
                0 < len(c.token_ids) <= max_new_tokens and all(isinstance(t, int) and t >= 0 for t in c.token_ids)
                for c in local))
            if not ctx.all_true(valid):
                raise ValueError("External backend returned invalid token responses")
            if ctx.device.type == "cuda":
                torch.cuda.synchronize(ctx.device)
                torch.cuda.empty_cache()  # Release sender-side staging before FSDP returns.
            engine.resume()
            engine.rollout_id += 1
            width = max(len(c.token_ids) for c in local)
            ids = torch.full((len(local), width), self.tokenizer.pad_token_id,
                             dtype=torch.long, device=ctx.device)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            for row, completion in enumerate(local):
                length = len(completion.token_ids)
                ids[row, :length] = torch.tensor(completion.token_ids, device=ctx.device)
                mask[row, :length] = True
            return RolloutResult(ids, mask, version, rollout_id)
        except BaseException:
            # Never resume CUDA training if receiver shutdown or transfer failed;
            # its GPU allocations/policy may be uncertain. Caller closes engine.
            engine.phase = Phase.FAILED
            raise
        finally:
            if ctx.rank == 0 and snapshot.exists():
                shutil.rmtree(snapshot)

    def close(self):
        def stop():
            if self.backend is not None:
                self.backend.close()
                self.backend = None
        self.ctx.root_call(stop)
