"""Collective adapter for the rank-0, single-writer HF best-checkpoint store."""
import json
import math
import os
from pathlib import Path
import shutil
from uuid import uuid4

import torch

from checkpointing.best import BestCheckpointStore
from .state import Phase
from .weights import export_model_only_hf


class DistributedBestCheckpoint:
    """ALL ranks call every public method, including constructor and export.

    Only rank 0 creates/accesses the cache (also works with node-local tmpfs).
    A winning candidate is streamed collectively before the store's rank-0
    callback publishes it. Never invoke engine collectives inside consider's
    rank-0-only callback. Failed exports preserve the previous complete best.
    """
    def __init__(self, engine, root, metric_name="accuracy", mode="max", *,
                 tokenizer=None, dtype=torch.bfloat16, save_assets=None):
        self.engine, self.context = engine, engine.context
        self.tokenizer, self.dtype, self.save_assets = tokenizer, dtype, save_assets
        self.store = None
        self.root = Path(root).expanduser().absolute()
        self.context.agree("best store", (str(self.root), metric_name, mode, str(dtype)))
        def create():
            self.store = BestCheckpointStore(self.root, metric_name, mode)
        self.context.root_call(create)

    def consider(self, step, score, metadata=None):
        self.engine._require(Phase.READY)
        self.context.agree("best candidate", (step, score))

        def prepare():
            if not isinstance(step, int) or isinstance(step, bool) or step < 0:
                raise ValueError("step must be a nonnegative integer")
            if isinstance(score, bool) or not math.isfinite(score):
                raise ValueError("score must be finite")
            json.dumps(metadata, allow_nan=False)
            self.store._assert_owned()
            previous = self.store.selection
            if previous is not None and not (
                score < previous["score"] if self.store.mode == "min" else score > previous["score"]
            ):
                return None
            return str(self.root / (".export-" + uuid4().hex))

        staged = self.context.root_call(prepare)
        if staged is None:
            return False
        staged = Path(staged)
        try:
            export_model_only_hf(self.engine, staged, tokenizer=self.tokenizer,
                                 dtype=self.dtype, save_assets=self.save_assets)
            return self.context.root_call(lambda: self.store.consider(
                step, score, lambda destination: os.rename(staged, destination), metadata))
        finally:
            # No collective here: preserve the original exception if a peer fails.
            if self.context.rank == 0 and staged.exists():
                shutil.rmtree(staged)

    def export(self, destination):
        self.context.agree("best destination", str(destination))
        return Path(self.context.root_call(lambda: str(self.store.export(destination))))

    def selection(self):
        return self.context.root_call(lambda: self.store.selection)

    def cleanup(self):
        self.context.root_call(lambda: self.store.cleanup())
