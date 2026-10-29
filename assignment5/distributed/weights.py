"""Collective FSDP export; generic receiver/writers live in checkpointing."""
from pathlib import Path
import torch
from checkpointing.interface import WeightSink
from checkpointing.hf import HFCheckpointSink, hf_asset_writer


def export_model_only_hf(engine, path, *, tokenizer=None, dtype=torch.bfloat16,
                         save_assets=None):
    """Collective live-shards -> HF export, without an intermediate DCP/model copy.

    ALL ranks call this; only rank 0 owns the destination and writes assets.
    Existing DCP checkpoints can first be loaded with engine.load_checkpoint.
    """
    path = Path(path).expanduser().absolute()
    engine.context.agree("HF export path", str(path))
    sink = None
    def prepare():
        nonlocal sink
        writer = save_assets if save_assets is not None else hf_asset_writer(engine.model, tokenizer, dtype)
        sink = HFCheckpointSink(path, writer)
    engine.context.root_call(prepare)
    engine.export_weights(sink, dtype=dtype)
    return path
