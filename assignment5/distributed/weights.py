"""Engine-neutral versioned weight receiver; no imports from alignment/vLLM."""
from typing import Protocol
from pathlib import Path
import json
import os
import shutil
from uuid import uuid4
import torch


def hf_asset_writer(model, tokenizer=None, dtype=torch.bfloat16):
    """Copy HF metadata without mutating the live FSDP model's config."""
    from copy import deepcopy
    config = deepcopy(model.config)
    config.dtype = dtype
    generation = deepcopy(getattr(model, "generation_config", None))

    def save(directory):
        config.save_pretrained(directory)
        if generation is not None:
            generation.save_pretrained(directory)
        if tokenizer is not None:
            tokenizer.save_pretrained(directory)

    return save


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


class WeightSink(Protocol):
    """Called only on rank 0. commit must acknowledge the published version.

    The receiver owns any required checkpoint-to-serving name/layout mapping.
    write receives a detached CPU tensor, one parameter/buffer at a time.
    Failed transactions must remain unavailable to generation until retried.
    """
    def begin(self, version: int) -> None: ...
    def write(self, name: str, tensor: torch.Tensor) -> None: ...
    def commit(self, version: int) -> int: ...
    def abort(self, version: int) -> None: ...


class HFCheckpointSink:
    """Rank-0 canonical HF checkpoint sink with bounded per-tensor CPU staging.

    save_assets(directory) must write config.json (and usually tokenizer files).
    This sink has no dependency on a particular Transformers model/tokenizer.
    It creates many small shards rather than retaining a full model in memory.
    """
    def __init__(self, path, save_assets):
        self.path = Path(path)
        self.save_assets = save_assets
        self.temporary = None
        self.version = None
        self.weight_map = {}
        self.total_bytes = 0

    def begin(self, version):
        if self.temporary is not None:
            raise RuntimeError("A weight transaction is already active")
        if self.path.exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_name(self.path.name + ".partial-" + uuid4().hex)
        self.temporary.mkdir()
        self.version = version
        self.weight_map, self.total_bytes = {}, 0

    def write(self, name, tensor):
        from safetensors.torch import save_file
        if self.temporary is None or name in self.weight_map:
            raise RuntimeError("No active transaction or duplicate weight name")
        if tensor.device.type != "cpu":
            raise ValueError("HF sink accepts CPU tensors")
        filename = f"model-{len(self.weight_map):06d}.safetensors"
        save_file({name: tensor.contiguous()}, str(self.temporary / filename), metadata={"format": "pt"})
        self.weight_map[name] = filename
        self.total_bytes += tensor.numel() * tensor.element_size()

    def commit(self, version):
        if self.temporary is None or version != self.version or not self.weight_map:
            raise RuntimeError("Invalid or empty weight transaction")
        self.save_assets(self.temporary)
        if not (self.temporary / "config.json").is_file():
            raise ValueError("save_assets must write config.json")
        index = {"metadata": {"total_size": self.total_bytes}, "weight_map": self.weight_map}
        (self.temporary / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
        (self.temporary / "policy_version.json").write_text(json.dumps({"policy_version": version}) + "\n")
        os.rename(self.temporary, self.path)
        self.temporary = None
        return version

    def abort(self, version):
        if self.temporary is not None:
            shutil.rmtree(self.temporary)
            self.temporary = None
