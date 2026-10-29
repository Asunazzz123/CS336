"""HF model-only writers preserving live parameters/config and tied views."""
import json
import os
import shutil
from pathlib import Path
from uuid import uuid4
import torch


def bf16_cpu_state_dict(model):
    """Export floating weights on CPU without breaking identical tied views."""
    exported = {}
    converted_views = {}
    for name, value in model.state_dict().items():
        # state_dict returns distinct Tensor objects for tied names. Cache their
        # complete view identity, rather than the Python object or storage alone.
        view = (value.device, value.dtype, value.untyped_storage().data_ptr(),
                value.storage_offset(), tuple(value.shape), tuple(value.stride()))
        if view not in converted_views:
            dtype = torch.bfloat16 if value.is_floating_point() else value.dtype
            converted_views[view] = value.detach().to(device="cpu", dtype=dtype, copy=True)
        exported[name] = converted_views[view]
    return exported


def save_bf16_model(model, destination):
    """Write BF16 model files while leaving live parameters/config untouched."""
    from copy import deepcopy

    state = bf16_cpu_state_dict(model)
    live_config = model.config
    live_generation_config = getattr(model, "generation_config", None)
    export_config = deepcopy(live_config)
    export_generation_config = deepcopy(live_generation_config)
    try:
        model.config = export_config
        if live_generation_config is not None:
            model.generation_config = export_generation_config
        model.save_pretrained(destination, state_dict=state)
        # Transformers infers config.dtype from the live FP32 parameters during
        # save_pretrained, even when the supplied state_dict contains BF16.
        model.config.dtype = torch.bfloat16
        model.config.save_pretrained(destination)
    finally:
        model.config = live_config
        if live_generation_config is not None:
            model.generation_config = live_generation_config


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
