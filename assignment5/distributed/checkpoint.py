"""Collective DCP checkpoints; publish only a complete directory."""
import json
import os
from pathlib import Path
import random
from uuid import uuid4

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_state_dict,
    set_model_state_dict, set_state_dict,
)

from .state import Phase

# Persisted format tag is independent of the import package name. Keep it
# stable across the nvidia_sys -> distributed source-directory rename.


def save(engine, path, *, include_optimizer=False, metadata=None):
    ctx = engine.context
    path = Path(path).expanduser().absolute()
    ctx.agree("checkpoint path", str(path))
    ctx.agree("checkpoint options", (include_optimizer, engine.policy_version, engine.rollout_id))

    def prepare():
        if ".partial-" in path.name:
            raise ValueError("The .partial- checkpoint name is reserved")
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        json.dumps(metadata)  # Fail before allocating a partial checkpoint.
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial-" + uuid4().hex)
        temporary.mkdir()
        return str(temporary)

    temporary = Path(ctx.root_call(prepare))
    engine.phase = Phase.CHECKPOINT
    try:
        engine._reshard()
        options = StateDictOptions(cpu_offload=True)
        if include_optimizer:
            model, optimizer = get_state_dict(engine.model, engine.optimizer, options=options)
            state = {"model": model, "optimizer": optimizer}
        else:
            state = {"model": get_model_state_dict(engine.model, options=options)}
        dcp.save(state, checkpoint_id=temporary / "shards")
        if include_optimizer:
            error = None
            try:
                torch.save({"torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state(ctx.device),
                            "python": random.getstate()}, temporary / f"rng-{ctx.rank}.pt")
            except Exception as exc:
                error = str(exc)
            if not ctx.all_true(error is None):
                raise RuntimeError(f"A rank could not save RNG state; local error: {error}")
        ctx.barrier()

        def publish():
            manifest = {"format": "nvidia_sys.dcp.v1", "world_size": ctx.world_size,
                        "include_optimizer": include_optimizer,
                        "policy_version": engine.policy_version, "rollout_id": engine.rollout_id,
                        "torch_version": torch.__version__, "metadata": metadata}
            (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            os.rename(temporary, path)

        ctx.root_call(publish)
        engine.phase = Phase.READY
        return path
    except BaseException:
        engine.phase = Phase.FAILED
        raise  # A .partial directory is deliberately never a loadable checkpoint.


def load(engine, path, *, restore_optimizer=False):
    ctx = engine.context
    path = Path(path).expanduser().absolute()
    ctx.agree("checkpoint load", (str(path), restore_optimizer))

    def read():
        if ".partial-" in path.name:
            raise ValueError("Refusing to load a partial checkpoint")
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("format") != "nvidia_sys.dcp.v1":
            raise ValueError("Unsupported checkpoint format")
        if restore_optimizer and not manifest["include_optimizer"]:
            raise ValueError("Model-only checkpoint cannot restore AdamW state")
        if restore_optimizer and manifest["world_size"] != ctx.world_size:
            raise ValueError("RNG continuation requires the same world size")
        return manifest

    manifest = ctx.root_call(read)
    if not engine.optimizer_on_gpu:
        raise RuntimeError("Load optimizer to CUDA before restoring a checkpoint")
    engine.phase = Phase.CHECKPOINT
    try:
        engine._reshard()
        options = StateDictOptions()
        if restore_optimizer:
            model, optimizer = get_state_dict(engine.model, engine.optimizer, options=options)
            state = {"model": model, "optimizer": optimizer}
            dcp.load(state, checkpoint_id=path / "shards")
            set_state_dict(engine.model, engine.optimizer, model_state_dict=state["model"],
                           optim_state_dict=state["optimizer"], options=options)
            error = None
            try:
                rng = torch.load(path / f"rng-{ctx.rank}.pt", map_location="cpu", weights_only=True)
                torch.set_rng_state(rng["torch"])
                torch.cuda.set_rng_state(rng["cuda"], ctx.device)
                random.setstate(rng["python"])
            except Exception as exc:
                error = str(exc)
            if not ctx.all_true(error is None):
                raise RuntimeError(f"A rank could not restore RNG; local error: {error}")
        else:
            state = {"model": get_model_state_dict(engine.model, options=options)}
            dcp.load(state, checkpoint_id=path / "shards")
            set_model_state_dict(engine.model, state["model"], options=options)
            engine.optimizer.state.clear()  # Never retain stale moments for newly loaded weights.
        engine.optimizer.zero_grad(set_to_none=True)
        engine.policy_version = manifest["policy_version"]
        engine.rollout_id = manifest["rollout_id"]
        engine.phase = Phase.READY
        return manifest["metadata"]
    except BaseException:
        engine.phase = Phase.FAILED
        raise
