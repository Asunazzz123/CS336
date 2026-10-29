"""Training selection and state persistence atop generic checkpoint writers."""
import json
import logging
import random
from dataclasses import asdict
import numpy as np
import torch
from checkpointing.best import BestCheckpointStore
from checkpointing.hf import save_bf16_model
from utils.serialization import json_value


def make_best_store(config, metric_name, mode):
    if config.best_checkpoint_dir is None:
        return None
    return BestCheckpointStore(config.best_checkpoint_dir, metric_name, mode)


def consider_best(runtime, config, store, step, metrics, *, eval_path, eval_examples):
    if store is None:
        return False
    def save(path):
        save_bf16_model(runtime.model, path)
        runtime.tokenizer.save_pretrained(path)
        (path / "config.json.run").write_text(json.dumps(json_value(asdict(config)), indent=2))
    return store.consider(step, metrics[store.metric_name], save, metadata={
        "checkpoint_dtype": "bfloat16", "live_parameter_dtype": str(next(runtime.model.parameters()).dtype),
        "metric_model": "live_policy_before_export", "optimizer_saved": False,
        "selection_split": "validation", "eval_path": str(eval_path), "eval_examples": eval_examples,
    })


def finish_best(store, output_dir):
    destination = store.export(output_dir / "checkpoint-best")
    (output_dir / "best_checkpoint.json").write_text(json.dumps(
        {**store.selection, "path": str(destination)}, indent=2) + "\n")
    try:
        store.cleanup()
    except OSError:
        logging.getLogger(__name__).warning("Best exported; RAM cleanup failed", exc_info=True)
    return destination


def save_checkpoint(runtime, config, step):
    destination = config.output_dir / f"checkpoint-{step:06d}"
    destination.mkdir(parents=True, exist_ok=False)
    # BF16 导出在 CPU 上完成；训练中的 FP32 主权重和配置保持原样。
    save_bf16_model(runtime.model, destination)
    runtime.tokenizer.save_pretrained(destination)
    training_state = {"step": step,
                      "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                      "numpy_rng": np.random.get_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    if getattr(config, "save_optimizer_state", True):
        training_state["optimizer"] = runtime.optimizer.state_dict()
    torch.save(training_state, destination / "training_state.pt")
    (destination / "config.json.run").write_text(json.dumps(json_value(asdict(config)), indent=2), encoding="utf-8")
