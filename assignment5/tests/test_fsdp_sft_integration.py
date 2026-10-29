"""SFT response weighting, padded rank tails and HF handoff contracts."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM

from alignment.sft import collate_sft_batch
from training.backends import fsdp_sft as runtime
from training.trainers.sft import train
from .test_distributed_runtime import LocalCollectives, local_engine
from .test_fsdp_grpo_integration import tiny_model
from .test_train_sft import tiny_sft_config
from .test_train_grpo import config


def records():
    return [{"input_ids": ids, "labels": [-100, *ids[1:]], "attention_mask": [1] * len(ids)}
            for ids in ([2, 3, 1], [4, 5, 6, 1], [7, 8, 1])]


def test_sft_tail_padding_preserves_exact_global_gradient_and_token_count():
    data = records()
    model = tiny_model().eval()
    reference = deepcopy(model)
    expected = runtime.loss_output(reference, collate_sft_batch(data, 0))
    expected.loss_sum.backward()
    weights, counts = [], []
    for rank in (0, 1):
        batches = runtime.padded_microbatches(data, [0, 1, 2], rank=rank, world_size=2,
            batch_size=1, accumulation=2, pad_id=0, device="cpu")
        outputs = [runtime.loss_output(model, b) for b in batches]
        weights.append(sum(output.weight for output in outputs))
        counts.append(len(outputs))
        for output in outputs:
            output.loss_sum.backward()
    assert counts == [2, 2] and sum(weights) == expected.weight == 7
    for actual, wanted in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, wanted.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("best_mode", [False, True])
def test_fsdp_sft_entry_runs_tail_batch_and_exports_hf_for_rl(tiny_sft_config, monkeypatch, best_mode):
    cfg = replace(tiny_sft_config, training_backend="fsdp2", save_optimizer_state=False,
                  best_checkpoint_dir=tiny_sft_config.output_dir.parent / "ram" if best_mode else None)
    class Context(LocalCollectives):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(runtime.NvidiaContext, "from_torchrun", lambda: Context())
    built = []
    def build(ctx, factory, config):
        engine = local_engine(factory())
        engine.context, engine.config = ctx, config
        built.append(engine)
        return engine
    monkeypatch.setattr(runtime, "NvidiaFSDPEngine", build)
    result = train(cfg)
    model = AutoModelForCausalLM.from_pretrained(result, local_files_only=True, dtype="auto")
    assert next(model.parameters()).dtype == torch.bfloat16
    assert json.loads((result / "config.json.run").read_text())["model_id"] == cfg.model_id
    assert (cfg.output_dir / "data/train.jsonl").is_file()
    assert (cfg.output_dir / "data/validation.jsonl").is_file()
    logs = [json.loads(line) for line in (cfg.output_dir / "metrics.jsonl").read_text().splitlines()]
    assert len([r for r in logs if r["split"] == "train"]) == 2
    if best_mode:
        assert result.name == "checkpoint-best"
        assert not cfg.best_checkpoint_dir.exists()


def test_rank_without_validation_examples_does_not_change_metric_denominator():
    engine = local_engine(tiny_model())
    engine.context.rank, engine.context.world_size = 1, 2
    data = records()[:1]
    def reduce(totals):
        assert totals.tolist() == [0, 0]
        totals.add_(torch.tensor([6.0, 2.0]))
    engine.context.sum = reduce
    metrics = runtime.evaluate_sft(engine, data, 0, 1)
    assert metrics["loss"] == 3.0 and metrics["supervised_tokens"] == 2


def test_no_best_grpo_hf_output_is_directly_reloadable(config, monkeypatch):
    from training.backends import fsdp_grpo as fsdp_grpo_runtime
    from training.configs import UnifiedGRPOConfig
    from training.trainers.grpo import train as train_grpo
    class Context(LocalCollectives):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(fsdp_grpo_runtime.NvidiaContext, "from_torchrun", lambda: Context())
    def build(ctx, factory, cfg):
        engine = local_engine(factory())
        engine.context, engine.config = ctx, cfg
        return engine
    monkeypatch.setattr(fsdp_grpo_runtime, "NvidiaFSDPEngine", build)
    cfg = UnifiedGRPOConfig(**{**vars(config), "training_backend": "fsdp2", "num_steps": 1,
        "save_optimizer_state": False, "checkpoint_format": "hf"})
    train_grpo(cfg)
    model = AutoModelForCausalLM.from_pretrained(cfg.output_dir / "checkpoint-000001",
                                              local_files_only=True, dtype="auto")
    assert next(model.parameters()).dtype == torch.bfloat16
    assert not list(cfg.output_dir.glob("resume-*"))
