"""BF16 checkpoint export must preserve live FP32 training state and ties."""
from dataclasses import dataclass, replace
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from training.configs import RuntimeConfig as TrainConfig
from checkpointing.hf import bf16_cpu_state_dict as _bf16_cpu_state_dict
from training.checkpoint_policy import save_checkpoint
from training.configs import SFTConfig


@pytest.fixture
def checkpoint_runtime(tmp_path):
    torch.manual_seed(17)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=7, n_embd=8, n_layer=1, n_head=2, n_positions=16,
        bos_token_id=1, eos_token_id=1, pad_token_id=0,
    ))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(
            {"<pad>": 0, "<eos>": 1, "<unk>": 2, "a": 3, "b": 4, "c": 5, "d": 6},
            unk_token="<unk>",
        )),
        pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    runtime = SimpleNamespace(model=model, tokenizer=tokenizer, optimizer=optimizer)
    config = TrainConfig("unused", tmp_path, tmp_path, tmp_path, tmp_path / "output")
    return runtime, config


def test_cpu_export_preserves_tied_views_and_buffer_types():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    model.tied = model.weight
    model.register_buffer("view_a", model.weight.detach()[1:])
    model.register_buffer("view_b", model.weight.detach()[1:])
    model.register_buffer("different_view", model.weight.detach()[:2])
    model.register_buffer("indices", torch.tensor([2**40 + 1, -7], dtype=torch.int64))
    model.register_buffer("mask", torch.tensor([True, False]))

    exported = _bf16_cpu_state_dict(model)

    assert all(tensor.device.type == "cpu" for tensor in exported.values())
    assert exported["weight"] is exported["tied"]
    assert exported["view_a"] is exported["view_b"]
    assert exported["view_a"] is not exported["different_view"]
    assert exported["weight"].dtype == torch.bfloat16
    assert exported["weight"].data_ptr() != model.weight.data_ptr()
    for name in ("indices", "mask"):
        torch.testing.assert_close(exported[name], getattr(model, name))
    for name in ("view_a", "view_b", "different_view"):
        torch.testing.assert_close(exported[name], getattr(model, name).to(torch.bfloat16))
    assert model.weight.dtype == torch.float32


@pytest.mark.parametrize("save_optimizer_state", [True, False])
def test_bf16_checkpoint_roundtrip_preserves_ties_and_live_state(
    checkpoint_runtime, save_optimizer_state,
):
    runtime, config = checkpoint_runtime
    config = replace(config, save_optimizer_state=save_optimizer_state)
    model = runtime.model
    live_config = model.config
    live_generation_config = model.generation_config
    config_before = live_config.to_dict()
    generation_before = live_generation_config.to_dict()
    parameters_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    save_checkpoint(runtime, config, 3)

    checkpoint = config.output_dir / "checkpoint-000003"
    stored = load_file(checkpoint / "model.safetensors")
    assert sum(tensor.numel() for tensor in stored.values()) == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert len({"transformer.wte.weight", "lm_head.weight"} & stored.keys()) == 1
    assert all(tensor.dtype == torch.bfloat16 for tensor in stored.values())
    assert json.loads((checkpoint / "config.json").read_text())["dtype"] == "bfloat16"
    reloaded = GPT2LMHeadModel.from_pretrained(checkpoint, dtype="auto")
    assert next(reloaded.parameters()).dtype == torch.bfloat16
    assert reloaded.transformer.wte.weight.data_ptr() == reloaded.lm_head.weight.data_ptr()
    for name, parameter in reloaded.named_parameters():
        torch.testing.assert_close(parameter, parameters_before[name].to(torch.bfloat16), rtol=0, atol=0)
    assert len(PreTrainedTokenizerFast.from_pretrained(checkpoint)) == len(runtime.tokenizer)

    assert model.config is live_config
    assert model.generation_config is live_generation_config
    assert model.config.to_dict() == config_before
    assert model.generation_config.to_dict() == generation_before
    for name, parameter in model.named_parameters():
        assert parameter.dtype == torch.float32
        torch.testing.assert_close(parameter, parameters_before[name], rtol=0, atol=0)
    training_state = torch.load(checkpoint / "training_state.pt", weights_only=False)
    assert training_state["step"] == 3
    assert ("optimizer" in training_state) == save_optimizer_state
    if save_optimizer_state:
        assert training_state["optimizer"]["state"]


def test_sft_config_and_legacy_config_keep_optimizer_state(checkpoint_runtime):
    runtime, config = checkpoint_runtime

    @dataclass
    class LegacySFTConfig:
        output_dir: object

    configurations = [
        SFTConfig("unused", config.train_path, config.output_dir / "sft"),
        LegacySFTConfig(config.output_dir / "legacy-sft"),
    ]
    for sft_config in configurations:
        save_checkpoint(runtime, sft_config, 1)
        saved = torch.load(sft_config.output_dir / "checkpoint-000001" / "training_state.pt", weights_only=False)
        assert saved["optimizer"]["state"]


def test_failed_model_export_restores_live_config(checkpoint_runtime, monkeypatch):
    runtime, config = checkpoint_runtime
    live_config = runtime.model.config
    live_generation_config = runtime.model.generation_config
    config_before = live_config.to_dict()

    def fail_save(*args, **kwargs):
        runtime.model.config.dtype = torch.bfloat16
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(runtime.model, "save_pretrained", fail_save)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        save_checkpoint(runtime, config, 1)
    assert runtime.model.config is live_config
    assert runtime.model.generation_config is live_generation_config
    assert live_config.to_dict() == config_before
    assert next(runtime.model.parameters()).dtype == torch.float32
