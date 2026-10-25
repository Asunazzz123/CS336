"""Numerical and checkpoint regressions for FP32 CPU Adam offload."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import train_grpo, train_grpo_full, train_sft
from scripts.train_grpo import (
    CPUOffloadOptimizer, RolloutBatch, Runtime, TrainConfig, optimizer_switcher,
)


@pytest.mark.parametrize("optimizer_name", ["adam", "adamw"])
def test_cpu_offload_matches_adam_with_missing_and_unequal_gradients(optimizer_name):
    initial = [torch.tensor([[1.25, -0.6], [0.1, 2.2]]), torch.tensor([0.3, -0.75])]
    live = [torch.nn.Parameter(value.clone()) for value in initial]
    reference = [torch.nn.Parameter(value.clone()) for value in initial]
    kwargs = dict(lr=0.003, weight_decay=0.07, betas=(0.8, 0.95), eps=1e-7)
    offloaded = CPUOffloadOptimizer(live, optimizer_name, **kwargs)
    optimizer_class = getattr(torch.optim, "Adam" if optimizer_name == "adam" else "AdamW")
    standard = optimizer_class(reference, foreach=False, **kwargs)
    for step in range(5):
        for index, (parameter, expected) in enumerate(zip(live, reference)):
            gradient = (torch.arange(parameter.numel()).reshape_as(parameter) - 0.3) * (step + 1)
            gradient = None if step == 2 and index == 1 else gradient
            parameter.grad = gradient
            expected.grad = None if gradient is None else gradient.clone()
        offloaded.step()
        standard.step()
        for parameter, expected, master in zip(live, reference, offloaded.master_parameters):
            assert torch.equal(parameter, expected)
            assert torch.equal(master, expected)
            assert master.device.type == "cpu" and master.dtype == torch.float32
        offloaded.zero_grad(set_to_none=step % 2 == 0)
        standard.zero_grad(set_to_none=step % 2 == 0)
        for parameter, expected, master in zip(live, reference, offloaded.master_parameters):
            if expected.grad is None:
                assert parameter.grad is None and master.grad is None
            else:
                assert torch.count_nonzero(parameter.grad) == 0
                assert torch.count_nonzero(master.grad) == 0


@pytest.mark.parametrize("optimizer_name", ["adam", "adamw"])
def test_cpu_offload_checkpoint_restores_master_weights_and_update(tmp_path, optimizer_name):
    # BF16 exported model weights alone would lose this small FP32 update.
    original = torch.nn.Parameter(torch.ones(3, dtype=torch.float32))
    optimizer = CPUOffloadOptimizer([original], optimizer_name, lr=1e-6)
    original.grad = torch.tensor([0.2, -0.7, 1.3])
    optimizer.step()
    destination = tmp_path / "optimizer.pt"
    torch.save(optimizer.state_dict(), destination)
    restored = torch.nn.Parameter(original.detach().bfloat16().float())
    assert not torch.equal(restored, original)
    loaded = CPUOffloadOptimizer([restored], optimizer_name, lr=0.1)
    old_groups = loaded.param_groups
    loaded.load_state_dict(torch.load(destination, weights_only=True))
    assert torch.equal(restored, original)
    assert loaded.param_groups is loaded.optimizer.param_groups
    assert loaded.param_groups is not old_groups
    loaded.param_groups[0]["foreach"] = False
    assert loaded.optimizer.param_groups[0]["foreach"] is False
    for live, opt in ((original, optimizer), (restored, loaded)):
        live.grad = torch.tensor([-0.9, 0.4, 0.1])
        opt.step()
    assert torch.equal(restored, original)
    assert torch.equal(loaded.master_parameters[0], optimizer.master_parameters[0])
    assert all(value.device.type == "cpu" for state in loaded.optimizer.state.values()
               for value in state.values() if torch.is_tensor(value))


def test_cpu_offload_deduplicates_tied_parameters_and_cleans_gradient_views():
    parameter = torch.nn.Parameter(torch.ones(3))
    optimizer = CPUOffloadOptimizer([parameter, parameter], "adamw", lr=0.01)
    assert len(optimizer.master_parameters) == 1
    parameter.grad = torch.arange(6.0)[::2]
    optimizer.step()
    optimizer.zero_grad(set_to_none=False)
    assert torch.count_nonzero(parameter.grad) == 0
    optimizer.zero_grad(set_to_none=True)
    assert parameter.grad is None and optimizer.master_parameters[0].grad is None


def test_optimizer_switcher_preserves_default_and_rejects_cpu_muon():
    model = torch.nn.Linear(3, 2)
    legacy = SimpleNamespace(optimizer_name="adamw", learning_rate=1e-4, weight_decay=0.0)
    assert type(optimizer_switcher(model, legacy)) is torch.optim.AdamW
    legacy.optimizer_device = "cpu"
    assert isinstance(optimizer_switcher(model, legacy), CPUOffloadOptimizer)
    legacy.optimizer_name = "muon"
    with pytest.raises(ValueError, match="only supports adam/adamw"):
        optimizer_switcher(model, legacy)


@pytest.mark.parametrize("module", [train_grpo, train_grpo_full, train_sft])
def test_optimizer_device_cli_default_and_forwarding(monkeypatch, module):
    monkeypatch.setattr("sys.argv", [str(module.__file__), "--model-id", "local-model"])
    parsed = module.parse_args()
    config = parsed.base if module is train_grpo_full else parsed
    assert config.optimizer_device == "model"
    monkeypatch.setattr("sys.argv", [str(module.__file__), "--model-id", "local-model",
                                    "--optimizer-device", "cpu"])
    parsed = module.parse_args()
    config = parsed.base if module is train_grpo_full else parsed
    assert config.optimizer_device == "cpu"
    if module is train_sft:
        assert module._runtime_config(config, Path("prompt.txt")).optimizer_device == "cpu"
    monkeypatch.setattr("sys.argv", [str(module.__file__), "--model-id", "local-model",
                                    "--optimizer-device", "cpu", "--optimizer-name", "muon"])
    with pytest.raises(SystemExit):
        module.parse_args()


@pytest.mark.parametrize("stage", ["sft", "rl", "grpo_full"])
def test_cpu_offload_integrates_with_accumulated_clipped_training(
    stage, tiny_train_model, tokenizer, tmp_path, monkeypatch,
):
    from alignment.sft import SFTExample, encode_sft_examples

    model = tiny_train_model
    reference = deepcopy(model)
    offloaded = CPUOffloadOptimizer(model.parameters(), "adamw", lr=1e-4, weight_decay=0.0)
    standard = torch.optim.AdamW(reference.parameters(), lr=1e-4, foreach=False,
                                weight_decay=0.0)
    base = TrainConfig("unused", tmp_path / "train", tmp_path / "eval", tmp_path / "prompt",
                       tmp_path / "output", group_size=2, gradient_accumulation_steps=2)
    reward = lambda config: lambda response, truth: {
        "reward": float(response == truth), "format_reward": 0.0,
        "answer_reward": float(response == truth),
    }
    monkeypatch.setattr(train_grpo, "reward_function", reward)
    monkeypatch.setattr(train_grpo_full, "reward_function", reward)
    rollouts = RolloutBatch(["Hello"] * 2, ["world", "another test"], ["world"] * 2)
    losses = []
    for live, optimizer in ((model, offloaded), (reference, standard)):
        if stage == "sft":
            records, _ = encode_sft_examples([
                SFTExample("Hello", "", "world", "Hello", response)
                for response in rollouts.responses
            ], tokenizer, max_seq_length=16)
            loss, _ = train_sft.train_sft_step(live, optimizer, records,
                                              tokenizer.pad_token_id, microbatch_size=1)
        else:
            runtime = Runtime(live, tokenizer, optimizer, live)
            if stage == "rl":
                loss, _ = train_grpo.train_on_rollouts(runtime, rollouts, base)
            else:
                loss, _ = train_grpo_full.train_on_rollouts_full(
                    runtime, rollouts, train_grpo_full.FullGRPOConfig(base),
                )
        losses.append(loss)
        assert all(parameter.grad is None for parameter in live.parameters())
    assert torch.equal(losses[0], losses[1])
    for actual, expected in zip(model.parameters(), reference.parameters()):
        assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("optimizer_name", ["adam", "adamw"])
def test_cuda_model_cpu_offload_matches_fp32_cpu_reference(optimizer_name):
    # Run on each visible device, validating the transfer boundaries on the server.
    for device_index in range(torch.cuda.device_count()):
        device = torch.device("cuda", device_index)
        parameter = torch.nn.Parameter(torch.tensor([0.75, -1.25, 0.1], device=device))
        reference = torch.nn.Parameter(parameter.detach().cpu())
        offloaded = CPUOffloadOptimizer([parameter], optimizer_name, lr=1e-6, weight_decay=0.03)
        reference_class = getattr(torch.optim, "Adam" if optimizer_name == "adam" else "AdamW")
        standard = reference_class([reference], lr=1e-6, weight_decay=0.03, foreach=False)
        for gradient in (torch.tensor([0.2, -0.3, 1.1]), torch.tensor([-0.7, 0.1, 0.5])):
            parameter.grad = gradient.to(device)
            reference.grad = gradient.clone()
            offloaded.step()
            standard.step()
            assert torch.equal(parameter.cpu(), reference)
            offloaded.zero_grad()
        assert parameter.grad is None
        assert all(value.device.type == "cpu" for state in offloaded.optimizer.state.values()
                   for value in state.values() if torch.is_tensor(value))
