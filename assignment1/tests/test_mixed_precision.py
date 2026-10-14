from unittest.mock import Mock

import pytest
import torch

from train.model import SwiGLU, TransformerLM
from train.optimizer import TinyAdamW
from train.precision import MixedPrecision
from train.training_utils import (
    cross_entropy,
    get_cosine_schedule_with_warmup,
    load_checkpoint,
    save_checkpoint,
)


def test_cpu_bf16_trains_with_fp32_parameters_gradients_and_optimizer_state():
    torch.manual_seed(42)
    model = TransformerLM(
        vocab_size=32,
        context_length=8,
        d_model=16,
        num_layers=1,
        num_heads=2,
        d_ff=32,
        rope_theta=10000.0,
    )
    optimizer = TinyAdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    precision = MixedPrecision(torch.device("cpu"), "bf16")
    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))
    original_head = model.lm_head.weight.detach().clone()

    with precision.autocast():
        logits = model(inputs)
        loss = cross_entropy(logits.float(), targets)

    assert logits.dtype == torch.bfloat16
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    precision.backward(loss)
    precision.unscale_(optimizer)
    for parameter in model.parameters():
        assert parameter.dtype == torch.float32
        assert parameter.grad is not None
        assert parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all()

    assert precision.step(optimizer, scheduler)
    assert scheduler.last_epoch == 1
    assert not torch.equal(model.lm_head.weight, original_head)
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()
        for name in ("exp_avg", "exp_avg_sq"):
            moment = optimizer.state[parameter][name]
            assert moment.dtype == torch.float32
            assert torch.isfinite(moment).all()


@pytest.mark.parametrize(
    "scales,successful_step",
    [((128.0, 64.0), False), ((64.0, 64.0), True), ((64.0, 128.0), True)],
)
def test_scheduler_advances_only_after_successful_scaled_step(scales, successful_step):
    precision = MixedPrecision(torch.device("cpu"), "fp32")
    precision.scaler = Mock()
    precision.scaler.is_enabled.return_value = True
    precision.scaler.get_scale.side_effect = scales
    # Optimizer.step() normally returns None even when an update succeeds.
    precision.scaler.step.return_value = None
    optimizer = Mock()
    scheduler = Mock()

    assert precision.step(optimizer, scheduler) is successful_step

    precision.scaler.step.assert_called_once_with(optimizer)
    precision.scaler.update.assert_called_once_with()
    assert scheduler.step.call_count == int(successful_step)


def test_checkpoint_restores_scaler_and_scheduler_for_identical_next_updates(tmp_path):
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = TinyAdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    precision = MixedPrecision(torch.device("cpu"), "bf16")
    precision.scaler = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=2)
    inputs, targets = torch.randn(4, 3), torch.randn(4, 2)

    def update(current_model, current_optimizer, current_scheduler, current_precision):
        current_optimizer.zero_grad(set_to_none=True)
        with current_precision.autocast():
            loss = (current_model(inputs).float() - targets).square().mean()
        current_precision.backward(loss)
        scaled_gradient = current_model.weight.grad.detach().clone()
        scale = current_precision.scaler.get_scale()
        current_precision.unscale_(current_optimizer)
        torch.testing.assert_close(current_model.weight.grad, scaled_gradient / scale)
        assert current_precision.step(current_optimizer, current_scheduler)

    update(model, optimizer, scheduler, precision)
    checkpoint = tmp_path / "mixed_precision.pt"
    save_checkpoint(model, optimizer, 1, checkpoint, scaler=precision.scaler, scheduler=scheduler)

    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = TinyAdamW(restored_model.parameters(), lr=0.1)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=3, gamma=0.9)
    restored_precision = MixedPrecision(torch.device("cpu"), "bf16")
    restored_precision.scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    iteration = load_checkpoint(
        checkpoint,
        restored_model,
        restored_optimizer,
        "cpu",
        scaler=restored_precision.scaler,
        scheduler=restored_scheduler,
    )

    assert iteration == 1
    assert restored_precision.scaler.state_dict() == precision.scaler.state_dict()
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    for _ in range(2):
        update(model, optimizer, scheduler, precision)
        update(restored_model, restored_optimizer, restored_scheduler, restored_precision)
    assert restored_precision.scaler.state_dict() == precision.scaler.state_dict()
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    for original, restored in zip(model.parameters(), restored_model.parameters(), strict=True):
        torch.testing.assert_close(original, restored, rtol=0, atol=0)
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.state[original][name], restored_optimizer.state[restored][name], rtol=0, atol=0
            )


def test_loading_legacy_checkpoint_restores_schedule_and_starts_fresh_scaler(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = TinyAdamW(model.parameters())
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2, num_training_steps=10)
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(1, 3)).square().sum().backward()
        optimizer.step()
        scheduler.step()
    checkpoint = tmp_path / "legacy.pt"
    save_checkpoint(model, optimizer, 4, checkpoint)
    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = TinyAdamW(restored_model.parameters())
    restored_scheduler = get_cosine_schedule_with_warmup(
        restored_optimizer, num_warmup_steps=2, num_training_steps=10
    )
    scaler = torch.amp.GradScaler("cpu", init_scale=16.0)
    scaler_state = scaler.state_dict()

    with pytest.warns(UserWarning, match="no GradScaler state"):
        assert load_checkpoint(
            checkpoint,
            restored_model,
            restored_optimizer,
            "cpu",
            scaler=scaler,
            scheduler=restored_scheduler,
        ) == 4
    assert scaler.state_dict() == scaler_state
    assert restored_scheduler.last_epoch == 4
    assert restored_scheduler.get_last_lr() == scheduler.get_last_lr()
    optimizer.step()
    scheduler.step()
    restored_optimizer.step()
    restored_scheduler.step()
    assert restored_scheduler.get_last_lr() == scheduler.get_last_lr()


def test_real_grad_scaler_skips_nonfinite_gradient_update():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = TinyAdamW([parameter], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    precision = MixedPrecision(torch.device("cpu"), "bf16")
    precision.scaler = torch.amp.GradScaler("cpu", init_scale=16.0)

    precision.backward((parameter * float("inf")).sum())
    precision.unscale_(optimizer)
    assert not precision.step(optimizer, scheduler)

    torch.testing.assert_close(parameter, torch.ones(2), rtol=0, atol=0)
    assert not optimizer.state
    assert scheduler.last_epoch == 0
    assert precision.scaler.get_scale() == 8.0


def test_cpu_fp16_is_rejected():
    with pytest.raises(ValueError, match="CUDA|cuda|FP16|fp16"):
        MixedPrecision(torch.device("cpu"), "fp16")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_silu_extreme_inputs_have_finite_low_precision_gradients(dtype):
    inputs = torch.tensor([-1000.0, -100.0, -20.0, 0.0, 20.0, 100.0], dtype=dtype, requires_grad=True)
    outputs = SwiGLU(2, 4).Silu(inputs)
    outputs.float().sum().backward()

    assert outputs.dtype == dtype
    assert torch.isfinite(outputs).all()
    assert torch.isfinite(inputs.grad).all()
