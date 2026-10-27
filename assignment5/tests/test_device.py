"""Shared device detection is lazy, explicit and independent of rank binding."""
from contextlib import contextmanager
import importlib

import pytest
import torch

from utils import device


def test_import_and_cpu_resolution_never_probe_or_bind_accelerators(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Import/CPU path must not inspect an accelerator")

    for name in ("is_available", "device_count", "current_device", "set_device"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch.backends.mps, "is_available", forbidden)
    importlib.reload(device)
    assert device.resolve_device("cpu") == torch.device("cpu")
    assert device.resolve_device("cpu:0") == torch.device("cpu")
    for spec in ("cpu", "mps"):
        with pytest.raises(ValueError, match="CUDA"):
            device.get_device_id(spec)
        with pytest.raises(ValueError, match="CUDA"):
            device.cuda_supports_bf16(spec)


def test_availability_is_not_cached_and_nvidia_is_not_rocm(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "12.9")
    monkeypatch.setattr(torch.version, "hip", None)
    assert device.cuda_available() and device.nvidia_cuda_available()
    monkeypatch.setattr(torch.version, "hip", "6.4")
    assert device.cuda_available() and not device.nvidia_cuda_available()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not device.cuda_available()


def test_visible_ordinals_do_not_parse_rank_or_rewrite_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: pytest.fail("resolve_device bound a device"))
    assert device.resolve_device("cuda:1") == torch.device("cuda:1")
    assert device.get_device_id("cuda:1") == 1
    assert device.get_device_id("cuda") == 0
    assert device.resolve_device("cuda").index is None
    with pytest.raises(ValueError, match="visible device"):
        device.resolve_device("cuda:2")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "3,7"


@pytest.mark.parametrize("spec", ["cuda:0", "mps"])
def test_unavailable_explicit_device_never_falls_back(monkeypatch, spec):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        device.resolve_device(spec)


def test_mps_is_shared_but_not_a_cuda_ordinal(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert device.mps_available()
    assert device.resolve_device("mps:0") == torch.device("mps")
    with pytest.raises(ValueError, match="CUDA"):
        device.get_device_id("mps")
    with pytest.raises(ValueError):
        device.resolve_device("mps:1")
    with pytest.raises(ValueError, match="Unsupported"):
        device.resolve_device("meta")


def test_bf16_queries_selected_card_and_restores_current_device(monkeypatch):
    current = [0]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    @contextmanager
    def selected(selected_device):
        previous = current[0]
        current[0] = selected_device.index
        try:
            yield
        finally:
            current[0] = previous

    monkeypatch.setattr(torch.cuda, "device", selected)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported",
                        lambda including_emulation: current[0] == 1 and not including_emulation)
    assert device.cuda_supports_bf16("cuda:1", including_emulation=False)
    assert current[0] == 0
    with pytest.raises(ValueError, match="CUDA"):
        device.cuda_supports_bf16("cpu")
