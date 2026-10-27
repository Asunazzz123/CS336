"""
Lazy device checks shared by algorithms, launchers and distributed backends.

No import-time probing, cached availability, persistent binding, rank parsing, or
silent fallback. CUDA indices always refer to the caller's visible devices.
"""
import torch


def cuda_available() -> bool:
    """PyTorch CUDA API availability, including ROCm builds."""
    return torch.cuda.is_available()


def nvidia_cuda_available() -> bool:
    return (torch.version.cuda is not None and torch.version.hip is None
            and cuda_available())


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return backend is not None and backend.is_available()


def resolve_device(spec: str | torch.device) -> torch.device:
    """Validate an explicitly selected CPU/CUDA/MPS device, without binding it."""
    device = torch.device(spec)
    if device.type == "cpu":
        if device.index not in (None, 0):
            raise ValueError("CPU device index must be omitted or zero")
        return torch.device("cpu")
    if device.type == "cuda":
        if not cuda_available():
            raise RuntimeError("Requested CUDA device is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA ordinal {device.index} is outside the visible device range")
        return device
    if device.type == "mps":
        if device.index not in (None, 0):
            raise ValueError("MPS device index must be omitted or zero")
        if not mps_available():
            raise RuntimeError("Requested MPS device is unavailable")
        return torch.device("mps")
    raise ValueError(f"Unsupported device type: {device.type}")


def get_device_id(spec: str | torch.device = "cuda") -> int:
    """Return a CUDA visible ordinal, never a physical ID or LOCAL_RANK.

    CPU/MPS are rejected. An omitted CUDA index may initialize CUDA through
    current_device(), only when this function is explicitly called.
    """
    device = torch.device(spec)
    if device.type != "cuda":
        raise ValueError("get_device_id requires a CUDA device")
    device = resolve_device(device)
    return device.index if device.index is not None else torch.cuda.current_device()


def cuda_supports_bf16(spec: str | torch.device = "cuda", *, including_emulation=True) -> bool:
    """Query a selected CUDA/ROCm card and restore the previous current card.

    This intentionally makes no claims about CPU or MPS BF16 support.
    """
    device = torch.device(spec)
    if device.type != "cuda":
        raise ValueError("cuda_supports_bf16 requires a CUDA device")
    device = resolve_device(device)
    with torch.cuda.device(device):
        return torch.cuda.is_bf16_supported(including_emulation=including_emulation)
