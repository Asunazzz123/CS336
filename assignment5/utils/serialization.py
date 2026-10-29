"""JSON-compatible conversion without importing training applications."""
from pathlib import Path
import torch


def json_value(value):
    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value
