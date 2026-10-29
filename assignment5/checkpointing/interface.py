"""Versioned CPU weight-stream receiver contract."""
from typing import Protocol
import torch


class WeightSink(Protocol):
    """Called only on rank 0. commit must acknowledge the published version.

    The receiver owns any required checkpoint-to-serving name/layout mapping.
    write receives a detached CPU tensor, one parameter/buffer at a time.
    Failed transactions must remain unavailable to generation until retried.
    """
    def begin(self, version: int) -> None: ...
    def write(self, name: str, tensor: torch.Tensor) -> None: ...
    def commit(self, version: int) -> int: ...
    def abort(self, version: int) -> None: ...

