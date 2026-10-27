"""Generation services independent of alignment algorithms and FSDP engines."""
from .interface import PolicySnapshot, GenerationBackend, TokenCompletion

__all__ = ["PolicySnapshot", "GenerationBackend", "TokenCompletion"]
