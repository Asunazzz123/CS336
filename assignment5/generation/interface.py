from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class PolicySnapshot:
    path: Path
    version: int


@dataclass(frozen=True)
class TokenCompletion:
    text: str
    token_ids: list[int]
    finish_reason: str | None


class GenerationBackend(Protocol):
    """Receiver acknowledges a fully loaded policy before serving token IDs."""
    def load_policy(self, snapshot: PolicySnapshot) -> int: ...
    def generate(self, prompt_ids: list[list[int]], sampling: dict) -> list[TokenCompletion]: ...
    def close(self) -> None: ...
