"""Application-owned samples, rollout batches and runtime handles."""
from dataclasses import dataclass
from typing import Any


@dataclass
class Question:
    question: str
    ground_truth: str


@dataclass
class RolloutBatch:
    # 同题回答连续排列，三个列表一一对应
    repeated_prompts: list[str]
    responses: list[str]
    repeated_ground_truths: list[str]


@dataclass
class Runtime:
    model: Any
    tokenizer: Any
    optimizer: Any
    generator: Any
