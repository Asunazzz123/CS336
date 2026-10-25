"""GSM8K preparation and response-only batches for supervised fine-tuning.

Records contain complete, unshifted token sequences.  A causal language model
shifts ``labels`` internally; a custom loss must perform that shift exactly once.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class SFTExample:
    question: str
    rationale: str
    ground_truth: str
    prompt: str
    response: str


def _validate_template(prompt_template: str, response_format: str) -> None:
    if response_format not in {"r1_zero", "boxed"}:
        raise ValueError("response_format must be 'r1_zero' or 'boxed'")
    if not isinstance(prompt_template, str) or not prompt_template.strip():
        raise ValueError("prompt_template must be a nonempty string")
    try:
        fields = [name for _, name, _, _ in Formatter().parse(prompt_template) if name is not None]
    except ValueError as exc:
        raise ValueError(f"Invalid prompt template: {exc}") from exc
    if not fields or set(fields) != {"question"}:
        raise ValueError("prompt_template must contain {question} and no other fields")
    if response_format == "r1_zero" and not prompt_template.rstrip().endswith("<think>"):
        raise ValueError("An r1_zero prompt_template must end with <think>")
    try:
        prompt_template.format(question="example")
    except (ValueError, KeyError, IndexError) as exc:
        raise ValueError(f"Invalid prompt template: {exc}") from exc


def load_gsm8k_examples(
    path: Path,
    prompt_template: str,
    response_format: str = "r1_zero",
) -> list[SFTExample]:
    """Read GSM8K JSONL and retain its worked solutions as SFT targets.

    ``####`` separates the original reasoning from the final answer.  Outer
    whitespace is stripped, while internal newlines and GSM8K ``<<...>>``
    annotations are preserved.  Malformed records report the source line.
    """
    _validate_template(prompt_template, response_format)
    path = Path(path)
    examples: list[SFTExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: Invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{location}: Each record must be a JSON object")
            question = record.get("question")
            answer = record.get("answer")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{location}: question must be a nonempty string")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"{location}: answer must be a nonempty string")
            if "####" not in answer:
                raise ValueError(f"{location}: answer must contain the #### final-answer separator")
            rationale, ground_truth = (part.strip() for part in answer.rsplit("####", 1))
            if not rationale or not ground_truth:
                raise ValueError(f"{location}: Both rationale and final answer must be nonempty")
            question = question.strip()
            prompt = prompt_template.format(question=question)
            if response_format == "r1_zero":
                response = rationale + " </think> <answer> " + ground_truth + " </answer>"
            else:
                response = rationale + "\n\\boxed{" + ground_truth + "}"
            examples.append(SFTExample(question, rationale, ground_truth, prompt, response))
    if not examples:
        raise ValueError(f"{path}: No GSM8K examples found")
    return examples


def split_examples(
    examples: list[SFTExample],
    validation_fraction: float,
    seed: int,
) -> tuple[list[SFTExample], list[SFTExample]]:
    """Split shuffled question groups, keeping duplicate questions together.

    The fraction is applied to unique question groups, rounded to the nearest
    integer and clamped to leave at least one group on each side.  Consequently
    the fraction of records may differ when a question occurs multiple times.
    A local RNG leaves the caller's global random state untouched.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    groups: dict[str, list[SFTExample]] = {}
    for example in examples:
        question = example.question.strip()
        if not question:
            raise ValueError("Cannot split an example with an empty question")
        groups.setdefault(question, []).append(example)
    if len(groups) < 2:
        raise ValueError("At least two unique questions are required for a train/validation split")
    questions = list(groups)
    random.Random(seed).shuffle(questions)
    validation_count = max(1, min(len(questions) - 1, round(len(questions) * validation_fraction)))
    validation = [example for question in questions[:validation_count] for example in groups[question]]
    train = [example for question in questions[validation_count:] for example in groups[question]]
    return train, validation


def encode_sft_examples(
    examples: list[SFTExample],
    tokenizer: Any,
    max_seq_length: int,
) -> tuple[list[dict[str, list[int]]], dict[str, int]]:
    """Encode prompt/response separately and supervise response tokens plus EOS.

    Overlong examples are dropped whole: truncation could remove the final
    answer while leaving a misleading supervised target.  Trailing EOS tokens
    are normalized to exactly one, including when an input already ends in EOS.
    """
    if not isinstance(max_seq_length, int) or isinstance(max_seq_length, bool) or max_seq_length < 2:
        raise ValueError("max_seq_length must be an integer of at least 2")
    eos_token_id = tokenizer.eos_token_id
    if not isinstance(eos_token_id, int) or isinstance(eos_token_id, bool) or eos_token_id < 0:
        raise ValueError("tokenizer.eos_token_id must be a nonnegative integer")
    records: list[dict[str, list[int]]] = []
    stats = {"total_examples": len(examples), "kept_examples": 0, "dropped_too_long": 0}
    for index, example in enumerate(examples):
        if not example.prompt.strip() or not example.response.strip():
            raise ValueError(f"Example {index}: prompt and response must be nonempty")
        prompt_ids = list(tokenizer.encode(example.prompt, add_special_tokens=False))
        response_ids = list(tokenizer.encode(example.response, add_special_tokens=False))
        if not prompt_ids or not response_ids:
            raise ValueError(f"Example {index}: prompt and response must each encode to at least one token")
        while response_ids and response_ids[-1] == eos_token_id:
            response_ids.pop()
        response_ids.append(eos_token_id)
        input_ids = prompt_ids + response_ids
        if len(input_ids) > max_seq_length:
            stats["dropped_too_long"] += 1
            continue
        records.append({
            "input_ids": input_ids,
            "labels": [-100] * len(prompt_ids) + response_ids,
            "attention_mask": [1] * len(input_ids),
        })
    stats["kept_examples"] = len(records)
    return records, stats


def collate_sft_batch(
    records: list[dict[str, list[int]]],
    pad_token_id: int,
) -> dict[str, Tensor]:
    """Right-pad complete sequences without masking real EOS tokens as padding."""
    if not records:
        raise ValueError("Cannot collate an empty SFT batch")
    if not isinstance(pad_token_id, int) or isinstance(pad_token_id, bool) or pad_token_id < 0:
        raise ValueError("pad_token_id must be a nonnegative integer")
    lengths = [len(record["input_ids"]) for record in records]
    if any(length == 0 for length in lengths):
        raise ValueError("SFT records must contain at least one token")
    max_length = max(lengths)
    input_ids = torch.full((len(records), max_length), pad_token_id, dtype=torch.long)
    labels = torch.full((len(records), max_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(records), max_length), dtype=torch.long)
    for row, (record, length) in enumerate(zip(records, lengths)):
        if len(record["labels"]) != length:
            raise ValueError("SFT input_ids and labels must have equal lengths")
        if "attention_mask" in record and (
            len(record["attention_mask"]) != length or any(value != 1 for value in record["attention_mask"])
        ):
            raise ValueError("SFT records must be unpadded, with an all-ones attention_mask")
        input_ids[row, :length] = torch.tensor(record["input_ids"], dtype=torch.long)
        labels[row, :length] = torch.tensor(record["labels"], dtype=torch.long)
        attention_mask[row, :length] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
