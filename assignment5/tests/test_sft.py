"""GSM8K SFT data boundaries and causal-label checks, without external models."""

import json
import random
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from alignment.sft import (
    SFTExample,
    collate_sft_batch,
    encode_sft_examples,
    load_gsm8k_examples,
    split_examples,
)


@pytest.fixture
def sft_tokenizer():
    vocab = {
        "<pad>": 0, "<eos>": 1, "<unk>": 2,
        "Question": 3, ":": 4, "one": 5, "Answer": 6,
        "reason": 7, "two": 8, "three": 9,
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )


def _example(question="one", response="reason two"):
    return SFTExample(
        question=question,
        rationale="reason",
        ground_truth="two",
        prompt=f"Question: {question} Answer:",
        response=response,
    )


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return path


def test_gsm8k_keeps_reasoning_and_matches_rl_response_format(tmp_path):
    path = _write_jsonl(tmp_path / "gsm8k.jsonl", [
        {"question": " How many? ", "answer": " Compute <<1+1=2>>2.\n#### 2 "},
    ])
    example, = load_gsm8k_examples(path, "User: {question}\nAssistant: <think>")
    assert example.question == "How many?"
    assert example.rationale == "Compute <<1+1=2>>2."
    assert example.ground_truth == "2"
    assert example.prompt.endswith("<think>")
    assert example.response.strip() == "Compute <<1+1=2>>2. </think> <answer> 2 </answer>"
    assert "####" not in example.response

    boxed, = load_gsm8k_examples(path, "Question: {question}\nAnswer:", response_format="boxed")
    assert example.rationale in boxed.response
    assert r"\boxed{2}" in boxed.response
    assert "<answer>" not in boxed.response


@pytest.mark.parametrize("bad_line", [
    '{"question": "q", "answer": "no final-answer separator"}',
    '{"question": "q", "answer": "reason #### "}',
    '{"question": " ", "answer": "reason #### 2"}',
    '{"question": "q", "answer": " #### 2"}',
    '{"question": "q"}',
    '{"question": "q", "answer": 2}',
    '{not valid JSON}',
])
def test_gsm8k_errors_identify_the_source_line(tmp_path, bad_line):
    path = tmp_path / "broken.jsonl"
    path.write_text(
        json.dumps({"question": "valid", "answer": "reason #### 2"}) + "\n" + bad_line,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"broken\.jsonl:2"):
        load_gsm8k_examples(path, "{question} <think>")


def test_r1_response_requires_the_prompt_to_open_think(tmp_path):
    path = _write_jsonl(tmp_path / "gsm8k.jsonl", [{"question": "q", "answer": "reason #### 2"}])
    with pytest.raises(ValueError, match="think"):
        load_gsm8k_examples(path, "{question}")


def test_validation_split_is_seeded_and_keeps_duplicate_questions_together():
    examples = [_example(str(index)) for index in range(10)]
    examples += [replace(examples[0], question=" 0 ", response="another solution")]
    original = list(examples)
    random_state = random.getstate()
    train, validation = split_examples(examples, validation_fraction=0.3, seed=19)
    assert random.getstate() == random_state
    assert (train, validation) == split_examples(examples, validation_fraction=0.3, seed=19)
    assert len(train) + len(validation) == len(examples)
    train_questions = {example.question.strip() for example in train}
    validation_questions = {example.question.strip() for example in validation}
    assert train_questions and validation_questions
    assert train_questions.isdisjoint(validation_questions)
    assert len([example for example in train + validation if example.question.strip() == "0"]) == 2
    assert examples == original


def test_causal_labels_supervise_the_first_response_token_and_eos(sft_tokenizer):
    example = _example()
    records, stats = encode_sft_examples([example], sft_tokenizer, max_seq_length=32)
    record, = records
    prompt_ids = sft_tokenizer.encode(example.prompt, add_special_tokens=False)
    response_ids = sft_tokenizer.encode(example.response, add_special_tokens=False)
    boundary = len(prompt_ids)
    assert record["input_ids"] == prompt_ids + response_ids + [sft_tokenizer.eos_token_id]
    assert record["labels"] == [-100] * boundary + response_ids + [sft_tokenizer.eos_token_id]
    assert record["attention_mask"] == [1] * len(record["input_ids"])
    assert stats["total_examples"] == stats["kept_examples"] == 1
    assert stats["dropped_too_long"] == 0

    # HF causal LM shifts internally: the final prompt position predicts the
    # first response token; pre-shifting labels would silently skip that target.
    batch = collate_sft_batch(records, pad_token_id=sft_tokenizer.pad_token_id)
    torch.manual_seed(7)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(sft_tokenizer), n_embd=8, n_layer=1, n_head=1,
        n_positions=32, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    ))
    output = model(**batch)
    shifted_labels = batch["labels"][:, 1:]
    assert shifted_labels[0, boundary - 1].item() == response_ids[0]
    assert (shifted_labels != -100).sum().item() == len(response_ids) + 1
    expected = F.cross_entropy(
        output.logits[:, :-1].reshape(-1, len(sft_tokenizer)),
        shifted_labels.reshape(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(output.loss, expected)


def test_padding_does_not_mask_a_real_eos_when_pad_equals_eos(sft_tokenizer):
    sft_tokenizer.pad_token = sft_tokenizer.eos_token
    records, _ = encode_sft_examples(
        [_example(response="two"), _example(response="reason two three")],
        sft_tokenizer,
        max_seq_length=32,
    )
    batch = collate_sft_batch(records, pad_token_id=sft_tokenizer.pad_token_id)
    for index, record in enumerate(records):
        length = len(record["input_ids"])
        assert batch["input_ids"][index, length - 1].item() == sft_tokenizer.eos_token_id
        assert batch["labels"][index, length - 1].item() == sft_tokenizer.eos_token_id
        assert batch["attention_mask"][index, length - 1].item() == 1
        assert torch.all(batch["input_ids"][index, length:] == sft_tokenizer.pad_token_id)
        assert torch.all(batch["labels"][index, length:] == -100)
        assert torch.all(batch["attention_mask"][index, length:] == 0)
    assert batch["input_ids"].dtype == torch.long


def test_existing_terminal_eos_is_not_duplicated(sft_tokenizer):
    example = _example(response="two <eos><eos>")
    records, _ = encode_sft_examples([example], sft_tokenizer, max_seq_length=32)
    record, = records
    assert record["labels"][-1] == sft_tokenizer.eos_token_id
    assert record["labels"][-2] == sft_tokenizer.convert_tokens_to_ids("two")


def test_overlength_examples_are_dropped_whole_instead_of_truncating_answers(sft_tokenizer):
    short = _example(response="two")
    full, _ = encode_sft_examples([short], sft_tokenizer, max_seq_length=32)
    exact_length = len(full[0]["input_ids"])
    long = replace(short, response="reason two three")
    records, stats = encode_sft_examples([short, long], sft_tokenizer, max_seq_length=exact_length)
    assert records == full
    assert stats["total_examples"] == 2
    assert stats["kept_examples"] == 1
    assert stats["dropped_too_long"] == 1
    assert records[0]["labels"][-2:] == [sft_tokenizer.convert_tokens_to_ids("two"), sft_tokenizer.eos_token_id]

    records, stats = encode_sft_examples([short], sft_tokenizer, max_seq_length=exact_length - 1)
    assert records == []
    assert stats["dropped_too_long"] == 1
