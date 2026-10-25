"""SFT updates and the SFT→RL checkpoint boundary using a local tiny model."""

import copy
import json
from dataclasses import replace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from alignment.sft import SFTExample, collate_sft_batch, encode_sft_examples
from scripts.train_grpo import RolloutBatch, TrainConfig, build_runtime, load_questions, train_on_rollouts
from scripts.train_sft import SFTConfig, evaluate_sft, train, train_sft_step


@pytest.fixture
def tiny_sft_config(tmp_path):
    vocab = {
        "<pad>": 0, "<eos>": 1, "<unk>": 2,
        "What": 3, "is": 4, "one": 5, "plus": 6, "two": 7,
        "three": 8, "four": 9, "?": 10, "reason": 11,
        "\\": 12, "boxed": 13, "{": 14, "}": 15,
        "2": 16, "3": 17, "4": 18, "5": 19,
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
    )
    model_path = tmp_path / "base-model"
    tokenizer.save_pretrained(model_path)
    torch.manual_seed(31)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(vocab), n_embd=16, n_layer=1, n_head=2, n_positions=64,
        bos_token_id=1, eos_token_id=1, pad_token_id=0,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    ))
    model.save_pretrained(model_path)
    data_path = tmp_path / "gsm8k.jsonl"
    data_path.write_text("\n".join(json.dumps({
        "question": f"What is {number} plus one ?",
        "answer": f"reason {answer} #### {answer}",
    }) for number, answer in [("one", "2"), ("two", "3"), ("three", "4"), ("four", "5")]), encoding="utf-8")
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("{question}", encoding="utf-8")
    return SFTConfig(
        model_id=str(model_path), train_path=data_path, output_dir=tmp_path / "sft-output",
        prompt_path=prompt_path, response_format="boxed", validation_fraction=0.25,
        batch_size=1, gradient_accumulation_steps=2, num_epochs=2, max_steps=2,
        learning_rate=1e-3, max_seq_length=64, eval_every=1, save_every=1,
        eval_examples=4, eval_generation_examples=0,
    )


def _variable_length_records(tokenizer):
    examples = [SFTExample(
        question="What is one ?", rationale="reason", ground_truth="2",
        prompt="What is one ?", response=response,
    ) for response in ("2", "reason two 2", "reason one plus two 3")]
    records, _ = encode_sft_examples(examples, tokenizer, max_seq_length=64)
    return records


def test_accumulated_update_matches_a_full_batch_with_unequal_responses(tiny_sft_config):
    tokenizer = AutoTokenizer.from_pretrained(tiny_sft_config.model_id)
    model = AutoModelForCausalLM.from_pretrained(tiny_sft_config.model_id)
    accumulated_model = copy.deepcopy(model)
    records = _variable_length_records(tokenizer)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    accumulated_optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=0.02)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    full_loss, _ = train_sft_step(
        model, optimizer, records, tokenizer.pad_token_id,
        microbatch_size=3, max_grad_norm=1e9,
    )
    accumulated_loss, _ = train_sft_step(
        accumulated_model, accumulated_optimizer, records, tokenizer.pad_token_id,
        microbatch_size=2, max_grad_norm=1e9,
    )
    assert not full_loss.requires_grad
    torch.testing.assert_close(accumulated_loss, full_loss)
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    for actual, expected in zip(accumulated_model.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in accumulated_model.parameters())


def test_evaluation_is_token_weighted_and_preserves_model_mode(tiny_sft_config):
    tokenizer = AutoTokenizer.from_pretrained(tiny_sft_config.model_id)
    model = AutoModelForCausalLM.from_pretrained(tiny_sft_config.model_id)
    records = _variable_length_records(tokenizer)
    model.train()
    individual = evaluate_sft(model, records, tokenizer.pad_token_id, batch_size=1)
    assert model.training
    together = evaluate_sft(model, records, tokenizer.pad_token_id, batch_size=3)
    assert individual["loss"] == pytest.approx(together["loss"], rel=1e-6)
    assert individual["perplexity"] == pytest.approx(together["perplexity"], rel=1e-6)
    assert individual["supervised_tokens"] == sum(sum(label != -100 for label in record["labels"][1:]) for record in records)
    assert all(parameter.grad is None for parameter in model.parameters())
    model.eval()
    evaluate_sft(model, records, tokenizer.pad_token_id, batch_size=2)
    assert not model.training
    batch = collate_sft_batch(records, tokenizer.pad_token_id)
    with torch.inference_mode():
        expected = model(**batch, use_cache=False).loss
    assert together["loss"] == pytest.approx(expected.item(), rel=1e-6)


def test_sft_saves_a_checkpoint_that_rl_can_load(tiny_sft_config):
    tiny_sft_config = replace(
        tiny_sft_config,
        gradient_checkpointing=True,
        eval_generation_examples=1,
        max_new_tokens=4,
        generation_batch_size=1,
    )
    checkpoint = train(tiny_sft_config)
    assert checkpoint.name == "checkpoint-000002"
    assert (checkpoint / "training_state.pt").exists()
    saved_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    assert saved_tokenizer.eos_token_id == 1
    train_path = tiny_sft_config.output_dir / "data" / "train.jsonl"
    validation_path = tiny_sft_config.output_dir / "data" / "validation.jsonl"
    train_questions = load_questions(train_path)
    validation_questions = load_questions(validation_path)
    assert {example.question for example in train_questions}.isdisjoint(
        example.question for example in validation_questions
    )
    assert len(train_questions) + len(validation_questions) == 4

    rl_config = TrainConfig(
        str(checkpoint), train_path, validation_path, tiny_sft_config.prompt_path,
        tiny_sft_config.output_dir / "rl-output", reward_type="gsm8k_numeric",
    )
    runtime = build_runtime(rl_config)
    original = AutoModelForCausalLM.from_pretrained(tiny_sft_config.model_id)
    assert any(not torch.equal(old, new) for old, new in zip(original.parameters(), runtime.model.parameters()))
    assert runtime.model.config.vocab_size == len(saved_tokenizer)
    records = [json.loads(line) for line in (tiny_sft_config.output_dir / "metrics.jsonl").read_text().splitlines()]
    assert any(record["step"] == 2 and record["split"] == "train" for record in records)
    assert {record["step"] for record in records if record["split"] == "eval_generation"} >= {0, 2}

    before_rl = [parameter.detach().clone() for parameter in runtime.model.parameters()]
    rollouts = RolloutBatch(
        ["What is one plus one ?"] * 4,
        [r"\boxed{2}<eos>", r"\boxed{3}<eos>"] * 2,
        ["2"] * 4,
    )
    loss, metadata = train_on_rollouts(runtime, rollouts, rl_config)
    assert torch.isfinite(loss)
    assert metadata["mean_reward"] == 0.5
    assert any(not torch.equal(old, new) for old, new in zip(before_rl, runtime.model.parameters()))
    assert all(parameter.grad is None for parameter in runtime.model.parameters())
    with pytest.raises(ValueError):
        train(tiny_sft_config)


def test_prepare_only_does_not_load_model_weights(tiny_sft_config, monkeypatch):
    def forbid_weights(*args, **kwargs):
        raise AssertionError("prepare_only must not load model weights or tokenizer")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", forbid_weights)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", forbid_weights)
    output = train(replace(tiny_sft_config, prepare_only=True))
    assert output == tiny_sft_config.output_dir
    assert (output / "data" / "train.jsonl").exists()
    assert (output / "data" / "validation.jsonl").exists()
    assert not list(output.glob("checkpoint-*"))


def test_explicit_validation_cannot_contain_training_questions(tiny_sft_config):
    with pytest.raises(ValueError):
        train(replace(tiny_sft_config, eval_path=tiny_sft_config.train_path))
