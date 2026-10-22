"""本地训练入口集成检查，无需下载模型或使用 GPU。"""
import json
from dataclasses import replace
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from scripts.train_grpo import TrainConfig, Question, RolloutBatch, build_runtime, load_questions, sample_questions, train, train_on_rollouts

@pytest.fixture
def config(tmp_path):
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "What": 3, "is": 4,
             "one": 5, "plus": 6, "two": 7, "?": 8, "\\": 9,
             "boxed": 10, "{": 11, "}": 12, "2": 13, "3": 14}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    model_path = tmp_path / "model"
    tokenizer.save_pretrained(model_path)
    torch.manual_seed(123)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_embd=16, n_layer=1, n_head=2, n_positions=64, bos_token_id=1, eos_token_id=1, pad_token_id=0))
    model.save_pretrained(model_path)
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("\n".join(json.dumps({"question": q, "answer": "reasoning #### 2"}) for q in ["What is one plus one ?", "What is two ?"]), encoding="utf-8")
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("{question}", encoding="utf-8")
    return TrainConfig(str(model_path), data_path, data_path, prompt_path, tmp_path / "output", max_new_tokens=4, eval_examples=2, reward_type="gsm8k_numeric", learning_rate=1e-3)


def test_loading_and_epoch_sampling(config):
    questions = load_questions(config.train_path)
    assert len(questions) == 2 and all(q.ground_truth == "2" for q in questions)
    many = [Question(str(i), str(i)) for i in range(8)]
    epoch = [sample_questions(many, 2, step, 42) for step in range(1, 5)]
    assert len({q.question for batch in epoch for q in batch}) == 8
    assert sample_questions(many, 2, 1, 42) == epoch[0]


@pytest.mark.parametrize("optimizer_name", ["adam", "adamw", "muon"])
def test_nonzero_advantage_updates_parameters(config, optimizer_name):
    if optimizer_name == "muon" and not hasattr(torch.optim, "Muon"):
        pytest.skip("PyTorch does not provide Muon")
    config = replace(config, optimizer_name=optimizer_name)
    runtime = build_runtime(config)
    before = [p.detach().clone() for p in runtime.model.parameters()]
    prompt = "What is one plus one ?"
    rollouts = RolloutBatch([prompt] * 4, ["\\boxed{2}<eos>", "\\boxed{3}<eos>"] * 2, ["2"] * 4)
    loss, metadata = train_on_rollouts(runtime, rollouts, config)
    assert torch.isfinite(loss)
    assert metadata["mean_reward"] == 0.5
    assert torch.isfinite(metadata["token_entropy"])
    assert any(not torch.equal(old, new) for old, new in zip(before, runtime.model.parameters()))
    assert all(p.grad is None for p in runtime.model.parameters())
    runtime.optimizer.load_state_dict(runtime.optimizer.state_dict())


def test_two_step_generation_training_and_reload(config):
    train(config)
    records = [json.loads(line) for line in (config.output_dir / "metrics.jsonl").read_text().splitlines()]
    assert [(r["step"], r["split"]) for r in records] == [(0, "eval"), (1, "train"), (1, "eval"), (2, "train"), (2, "eval")]
    checkpoint = config.output_dir / "checkpoint-000002"
    model = GPT2LMHeadModel.from_pretrained(checkpoint)
    assert model.config.vocab_size == 15
    assert (checkpoint / "training_state.pt").exists()
    assert all(len(r["examples"]) > 0 for r in records)
    with pytest.raises(ValueError, match="输出目录非空"):
        train(config)
