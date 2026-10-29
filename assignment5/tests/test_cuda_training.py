"""Offline CUDA smoke tests for Qwen3.5 hybrid attention training and decoding.

Run on the target GPU machine with ``pytest tests/test_cuda_training.py -q``.
Each visible GPU (up to two) executes SFT, basic RL, and full GRPO separately.
"""

import gc
from dataclasses import replace

import pytest
import torch

from alignment.sft import SFTExample, encode_sft_examples
from training.types import RolloutBatch, Runtime
from training.configs import UnifiedGRPOConfig as TrainConfig
from training.trainers.grpo import train_rollout_batch as train_on_rollouts
from training.trainers.sft import train_sft_step


CUDA_DEVICES = [f"cuda:{index}" for index in range(min(torch.cuda.device_count(), 2))]
pytestmark = pytest.mark.skipif(not CUDA_DEVICES, reason="CUDA GPU required")


class CheckedSGD(torch.optim.SGD):
    """Inspect real backward gradients before the training step clears them."""

    def step(self, closure=None):
        gradients = [parameter.grad for group in self.param_groups
                     for parameter in group["params"] if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) for gradient in gradients)
        self.checked_gradients = True
        return super().step(closure)


@pytest.fixture(params=CUDA_DEVICES or ["cuda:0"])
def cuda_runtime(request):
    # Import only after CUDA skip handling: local Transformers versions may not
    # yet provide Qwen3.5, and this test must not trigger model downloads.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig

    device = request.param
    torch.manual_seed(7)
    backend = Tokenizer(WordLevel({
        "<pad>": 0, "<eos>": 1, "<unk>": 2, "Question": 3, "Answer": 4,
        "2": 5, "3": 6, "\\": 7, "boxed": 8, "{": 9, "}": 10,
    }, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>",
        unk_token="<unk>",
    )
    config = Qwen3_5TextConfig(
        vocab_size=len(tokenizer), hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=128, linear_conv_kernel_dim=4, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_num_key_heads=2, linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"], attention_dropout=0.0,
        max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=1,
    )
    model = Qwen3_5ForCausalLM(config).to(device=device, dtype=torch.float32)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    optimizer = CheckedSGD(model.parameters(), lr=0.01)
    yield Runtime(model, tokenizer, optimizer, model)
    del optimizer, model
    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()


@pytest.mark.parametrize("stage", ["sft", "rl", "grpo_full"])
def test_qwen35_amp_checkpoint_training_and_cached_decode(cuda_runtime, stage, tmp_path):
    runtime = cuda_runtime
    model, tokenizer = runtime.model, runtime.tokenizer
    device = next(model.parameters()).device
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    before = [parameter.detach().clone() for parameter in model.parameters()]
    base = TrainConfig(
        model_id="offline-tiny-qwen35", train_path=tmp_path / "train.jsonl",
        eval_path=tmp_path / "eval.jsonl", prompt_path=tmp_path / "prompt.txt",
        output_dir=tmp_path / "output", device=str(device), dtype="float32",
        autocast="bfloat16", gradient_checkpointing=True, group_size=2,
        gradient_accumulation_steps=2,
    )
    responses = [r"\boxed{2}", r"\boxed{3}"]
    if stage == "sft":
        records, _ = encode_sft_examples([
            SFTExample("Question", "", "2", "Question Answer", response)
            for response in responses
        ], tokenizer, max_seq_length=64)
        loss, _ = train_sft_step(
            model, runtime.optimizer, records, tokenizer.pad_token_id,
            microbatch_size=1, autocast="bfloat16",
        )
    else:
        rollouts = RolloutBatch(["Question Answer"] * 2, responses, ["2"] * 2)
        if stage == "rl":
            loss, _ = train_on_rollouts(runtime, rollouts, base)
        else:
            loss, _ = train_on_rollouts(runtime, rollouts, replace(base, num_epochs_per_rollout=3))
    torch.cuda.synchronize(device)
    assert torch.isfinite(loss)
    assert runtime.optimizer.checked_gradients
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    del before

    # Exercise both hybrid attention cache implementations after the update.
    model.eval()
    input_ids = torch.tensor([[3, 4]], device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            use_cache=True, do_sample=False, max_new_tokens=2,
            pad_token_id=0, eos_token_id=None,
        )
    torch.cuda.synchronize(device)
    assert output.shape == (1, 4)
    assert output.device == device

