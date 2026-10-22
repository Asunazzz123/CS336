# CS336 Assignment 5: Reasoning RL (GRPO)

GRPO-based RL training framework for LLM alignment. Core implementation of
on-policy and off-policy policy gradient with group-normalized rewards.

## Setup

```bash
uv sync
```

## Structure

```
alignment/
    grpo.py          # GRPO core: tokenization, log-probs, rewards, losses, train step
    checkpoint.py    # Model/tokenizer loading helper
    vllm_utils.py    # vLLM generation server + weight sync
    drgrpo_grader.py # DR-GRPO / r1_zero reward functions
    prompts/         # Prompt templates (r1_zero, question_only, three-shot)

scripts/
    train_grpo.py    # End-to-end training loop (transformers or vllm backend)

data/
    gsm8k/           # GSM8K train/test splits (JSONL)

tests/
    test_grpo.py         # Unit tests for all GRPO components
    test_train_grpo.py   # Integration tests for the training loop
```

## Run tests

```bash
uv run pytest
```

## Quick training run (CPU smoke test)

```bash
uv run python scripts/train_grpo.py \
    --model-id <hf-model-id> \
    --num-steps 2 \
    --prompts-per-batch 2 \
    --group-size 2 \
    --device cpu \
    --output-dir outputs/smoke
```
