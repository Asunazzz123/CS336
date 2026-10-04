# Assignment 1 training core

The core training code is managed independently in this CS336 repository,
under `assignment1/`. Learn no longer retains the former extracted copy.
The original learning files remain in Learn under `assignment1-basics`.

```text
assignment1/
  train/       model, tokenizer, optimizer, training entry point and utilities
  scripts/     tokenizer inspection tools
  tests/       core tests, reference fixtures and snapshots
  pyproject.toml
```

The BPE pre-tokenization experiments, experiment outputs, course PDF, submission
script, virtual environments and caches are excluded. BPE tokenizer code and
its core tests are retained because tokenization is required by training.
Reference fixture files in tests are required test inputs, not trained outputs.

Run from this directory using the existing agent environment:

```sh
conda run -n agent python -m train.train --help
conda run -n agent python -m pytest tests/test_optimizer.py tests/test_nn_utils.py
conda run -n agent python scripts/inspect_tokenizer.py
```

The extraction updates package imports and paths, preserving the current
training algorithms. Startup and uint16 batch conversion have been fixed.
Tokenizer artifact loading and generation context management still need work.
A full 10,000-step TinyStories run has been completed on a single RTX 5090,
as documented below. Multi-GPU validation has not been completed.

## Single-device performance probes

Single-device runs enable performance probes by default. The first five steps
of each invocation are excluded from stable throughput measurements. Change
this with `--probe_warmup_steps N`, or disable probes with `--disable_probe`.
Warmup steps are real training updates and still count toward total tokens.

The output directory receives `metrics.jsonl` (append-only events with run IDs)
and `performance_summary.json` (latest invocation summary). Training logs report
loss, learning rate used and next learning rate, total/session token counts,
mean/std step time, tokens/s and mean phase times for data loading including
transfer, forward/loss, backward including zero-grad, clipping and optimizer/
scheduler. Validation, generation and checkpoint times are separate overheads.

CUDA phase timings synchronize the selected device before and after each
phase. This intentionally adds instrumentation overhead and prevents overlap
between phases; disable probes to measure the uninstrumented training path.
Peak allocated/reserved CUDA bytes cover measured training steps, excluding
validation and generation. CPU runs report these memory fields as null.
DDP runs do not enable these probes; local rank timing is not a distributed
throughput benchmark. Resume starts a new measurement window and warmup.

## Verified TinyStories training result (2026-10-04)

This run was executed on AutoDL using an NVIDIA GeForce RTX 5090 (32 GB),
driver 580.105.08, Python 3.12 and PyTorch 2.11.0+cu130 (CUDA 13.0 build).
Training used FP32 with `torch.compile` disabled and synchronized performance
probes enabled; the first 10 training steps were excluded from timing statistics.

### Model and data

- Model: 43,803,136 parameters; 8 layers, hidden size 512, 8 attention heads,
  SwiGLU intermediate size 2048 and context length 256. Input embeddings and
  the output head have separate weights.
- Tokenizer: byte-level BPE with GPT-2 regex pre-tokenization, vocabulary size
  10,000 and 9,743 merges. `<|endoftext|>` has token ID 256.
- The tokenizer was trained on the entire TinyStories V2 GPT4 training text
  (2,227,753,162 bytes), using 8 CPU workers, in 32.66 seconds.
- Encoded training data: 541,229,347 tokens, stored in a 1,082,458,822-byte
  `uint16` NumPy file. Validation data: 5,465,883 tokens in 10,931,894 bytes.
- Encoding both splits took 43.94 seconds. Every shard passed an exact byte
  roundtrip check; final files passed SHA256 checks. CUDA batch loading was
  verified to return `torch.int64` tensors with the correct next-token shift.

Upstream data: [TinyStories V2 GPT4 training text](https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStoriesV2-GPT4-train.txt)
and [validation text](https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStoriesV2-GPT4-valid.txt).

### Training configuration and measurements

The run used batch size 16, 10,000 optimizer steps, peak learning rate `3e-4`,
500 warmup steps, minimum learning-rate ratio 0.1, AdamW betas `(0.9, 0.95)`,
weight decay 0.1, gradient clipping at 1.0 and seed 42. Validation ran every
500 steps with 50 random batches; checkpoints were saved every 1,000 steps.

| Metric | Latest completed run |
| --- | ---: |
| Training device | NVIDIA GeForce RTX 5090, 32 GB |
| Final iteration | 10,000 |
| Tokens processed | 40,960,000 |
| Session wall time | 785.38 s (13 min 5 s) |
| Measured steps / excluded warmup steps | 9,990 / 10 |
| Mean training step time | 75.43 ms |
| Step time standard deviation | 8.92 ms |
| Training throughput | 54,300 tokens/s |
| Measured training-step time | 753.57 s |
| Validation overhead | 22.31 s |
| Checkpoint overhead | 4.54 s |
| Training-step peak allocated CUDA memory | 3.77 GiB |
| Training-step peak reserved CUDA memory | 4.11 GiB |
| Last training batch loss | 1.7583 |
| Final validation loss | 1.693185 |
| Validation perplexity, `exp(loss)` | approximately 5.44 |

Training throughput excludes validation and checkpoint overhead. CUDA peaks
cover measured training steps, not the entire session. Two same-configuration
runs (`141826` and `142021`) overlapped in time and both completed; therefore
these measurements are not an isolated single-GPU benchmark. Their recorded
validation curves agree. The latest run is reported in the table above.

| Iteration | Validation loss |
| --- | ---: |
| 500 | 3.0243 |
| 1,000 | 2.5762 |
| 2,000 | 2.2381 |
| 5,000 | 1.8429 |
| 8,500 | 1.7003 |
| 10,000 | 1.6932 |

The final validation loss is the lowest recorded value in this run, without
a sustained increase in the validation curve. Each evaluation samples
204,800 tokens across 50 batches; it is not a full-validation-corpus loss.
The processed-token budget is approximately 7.6% of the training corpus token
count, with random-window sampling, so this is not a full epoch or evidence of
complete convergence.

### Checkpoint and generation checks

The final checkpoint loaded successfully at iteration 10,000, and every model
parameter was finite. Three prompts were sampled with temperature 0.8 and a
limit of 120 new tokens. The model produces recognizable short English
stories, but grammar and narrative consistency remain imperfect. For example,
the bird story generated:

> Lily was happy to see her hurt wing. She some strong wings made her wing happier and better.

This demonstrates learning of the TinyStories language patterns while exposing
remaining subject-reference, grammar and story-logic errors. No claim of
general language-model quality is made from this small sample.

AutoDL artifact locations (these files are not stored in this Git repository):

```text
/root/autodl-tmp/CS336/assignment1/
  tokenizers/tinystories-10k/tokenizer.json
  tokenizers/tinystories-10k/training_metadata.json
  data/encoded/tinystories-10k/train.npy
  data/encoded/tinystories-10k/valid.npy
  data/encoded/tinystories-10k/encoding_metadata.json
  data/encoded/tinystories-10k/loader_verification.json
  outputs/tinystories_10k_20261004_142021/
    config.json
    final_model.pt
    checkpoint_10000.pt
    metrics.jsonl
    performance_summary.json
    result_review.json
```

The AutoDL-only helpers `scripts/train_tokenizer.py` and `scripts/prepare_data.py`
were added for this run and have not yet been copied into this local checkout.
The tokenizer artifact uses the lossless `cs336-bpe-hex-v1` JSON format and is
loaded by the remote helper's `load_tokenizer` function. The current training
CLI's `--vocab_path` / `--merges_path` loader does not support this format yet;
training-time generation was disabled, and the samples above were generated
after training by a separate checkpoint review script.

## Independent Git management

Develop, commit and push from the sibling CS336 repository:

```sh
cd ../CS336
git add -- assignment1
git commit -m "Update Assignment 1 training code"
git push
```

These commands assume the starting directory is the Learn repository root.
Configure a CS336 remote before pushing. Learn no longer tracks its local
`learning_code/CS336/assignment1` copy, and pushing Learn does not publish new
changes from that copy. The copies do not synchronize automatically.

The original subtree import and Learn `assignment1-core` branch remain as
historical references. Do not use the former Learn subtree synchronization
commands for ongoing development. Existing Learn history still contains the
initial extraction; ignoring the directory does not rewrite old commits.
