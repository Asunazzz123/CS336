# Assignment 1 training core

The core training code is managed independently in the sibling CS336 repository,
under `CS336/assignment1/`. This copy inside Learn is ignored by Learn Git.
The original learning files remain in `assignment1-basics`.

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
Full real-corpus training and multi-GPU validation have not been completed.

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
