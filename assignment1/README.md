# Assignment 1 training core

This directory is the core training snapshot maintained as the `assignment1`
subtree of the CS336 repository. The original learning files remain in Learn.

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
training algorithms. Previously identified training issues remain: the local
json import inside main, uint16 batch conversion, tokenizer artifact loading
and generation context management. This extraction is not a claim that full
training has passed.

## Subtree synchronization

Learn prefix: `learning_code/CS336/assignment1`

CS336 prefix: `assignment1`

From Learn, split the current core directory:

```sh
git subtree split --prefix=learning_code/CS336/assignment1 -b assignment1-core
```

From CS336, merge updates from the local Learn repository:

```sh
git subtree pull --prefix=assignment1 ../Learn assignment1-core --squash
```

For updates made in CS336, split its prefix into a branch:

```sh
git subtree split --prefix=assignment1 -b assignment1-core
```

Then from Learn:

```sh
git subtree pull --prefix=learning_code/CS336/assignment1 ../CS336 assignment1-core --squash
```

Split branches must be refreshed before pulling. Neither split command pushes
to GitHub. Develop in one copy at a time and synchronize explicitly.
