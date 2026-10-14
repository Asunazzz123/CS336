# Assignment 1: TinyStories Training Results

A custom Transformer language model was trained on TinyStories using an **NVIDIA GeForce RTX 5090 (32 GB)**. The experiments include an initial FP32 baseline and two completed sweeps of 64 hyperparameter configurations each, using FP32 and BF16 mixed precision. Each sweep run processes exactly **40,960,000 training tokens** to compare learning rate, batch size, and warmup fraction under the same token budget.



## FP32 Training Result

### Model and Dataset Size

| Property | Value |
| --- | ---: |
| Trainable parameters | 43,803,136 (approximately 43.8 million) |
| Transformer layers | 8 |
| Hidden dimension | 512 |
| Attention heads | 8 |
| SwiGLU intermediate dimension | 2048 |
| Context length | 256 tokens |
| Vocabulary size | 10,000 |
| BPE merges | 9,743 |
| Training dataset tokens | 541,229,347 |
| Validation dataset tokens | 5,465,883 |

The input embeddings and output head have separate weights. The tokenizer uses byte-level BPE with GPT-2 regex pre-tokenization. The special token `<|endoftext|>` has ID 256.

The data comes from the [TinyStories V2 GPT4 training split](https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStoriesV2-GPT4-train.txt) and [validation split](https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStoriesV2-GPT4-valid.txt). Tokenizer training used the entire training text (2,227,753,162 bytes) and took 32.66 seconds. Encoding both splits took 43.94 seconds; every shard passed an exact byte roundtrip check.

Each model processes a token budget equivalent to approximately **7.6%** of the training corpus token count. Training samples random windows, so this fraction does not represent a complete pass through that portion of the corpus.

### Experimental Design and Training Budget

The full Cartesian product of learning rates, batch sizes, and warmup fractions contains **4 × 4 × 4 = 64 independent runs**. All 64 runs completed successfully.

| Variable | Candidate values |
| --- | --- |
| Peak learning rate | `1e-4`, `3e-4`, `1e-3`, `3e-3` |
| Batch size | 16, 32, 64, 128 |
| Warmup fraction | 0%, 1%, 3%, 5% |

Every run processes **40,960,000 tokens**, for a combined total of **2,621,440,000 tokens** across the sweep. Update counts and warmup steps vary with batch size:

| Batch size | Tokens per step | Optimizer updates | Warmup steps: 0% / 1% / 3% / 5% |
| ---: | ---: | ---: | --- |
| 16 | 4,096 | 10,000 | 0 / 100 / 300 / 500 |
| 32 | 8,192 | 5,000 | 0 / 50 / 150 / 250 |
| 64 | 16,384 | 2,500 | 0 / 25 / 75 / 125 |
| 128 | 32,768 | 1,250 | 0 / 13 / 38 / 63 |

Warmup steps are rounded up. For batch size 128, the actual fractions are therefore 0%, 1.04%, 3.04%, and 5.04%.

All other settings remain fixed: FP32 training, AdamW betas `(0.9, 0.95)`, weight decay 0.1, gradient clipping threshold 1.0, and random seed 42. After warmup, the learning rate follows cosine decay to 10% of its peak value.

Each run has 10 evaluations, spaced by 4,096,000 training tokens. An evaluation uses 50 batches of size 16, totaling **204,800 validation tokens**. The same fixed windows are used across all runs and evaluations, and validation sampling does not consume the training RNG state. These losses estimate performance on that fixed validation sample, not the entire validation corpus.

### Best Training Result

Among the 64 tested configurations, the lowest final validation loss was obtained with:

| Property | Best configuration and result |
| --- | ---: |
| Peak learning rate | `3e-3` |
| Batch size | 128 |
| Warmup | 5% (63 steps; actual fraction 5.04%) |
| Optimizer updates | 1,250 |
| Training tokens | 40,960,000 |
| Final validation loss | **1.556914** |
| Validation perplexity | **4.744157** |
| Run duration | 403.23 seconds (approximately 6 min 43 s) |
| Training throughput | 104,396 tokens/s |
| Peak allocated memory during training steps | 26.60 GiB |

The following plots use this configuration as the baseline. Each panel varies one hyperparameter while holding the other two at their best values. All curves use the same training-token budget.

#### Validation Loss Curves

The horizontal axis measures processed training tokens, allowing comparison across different batch sizes and update counts. Curves show unsmoothed measurements.

![Validation loss over training tokens, with other variables fixed at their best values](result/asset/FP32/validation_learning_curves.png)

#### Final Validation Loss versus Hyperparameters

Within these conditional slices, increasing learning rate, batch size, and warmup fraction reduced final validation loss. The three panels use separate vertical-axis ranges, and each point is labeled with its loss.

![Final validation loss versus one hyperparameter, with the others fixed at their best values](result/asset/FP32/final_validation_sensitivity.png)

These are conditional slices through the best configuration; they do not establish that hyperparameter interactions are absent. The best configuration lies at the boundary of the tested range, so conclusions apply only to the tested candidates. The result is also selected on the tuning validation set, rather than evaluated on an independent test set.

#### Training Loss Curves

The following plots show unsmoothed losses from individual training batches. Fluctuation levels differ with batch size; training quality should also be assessed using the fixed-sample validation losses.

![Training batch loss over training tokens, with other variables fixed at their best values](result/asset/FP32/training_learning_curves.png)

### Runtime and Memory

Results below are grouped by batch size, with 16 learning-rate/warmup combinations per row. Run duration includes training, validation, and result saving. Throughput measures training steps only.

| Batch size | Mean run duration | Mean training throughput | Peak allocated memory during training steps |
| ---: | ---: | ---: | ---: |
| 16 | 631.03 s (10.52 min) | 66,625 tokens/s | 3.77 GiB |
| 32 | 406.06 s (6.77 min) | 104,117 tokens/s | 7.03 GiB |
| 64 | 396.98 s (6.62 min) | 106,238 tokens/s | 13.56 GiB |
| 128 | 403.77 s (6.73 min) | 104,301 tokens/s | 26.60 GiB |

The sum of all 64 run durations is **29,405.41 seconds, approximately 8 hours 10 minutes**. Throughput improves substantially from batch size 16 to 32, then remains broadly similar at 64 and 128 while memory usage increases. Synchronized performance probes were enabled throughout this sweep, so reported throughput includes their overhead. Memory peaks cover measured training steps only.

**Runtime hypothesis.** The higher runtime at batch size 16 may be largely driven by CPU-side kernel dispatch and frequent probe synchronization. The probes introduce approximately 12 explicit CUDA synchronizations per training step. With a fixed token budget, batch size 16 requires 10,000 steps, compared with 1,250 at batch size 128, resulting in eight times as many probe synchronization calls. The observed saturation of one CPU core alongside roughly 50–60% GPU utilization is consistent with substantial CPU-side overhead. This remains a hypothesis: a controlled comparison with probes disabled is needed to distinguish their contribution from eager kernel dispatch, the custom optimizer, and other synchronization points.

### Initial Baseline and Generation Quality

Before the sweep, a baseline model was trained for 10,000 steps using learning rate `3e-4`, batch size 16, and 5% warmup. It also processed 40,960,000 tokens:

| Metric | Baseline result |
| --- | ---: |
| Final validation loss | 1.693185 |
| Validation perplexity | approximately 5.44 |
| Total duration | 785.38 s (approximately 13 min 5 s) |
| Training throughput | 54,300 tokens/s |
| Peak allocated memory during training steps | 3.77 GiB |

This baseline used randomly sampled validation windows, unlike the fixed windows used in the later sweep. Two baseline runs overlapped in time, so their speed does not represent isolated GPU performance.

The baseline checkpoint loaded successfully and passed finite-parameter checks. With temperature 0.8 and a limit of 120 new tokens, it produced recognizable short English stories, but still showed errors in subject reference, grammar, and narrative logic. For example:

> Lily was happy to see her hurt wing. She some strong wings made her wing happier and better.

These samples came from the initial baseline model. The best sweep model has not yet undergone the same generation assessment. Lower validation loss indicates improved prediction on the fixed validation sample; it alone does not establish a corresponding improvement in story quality.


## Mixed-Precision Training Result

### Model and Dataset Size

The BF16 sweep reuses the model architecture, tokenizer, and encoded datasets from the FP32 experiments. Each configuration trains a fresh model; the sweep does not resume the FP32 checkpoints.

| Property | Value |
| --- | ---: |
| Trainable parameters | 43,803,136 (approximately 43.8 million) |
| Transformer layers | 8 |
| Hidden dimension | 512 |
| Attention heads | 8 |
| SwiGLU intermediate dimension | 2048 |
| Context length | 256 tokens |
| Vocabulary size | 10,000 |
| BPE merges | 9,743 |
| Training dataset tokens | 541,229,347 |
| Validation dataset tokens | 5,465,883 |

The input embeddings and output head have separate weights. The same byte-level BPE tokenizer, special token `<|endoftext|>` (ID 256), and TinyStories V2 GPT4 train/validation splits are used. No tokenizer retraining or data re-encoding is included in the BF16 sweep duration.

Each model processes a token budget equivalent to approximately **7.6%** of the training corpus token count. As in the FP32 sweep, random-window sampling does not represent a complete pass through that portion of the corpus.

### Experimental Design and Training Budget

The BF16 sweep repeats the complete **4 × 4 × 4 = 64 independent runs** from the FP32 experiment. All 64 runs completed successfully. The exported plans and per-run configurations match except for precision and output directory.

| Variable | Candidate values |
| --- | --- |
| Peak learning rate | `1e-4`, `3e-4`, `1e-3`, `3e-3` |
| Batch size | 16, 32, 64, 128 |
| Warmup fraction | 0%, 1%, 3%, 5% |

Every run processes **40,960,000 tokens**, for a combined total of **2,621,440,000 tokens** across the BF16 sweep. Update counts and warmup steps are unchanged:

| Batch size | Tokens per step | Optimizer updates | Warmup steps: 0% / 1% / 3% / 5% |
| ---: | ---: | ---: | --- |
| 16 | 4,096 | 10,000 | 0 / 100 / 300 / 500 |
| 32 | 8,192 | 5,000 | 0 / 50 / 150 / 250 |
| 64 | 16,384 | 2,500 | 0 / 25 / 75 / 125 |
| 128 | 32,768 | 1,250 | 0 / 13 / 38 / 63 |

Warmup steps are rounded up. For batch size 128, the actual fractions are 0%, 1.04%, 3.04%, and 5.04%. All other training settings remain fixed: AdamW betas `(0.9, 0.95)`, weight decay 0.1, gradient clipping threshold 1.0, and random seed 42. After warmup, the learning rate follows cosine decay to 10% of its peak value.

Each run has 10 evaluations, spaced by 4,096,000 training tokens. Each evaluation uses 50 batches of size 16, totaling **204,800 validation tokens**. The fixed validation windows and seed 1234 match the FP32 sweep, and validation sampling does not consume the training RNG state. Validation uses BF16 autocast with cross-entropy computed in FP32; these losses estimate performance on the fixed validation sample, not the entire validation corpus.

#### Precision Policy

The default training entry point, `scripts/train_mixed_precision.py`, uses **BF16 + FP32 automatic mixed precision (AMP)**. Precision is assigned as follows:

| Computation or data | Precision |
| --- | --- |
| Q, K, V projections and the attention output projection | BF16 matrix multiplication |
| Attention matrix multiplications: $QK^\top$ and $PV$ | BF16 matrix multiplication |
| The three FFN linear layers, SiLU, and elementwise gating | BF16 |
| Final LM head and output logits | BF16 |
| RoPE rotation | FP32 internally, then cast back to BF16 |
| RMSNorm and attention softmax | FP32 internally, then cast back to the input dtype |
| Embedding outputs and residual additions | FP32 |
| Cross-entropy loss | FP32, after converting logits to FP32 |
| Parameter storage, final `param.grad` buffers, AdamW moment buffers, and parameter updates | FP32 |

BF16 matrix multiplication refers to the operand and output dtypes; the internal accumulation precision depends on the backend kernel. Backward matrix multiplications use the dtypes selected for the corresponding forward operations, while final parameter gradients accumulate in FP32 buffers. In this implementation, RMSNorm receives and returns FP32 tensors, and attention softmax returns BF16 tensors.

The complete sweep entry point is `scripts/train_mixed_precision_sweep.py`; the default BF16 output directory is `outputs/sweep_tinystories_10k_bf16`.

### Best Training Result

Among the 64 tested BF16 configurations, the lowest final validation loss was obtained with:

| Property | Best configuration and result |
| --- | ---: |
| Peak learning rate | `3e-3` |
| Batch size | 128 |
| Warmup | 5% (63 steps; actual fraction 5.04%) |
| Optimizer updates | 1,250 |
| Training tokens | 40,960,000 |
| Final validation loss | **1.558090** |
| Validation perplexity | **4.749739** |
| Run duration | 249.96 seconds (approximately 4 min 10 s) |
| Training throughput | 169,981 tokens/s |
| Peak allocated memory during training steps | 17.29 GiB |

The selected hyperparameters are the same as the FP32 sweep's best configuration. Its BF16 validation loss is 0.001176 higher than the recorded FP32 value of 1.556914. The following plots use the BF16 best configuration as the baseline. Each panel varies one hyperparameter while holding the other two at their best values, with the same training-token budget in every run.

#### Validation Loss Curves

The horizontal axis measures processed training tokens, allowing comparison across different batch sizes and update counts. Curves show the 10 unsmoothed validation measurements from each selected BF16 run.

![BF16 validation loss over training tokens, with other variables fixed at their best values](result/asset/Mixed-Precision/bf16_validation_learning_curves.png)

#### Final Validation Loss versus Hyperparameters

Within these conditional slices, increasing learning rate, batch size, and warmup fraction reduced final validation loss while the other settings remained fixed at the BF16 best values. The three panels use separate vertical-axis ranges, and each point is labeled with its loss.

![BF16 final validation loss versus one hyperparameter, with the others fixed at their best values](result/asset/Mixed-Precision/bf16_final_validation_sensitivity.png)

These slices do not establish that hyperparameter interactions are absent. The selected configuration lies at the boundary of the tested range, so conclusions apply only to the tested candidates. It is selected on the tuning validation set, rather than evaluated on an independent test set.

#### Training Loss Curves

The following plots show unsmoothed losses from individual training batches at the recorded logging steps. Fluctuation levels differ with batch size; training quality should also be assessed using the fixed-sample validation losses.

![BF16 training batch loss over training tokens, with other variables fixed at their best values](result/asset/Mixed-Precision/bf16_training_learning_curves.png)

### Runtime and Memory

Results below are grouped by batch size, with 16 learning-rate/warmup combinations per row. Run duration includes training, validation, and result saving. Throughput measures training steps only, excluding the first 10 warmup steps. These measurement warmup steps are separate from learning-rate warmup.

| Batch size | Mean run duration | Mean training throughput | Peak allocated memory during training steps |
| ---: | ---: | ---: | ---: |
| 16 | 650.33 s (10.84 min) | 64,502 tokens/s | 2.68 GiB |
| 32 | 338.19 s (5.64 min) | 125,359 tokens/s | 4.76 GiB |
| 64 | 233.36 s (3.89 min) | 182,959 tokens/s | 8.94 GiB |
| 128 | 249.92 s (4.17 min) | 170,082 tokens/s | 17.29 GiB |

The sum of all 64 BF16 run durations is **23,548.73 seconds, approximately 6 hours 32 minutes 29 seconds**, compared with 29,405.41 seconds for FP32: a **19.9% reduction** in cumulative duration. Relative to the FP32 means, training throughput is 0.968×, 1.204×, 1.722×, and 1.631× at batch sizes 16, 32, 64, and 128, respectively. Peak allocated memory is lower by 28.9%, 32.3%, 34.1%, and 35.0%. The highest mean throughput occurs at batch size 64; increasing to 128 raises memory use without improving throughput.

Synchronized performance probes were enabled throughout both sweeps, so the reported throughput includes their overhead. Memory peaks cover measured training steps only. The recorded GPU and PyTorch versions match, but these are historical comparisons from separate executions, not a contemporaneous controlled benchmark. BF16 validation uses autocast, whereas FP32 validation uses FP32; the BF16 implementation also uses the mathematically equivalent `x * sigmoid(x)` SiLU expression instead of the earlier `x / (1 + exp(-x))` expression. Consequently, the comparison is not a strict precision-only ablation.

**Runtime hypothesis.** Batch size 16 remains the slowest setting and its mean training throughput is 3.2% lower than FP32. CPU-side kernel dispatch, the custom FP32 optimizer, and synchronization may limit the benefit of faster BF16 matrix multiplications at this batch size. The probe still introduces approximately 12 explicit CUDA synchronizations per training step, with eight times as many steps at batch size 16 as at 128 under the same token budget. These observations do not isolate the probe's contribution: a controlled probe-on/probe-off comparison and a short CPU/CUDA timeline are needed to distinguish synchronization, kernel dispatch, and GPU execution costs.


