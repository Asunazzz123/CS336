from __future__ import annotations

from typing import Callable, Literal

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from alignment.grpo import (
    run_tokenize_prompt_and_output,
    run_get_response_log_probs,
    run_compute_rollout_rewards,
    run_compute_group_normalized_rewards,
    run_compute_policy_gradient_loss,
    run_aggregate_loss_across_microbatch,
    run_grpo_train_step,
)


