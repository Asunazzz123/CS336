"""GRPO helpers shared by training and test adapters."""
from __future__ import annotations
from typing import Callable, Literal
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask aligned with
    labels that is 1 for response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str]
            List of prompt strings.
        output_strs: list[str]
            List of output strings.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor].
            Let prompt_and_output_lens be a list containing the lengths of the
            concatenated tokenized prompt and output strings. Then the returned
            dictionary should have the following keys:

            input_ids
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): the tokenized
                prompt and output strings, with the final token sliced off.
            labels
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): shifted input
                ids, i.e., the input ids without the first token.
            response_mask
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): a mask aligned
                with labels, with value 1 where the corresponding label token
                is part of the response and 0 otherwise.
    """

    # 1. tokenize prompt / output
    prompt_tokens = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]

    output_tokens = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]

    batch_size = len(prompt_strs)

    # 2. 分别记录长度
    prompt_lens = torch.tensor(
        [len(x) for x in prompt_tokens],
        dtype=torch.long,
    )

    output_lens = torch.tensor(
        [len(x) for x in output_tokens],
        dtype=torch.long,
    )

    prompt_and_output_lens = prompt_lens + output_lens
    max_len = prompt_and_output_lens.max().item()

    pad_token_id = tokenizer.pad_token_id

    #  token size [B, max_len]
    token_ids = torch.full(
        (batch_size, max_len),
        fill_value=pad_token_id,
        dtype=torch.long,
    )

    response_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.bool,
    )

    # 3. 写入 prompt + output
    for i, (prompt, output) in enumerate(zip(prompt_tokens, output_tokens)):
        p_len = len(prompt)
        o_len = len(output)

        token_ids[i, :p_len] = torch.tensor(prompt)
        token_ids[i, p_len:p_len + o_len] = torch.tensor(output)

        # response 对应的位置
        response_mask[i, p_len:p_len + o_len] = True

    # 4. next-token prediction 对齐
    input_ids = token_ids[:, :-1]
    labels = token_ids[:, 1:]
    response_mask = response_mask[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }
    # raise NotImplementedError


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens)
    from a causal language model, and optionally the entropy of the model's
    next-token distribution.

    Args:
        model: PreTrainedModel
            HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor
            shape (batch_size, sequence_length), concatenated prompt + response
            tokens as produced by your tokenization method.
        labels: torch.Tensor
            shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool
            If True, also return per-token entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs"
                shape (batch_size, sequence_length), conditional
                log-probabilities log p_(theta)(x_t | x_(<t)).
            "token_entropy"
                optional, shape (batch_size, sequence_length), per-token
                entropy for each position (present only if
                return_token_entropy=True).
    """
    # 前向传播, 输出的Tensor 是 [Batch,Sequence,Token]
    pad_token_id = getattr(model.config, "pad_token_id", None)
    attention_mask = input_ids.ne(pad_token_id) if pad_token_id is not None else None
    outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits


    log_probs_all = torch.log_softmax(
        logits,
        dim=-1
    )

    # 提取每一个位置的目标Token ID
    log_probs = torch.gather(
        log_probs_all,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    result = {
        "log_probs": log_probs,
    }

    if return_token_entropy:
        probs = torch.softmax(logits, dim=-1)
        token_entropy = -torch.sum(
            probs * log_probs_all,
            dim=-1
        )
        result["token_entropy"] = token_entropy
    return result

    # raise NotImplementedError


def run_compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for a list of rollout responses, along with metadata for
    the reward components.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            raw_rewards
                shape (rollout_batch_size,). Unnormalized rewards for each
                rollout response.
            metadata
                Reward statistics to log. At minimum, include the mean total
                and format rewards over the rollout batch.
    """
    # 将生成的答案和标准答案长度对齐
    assert len(rollout_responses) == len(repeated_ground_truths)

    reward_dicts = [
        # 根据 reward_fn 的打分规则进行打分
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(
            rollout_responses,
            repeated_ground_truths,
        )
    ]
    # 获得每条回答的总奖励
    raw_rewards = torch.tensor(
        [r["reward"] for r in reward_dicts],
        dtype=torch.float32,
    )
    # 构建整体日志，包含
    metadata = {
        # 整体总奖励的均值
        "mean_reward": sum(r["reward"] for r in reward_dicts) / len(reward_dicts),
        # 回答形式的奖励均值
        "mean_format_reward": sum(
            r["format_reward"] for r in reward_dicts
        ) / len(reward_dicts),
        # 回答答案的奖励均值
        "mean_answer_reward": sum(
            r["answer_reward"] for r in reward_dicts
        ) / len(reward_dicts),
    }

    return raw_rewards, metadata


    # raise NotImplementedError


def run_compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            For this problem, support mean, which subtracts the per-group mean
            reward. Later, none will mean no baseline subtraction.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            For this problem, support std, which divides by the per-group
            standard deviation. Later, none will mean no normalization and
            mean will mean divide by the per-group mean reward.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """
    raw_rewards = raw_rewards.reshape(-1,group_size)

    # advantage = (r-mu)/(delta+eps)
    reward_mean = raw_rewards.mean(dim=1, keepdim=True) if baseline == "mean" else 0
    match advantage_normalizer:
        case "std":
            divisor = raw_rewards.std(dim=1, keepdim=True) + advantage_eps

        case "mean":
            divisor = raw_rewards.mean(dim=1, keepdim= True) + advantage_eps

        case "none":
            divisor = 1

        case _:
            raise ValueError("Error Input advantage_normalizer")


    advantage = ((raw_rewards - reward_mean)/divisor).reshape(-1)

    metadata = {
        "mean": raw_rewards.mean().item(),
        "std": raw_rewards.std().item(),
        "max": raw_rewards.max().item(),
        "min": raw_rewards.min().item()
    }

    return advantage, metadata

    # raise NotImplementedError


def run_compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.
    """
    advantages = raw_rewards_or_advantages.reshape(-1,1)

    match importance_reweighting_method:

        case "none":
            l =  - advantages * policy_log_probs

        case "noclip":
            assert old_log_probs is not None
            l = - torch.exp(policy_log_probs - old_log_probs) * advantages

        case "grpo":
            # grpo token 裁剪计算: L= - min (ratio,ratio_clip*advantage), 即裁剪前后的对数差的min
            assert old_log_probs is not None
            assert cliprange is not None
            ratio = torch.exp(policy_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(
                ratio,
                min = 1 - cliprange,
                max = 1 + cliprange
            )
            normal_object = torch.exp(policy_log_probs - old_log_probs) * advantages
            clipped_object = clipped_ratio * advantages
            l = -torch.minimum(normal_object,clipped_object)

        case "gspo":
            # gspo 根据token response_mask 对有效概率差加权，并使用和grpo 相似的裁剪
            assert old_log_probs is not None
            assert cliprange is not None
            assert response_mask is not None

            # 每条回答分别计算，形状 [B, 1]
            log_ratio = policy_log_probs - old_log_probs
            response_lengths = response_mask.sum(dim=1, keepdim=True)
            assert torch.all(response_lengths > 0)

            mean_log_ratio = (
                (response_mask * log_ratio).sum(dim=1, keepdim=True)
                / response_lengths
            )
            ratio = torch.exp(mean_log_ratio)

            clipped_ratio = torch.clamp(
                ratio,
                min=1 - cliprange,
                max=1 + cliprange,
            )

            sequence_loss = -torch.minimum(
                ratio * advantages,
                clipped_ratio * advantages,
            )  # [B, 1]

            l = sequence_loss.expand_as(policy_log_probs)  # [B, T]
        case _:
            raise NameError("Error Input importance_reweighting_method")
    # metadata = {
    #     "clip_fraction": (
    #             clipped_object < normal_object
    #     ).float().mean().detach(),
    # }
    metadata = {}
    return l,metadata


def run_aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate the per-token policy-gradient loss according to the response
    mask and loss-normalization strategy.

    Args:
        per_token_policy_gradient_loss: torch.Tensor
            Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in
            the training loop).
        mask
            torch.Tensor of shape (batch_size, sequence_length) denoting which
            positions should be included in the loss.
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant.
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        loss: torch.Tensor
            A scalar containing the average loss. Make sure you can later call
            backward on this loss.
    """
    loss_masked = per_token_policy_gradient_loss * mask
    match loss_normalization:
        case "sequence":
            # 每个 batch 中的有效回答token的loss 的平均
            seq_loss = (
                loss_masked.sum(dim = 1) / mask.sum(dim = 1)
            )
            # 对 batch 间的平均loss 求平均
            loss = seq_loss.mean()
        case "constant":
            assert normalization_constant is not None
            assert normalization_constant > 0

            # 将每一个batch 的每一个seq tokens混合，求和后除以输入的常数
            loss = loss_masked.sum() / normalization_constant
    return loss
    # raise NotImplementedError


def run_grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute forward-and-backward passes, with gradient_accumulation_steps
    microbatches.

    Args:
        model: PreTrainedModel
            HuggingFace model to train.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.
        optimizer: Optimizer
            Optimizer for the model.
        gradient_accumulation_steps: int
            Number of microbatches per optimizer step.
        max_grad_norm: float | None
            If not None, clip the gradient norm to this value before calling
            optimizer.step().
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        repeated_prompts: list[str]
            The prompts for the examples. The length of this list is
            rollout_batch_size, because the prompt for each example is repeated
            group_size times.
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            If mean, subtract the per-group mean reward; if none, do nothing.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            If std, divide by the per-group standard deviation; if none, do
            nothing; if mean, divide by the per-group mean reward.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style token-level
            reweighting and clipping; "gspo": do GSPO-style sequence-level
            reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant (fixed
            for all of training).
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss
                scalar tensor. The batch loss, adjusted for gradient
                accumulation. We return this so we can log it.
            metadata
                Dict with metadata from the underlying loss call, gradient norm
                before clipping, and any other statistics you might want to log.
    """
    metadata = {}
    # 1. 对整个 rollout batch 打分，得到每条回答的原始奖励
    raw_rewards, metadata_step1 = run_compute_rollout_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths
    )

    # 2. 按题目分组计算 advantages，在划分 microbatch 前完成组内比较
    advantages , metadata_step2 = run_compute_group_normalized_rewards(
        raw_rewards=raw_rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer
    )
    # advantages 作为固定训练信号，放到模型所在设备
    device = next(model.parameters()).device
    advantages = advantages.detach().to(device=device)


    # 3. 对 prompt 和回答 Tokenize，构造输入、标签和回答 mask
    tokenized = run_tokenize_prompt_and_output(
        prompt_strs=repeated_prompts,
        output_strs=rollout_responses,
        tokenizer=tokenizer
    )
    # 将训练张量移到模型所在设备，旧策略 log_probs 不参与反向传播
    input_ids = tokenized["input_ids"].to(device=device)
    labels = tokenized["labels"].to(device=device)
    response_mask = tokenized["response_mask"].to(device=device)
    if old_log_probs is not None:
        old_log_probs = old_log_probs.detach().to(device=device)
    # 4. 根据梯度累积步数，将 batch 划分为等大小的 microbatch
    batch_size = input_ids.shape[0]

    assert gradient_accumulation_steps > 0
    assert batch_size > 0
    assert batch_size % gradient_accumulation_steps == 0

    microbatch_size = batch_size // gradient_accumulation_steps

    # 5. 清空旧梯度，初始化累计 loss，并融合奖励和优势的 metadata
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)
    response_tokens = response_mask.sum()
    if torch.any(response_mask.sum(dim=1) == 0):
        raise ValueError("每条回答必须至少包含一个有效 token")
    metadata |= metadata_step1
    metadata |= metadata_step2

    # 6. 对各个 microbatch 分别前向和反向传播，累积梯度
    for start in range(0,batch_size,microbatch_size):
        end = start + microbatch_size
        # 6.1 切片当前 microbatch，计算每个目标 token 的 log_probs
        result = run_get_response_log_probs(
            model=model,
            input_ids=input_ids[start:end],
            labels=labels[start:end],
            return_token_entropy=True,
        )
        entropy_sum += (
            result["token_entropy"].detach() * response_mask[start:end]
        ).sum()
        # 6.2 同步切片 advantages、旧策略 log_probs 和 mask，计算逐 token 损失
        per_token_loss, metadata_step3 = run_compute_policy_gradient_loss(
            raw_rewards_or_advantages=advantages[start:end],
            policy_log_probs=result["log_probs"],
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=(
                old_log_probs[start:end] if old_log_probs is not None else None
            ),
            cliprange=cliprange,
            response_mask=response_mask[start:end],
        )
        # 6.3 用回答 mask 筛选有效 token，将损失聚合为标量
        loss = run_aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_loss,
            mask=response_mask[start:end],
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )

        # sequence 模式按累积步数缩放；constant 模式已使用完整 batch 的固定分母
        if loss_normalization == "sequence":
            loss = loss / gradient_accumulation_steps

        # 6.4 累积梯度，循环内不清空梯度，也不更新参数
        loss.backward()
        # 6.5 累加用于日志的 loss，取各 microbatch 统计量的平均值
        total_loss = total_loss + loss.detach()
        for key, value in metadata_step3.items():
            metadata[key] = metadata.get(key, 0) + value.detach() / gradient_accumulation_steps

    metadata["token_entropy"] = entropy_sum / response_tokens
    metadata["response_tokens"] = response_tokens.detach()
    # 7. 所有梯度累积完成后统一裁剪，并记录裁剪前的梯度范数
    if max_grad_norm is not None:
        metadata["grad_norm"] = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm
        ).detach()

    # 8. 更新一次模型参数，清空梯度，为下一次训练步做准备
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    # 9. 返回完整 batch 的累计 loss 和 metadata
    return total_loss, metadata

    # raise NotImplementedError

