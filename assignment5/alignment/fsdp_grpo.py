"""GRPO adapter: algorithms live here; FSDP collectives/AdamW live in engine."""
from dataclasses import dataclass

import torch

from distributed import LossOutput, Phase
from .grpo import run_compute_group_normalized_rewards, run_compute_policy_gradient_loss


@dataclass
class GRPOBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    response_mask: torch.Tensor
    advantages: torch.Tensor

    def slice(self, start, end):
        return GRPOBatch(*(getattr(self, name)[start:end] for name in self.__dataclass_fields__))


def batch_from_rollout(prompt_ids, prompt_mask, rollout, rewards, *, group_size,
                       baseline="mean", advantage_normalizer="std"):
    """Keep sampled IDs, including actual EOS; never decode/re-tokenize policy data.

    Entire groups must reside on one rank before computing advantages.
    Left padding uses explicit attention/position masks during policy scoring.
    """
    if group_size < 2 or len(rewards) != len(prompt_ids) or len(rewards) % group_size:
        raise ValueError("Each rank must own complete reward groups of size >=2")
    advantages, _ = run_compute_group_normalized_rewards(
        torch.as_tensor(rewards, dtype=torch.float32), group_size,
        baseline=baseline, advantage_normalizer=advantage_normalizer)
    device = prompt_ids.device
    ids = torch.cat((prompt_ids, rollout.response_ids.to(device)), dim=1)
    attention = torch.cat((prompt_mask, rollout.response_mask.to(device)), dim=1).long()
    response = torch.cat((torch.zeros_like(prompt_mask, dtype=torch.bool),
                          rollout.response_mask.to(device)), dim=1)
    return GRPOBatch(ids[:, :-1], attention[:, :-1], ids[:, 1:], response[:, 1:],
                     advantages.detach().to(device))


def policy_log_probs(model, batch):
    position_ids = (batch.attention_mask.cumsum(-1) - 1).clamp_min(0)
    logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                   position_ids=position_ids, use_cache=False).logits.float()
    return logits.gather(-1, batch.labels.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)


class FSDPGRPOTrainer:
    def __init__(self, engine, *, num_epochs_per_rollout=1, gradient_accumulation_steps=1,
                 importance_reweighting_method="grpo", cliprange=0.2,
                 loss_normalization="sequence"):
        if num_epochs_per_rollout < 1 or gradient_accumulation_steps < 1:
            raise ValueError("Epoch and accumulation counts must be positive")
        if importance_reweighting_method not in ("none", "noclip", "grpo", "gspo"):
            raise ValueError("Unknown importance reweighting method")
        if loss_normalization not in ("sequence", "constant"):
            raise ValueError("Unknown loss normalization")
        if not 0 < cliprange < float("inf"):
            raise ValueError("cliprange must be finite and positive")
        self.engine = engine
        self.epochs, self.accumulation = num_epochs_per_rollout, gradient_accumulation_steps
        self.method, self.cliprange = importance_reweighting_method, cliprange
        self.normalization = loss_normalization
        # Checkpoint recomputation needs train mode. Disable module dropout so
        # old and new policies describe the same stochastic policy (token sampling).
        for module in engine.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

    def update(self, batch, *, policy_version):
        engine, ctx = self.engine, self.engine.context
        engine._require(Phase.READY)
        ctx.agree("GRPO update", (self.epochs, self.accumulation, self.method,
                                  self.cliprange, self.normalization, policy_version))
        valid = (policy_version == engine.policy_version and len(batch.input_ids) > 0
                 and len(batch.input_ids) % self.accumulation == 0
                 and bool((batch.response_mask.sum(-1) > 0).all())
                 and bool(torch.isfinite(batch.advantages).all()))
        if not ctx.all_true(valid):
            raise ValueError("Stale policy, incomplete microbatches, empty responses or invalid advantages")
        size = len(batch.input_ids) // self.accumulation
        batches = [batch.slice(start, start + size) for start in range(0, len(batch.input_ids), size)]
        # Single epoch is on-policy: no stale-ratio pass. Multi epoch fixes the
        # rollout policy once, before ANY optimizer step, for all later epochs.
        method = self.method if self.epochs > 1 else "none"
        old = [None] * len(batches)
        previous_mode = engine.model.training
        try:
            if method != "none":
                engine.model.eval()
                with torch.no_grad():
                    old = [policy_log_probs(engine.model, b).detach() for b in batches]

            def loss(model, item):
                microbatch, reference = item
                model.train(engine.config.gradient_checkpointing)
                log_probs = policy_log_probs(model, microbatch)
                per_token, _ = run_compute_policy_gradient_loss(
                    microbatch.advantages, log_probs, method, reference,
                    self.cliprange, microbatch.response_mask)
                masked = per_token * microbatch.response_mask
                if self.normalization == "sequence":
                    return LossOutput((masked.sum(-1) / microbatch.response_mask.sum(-1)).sum(),
                                      len(microbatch.input_ids))
                return LossOutput(masked.sum(), int(microbatch.response_mask.sum().item()))

            return [engine.train_step(list(zip(batches, old)), loss) for _ in range(self.epochs)]
        finally:
            engine.model.train(previous_mode)
