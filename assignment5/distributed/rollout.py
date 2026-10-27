"""Correctness-first DP/FSDP decoding; every rank executes matching forwards."""
import torch

from .engine import RolloutResult


@torch.no_grad()
def generate(engine, input_ids, attention_mask, config):
    context = engine.context
    valid = (input_ids.ndim == attention_mask.ndim == 2
             and input_ids.shape == attention_mask.shape and input_ids.numel() > 0
             and input_ids.dtype == torch.long)
    if not context.all_true(valid):
        raise ValueError("Each rank needs a nonempty [batch, length] long input_ids and matching mask")
    ids = input_ids.to(context.device)
    mask = attention_mask.to(context.device)
    limit = getattr(getattr(engine.model, "config", None), "max_position_embeddings", None)
    valid = (bool(((mask == 0) | (mask == 1)).all().item())
             and bool((mask[:, -1] == 1).all().item())
             and bool((mask[:, 1:] >= mask[:, :-1]).all().item())
             and bool((ids >= 0).all().item())
             and (limit is None or ids.shape[1] + config.max_new_tokens <= limit))
    if not context.all_true(valid):
        raise ValueError("Use left padding, binary masks, valid token IDs and a supported context length")
    mask = mask.long()
    finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=context.device)
    responses, response_masks = [], []
    rng = torch.Generator(device=context.device).manual_seed(
        engine.config.seed + context.rank + 1009 * engine.policy_version + 104729 * engine.rollout_id)
    for _ in range(config.max_new_tokens):
        # Cache deliberately disabled: generic FSDP hooks see the SAME number
        # of forwards on every rank, including ranks whose sequences ended.
        position_ids = (mask.cumsum(dim=-1) - 1).clamp_min(0)
        output = engine.model(input_ids=ids, attention_mask=mask,
                              position_ids=position_ids, use_cache=False)
        logits = output.logits[:, -1, :].float()
        valid = (bool(torch.isfinite(logits).all().item())
                 and config.pad_token_id < logits.shape[-1]
                 and (config.eos_token_id is None or config.eos_token_id < logits.shape[-1]))
        if not context.all_true(valid):
            raise ValueError("Invalid generation logits or special token IDs")
        if config.temperature == 0:
            sampled = logits.argmax(dim=-1)
        else:
            probabilities = torch.softmax(logits / config.temperature, dim=-1)
            if not context.all_true(bool(torch.isfinite(probabilities).all().item())):
                raise ValueError("Non-finite sampling probabilities; check temperature")
            sampled = torch.multinomial(probabilities, 1, generator=rng).squeeze(-1)
        active = ~finished
        sampled = torch.where(active, sampled, config.pad_token_id)
        responses.append(sampled)
        response_masks.append(active)
        ids = torch.cat((ids, sampled[:, None]), dim=1)
        mask = torch.cat((mask, active[:, None].long()), dim=1)
        if config.eos_token_id is not None:
            finished |= active & sampled.eq(config.eos_token_id)
        # Never exit on local EOS: that would deadlock another rank's all-gathers.
        if context.all_true(bool(finished.all().item())):
            break
    return RolloutResult(torch.stack(responses, dim=1), torch.stack(response_masks, dim=1),
                         engine.policy_version, engine.rollout_id)
