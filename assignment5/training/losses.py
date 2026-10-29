"""Backend-neutral differentiable SFT response-token loss."""
import torch
import torch.nn.functional as F
from profiling.nvtx import nvtx_range
from training.reproducibility import autocast_context


def sft_loss_sum(model, batch: dict[str, torch.Tensor], autocast: str = "none") -> tuple[torch.Tensor, int]:
    device = next(model.parameters()).device
    with nvtx_range("sft.batch_to_device", device):
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
    targets = batch["labels"][:, 1:]
    supervised = targets.ne(-100)
    token_count = int(supervised.sum().item())
    if token_count == 0:
        raise ValueError("An SFT batch has no supervised next-token targets")
    with nvtx_range("sft.forward", device), autocast_context(model, autocast):
        logits = model(batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
    # Select response positions before the FP32 CE, keeping prompt/padding out.
    with nvtx_range("sft.loss", device):
        selected_logits = logits[:, :-1, :][supervised]
        loss = F.cross_entropy(selected_logits.float(), targets[supervised], reduction="sum")
    return loss, token_count
