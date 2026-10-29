"""Token-weighted SFT NLL evaluation without optimizer updates."""
import math
import torch
from alignment.sft import collate_sft_batch
from training.losses import sft_loss_sum as _loss_sum


def evaluate_sft(model, records, pad_token_id, batch_size, autocast: str = "none"):
    """Token-weighted held-out teacher-forcing NLL; restore the caller's mode."""
    if not records or batch_size <= 0:
        raise ValueError("Validation records and a positive batch_size are required")
    previous_mode = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    try:
        with torch.inference_mode():
            for start in range(0, len(records), batch_size):
                batch = collate_sft_batch(records[start:start + batch_size], pad_token_id)
                loss, tokens = _loss_sum(model, batch, autocast)
                total_loss += loss.item()
                total_tokens += tokens
    finally:
        model.train(previous_mode)
    mean_loss = total_loss / total_tokens
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss) if mean_loss < 700 else None,
            "supervised_tokens": total_tokens, "num_examples": len(records)}
