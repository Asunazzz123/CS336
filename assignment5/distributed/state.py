"""State transitions and explicit, synchronous optimizer-state residency."""
from enum import Enum
import torch


class Phase(str, Enum):
    READY = "ready"
    TRAINING = "training"
    ROLLOUT = "rollout"
    CHECKPOINT = "checkpoint"
    PARKED = "parked"
    FAILED = "failed"
    CLOSED = "closed"


def move_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, dict):
        return {k: move_tree(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_tree(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_tree(v, device) for v in value)
    return value


@torch.no_grad()
def move_optimizer_state(optimizer: torch.optim.Optimizer, device):
    """Preserve parameter keys/identities and leave ordinary AdamW step on CPU.

    This moves inactive state; optimizer.step() is always performed on CUDA.
    FSDP2 moments are DTensors and retain their sharding through Tensor.to().
    """
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            for key, value in optimizer.state.get(parameter, {}).items():
                if key == "step" and not group.get("capturable", False) and not group.get("fused", False):
                    target = "cpu"
                else:
                    target = device
                optimizer.state[parameter][key] = move_tree(value, target)
