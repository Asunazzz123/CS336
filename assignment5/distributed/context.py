"""One NVIDIA CUDA worker per torchrun rank; owns only groups it creates."""
from dataclasses import dataclass
from datetime import timedelta
import os
from typing import Callable, TypeVar

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from utils.device import nvidia_cuda_available, resolve_device

T = TypeVar("T")


@dataclass
class NvidiaContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    mesh: object
    owns_group: bool
    closed: bool = False

    @classmethod
    def from_torchrun(cls, timeout_seconds: int = 180):
        if not nvidia_cuda_available():
            raise RuntimeError("distributed requires NVIDIA CUDA; there is no CPU/MPS fallback")
        try:
            rank, size, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
        except (KeyError, ValueError) as error:
            raise RuntimeError("Launch with torchrun and RANK/WORLD_SIZE/LOCAL_RANK") from error
        if size < 2 or not 0 <= rank < size or local < 0:
            raise ValueError("Require >=2 ranks and a valid visible local CUDA ordinal")
        device = resolve_device(f"cuda:{local}")
        torch.cuda.set_device(local)
        owned = not dist.is_initialized()
        try:
            if owned:
                dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_seconds))
            if dist.get_backend() != "nccl" or (dist.get_rank(), dist.get_world_size()) != (rank, size):
                raise RuntimeError("Existing process group must match the torchrun NCCL world")
            mesh = init_device_mesh("cuda", (size,), mesh_dim_names=("fsdp",))
        except BaseException:
            if owned and dist.is_initialized():
                dist.destroy_process_group()
            raise
        return cls(rank, size, local, device, mesh, owned)

    def sum(self, tensor: torch.Tensor):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def all_true(self, value: bool):
        flag = torch.tensor(int(value), device=self.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def agree(self, name: str, value):
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        if any(other != values[0] for other in values):
            raise RuntimeError(f"Ranks disagree on {name}: {values}")

    def gather_objects(self, value):
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def root_call(self, fn: Callable[[], T]) -> T:
        """Propagate rank-0 I/O/sink errors before the next collective."""
        result = [None]
        if self.rank == 0:
            try:
                result[0] = (True, fn())
            except Exception as error:
                result[0] = (False, f"{type(error).__name__}: {error}")
        dist.broadcast_object_list(result, src=0)
        ok, value = result[0]
        if not ok:
            raise RuntimeError(f"Rank-0 operation failed: {value}")
        return value

    def barrier(self):
        dist.barrier(device_ids=[self.local_rank])

    def close(self):
        if not self.closed:
            # No barrier on error: a failed peer must not hang cleanup forever.
            if self.owns_group and dist.is_initialized():
                dist.destroy_process_group()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def local_call(ctx, fn):
    """Surface tokenization/reward errors on all peers before the next forward."""
    result, error = None, None
    try:
        result = fn()
    except Exception as exc:
        error = exc
    if not ctx.all_true(error is None):
        raise RuntimeError(f"Rank-local preparation failed; local error: {error}") from error
    return result
