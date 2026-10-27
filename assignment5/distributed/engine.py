"""Standalone NVIDIA FSDP2 engine: no dataset, reward, trainer or launcher imports."""
from dataclasses import dataclass
from typing import Callable, Iterable, Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy, OffloadPolicy
from torch.distributed.tensor import DTensor

from .config import FSDPConfig, GenerationConfig
from .context import NvidiaContext
from .state import Phase, move_optimizer_state
from utils.device import nvidia_cuda_available, cuda_supports_bf16


@dataclass
class LossOutput:
    loss_sum: torch.Tensor
    weight: int  # Effective tokens/examples for this local microbatch, not a mean.


@dataclass
class StepResult:
    loss: float
    global_weight: int
    grad_norm: float
    policy_version: int


@dataclass
class RolloutResult:
    response_ids: torch.Tensor
    response_mask: torch.Tensor  # Includes actual EOS, excludes filler after EOS.
    policy_version: int
    rollout_id: int


def validate_shared_parameters(model, block_names):
    """Tied root embedding/head are supported; cross-unit ties are not."""
    owners = {}
    names = sorted(block_names, key=len, reverse=True)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        owner = next((block for block in names if name.startswith(block + ".")), "<root>")
        owners.setdefault(id(parameter), set()).add(owner)
    if any(len(units) > 1 for units in owners.values()):
        raise ValueError("A tied parameter crosses FSDP block boundaries; keep its aliases in one unit")


@torch.no_grad()
def normalize_and_clip(parameters, context, global_weight: int, max_norm: float | None):
    """FSDP averaged gradients -> global token mean; norm over unique shards."""
    if global_weight <= 0:
        raise ValueError("The global loss weight must be positive")
    grads = [p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
             for p in parameters if p.grad is not None]
    squared = torch.zeros((), dtype=torch.float64, device=context.device)
    for grad in grads:
        grad.mul_(context.world_size / global_weight)
        squared += torch.linalg.vector_norm(grad, dtype=torch.float32).double().square()
    context.sum(squared)
    norm = squared.sqrt()
    if not torch.isfinite(norm).item():
        raise FloatingPointError("Non-finite global gradient norm; update rejected")
    if max_norm is not None:
        scale = (max_norm / (norm + 1e-6)).clamp(max=1).float()
        for grad in grads:
            grad.mul_(scale)
    return float(norm.item())


class NvidiaFSDPEngine:
    """All public collective operations must be called by ALL ranks in order.

    Factory returns a materialized model (typically CPU); one unsharded model
    must fit each GPU during initialization. Full training state is sharded.
    CUDA AdamW runs on local FP32 shards, never on CPU.
    """
    def __init__(self, context: NvidiaContext, model_factory: Callable[[], torch.nn.Module],
                 config: FSDPConfig | None = None):
        if context.closed or context.device.type != "cuda" or not nvidia_cuda_available():
            raise RuntimeError("Engine requires an open NVIDIA CUDA context")
        self.context = context
        self.config = config or FSDPConfig()
        self.phase = Phase.READY
        self.policy_version = 0
        self.rollout_id = 0
        self.optimizer_on_gpu = True
        context.agree("engine config", repr(self.config))
        if self.config.compute_dtype == torch.bfloat16 and not cuda_supports_bf16(context.device):
            raise RuntimeError("This CUDA device does not support BF16; use compute_dtype=float32")
        torch.manual_seed(self.config.seed)
        model = model_factory()
        if any(p.is_meta for p in model.parameters()):
            raise ValueError("Factory must materialize parameters; meta init is not supported yet")
        if not any(p.requires_grad for p in model.parameters()):
            raise ValueError("Model has no trainable parameters")
        names = self.config.block_class_names or tuple(getattr(model, "_no_split_modules", ()) or ())
        blocks = [(name, module) for name, module in model.named_modules()
                  if name and type(module).__name__ in names]
        if not blocks:
            raise ValueError("Specify block_class_names or a model with matching _no_split_modules")
        validate_shared_parameters(model, [name for name, _ in blocks])
        context.agree("model structure", [(name, tuple(p.shape), str(p.dtype))
                                          for name, p in list(model.named_parameters()) + list(model.named_buffers())])
        context.agree("FSDP block boundaries", [name for name, _ in blocks])
        if self.config.gradient_checkpointing:
            if not hasattr(model, "gradient_checkpointing_enable"):
                raise ValueError("This model does not expose gradient_checkpointing_enable")
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = model.to(device=context.device, dtype=self.config.master_dtype)
        with torch.no_grad():
            for tensor in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(tensor, src=0)
        kwargs = dict(mesh=context.mesh, reshard_after_forward=True,
                      mp_policy=MixedPrecisionPolicy(param_dtype=self.config.compute_dtype,
                                                     reduce_dtype=self.config.reduce_dtype),
                      offload_policy=OffloadPolicy())
        # Bottom-up units, then remaining root parameters (including tied embeddings).
        for _, block in sorted(blocks, key=lambda item: item[0].count("."), reverse=True):
            fully_shard(block, **kwargs)
        fully_shard(model, **kwargs)
        self.model = model
        self.optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=self.config.learning_rate,
            betas=self.config.betas, eps=self.config.eps, weight_decay=self.config.weight_decay,
            foreach=False, fused=False,
        )

    def _require(self, *allowed):
        if self.context.closed:
            raise RuntimeError("Distributed context is closed")
        self.context.agree("engine state", (self.phase.value, self.optimizer_on_gpu,
                                            self.policy_version, self.rollout_id))
        if self.phase not in allowed:
            raise RuntimeError(f"Operation requires {allowed}; engine is {self.phase.value}")

    def _reshard(self):
        for module in self.model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()

    def train_step(self, microbatches: Iterable[Any],
                   loss_fn: Callable[[torch.nn.Module, Any], LossOutput]) -> StepResult:
        """loss_fn performs the forward and returns an unnormalized scalar loss.

        Each rank must execute the same positive number of microbatches, with
        the same model control flow. Gradients stay synchronized AND sharded
        on every backward: no no_sync() full-gradient accumulation spike.
        """
        self._require(Phase.READY)
        self.context.agree("training cursor", self.policy_version)
        self.context.agree("optimizer residency", self.optimizer_on_gpu)
        if not self.optimizer_on_gpu:
            raise RuntimeError("Load optimizer state before a training step")
        batches = list(microbatches)
        self.context.agree("microbatch count", len(batches))
        if not batches:
            raise ValueError("At least one microbatch per rank is required")
        self.phase = Phase.TRAINING
        try:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(2, dtype=torch.float64, device=self.context.device)
            for batch in batches:
                output = loss_fn(self.model, batch)
                valid = (isinstance(output, LossOutput) and isinstance(output.loss_sum, torch.Tensor)
                         and isinstance(output.weight, int)
                         and output.weight >= 0 and output.loss_sum.numel() == 1
                         and output.loss_sum.device == self.context.device
                         and output.loss_sum.requires_grad
                         and bool(torch.isfinite(output.loss_sum.detach()).item()))
                if not self.context.all_true(valid):
                    raise ValueError("Each callback must return a finite differentiable scalar LossOutput")
                output.loss_sum.backward()
                totals[0] += output.loss_sum.detach().double().reshape(())
                totals[1] += output.weight
            self.context.sum(totals)
            weight = int(totals[1].item())
            norm = normalize_and_clip(self.model.parameters(), self.context, weight,
                                      self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.policy_version += 1
            self.phase = Phase.READY
            return StepResult(float(totals[0].item()) / weight, weight, norm, self.policy_version)
        except BaseException:
            self.phase = Phase.FAILED
            self.optimizer.zero_grad(set_to_none=True)
            raise

    def offload_optimizer(self):
        self._require(Phase.READY)
        try:
            move_optimizer_state(self.optimizer, "cpu")
            self.optimizer_on_gpu = False
        except BaseException:
            self.phase = Phase.FAILED
            raise

    def load_optimizer(self):
        self._require(Phase.READY)
        try:
            move_optimizer_state(self.optimizer, self.context.device)
            self.optimizer_on_gpu = True
        except BaseException:
            self.phase = Phase.FAILED
            raise

    def park(self):
        """Release this rank's CUDA model AND optimizer shards at a phase boundary."""
        self._require(Phase.READY)
        if not self.context.all_true(all(p.grad is None for p in self.model.parameters())):
            raise RuntimeError("Cannot park with live gradients")
        try:
            self.offload_optimizer()
            self._reshard()
            identities = [id(p) for p in self.model.parameters()]
            self.model.to("cpu", non_blocking=False)
            if identities != [id(p) for p in self.model.parameters()]:
                raise RuntimeError("This PyTorch conversion replaced optimizer parameter identities")
            torch.cuda.synchronize(self.context.device)
            torch.cuda.empty_cache()
            self.phase = Phase.PARKED
        except BaseException:
            self.phase = Phase.FAILED
            raise

    def resume(self):
        self._require(Phase.PARKED)
        try:
            identities = [id(p) for p in self.model.parameters()]
            self.model.to(self.context.device, non_blocking=False)
            if identities != [id(p) for p in self.model.parameters()]:
                raise RuntimeError("Resume replaced optimizer parameter identities")
            move_optimizer_state(self.optimizer, self.context.device)
            self.optimizer_on_gpu = True
            self.phase = Phase.READY
        except BaseException:
            self.phase = Phase.FAILED
            raise

    @torch.no_grad()
    def rollout(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                config: GenerationConfig | None = None) -> RolloutResult:
        from .rollout import generate
        config = config or GenerationConfig()
        self._require(Phase.READY)
        self.context.agree("generation config", repr(config))
        self.context.agree("rollout cursor", (self.policy_version, self.rollout_id))
        was_training = self.model.training
        restore_optimizer = self.optimizer_on_gpu and config.offload_optimizer
        if restore_optimizer:
            self.offload_optimizer()
        self.phase = Phase.ROLLOUT
        try:
            self.model.eval()
            result = generate(self, input_ids, attention_mask, config)
            self.rollout_id += 1
            self._reshard()
            self.model.train(was_training)
            if restore_optimizer:
                move_optimizer_state(self.optimizer, self.context.device)
                self.optimizer_on_gpu = True
            self.phase = Phase.READY
            return result
        except BaseException:
            self.phase = Phase.FAILED
            raise

    @torch.no_grad()
    def export_weights(self, sink, dtype=torch.bfloat16):
        """ALL ranks gather each tensor; only rank 0 writes, with error broadcast.

        This is a DP-shards -> canonical tensor stream, not a vLLM TP loader.
        An external receiver must implement its own layout/name mapping and
        cache invalidation before acknowledging commit(version).
        """
        self._require(Phase.READY)
        self.context.agree("weight export", (self.policy_version, str(dtype)))
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Export dtype must be float32 or bfloat16")
        try:
            self._reshard()
            state = self.model.state_dict()
            if not self.context.all_true(all(isinstance(value, torch.Tensor) for value in state.values())):
                raise TypeError("Only tensor model state can be streamed")
            self.context.agree("weight schema", [(name, tuple(value.shape), str(value.dtype))
                                                  for name, value in state.items()])
            self.context.root_call(lambda: sink.begin(self.policy_version))
            for name, tensor in state.items():
                full = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
                self.context.root_call(lambda: sink.write(
                    name, full.detach().to(device="cpu", dtype=dtype if full.is_floating_point() else full.dtype,
                                           non_blocking=False)))
                del full
            acknowledged = self.context.root_call(lambda: sink.commit(self.policy_version))
            if acknowledged != self.policy_version:
                raise RuntimeError("Receiver acknowledged a different policy version")
        except BaseException:
            self.phase = Phase.FAILED
            try:
                self.context.root_call(lambda: sink.abort(self.policy_version))
            except Exception:
                pass  # Preserve the transfer failure, never allow generation to continue.
            raise

    def save_checkpoint(self, path, *, include_optimizer=False, metadata=None):
        from .checkpoint import save
        self._require(Phase.READY)
        return save(self, path, include_optimizer=include_optimizer, metadata=metadata)

    def export_model_only_hf(self, path, **kwargs):
        """ALL ranks participate; rank 0 publishes a model-only HF checkpoint."""
        from .weights import export_model_only_hf
        return export_model_only_hf(self, path, **kwargs)

    def load_checkpoint(self, path, *, restore_optimizer=False):
        from .checkpoint import load
        self._require(Phase.READY)
        return load(self, path, restore_optimizer=restore_optimizer)

    def close(self):
        """Drop engine-owned tensors; context lifetime remains the caller's responsibility."""
        if self.phase != Phase.CLOSED:
            self.optimizer.zero_grad(set_to_none=True)
            del self.optimizer
            del self.model
            torch.cuda.empty_cache()
            self.phase = Phase.CLOSED


def local_hf_factory(path: str, **kwargs):
    """Optional Transformers factory. No tokenizer/data loading and no downloads."""
    def build():
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32,
                                                   local_files_only=True, **kwargs)
    return build
