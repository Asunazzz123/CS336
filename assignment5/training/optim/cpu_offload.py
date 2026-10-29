"""Explicit CPU Adam/AdamW placement, independent of trainers."""
import torch
from profiling.nvtx import nvtx_range


class CPUOffloadOptimizer:
    """Adam/AdamW with FP32 CPU masters and moments, and live model parameters.

    Gradients are clipped/accumulated on the model device by the caller. Each
    step copies those gradients to CPU, updates the masters, then copies the
    updated parameters back. Copies are synchronous so the next forward sees
    the completed update. This saves GPU optimizer memory at a transfer cost.
    """

    def __init__(self, params, optimizer_name: str, **kwargs):
        if optimizer_name not in ("adam", "adamw"):
            raise ValueError("CPU optimizer offload only supports adam/adamw")
        self.model_parameters = list({id(p): p for p in params}.values())
        self.master_parameters = [
            torch.nn.Parameter(p.detach().to(device="cpu", dtype=torch.float32, copy=True),
                               requires_grad=True)
            for p in self.model_parameters
        ]
        self.optimizer_name = optimizer_name
        optimizer_class = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
        # A parameter-sized foreach temporary is expensive for a large model.
        self.optimizer = optimizer_class(self.master_parameters, foreach=False, **kwargs)

    @property
    def param_groups(self):
        # Return the actual groups, including after load_state_dict replaces them.
        return self.optimizer.param_groups

    @torch.no_grad()
    def step(self):
        device = self.model_parameters[0].device
        with nvtx_range("optimizer.offload.grad_to_cpu", device):
            for parameter, master in zip(self.model_parameters, self.master_parameters):
                master.grad = (
                    None if parameter.grad is None else
                    parameter.grad.detach().to(device="cpu", dtype=torch.float32, copy=True)
                )
        with nvtx_range("optimizer.offload.cpu_adamw", device):
            result = self.optimizer.step()
        with nvtx_range("optimizer.offload.param_to_gpu", device):
            for parameter, master in zip(self.model_parameters, self.master_parameters):
                parameter.copy_(master)
        return result

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for parameter in self.model_parameters:
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    if parameter.grad.grad_fn is not None:
                        parameter.grad.detach_()
                    else:
                        parameter.grad.requires_grad_(False)
                    parameter.grad.zero_()

    def state_dict(self):
        # Masters are needed for exact FP32 optimizer continuation: model export
        # may be BF16 even when live training parameters are FP32.
        return {
            "optimizer_name": self.optimizer_name,
            "master_parameters": [p.detach() for p in self.master_parameters],
            "optimizer": self.optimizer.state_dict(),
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        if state.get("optimizer_name") != self.optimizer_name:
            raise ValueError("CPU optimizer type does not match checkpoint")
        saved_parameters = state["master_parameters"]
        if len(saved_parameters) != len(self.master_parameters):
            raise ValueError("CPU master parameter count does not match checkpoint")
        for master, saved in zip(self.master_parameters, saved_parameters):
            if master.shape != saved.shape:
                raise ValueError("CPU master parameter shape does not match checkpoint")
        self.optimizer.load_state_dict(state["optimizer"])
        for parameter, master, saved in zip(
            self.model_parameters, self.master_parameters, saved_parameters
        ):
            master.copy_(saved)
            parameter.copy_(master)
        self.zero_grad(set_to_none=True)
