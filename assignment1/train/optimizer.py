import torch
from torch.optim import Optimizer, AdamW

class TinyAdamW(Optimizer):
    """
    Tiny Realization of AdamW Optimizer Built with `torch.optim.Optimizer`

    Args:
        params: (torch.tensor) Specifies what Tensors should be optimized.
        lr: (float) Learning rate, control the speed of gradient decent. Defaults to 1e-3
        betas: (tuple[float, float]) Exponential decay rates for the first and
        second moment estimates of the gradients. Defaults to (0.9, 0.999).
        eps: (float) Small constant added to the denominator for numerical
        stability. Defaults to 1e-8.
        weight_decay: (float) Decoupled weight decay coefficient. Parameters are
        multiplied by (1 - lr * weight_decay) at each optimization step.
        Defaults to 0.01.
    """
    def __init__(
        self,
        params,
        lr= 1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    ):
        defaults = {
            "lr":lr,
            "betas":betas,
            "eps":eps,
            "weight_decay":weight_decay
        }

        super().__init__(params,defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                # m_t = β1 m_{t-1} + (1-β1) g_t
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # v_t = β2 v_{t-1} + (1-β2) g_t²
                v.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1 - beta2,
                )

                t = state["step"]

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                p.mul_(1 - lr * wd)

                p.addcdiv_(
                    m_hat,
                    v_hat.sqrt().add_(eps),
                    value=-lr,
                )


        return loss