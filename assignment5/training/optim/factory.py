"""Optimizer composition and selection."""
import torch
from training.optim.cpu_offload import CPUOffloadOptimizer


class CombinedOptimizer:
    """Muon 矩阵参数与 AdamW 辅助参数共享 step / zero_grad 接口。"""
    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.param_groups = [g for optimizer in optimizers for g in optimizer.param_groups]

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state):
        if len(state["optimizers"]) != len(self.optimizers):
            raise ValueError("优化器数量与 checkpoint 不一致")
        for optimizer, saved in zip(self.optimizers, state["optimizers"]):
            optimizer.load_state_dict(saved)


def optimizer_switcher(model, config):
    params = [p for p in model.parameters() if p.requires_grad]
    kwargs = dict(lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer_device = getattr(config, "optimizer_device", "model")
    if optimizer_device == "cpu":
        return CPUOffloadOptimizer(params, config.optimizer_name, **kwargs)
    if optimizer_device != "model":
        raise ValueError(f"未知优化器设备：{optimizer_device}")
    if config.optimizer_name == "adam":
        return torch.optim.Adam(params, **kwargs)
    if config.optimizer_name == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    if config.optimizer_name != "muon":
        raise ValueError(f"未知优化器：{config.optimizer_name}")
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("当前 PyTorch 不提供 torch.optim.Muon；请使用 adam/adamw")
    # 排除 embedding / 输出层，Muon 仅更新隐藏层二维矩阵
    excluded = {id(p) for module in (model.get_input_embeddings(), model.get_output_embeddings())
                if module is not None for p in module.parameters()}
    matrices = [p for p in params if p.ndim == 2 and id(p) not in excluded]
    auxiliary = [p for p in params if p.ndim != 2 or id(p) in excluded]
    optimizers = []
    if matrices:
        optimizers.append(torch.optim.Muon(matrices, **kwargs))
    if auxiliary:
        optimizers.append(torch.optim.AdamW(auxiliary, **kwargs))
    return CombinedOptimizer(optimizers)
