"""Optimizers and training utilities."""

import math
from collections.abc import Callable, Iterable
from typing import Optional

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, betas, weight_decay: float, eps: float, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "b1": betas[0], "b2": betas[1], "decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["b1"]
            b2 = group["b2"]
            weight_decay = group["decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                grad = p.grad.data
                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * grad**2
                lrt = lr * ((1 - b2**t) ** 0.5) / (1 - b1**t)
                p.data -= lrt * m / (v**0.5 + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


def get_lr_learning_rate(
    t: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if t < warmup_iters:
        return t / warmup_iters * max_learning_rate
    elif t <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1 + math.cos(math.pi * (t - warmup_iters) / (cosine_cycle_iters - warmup_iters))
        ) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, epsilon: float = 1e-6
):
    total_norm = torch.sqrt(
        sum(p.grad.detach().pow(2).sum() for p in parameters if p.grad is not None)
    )
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + epsilon)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale)
