from typing import Any, Callable
from typing import Optional
from cs336_basics.model import transformer
import torch
from torch import nn
import math

# inputs: batch_size, vocab_size
# targets: batch_size
# return: The average cross-entropy loss
def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    # cp = -log(exp(input)/sum(exp(input)))
    # cp = - input + log(sum(exp((nput)))
    # cp = - (input-maxv) + log(sum(exp(input-maxv)))
    maxv = inputs.amax(-1, keepdim=True)
    log_exp_sum = maxv + (inputs-maxv).exp().sum(-1, keepdim=True).log()
    log_probs = inputs - log_exp_sum
    ll = -torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
    return ll.mean()


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
                p.data -= lr / math.sqrt(t+1) * grad
                state['t'] = t+1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                t = state.get('t', 1)
                lrt = lr*((1-b2**t)**0.5)/(1-b1**t)
                p.data -= lr*weight_decay*p.data
                m = state.get('m', 0)
                m = b1*m + (1-b1)*grad
                v = state.get('v', 0)
                v = b2*v + (1-b2)*grad**2
                p.data -= lrt*m/(v**0.5+eps)
                state['t'] = t+1
                state['m'] = m
                state['v'] = v
        return loss
    
    