from typing import Callable
import torch
from torch import optim

# flops:
#          amax, -, exp, sum, log, neg, add, mean
# activations:
#          logits, logits.exp: BLV*2
#          maxv, logexpsum: BL*2
#          target: BL
def cross_entropy(logits:torch.Tensor, target:torch.Tensor):
    # -log softmax [target]
    # -log logits.exp / logits.exp.sum() [target]
    # -logits + logits.exp.sum().log() [target]
    maxv = logits.amax(dim=-1, keepdim=True) #logits, maxv
    logits = logits - maxv
    expsumlog = logits.exp().sum(dim=-1, keepdim=True).log() #logits.exp, logexpsum
    probs = -logits + expsumlog
    ll = torch.gather(probs, -1, target.unsqueeze(-1)).squeeze(-1) #target
    return ll.mean()

class AdamW(optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure: Callable|None=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 0)
                t += 1
                m = state.get('m', 0)
                v = state.get('v', 0)
                lrt = lr*((1-b2**t)**0.5)/(1-b1**t)
                p.data -= lr*weight_decay*p.data
                m = b1*m + (1-b1)*p.grad
                v = b2*v + (1-b2)*p.grad**2
                p.data -= lrt*m/(v**0.5+eps)
                state['t'] = t
                state['m'] = m
                state['v'] = v
        return loss
    