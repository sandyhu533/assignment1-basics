from typing import Callable, Optional
import torch

def cross_entropy(logits:torch.Tensor, targets:torch.Tensor):
    # - log softmax(logits)
    # - log( exp(logits) / sum(exp(logits)))
    # - log( exp(logits-max) / sum(exp(logits-max)))
    # - logits + max + log(sum(exp(logits-max)))
    # [target]
    
    maxv = logits.amax(-1, keepdim=True)
    logsumexp = maxv + (logits-maxv).exp().sum(-1, keepdim=True).log()
    log_probs = logits - logsumexp
    ll = -torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
    return ll.mean()
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                t = state.get('t', 1)
                m = state.get('m', 0)
                v = state.get('v', 0)
                lrt = lr*((1-b2**t)**0.5)/(1-b1**t)
                p.data -= lr*weight_decay*p.data
                m = b1*m + (1-b1)*grad
                v = b2*v + (1-b2)*grad**2
                p.data -= lrt*m/(v**0.5+eps)
                state['t'] = t+1
                state['m'] = m
                state['v'] = v
        return loss


