import torch

# activations:
    # logitis ~ BCV
    # logits.exp() ~ BCV
    # target ~ BC
def cross_entropy(logits:torch.Tensor, target:torch.Tensor):
    # ce = -log ( l.exp / l.exp.sum)
    # ce = - l + l.exp.sum.log
    # ce = -l + max + (l-max).exp.sum.log
    maxv = logits.amax(dim=-1, keepdim=True)
    logits = logits - maxv
    logexpsum = logits.exp().sum(dim=-1, keepdim=True).log()
    loss = -logits + logexpsum
    l = loss.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    return l.mean()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, weight_decay, eps):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                t += 1
                m = state.get("m", 0)
                v = state.get("v", 0)
                lrt = lr*((1-b2**t)**0.5)/(1-b1**t)
                p.data -= weight_decay*lr*p.data
                m = b1*m+(1-b1)*p.grad
                v = b2*v+(1-b2)*p.grad**2
                p.data -= lrt*m/(v**0.5+eps)
                state["t"] = t
                state["m"] = m
                state["v"] = v
        return loss