import torch

# -log exp(x-xmax) - log(sum(exp(x-xmax)))
def cross_entropy(logits:torch.Tensor, target:torch.Tensor):
    maxv = logits.amax(dim=-1, keepdim=True)
    logexpsum = (logits-maxv).exp().sum(dim=-1, keepdim=True).log()
    probs = logits - maxv - logexpsum
    loss = -probs.gather(dim=-1, index=target.unsqueeze(dim=-1)).squeeze(dim=-1)
    return loss.mean()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
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