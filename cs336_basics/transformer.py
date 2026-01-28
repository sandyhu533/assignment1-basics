import torch
import torch.nn as nn
from einops import rearrange, einsum
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        shape = (out_features, in_features)
        tensor = torch.empty(shape, **factory_kwargs)
        
        std = (2/(in_features + out_features))**0.5
        nn.init.trunc_normal_(tensor, mean=0, std=std, a=-3*std, b=3*std)
        
        self.W = nn.Parameter(tensor)
    
    def forward(self, x: torch.Tensor):
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        shape = (num_embeddings, embedding_dim)
        tensor = torch.empty(shape, **factory_kwargs)
        
        nn.init.trunc_normal_(tensor, mean=0, std=1, a=-3, b=3)
        self.W = nn.Parameter(tensor)
    
    def forward(self, token_ids: torch.Tensor):
        return self.W[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        tensor = torch.ones((d_model,), **factory_kwargs)
        self.g = nn.Parameter(tensor)
        self.eps = eps
        
    
    def forward(self, x:torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(ms + self.eps)
        
        result = x * rms * self.g
        
        return result.to(in_dtype)

class Swiglu(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.W1 = Linear(d_model, d_ff, device, dtype)
        self.W3 = Linear(d_model, d_ff, device, dtype)
        self.W2 = Linear(d_ff, d_model, device, dtype)
    
    def forward(self, x: torch.Tensor):
        a = self.W1(x)
        silu = a * torch.sigmoid(a)
        return self.W2(silu * self.W3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        shape = (max_seq_len, d_k, d_k)
        
        # Version 1: for loop
        # tensor = torch.zeros(shape, **factory_kwargs)
        # for j in range(max_seq_len):
        #     for k in range(d_k // 2):
        #         v = j / (theta**(2*k/d_k))
        #         tensor[j, 2*k:2*k+2, 2*k:2*k+2] = torch.tensor(
        #             [
        #                 [math.cos(v), -math.sin(v)],
        #                 [math.sin(v), math.cos(v)]
        #             ]
        #         )        
        # self.register_buffer("rotate", tensor)
        
        # Version 2: matmul
        dim_indices = torch.arange(0, d_k, 2, **factory_kwargs).float()
        inv_freq = 1 / (theta ** (dim_indices / d_k))
        
        pos = torch.arange(max_seq_len, **factory_kwargs).float()
        angles = torch.outer(pos, inv_freq)
        
        R = torch.zeros((max_seq_len, d_k, d_k), **factory_kwargs)
        
        cos_v = torch.cos(angles)
        sin_v = torch.sin(angles)
        
        idx = torch.arange(d_k // 2, **factory_kwargs)
        R[:, 2*idx, 2*idx] = cos_v   # 左上角 (0,0), (2,2)...
        R[:, 2*idx, 2*idx+1] = -sin_v  # 右上角 (0,1), (2,3)...
        R[:, 2*idx+1, 2*idx] = sin_v   # 左下角 (1,0), (3,2)...
        R[:, 2*idx+1, 2*idx+1] = cos_v   # 右下角 (1,1), (3,3)...
        
        self.register_buffer("rotate", R)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        d_k = x.shape[-1]
        R = self.rotate
        
        rorate = R[token_positions]
        return einsum(rorate, x, '... s i j, ... s j -> ... s i')

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_heads = num_heads
        self.rope = None

        shape = (d_model, d_model)
        Q = torch.rand(shape,**factory_kwargs)
        K = torch.rand(shape,**factory_kwargs)
        V = torch.rand(shape,**factory_kwargs)
        O = torch.rand(shape, **factory_kwargs)
        
        self.Q = nn.Parameter(Q)
        self.K = nn.Parameter(K)
        self.V = nn.Parameter(V)
        self.O = nn.Parameter(O)
    
    @classmethod
    def with_rope(cls, d_model:int, num_heads:int, max_seq_len: int, theta: float, device=None, dtype=None):
        instance = cls(d_model,num_heads, device, dtype)  
        rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
        instance.rope = rope
        return instance
            
    def forward(self, x, token_positions=None): # x: (... sequence_length d_in)
        num_heads = self.num_heads
        
        p_proj = einsum(self.Q, x, "d_model d_in, ... seq_len d_in -> ... seq_len d_model")
        k_proj = einsum(self.K, x, "d_model d_in, ... seq_len d_in -> ... seq_len d_model")
        v_proj = einsum(self.V, x, "d_model d_in, ... seq_len d_in -> ... seq_len d_model")
        
        q = rearrange(p_proj, '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', num_heads=num_heads)
        k = rearrange(k_proj, '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', num_heads=num_heads)
        v = rearrange(v_proj, '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', num_heads=num_heads)       
        
        seq_len = x.shape[-2]

        if self.rope != None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
                
        mask = torch.triu(torch.ones(seq_len,seq_len), diagonal=1) == 0
        
        res = scaled_dot_product_attention(q, k, v, mask)
        
        s = rearrange(res, '... h s k -> ... s (h k)')
        s = einsum(self.O, s, "d_v d_model, ... seq d_model -> ... seq d_v")
        
        return s
        
    
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    qk = einsum(q, k, '... q d_k, ... k d_k -> ... q k')
    qk = qk / d_k**0.5
    # masking
    if mask is not None:
        qk = qk.masked_fill(mask == False, float('-inf'))
    qk = softmax(qk, -1)
    qkv = einsum(qk, v, '... q k, ... k d_v -> ... q d_v')
    return qkv

def softmax(x, i):
    max_v = torch.max(x, dim=i, keepdim=True).values
    exp = torch.exp(x - max_v)
    exp_sum = torch.sum(exp, dim=i, keepdim=True)
    return exp / exp_sum

class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, max_seq_len:int, theta:float, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
            
        self.norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.mha = CausalMultiHeadSelfAttention.with_rope(d_model=d_model, num_heads=num_heads, 
                                                          max_seq_len=max_seq_len, theta=theta)
        self.ff = Swiglu(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x):
        norm1 = self.norm(x)
        seq_len = x.shape[-2]
        att = self.mha(norm1, torch.arange(0, seq_len))
        x = x + att
        norm2 = self.norm2(x)
        ff = self.ff(norm2)
        x = x + ff
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size:int, context_length:int, d_model:int, num_layer:int, num_heads:int, d_ff:int, rope_theta:int):
        super().__init__()
        
        self.embeddings = Embedding(vocab_size, d_model)
        self.blocks = [TransformerBlock(d_model=d_model, num_heads=num_heads, 
                                        d_ff=d_ff, max_seq_len=context_length, 
                                        theta=rope_theta) for _ in range(num_layer)]
        self.norm = RMSNorm(d_model=d_model)
        self.liner = Linear(d_model, vocab_size)
    
    def forward(self, x):
        # x = (b, s)
        print(f'input={x.shape}')
        x = self.embeddings(x)
        # x = (b, s, d_model)
        print(f'emb={x.shape}')
        for block in self.blocks:
            x = block(x)
        # x = (b, s, d_model)
        print(f'block={x.shape}')
        x = self.norm(x)
        # x = (b, s, d_model)
        print(f'norm={x.shape}')
        x = self.liner(x)
        # x = (b, s, vocab_size)
        print(f'final={x.shape}')
        return x

def softmax(x, i):
    max_v = torch.max(x, dim=i, keepdim=True).values
    exp = torch.exp(x - max_v)
    exp_sum = torch.sum(exp, dim=i, keepdim=True)
    return exp / exp_sum

def cross_entropy(inputs, targets):
    max_v = torch.max(inputs, dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs - max_v), dim=-1, keepdim=True)) + max_v
    log_probs = inputs - log_sum_exp    
    
    rows = torch.arange(inputs.size(0))
    loss = -log_probs[rows, targets]
    return loss.mean()

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, betas:float, weight_decay:float, eps:float, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "b1": betas[0], "b2": betas[1], "decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, b1, b2, weight_decay, eps = group["lr"], group["b1"], group["b2"], group["decay"], group["eps"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                
                grad = p.grad.data # Get the gradient of loss with respect to p.
                m = b1*m+(1-b1)*grad
                v = b2*v+(1-b2)*grad**2
                
                lrt = lr*((1-b2**t)**0.5)/(1-b1**t)
                p.data -= lrt*m/(v**0.5+eps)
                p.data -= lr*weight_decay*p.data
                
                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss

def get_lr_learning_rate(t: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int):
    if t < warmup_iters:
        return t/warmup_iters * max_learning_rate
    elif t <= cosine_cycle_iters:
        return min_learning_rate + 0.5*(1+math.cos(math.pi*(t-warmup_iters)/(cosine_cycle_iters-warmup_iters)))*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, epsilon: float = 1e-6):
    # Total L2 = sqrt(sum over params of (p.grad^2).sum())
    total_norm = torch.sqrt(
        sum(p.grad.detach().pow(2).sum() for p in parameters if p.grad is not None)
    )
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + epsilon)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale)

if __name__ == '__main__':
    # m = Linear(1, 2)
    # x = torch.Tensor([1])
    # print(x.shape)
    # m(x)
    
    # e = Embedding(10, 2)
    # token_ids = torch.rand_like(torch.empty((10,)))
    # print(token_ids.shape)
    # ebs = e([0,2,3,4,7,9])
    # print(ebs)
    
    # rms = RMSNorm(10)
    # a = torch.rand_like(torch.empty((10,)))
    # print(a)
    # b = rms(a)
    # print(b)
    
    # swiglu = Swiglu(3, 8)
    # x = torch.rand_like(torch.empty((3,)))
    # print(x)
    # y = swiglu(x)
    # print(y)
    
    # d_k = 4
    # seq_len = 3
    # rope = RotaryPositionalEmbedding(1, d_k, 10)
    # x = torch.rand_like(torch.empty((seq_len,d_k)))
    # print(x)
    # y = rope(x, torch.tensor([i for i in range(0,seq_len)]))
    # print(y)
    
    # x = torch.rand_like(torch.empty(2,2,2)) + 10
    # print(x)
    # print(softmax(x, 0))
    
    # att = CausalMultiHeadSelfAttention(10,2)
    # x = torch.rand((10,10))
    # res=att(x)
    # print(f'res={res}')
    # print(f'res.shape={res.shape}')
    
    # lm = TransformerLM(vocab_size=100, context_length=5, d_model=64, num_layer=2, num_heads=4, d_ff=128, rope_theta=0.1)
    # x = torch.rand(10, 5).int()
    
    # y = cross_entropy(torch.tensor([[2,1], [2,1]]), torch.tensor([0, 0]))
    # print(y)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e3)
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.