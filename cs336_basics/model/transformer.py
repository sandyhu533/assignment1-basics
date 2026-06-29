import torch
from torch import nn
from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None):
        super().__init__()
        weight = torch.zeros((out_features, in_features), device=device)
        std = (2/(in_features+out_features))**0.5
        nn.init.trunc_normal_(weight, 0, std, -3*std, 3*std)
        self.weight = nn.Parameter(weight)
    
    def forward(self, x):
        return einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None):
        super().__init__()
        weight = torch.zeros((num_embeddings, embedding_dim), device=None)
        std = 1
        nn.init.trunc_normal_(weight, 0, std, -3*std, 3*std)
        self.weight = nn.Parameter(weight)
    
    def forward(self, x):
        return self.weight[x]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None):
        super().__init__()
        weight = torch.ones((d_model,), device=None)
        self.weight = nn.Parameter(weight)
        self.eps = eps
    
    def forward(self, x:torch.Tensor):
        d_model = x.size(-1)
        rms = (1/d_model*x.pow(2).sum(dim=-1, keepdim=True)+self.eps)**0.5
        return x/rms*self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device)
        self.w3 = Linear(d_model, d_ff, device)
        self.w2 = Linear(d_ff, d_model, device)
    
    def forward(self, x):
        w1x = self.w1(x)
        gate = w1x * torch.sigmoid(w1x)
        return self.w2(gate*self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        k = 1/theta**(torch.arange(0, d_k, 2, device=device)/d_k)
        pos = torch.arange(0, max_seq_len, 1, device=device)
        angle = torch.outer(pos, k)
        self.register_buffer("cos_cache", torch.cos(angle), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angle), persistent=False)
    
    def forward(self, x, token_positions):
        sin = self.sin_cache[token_positions]
        cos = self.cos_cache[token_positions]
        pair = rearrange(x, "... (d two) -> ... d two", two=2)
        even, odd = pair[...,0], pair[...,1]
        new_even, new_odd = cos*even-sin*odd, sin*even+cos*odd
        new_pair = torch.stack((new_even, new_odd), dim=-1)
        return rearrange(new_pair, "... d two -> ... (d two)")

def softmax(x:torch.Tensor, dim=-1):
    xmax = x.amax(dim=dim, keepdim=True)
    x = (x-xmax).exp()
    return x/x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask=None):
    d_k = q.size(-1)
    qk = einsum(q, k, "... query d_k, ... key d_k -> ... query key")/d_k**0.5
    if mask is not None:
        qk = torch.where(mask, qk, -float('inf'))
    qk = softmax(qk, dim=-1)
    return einsum(qk, v, "... query key, ... key d_k -> ... query d_k")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None):
        super().__init__()
        self.q_proj = Linear(d_model, d_model, device)
        self.k_proj = Linear(d_model, d_model, device)
        self.v_proj = Linear(d_model, d_model, device)
        self.output_proj = Linear(d_model, d_model, device)
        self.num_heads = num_heads
        self.rope = rope
    
    def forward(self, x, token_positions=None):
        q = rearrange(self.q_proj(x), "... cl (num_heads d_k) -> ... num_heads cl d_k", num_heads=self.num_heads)
        k = rearrange(self.k_proj(x), "... cl (num_heads d_k) -> ... num_heads cl d_k", num_heads=self.num_heads)
        v = rearrange(self.v_proj(x), "... cl (num_heads d_k) -> ... num_heads cl d_k", num_heads=self.num_heads)
        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        cl = q.size(-2)
        mask = torch.tril(torch.ones((cl, cl), dtype=torch.bool, device=q.device))
        qv = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(qv, "... num_heads cl d_k -> ... cl (num_heads d_k)")
        return self.output_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, rope=None, device=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device)
        self.ln2 = RMSNorm(d_model, device=device)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope, device)
        self.ffn = SwiGLU(d_model, d_ff, device)
    
    def forward(self, x, token_postions=None):
        y = self.ln1(x)
        y = self.attn(y, token_postions)
        x = x + y
        y = self.ln2(x)
        y = self.ffn(y)
        return x + y
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, num_layers, context_length, d_model, d_ff, num_heads, rope=None, device=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, d_ff, num_heads, rope, device) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device)
    
    def forward(self, x, token_positions=None):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)