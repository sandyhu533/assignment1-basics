import torch
from torch import nn
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        weight = torch.zeros((out_features, in_features),dtype=dtype, device=device)
        std = (2/(in_features+out_features))**0.5
        weight = nn.init.trunc_normal_(weight, 0, std, std*-3, std*3)
        self.weight = nn.Parameter(weight)
    
    def forward(self, x):
        return einsum(self.weight, x, 
                    "... out_features in_features, ... in_features -> ... out_features")

class Embedding(nn.Module):
    def __init__(self, embedding_nums, embedding_dim, device=None, dtype=None):
        super().__init__()
        weight = torch.zeros((embedding_nums, embedding_dim), dtype=dtype, device=device)
        weight = nn.init.trunc_normal_(weight, 0, 1, -3, 3)
        self.weight = nn.Parameter(weight)
    
    def forward(self, token_ids):
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        weight = torch.ones((d_model,), dtype=dtype, device=device)
        self.weight = nn.Parameter(weight)
        self.eps = eps
    
    def forward(self, x: torch.Tensor):
        d_model = x.size(-1)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (1/d_model*x.pow(2).sum(dim=-1, keepdim=True)+self.eps)**0.5
        res = x/rms*self.weight
        return res.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
    
    def forward(self, x: torch.Tensor):
        w1x = self.w1(x)
        silu = w1x * nn.functional.sigmoid(w1x)
        w1w2 = silu * self.w3(x)
        return self.w2(w1w2)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        i = torch.arange(0, d_k, 2, device=device)
        freqs = 1/(theta**(i/d_k))
        positions = torch.arange(0, max_seq_len, 1, device=device)
        angles = torch.outer(positions, freqs)
        cos_table = torch.cos(angles)
        sin_table = torch.sin(angles)
        self.register_buffer("cos_cache", cos_table, persistent=False)
        self.register_buffer("sin_cache", sin_table, persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        x_pair = rearrange(x, "... (d two) -> ... d two", two=2)
        x_even = x_pair[...,0]
        x_odd = x_pair[...,1]
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)
        x_even_new = x_even * cos - x_odd * sin
        x_odd_new = x_even * sin + x_odd * cos
        res = torch.stack([x_even_new, x_odd_new], dim=-1)
        return rearrange(res, "... d two -> ... (d two)")
    

def softmax(v: torch.Tensor, i: int):
    maxv = v.amax(dim=i, keepdim=True)
    v = v.sub(maxv).exp()
    return v / v.sum(dim=i, keepdim=True)

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
    d_k = q.size(-1)
    scores = einsum(q, k, "... q d_k, ... k d_k -> ... q k")
    scores = scores/(d_k**0.5)
    if mask is not None:
        scores = torch.where(mask, scores, float('-inf'))
    scores = softmax(scores, -1)
    return einsum(scores, v, "... q k, ... k d_k -> ... q d_k")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope = None, device=None, dtype=None):
        super().__init__()
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)
        self.rope = rope
        self.d_k = d_model // num_heads
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None):
        q = rearrange(self.q_proj(x), "... seq (num_heads d_k) -> ... num_heads seq d_k", d_k = self.d_k)
        k = rearrange(self.k_proj(x), "... seq (num_heads d_k) -> ... num_heads seq d_k", d_k = self.d_k)
        v = rearrange(self.v_proj(x), "... seq (num_heads d_k) -> ... num_heads seq d_k", d_k = self.d_k)
        seq_len = x.size(-2)
        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        qkv = scaled_dot_product_attention(q, k, v, mask)
        att = rearrange(qkv, "... num_heads seq d_k -> ... seq (num_heads d_k)")
        return self.output_proj(att)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope=None, device=None, dtype=None):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope, device, dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        
    def forward(self, x, position_ids):
        y = self.ln1(x)
        y = self.attn(y, position_ids)
        x = y + x
        y = self.ln2(x)
        y = self.ffn(y)
        return y + x
        
class TransformerLM(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, vocab_size, context_length, num_layers, rope=None, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, rope, device, dtype) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)
    
    def forward(self, token_ids, position_ids=None):
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, position_ids)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
