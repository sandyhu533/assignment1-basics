import torch
from torch import nn
from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        w = torch.zeros((out_features, in_features), device=device, dtype=dtype)
        std = (2/(in_features+out_features))**0.5
        torch.nn.init.trunc_normal_(w, 0, std, -3*std, 3*std)
        self.weight = nn.Parameter(w)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_emb, emb_dim, device=None, dtype=None):
        super().__init__()
        w = torch.zeros((num_emb, emb_dim), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(w, 0, 1, -3,3)
        self.weight = nn.Parameter(w)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        g = torch.ones((d_model,), device=device, dtype=dtype)
        self.weight = nn.Parameter(g)
        self.eps = eps
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(1/self.d_model*x.pow(2).sum(dim=-1, keepdim=True)+self.eps)
        result = x / rms * self.weight
        return result.to(in_dtype)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        return self.w2(gate*torch.sigmoid(gate)*self.w3(x))

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

def softmax(x: torch.Tensor, i) -> torch.Tensor:
    m = x.amax(dim=i, keepdim=True)
    exp_x = (x-m).exp()
    return exp_x / exp_x.sum(dim=i, keepdim=True)

def scaled_dot_product_attention(q, k, v, mask=None) -> torch.Tensor:
    d_k = q.size(-1)
    qk = einsum(q, k, "... query d_k, ... key d_k -> ... query key")
    qk = qk/(d_k**0.5)        
    if mask is not None:
        qk = torch.where(mask, qk, float('-inf'))
    qk = softmax(qk, -1)
    qkv = einsum(qk, v, "... query key, ... key d_v -> ... query d_v")
    return qkv

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        super().__init__()
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, self.d_k * num_heads, device, dtype)
        self.k_proj = Linear(d_model, self.d_k * num_heads, device, dtype)  
        self.v_proj = Linear(d_model, self.d_k * num_heads, device, dtype)        
        self.output_proj = Linear(d_model, d_model, device, dtype)
        self.rope = rope
    
    def forward(self, x, token_positions=None):
        q_proj = rearrange(self.q_proj(x), "... seq_length (num_heads d_k) -> ... num_heads seq_length d_k", d_k = self.d_k)
        k_proj = rearrange(self.k_proj(x), "... seq_length (num_heads d_k) -> ... num_heads seq_length d_k", d_k = self.d_k)
        if token_positions is not None:
            q_proj = self.rope(q_proj, token_positions)
            k_proj = self.rope(k_proj, token_positions)
        v_proj = rearrange(self.v_proj(x), "... seq_length (num_heads d_k) -> ... num_heads seq_length d_k", d_k = self.d_k)
        mask = torch.tril(torch.ones((x.size(-2), x.size(-2))))
        att = scaled_dot_product_attention(q_proj, k_proj, v_proj, mask= mask == 1)
        out = rearrange(att, "... num_heads seq_length d_k -> ... seq_length (num_heads d_k)")
        return self.output_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope=None, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x, token_positions=None):
        y = self.ln1(x)
        y = self.attn(y, token_positions)
        x = x + y
        y = self.ln2(x)
        y = self.ffn(y)
        x = x + y
        return x
        
class TransformerLM(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, vocab_size, context_length, num_layers, rope=None, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList( [TransformerBlock(d_model, num_heads, d_ff, rope, device, dtype) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)
    
    def forward(self, x, token_positions=None):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)