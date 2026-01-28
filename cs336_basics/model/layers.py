"""Neural network layers: Linear, Embedding, RMSNorm, Swiglu, RoPE, Attention."""

import math

import torch
import torch.nn as nn
from einops import rearrange, einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        shape = (out_features, in_features)
        tensor = torch.empty(shape, **factory_kwargs)
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(tensor, mean=0, std=std, a=-3 * std, b=3 * std)
        self.W = nn.Parameter(tensor)

    def forward(self, x: torch.Tensor):
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        shape = (num_embeddings, embedding_dim)
        tensor = torch.empty(shape, **factory_kwargs)
        nn.init.trunc_normal_(tensor, mean=0, std=1, a=-3, b=3)
        self.W = nn.Parameter(tensor)

    def forward(self, token_ids: torch.Tensor):
        return self.W[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        tensor = torch.ones((d_model,), **factory_kwargs)
        self.g = nn.Parameter(tensor)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(ms + self.eps)
        result = x * rms * self.g
        return result.to(in_dtype)


class Swiglu(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
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
        factory_kwargs = {"device": device, "dtype": dtype}
        dim_indices = torch.arange(0, d_k, 2, **factory_kwargs).float()
        inv_freq = 1 / (theta ** (dim_indices / d_k))
        pos = torch.arange(max_seq_len, **factory_kwargs).float()
        angles = torch.outer(pos, inv_freq)
        R = torch.zeros((max_seq_len, d_k, d_k), **factory_kwargs)
        cos_v = torch.cos(angles)
        sin_v = torch.sin(angles)
        idx = torch.arange(d_k // 2, **factory_kwargs)
        R[:, 2 * idx, 2 * idx] = cos_v
        R[:, 2 * idx, 2 * idx + 1] = -sin_v
        R[:, 2 * idx + 1, 2 * idx] = sin_v
        R[:, 2 * idx + 1, 2 * idx + 1] = cos_v
        self.register_buffer("rotate", R)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        rorate = self.rotate[token_positions]
        return einsum(rorate, x, "... s i j, ... s j -> ... s i")


def softmax(x, dim):
    max_v = torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(x - max_v)
    exp_sum = torch.sum(exp, dim=dim, keepdim=True)
    return exp / exp_sum


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    qk = einsum(q, k, "... q d_k, ... k d_k -> ... q k")
    qk = qk / (d_k**0.5)
    if mask is not None:
        qk = qk.masked_fill(mask == False, float("-inf"))
    qk = softmax(qk, -1)
    qkv = einsum(qk, v, "... q k, ... k d_v -> ... q d_v")
    return qkv


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_heads = num_heads
        self.rope = None
        shape = (d_model, d_model)
        self.Q = nn.Parameter(torch.rand(shape, **factory_kwargs))
        self.K = nn.Parameter(torch.rand(shape, **factory_kwargs))
        self.V = nn.Parameter(torch.rand(shape, **factory_kwargs))
        self.O = nn.Parameter(torch.rand(shape, **factory_kwargs))

    @classmethod
    def with_rope(
        cls,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        instance = cls(d_model, num_heads, device, dtype)
        instance.rope = RotaryPositionalEmbedding(
            theta, d_model // num_heads, max_seq_len, device, dtype
        )
        return instance

    def forward(self, x, token_positions=None):
        num_heads = self.num_heads
        p_proj = einsum(self.Q, x, "d_model d_in, ... seq_len d_in -> ... seq_len d_model")
        k_proj = einsum(self.K, x, "d_model d_in, ... seq_len d_in -> ... seq_len d_model")
        v_proj = einsum(self.V, x, "d_model d_in, ... seq_len d_in -> ... seq_len d_model")
        q = rearrange(
            p_proj, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=num_heads
        )
        k = rearrange(
            k_proj, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=num_heads
        )
        v = rearrange(
            v_proj, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=num_heads
        )
        seq_len = x.shape[-2]
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == 0
        res = scaled_dot_product_attention(q, k, v, mask)
        s = rearrange(res, "... h s k -> ... s (h k)")
        s = einsum(self.O, s, "d_v d_model, ... seq d_model -> ... seq d_v")
        return s
