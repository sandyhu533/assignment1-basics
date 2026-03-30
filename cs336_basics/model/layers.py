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
        # Indices must be long/int for tensor indexing (not float from factory_kwargs)
        idx = torch.arange(d_k // 2, dtype=torch.long, device=factory_kwargs.get("device"))
        R[:, 2 * idx, 2 * idx] = cos_v
        R[:, 2 * idx, 2 * idx + 1] = -sin_v
        R[:, 2 * idx + 1, 2 * idx] = sin_v
        R[:, 2 * idx + 1, 2 * idx + 1] = cos_v
        self.register_buffer("rotate", R, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        # R: (max_seq_len, d_k, d_k), token_positions indexes positions
        R = self.rotate[token_positions]  # (..., seq_len, d_k, d_k) or (seq_len, d_k, d_k)
        if x.dim() == 3:
            # x: (..., seq_len, d_k) e.g. (batch, seq_len, d_k) from run_rope
            # out[..., s, :] = R[..., s, :, :] @ x[..., s, :]
            return einsum(R, x, "... s i j, ... s j -> ... s i")
        else:
            # x: (batch, num_heads, seq_len, d_k) from MHA
            # Apply same R per position: transpose to (batch, seq_len, num_heads, d_k)
            x = x.transpose(1, 2)  # (batch, seq_len, num_heads, d_k)
            # R may be (seq_len, d_k, d_k) or (batch, seq_len, d_k, d_k)
            if R.dim() == 3:
                R = R.unsqueeze(0)  # (1, seq_len, d_k, d_k) for broadcasting with batch
            out = einsum(R, x, "... s i j, ... s h j -> ... s h i")
            return out.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)


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
        
        # 1. 创建空 Tensor
        q_tensor = torch.empty(shape, **factory_kwargs)
        k_tensor = torch.empty(shape, **factory_kwargs)
        v_tensor = torch.empty(shape, **factory_kwargs)
        o_tensor = torch.empty(shape, **factory_kwargs)
        
        # 2. 计算标准差 (参考 Xavier Init 或 Assignment 要求)
        # 通常 Attention 层的 std 使用 1/sqrt(d_model)
        std = 1.0 / (d_model ** 0.5)
        
        # 3. 使用 trunc_normal_ 初始化 (保持均值为 0)
        nn.init.trunc_normal_(q_tensor, mean=0.0, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(k_tensor, mean=0.0, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(v_tensor, mean=0.0, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(o_tensor, mean=0.0, std=std, a=-3 * std, b=3 * std)
        
        self.Q = nn.Parameter(q_tensor)
        self.K = nn.Parameter(k_tensor)
        self.V = nn.Parameter(v_tensor)
        self.O = nn.Parameter(o_tensor)

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

    def forward(self, x, token_positions=None, past_kv=None, use_cache=False):
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
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        new_kv = (k, v) if use_cache else None

        q_len = q.shape[-2]
        kv_len = k.shape[-2]
        past_len = kv_len - q_len
        mask = torch.triu(torch.ones(q_len, kv_len, device=q.device), diagonal=past_len + 1) == 0
        res = scaled_dot_product_attention(q, k, v, mask)
        s = rearrange(res, "... h s k -> ... s (h k)")
        s = einsum(self.O, s, "d_v d_model, ... seq d_model -> ... seq d_v")
        if use_cache:
            return s, new_kv
        return s
