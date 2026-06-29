import torch
from torch import nn
from einops import einsum, rearrange
from cs336_basics.model.transformer import *

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None):
        super().__init__()
        assert d_model % num_heads == 0
        d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device)
        self.k_proj = Linear(d_model, d_k, device)
        self.v_proj = Linear(d_model, d_k, device)
        self.output_proj = Linear(d_model, d_model, device)
        self.h = num_heads
        self.rope = rope
    
    def forward(self, x:torch.Tensor, token_positions=None) -> torch.Tensor:
        B, L, _ = x.shape
        q = rearrange(self.q_proj(x), "b l (h d)-> b h l d", h=self.h)
        k:torch.Tensor = rearrange(self.k_proj(x), "b l (1 d)->b 1 l d")
        v = rearrange(self.v_proj(x), "b l (1 d)->b 1 l d")
        if self.rope is not None and token_positions is not None:
            k = self.rope(k, token_positions)
            v = self.rope(v, token_positions)
        k = k.expand(-1, self.h, -1, -1)
        v = v.expand(-1, self.h, -1, -1)
        mask = torch.tril(torch.ones((L, L), dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.output_proj(out)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, rope=None, device=None):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0
        d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device)
        self.k_proj = Linear(d_model, d_k*num_kv_heads, device)
        self.v_proj = Linear(d_model, d_k*num_kv_heads, device)
        self.output_proj = Linear(d_model, d_model, device)
        self.h = num_heads
        self.kv_h = num_kv_heads
        self.rope = rope
    
    def forward(self, x:torch.Tensor, token_positions=None) -> torch.Tensor:
        B, L, _ = x.shape
        q = rearrange(self.q_proj(x), "b l (h d)-> b h l d", h=self.h)
        k:torch.Tensor = rearrange(self.k_proj(x), "b l (h d)->b h l d", h = self.kv_h)
        v = rearrange(self.v_proj(x), "b l (h d)->b h l d", h = self.kv_h)
        if self.rope is not None and token_positions is not None:
            k = self.rope(k, token_positions)
            v = self.rope(v, token_positions)
        k = k.repeat_interleave(self.h // self.kv_h, dim=1)
        v = v.repeat_interleave(self.h // self.kv_h, dim=1)
        mask = torch.tril(torch.ones((L, L), dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.output_proj(out)

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_c:int, d_c_q:int=None, d_k_rope:int=None, rope=None, device=None):
        super().__init__()
        assert d_model % num_heads == 0
        if d_c_q is None: d_c_q = d_c
        d_k = d_model // num_heads
        d_k_nope = d_k - d_k_rope if d_k_rope is not None else d_k
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_c = d_c
        self.d_c_q = d_c_q
        self.d_k_rope = d_k_rope
        self.d_k_nope = d_k_nope
        self.rope = rope
        
        self.w_dkv = Linear(d_model, d_c, device)
        self.w_dq = Linear(d_model, d_c_q, device)
        self.w_uk = Linear(d_c, d_k_nope*num_heads, device)
        self.w_uv = Linear(d_c, d_k*num_heads, device)
        self.w_uq = Linear(d_c_q, d_k_nope*num_heads, device)
        
        if d_k_rope is not None:
            self.w_rk = Linear(d_model, d_k_rope, device)
            self.w_rq = Linear(d_c, d_k_rope*num_heads, device)
        self.w_o = Linear(d_model, d_model, device)
    
    def forward(self, x:torch.Tensor, token_positions=None) -> torch.Tensor:
        B, L, _ = x.shape
        ckv = self.w_dkv(x)
        cq = self.w_dq(x)
        uk = rearrange(self.w_uk(ckv), "b l (h d) -> b h l d", h=self.num_heads)
        V = rearrange(self.w_uv(ckv), "b l (h d) -> b h l d", h=self.num_heads)
        uq = rearrange(self.w_uq(cq), "b l (h d) -> b h l d", h=self.num_heads)
        
        if self.rope is not None and token_positions is not None and self.d_k_rope is not None:
            rk:torch.Tensor = rearrange(self.w_rk(x), "b l (h d) -> b h l d", h=1)
            rq = rearrange(self.w_rq(cq), "b l (h d) -> b h l d", h=self.num_heads)
            rk = self.rope(rk, token_positions)
            rq = self.rope(rq, token_positions)
            rk = rk.expand(-1, self.num_heads, -1, -1)
        
            K = torch.concat([uk, rk], dim=-1)
            Q = torch.concat([uq, rq], dim=-1)
        else:
            K = uk
            Q = uq
        mask = torch.tril(torch.ones((L, L), dtype=torch.bool, device=x.device))
        res = scaled_dot_product_attention(Q, K, V, mask)
        
        res = rearrange(res, "b h l d -> b l (h d)")
        return self.w_o(res)