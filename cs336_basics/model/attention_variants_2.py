import torch
from torch import nn
from einops import rearrange
from cs336_basics.model.transformer import *

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, rope=None, device=None):
        super().__init__()
        d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device)
        self.k_proj = Linear(d_model, d_k, device)
        self.v_proj = Linear(d_model, d_k, device)
        self.output_proj = Linear(d_model, d_model, device)
        self.num_heads = num_heads
        self.rope = rope
    
    def forward(self, x:torch.Tensor, token_positions=None) -> torch.Tensor:
        q = rearrange(self.q_proj(x), "... context_length (num_heads d_k) -> ... num_heads context_length d_k", num_heads=self.num_heads)
        k = rearrange(self.k_proj(x), "... context_length (num_heads d_k) -> ... num_heads context_length d_k", num_heads=1)
        v =  rearrange(self.v_proj(x), "... context_length (num_heads d_k) -> ... num_heads context_length d_k", num_heads=1)
        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        context_length = q.size(-2)
        mask = torch.tril(torch.ones((context_length, context_length), device=x.device, dtype=torch.bool))
        k = torch.broadcast_to(k, q.shape)
        v = torch.broadcast_to(v, q.shape)
        res = scaled_dot_product_attention(q, k, v, mask)
        return self.output_proj(
            rearrange(res, "... num_heads context_length d_k -> ... context_length (num_heads d_k)")
        )

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, rope=None, device=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device)
        self.k_proj = Linear(d_model, num_kv_heads*d_k, device)
        self.v_proj = Linear(d_model, num_kv_heads*d_k, device)
        self.output_proj = Linear(d_model, d_model, device)
        self.rope = rope
    
    def forward(self, x:torch.Tensor, token_positions=None):
        q = rearrange(self.q_proj(x), "... context_length (num_heads d_k) -> ... num_heads context_length d_k", num_heads=self.num_heads)
        k = rearrange(self.k_proj(x), "... context_length (num_heads d_k) -> ... num_heads context_length d_k", num_heads=self.num_kv_heads)
        v =  rearrange(self.v_proj(x), "... context_length (num_heads d_k) -> ... num_heads context_length d_k", num_heads=self.num_kv_heads)
        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        context_length = q.size(-2)
        mask = torch.tril(torch.ones((context_length, context_length), device=x.device, dtype=torch.bool))
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        res = scaled_dot_product_attention(q, k, v, mask)
        return self.output_proj(
            rearrange(res, "... num_heads context_length d_k -> ... context_length (num_heads d_k)")
        )

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_c: int,d_c_q:int=None, d_k_rope:int=None, rope=None, device=None):
        super().__init__()
        if d_k_rope is None:
            d_k_rope = 0
        if d_c_q is None:
            d_c_q = d_c
        d_k = d_model // num_heads
        d_k_nope = d_k - d_k_rope
        self.w_dkv = Linear(d_model, d_c, device)
        self.w_uk = Linear(d_c, num_heads*d_k_nope)
        self.w_uv = Linear(d_c, num_heads*d_k)
        self.w_dq = Linear(d_model, d_c_q)
        self.w_uq = Linear(d_c_q, num_heads*d_k_nope)
        if d_k_rope > 0:
            self.w_qr = Linear(d_c_q, num_heads*d_k_rope)
            self.w_kr = Linear(d_model, d_k_rope)
        self.output_proj = Linear(d_model, d_model)
        self.num_heads = num_heads
        self.d_k_nope = d_k_nope
        self.d_k = d_k
        self.d_k_rope = d_k_rope
        self.d_c = d_c
        self.d_c_q = d_c_q   # already resolved: None → d_c above
        self.rope = rope
    
    def forward(self, x:torch.Tensor, token_positions=None) -> torch.Tensor:
        c_kv = self.w_dkv(x) # d_c
        k_nope = rearrange(self.w_uk(c_kv), "... context_length (num_heads d_k) -> ... num_heads context_length d_k", num_heads=self.num_heads)
        v_nope = rearrange(self.w_uv(c_kv), "... context_length (num_heads d_k) -> ... num_heads context_length d_k", num_heads=self.num_heads)
        
        c_q = self.w_dq(x) # d_c_q
        q_nope = rearrange(self.w_uq(c_q), "... context_length (num_heads d_k) -> ... num_heads context_length d_k", num_heads=self.num_heads)
        
        if token_positions is not None and self.rope is not None:
            k_rope = rearrange(self.w_kr(x), "... context_length (num_heads d_k) -> ... num_heads context_length d_k", num_heads=1)
            q_rope = rearrange(self.w_qr(c_q), "... context_length (num_heads d_k) -> ... num_heads context_length d_k", num_heads=self.num_heads)
            k_rope = self.rope(k_rope, token_positions)
            q_rope = self.rope(q_rope, token_positions)
            k_rope = k_rope.repeat_interleave(self.num_heads, dim=1)
            K = torch.concat([k_nope, k_rope], dim=-1)
            Q = torch.concat([q_nope, q_rope], dim=-1)
        else:
            K = k_nope
            Q = q_nope
                
        context_length = Q.size(-2)
        mask = torch.tril(torch.ones((context_length, context_length), device=x.device, dtype=torch.bool))
        res = scaled_dot_product_attention(Q, K, v_nope, mask)
        
        return self.output_proj(
            rearrange(res, "... num_heads context_length d_k -> ... context_length (num_heads d_k)")
        )


class MultiHeadLatentAttentionAbsorbed(MultiHeadLatentAttention):
    """
    MLA Stage 2: weight absorption — K_nope and V are never explicitly materialized.

    Key insight (decoupled RoPE case):
        score = (Q_nope @ K_nope^T + Q_rope @ K_rope^T) / sqrt(d_k)
              = (c_Q @ W_combined @ c_KV^T + Q_rope @ K_rope^T) / sqrt(d_k)

    where W_combined[h] = W_UQ[h]^T @ W_UK[h]  ∈ R^{d_c_q × d_c}
    computed on the fly from existing weights — no new parameters.

    Value absorption (both cases):
        out[h] = attn[h] @ V[h] = (attn[h] @ c_KV) @ W_UV[h]^T
               = context[h] @ W_UV[h]^T,  context ∈ R^{B × L × d_c}

    RoPE is nonlinear (position-dependent rotation), so k_rope must remain explicit.
    Everything else (K_nope and V) is absorbed, reducing peak memory during decode.

    Shares all weights with Stage 1 (same __init__). Mathematically identical output.
    """

    def forward(self, x: torch.Tensor, token_positions=None) -> torch.Tensor:
        B, L, _ = x.shape
        d_k = self.d_k
        d_k_nope = self.d_k_nope

        # ---------- latents ----------
        c_kv = self.w_dkv(x)   # [B, L, d_c]
        c_q  = self.w_dq(x)    # [B, L, d_c_q]

        # ---------- fused QK weight: W_combined[h] = W_UQ[h]^T @ W_UK[h] ----------
        # w_uq.weight: [H*d_k_nope, d_c_q]  →  [H, d_k_nope, d_c_q]
        # w_uk.weight: [H*d_k_nope, d_c]    →  [H, d_k_nope, d_c]
        W_UQ = self.w_uq.weight.reshape(self.num_heads, d_k_nope, self.d_c_q)
        W_UK = self.w_uk.weight.reshape(self.num_heads, d_k_nope, self.d_c)
        # bmm: [H, d_c_q, d_k_nope] @ [H, d_k_nope, d_c] = [H, d_c_q, d_c]
        W_combined = torch.bmm(W_UQ.permute(0, 2, 1), W_UK)

        # ---------- absorbed Q: project c_q into KV latent space ----------
        # q_abs[b,h,l,c] = sum_cq c_q[b,l,cq] * W_combined[h,cq,c]
        q_abs = torch.einsum("blq,hqc->bhlc", c_q, W_combined)   # [B, H, L, d_c]

        # ---------- nope score in latent space (no K materialization) ----------
        score = torch.einsum("bhqc,bkc->bhqk", q_abs, c_kv) / (d_k ** 0.5)

        # ---------- rope score (k_rope cannot be absorbed — RoPE is nonlinear) ----------
        if self.d_k_rope > 0 and self.rope is not None and token_positions is not None:
            q_rope = rearrange(self.w_qr(c_q), "b l (h d) -> b h l d", h=self.num_heads)
            k_rope = rearrange(self.w_kr(x), "b l d -> b 1 l d")  # single shared head
            q_rope = self.rope(q_rope, token_positions)
            k_rope = self.rope(k_rope, token_positions)
            # k_rope broadcasts: [B,1,L,d_kr] → [B,H,L,L] via g=1
            score = score + torch.einsum("bhqd,bgkd->bhqk", q_rope, k_rope) / (d_k ** 0.5)

        # ---------- causal mask + softmax ----------
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
        score = torch.where(causal, score, score.new_full((), float("-inf")))
        attn = softmax(score, dim=-1)   # [B, H, L, L]

        # ---------- absorbed value: aggregate c_kv, then project ----------
        # context[b,h,q,c] = sum_k attn[b,h,q,k] * c_kv[b,k,c]
        context = torch.einsum("bhqk,bkc->bhqc", attn, c_kv)       # [B, H, L, d_c]
        # w_uv.weight: [H*d_k, d_c] → [H, d_k, d_c]
        W_UV = self.w_uv.weight.reshape(self.num_heads, d_k, self.d_c)
        # out[b,h,l,dk] = sum_c context[b,h,l,c] * W_UV[h,dk,c]
        out = torch.einsum("bhlc,hdc->bhld", context, W_UV)          # [B, H, L, d_k]

        out = rearrange(out, "b h l d -> b l (h d)")
        return self.output_proj(out)
