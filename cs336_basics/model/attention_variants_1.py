import torch
from torch import nn
from einops import rearrange
from .transformer import Linear, scaled_dot_product_attention


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention [Ainslie et al., 2023].

    Uses num_kv_heads shared K/V heads across num_heads query heads.
    Each group of (num_heads // num_kv_heads) query heads shares one K/V head.

    Special cases:
        num_kv_heads == num_heads  =>  standard MHA
        num_kv_heads == 1          =>  MQA (Multi-Query Attention)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        rope=None,
        device=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.groups = num_heads // num_kv_heads
        d_k = d_model // num_heads

        self.q_proj = Linear(d_model, num_heads * d_k, device)
        self.k_proj = Linear(d_model, num_kv_heads * d_k, device)
        self.v_proj = Linear(d_model, num_kv_heads * d_k, device)
        self.output_proj = Linear(d_model, d_model, device)
        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions=None):
        B, L, _ = x.shape
        q = rearrange(self.q_proj(x), "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b l (g d) -> b g l d", g=self.num_kv_heads)
        v = rearrange(self.v_proj(x), "b l (g d) -> b g l d", g=self.num_kv_heads)

        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Tile each KV head to serve self.groups query heads
        k = k.repeat_interleave(self.groups, dim=1)  # [B, H, L, d_k]
        v = v.repeat_interleave(self.groups, dim=1)

        mask = torch.tril(torch.ones((L, L), dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.output_proj(out)


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention [Shazeer, 2019].

    Single shared K/V head for all H query heads.
    Equivalent to GroupedQueryAttention(d_model, num_heads, num_kv_heads=1)
    but written explicitly for clarity.

    KV cache savings vs MHA: H× smaller (stores d_k instead of H*d_k per head type).
    """

    def __init__(self, d_model: int, num_heads: int, rope=None, device=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        d_k = d_model // num_heads

        self.q_proj = Linear(d_model, num_heads * d_k, device)  # H query heads
        self.k_proj = Linear(d_model, d_k, device)               # 1 key head
        self.v_proj = Linear(d_model, d_k, device)               # 1 value head
        self.output_proj = Linear(d_model, d_model, device)
        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions=None):
        B, L, _ = x.shape
        q = rearrange(self.q_proj(x), "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b l d -> b 1 l d")
        v = rearrange(self.v_proj(x), "b l d -> b 1 l d")

        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        k = k.expand(-1, self.num_heads, -1, -1)
        v = v.expand(-1, self.num_heads, -1, -1)

        mask = torch.tril(torch.ones((L, L), dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.output_proj(out)


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention [DeepSeek-AI, 2024].

    Compresses the KV cache via low-rank projection:
        c_KV = W_DKV @ x          [B, L, d_c]   <- what gets cached at inference
        K    = W_UK  @ c_KV       [B, H, L, d_k]
        V    = W_UV  @ c_KV       [B, H, L, d_k]
        c_Q  = W_DQ  @ x          [B, L, d_c_q]
        Q    = W_UQ  @ c_Q        [B, H, L, d_k]

    With decoupled RoPE (set d_k_rope > 0):
        K_nope = W_UK @ c_KV                         [B, H, L, d_k_nope]
        k_rope = W_KR @ x  (shared, not compressed)  [B, 1, L, d_k_rope]  <- also cached
        K = cat([K_nope, k_rope.expand(H)], dim=-1)  [B, H, L, d_k]
        Similarly for Q.

    KV cache per token position:
        MHA: 2 * H * d_k
        MLA (no rope): d_c
        MLA (decoupled rope): d_c + d_k_rope
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_c: int,
        d_c_q: int = None,
        d_k_rope: int = None,
        rope=None,
        device=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k_rope = d_k_rope
        d_k = d_model // num_heads
        d_k_nope = d_k - (d_k_rope or 0)
        if d_c_q is None:
            d_c_q = d_c

        # KV compression path
        self.w_dkv = Linear(d_model, d_c, device)               # x -> c_KV
        self.w_uk = Linear(d_c, num_heads * d_k_nope, device)   # c_KV -> K_nope
        self.w_uv = Linear(d_c, num_heads * d_k, device)        # c_KV -> V

        # Q compression path
        self.w_dq = Linear(d_model, d_c_q, device)              # x -> c_Q
        self.w_uq = Linear(d_c_q, num_heads * d_k_nope, device) # c_Q -> Q_nope

        # Decoupled RoPE projections (optional)
        if d_k_rope is not None:
            self.w_qr = Linear(d_c_q, num_heads * d_k_rope, device)  # c_Q -> q_rope
            self.w_kr = Linear(d_model, d_k_rope, device)             # x -> k_rope (shared)

        self.output_proj = Linear(d_model, d_model, device)
        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions=None):
        B, L, _ = x.shape

        # KV path through compressed latent
        c_kv = self.w_dkv(x)  # [B, L, d_c]
        k = rearrange(self.w_uk(c_kv), "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(self.w_uv(c_kv), "b l (h d) -> b h l d", h=self.num_heads)

        # Q path through compressed latent
        c_q = self.w_dq(x)
        q = rearrange(self.w_uq(c_q), "b l (h d) -> b h l d", h=self.num_heads)

        if self.d_k_rope is not None:
            # Decoupled RoPE: position-sensitive portions computed separately
            q_rope = rearrange(self.w_qr(c_q), "b l (h d) -> b h l d", h=self.num_heads)
            k_rope = rearrange(self.w_kr(x), "b l d -> b 1 l d")  # shared across heads

            if self.rope is not None and token_positions is not None:
                q_rope = self.rope(q_rope, token_positions)
                k_rope = self.rope(k_rope, token_positions)

            q = torch.cat([q, q_rope], dim=-1)                          # [B, H, L, d_k]
            k = torch.cat([k, k_rope.expand(B, self.num_heads, L, -1)], dim=-1)
        elif self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = torch.tril(torch.ones((L, L), dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.output_proj(out)
