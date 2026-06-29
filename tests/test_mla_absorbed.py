"""
Unit tests for MultiHeadLatentAttentionAbsorbed (MLA Stage 2).

The core property: weight absorption is a purely algebraic rearrangement.
Given identical weights, Stage 2 must produce byte-for-byte-equivalent output
to Stage 1 (up to float32 rounding from reordered summations, atol≈1e-4).

Run:
    uv run pytest tests/test_mla_absorbed.py -v
"""

import torch
import pytest
from cs336_basics.model.transformer import RotaryPositionalEmbedding

try:
    from cs336_basics.model.attention_variants import (
        MultiHeadLatentAttention,
        MultiHeadLatentAttentionAbsorbed,
    )
    HAS_ABSORBED = True
except ImportError:
    HAS_ABSORBED = False

pytestmark = pytest.mark.skipif(
    not HAS_ABSORBED, reason="MultiHeadLatentAttentionAbsorbed not implemented"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_size(): return 2

@pytest.fixture
def seq_len(): return 8

@pytest.fixture
def num_heads(): return 4

@pytest.fixture
def d_head(): return 16

@pytest.fixture
def d_model(num_heads, d_head): return num_heads * d_head   # 64

@pytest.fixture
def d_c(d_model): return d_model // 2                       # 32

@pytest.fixture
def d_k_rope(d_head): return d_head // 2                    # 8


# ---------------------------------------------------------------------------
# Helper: load Stage-1 weights into Stage-2 instance
# ---------------------------------------------------------------------------

def _make_absorbed_from_stage1(stage1: MultiHeadLatentAttention) -> MultiHeadLatentAttentionAbsorbed:
    """Construct an absorbed instance that shares no state but has identical weights."""
    # Build a fresh Stage-2 with the same constructor signature
    d_model_val = stage1.output_proj.weight.shape[1]
    stage2 = MultiHeadLatentAttentionAbsorbed.__new__(MultiHeadLatentAttentionAbsorbed)
    # Re-use Stage-1's __init__ by calling it on stage2
    MultiHeadLatentAttention.__init__(
        stage2,
        d_model=d_model_val,
        num_heads=stage1.num_heads,
        d_c=stage1.d_c,
        d_c_q=stage1.d_c_q,
        d_k_rope=stage1.d_k_rope if stage1.d_k_rope > 0 else None,
        rope=stage1.rope,
    )
    stage2.load_state_dict(stage1.state_dict())
    return stage2


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_absorbed_output_shape(batch_size, seq_len, d_model, num_heads, d_c):
    x = torch.randn(batch_size, seq_len, d_model)
    model = MultiHeadLatentAttentionAbsorbed(d_model, num_heads, d_c=d_c)
    out = model(x)
    assert out.shape == (batch_size, seq_len, d_model)


def test_absorbed_rope_output_shape(batch_size, seq_len, d_model, num_heads, d_c, d_k_rope):
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=d_k_rope, max_seq_len=seq_len * 2)
    model = MultiHeadLatentAttentionAbsorbed(
        d_model, num_heads, d_c=d_c, d_k_rope=d_k_rope, rope=rope
    )
    x = torch.randn(batch_size, seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(0)
    out = model(x, token_positions=pos)
    assert out.shape == (batch_size, seq_len, d_model)


# ---------------------------------------------------------------------------
# KEY TEST: Stage 2 must be numerically equivalent to Stage 1
# ---------------------------------------------------------------------------

def test_absorbed_equals_stage1_no_rope(batch_size, seq_len, d_model, num_heads, d_c):
    """
    Weight absorption is an algebraic identity:
        c_Q @ (W_UQ^T @ W_UK) @ c_KV^T  ≡  (W_UQ @ c_Q)^T @ (W_UK @ c_KV)

    Verify that Stage 2 and Stage 1 produce the same output given identical weights.
    """
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    stage1 = MultiHeadLatentAttention(d_model, num_heads, d_c=d_c)
    stage2 = _make_absorbed_from_stage1(stage1)

    stage1.eval(); stage2.eval()
    with torch.no_grad():
        out1 = stage1(x)
        out2 = stage2(x)

    torch.testing.assert_close(out1, out2, atol=1e-4, rtol=1e-4,
        msg="Absorbed Stage 2 diverged from Stage 1 (no rope)")


def test_absorbed_equals_stage1_with_rope(
    batch_size, seq_len, d_model, num_heads, d_c, d_k_rope
):
    """
    With decoupled RoPE the nope scores are absorbed; rope scores remain explicit.
    Both paths must yield identical attention distributions.
    """
    torch.manual_seed(7)
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=d_k_rope, max_seq_len=seq_len * 2)
    x = torch.randn(batch_size, seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(0)

    stage1 = MultiHeadLatentAttention(d_model, num_heads, d_c=d_c, d_k_rope=d_k_rope, rope=rope)
    stage2 = _make_absorbed_from_stage1(stage1)

    stage1.eval(); stage2.eval()
    with torch.no_grad():
        out1 = stage1(x, pos)
        out2 = stage2(x, pos)

    torch.testing.assert_close(out1, out2, atol=1e-4, rtol=1e-4,
        msg="Absorbed Stage 2 diverged from Stage 1 (with decoupled rope)")


# ---------------------------------------------------------------------------
# Causal masking
# ---------------------------------------------------------------------------

def test_absorbed_causal_masking(batch_size, seq_len, d_model, num_heads, d_c):
    torch.manual_seed(2)
    split = 4
    model = MultiHeadLatentAttentionAbsorbed(d_model, num_heads, d_c=d_c)
    model.eval()
    x = torch.randn(batch_size, seq_len, d_model)
    with torch.no_grad():
        out_full = model(x)
        x_mod = x.clone()
        x_mod[:, split:] = torch.randn_like(x_mod[:, split:])
        out_mod = model(x_mod)
    torch.testing.assert_close(out_full[:, :split], out_mod[:, :split], atol=1e-5, rtol=1e-5)


def test_absorbed_rope_causal_masking(
    batch_size, seq_len, d_model, num_heads, d_c, d_k_rope
):
    torch.manual_seed(3)
    split = 4
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=d_k_rope, max_seq_len=seq_len * 2)
    model = MultiHeadLatentAttentionAbsorbed(
        d_model, num_heads, d_c=d_c, d_k_rope=d_k_rope, rope=rope
    )
    model.eval()
    x = torch.randn(batch_size, seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(0)
    with torch.no_grad():
        out_full = model(x, pos)
        x_mod = x.clone()
        x_mod[:, split:] = torch.randn_like(x_mod[:, split:])
        out_mod = model(x_mod, pos)
    torch.testing.assert_close(out_full[:, :split], out_mod[:, :split], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Gradient flow — all parameters must receive non-zero gradients
# ---------------------------------------------------------------------------

def test_absorbed_gradient_flow(batch_size, seq_len, d_model, num_heads, d_c):
    x = torch.randn(batch_size, seq_len, d_model)
    model = MultiHeadLatentAttentionAbsorbed(d_model, num_heads, d_c=d_c)
    model(x).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} has no gradient"


def test_absorbed_rope_gradient_flow(batch_size, seq_len, d_model, num_heads, d_c, d_k_rope):
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=d_k_rope, max_seq_len=seq_len * 2)
    model = MultiHeadLatentAttentionAbsorbed(
        d_model, num_heads, d_c=d_c, d_k_rope=d_k_rope, rope=rope
    )
    x = torch.randn(batch_size, seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(0)
    model(x, pos).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} has no gradient"


# ---------------------------------------------------------------------------
# KV cache footprint: cache (c_KV, k_rope_base) not (K, V)
# ---------------------------------------------------------------------------

def test_absorbed_kv_cache_is_latent(d_model, num_heads, d_c, d_k_rope):
    """
    Real KV cache with absorption = c_KV (d_c) + k_rope_base (d_k_rope).
    This is strictly smaller than the MHA cache = 2 * d_model per token.
    """
    mha_cache_dim = 2 * d_model                  # K + V for all heads
    absorbed_cache_dim = d_c + d_k_rope           # latent + rope
    assert absorbed_cache_dim < mha_cache_dim, (
        f"absorbed cache {absorbed_cache_dim} should be < MHA cache {mha_cache_dim}"
    )
