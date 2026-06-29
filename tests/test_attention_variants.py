"""
Unit tests for MQA, GQA, and MLA attention variants.

Run with:
    uv run pytest tests/test_attention_variants.py -v
"""

import torch
import pytest
from einops import rearrange

from cs336_basics.model.transformer import *
from cs336_basics.model.attention_variants import *

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 8


@pytest.fixture
def num_heads():
    return 4


@pytest.fixture
def d_head():
    return 16


@pytest.fixture
def d_model(num_heads, d_head):
    return num_heads * d_head  # 64


# ---------------------------------------------------------------------------
# MQA — shape
# ---------------------------------------------------------------------------


def test_mqa_output_shape(batch_size, seq_len, d_model, num_heads):
    x = torch.randn(batch_size, seq_len, d_model)
    mqa = MultiQueryAttention(d_model, num_heads)
    out = mqa(x)
    assert out.shape == (batch_size, seq_len, d_model)


def test_mqa_kv_param_count(d_model, num_heads, d_head):
    """MQA has H× fewer KV params than MHA."""
    mha = MultiHeadSelfAttention(d_model, num_heads)
    mqa = MultiQueryAttention(d_model, num_heads)

    mha_kv = mha.k_proj.weight.numel() + mha.v_proj.weight.numel()
    mqa_kv = mqa.k_proj.weight.numel() + mqa.v_proj.weight.numel()

    assert mqa_kv * num_heads == mha_kv
    assert mqa_kv == 2 * d_head * d_model  # one head per KV type


# ---------------------------------------------------------------------------
# GQA — shape and param count
# ---------------------------------------------------------------------------


def test_gqa_output_shape(batch_size, seq_len, d_model, num_heads):
    x = torch.randn(batch_size, seq_len, d_model)
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=2)
    out = gqa(x)
    assert out.shape == (batch_size, seq_len, d_model)


def test_gqa_kv_param_count_scales_with_num_kv_heads(d_model, num_heads):
    """KV parameter count is linear in num_kv_heads."""
    gqa_g1 = GroupedQueryAttention(d_model, num_heads, num_kv_heads=1)
    gqa_g2 = GroupedQueryAttention(d_model, num_heads, num_kv_heads=2)
    gqa_g4 = GroupedQueryAttention(d_model, num_heads, num_kv_heads=4)

    kv = lambda m: m.k_proj.weight.numel() + m.v_proj.weight.numel()
    assert kv(gqa_g2) == 2 * kv(gqa_g1)
    assert kv(gqa_g4) == 4 * kv(gqa_g1)


# ---------------------------------------------------------------------------
# Equivalence: GQA(G=H) == MHA with the same weights
# ---------------------------------------------------------------------------


def test_gqa_full_kv_equals_mha(batch_size, seq_len, d_model, num_heads):
    """GQA with num_kv_heads == num_heads must produce identical output to MHA."""
    torch.manual_seed(0)
    x = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadSelfAttention(d_model, num_heads)
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=num_heads)

    gqa.q_proj.weight.data = mha.q_proj.weight.data.clone()
    gqa.k_proj.weight.data = mha.k_proj.weight.data.clone()
    gqa.v_proj.weight.data = mha.v_proj.weight.data.clone()
    gqa.output_proj.weight.data = mha.output_proj.weight.data.clone()

    torch.testing.assert_close(mha(x), gqa(x), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Equivalence: MQA == GQA(G=1) with the same weights
# ---------------------------------------------------------------------------


def test_mqa_equals_gqa_g1(batch_size, seq_len, d_model, num_heads):
    """MQA is exactly GQA with a single KV head group."""
    torch.manual_seed(1)
    x = torch.randn(batch_size, seq_len, d_model)

    mqa = MultiQueryAttention(d_model, num_heads)
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=1)

    gqa.q_proj.weight.data = mqa.q_proj.weight.data.clone()
    gqa.k_proj.weight.data = mqa.k_proj.weight.data.clone()
    gqa.v_proj.weight.data = mqa.v_proj.weight.data.clone()
    gqa.output_proj.weight.data = mqa.output_proj.weight.data.clone()

    torch.testing.assert_close(mqa(x), gqa(x), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Causal masking — MQA / GQA
# ---------------------------------------------------------------------------


def _causal_masking_check(module, batch_size, seq_len, d_model, split=4):
    """
    Verify that output at positions [0, split) is unchanged when tokens
    at positions [split, seq_len) are replaced with random noise.
    """
    torch.manual_seed(2)
    x = torch.randn(batch_size, seq_len, d_model)
    module.eval()
    with torch.no_grad():
        out_full = module(x)

    x_mod = x.clone()
    x_mod[:, split:, :] = torch.randn_like(x_mod[:, split:, :])
    with torch.no_grad():
        out_mod = module(x_mod)

    torch.testing.assert_close(
        out_full[:, :split, :], out_mod[:, :split, :], atol=1e-5, rtol=1e-5
    )


def test_mqa_causal_masking(batch_size, seq_len, d_model, num_heads):
    mqa = MultiQueryAttention(d_model, num_heads)
    _causal_masking_check(mqa, batch_size, seq_len, d_model)


def test_gqa_causal_masking(batch_size, seq_len, d_model, num_heads):
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=2)
    _causal_masking_check(gqa, batch_size, seq_len, d_model)


# ---------------------------------------------------------------------------
# MLA — shape
# ---------------------------------------------------------------------------


def test_mla_output_shape(batch_size, seq_len, d_model, num_heads):
    d_c = d_model // 2
    x = torch.randn(batch_size, seq_len, d_model)
    mla = MultiHeadLatentAttention(d_model, num_heads, d_c=d_c)
    out = mla(x)
    assert out.shape == (batch_size, seq_len, d_model)


def test_mla_decoupled_rope_output_shape(batch_size, seq_len, d_model, num_heads, d_head):
    d_k_rope = d_head // 2  # rope portion = half the head dim
    d_c = d_model // 2
    rope = RotaryPositionalEmbedding(
        theta=10000.0, d_k=d_k_rope, max_seq_len=seq_len * 2
    )
    mla = MultiHeadLatentAttention(
        d_model, num_heads, d_c=d_c, d_k_rope=d_k_rope, rope=rope
    )
    x = torch.randn(batch_size, seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(0)  # [1, L]
    out = mla(x, token_positions=pos)
    assert out.shape == (batch_size, seq_len, d_model)


# ---------------------------------------------------------------------------
# MLA — KV cache memory analysis
# ---------------------------------------------------------------------------


def test_mla_kv_cache_smaller_than_mha(d_model, num_heads):
    """KV latent dimension d_c must be smaller than the equivalent MHA KV cache."""
    d_c = d_model // 2
    mla = MultiHeadLatentAttention(d_model, num_heads, d_c=d_c)

    mha_kv_cache_dim = 2 * d_model  # 2 * H * d_k per token position
    mla_kv_cache_dim = d_c           # only c_KV per token position

    assert mla_kv_cache_dim < mha_kv_cache_dim, (
        f"MLA d_c={mla_kv_cache_dim} should be less than MHA KV dim={mha_kv_cache_dim}"
    )


def test_mla_latent_reconstruction(batch_size, seq_len, d_model, num_heads):
    """
    The KV latent c_KV is computed by w_dkv; K and V are deterministic up-projections
    from c_KV alone. Verify that re-computing from the same c_KV gives identical K, V.
    """
    d_c = d_model // 2
    mla = MultiHeadLatentAttention(d_model, num_heads, d_c=d_c)
    mla.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    with torch.no_grad():
        c_kv_1 = mla.w_dkv(x)
        k_1 = mla.w_uk(c_kv_1)
        v_1 = mla.w_uv(c_kv_1)

        # Simulate caching c_KV and re-using it
        c_kv_2 = c_kv_1.clone()
        k_2 = mla.w_uk(c_kv_2)
        v_2 = mla.w_uv(c_kv_2)

    torch.testing.assert_close(k_1, k_2)
    torch.testing.assert_close(v_1, v_2)


# ---------------------------------------------------------------------------
# MLA — causal masking
# ---------------------------------------------------------------------------


def test_mla_causal_masking(batch_size, seq_len, d_model, num_heads):
    d_c = d_model // 2
    mla = MultiHeadLatentAttention(d_model, num_heads, d_c=d_c)
    _causal_masking_check(mla, batch_size, seq_len, d_model)


def test_mla_decoupled_rope_causal_masking(
    batch_size, seq_len, d_model, num_heads, d_head
):
    d_k_rope = d_head // 2
    d_c = d_model // 2
    rope = RotaryPositionalEmbedding(
        theta=10000.0, d_k=d_k_rope, max_seq_len=seq_len * 2
    )
    mla = MultiHeadLatentAttention(
        d_model, num_heads, d_c=d_c, d_k_rope=d_k_rope, rope=rope
    )

    torch.manual_seed(3)
    x = torch.randn(batch_size, seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(0)
    split = 4

    mla.eval()
    with torch.no_grad():
        out_full = mla(x, pos)
        x_mod = x.clone()
        x_mod[:, split:, :] = torch.randn_like(x_mod[:, split:, :])
        out_mod = mla(x_mod, pos)

    torch.testing.assert_close(
        out_full[:, :split, :], out_mod[:, :split, :], atol=1e-5, rtol=1e-5
    )


# ---------------------------------------------------------------------------
# Gradient flow — all parameters must receive gradients
# ---------------------------------------------------------------------------


def test_gqa_gradient_flow(batch_size, seq_len, d_model, num_heads):
    x = torch.randn(batch_size, seq_len, d_model)
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=2)
    loss = gqa(x).sum()
    loss.backward()
    for name, p in gqa.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} has no gradient"


def test_mla_gradient_flow(batch_size, seq_len, d_model, num_heads):
    d_c = d_model // 2
    x = torch.randn(batch_size, seq_len, d_model)
    mla = MultiHeadLatentAttention(d_model, num_heads, d_c=d_c)
    loss = mla(x).sum()
    loss.backward()
    for name, p in mla.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} has no gradient"


def test_mla_decoupled_rope_gradient_flow(batch_size, seq_len, d_model, num_heads, d_head):
    d_k_rope = d_head // 2
    d_c = d_model // 2
    rope = RotaryPositionalEmbedding(
        theta=10000.0, d_k=d_k_rope, max_seq_len=seq_len * 2
    )
    mla = MultiHeadLatentAttention(
        d_model, num_heads, d_c=d_c, d_k_rope=d_k_rope, rope=rope
    )
    x = torch.randn(batch_size, seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(0)
    loss = mla(x, pos).sum()
    loss.backward()
    for name, p in mla.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} has no gradient"
