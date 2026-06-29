import torch
import pytest
from cs336_basics.model.transformer import MultiHeadSelfAttention
from cs336_basics.model.attention_variants import MultiQueryAttention

try:
    from cs336_basics.model.attention_variants import GroupedQueryAttention
    HAS_GQA = True
except ImportError:
    HAS_GQA = False

pytestmark = pytest.mark.skipif(not HAS_GQA, reason="GroupedQueryAttention not implemented")


@pytest.fixture
def batch_size(): return 2
@pytest.fixture
def seq_len(): return 8
@pytest.fixture
def num_heads(): return 4
@pytest.fixture
def d_head(): return 16
@pytest.fixture
def d_model(num_heads, d_head): return num_heads * d_head


def test_gqa_output_shape(batch_size, seq_len, d_model, num_heads):
    x = torch.randn(batch_size, seq_len, d_model)
    out = GroupedQueryAttention(d_model, num_heads, num_kv_heads=2)(x)
    assert out.shape == (batch_size, seq_len, d_model)


def test_gqa_kv_param_count_scales_with_num_kv_heads(d_model, num_heads):
    """KV parameter count is proportional to num_kv_heads."""
    kv = lambda m: m.k_proj.weight.numel() + m.v_proj.weight.numel()
    g1 = GroupedQueryAttention(d_model, num_heads, num_kv_heads=1)
    g2 = GroupedQueryAttention(d_model, num_heads, num_kv_heads=2)
    g4 = GroupedQueryAttention(d_model, num_heads, num_kv_heads=4)
    assert kv(g2) == 2 * kv(g1)
    assert kv(g4) == 4 * kv(g1)


def test_gqa_full_kv_heads_equals_mha(batch_size, seq_len, d_model, num_heads):
    """GQA(num_kv_heads=num_heads) must be identical to MHA given the same weights."""
    torch.manual_seed(0)
    x = torch.randn(batch_size, seq_len, d_model)
    mha = MultiHeadSelfAttention(d_model, num_heads)
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=num_heads)
    gqa.q_proj.weight.data = mha.q_proj.weight.data.clone()
    gqa.k_proj.weight.data = mha.k_proj.weight.data.clone()
    gqa.v_proj.weight.data = mha.v_proj.weight.data.clone()
    gqa.output_proj.weight.data = mha.output_proj.weight.data.clone()
    torch.testing.assert_close(mha(x), gqa(x), atol=1e-5, rtol=1e-5)


def test_gqa_single_kv_head_equals_mqa(batch_size, seq_len, d_model, num_heads):
    """GQA(num_kv_heads=1) must be identical to MQA given the same weights."""
    torch.manual_seed(1)
    x = torch.randn(batch_size, seq_len, d_model)
    mqa = MultiQueryAttention(d_model, num_heads)
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=1)
    gqa.q_proj.weight.data = mqa.q_proj.weight.data.clone()
    gqa.k_proj.weight.data = mqa.k_proj.weight.data.clone()
    gqa.v_proj.weight.data = mqa.v_proj.weight.data.clone()
    gqa.output_proj.weight.data = mqa.output_proj.weight.data.clone()
    torch.testing.assert_close(mqa(x), gqa(x), atol=1e-5, rtol=1e-5)


def test_gqa_causal_masking(batch_size, seq_len, d_model, num_heads):
    torch.manual_seed(2)
    split = 4
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=2)
    gqa.eval()
    x = torch.randn(batch_size, seq_len, d_model)
    with torch.no_grad():
        out_full = gqa(x)
        x_mod = x.clone()
        x_mod[:, split:] = torch.randn_like(x_mod[:, split:])
        out_mod = gqa(x_mod)
    torch.testing.assert_close(out_full[:, :split], out_mod[:, :split], atol=1e-5, rtol=1e-5)


def test_gqa_gradient_flow(batch_size, seq_len, d_model, num_heads):
    x = torch.randn(batch_size, seq_len, d_model)
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=2)
    gqa(x).sum().backward()
    for name, p in gqa.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} has no gradient"
