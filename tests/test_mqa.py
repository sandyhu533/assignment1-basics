import torch
import pytest
from cs336_basics.model.transformer import MultiHeadSelfAttention
from cs336_basics.model.attention_variants import MultiQueryAttention


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


def test_mqa_output_shape(batch_size, seq_len, d_model, num_heads):
    x = torch.randn(batch_size, seq_len, d_model)
    out = MultiQueryAttention(d_model, num_heads)(x)
    assert out.shape == (batch_size, seq_len, d_model)


def test_mqa_kv_param_count(d_model, num_heads, d_head):
    """K+V params are H× smaller than MHA."""
    mha = MultiHeadSelfAttention(d_model, num_heads)
    mqa = MultiQueryAttention(d_model, num_heads)
    mha_kv = mha.k_proj.weight.numel() + mha.v_proj.weight.numel()
    mqa_kv = mqa.k_proj.weight.numel() + mqa.v_proj.weight.numel()
    assert mqa_kv * num_heads == mha_kv
    assert mqa_kv == 2 * d_head * d_model


def test_mqa_causal_masking(batch_size, seq_len, d_model, num_heads):
    """Output at positions [0, split) must not change when future tokens are perturbed."""
    torch.manual_seed(2)
    split = 4
    mqa = MultiQueryAttention(d_model, num_heads)
    mqa.eval()
    x = torch.randn(batch_size, seq_len, d_model)
    with torch.no_grad():
        out_full = mqa(x)
        x_mod = x.clone()
        x_mod[:, split:] = torch.randn_like(x_mod[:, split:])
        out_mod = mqa(x_mod)
    torch.testing.assert_close(out_full[:, :split], out_mod[:, :split], atol=1e-5, rtol=1e-5)
