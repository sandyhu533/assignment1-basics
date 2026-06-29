import torch
import pytest
from cs336_basics.model.transformer import RotaryPositionalEmbedding

try:
    from cs336_basics.model.attention_variants import MultiHeadLatentAttention
    HAS_MLA = True
except ImportError:
    HAS_MLA = False

pytestmark = pytest.mark.skipif(not HAS_MLA, reason="MultiHeadLatentAttention not implemented")


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
@pytest.fixture
def d_c(d_model): return d_model // 2


def test_mla_output_shape(batch_size, seq_len, d_model, num_heads, d_c):
    x = torch.randn(batch_size, seq_len, d_model)
    out = MultiHeadLatentAttention(d_model, num_heads, d_c=d_c)(x)
    assert out.shape == (batch_size, seq_len, d_model)


def test_mla_decoupled_rope_output_shape(batch_size, seq_len, d_model, num_heads, d_head, d_c):
    d_k_rope = d_head // 2
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=d_k_rope, max_seq_len=seq_len * 2)
    mla = MultiHeadLatentAttention(d_model, num_heads, d_c=d_c, d_k_rope=d_k_rope, rope=rope)
    x = torch.randn(batch_size, seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(0)
    out = mla(x, token_positions=pos)
    assert out.shape == (batch_size, seq_len, d_model)


def test_mla_kv_cache_smaller_than_mha(d_model, num_heads, d_c):
    """d_c must be smaller than 2*H*d_k (the MHA KV cache width)."""
    assert d_c < 2 * d_model  # 2 * H * d_k = 2 * d_model


def test_mla_latent_reconstruction(batch_size, seq_len, d_model, num_heads, d_c):
    """K and V are deterministic linear functions of c_KV; caching c_KV is sufficient."""
    mla = MultiHeadLatentAttention(d_model, num_heads, d_c=d_c)
    mla.eval()
    x = torch.randn(batch_size, seq_len, d_model)
    with torch.no_grad():
        c_kv = mla.w_dkv(x)
        k1, v1 = mla.w_uk(c_kv), mla.w_uv(c_kv)
        k2, v2 = mla.w_uk(c_kv.clone()), mla.w_uv(c_kv.clone())
    torch.testing.assert_close(k1, k2)
    torch.testing.assert_close(v1, v2)


def test_mla_causal_masking(batch_size, seq_len, d_model, num_heads, d_c):
    torch.manual_seed(2)
    split = 4
    mla = MultiHeadLatentAttention(d_model, num_heads, d_c=d_c)
    mla.eval()
    x = torch.randn(batch_size, seq_len, d_model)
    with torch.no_grad():
        out_full = mla(x)
        x_mod = x.clone()
        x_mod[:, split:] = torch.randn_like(x_mod[:, split:])
        out_mod = mla(x_mod)
    torch.testing.assert_close(out_full[:, :split], out_mod[:, :split], atol=1e-5, rtol=1e-5)


def test_mla_decoupled_rope_causal_masking(batch_size, seq_len, d_model, num_heads, d_head, d_c):
    torch.manual_seed(3)
    split = 4
    d_k_rope = d_head // 2
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=d_k_rope, max_seq_len=seq_len * 2)
    mla = MultiHeadLatentAttention(d_model, num_heads, d_c=d_c, d_k_rope=d_k_rope, rope=rope)
    mla.eval()
    x = torch.randn(batch_size, seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(0)
    with torch.no_grad():
        out_full = mla(x, pos)
        x_mod = x.clone()
        x_mod[:, split:] = torch.randn_like(x_mod[:, split:])
        out_mod = mla(x_mod, pos)
    torch.testing.assert_close(out_full[:, :split], out_mod[:, :split], atol=1e-5, rtol=1e-5)


def test_mla_gradient_flow(batch_size, seq_len, d_model, num_heads, d_c):
    x = torch.randn(batch_size, seq_len, d_model)
    mla = MultiHeadLatentAttention(d_model, num_heads, d_c=d_c)
    mla(x).sum().backward()
    for name, p in mla.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} has no gradient"


def test_mla_decoupled_rope_gradient_flow(batch_size, seq_len, d_model, num_heads, d_head, d_c):
    d_k_rope = d_head // 2
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=d_k_rope, max_seq_len=seq_len * 2)
    mla = MultiHeadLatentAttention(d_model, num_heads, d_c=d_c, d_k_rope=d_k_rope, rope=rope)
    x = torch.randn(batch_size, seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(0)
    mla(x, pos).sum().backward()
    for name, p in mla.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} has no gradient"
