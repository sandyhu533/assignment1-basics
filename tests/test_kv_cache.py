"""Tests for KV cache correctness in attention, transformer, and generation."""

import torch
import pytest

from cs336_basics.model import TransformerLM
from cs336_basics.model.layers import CausalMultiHeadSelfAttention
from cs336_basics.model.transformer import TransformerBlock
from cs336_basics.generation import generate, generate_with_kv_cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_config():
    return dict(
        vocab_size=128,
        context_length=64,
        d_model=32,
        num_layer=2,
        num_heads=4,
        d_ff=64,
        rope_theta=10000.0,
    )


@pytest.fixture
def model(model_config):
    torch.manual_seed(42)
    m = TransformerLM(**model_config)
    m.eval()
    return m


@pytest.fixture
def prompt():
    torch.manual_seed(0)
    return torch.randint(0, 128, (1, 8))


# ---------------------------------------------------------------------------
# Attention-level tests
# ---------------------------------------------------------------------------

class TestAttentionKVCache:
    """Verify that CausalMultiHeadSelfAttention produces identical results
    with and without KV cache, when called incrementally."""

    def test_prefill_matches_no_cache(self):
        """Prefill with use_cache=True should give the same output as without cache."""
        torch.manual_seed(7)
        d_model, num_heads, seq_len = 32, 4, 10
        mha = CausalMultiHeadSelfAttention.with_rope(d_model, num_heads, 64, 10000.0)
        mha.eval()
        x = torch.randn(2, seq_len, d_model)
        positions = torch.arange(seq_len).unsqueeze(0).expand(2, -1)

        out_no_cache = mha(x, positions)
        out_cache, kv = mha(x, positions, use_cache=True)

        torch.testing.assert_close(out_no_cache, out_cache, atol=1e-5, rtol=1e-5)

    def test_incremental_decode_matches_full(self):
        """Running token-by-token with KV cache should match a full-sequence forward."""
        torch.manual_seed(8)
        d_model, num_heads, total_len = 32, 4, 12
        mha = CausalMultiHeadSelfAttention.with_rope(d_model, num_heads, 64, 10000.0)
        mha.eval()
        x_full = torch.randn(1, total_len, d_model)
        positions_full = torch.arange(total_len).unsqueeze(0)

        out_full = mha(x_full, positions_full)

        # Incremental: prefill first 8, then decode 4 one at a time
        prefill_len = 8
        x_prefill = x_full[:, :prefill_len]
        pos_prefill = positions_full[:, :prefill_len]
        out_prefill, past_kv = mha(x_prefill, pos_prefill, use_cache=True)

        outputs = [out_prefill]
        kv = past_kv
        for t in range(prefill_len, total_len):
            x_t = x_full[:, t:t+1]
            pos_t = torch.tensor([[t]])
            out_t, kv = mha(x_t, pos_t, past_kv=kv, use_cache=True)
            outputs.append(out_t)

        out_incremental = torch.cat(outputs, dim=1)
        torch.testing.assert_close(out_full, out_incremental, atol=1e-5, rtol=1e-5)

    def test_cache_shapes(self):
        """KV cache tensors should have expected shapes."""
        torch.manual_seed(9)
        d_model, num_heads = 32, 4
        d_k = d_model // num_heads
        mha = CausalMultiHeadSelfAttention.with_rope(d_model, num_heads, 64, 10000.0)
        mha.eval()

        batch_size, seq_len = 2, 6
        x = torch.randn(batch_size, seq_len, d_model)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        _, (k_cache, v_cache) = mha(x, positions, use_cache=True)
        assert k_cache.shape == (batch_size, num_heads, seq_len, d_k)
        assert v_cache.shape == (batch_size, num_heads, seq_len, d_k)

        # After one decode step, seq dimension should grow by 1
        x_new = torch.randn(batch_size, 1, d_model)
        pos_new = torch.full((batch_size, 1), seq_len, dtype=torch.long)
        _, (k_cache2, v_cache2) = mha(x_new, pos_new, past_kv=(k_cache, v_cache), use_cache=True)
        assert k_cache2.shape == (batch_size, num_heads, seq_len + 1, d_k)
        assert v_cache2.shape == (batch_size, num_heads, seq_len + 1, d_k)


# ---------------------------------------------------------------------------
# TransformerBlock-level tests
# ---------------------------------------------------------------------------

class TestTransformerBlockKVCache:
    def test_prefill_matches_no_cache(self):
        torch.manual_seed(10)
        block = TransformerBlock(d_model=32, num_heads=4, d_ff=64, max_seq_len=64, theta=10000.0)
        block.eval()
        x = torch.randn(1, 10, 32)
        positions = torch.arange(10).unsqueeze(0)

        out_no_cache = block(x, token_positions=positions)
        out_cache, _ = block(x, token_positions=positions, use_cache=True)

        torch.testing.assert_close(out_no_cache, out_cache, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# TransformerLM-level tests
# ---------------------------------------------------------------------------

class TestTransformerLMKVCache:
    def test_prefill_logits_match(self, model, prompt):
        """Logits from prefill with cache should match logits without cache."""
        logits_no_cache = model(prompt)
        logits_cache, _ = model(prompt, use_cache=True)
        torch.testing.assert_close(logits_no_cache, logits_cache, atol=1e-5, rtol=1e-5)

    def test_incremental_logits_match(self, model):
        """Token-by-token decode with cache should match full-sequence forward."""
        torch.manual_seed(1)
        seq = torch.randint(0, 128, (1, 12))

        logits_full = model(seq)

        # Prefill on first 8 tokens
        prefill = seq[:, :8]
        logits_prefill, past_kv = model(prefill, use_cache=True)
        all_logits = [logits_prefill]

        # Decode remaining tokens one at a time
        for t in range(8, 12):
            token = seq[:, t:t+1]
            positions = torch.tensor([[t]])
            logits_t, past_kv = model(token, token_positions=positions, past_kv_list=past_kv, use_cache=True)
            all_logits.append(logits_t)

        logits_incremental = torch.cat(all_logits, dim=1)
        torch.testing.assert_close(logits_full, logits_incremental, atol=1e-4, rtol=1e-4)

    def test_backward_compatible(self, model, prompt):
        """Without use_cache, model should return plain logits tensor (not tuple)."""
        out = model(prompt)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 8, 128)


# ---------------------------------------------------------------------------
# Generation-level tests
# ---------------------------------------------------------------------------

class TestGenerateWithKVCache:
    def test_output_matches_naive(self, model, prompt):
        """With temperature=0, KV-cached generation should match naive generation."""
        out_naive = generate(model, prompt, max_new_tokens=20, temperature=0.0)
        out_cached = generate_with_kv_cache(model, prompt, max_new_tokens=20, temperature=0.0)
        torch.testing.assert_close(out_naive, out_cached)

    def test_output_matches_naive_longer(self, model):
        """Test with a longer generation to stress the cache."""
        torch.manual_seed(3)
        prompt = torch.randint(0, 128, (1, 4))
        out_naive = generate(model, prompt, max_new_tokens=40, temperature=0.0)
        out_cached = generate_with_kv_cache(model, prompt, max_new_tokens=40, temperature=0.0)
        torch.testing.assert_close(out_naive, out_cached)

    def test_1d_prompt(self, model):
        """Should handle 1D prompt (no batch dim) the same as naive."""
        torch.manual_seed(5)
        prompt = torch.randint(0, 128, (6,))
        out_naive = generate(model, prompt, max_new_tokens=10, temperature=0.0)
        out_cached = generate_with_kv_cache(model, prompt, max_new_tokens=10, temperature=0.0)
        torch.testing.assert_close(out_naive, out_cached)

    def test_eos_stops_early(self, model):
        """Generation should stop when EOS token is produced."""
        torch.manual_seed(99)
        prompt = torch.randint(0, 128, (1, 4))
        out = generate_with_kv_cache(model, prompt, max_new_tokens=100, temperature=0.0, eos_token_id=0)
        assert out.shape[-1] <= 104

    def test_output_length(self, model, prompt):
        """Generated output should have at most prompt_len + max_new_tokens tokens."""
        max_new = 15
        out = generate_with_kv_cache(model, prompt, max_new_tokens=max_new, temperature=0.0)
        assert out.shape[-1] <= prompt.shape[-1] + max_new
