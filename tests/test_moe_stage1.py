"""
Stage 1 unit tests for the sparse MoE layer (NO load balancing).

MoELayer.forward(x) -> Tensor   # only the shared+routed sum, no aux loss.

Run:
    uv run pytest tests/test_moe_stage1.py -v
"""

import torch
import pytest

try:
    from cs336_basics.model.moe import MoELayer
    HAS_MOE = True
except ImportError:
    HAS_MOE = False

pytestmark = pytest.mark.skipif(not HAS_MOE, reason="MoELayer not implemented")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_size(): return 2

@pytest.fixture
def seq_len(): return 6

@pytest.fixture
def d_model(): return 32

@pytest.fixture
def d_ffn(): return 64

@pytest.fixture
def n_shared(): return 2

@pytest.fixture
def n_routed(): return 8

@pytest.fixture
def top_k(): return 3


# ---------------------------------------------------------------------------
# 1. Output shape — Stage 1 returns ONE tensor (no aux loss)
# ---------------------------------------------------------------------------

def test_output_shape(batch_size, seq_len, d_model, d_ffn, n_shared, n_routed, top_k):
    torch.manual_seed(0)
    model = MoELayer(d_model, d_ffn, n_shared, n_routed, top_k)
    x = torch.randn(batch_size, seq_len, d_model)
    out = model(x)
    assert isinstance(out, torch.Tensor), (
        "Stage 1 forward must return a single Tensor (no aux loss yet)"
    )
    assert out.shape == (batch_size, seq_len, d_model), (
        f"Expected {(batch_size, seq_len, d_model)}, got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 2. Shared-only (n_routed=0)
# ---------------------------------------------------------------------------

def test_shared_only(batch_size, seq_len, d_model, d_ffn):
    model = MoELayer(d_model, d_ffn, n_shared=2, n_routed=0, top_k=0)
    x = torch.randn(batch_size, seq_len, d_model)
    out = model(x)
    assert out.shape == (batch_size, seq_len, d_model)


# ---------------------------------------------------------------------------
# 3. Routed-only (n_shared=0)
# ---------------------------------------------------------------------------

def test_routed_only(batch_size, seq_len, d_model, d_ffn, n_routed, top_k):
    model = MoELayer(d_model, d_ffn, n_shared=0, n_routed=n_routed, top_k=top_k)
    x = torch.randn(batch_size, seq_len, d_model)
    out = model(x)
    assert out.shape == (batch_size, seq_len, d_model)


# ---------------------------------------------------------------------------
# 4. Router indices: shape [T, top_k], values in [0, n_routed)
# ---------------------------------------------------------------------------

def test_router_indices_shape_and_range(batch_size, seq_len, d_model, d_ffn, n_shared, n_routed, top_k):
    model = MoELayer(d_model, d_ffn, n_shared, n_routed, top_k)
    x = torch.randn(batch_size, seq_len, d_model)
    model(x)

    assert hasattr(model, "last_router_indices"), (
        "MoELayer must expose self.last_router_indices after forward"
    )
    idx = model.last_router_indices
    T = batch_size * seq_len
    assert idx.shape == (T, top_k), f"Expected ({T}, {top_k}), got {tuple(idx.shape)}"
    assert idx.dtype in (torch.int64, torch.long), f"indices must be int64, got {idx.dtype}"
    assert idx.min() >= 0 and idx.max() < n_routed, (
        f"indices must be in [0, {n_routed}), got [{idx.min()}, {idx.max()}]"
    )


# ---------------------------------------------------------------------------
# 5. Router probs: shape [T, n_routed], each row is a valid distribution
# ---------------------------------------------------------------------------

def test_router_probs_distribution(batch_size, seq_len, d_model, d_ffn, n_shared, n_routed, top_k):
    model = MoELayer(d_model, d_ffn, n_shared, n_routed, top_k)
    x = torch.randn(batch_size, seq_len, d_model)
    model(x)

    assert hasattr(model, "last_router_probs"), (
        "MoELayer must expose self.last_router_probs after forward"
    )
    probs = model.last_router_probs
    T = batch_size * seq_len
    assert probs.shape == (T, n_routed), f"Expected ({T}, {n_routed}), got {tuple(probs.shape)}"
    assert (probs >= 0).all(), "softmax probabilities must be non-negative"
    row_sums = probs.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones(T), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 6. top_k == n_routed → all experts selected for every token
# ---------------------------------------------------------------------------

def test_all_experts_when_topk_equals_n_routed(batch_size, seq_len, d_model, d_ffn):
    n_routed = 4
    model = MoELayer(d_model, d_ffn, n_shared=1, n_routed=n_routed, top_k=n_routed)
    x = torch.randn(batch_size, seq_len, d_model)
    model(x)
    for row in model.last_router_indices:
        assert set(row.tolist()) == set(range(n_routed)), (
            f"top_k==n_routed must select all experts; got {sorted(row.tolist())}"
        )


# ---------------------------------------------------------------------------
# 7. Deterministic — routing is argmax, not sampled
# ---------------------------------------------------------------------------

def test_deterministic(batch_size, seq_len, d_model, d_ffn, n_shared, n_routed, top_k):
    model = MoELayer(d_model, d_ffn, n_shared, n_routed, top_k)
    model.eval()
    x = torch.randn(batch_size, seq_len, d_model)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0,
        msg="MoELayer forward is non-deterministic")


# ---------------------------------------------------------------------------
# 8. Gradient flow — every parameter gets a gradient
# ---------------------------------------------------------------------------

def test_gradient_flow(batch_size, seq_len, d_model, d_ffn, n_shared, n_routed, top_k):
    """
    Use a large batch so that every routed expert is selected by at least one token
    (an unselected expert legitimately has zero gradient — we skip those).

    Relies on the naming convention from the guide:
        self.router          : Linear(d_model, n_routed)
        self.shared_experts  : nn.ModuleList of SwiGLU
        self.routed_experts  : nn.ModuleList of SwiGLU
    """
    torch.manual_seed(1)
    big_batch, big_seq = 8, 16   # T = 128 tokens, top_k=3 → ~384 selections over 8 experts
    model = MoELayer(d_model, d_ffn, n_shared, n_routed, top_k)
    x = torch.randn(big_batch, big_seq, d_model)
    model(x).sum().backward()

    selected = set(model.last_router_indices.flatten().tolist())

    # router + shared experts must always have gradient
    assert model.router.weight.grad is not None and model.router.weight.grad.abs().sum() > 0, \
        "router has no gradient"
    for e in model.shared_experts:
        for n, p in e.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, f"shared expert {n}: no gradient"

    # routed experts: only the ones actually selected must have gradient
    for i, e in enumerate(model.routed_experts):
        if i not in selected:
            continue
        for n, p in e.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, f"routed expert {i}.{n}: no gradient"


# ---------------------------------------------------------------------------
# 9. Sparsity — mutating an unselected expert must not change the output
# ---------------------------------------------------------------------------

def test_unselected_expert_does_not_affect_output(batch_size, seq_len, d_model, d_ffn):
    """
    With top_k small and n_routed large, find an expert that NO token selected,
    perturb its weights, and confirm the output is unchanged (routing is truly sparse).
    """
    torch.manual_seed(5)
    n_routed, top_k = 16, 1
    model = MoELayer(d_model, d_ffn, n_shared=1, n_routed=n_routed, top_k=top_k)
    model.eval()
    x = torch.randn(batch_size, seq_len, d_model)

    with torch.no_grad():
        out_before = model(x)
        selected = set(model.last_router_indices.flatten().tolist())
        unselected = [e for e in range(n_routed) if e not in selected]
        if not unselected:
            pytest.skip("no unselected expert in this configuration")
        victim = unselected[0]

        # Perturb every parameter of the unselected routed expert.
        for p in model.routed_experts[victim].parameters():
            p.add_(torch.randn_like(p) * 10.0)

        out_after = model(x)

    torch.testing.assert_close(out_before, out_after, atol=1e-6, rtol=1e-6,
        msg="Perturbing an unselected expert changed the output — routing is not sparse")
