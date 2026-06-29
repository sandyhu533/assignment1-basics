"""
Stage 2 unit tests for the expert-level load-balance loss.

MoELayerBalanced(MoELayer).forward(x) -> (output, balance_loss)
  - output must be IDENTICAL to Stage-1 MoELayer given the same weights.
  - balance_loss = α · Σ_i f_i · P_i   (f_i detached, P_i differentiable).

Run:
    uv run pytest tests/test_moe_stage2.py -v
"""

import torch
import pytest

try:
    from cs336_basics.model.moe import MoELayer, MoELayerBalanced
    HAS_MOE = True
except ImportError:
    HAS_MOE = False

pytestmark = pytest.mark.skipif(not HAS_MOE, reason="MoELayerBalanced not implemented")


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


def _make_stage1_from_stage2(stage2: "MoELayerBalanced") -> MoELayer:
    """A bare Stage-1 layer sharing Stage-2's weights (for equivalence checks)."""
    stage1 = MoELayer(
        d_model=stage2.router.weight.shape[1],
        d_ffn=stage2.routed_experts[0].w1.weight.shape[0] if len(stage2.routed_experts) else 1,
        n_shared=len(stage2.shared_experts),
        n_routed=len(stage2.routed_experts),
        top_k=stage2.top_k,
    )
    stage1.load_state_dict(
        {k: v for k, v in stage2.state_dict().items() if k in stage1.state_dict()}
    )
    return stage1


# ---------------------------------------------------------------------------
# 1. Forward returns (output, loss)
# ---------------------------------------------------------------------------

def test_forward_returns_tuple(batch_size, seq_len, d_model, d_ffn, n_shared, n_routed, top_k):
    torch.manual_seed(0)
    model = MoELayerBalanced(d_model, d_ffn, n_shared, n_routed, top_k)
    x = torch.randn(batch_size, seq_len, d_model)
    result = model(x)
    assert isinstance(result, tuple) and len(result) == 2, (
        "Stage 2 forward must return a (output, balance_loss) tuple"
    )
    out, loss = result
    assert out.shape == (batch_size, seq_len, d_model)
    assert loss.shape == (), f"balance_loss must be a scalar, got shape {tuple(loss.shape)}"


# ---------------------------------------------------------------------------
# 2. EQUIVALENCE: output must match Stage-1 exactly (loss is auxiliary only)
# ---------------------------------------------------------------------------

def test_output_matches_stage1(batch_size, seq_len, d_model, d_ffn, n_shared, n_routed, top_k):
    """
    The balance loss is an auxiliary training signal; it must NOT alter the forward output.
    Stage 2's output must equal a Stage-1 layer with identical weights.
    """
    torch.manual_seed(42)
    stage2 = MoELayerBalanced(d_model, d_ffn, n_shared, n_routed, top_k)
    stage1 = _make_stage1_from_stage2(stage2)

    stage1.eval(); stage2.eval()
    x = torch.randn(batch_size, seq_len, d_model)
    with torch.no_grad():
        out1 = stage1(x)
        out2, _ = stage2(x)

    torch.testing.assert_close(out1, out2, atol=1e-6, rtol=1e-6,
        msg="Stage-2 output diverged from Stage-1 — the loss must not touch the forward path")


# ---------------------------------------------------------------------------
# 3. Balance loss is non-negative
# ---------------------------------------------------------------------------

def test_aux_loss_nonneg(batch_size, seq_len, d_model, d_ffn, n_shared, n_routed, top_k):
    for seed in range(5):
        torch.manual_seed(seed)
        model = MoELayerBalanced(d_model, d_ffn, n_shared, n_routed, top_k)
        x = torch.randn(batch_size, seq_len, d_model)
        _, loss = model(x)
        assert loss.item() >= 0.0, f"balance_loss must be ≥ 0, got {loss.item()}"


# ---------------------------------------------------------------------------
# 4. Perfect-balance value == α (normalization sanity)
# ---------------------------------------------------------------------------

def test_uniform_router_loss_equals_alpha(d_model, d_ffn, n_routed, top_k):
    """
    Sanity-check the f_i normalization factor N_r/(K_r·T).

    With a zero router (all logits equal) and constant input, P_i = 1/N_r for every
    expert, and the top-K picks some fixed K experts each with f_i = N_r/K_r. Then
        L/α = Σ_i f_i·P_i = K · (N_r/K_r)·(1/N_r) = K/K_r = 1   (since K == K_r)
    so the loss equals exactly α. If you get N_r·α or α/N_r, your f_i factor is wrong.
    """
    alpha = 0.01
    B, L = 4, 8
    model = MoELayerBalanced(d_model, d_ffn, n_shared=1, n_routed=n_routed, top_k=top_k,
                             balance_coeff=alpha)
    with torch.no_grad():
        model.router.weight.zero_()          # uniform softmax over experts
    model.eval()
    x = torch.ones(B, L, d_model)            # constant input → identical routing per token
    with torch.no_grad():
        _, loss = model(x)
    torch.testing.assert_close(loss, torch.tensor(alpha), atol=1e-6, rtol=1e-5,
        msg=f"uniform-router balance loss should equal α={alpha}, got {loss.item()} "
            f"(check the N_r/(K_r·T) normalization on f_i)")


# ---------------------------------------------------------------------------
# 5. Loss is zero when there are no routed experts
# ---------------------------------------------------------------------------

def test_aux_loss_zero_without_routed(batch_size, seq_len, d_model, d_ffn):
    model = MoELayerBalanced(d_model, d_ffn, n_shared=2, n_routed=0, top_k=0)
    x = torch.randn(batch_size, seq_len, d_model)
    _, loss = model(x)
    assert loss.item() == 0.0, f"loss must be 0 when n_routed=0, got {loss.item()}"


# ---------------------------------------------------------------------------
# 6. Balance loss back-propagates to the router (via P_i, not f_i)
# ---------------------------------------------------------------------------

def test_aux_loss_gradient_to_router(batch_size, seq_len, d_model, d_ffn, n_shared, n_routed, top_k):
    """
    P_i = mean softmax probability is differentiable, so loss.backward() alone must
    place a gradient on the router. (f_i comes from top-K and is detached.)
    """
    torch.manual_seed(2)
    model = MoELayerBalanced(d_model, d_ffn, n_shared, n_routed, top_k)
    x = torch.randn(batch_size, seq_len, d_model)
    _, loss = model(x)
    loss.backward()

    assert model.router.weight.grad is not None and model.router.weight.grad.abs().sum() > 0, (
        "balance loss must produce a gradient on self.router (through P_i)"
    )


# ---------------------------------------------------------------------------
# 7. Loss scales linearly with balance_coeff
# ---------------------------------------------------------------------------

def test_balance_loss_scales_with_coeff(batch_size, seq_len, d_model, d_ffn, n_shared, n_routed, top_k):
    torch.manual_seed(3)
    x = torch.randn(batch_size, seq_len, d_model)

    base = MoELayerBalanced(d_model, d_ffn, n_shared, n_routed, top_k, balance_coeff=0.001)
    dbl  = MoELayerBalanced(d_model, d_ffn, n_shared, n_routed, top_k, balance_coeff=0.002)
    dbl.load_state_dict(base.state_dict())

    base.eval(); dbl.eval()
    with torch.no_grad():
        _, l_base = base(x)
        _, l_dbl  = dbl(x)

    torch.testing.assert_close(l_dbl, l_base * 2, atol=1e-7, rtol=1e-5,
        msg=f"doubling balance_coeff should double the loss: {l_base.item()}×2 ≠ {l_dbl.item()}")


# ---------------------------------------------------------------------------
# 8. Uniform routing has lower balance loss than skewed routing
# ---------------------------------------------------------------------------

def test_balanced_routing_has_lower_loss(d_model, d_ffn, n_routed, top_k):
    """
    Drive the router toward a single expert vs. a uniform spread (by setting the
    router weights) and confirm the imbalanced case has strictly higher loss.

    Uniform: all P_i = 1/N_r, top-K hits K experts each with f_i = N_r/K_r → L = α.
    Skewed:  P concentrates on expert 0 (which is always selected) → Σ f_i P_i ≈ N_r/K_r → L ≈ (N_r/K_r)·α > α.

    Use a constant input so `logit = x · centroid` is deterministic and expert 0
    truly dominates for every token (with random x its sign would flip the affinity).
    """
    B, L = 4, 8
    x = torch.ones(B, L, d_model)

    # Uniform router: all centroids equal → uniform softmax.
    uniform = MoELayerBalanced(d_model, d_ffn, n_shared=1, n_routed=n_routed, top_k=top_k)
    with torch.no_grad():
        uniform.router.weight.zero_()               # all logits equal → uniform probs
    uniform.eval()

    # Skewed router: expert 0 dominates strongly for every token.
    skewed = MoELayerBalanced(d_model, d_ffn, n_shared=1, n_routed=n_routed, top_k=top_k)
    with torch.no_grad():
        skewed.router.weight.zero_()
        skewed.router.weight[0] = 100.0              # huge affinity for expert 0
    skewed.eval()

    with torch.no_grad():
        _, loss_uniform = uniform(x)
        _, loss_skewed = skewed(x)

    assert loss_skewed.item() > loss_uniform.item(), (
        f"skewed routing loss {loss_skewed.item()} should exceed uniform {loss_uniform.item()}"
    )
