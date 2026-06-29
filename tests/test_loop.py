"""Unit tests for cs336_basics.practice.loop — training + decode practice.

Lazy imports + pytest.skip: every test for an unimplemented function skips
instead of erroring, so you can work on functions in any order.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
import torch
from torch import Tensor, nn

_mod = importlib.import_module("cs336_basics.practice.loop")


def _get(name):
    fn = getattr(_mod, name, None)
    if fn is None:
        pytest.skip(f"{name} not implemented yet")
    return fn


def apply_temperature(*a, **k):       return _get("apply_temperature")(*a, **k)
def top_k_filter(*a, **k):            return _get("top_k_filter")(*a, **k)
def top_p_filter(*a, **k):            return _get("top_p_filter")(*a, **k)
def sample_next_token(*a, **k):       return _get("sample_next_token")(*a, **k)
def decode(*a, **k):                  return _get("decode")(*a, **k)
def cross_entropy_loss(*a, **k):      return _get("cross_entropy_loss")(*a, **k)
def gradient_l2_norm(*a, **k):        return _get("gradient_l2_norm")(*a, **k)
def clip_gradient_l2_norm(*a, **k):   return _get("clip_gradient_l2_norm")(*a, **k)
def cosine_lr_schedule(*a, **k):      return _get("cosine_lr_schedule")(*a, **k)
def get_batch(*a, **k):               return _get("get_batch")(*a, **k)
def save_checkpoint(*a, **k):         return _get("save_checkpoint")(*a, **k)
def load_checkpoint(*a, **k):         return _get("load_checkpoint")(*a, **k)
def train_step(*a, **k):              return _get("train_step")(*a, **k)


# Wrap any NotImplementedError in pytest.skip so partially-filled functions
# also skip cleanly instead of erroring.
@pytest.fixture(autouse=True)
def _skip_not_implemented():
    try:
        yield
    except NotImplementedError as e:
        pytest.skip(f"NotImplementedError: {e}")


# ============================================================
# Decoding primitives
# ============================================================

def test_apply_temperature_one_is_identity():
    logits = torch.randn(4, 10)
    out = apply_temperature(logits, 1.0)
    torch.testing.assert_close(out, logits)


def test_apply_temperature_scales():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = apply_temperature(logits, 2.0)
    torch.testing.assert_close(out, logits / 2.0)


def test_apply_temperature_zero_is_greedy():
    logits = torch.tensor([1.0, 5.0, 2.0, 3.0])
    out = apply_temperature(logits, 0.0)
    probs = torch.softmax(out.float(), dim=-1)
    assert probs.argmax().item() == 1
    assert probs[1].item() > 0.999


def test_top_k_filter_keeps_only_k():
    logits = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])
    out = top_k_filter(logits, 2)
    assert out[1].item() == 5.0
    assert out[3].item() == 4.0
    for i in [0, 2, 4]:
        assert out[i].item() == float("-inf")


def test_top_k_filter_batched():
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0],
                           [9.0, 1.0, 1.0, 1.0, 8.0]])
    out = top_k_filter(logits, 2)
    # row 0: keep idx 1, 3
    assert out[0, 1].item() == 5.0 and out[0, 3].item() == 4.0
    # row 1: keep idx 0, 4
    assert out[1, 0].item() == 9.0 and out[1, 4].item() == 8.0


def test_top_k_filter_none_is_identity():
    logits = torch.randn(8)
    torch.testing.assert_close(top_k_filter(logits, None), logits)


def test_top_p_filter_p_1_is_identity():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    torch.testing.assert_close(top_p_filter(logits, 1.0), logits)


def test_top_p_filter_keeps_only_top():
    # softmax([0, 10]) ≈ [4.5e-5, 0.999955] → p=0.5 should keep only idx 1
    logits = torch.tensor([0.0, 10.0])
    out = top_p_filter(logits, 0.5)
    assert out[0].item() == float("-inf")
    assert out[1].item() == 10.0


def test_top_p_filter_always_keeps_argmax():
    # Argmax alone exceeds p; the set must still include it (no empty set).
    logits = torch.tensor([0.0, 100.0, 0.0])
    out = top_p_filter(logits, 0.1)
    assert out[1].item() == 100.0
    assert torch.isfinite(out).any()


def test_sample_next_token_greedy_when_temp_zero():
    logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
    tok = sample_next_token(logits, temperature=0.0)
    assert tok.shape == (1,)
    assert tok.item() == 1


def test_sample_next_token_top_k_1_is_argmax():
    logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
    tok = sample_next_token(logits, temperature=1.0, top_k=1)
    assert tok.item() == 1


def test_sample_next_token_reproducible():
    torch.manual_seed(0)
    logits = torch.randn(4, 100)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    t1 = sample_next_token(logits, temperature=1.0, generator=g1)
    t2 = sample_next_token(logits, temperature=1.0, generator=g2)
    assert torch.equal(t1, t2)


def test_sample_next_token_distribution():
    # Sample many times from a known distribution; empirical freq should match.
    logits = torch.log(torch.tensor([[0.1, 0.6, 0.3]]))  # exact probs
    g = torch.Generator().manual_seed(0)
    samples = torch.stack([sample_next_token(logits, generator=g) for _ in range(5000)])
    samples = samples.flatten()
    freqs = torch.bincount(samples, minlength=3).float() / samples.numel()
    torch.testing.assert_close(freqs, torch.tensor([0.1, 0.6, 0.3]), atol=0.03, rtol=0)


# ============================================================
# Decode loop — uses a tiny mock model that always favors one token.
# ============================================================

class _ConstantLogitsModel(nn.Module):
    """Returns logits favoring `favored_token` at every position. shape: [B,T,V]."""
    def __init__(self, vocab_size: int, favored_token: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.favored_token = favored_token

    def forward(self, x: Tensor) -> Tensor:
        B, T = x.shape
        logits = torch.zeros(B, T, self.vocab_size)
        logits[..., self.favored_token] = 100.0
        return logits


def test_decode_greedy_appends_favored_token():
    model = _ConstantLogitsModel(vocab_size=10, favored_token=7)
    prompt = torch.tensor([[1, 2, 3]])
    out = decode(model, prompt, max_new_tokens=5, temperature=0.0)
    assert out.shape == (1, 8)
    assert torch.equal(out[0, :3], torch.tensor([1, 2, 3]))
    assert torch.equal(out[0, 3:], torch.tensor([7, 7, 7, 7, 7]))


def test_decode_respects_max_new_tokens():
    model = _ConstantLogitsModel(vocab_size=10, favored_token=4)
    prompt = torch.tensor([[1, 2]])
    out = decode(model, prompt, max_new_tokens=3, temperature=0.0)
    assert out.shape[1] == 5  # 2 prompt + 3 new


def test_decode_stops_on_eos():
    model = _ConstantLogitsModel(vocab_size=10, favored_token=5)
    prompt = torch.tensor([[1, 2]])
    out = decode(model, prompt, max_new_tokens=10, temperature=0.0, eos_id=5)
    # Should generate one token (5), include it, and stop.
    assert out.shape[1] == 3
    assert out[0, -1].item() == 5


# ============================================================
# Cross entropy
# ============================================================

def test_cross_entropy_matches_torch():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7)
    targets = torch.randint(0, 7, (2, 3))
    loss = cross_entropy_loss(logits, targets)
    ref = torch.nn.functional.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
    torch.testing.assert_close(loss, ref)


def test_cross_entropy_zero_when_perfect():
    # One-hot logits → loss ≈ 0
    logits = torch.tensor([[[-1e9, 1e9, -1e9, -1e9]]])  # argmax=1
    targets = torch.tensor([[1]])
    loss = cross_entropy_loss(logits, targets)
    assert loss.item() < 1e-3


def test_cross_entropy_stable_with_large_logits():
    # exp(1000) overflows; a stable implementation handles this.
    logits = torch.tensor([[[0.0, 1000.0, 0.0]]])
    targets = torch.tensor([[1]])
    loss = cross_entropy_loss(logits, targets)
    assert torch.isfinite(loss), "cross_entropy should be numerically stable (log-sum-exp)"
    assert loss.item() < 1e-3


def test_cross_entropy_uniform_logits_gives_log_vocab():
    # Uniform logits → loss = log(V)
    V = 32
    logits = torch.zeros(1, 1, V)
    targets = torch.zeros(1, 1, dtype=torch.long)
    loss = cross_entropy_loss(logits, targets)
    torch.testing.assert_close(loss, torch.tensor(math.log(V)), atol=1e-5, rtol=0)


def test_cross_entropy_backprops():
    logits = torch.randn(2, 3, 5, requires_grad=True)
    targets = torch.randint(0, 5, (2, 3))
    loss = cross_entropy_loss(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


# ============================================================
# Gradient norm + clip
# ============================================================

def test_gradient_l2_norm_known():
    p1 = nn.Parameter(torch.zeros(3))
    p2 = nn.Parameter(torch.zeros(4))
    p1.grad = torch.tensor([3.0, 0.0, 0.0])
    p2.grad = torch.tensor([0.0, 4.0, 0.0, 0.0])
    # global L2 = sqrt(9 + 16) = 5
    norm = gradient_l2_norm([p1, p2])
    assert math.isclose(float(norm), 5.0, abs_tol=1e-6)


def test_gradient_l2_norm_skips_none():
    p1 = nn.Parameter(torch.zeros(3))
    p2 = nn.Parameter(torch.zeros(3))
    p1.grad = torch.tensor([3.0, 4.0, 0.0])
    # p2.grad stays None
    norm = gradient_l2_norm([p1, p2])
    assert math.isclose(float(norm), 5.0, abs_tol=1e-6)


def test_clip_no_op_when_below_threshold():
    p = nn.Parameter(torch.zeros(3))
    p.grad = torch.tensor([0.6, 0.8, 0.0])  # norm=1
    pre = clip_gradient_l2_norm([p], max_norm=10.0)
    assert math.isclose(float(pre), 1.0, abs_tol=1e-5)
    torch.testing.assert_close(p.grad, torch.tensor([0.6, 0.8, 0.0]))


def test_clip_scales_when_above_threshold():
    p = nn.Parameter(torch.zeros(3))
    p.grad = torch.tensor([3.0, 4.0, 0.0])  # norm=5
    pre = clip_gradient_l2_norm([p], max_norm=1.0)
    assert math.isclose(float(pre), 5.0, abs_tol=1e-5)
    post = p.grad.norm()
    assert math.isclose(float(post), 1.0, abs_tol=1e-4)


def test_clip_returns_pre_clip_norm():
    # The return value should be the norm BEFORE scaling (for logging).
    p = nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([6.0, 8.0])  # norm=10
    pre = clip_gradient_l2_norm([p], max_norm=2.0)
    assert math.isclose(float(pre), 10.0, abs_tol=1e-5)


# ============================================================
# Cosine LR schedule
# ============================================================

def test_cosine_lr_at_warmup_end_is_max():
    lr = cosine_lr_schedule(step=10, warmup_steps=10, total_steps=100,
                            max_lr=1.0, min_lr=0.1)
    assert math.isclose(lr, 1.0, abs_tol=1e-9)


def test_cosine_lr_warmup_is_linear():
    lr_half = cosine_lr_schedule(step=5, warmup_steps=10, total_steps=100,
                                 max_lr=1.0, min_lr=0.1)
    assert math.isclose(lr_half, 0.5, abs_tol=1e-9)


def test_cosine_lr_decay_to_min():
    lr_end = cosine_lr_schedule(step=100, warmup_steps=10, total_steps=100,
                                max_lr=1.0, min_lr=0.1)
    assert math.isclose(lr_end, 0.1, abs_tol=1e-6)


def test_cosine_lr_past_end_stays_min():
    lr_past = cosine_lr_schedule(step=500, warmup_steps=10, total_steps=100,
                                 max_lr=1.0, min_lr=0.1)
    assert math.isclose(lr_past, 0.1, abs_tol=1e-6)


def test_cosine_lr_monotonic_decay():
    lrs = [cosine_lr_schedule(s, warmup_steps=10, total_steps=100,
                              max_lr=1.0, min_lr=0.1) for s in range(10, 101)]
    for i in range(len(lrs) - 1):
        assert lrs[i] >= lrs[i + 1] - 1e-9, f"non-monotonic at i={i}: {lrs[i]} -> {lrs[i+1]}"


def test_cosine_lr_midpoint():
    # Midpoint of cosine half-period: lr = (max + min) / 2
    lr_mid = cosine_lr_schedule(step=55, warmup_steps=10, total_steps=100,
                                max_lr=1.0, min_lr=0.0)
    assert math.isclose(lr_mid, 0.5, abs_tol=1e-6)


# ============================================================
# Data + checkpoint
# ============================================================

def test_get_batch_shapes_and_next_token_shift():
    data = np.arange(1000, dtype=np.int64)
    x, y = get_batch(data, batch_size=4, context_length=8, device="cpu")
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert x.dtype in (torch.int64, torch.long)
    # y is x shifted by +1 in the underlying token stream
    assert torch.equal(y, x + 1)


def test_get_batch_random_starts():
    data = np.arange(10000, dtype=np.int64)
    starts = {get_batch(data, 1, 4, "cpu")[0][0, 0].item() for _ in range(80)}
    assert len(starts) > 10, "starts should be sampled — got too few unique values"


def test_get_batch_in_bounds():
    data = np.arange(100, dtype=np.int64)
    for _ in range(20):
        x, y = get_batch(data, batch_size=2, context_length=16, device="cpu")
        assert x.max().item() < len(data)
        assert y.max().item() < len(data)


def test_save_load_checkpoint_roundtrip(tmp_path):
    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Run one optimizer step so optimizer.state is populated
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, step=42, path=str(path))

    model2 = nn.Linear(4, 3)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    loaded_step = load_checkpoint(str(path), model2, optimizer2)

    assert loaded_step == 42
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        torch.testing.assert_close(p1, p2)


# ============================================================
# Training step — orchestration
# ============================================================

class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 7, d: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.embed(x))


def test_train_step_returns_loss_and_grad_norm():
    torch.manual_seed(0)
    lm = _TinyLM()
    optimizer = torch.optim.AdamW(lm.parameters(), lr=1e-3)
    x = torch.randint(0, 7, (2, 4))
    y = torch.randint(0, 7, (2, 4))
    out = train_step(lm, (x, y), optimizer, max_grad_norm=1.0)
    assert "loss" in out and "grad_norm" in out
    assert out["loss"] > 0
    assert out["grad_norm"] >= 0


def test_train_step_loss_decreases_on_overfit():
    """Single fixed batch repeated → loss must drop substantially."""
    torch.manual_seed(0)
    lm = _TinyLM()
    optimizer = torch.optim.AdamW(lm.parameters(), lr=1e-2)
    x = torch.randint(0, 7, (1, 4))
    y = torch.randint(0, 7, (1, 4))
    first = train_step(lm, (x, y), optimizer)["loss"]
    for _ in range(100):
        train_step(lm, (x, y), optimizer)
    last = train_step(lm, (x, y), optimizer)["loss"]
    assert last < first * 0.5, f"overfitting failed: {first:.3f} -> {last:.3f}"


def test_train_step_zeros_grad_between_calls():
    """Two calls in a row shouldn't accumulate grads from the first into the second."""
    torch.manual_seed(0)
    lm = _TinyLM()
    optimizer = torch.optim.AdamW(lm.parameters(), lr=0.0)  # lr=0 → params unchanged
    x = torch.randint(0, 7, (2, 4))
    y = torch.randint(0, 7, (2, 4))
    g1 = train_step(lm, (x, y), optimizer)["grad_norm"]
    g2 = train_step(lm, (x, y), optimizer)["grad_norm"]
    # With lr=0 and same batch, if grads were accumulating, g2 would be ~2× g1.
    assert math.isclose(g1, g2, rel_tol=1e-3), f"grads appear to accumulate: {g1} vs {g2}"
