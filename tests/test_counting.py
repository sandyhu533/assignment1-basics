"""Unit tests for cs336_basics.practice.math — transformer counting practice.

All counter functions return a dict with a 'total' key plus per-module breakdown
keys, so a failing test points at the specific sub-module whose formula is wrong.

Two configurations:
  A: tiny, fully hand-computable (V=10, T=8, L=1, d=4, H=2, d_ff=12, B=1)
  B: small sanity (V=1000, T=128, L=4, d=128, H=4, d_ff=512, B=2)
"""

import importlib

import pytest

_mod = importlib.import_module("cs336_basics.practice.math")


def _get(name):
    """Return function `name` from math.py, or skip the test if not implemented yet."""
    fn = getattr(_mod, name, None)
    if fn is None:
        pytest.skip(f"{name} not implemented yet")
    return fn


def round_d_ff(*a, **k):                  return _get("round_d_ff")(*a, **k)
def transformer_params(*a, **k):          return _get("transformer_params")(*a, **k)
def transformer_forward_flops(*a, **k):   return _get("transformer_forward_flops")(*a, **k)
def activation_elements(*a, **k):         return _get("activation_elements")(*a, **k)
def adamw_peak_memory_bytes(*a, **k):     return _get("adamw_peak_memory_bytes")(*a, **k)
def adamw_step_flops(*a, **k):            return _get("adamw_step_flops")(*a, **k)


# ---------------------------------------------------------------------------
# Config A — tiny, hand-computable
# ---------------------------------------------------------------------------
CFG_A = dict(
    vocab_size=10,
    context_length=8,
    num_layers=1,
    d_model=4,
    num_heads=2,
    d_ff=12,
)
BATCH_A = 1

CFG_A_PARAMS_KW = dict(
    vocab_size=CFG_A["vocab_size"],
    num_layers=CFG_A["num_layers"],
    d_model=CFG_A["d_model"],
    num_heads=CFG_A["num_heads"],
    d_ff=CFG_A["d_ff"],
)


def test_params_tiny():
    n = transformer_params(**CFG_A_PARAMS_KW)
    assert n["token_embedding"] == 40
    assert n["attention"] == 64          # L · 4 · d² = 1·4·16
    assert n["ffn"] == 144               # L · 3 · d · d_ff = 1·3·48
    assert n["norms"] == 12              # (2L + 1) · d = 3·4
    assert n["lm_head"] == 40
    assert n["total"] == 300


def test_forward_flops_tiny():
    f = transformer_forward_flops(batch_size=BATCH_A, **CFG_A)
    assert f["attention_qkvo"] == 1024        # L · 4 · 2·B·T·d² = 1·4·256
    assert f["attention_scores"] == 512       # L · 2·B·T²·d
    assert f["attention_softmax_v"] == 512    # L · 2·B·T²·d
    assert f["ffn"] == 2304                   # L · 3 · 2·B·T·d·d_ff
    assert f["lm_head"] == 640                # 2·B·T·d·V
    assert f["total"] == 4992


def test_activations_tiny():
    a = activation_elements(batch_size=BATCH_A, **CFG_A)
    assert a["attention"] == 160          # 5·B·T·d·L: x, q, k, v, attn_out
    assert a["attention_T2"] == 384       # 3·B·L·H·T² (manual softmax: amax-input + xexp + softmax-out)
    assert a["ffn"] == 512                # 5·B·T·d_ff·L + B·T·d·L (w1x, sig, gate, w3x, product + x input)
    assert a["norms"] == 192              # 2·(2L+1)·B·T·d (manual RMSNorm saves x + x/rms per norm)
    assert a["lm_head"] == 32             # B·T·d (lm_head matmul saves its input — ln_final's output)
    assert a["ce"] == 160                 # 2·B·T·V (manual CE saves logits + logits.exp)
    assert a["total"] == 1440


def test_adamw_memory_tiny():
    mem = adamw_peak_memory_bytes(batch_size=BATCH_A, **CFG_A)
    assert mem["params"] == 1200          # 4 · 300
    assert mem["gradients"] == 1200       # 4 · 300
    assert mem["optimizer"] == 2400       # 2 · 4 · 300  (m + v)
    assert mem["activations"] == 5760     # 4 · 1440
    assert mem["total"] == 10560


def test_adamw_step_flops_tiny():
    # N = 300; per-param ops from CS336 Algorithm 1 lines 8-11.
    f = adamw_step_flops(**CFG_A_PARAMS_KW)
    assert f["weight_decay"] == 600       # 2N: (αλ)·θ + θ - (αλ)·θ
    assert f["m_update"] == 900           # 3N: β₁·m + (1-β₁)·g (2 mul + 1 add)
    assert f["v_update"] == 1200          # 4N: g² + β₂·v + (1-β₂)·g² + add
    assert f["param_update"] == 1500      # 5N: √v + +ε + m/(√v+ε) + ·α_t + θ -
    assert f["total"] == 4200             # 14N


# ---------------------------------------------------------------------------
# Config B — small but non-trivial
# ---------------------------------------------------------------------------
CFG_B = dict(
    vocab_size=1000,
    context_length=128,
    num_layers=4,
    d_model=128,
    num_heads=4,
    d_ff=512,
)
BATCH_B = 2

CFG_B_PARAMS_KW = dict(
    vocab_size=CFG_B["vocab_size"],
    num_layers=CFG_B["num_layers"],
    d_model=CFG_B["d_model"],
    num_heads=CFG_B["num_heads"],
    d_ff=CFG_B["d_ff"],
)


def test_params_small():
    n = transformer_params(**CFG_B_PARAMS_KW)
    assert n["token_embedding"] == 128_000
    assert n["attention"] == 262_144      # 4 · 4 · 128²
    assert n["ffn"] == 786_432            # 4 · 3 · 128 · 512
    assert n["norms"] == 1_152            # (2·4 + 1) · 128
    assert n["lm_head"] == 128_000
    assert n["total"] == 1_305_728


def test_forward_flops_small():
    f = transformer_forward_flops(batch_size=BATCH_B, **CFG_B)
    assert f["attention_qkvo"] == 134_217_728
    assert f["attention_scores"] == 33_554_432
    assert f["attention_softmax_v"] == 33_554_432
    assert f["ffn"] == 402_653_184
    assert f["lm_head"] == 65_536_000
    assert f["total"] == 669_515_776


def test_activations_small():
    a = activation_elements(batch_size=BATCH_B, **CFG_B)
    assert a["attention"] == 655_360         # 5·B·T·d·L
    assert a["attention_T2"] == 1_572_864    # 3·B·L·H·T² (manual softmax)
    assert a["ffn"] == 2_752_512             # 5·B·T·d_ff·L + B·T·d·L
    assert a["norms"] == 589_824             # 2·(2L+1)·B·T·d (manual RMSNorm)
    assert a["lm_head"] == 32_768            # B·T·d (lm_head matmul saves its input)
    assert a["ce"] == 512_000                # 2·B·T·V (manual CE saves logits + logits.exp)
    assert a["total"] == 6_115_328


def test_adamw_memory_small():
    mem = adamw_peak_memory_bytes(batch_size=BATCH_B, **CFG_B)
    assert mem["params"] == 5_222_912
    assert mem["gradients"] == 5_222_912
    assert mem["optimizer"] == 10_445_824
    assert mem["activations"] == 24_461_312   # 4 · 6_115_328
    assert mem["total"] == 45_352_960


def test_adamw_step_flops_small():
    # N = 1_305_728
    f = adamw_step_flops(**CFG_B_PARAMS_KW)
    assert f["weight_decay"] == 2_611_456
    assert f["m_update"] == 3_917_184
    assert f["v_update"] == 5_222_912
    assert f["param_update"] == 6_528_640
    assert f["total"] == 18_280_192


# ---------------------------------------------------------------------------
# Self-consistency: every dict's 'total' must equal the sum of its other values
# ---------------------------------------------------------------------------
def _sum_non_total(d):
    return sum(v for k, v in d.items() if k != "total")


def test_params_dict_total_equals_sum():
    for kw in [CFG_A_PARAMS_KW, CFG_B_PARAMS_KW]:
        d = transformer_params(**kw)
        assert d["total"] == _sum_non_total(d)


def test_forward_flops_dict_total_equals_sum():
    for cfg, b in [(CFG_A, BATCH_A), (CFG_B, BATCH_B)]:
        d = transformer_forward_flops(batch_size=b, **cfg)
        assert d["total"] == _sum_non_total(d)


def test_activations_dict_total_equals_sum():
    for cfg, b in [(CFG_A, BATCH_A), (CFG_B, BATCH_B)]:
        d = activation_elements(batch_size=b, **cfg)
        assert d["total"] == _sum_non_total(d)


# ---------------------------------------------------------------------------
# Sanity / invariants
# ---------------------------------------------------------------------------
def test_round_d_ff_is_multiple_of_64():
    for d in [128, 512, 768, 1024, 4096, 8192]:
        assert round_d_ff(d) % 64 == 0, f"d_model={d}: d_ff={round_d_ff(d)} not multiple of 64"


def test_round_d_ff_close_to_8_over_3():
    for d in [128, 512, 768, 1024, 4096]:
        assert abs(round_d_ff(d) - (8 / 3) * d) <= 64


def test_default_d_ff_used_when_none():
    kwargs = dict(vocab_size=100, num_layers=2, d_model=128, num_heads=4)
    n_default = transformer_params(**kwargs)
    n_explicit = transformer_params(**kwargs, d_ff=round_d_ff(128))
    assert n_default == n_explicit


def test_params_scale_linearly_in_layers():
    # Adding a layer should add exactly per_block params (norms + attn + ffn).
    common = dict(vocab_size=100, d_model=64, num_heads=4, d_ff=128)
    n1 = transformer_params(num_layers=1, **common)["total"]
    n2 = transformer_params(num_layers=2, **common)["total"]
    n5 = transformer_params(num_layers=5, **common)["total"]
    per_block = n2 - n1
    assert n5 - n1 == 4 * per_block


def test_flops_scale_linearly_in_batch():
    f1 = transformer_forward_flops(batch_size=1, **CFG_A)["total"]
    f4 = transformer_forward_flops(batch_size=4, **CFG_A)["total"]
    assert f4 == 4 * f1


def test_flops_quadratic_in_context_for_attention():
    cfg_t8 = dict(CFG_A); cfg_t8["context_length"] = 8
    cfg_t16 = dict(CFG_A); cfg_t16["context_length"] = 16
    f8 = transformer_forward_flops(batch_size=1, **cfg_t8)["total"]
    f16 = transformer_forward_flops(batch_size=1, **cfg_t16)["total"]
    assert 2 * f8 < f16 < 4 * f8, f"f8={f8}, f16={f16} — attention should be super-linear in T"


def test_adamw_optimizer_is_2x_params():
    mem = adamw_peak_memory_bytes(batch_size=BATCH_A, **CFG_A)
    assert mem["optimizer"] == 2 * mem["params"]


def test_adamw_total_equals_sum():
    mem = adamw_peak_memory_bytes(batch_size=BATCH_A, **CFG_A)
    assert mem["total"] == mem["params"] + mem["gradients"] + mem["optimizer"] + mem["activations"]


def test_adamw_step_flops_equals_14N():
    # All four buckets sum to 14·N (2 + 3 + 4 + 5).
    for kw in [CFG_A_PARAMS_KW, CFG_B_PARAMS_KW]:
        N = transformer_params(**kw)["total"]
        f = adamw_step_flops(**kw)
        assert f["total"] == 14 * N, f"AdamW step total should be 14·N, got {f['total']} vs 14·{N}={14*N}"


def test_adamw_step_flops_dict_total_equals_sum():
    for kw in [CFG_A_PARAMS_KW, CFG_B_PARAMS_KW]:
        f = adamw_step_flops(**kw)
        assert f["total"] == _sum_non_total(f)


def test_adamw_step_flops_scales_with_N():
    # Doubling num_layers grows N by per-block params; AdamW step FLOPs must scale identically.
    common = dict(vocab_size=100, d_model=64, num_heads=4, d_ff=128)
    n1 = transformer_params(num_layers=1, **common)["total"]
    n3 = transformer_params(num_layers=3, **common)["total"]
    f1 = adamw_step_flops(num_layers=1, **common)["total"]
    f3 = adamw_step_flops(num_layers=3, **common)["total"]
    assert f3 / f1 == n3 / n1


def test_bf16_halves_param_memory():
    fp32 = adamw_peak_memory_bytes(batch_size=BATCH_A, bytes_per_element=4, **CFG_A)
    bf16 = adamw_peak_memory_bytes(batch_size=BATCH_A, bytes_per_element=2, **CFG_A)
    assert bf16["params"] == fp32["params"] // 2
    assert bf16["total"] == fp32["total"] // 2


# ===========================================================================
# Insight tests — derive from the breakdown dicts directly (no extra functions)
# ===========================================================================
def test_attention_flops_grow_quadratically_in_T():
    # attention_scores + attention_softmax_v ∝ T²; all other terms ∝ T.
    cfg_t8 = dict(CFG_A);  cfg_t8["context_length"] = 8
    cfg_t16 = dict(CFG_A); cfg_t16["context_length"] = 16
    f8 = transformer_forward_flops(batch_size=1, **cfg_t8)
    f16 = transformer_forward_flops(batch_size=1, **cfg_t16)
    attn_t2_8 = f8["attention_scores"] + f8["attention_softmax_v"]
    attn_t2_16 = f16["attention_scores"] + f16["attention_softmax_v"]
    assert attn_t2_16 == 4 * attn_t2_8
    # Param-matmul FLOPs scale linearly
    param_8 = f8["attention_qkvo"] + f8["ffn"] + f8["lm_head"]
    param_16 = f16["attention_qkvo"] + f16["ffn"] + f16["lm_head"]
    assert param_16 == 2 * param_8


def test_2ND_approximation():
    # 2·N·D ≈ attention_qkvo + ffn + lm_head (all Linear-layer matmuls).
    # Gap = 2·D·(V·d for embedding + norm_params), since those add to N but contribute 0 FLOPs.
    N = transformer_params(**CFG_B_PARAMS_KW)["total"]
    D = BATCH_B * CFG_B["context_length"]
    approx = 2 * N * D
    f = transformer_forward_flops(batch_size=BATCH_B, **CFG_B)
    linear_matmuls = f["attention_qkvo"] + f["ffn"] + f["lm_head"]
    gap = approx - linear_matmuls
    assert gap > 0
    assert gap / approx < 0.15, f"2ND approximation off by {gap/approx:.1%}"


def test_attention_T2_activation_dominates_at_long_context():
    # For Config B (d=128, L=4, H=4, V=1000), short context: T² < T-terms; long context: T² wins.
    cfg = dict(CFG_B); cfg["context_length"] = 256
    a = activation_elements(batch_size=1, **cfg)
    rest = a["attention"] + a["ffn"] + a["norms"] + a["lm_head"] + a["ce"]
    assert a["attention_T2"] < rest

    cfg = dict(CFG_B); cfg["context_length"] = 1024
    a = activation_elements(batch_size=1, **cfg)
    rest = a["attention"] + a["ffn"] + a["norms"] + a["lm_head"] + a["ce"]
    assert a["attention_T2"] > rest


def test_attention_T2_activation_scales_with_L_and_H():
    # attention_T2 = 2·B·L·H·T² → exactly proportional to L and H independently.
    base = activation_elements(batch_size=1, **CFG_B)
    cfg_2L = dict(CFG_B); cfg_2L["num_layers"] = CFG_B["num_layers"] * 2
    cfg_2H = dict(CFG_B); cfg_2H["num_heads"] = CFG_B["num_heads"] * 2
    assert activation_elements(batch_size=1, **cfg_2L)["attention_T2"] == 2 * base["attention_T2"]
    assert activation_elements(batch_size=1, **cfg_2H)["attention_T2"] == 2 * base["attention_T2"]
