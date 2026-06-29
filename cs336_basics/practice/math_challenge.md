# Transformer Counting Challenge

Fill in `cs336_basics/practice/math.py`.

## Run tests

```bash
# Run everything (stop on first failure)
uv run pytest tests/test_counting.py -x

# Run a single test by exact name
uv run pytest tests/test_counting.py::test_params_tiny -v

# Run all tests for one function (substring match)
uv run pytest tests/test_counting.py -k params -v          # transformer_params tests
uv run pytest tests/test_counting.py -k "flops and tiny" -v  # combine filters

# Run one exercise at a time (by test name prefix)
uv run pytest tests/test_counting.py -k "params or forward_flops or activations or adamw"  # ex 1
uv run pytest tests/test_counting.py -k "flops_linear or flops_attention or training_flops or 2ND"  # ex 2
uv run pytest tests/test_counting.py -k "activation_breakdown or quadratic"  # ex 3
uv run pytest tests/test_counting.py -k "adamw_step"  # AdamW step FLOPs

# Show full diff on assertion failure
uv run pytest tests/test_counting.py::test_params_tiny -vv
```

Suggested order: `round_d_ff` → `transformer_params` → `transformer_forward_flops` → `activation_elements` → `adamw_peak_memory_bytes` → `adamw_step_flops` → exercise 2 → exercise 3.

## Architecture (CS336 spec)

- Decoder-only causal LM, Pre-Norm with RMSNorm, SwiGLU FFN, MHA + RoPE
- No bias in Linear; no weight tying between `token_embedding` and `lm_head`
- `d_ff` defaults to `round_d_ff(d_model)` when not given

## Conventions

- **FLOPs**: matmul `A(m,k) @ B(k,n)` = `2·m·n·k`. Ignore elementwise (norms, SiLU, softmax, masks, residuals).
- **Activations saved per block**: ln1 in, q/k/v, QKᵀ scores, softmax, attn out, o_proj, ln2 in, W1, W3, SiLU·W3, W2.
  Outside: final RMSNorm out, LM head logits.
- **Memory**: fp32 = 4 bytes/elem default.

## Function signatures to implement

Every counter returns a **dict** so when the total mismatches you can see which sub-module is off.

```python
from typing import Optional

def round_d_ff(d_model: int) -> int: ...

# ---- Exercise 1: basics ----
def transformer_params(vocab_size, num_layers, d_model, num_heads, d_ff=None) -> dict: ...
#   keys: 'token_embedding', 'attention', 'ffn', 'norms', 'lm_head', 'total'
#         (each value = total across ALL layers, e.g. 'attention' = L·4·d²)

def transformer_forward_flops(batch_size, context_length, vocab_size,
                              num_layers, d_model, num_heads, d_ff=None) -> dict: ...
#   keys: 'attention_qkvo', 'attention_scores', 'attention_softmax_v',
#         'ffn', 'lm_head', 'total'

def activation_elements(batch_size, context_length, vocab_size,
                        num_layers, d_model, num_heads, d_ff=None) -> dict: ...
#   keys: 'attention', 'attention_T2', 'ffn', 'norms', 'logits', "lm_head", 'total'
#         (attention_T2 is the only T²-scaling bucket — split out on purpose so you can see
#          when it dominates everything else combined. 'norms' = all RMSNorm saved inputs,
#          ln1 + ln2 per block plus the final norm.)

def adamw_peak_memory_bytes(batch_size, context_length, vocab_size,
                            num_layers, d_model, num_heads,
                            d_ff=None, bytes_per_element=4) -> dict: ...
#   keys: 'params', 'gradients', 'optimizer', 'activations', 'total'

def adamw_step_flops(vocab_size, num_layers, d_model, num_heads, d_ff=None) -> dict: ...
#   keys: 'weight_decay', 'm_update', 'v_update', 'param_update', 'total'
#   Per the CS336 AdamW pseudocode: α_t is a scalar (line 7, ignore),
#   then 4 per-parameter steps (lines 8-11). Each value = ops across ALL params N.
#   No batch_size: optimizer step is per-param, independent of B and T.
```

## Insights the breakdown dicts let you derive (interview-relevant)

The detailed dicts make these computations trivial — practice reading them:

1. **2·N·D rule** covers only Linear-layer matmuls (those with learnable weights):
   `attention_qkvo + ffn + lm_head ≈ 2 · N · (B·T)`. Small gap because embedding contributes V·d to N but 0 FLOPs (it's a gather), and norms contribute O(d·L) to N but only elementwise FLOPs (uncounted).

2. **Full forward FLOPs = 2·N·D + attention T² overhead**:
   `attention_scores + attention_softmax_v` = 4·B·L·T²·d. These have **no learnable parameters**, so they fall outside the 2·N·D framework entirely. At short T they're negligible; at long T they dominate.

3. **Training ≈ 6·N·D**: backward ≈ 2× forward, so one step ≈ 3 × forward total (same caveat about T² overhead).

4. **Attention T² term**: `attention_scores` + `attention_softmax_v` scale as T², everything else as T. At long context this dominates → FlashAttention motivation.

4. **Activation T² term**: `attention_T2` = 2·B·L·H·T². Compare it to `attention + ffn + norms + logits` (the linear-in-T buckets) to find the crossover T.

5. **AdamW = 16 bytes/param** in fp32: `params + gradients + optimizer` always = 16·N bytes (4 each for w, g, m, v).

6. **AdamW step ≈ 14·N FLOPs** (per `adamw_step_flops`): tiny vs the ~6·N·D of fwd+bwd when B·T ≫ 1.
   The interesting ratio is **arithmetic intensity = 14·N FLOPs / 16·N bytes < 1 FLOP/byte** → optimizer step is **DRAM-bandwidth bound**, not compute bound.
   This is why ZeRO-1 (optimizer state sharding) is nearly free: you avoid 8·N bytes of m+v with negligible compute overhead.

## Test configurations (for sanity)

- **Tiny**: `V=10, T=8, L=1, d=4, H=2, d_ff=12, B=1` → hand-computable
- **Small**: `V=1000, T=128, L=4, d=128, H=4, d_ff=512, B=2`

Invariants the tests also check: params linear in L, FLOPs linear in B and super-linear in T, training = 3 × forward, attention_part scales as T², quadratic activations scale as L·H·T², bf16 halves memory.

## What this trains

- 2·N·D forward-FLOP rule (and why it diverges from real FLOPs at small N or long T)
- Where the T² activation term starts to dominate (FlashAttention motivation)
- 16-bytes-per-param AdamW memory intuition (fp32: params 4 + grad 4 + m 4 + v 4)
