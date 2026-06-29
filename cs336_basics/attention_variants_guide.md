# 3.5 Attention Variants: MQA, GQA, and MLA

This section extends the multi-head self-attention (MHA) implementation from Section 3.4.4
by introducing three variants designed to reduce the **KV cache memory bottleneck** that
dominates transformer inference.

---

## Background: The KV Cache Bottleneck

During autoregressive decoding, each new token must attend to all previous tokens. To avoid
recomputing K and V projections at every step, systems cache them:

```
KV cache size per token = 2 × H × d_k × num_layers × dtype_bytes
```

For LLaMA-3-70B (H=64, d_k=128, L=80 layers, bf16): **≈ 2.6 MB per token**.
At a batch size of 128 with 4K context, the KV cache alone requires **≈ 1.3 TB** of GPU memory.

The variants in this section reduce that cost by reducing the number of KV heads (MQA/GQA)
or compressing KV into a low-rank latent vector (MLA).

All three variants **share the same output interface** as MHA:

```
forward(x: [B, L, d_model], token_positions=None) -> [B, L, d_model]
```

Implementations live in `cs336_basics/model/attention_variants.py`.

---

## 3.5.1 Multi-Query Attention (MQA)

### Principle

Multi-Query Attention [N. Shazeer, 2019] keeps H query heads but reduces to a **single
shared key and value head**. All H query heads attend against the same K and V projections.

This reduces the KV cache by a factor of H:

```
MHA KV cache: 2 × H × d_k  per token
MQA KV cache: 2 × 1 × d_k  per token   (H× savings)
```

The Q projection is unchanged; only W_K and W_V are shrunk.

### Formulation

Let x ∈ R^{d_model} be a single token embedding, H the number of heads, d_k = d_model / H.

```
Q_h = W_Q^h x    for h = 1, ..., H      (H query projections, each R^{d_k})
K   = W_K x                              (one shared key projection, R^{d_k})
V   = W_V x                              (one shared value projection, R^{d_k})

head_h = Attention(Q_h, K, V)
output  = W_O [ head_1 ; ... ; head_H ]
```

where Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.

In practice, Q is projected jointly as W_Q ∈ R^{H·d_k × d_model} and then split;
K and V use W_K, W_V ∈ R^{d_k × d_model} (single-head projections).

During SDPA, K and V are expanded (via `expand`, not `repeat_interleave`) to shape
[B, H, L, d_k] via broadcasting — no extra memory is allocated for the expanded view.

### Parameter Comparison


| Module | Q params | K params | V params | O params | Total        |
| ------ | -------- | -------- | -------- | -------- | ------------ |
| MHA    | d²       | d²       | d²       | d²       | 4d²          |
| MQA    | d²       | d·d_k    | d·d_k    | d²       | 2d² + 2d·d_k |


where d = d_model, d_k = d/H. The savings grow with H.

---

> **Problem (mqa): Implement Multi-Query Attention (MQA)**
>
> **Deliverable**: Implement `MultiQueryAttention` in `cs336_basics/model/attention_variants.py`.
>
> **Interface**:
>
> ```python
> def __init__(self, d_model: int, num_heads: int, rope=None, device=None)
> def forward(self, x: Tensor, token_positions=None) -> Tensor
> ```
>
> **Make sure to**:
>
> - Use a single K and V projection (shape `[d_k, d_model]` each, not `[H·d_k, d_model]`)
> - Expand K and V to `[B, H, L, d_k]` before calling `scaled_dot_product_attention`
> - Apply causal masking (lower-triangular `torch.tril` mask)
> - Optionally apply RoPE to both Q and K when `rope` is provided
>
> **Note**: Use `tensor.expand(...)` rather than `repeat_interleave` to broadcast K and V
> across heads — expand is a zero-copy view, while repeat_interleave allocates new memory.
>
> **Reference**: N. Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need," 2019.
>
> To test: `uv run pytest tests/test_mqa.py -v`

---

## 3.5.2 Grouped Query Attention (GQA)

### Principle

Grouped Query Attention [J. Ainslie et al., 2023] is a generalization that interpolates
between MHA and MQA. Instead of one shared KV head (MQA) or H independent KV heads (MHA),
GQA uses **G KV head groups** (1 ≤ G ≤ H, H % G == 0).

Each group of H/G query heads shares a single K/V head:

```
Query heads:   H (unchanged)
KV heads:      G
Group size:    H / G  (query heads per KV head)

MHA  = GQA(G = H)
MQA  = GQA(G = 1)
```

KV cache reduction vs MHA: **H/G×** (G=2 → 2×, G=4 → 4× savings).

LLaMA 3 (70B): H=64, G=8, group_size=8. Mistral 7B: H=32, G=8, group_size=4.

### Formulation

```
Q  ∈ R^{H·d_k}     split to [B, H, L, d_k]    (W_Q ∈ R^{H·d_k × d_model})
K  ∈ R^{G·d_k}     split to [B, G, L, d_k]    (W_K ∈ R^{G·d_k × d_model})
V  ∈ R^{G·d_k}     split to [B, G, L, d_k]    (W_V ∈ R^{G·d_k × d_model})
```

Before SDPA, K and V are tiled from G heads to H heads:

```python
K = K.repeat_interleave(H // G, dim=1)    # [B, G, L, d_k] -> [B, H, L, d_k]
V = V.repeat_interleave(H // G, dim=1)
```

This creates a new tensor (unlike `expand`) because each KV head's data is logically
distinct — `repeat_interleave` makes the tiling explicit for the matmul.

### Key Implementation Insight: Converting MHA Checkpoints to GQA

The [Ainslie et al., 2023] paper shows that MHA models can be **distilled** into GQA
by mean-pooling the H pre-trained KV heads into G groups. This lets you convert a trained
MHA model into a GQA model cheaply, retaining most quality with H/G× inference speedup.

---

> **Problem (gqa): Implement Grouped Query Attention (GQA)**
>
> **Deliverable**: Implement `GroupedQueryAttention` in `cs336_basics/model/attention_variants.py`.
>
> **Interface**:
>
> ```python
> def __init__(self, d_model: int, num_heads: int, num_kv_heads: int,
>              rope=None, device=None)
> def forward(self, x: Tensor, token_positions=None) -> Tensor
> ```
>
> **Make sure to**:
>
> - Project K and V to `num_kv_heads` heads (not `num_heads`)
> - Tile K and V to `num_heads` heads before SDPA via `repeat_interleave(groups, dim=1)`
> - Verify: `GQA(num_kv_heads=num_heads)` must produce identical output to MHA
> given the same weights (tested by `test_gqa_full_kv_heads_equals_mha`)
> - Verify: `GQA(num_kv_heads=1)` must produce identical output to MQA
> given the same weights (tested by `test_gqa_single_kv_head_equals_mqa`)
>
> **Reference**: J. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer
> Models from Multi-Head Checkpoints," EMNLP 2023.
>
> To test: `uv run pytest tests/test_gqa.py -v`

---

## 3.5.3 Multi-head Latent Attention (MLA)

### Principle

MLA [DeepSeek-AI, 2024] takes a different approach: instead of reducing the *number* of KV
heads, it **compresses K and V through a low-rank bottleneck**. The compressed latent vector
c_KV is what gets cached at inference, and K/V are reconstructed via learned up-projections.

The key insight is that if the rank of the KV information is low (i.e., d_c << 2·H·d_k),
we can cache d_c values instead of 2·H·d_k values per token — a substantial reduction.

```
KV cache memory per token:
    MHA: 2 × H × d_k  bytes (e.g., 2×128×128 = 32,768 for DeepSeek-V2)
    MLA: d_c           bytes (e.g., 512 for DeepSeek-V2)
    Savings: ~64×
```

### Formulation: Simplified MLA (KV Compression Only)

```
c_KV = W_DKV x    ∈ R^{d_c}          # down-project to KV latent
K    = W_UK  c_KV ∈ R^{H·d_k}        # up-project to K
V    = W_UV  c_KV ∈ R^{H·d_k}        # up-project to V

c_Q  = W_DQ  x    ∈ R^{d_c_q}        # down-project to Q latent
Q    = W_UQ  c_Q  ∈ R^{H·d_k}        # up-project to Q
```

At inference, only `c_KV ∈ R^{d_c}` needs to be stored per token. K and V are
reconstructed on the fly during attention.

### Formulation: Decoupled RoPE (Full MLA)

Standard RoPE is position-dependent and cannot be applied to the KV latent before caching
(position information would be baked in, breaking prefix reuse). MLA solves this by using
**decoupled RoPE**: a separate position-sensitive component that bypasses the latent bottleneck.

```
# Non-positional (nope) part — goes through the low-rank bottleneck
K_nope = W_UK @ c_KV     ∈ R^{H·d_k_nope}  per head
Q_nope = W_UQ @ c_Q      ∈ R^{H·d_k_nope}

# Positional (rope) part — NOT compressed, RoPE applied to these
k_rope = W_KR @ x        ∈ R^{d_k_rope}  shared across heads, cached separately
q_rope = W_QR @ c_Q      ∈ R^{H·d_k_rope}

RoPE(q_rope), RoPE(k_rope)

# Concatenate to form full Q and K (d_k = d_k_nope + d_k_rope)
K = [ K_nope ; k_rope.expand(H) ]
Q = [ Q_nope ; q_rope           ]
```

KV cache with decoupled RoPE: `d_c + d_k_rope` per token
(vs 2·H·d_k for MHA; for DeepSeek-V2: 512+64=576 vs 32,768 → 57×).

### Weight Matrix Summary


| Weight | Shape               | Role                                            |
| ------ | ------------------- | ----------------------------------------------- |
| W_DKV  | [d_c, d_model]      | KV down-projection (cached output)              |
| W_UK   | [H·d_k_nope, d_c]   | K up-projection                                 |
| W_UV   | [H·d_k, d_c]        | V up-projection (full d_k)                      |
| W_DQ   | [d_c_q, d_model]    | Q down-projection                               |
| W_UQ   | [H·d_k_nope, d_c_q] | Q nope up-projection                            |
| W_QR   | [H·d_k_rope, d_c_q] | Q rope projection (decoupled RoPE only)         |
| W_KR   | [d_k_rope, d_model] | K rope projection, shared (decoupled RoPE only) |
| W_O    | [d_model, d_model]  | Output projection                               |


### Interview Hook: Why Can't You Apply RoPE to c_KV?

If you naively apply RoPE to K = W_UK @ c_KV, the position encoding is baked into K before
caching. Then when a new query attends at a different position, the cached K has a stale
position baked in — you'd need to recompute K from scratch, defeating the cache.

The decoupled RoPE solves this by caching `c_KV` (position-free) and `k_rope` (the
position-sensitive component, which is small: only d_k_rope << H·d_k). During decoding:

1. Load `c_KV` from cache; reconstruct `K_nope = W_UK @ c_KV`.
2. Load `k_rope` from cache; apply RoPE for the current sequence offset.
3. Concatenate: `K = [K_nope; RoPE(k_rope)]`.

---

> **Problem (mla): Implement Multi-head Latent Attention (MLA)**
>
> **Deliverable**: Implement `MultiHeadLatentAttention` in `cs336_basics/model/attention_variants.py`.
>
> **Interface**:
>
> ```python
> def __init__(self, d_model: int, num_heads: int, d_c: int,
>              d_c_q: int = None,     # defaults to d_c
>              d_k_rope: int = None,  # None = no decoupled RoPE
>              rope = None,
>              device = None)
> def forward(self, x: Tensor, token_positions=None) -> Tensor
> ```
>
> **Simplified MLA** (d_k_rope=None):
>
> - W_DKV: [d_model → d_c], W_UK: [d_c → H·d_k], W_UV: [d_c → H·d_k]
> - W_DQ: [d_model → d_c_q], W_UQ: [d_c_q → H·d_k]
> - Optional: apply RoPE to full Q and K before SDPA
>
> **Full MLA** (d_k_rope > 0):
>
> - d_k_nope = (d_model // num_heads) - d_k_rope
> - W_UK: [d_c → H·d_k_nope], W_UQ: [d_c_q → H·d_k_nope]
> - W_UV: [d_c → H·d_k] (V uses full head dim, no rope portion)
> - W_QR: [d_c_q → H·d_k_rope], W_KR: [d_model → d_k_rope]
> - Apply RoPE to q_rope and k_rope only; concatenate nope+rope along last dim
>
> **Note**: The rope object passed in must be initialized with `d_k=d_k_rope`
> (not `d_k=d_model // num_heads`) when using decoupled RoPE.
>
> **Note**: V always has shape [B, H, L, d_k] (full head dim). Only K and Q use the
> nope/rope split. The output after SDPA is [B, L, H·d_k] = [B, L, d_model].
>
> **Reference**: DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-
> of-Experts Language Model," 2024. See Section 2.1 (MLA) and Section 2.1.2 (decoupled RoPE).
>
> To test: `uv run pytest tests/test_mla.py -v`

---

## 3.5.4 MLA Stage 2 — Weight Absorption

### Principle

The KV latent `c_KV` is stored in cache. During decode, the naive Stage-1 approach
**materializes** K_nope and V by applying W_UK and W_UV to every cached c_KV — this
costs O(H · L · d_k_nope) memory and compute per step.

Weight absorption avoids this by fusing the projection weights into the attention score
and the output aggregation, so **K_nope and V are never explicitly formed**.

### Algebraic Identity: Absorbing K

Start from the nope attention score for head h:

```
q_nope_h^T · K_nope_h_i
  = (W_UQ_h · c_Q)^T · (W_UK_h · c_KV_i)
  = c_Q^T · (W_UQ_h^T · W_UK_h) · c_KV_i
  = c_Q^T · W_combined_h · c_KV_i
```

where the fused weight W_combined_h = W_UQ_h^T @ W_UK_h ∈ R^{d_c_q × d_c}.

Define the **absorbed query**: q_abs_h = c_Q @ W_combined_h ∈ R^{d_c}.
Then the nope score becomes simply `q_abs_h · c_KV_i`, computed directly in the
**latent space** (dim d_c) without ever building K_nope (dim d_k_nope per head).

For all heads simultaneously:

```
W_combined = bmm(W_UQ.T, W_UK)          # [H, d_c_q, d_c]
q_abs = einsum("blq,hqc->bhlc", c_Q, W_combined)     # [B, H, L, d_c]
score_nope = einsum("bhqc,bkc->bhqk", q_abs, c_KV) / sqrt(d_k)
```

### Algebraic Identity: Absorbing V

The output for head h:

```
out_h = sum_k attn_h_k · V_h_k
      = sum_k attn_h_k · W_UV_h · c_KV_k
      = W_UV_h · (sum_k attn_h_k · c_KV_k)
      = W_UV_h · context_h
```

where context_h = attn_h @ c_KV ∈ R^{d_c} (aggregate c_KV instead of V).
V is never materialized; the projection W_UV_h is applied **after** the softmax:

```
context = einsum("bhqk,bkc->bhqc", attn, c_KV)       # [B, H, L, d_c]
W_UV_h  = w_uv.weight.reshape(H, d_k, d_c)            # [H, d_k, d_c]
out     = einsum("bhlc,hdc->bhld", context, W_UV_h)   # [B, H, L, d_k]
```

### Why k_rope Cannot Be Absorbed

RoPE applies a **position-dependent** rotation matrix R(pos):

```
q_rope_h · RoPE(pos_q) · k_rope · RoPE(pos_k)^T
```

The rotation matrices differ per position, so they cannot be fused with static
weight matrices. k_rope must remain explicit and is broadcast across all H heads:

```
k_rope = rope(w_kr(x).reshape(..., 1, L, d_k_rope), pos)   # [B, 1, L, d_k_rope]
score_rope = einsum("bhqd,bgkd->bhqk", q_rope, k_rope) / sqrt(d_k)
```

### Real KV Cache During Decode

```
Stage 1 cache per token:  K_nope (H·d_k_nope) + V (H·d_k) + k_rope (d_k_rope)
Stage 2 cache per token:  c_KV   (d_c)         + k_rope (d_k_rope)
```

Since d_c << H·d_k for typical MLA configurations, Stage 2 eliminates the O(H·d_k)
portion from the cache.

### Summary: What Changes Between Stage 1 and Stage 2

| Quantity   | Stage 1                    | Stage 2                              |
| ---------- | -------------------------- | ------------------------------------ |
| K_nope     | W_UK @ c_KV  [B,H,L,d_k_nope] | **not materialized**            |
| Q_nope     | W_UQ @ c_Q   [B,H,L,d_k_nope] | q_abs = c_Q @ W_combined [B,H,L,d_c]|
| Attn score | SDPA(Q, K, V)              | score_nope + score_rope (separate)   |
| V          | W_UV @ c_KV  [B,H,L,d_k]  | **not materialized**                 |
| Value agg  | attn @ V                   | (attn @ c_KV) @ W_UV^T              |
| k_rope     | explicit, expanded to H    | explicit, shape [B,1,L,d_k_rope]     |
| Parameters | same                       | same (W_combined computed on the fly)|

Mathematically Stage 2 is **identical** to Stage 1. Floating-point differences
(reordered summations) are at the ~1e-4 level in float32.

---

> **Problem (mla_absorbed): Implement MLA Stage 2 — Weight Absorption**
>
> **Deliverable**: Implement `MultiHeadLatentAttentionAbsorbed` in
> `cs336_basics/model/attention_variants.py`.
>
> **Recommended approach**: subclass `MultiHeadLatentAttention` and override only
> `forward` — the weight structure (all projection layers) is identical.
>
> **Interface** (same as Stage 1):
>
> ```python
> class MultiHeadLatentAttentionAbsorbed(MultiHeadLatentAttention):
>     def forward(self, x: Tensor, token_positions=None) -> Tensor
> ```
>
> **forward must**:
>
> 1. Compute `c_KV = w_dkv(x)` and `c_Q = w_dq(x)` (same as Stage 1)
> 2. Fuse nope projections: `W_combined[h] = W_UQ[h]^T @ W_UK[h]`
>    via `torch.bmm(w_uq.weight.reshape(H,d_k_nope,d_c_q).permute(0,2,1), w_uk.weight.reshape(H,d_k_nope,d_c))`
> 3. Compute absorbed query: `q_abs = einsum("blq,hqc->bhlc", c_Q, W_combined)` → [B,H,L,d_c]
> 4. Nope score: `einsum("bhqc,bkc->bhqk", q_abs, c_KV) / sqrt(d_k)` → [B,H,L,L]
> 5. If rope: compute `q_rope`, `k_rope` (single head, no expand), apply RoPE,
>    add `einsum("bhqd,bgkd->bhqk", q_rope, k_rope) / sqrt(d_k)` — g=1 broadcasts
> 6. Apply causal mask and softmax
> 7. Aggregate: `context = einsum("bhqk,bkc->bhqc", attn, c_KV)` → [B,H,L,d_c]
> 8. Project: `W_UV = w_uv.weight.reshape(H, d_k, d_c)`;
>    `out = einsum("bhlc,hdc->bhld", context, W_UV)` → [B,H,L,d_k]
> 9. Merge heads and apply `output_proj`
>
> **Note**: Scale both score_nope and score_rope by `1/sqrt(d_k)` (full head dim),
> not by `1/sqrt(d_c)` or `1/sqrt(d_k_nope)`. This preserves numerical equivalence
> with Stage 1 where `q.size(-1) == d_k`.
>
> **Note**: Add `self.d_c` and `self.d_c_q` to `MultiHeadLatentAttention.__init__`
> so that `MultiHeadLatentAttentionAbsorbed.forward` can access them via `self`.
>
> **Reference**: DeepSeek-AI, "DeepSeek-V2," 2024, Section 2.1.3 (KV Cache during inference).
>
> To test: `uv run pytest tests/test_mla_absorbed.py -v`

---

## Unit Tests

每个变体对应独立测试文件，互不干扰。

### MQA Tests (`tests/test_mqa.py`)

```python
test_mqa_output_shape              # output shape is [B, L, d_model]
test_mqa_kv_param_count            # K+V params are H× smaller than MHA
test_mqa_causal_masking            # future tokens don't affect past outputs
```

### GQA Tests (`tests/test_gqa.py`)

```python
test_gqa_output_shape                          # output shape is [B, L, d_model]
test_gqa_kv_param_count_scales_with_num_kv_heads  # KV params linear in num_kv_heads
test_gqa_full_kv_heads_equals_mha              # GQA(G=H) ≡ MHA with same weights
test_gqa_single_kv_head_equals_mqa             # GQA(G=1) ≡ MQA with same weights
test_gqa_causal_masking                        # future tokens don't affect past outputs
test_gqa_gradient_flow                         # all parameters receive gradients
```

### MLA Tests (`tests/test_mla.py`)

```python
test_mla_output_shape                  # output shape is [B, L, d_model]
test_mla_decoupled_rope_output_shape   # full MLA with decoupled RoPE
test_mla_kv_cache_smaller_than_mha     # d_c < 2·H·d_k (cache savings)
test_mla_latent_reconstruction         # K, V are deterministic from c_KV
test_mla_causal_masking                # future tokens don't affect past outputs
test_mla_decoupled_rope_causal_masking # causal masking with decoupled RoPE
test_mla_gradient_flow                 # all parameters receive gradients
test_mla_decoupled_rope_gradient_flow  # gradients through all decoupled-RoPE paths
```

### MLA Absorbed Tests (`tests/test_mla_absorbed.py`)

```python
test_absorbed_output_shape             # output shape is [B, L, d_model]
test_absorbed_rope_output_shape        # same with decoupled RoPE
test_absorbed_equals_stage1_no_rope    # KEY: Stage 2 ≡ Stage 1 numerically (no rope)
test_absorbed_equals_stage1_with_rope  # KEY: Stage 2 ≡ Stage 1 numerically (with rope)
test_absorbed_causal_masking           # future tokens don't affect past outputs
test_absorbed_rope_causal_masking      # same with decoupled RoPE
test_absorbed_gradient_flow            # all parameters receive gradients
test_absorbed_rope_gradient_flow       # same with decoupled RoPE
test_absorbed_kv_cache_is_latent       # cache size = d_c + d_k_rope < 2·d_model
```

The two `equals_stage1` tests are the most important: they verify the algebraic
identities (K absorption and V absorption) hold numerically in float32.
If Stage 2 diverges from Stage 1, the most likely bugs are:
- Wrong scaling factor (using `d_c` or `d_k_nope` instead of `d_k` in `/ sqrt(...)`)
- Wrong reshape order for `W_UQ` or `W_UK` (shape must be `[H, d_k_nope, d_c_q/d_c]`)
- Missing or wrong einsum contraction index

---

## Running the Tests

### Run each variant independently

```bash
uv run pytest tests/test_mqa.py -v
uv run pytest tests/test_gqa.py -v
uv run pytest tests/test_mla.py -v
uv run pytest tests/test_mla_absorbed.py -v
```

### Run a single test with verbose output

```bash
uv run pytest tests/test_gqa.py::test_gqa_full_kv_heads_equals_mha -vv
uv run pytest tests/test_mla.py::test_mla_latent_reconstruction -vv
uv run pytest tests/test_mla_absorbed.py::test_absorbed_equals_stage1_with_rope -vv
```

### Run all four together

```bash
uv run pytest tests/test_mqa.py tests/test_gqa.py tests/test_mla.py tests/test_mla_absorbed.py -v
```

### Stop on first failure

```bash
uv run pytest tests/test_mqa.py -x -v
uv run pytest tests/test_gqa.py -x -v
uv run pytest tests/test_mla_absorbed.py -x -v
```

---

## References

1. N. Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need," arXiv 2019.
  [https://arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150)
2. J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón, S. Sanghai,
  "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints,"
   EMNLP 2023. [https://arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)
3. DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts
  Language Model," 2024. [https://arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434)
   (See Section 2.1 for MLA formulation, Section 2.1.2 for decoupled RoPE)
4. A. Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
  (Baseline MHA formulation referenced throughout)

---

## KV Cache Memory Comparison (summary table)

Assume H query heads, d_k = d_model / H head dimension:


| Method           | KV cache per token              | Reduction vs MHA  |
| ---------------- | ------------------------------- | ----------------- |
| MHA              | 2 · H · d_k                     | 1×                |
| GQA(G)           | 2 · G · d_k                     | H/G×              |
| MQA              | 2 · d_k                         | H×                |
| MLA Stage 1      | d_c (+ d_k_rope)                | ~57× (DeepSeek)   |
| MLA Stage 2      | d_c + d_k_rope (same as Stage 1)| ~57× (DeepSeek)   |

Note: Stage 1 and Stage 2 have the same **cache size** — the difference is in
**decode compute**. Stage 1 expands c_KV → K_nope ([B,H,L,d_k_nope]) and
c_KV → V ([B,H,L,d_k]) every step; Stage 2 avoids both expansions by absorbing
W_UK and W_UV into the attention arithmetic.


For DeepSeek-V2 (H=128, d_k=128, d_c=512, d_k_rope=64):

```
MHA  KV cache per token: 2 × 128 × 128 = 32,768 elements
MLA  KV cache per token: 512 + 64     =     576 elements
Reduction: 32,768 / 576 ≈ 56.9×
```

