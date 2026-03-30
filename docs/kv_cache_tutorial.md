# KV Cache for Efficient Autoregressive Generation

## 1. The Problem: Why Naive Generation Is Slow

Recall from Section 6 of the assignment that autoregressive text generation works by repeatedly:

1. Running the full model on the current token sequence to get next-token logits.
2. Sampling a new token from the logits.
3. Appending the new token to the sequence and repeating.

In our current implementation (`generate()` in `generation.py`), **every step re-runs the entire Transformer on the full prefix**. Let's analyze why this is wasteful.

### Attention Complexity Per Step

Consider a single attention layer processing a sequence of length $n$ with head dimension $d_k$ and $h$ heads:

- **Q, K, V projections**: $3 \times n \times d_{\text{model}}^2$ FLOPs (project all $n$ tokens)
- **$QK^\top$ matrix multiply**: $n^2 \times d_k \times h$ FLOPs
- **Weighted sum with V**: $n^2 \times d_k \times h$ FLOPs
- **Output projection**: $n \times d_{\text{model}}^2$ FLOPs

For $L$ layers, total attention work per step: $O(L \cdot (n \cdot d^2 + n^2 \cdot d))$.

Over $T$ generation steps (where the prefix grows from $n_0$ to $n_0 + T$), the total work is:

$$\sum_{t=0}^{T-1} O\left(L \cdot \left((n_0+t) \cdot d^2 + (n_0+t)^2 \cdot d\right)\right) = O\left(L \cdot T \cdot n \cdot d^2 + L \cdot T \cdot n^2 \cdot d\right)$$

where $n = n_0 + T$ is the final sequence length. The $n^2$ term makes this **quadratic in sequence length per step**.

### The Key Observation

When generating the $(n+1)$-th token, the model computes attention as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Due to the **causal mask**, the output at position $n$ only depends on keys and values at positions $0, 1, \ldots, n$. Crucially, **the keys and values at positions $0$ through $n-1$ are identical to what we computed in the previous step**. We are recomputing them from scratch every time!

## 2. KV Cache: The Solution

The KV Cache optimization is simple: **cache the Key and Value tensors from previous positions and reuse them**.

### Two-Phase Generation

With KV Cache, generation splits into two phases:

**Phase 1 — Prefill**: Process the entire prompt in one forward pass (just like training). Save the K and V tensors from every layer.

**Phase 2 — Decode**: For each new token:
1. Run *only the new token* through the model.
2. Compute Q, K, V projections for this single token.
3. Concatenate the new K, V with the cached K, V from previous steps.
4. Compute attention: Q (1 token) attending to all cached K, V (full sequence).
5. Update the cache with the new K, V.

```
Prefill (process full prompt):
┌─────────────────────────────────┐
│  tokens: [t₀, t₁, t₂, ..., tₙ] │
│                                   │
│  Forward pass → logits            │
│  Save: K_cache, V_cache per layer │
│  K_cache shape: (batch, heads,    │
│                   n, d_k)         │
└─────────────────────────────────┘
                 │
                 ▼
Decode step i (process 1 token):
┌─────────────────────────────────┐
│  token: [t_{n+i}]                │
│                                   │
│  q = W_Q @ embed(t_{n+i})  [1×d] │
│  k_new = W_K @ embed(...)  [1×d] │
│  v_new = W_V @ embed(...)  [1×d] │
│                                   │
│  K_cache = cat(K_cache, k_new)    │
│  V_cache = cat(V_cache, v_new)    │
│                                   │
│  attn = softmax(q @ K_cache^T) @  │
│         V_cache                   │
│  → next token logits              │
└─────────────────────────────────┘
```

### Complexity With KV Cache

At decode step $t$ (sequence length $n_0 + t$):

- **Projections**: Only 1 token → $O(d^2)$ instead of $O(n \cdot d^2)$
- **$QK^\top$**: $1 \times (n_0 + t)$ → $O(n \cdot d)$ instead of $O(n^2 \cdot d)$
- **Weighted sum**: Same as above → $O(n \cdot d)$

Over $T$ decode steps, total attention work:

$$O\left(L \cdot T \cdot d^2 + L \cdot T \cdot n \cdot d\right)$$

Compared to naive $O(L \cdot T \cdot n \cdot d^2 + L \cdot T \cdot n^2 \cdot d)$, this is a **factor of $\sim n$ speedup** on the projection terms and the attention terms.

## 3. Memory vs. Compute Tradeoff

KV Cache is not free — it trades **memory** for **compute**.

### Memory Cost

For each layer, we store K and V tensors:

$$\text{Memory per layer} = 2 \times \text{batch\_size} \times \text{num\_heads} \times \text{seq\_len} \times d_k \times \text{bytes\_per\_element}$$

For $L$ layers:

$$\text{Total KV Cache Memory} = 2 \times L \times B \times h \times n \times d_k \times \text{sizeof(dtype)}$$

Since $h \times d_k = d_{\text{model}}$, this simplifies to:

$$\boxed{\text{KV Cache Memory} = 2 \times L \times B \times n \times d_{\text{model}} \times \text{sizeof(dtype)}}$$

### Example: GPT-2 XL Generating 1024 Tokens

| Parameter | Value |
|-----------|-------|
| $L$ (num_layers) | 48 |
| $d_{\text{model}}$ | 1600 |
| $n$ (seq_len) | 1024 |
| $B$ (batch_size) | 1 |
| dtype | float32 (4 bytes) |

$$\text{KV Cache} = 2 \times 48 \times 1 \times 1024 \times 1600 \times 4 = 629\text{ MB}$$

This is significant, but the model parameters alone take $\sim$8.5 GB in float32. The KV cache adds roughly 7% overhead for 1024-token generation, while providing a dramatic speedup.

## 4. RoPE Compatibility

Our model uses **Rotary Positional Embeddings (RoPE)**, which applies position-dependent rotations to the Q and K vectors. This is important for KV Cache:

- RoPE is applied to K **before** it enters the cache. The cached K vectors already have their positional information baked in.
- During the decode phase, the new token's Q and K must use the **absolute position** (e.g., position $n+t$), not position 0.
- This means `token_positions` must be passed explicitly during generation, rather than computed from `torch.arange(seq_len)`.

In our implementation, `RotaryPositionalEmbedding` already accepts explicit `token_positions`, so this is a matter of passing the correct values.

## 5. Implementation Guide

Here is a step-by-step guide for implementing KV Cache in the existing codebase.

### Step 1: Modify `CausalMultiHeadSelfAttention.forward()`

**File**: `cs336_basics/model/layers.py`

Add an optional `past_kv` parameter and return the updated cache:

```python
def forward(self, x, token_positions=None, past_kv=None):
    # 1. Project Q, K, V from x (x may be a single token during decode)
    # 2. Rearrange into multi-head format
    # 3. Apply RoPE with absolute token_positions
    
    # 4. If past_kv is provided, concatenate with new K, V
    if past_kv is not None:
        past_k, past_v = past_kv
        k = torch.cat([past_k, k], dim=-2)  # along seq dimension
        v = torch.cat([past_v, v], dim=-2)
    new_kv = (k, v)
    
    # 5. Build causal mask with shape (q_len, kv_len), NOT (seq_len, seq_len)
    q_len = x.shape[-2]
    kv_len = k.shape[-2]
    # The new query tokens can attend to all KV tokens up to their position
    
    # 6. Run scaled_dot_product_attention, output projection
    # 7. Return (output, new_kv) — caller decides whether to use the cache
```

**Critical detail**: The causal mask changes. During prefill, `q_len == kv_len` and you get the standard lower-triangular mask. During decode with `q_len == 1`, the single query token can attend to all `kv_len` cached positions, so the mask is all-True (shape `(1, kv_len)`).

### Step 2: Modify `TransformerBlock`

**File**: `cs336_basics/model/transformer.py`

- Accept `token_positions` as a parameter instead of computing it internally.
- Accept and return `past_kv` for this block's attention layer.

### Step 3: Modify `TransformerLM`

**File**: `cs336_basics/model/transformer.py`

- Accept `past_kv_list` (one cache entry per layer) and `token_positions`.
- Return `(logits, new_kv_list)` when using cache.
- **Backward compatibility**: When `past_kv_list is None` and `token_positions is None`, behave exactly as before (for training). This ensures existing training code is unaffected.

### Step 4: Implement `generate_with_kv_cache()`

**File**: `cs336_basics/generation.py`

```python
def generate_with_kv_cache(model, prompt, max_new_tokens, ...):
    # Prefill: run full prompt, get (logits, kv_cache)
    logits, past_kv = model(prompt, use_cache=True)
    
    for step in range(max_new_tokens):
        next_token = sample(logits[:, -1, :])
        # Decode: run single token with cache
        position = torch.tensor([[prompt_len + step]])
        logits, past_kv = model(
            next_token.unsqueeze(-1),
            token_positions=position,
            past_kv_list=past_kv,
        )
        generated.append(next_token)
    
    return torch.cat(generated, dim=-1)
```

### Step 5: Test and Benchmark

Run the provided tests to verify that KV-cached generation produces **identical** outputs to naive generation (with `temperature=0`). Then use the benchmark script to measure the speedup.

## 6. Further Reading

- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) — Tay et al., 2020
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) — Touvron et al., 2023
  (Section 3.2 discusses grouped-query attention, which reduces KV cache size)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) — Kwon et al., 2023
  (Efficient memory management for KV caches in serving)
