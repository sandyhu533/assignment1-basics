# 3.6 Mixture of Experts (MoE)

This section implements the **DeepSeekMoE** architecture from DeepSeek-V2, replacing the
dense FFN sublayer with a sparse mixture of experts.

We build it in **two stages**, mirroring the MLA Stage-1 / Stage-2 split:


| Stage       | What it adds                                                         | Forward signature                |
| ----------- | -------------------------------------------------------------------- | -------------------------------- |
| **Stage 1** | Sparse top-K routing + shared/routed experts. **No load balancing.** | `forward(x) -> Tensor`           |
| **Stage 2** | Adds the expert-level load-balance auxiliary loss.                   | `forward(x) -> (Tensor, Tensor)` |


Stage 2 **does not change the forward output** — it only attaches an extra training signal.
The two stages must produce identical `output` given identical weights (an equivalence test).

**Reference**: DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts
Language Model," 2024. Section 2.2 (DeepSeekMoE), Eq. 12–14.

---

## Background: The Dense FFN Bottleneck

A standard transformer FFN processes every token with the same weights:

```
FFN(x) = W_down · SwiGLU(x)     params: 3 × d_model × d_ff
FLOPs per token: ~6 × d_model × d_ff
```

Scaling up requires either wider models (more params per token, more memory) or deeper models
(more layers, more latency). MoE decouples **model capacity** from **per-token compute** by
keeping only a fraction of experts active for each token.

```
Dense FFN:  every token uses the same d_ff neurons
MoE FFN:    each token routes to K out of N expert FFNs
            active params per token: ~K/N × total params
            total params can be N× larger for the same FLOPs
```

DeepSeek-V2 (236B parameters) uses 160 routed experts but activates only 6 per token,
matching a ~21B active-parameter dense model in compute cost.

---

## 3.6.1 DeepSeekMoE Architecture (shared by both stages)

### Expert Types

Each MoE layer has two classes of experts:

```
N_s  shared experts     — always active for every token
N_r  routed experts     — selected per-token via top-K gating
```

Shared experts act as a stable backbone that all tokens use; routed experts specialize.
In DeepSeek-V2: N_s = 2, N_r = 160, K_r = 6.

### Output Formula

For token t with hidden state u_t ∈ R^{d_model}:

```
h'_t = u_t                                              # residual
      + Σ_{i=1}^{N_s}     FFN_i^(shared)(u_t)           # shared experts (always)
      + Σ_{i ∈ T_t}  g_{t,i} · FFN_i^(routed)(u_t)     # routed experts (sparse)
```

where:

- T_t = indices of the top-K_r routed experts chosen for token t
- g_{t,i} = routing gate score
- Each expert FFN is an independent SwiGLU network

> **Note on the residual**: whether the `+ u_t` residual lives inside `MoELayer` or in the
> surrounding `TransformerBlock` is a convention choice. This guide keeps the residual
> **outside** (in the block), so `MoELayer.forward` returns only the shared+routed sum.

### Expert FFN Structure

Each expert is an independent SwiGLU FFN identical in form to the dense FFN
(`cs336_basics/model/transformer.py::SwiGLU`):

```
Expert_i(x) = W_down_i · (SiLU(W_gate_i · x) ⊙ (W_up_i · x))

W_gate_i, W_up_i  ∈ R^{d_ffn × d_model}
W_down_i          ∈ R^{d_model × d_ffn}
```

You can reuse the existing `SwiGLU` module directly as one expert. The hidden dimension
d_ffn is typically chosen so that the K_r active routed experts together roughly match the
dense FFN FLOPs; for this assignment, accept d_ffn as a hyperparameter.

---

## 3.6.2 Stage 1 — Sparse Routing (no load balancing)

### Routing Mechanism

The router embeds each routed expert as a learnable centroid vector e_i ∈ R^{d_model}:

```
# Step 1: affinity scores for all N_r routed experts (dot product with each centroid)
logits_{t,:} = u_t @ E^T            ∈ R^{N_r}     where E ∈ R^{N_r × d_model}

# Step 2: normalize across the routed-expert pool
s_{t,:} = Softmax(logits_{t,:})     ∈ R^{N_r}

# Step 3: select the top-K_r experts
T_t = argtopk(s_{t,:}, K_r)

# Step 4: gate scores for the selected experts
g_{t,i} = s_{t,i}   if i ∈ T_t  else  0
```

Design choice (DeepSeek-V2): gate scores are the **raw softmax values** s_{t,i}, *not*
renormalized over the top-K. So Σ_{i∈T_t} g_{t,i} < 1 in general — the dropped experts'
probability mass is simply discarded. (Renormalizing is a valid alternative; pick one and
keep the tests consistent.)

### Aggregation: scatter-add over (token, expert) pairs

The performance-critical part of any MoE layer is turning the per-token top-K selection into
a batched expert computation. The standard trick: flatten all `(token, expert)` pairs, run
each expert on its assigned tokens, then scatter-add results back to token positions.

```python
# x_flat: [T, d_model], T = B*L
# indices: [T, K]   (top-K expert id per token)
# gates:   [T, K]   (gate score per selected expert)

result = torch.zeros_like(x_flat)               # [T, d_model]
for e in range(n_routed):                        # loop over experts (vectorized inside)
    token_mask = (indices == e)                  # [T, K] which slots picked expert e
    if not token_mask.any():
        continue
    tok_ids = token_mask.any(dim=1).nonzero(as_tuple=True)[0]   # tokens that picked e
    contrib = experts[e](x_flat[tok_ids])        # run expert e once on its tokens
    # weight each token by the gate score it gave to expert e
    w = (gates * token_mask).sum(dim=1)[tok_ids] # [n_tok_for_e]
    result[tok_ids] += w.unsqueeze(-1) * contrib
```

This loops over **experts** (N_r iterations), not tokens — each expert runs at most once per
forward. An alternative is the fully-vectorized `scatter_add`_ form; either is acceptable as
long as gradient flows to every expert that was selected.

> **Interview hook**: "How would you make MoE routing efficient on GPU?" The answer is
> exactly this gather→batched-expert→scatter pattern (this is what `grouped_gemm` /
> megablocks / the SGLang+vLLM fused MoE kernels do). The naive per-token Python loop is
> O(T·K) kernel launches; the expert-grouped form is O(N_r).

### What Stage 1 deliberately omits

Without a balance loss, routing tends to **collapse**: a few experts get all the tokens
("rich-get-richer", because gradient reinforces high-usage experts). Stage 1 will still train
and pass correctness tests, but the expert utilization will be skewed. Fixing that is Stage 2.

---

> **Problem (moe_stage1): Implement the sparse MoE layer (no balancing)**
>
> **Deliverable**: Implement `MoELayer` in `cs336_basics/model/moe.py` (experts are `SwiGLU`).
>
> ```python
> class MoELayer(nn.Module):
>     def __init__(
>         self,
>         d_model: int,
>         d_ffn: int,
>         n_shared: int,       # N_s: always-active experts
>         n_routed: int,       # N_r: routed expert pool size
>         top_k: int,          # K_r: routed experts per token
>         device=None,
>     )
>     def forward(self, x: Tensor) -> Tensor        # [B, L, d_model] → [B, L, d_model]
> ```
>
> **Naming convention** (the tests rely on these attribute names):
>
> - `self.router`          → `Linear(d_model, n_routed)` — its `.weight` `[n_routed, d_model]`
>                      is the matrix of expert centroids
> - `self.shared_experts`  → `nn.ModuleList` of experts (length `n_shared`)
> - `self.routed_experts`  → `nn.ModuleList` of experts (length `n_routed`)
>
> Each expert is just a `SwiGLU(d_model, d_ffn)` — no separate `Expert` class is needed,
> though you may wrap one for clarity.
>
> **Router**: compute `logits = self.router(x_flat)`  (= `x_flat @ router.weight.T`),
> softmax over `n_routed`, then `torch.topk(probs, top_k)`.
>
> **Expose routing state** (consumed by tests and by Stage 2):
>
> - `self.last_router_probs`   → `[T, n_routed]`  full softmax distribution
> - `self.last_router_indices` → `[T, top_k]`     selected expert ids (int64)
>
> **Make sure to**:
>
> - Sum all `n_shared` shared experts (always active) using the original `x`.
> - Weight each routed expert output by its gate score `s_{t,i}`.
> - Aggregate routed outputs with the gather→expert→scatter pattern above.
> - Handle `n_routed == 0` (no routing; `last_router_`* can be empty tensors).
> - Handle `n_shared == 0` (pure routing).
> - Handle `top_k == n_routed` (all routed experts selected; degenerates to dense-MoE).
>
> To test: `uv run pytest tests/test_moe_stage1.py -v`

### Stage 1 test guide (`tests/test_moe_stage1.py`)


| Test                                            | What it checks                                                           |
| ----------------------------------------------- | ------------------------------------------------------------------------ |
| `test_output_shape`                             | forward returns a single `[B, L, d_model]` tensor                        |
| `test_shared_only`                              | `n_routed=0` works (only shared experts)                                 |
| `test_routed_only`                              | `n_shared=0` works (pure routing)                                        |
| `test_router_indices_shape_and_range`           | `last_router_indices` is `[T, top_k]`, values in `[0, n_routed)`         |
| `test_router_probs_distribution`                | `last_router_probs` is `[T, n_routed]`, each row sums to 1               |
| `test_all_experts_when_topk_equals_n_routed`    | `top_k==n_routed` → every expert id selected                             |
| `test_deterministic`                            | same input → identical output (routing is argmax, not sampled)           |
| `test_gradient_flow`                            | every parameter (shared, routed, router) gets non-zero gradient          |
| `test_unselected_expert_does_not_affect_output` | mutating an expert no token routed to leaves output unchanged (sparsity) |


---

## 3.6.3 Stage 2 — Expert-Level Load Balance Loss

### Why it's needed

Backprop reinforces whichever experts already win tokens, so unbalanced routing is a stable
attractor. Two failure modes:

1. **Capacity waste**: a few experts do all the work; the rest are dead parameters.
2. **Hardware skew**: in distributed MoE, overloaded experts' GPUs become stragglers.

The auxiliary balance loss pushes the router toward uniform expert utilization.

### Expert-Level Balance Loss (DeepSeek-V2, Eq. 12–14)

```
L_ExpBal = α · Σ_{i=1}^{N_r} f_i · P_i

f_i = (N_r / (K_r · T)) · Σ_{t=1}^{T} 1[ i ∈ T_t ]      # load: normalized selection count
P_i = (1 / T)           · Σ_{t=1}^{T} s_{t,i}            # mean routing probability
```

where T = batch_size × seq_len, α is a small coefficient (DeepSeek-V2: α ≈ 0.003).

**Normalization sanity check** — at perfect balance:

- each expert is selected `T·K_r/N_r` times → `f_i = (N_r/(K_r·T))·(T·K_r/N_r) = 1`
- softmax is uniform → `P_i = 1/N_r`
- `L = α · Σ_i 1 · (1/N_r) = α · N_r · (1/N_r) = α`

So perfect balance gives exactly **L = α** — this is the value the router is driven
toward, and it's a clean check on the `N_r/(K_r·T)` normalization factor.

> **Caution**: α is *not* a hard lower bound. The true infimum of `Σ f_i P_i` (subject
> to `Σ f_i = N_r`, `Σ P_i = 1`, both ≥ 0) is 0, reached when load and probability are
> anti-correlated. In practice f_i comes from the top-K of P_i, so the two are positively
> correlated and L hovers near or above α — but don't assert `L ≥ α` as an invariant.

### The key gradient subtlety: detach f_i

```
f_i  comes from argtopk → discrete → NOT differentiable.
P_i  is a mean of softmax outputs → differentiable.
```

f_i must be **detached** (`f_i = f_i.detach()`). The gradient flows only through P_i:

```
∂L/∂s_{t,i} = α · f_i · (1/T)        # f_i is a constant scaling factor here
```

This is the **straight-through estimator** pattern: the discrete load measurement f_i acts as
a per-expert weight on the differentiable probability term. Intuitively — "for each
overloaded expert (large f_i), apply downward pressure proportional to its probability mass."

> Note: `torch.topk` already breaks the gradient path (indices are discrete), so in practice
> f_i carries no gradient even without `.detach()`. We detach anyway for clarity and to avoid
> a subtle bug if you ever build f_i from a soft/differentiable count.

### Equivalence to Stage 1

The balance loss is **auxiliary** — it is added to the training loss, never to the forward
output. Therefore:

```
MoELayerBalanced(x)[0]  ==  MoELayer(x)        # identical output, same weights
```

This is the Stage-1 / Stage-2 equivalence test, analogous to MLA absorption.

---

> **Problem (moe_stage2): Add the expert-level load-balance loss**
>
> **Deliverable**: Implement `MoELayerBalanced` in `cs336_basics/model/moe.py`.
>
> **Recommended approach**: subclass `MoELayer` and reuse its routing. The forward computes
> the same output, then derives the loss from `self.last_router_probs` / `self.last_router_indices`.
>
> ```python
> class MoELayerBalanced(MoELayer):
>     def __init__(self, d_model, d_ffn, n_shared, n_routed, top_k,
>                  balance_coeff: float = 0.001, device=None)
>     def forward(self, x: Tensor) -> tuple[Tensor, Tensor]
>     #   returns (output [B, L, d_model], balance_loss scalar)
> ```
>
> **Balance loss**:
>
> - `probs   = self.last_router_probs`   # [T, n_routed]
> - `indices = self.last_router_indices` # [T, top_k]
> - `f_i = (n_routed / (top_k * T)) * count(i in indices)`  →  **detach**
> - `P_i = probs.mean(dim=0)`
> - `loss = balance_coeff * (f * P).sum()`
>
> **Make sure to**:
>
> - Return `torch.tensor(0.0)` for the loss when `n_routed == 0` (nothing to balance).
> - Keep `output` byte-for-byte equal to Stage 1 (only the second return value is new).
>
> To test: `uv run pytest tests/test_moe_stage2.py -v`

### Stage 2 test guide (`tests/test_moe_stage2.py`)


| Test                                    | What it checks                                                          |
| --------------------------------------- | ----------------------------------------------------------------------- |
| `test_forward_returns_tuple`            | forward returns `(output, loss)` with loss a scalar                     |
| `test_output_matches_stage1`            | output identical to `MoELayer` given the same weights                   |
| `test_aux_loss_nonneg`                  | balance loss ≥ 0 for any input                                          |
| `test_uniform_router_loss_equals_alpha` | zero-router + constant input → loss == α exactly (normalization sanity) |
| `test_aux_loss_zero_without_routed`     | `n_routed=0` → loss == 0                                                |
| `test_aux_loss_gradient_to_router`      | `loss.backward()` puts gradient on `router` (via P_i)                   |
| `test_balance_loss_scales_with_coeff`   | doubling `balance_coeff` exactly doubles the loss                       |
| `test_balanced_routing_has_lower_loss`  | hand-built uniform routing < skewed routing                             |


---

## 3.6.4 MoE vs MLA: Two Axes of Efficiency


| Dimension           | MoE                                 | MLA                           |
| ------------------- | ----------------------------------- | ----------------------------- |
| Bottleneck targeted | FFN compute per token               | KV-cache memory               |
| Mechanism           | Sparse expert routing               | Low-rank KV compression       |
| Tradeoff            | Load imbalance, routing instability | KV approximation quality      |
| Scaling direction   | Model capacity without FLOPs        | Longer context without memory |


Both ship together in DeepSeek-V2: MLA for attention, DeepSeekMoE for the FFN.

---

## Common Debugging Pitfalls

**NaN with n_routed=0**: softmax over an empty expert dim. Guard the routed branch entirely
when `n_routed == 0`.

**Some parameters get no gradient (Stage 1 `test_gradient_flow` fails)**: usually the
scatter/gather drops the gradient. If you index `x_flat[tok_ids]`, run the expert, and write
back with `result[tok_ids] += ...`, gradient flows fine. If instead you build outputs with an
in-place op on a tensor that's detached or pre-allocated wrong, it breaks. Also: an expert
that *no* token selected will legitimately have zero gradient — size your test (`T`, `top_k`,
`n_routed`) so every expert is hit, or exclude unselected experts from the assertion.

**Stage 2 output diverges from Stage 1**: you accidentally folded the gate renormalization or
the loss into the forward path. The loss must be a pure read-off of routing stats; it must not
touch `output`.

**Balance loss gradient is zero on the router**: you detached `probs` (P_i) instead of (or in
addition to) f_i. Only **f_i** is detached; P_i must stay on the graph.

**Loss floor isn't α**: f_i normalization factor `N_r/(K_r·T)` is wrong or missing. Re-derive
with the perfect-balance check above.