# Training + Decode Loop Practice

Fill in `cs336_basics/practice/loop.py`. Every function raises
`NotImplementedError`; tests skip unimplemented ones, so work in any order.

## Run tests

```bash
# Run everything (stop on first failure)
uv run pytest tests/test_loop.py -x

# By group
uv run pytest tests/test_loop.py -k "temperature or top_k or top_p or sample"  # decode primitives
uv run pytest tests/test_loop.py -k "decode_"                                   # decode loop
uv run pytest tests/test_loop.py -k "cross_entropy"                             # CE
uv run pytest tests/test_loop.py -k "gradient or clip"                          # grad norm + clip
uv run pytest tests/test_loop.py -k "cosine_lr"                                 # LR schedule
uv run pytest tests/test_loop.py -k "get_batch"                                 # data loader
uv run pytest tests/test_loop.py -k "checkpoint"                                # save/load
uv run pytest tests/test_loop.py -k "train_step"                                # orchestration

# One test, verbose
uv run pytest tests/test_loop.py::test_cross_entropy_stable_with_large_logits -vv
```

## Suggested order (≈ dependency order)

1. `apply_temperature` → `top_k_filter` → `top_p_filter` → `sample_next_token` (pure functions, fastest feedback)
2. `decode` (uses the four above + a mock model from the test file)
3. `cross_entropy_loss` (numerical stability matters)
4. `gradient_l2_norm` → `clip_gradient_l2_norm` (clip uses norm)
5. `cosine_lr_schedule` (pure float function)
6. `get_batch` (data loader)
7. `save_checkpoint` → `load_checkpoint` (round-trip test)
8. `train_step` (glues everything together)

## Function contracts (full spec in docstrings)

```python
# Decoding primitives (pure)
apply_temperature(logits, temperature) -> Tensor
top_k_filter(logits, k) -> Tensor
top_p_filter(logits, p) -> Tensor
sample_next_token(logits, temperature=1.0, top_k=None, top_p=None, generator=None) -> Tensor

# Decode loop
decode(model, prompt_ids, max_new_tokens,
       temperature=1.0, top_k=None, top_p=None,
       eos_id=None, generator=None) -> Tensor

# Training primitives
cross_entropy_loss(logits, targets) -> Tensor                              # scalar
gradient_l2_norm(params) -> Tensor                                          # scalar
clip_gradient_l2_norm(params, max_norm, eps=1e-6) -> Tensor                 # pre-clip norm
cosine_lr_schedule(step, warmup_steps, total_steps, max_lr, min_lr) -> float

# Data + checkpoint
get_batch(data, batch_size, context_length, device="cpu") -> (Tensor, Tensor)
save_checkpoint(model, optimizer, step, path) -> None
load_checkpoint(path, model, optimizer) -> int

# Training step
train_step(model, batch, optimizer, max_grad_norm=None) -> dict
#   returns {"loss": float, "grad_norm": float}
```

## What this trains

- **Decoding hygiene**: temperature/top-k/top-p ordering, the always-include-argmax invariant for top-p, reproducibility with `torch.Generator`.
- **Numerical stability**: CE on logits that overflow a naive `exp(·)` → log-sum-exp.
- **Optimizer plumbing**: zero_grad placement, the clip-before-step ordering, returning the *pre-clip* norm for logging.
- **Data pipeline**: random-window sampling from a flat token stream (the standard pre-training data layout).
- **Checkpoint round-trip**: model state + optimizer state + step counter — what you need to resume mid-training.

## Interview hooks

- **Why log-sum-exp in CE?** Mixed-precision training routinely produces logits with magnitude > 50; naive `exp(·)` overflows in bf16 (max ≈ 3.4e38) and underflows in fp16 (max ≈ 6.5e4).
- **Why clip *before* step?** Clipping after `optimizer.step()` corrupts the m/v estimates AdamW already absorbed.
- **Why store the pre-clip norm?** Monitoring the unclipped norm catches gradient explosion early; the post-clip norm is bounded by construction and tells you nothing.
- **Why a separate `get_batch` instead of `DataLoader`?** Pre-training data is one giant token stream — uniform window sampling is both simpler and uniformly distributed over all positions, which `DataLoader` over fixed shards is not.
