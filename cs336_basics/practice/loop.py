"""Practice scaffold: training loop + decoding loop.

Fill in every body that says `raise NotImplementedError`. Tests live at
`tests/test_loop.py` and skip any function still unimplemented, so you can
work in any order and re-run tests after each function.

Docstrings describe the CONTRACT (input/output/edge cases). They do NOT tell
you the algorithm — that's the exercise.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn


# ============================================================
# Decoding primitives — pure functions on a logits tensor.
# logits last dim is always `vocab_size`.
# ============================================================

def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Rescale logits by temperature.

    temperature > 0 : standard scaling.
    temperature == 0: greedy — after a subsequent softmax the argmax token
                       must have probability ≈ 1 and all others ≈ 0.

    Does NOT modify `logits` in place.
    """
    if temperature > 0:
        return logits/temperature
    out = torch.full_like(logits, float('-inf'))
    out.scatter_(-1, torch.argmax(logits, -1, keepdim=True), 0)
    return out


def top_k_filter(logits: Tensor, k: int | None) -> Tensor:
    """Keep only the top-k values along the last dim; mask the rest to -inf.

    k is None or k >= vocab_size → return logits unchanged.
    Shape preserved.
    """
    if k is None or k >= logits.size(-1): return logits
    values, indices = torch.topk(logits, k, dim=-1)
    out = torch.full_like(logits, float('-inf'))
    out.scatter_(-1, indices, values)
    return out


def top_p_filter(logits: Tensor, p: float | None) -> Tensor:
    """Nucleus filter: keep the smallest set whose softmax mass ≥ p; mask others to -inf.

    p is None or p >= 1.0 → return logits unchanged.
    The kept set must ALWAYS include the argmax token, even if its single
    probability already exceeds p.
    Shape preserved.
    """
    if p is None or p >= 1.0: return logits
    probs = logits.softmax(dim=-1)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    sumlogits = sorted_probs.cumsum(dim=-1)
    
    sorted_mask = sumlogits < p
    sorted_mask.scatter_(-1, sorted_probs.argmax(-1, keepdim=True), True)
    mask = torch.empty_like(sorted_mask).scatter_(-1, sorted_idx, sorted_mask)
    return torch.where(mask, logits, float('-inf'))

def sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one token per row.

    logits: shape [..., vocab_size].
    Returns: long tensor of shape [...] (one fewer dim).
    Pipeline order: temperature → top_k → top_p → softmax → sample.
    Use `generator` for reproducibility.
    """
    logits = apply_temperature(logits, temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


# ============================================================
# Decode loop.
# ============================================================

def decode(
    model: nn.Module,
    prompt_ids: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Autoregressive decode.

    prompt_ids: long tensor [batch, prompt_len].
    Returns:    long tensor [batch, prompt_len + N] where N ≤ max_new_tokens.
                Includes the prompt. May stop early when ALL rows have produced
                eos_id at least once (if eos_id is given).

    No KV cache — recompute the whole sequence each step. (Cache is a separate
    exercise; doing it here would couple this scaffold to the model internals.)
    """
    for _ in range(max_new_tokens):
        logits = model(prompt_ids)[:,-1,:]
        next_token_id = sample_next_token(logits, temperature, top_k, top_p, generator)
        if next_token_id == eos_id:
            prompt_ids = torch.cat([prompt_ids, next_token_id.unsqueeze(-1)], dim=-1)
            break
        prompt_ids = torch.cat([prompt_ids, next_token_id.unsqueeze(-1)], dim=-1)
    return prompt_ids
    


# ============================================================
# Training primitives.
# ============================================================

def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean cross-entropy over batch and time.

    logits:  float [batch, seq, vocab]
    targets: long  [batch, seq]
    Returns: scalar tensor (mean over all batch * seq positions).

    Must be numerically stable for logits whose magnitude exceeds ~50.
    Implement the math yourself; do not just call `F.cross_entropy`.
    """
    maxv = logits.amax(-1, keepdim=True)
    logsumexp = maxv + (logits-maxv).exp().sum(-1, keepdim=True).log()
    log_probs = logits - logsumexp
    ll = -torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
    return ll.mean()


def gradient_l2_norm(params: Iterable[nn.Parameter]) -> Tensor:
    """Global L2 norm over all `.grad` tensors in `params`.

    Params with `grad is None` are skipped.
    Returns a 0-dim tensor on the device of the first non-None grad.
    """
    raise NotImplementedError


def clip_gradient_l2_norm(
    params: Iterable[nn.Parameter], max_norm: float, eps: float = 1e-6
) -> Tensor:
    """In-place scale `.grad` by `min(1, max_norm / (total_norm + eps))`.

    Returns the PRE-clip global norm (for logging).
    """
    raise NotImplementedError


def cosine_lr_schedule(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Linear warmup then half-cosine decay to min_lr.

    Regions:
      step < warmup_steps               : linear from 0 → max_lr
      warmup_steps ≤ step ≤ total_steps : cosine from max_lr → min_lr
      step > total_steps                : stays at min_lr

    Edge case: warmup_steps == 0 → at step 0 return max_lr.
    """
    raise NotImplementedError


# ============================================================
# Data + checkpoint.
# ============================================================

def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Sample (input, target) windows from a flat token array for next-token training.

    data: 1-D int array of token ids, length N.
    Returns (x, y), each long tensor [batch_size, context_length] on `device`:
      x[b]  = data[start_b : start_b + context_length]
      y[b]  = data[start_b + 1 : start_b + context_length + 1]
    `start_b` is sampled uniformly from [0, N - context_length - 1].
    """
    raise NotImplementedError


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str,
) -> None:
    """Persist model + optimizer + step to `path` via torch.save."""
    raise NotImplementedError


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model + optimizer in place from `path`. Returns the saved step."""
    raise NotImplementedError


# ============================================================
# Training step.
# ============================================================

def train_step(
    model: nn.Module,
    batch: tuple[Tensor, Tensor],
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    """One full training step: fwd → loss → bwd → (optional clip) → step → zero_grad.

    batch: (x, y) from `get_batch`.
    Returns {'loss': float, 'grad_norm': float}.
    `grad_norm` is the PRE-clip global norm. When max_grad_norm is None, still
    report the grad norm (no clipping applied).
    """
    x, target = batch
    logits = model(x)
    loss = cross_entropy_loss(logits, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return {
        'loss': loss,
        'grad_norm': 0
    }
