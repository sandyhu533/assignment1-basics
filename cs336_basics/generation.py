"""Text generation utilities."""

import torch

from cs336_basics.model import TransformerLM


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """
    Sample next token from logits with optional temperature and top-k.

    Args:
        logits: Shape (..., vocab_size).
        temperature: Sampling temperature (higher = more random).
        top_k: If set, only sample from top-k tokens.

    Returns:
        Sampled token IDs of shape (...).
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        logits = logits.clone()
        logits[logits < v[..., -1, None]] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def generate(
    model: TransformerLM,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    Autoregressively generate tokens given a prompt.

    Args:
        model: TransformerLM model.
        prompt: Token IDs of shape (batch, seq_len) or (seq_len,).
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling (None to disable).
        eos_token_id: Stop generation when this token is produced (None to disable).

    Returns:
        Generated token IDs.
    """
    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    context_length = model.blocks[0].mha.rope.rotate.shape[0]
    generated = prompt.clone()

    for _ in range(max_new_tokens):
        context = generated[:, -context_length:]
        with torch.no_grad():
            logits = model(context)
        next_logits = logits[:, -1, :]
        next_token = sample_next_token(next_logits, temperature, top_k)
        generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    if squeeze:
        generated = generated.squeeze(0)
    return generated
