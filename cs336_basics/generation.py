"""Text generation utilities."""

import torch

from cs336_basics.model import TransformerLM, softmax


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
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
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
    elif top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        logits = logits.clone()
        logits[logits < v[..., -1, None]] = float("-inf")
    probs = softmax(logits, dim=-1)
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
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
        generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

    if squeeze:
        generated = generated.squeeze(0)
    return generated


def generate_with_kv_cache(
    model: TransformerLM,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    Autoregressively generate tokens using KV cache for efficient decoding.

    Phase 1 (prefill): Run the full prompt through the model, caching K and V.
    Phase 2 (decode): For each new token, run only a single token through the
    model, reusing cached K/V from all previous positions.

    Args:
        model: TransformerLM model.
        prompt: Token IDs of shape (batch, seq_len) or (seq_len,).
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling (None to disable).
        eos_token_id: Stop generation when this token is produced (None to disable).

    Returns:
        Generated token IDs including the original prompt.
    """
    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    context_length = model.blocks[0].mha.rope.rotate.shape[0]
    generated = prompt.clone()
    prompt_len = prompt.shape[1]

    with torch.no_grad():
        # Phase 1: Prefill — run full prompt, collect KV cache
        context = prompt[:, -context_length:]
        actual_prompt_len = context.shape[1]
        positions = torch.arange(actual_prompt_len, device=prompt.device).unsqueeze(0).expand(prompt.shape[0], -1)
        logits, past_kv_list = model(context, token_positions=positions, use_cache=True)
        next_logits = logits[:, -1, :]
        next_token = sample_next_token(next_logits, temperature, top_k)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            if squeeze:
                generated = generated.squeeze(0)
            return generated
        generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

        # Phase 2: Decode — one token at a time with KV cache
        for step in range(1, max_new_tokens):
            cur_pos = actual_prompt_len + step - 1
            if cur_pos >= context_length:
                break
            token_input = next_token.unsqueeze(-1)
            positions = torch.full(
                (prompt.shape[0], 1), cur_pos, dtype=torch.long, device=prompt.device
            )
            logits, past_kv_list = model(
                token_input, token_positions=positions, past_kv_list=past_kv_list, use_cache=True
            )
            next_logits = logits[:, -1, :]
            next_token = sample_next_token(next_logits, temperature, top_k)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

    if squeeze:
        generated = generated.squeeze(0)
    return generated
