"""Loss functions."""

import torch


def cross_entropy(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute average cross-entropy loss for next-token prediction.

    Args:
        outputs: Logits of shape (..., vocab_size).
        targets: Target token IDs of shape (...).

    Returns:
        Scalar loss.
    """
    outputs = outputs.view(-1, outputs.size(-1))
    targets = targets.view(-1)
    max_v = torch.max(outputs, dim=-1, keepdim=True).values
    log_sum_exp = (
        torch.log(torch.sum(torch.exp(outputs - max_v), dim=-1, keepdim=True)) + max_v
    )
    log_probs = outputs - log_sum_exp
    rows = torch.arange(outputs.size(0), device=outputs.device)
    loss = -log_probs[rows, targets]
    return loss.mean()
