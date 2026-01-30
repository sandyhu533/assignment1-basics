"""Checkpoint save/load utilities."""

import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Serialize model, optimizer, and iteration to disk.

    Args:
        model: Model whose state_dict to save.
        optimizer: Optimizer whose state_dict to save.
        iteration: Training iteration count.
        out: Path or file-like object to write to.
    """
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)


def _strip_compile_prefix(state_dict: dict) -> dict:
    """Remove '_orig_mod.' prefix from state_dict keys (from torch.compile)."""
    prefix = "_orig_mod."
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k.removeprefix(prefix): v for k, v in state_dict.items()}


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Restore model and optimizer from a checkpoint.

    Args:
        src: Path or file-like object to load from.
        model: Model to restore state into.
        optimizer: Optimizer to restore state into.

    Returns:
        The iteration count stored in the checkpoint.
    """
    obj = torch.load(src, weights_only=False)
    model_state = _strip_compile_prefix(obj["model"])
    model.load_state_dict(model_state)
    optimizer_state = _strip_compile_prefix(obj["optimizer"])
    optimizer.load_state_dict(optimizer_state)
    return obj["iteration"]
