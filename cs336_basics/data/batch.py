"""Batch sampling for language modeling."""

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample language modeling input sequences and labels from a 1D token array.

    Randomly samples batch_size starting indices from [0, size - context_length),
    then extracts (input, label) pairs where labels are inputs shifted by 1.

    Args:
        dataset: 1D numpy array of integer token IDs.
        batch_size: Number of sequences to sample.
        context_length: Length of each input/label sequence.
        device: PyTorch device string (e.g. 'cpu', 'cuda:0').

    Returns:
        Tuple of (inputs, labels), each of shape (batch_size, context_length).
    """
    # Avoid np.asarray on memmap (would copy entire array to RAM)
    if hasattr(dataset, "ravel"):
        data = dataset.ravel() if dataset.ndim > 1 else dataset
    else:
        data = np.asarray(dataset).ravel()
    size = data.size
    num_starts = max(0, size - context_length)
    batch_size = min(batch_size, num_starts)

    if batch_size <= 0:
        inputs = torch.zeros(0, context_length, dtype=torch.long, device=device)
        labels = torch.zeros(0, context_length, dtype=torch.long, device=device)
        return inputs, labels

    start_indices = np.random.choice(num_starts, size=batch_size, replace=True)
    x_batch = np.stack([data[i : i + context_length] for i in start_indices])
    y_batch = np.stack([data[i + 1 : i + 1 + context_length] for i in start_indices])

    inputs = torch.from_numpy(x_batch.astype(np.int64)).to(device)
    labels = torch.from_numpy(y_batch.astype(np.int64)).to(device)
    return inputs, labels
