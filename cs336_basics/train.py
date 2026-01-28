"""
Training utilities and CLI.

Re-exports get_batch, save_checkpoint, load_checkpoint from cs336_basics.data
for backward compatibility with tests/adapters.py.
"""

import argparse
import numpy as np
import torch

from cs336_basics.data import get_batch, save_checkpoint, load_checkpoint
from cs336_basics.model import TransformerLM, AdamW, cross_entropy

# Re-export for adapters
__all__ = ["get_batch", "save_checkpoint", "load_checkpoint", "main"]


def main():
    parser = argparse.ArgumentParser(description="Train Transformer LLM.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to .npy tokenized data")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to save checkpoint")
    parser.add_argument(
        "--load_checkpoint",
        type=int,
        default=0,
        choices=[0, 1],
        help="1 to resume from checkpoint, 0 to train from scratch",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--context_length", type=int, default=128, help="Context length")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=None, help="FFN dimension (default: 4*d_model/3)")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta")
    parser.add_argument("--train_steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--b2", type=float, default=0.99, help="Adam beta2")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")

    args = parser.parse_args()
    d_ff = args.d_ff or (4 * args.d_model // 3)

    data = np.load(args.input_file, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.ndim > 1:
        data = data.ravel()
    else:
        data = np.asarray(data).ravel()
    vocab_size = int(np.max(data)) + 1

    inputs, labels = get_batch(
        data, args.batch_size, args.context_length, args.device
    )

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layer=args.num_layers,
        num_heads=args.num_heads,
        d_ff=d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
        dtype=torch.float32,
    )
    optimizer = AdamW(
        model.parameters(),
        (args.b1, args.b2),
        args.weight_decay,
        args.eps,
        args.lr,
    )

    step = 0
    if args.load_checkpoint == 1:
        step = load_checkpoint(args.checkpoint_dir, model, optimizer)

    for _ in range(args.train_steps):
        model.zero_grad()
        step += 1
        outputs = model(inputs)
        loss = cross_entropy(outputs, labels)
        print(f"step={step} loss={loss.item():.4f}")
        loss.backward()
        optimizer.step()
        inputs, labels = get_batch(
            data, args.batch_size, args.context_length, args.device
        )

    ckpt_dir = os.path.dirname(args.checkpoint_dir)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    save_checkpoint(model, optimizer, step, args.checkpoint_dir)
    print(f"Checkpoint saved at step {step}")


if __name__ == "__main__":
    main()
