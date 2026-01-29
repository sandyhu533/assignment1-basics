"""
Training utilities and CLI.

Re-exports get_batch, save_checkpoint, load_checkpoint from cs336_basics.data
for backward compatibility with tests/adapters.py.
"""

import argparse
import json
import os
import re

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
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help="Vocab size (from tokenizer). If not set, computed from data (slower for large files)",
    )
    parser.add_argument(
        "--mmap",
        action="store_true",
        default=True,
        help="Use memory-mapped loading for large datasets (default: True)",
    )
    parser.add_argument("--no-mmap", action="store_false", dest="mmap", help="Load full array into RAM")
    parser.add_argument(
        "--run_note",
        type=str,
        default="",
        help="Note for this run (e.g. 'baseline', 'lr=1e-4'). Used in loss plot title/legend and log filename.",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Do not save loss plot (only save loss log JSON)",
    )

    args = parser.parse_args()
    d_ff = args.d_ff or (4 * args.d_model // 3)

    # Memory-efficient loading with np.memmap for large datasets
    if args.mmap:
        data = np.load(args.input_file, mmap_mode="r", allow_pickle=False)
    else:
        data = np.load(args.input_file, allow_pickle=True)
    if data.ndim > 1:
        data = data.ravel()
    if args.vocab_size is not None:
        vocab_size = args.vocab_size
    else:
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

    steps_log = []
    losses_log = []

    for _ in range(args.train_steps):
        model.zero_grad()
        step += 1
        outputs = model(inputs)
        loss = cross_entropy(outputs, labels)
        loss_val = loss.item()
        steps_log.append(step)
        losses_log.append(loss_val)
        print(f"step={step} loss={loss_val:.4f}")
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

    # Save loss log and plot
    note_safe = re.sub(r"[^\w\-]", "_", args.run_note)[:32] if args.run_note else ""
    log_name = f"train_loss_{note_safe}.json" if note_safe else "train_loss.json"
    plot_name = f"loss_{note_safe}.png" if note_safe else "loss.png"
    log_path = os.path.join(ckpt_dir, log_name) if ckpt_dir else log_name
    plot_path = os.path.join(ckpt_dir, plot_name) if ckpt_dir else plot_name

    loss_log = {
        "run_note": args.run_note or "(no note)",
        "steps": steps_log,
        "losses": losses_log,
        "checkpoint_dir": args.checkpoint_dir,
    }
    with open(log_path, "w") as f:
        json.dump(loss_log, f, indent=2)
    print(f"Loss log saved to {log_path}")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            plt.plot(steps_log, losses_log, color="C0", linewidth=1)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            title = f"Training Loss ({args.run_note})" if args.run_note else "Training Loss"
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Loss plot saved to {plot_path}")
        except Exception as e:
            print(f"Could not save loss plot: {e}")


if __name__ == "__main__":
    main()
