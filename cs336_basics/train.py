"""
Training utilities and CLI.

Re-exports get_batch, save_checkpoint, load_checkpoint from cs336_basics.data
for backward compatibility with tests/adapters.py.
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime

import numpy as np
import torch

from cs336_basics.data import get_batch, save_checkpoint, load_checkpoint
from cs336_basics.model import TransformerLM, AdamW, cross_entropy, get_lr_learning_rate

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
    # Cosine LR schedule with warmup (used by get_lr_learning_rate)
    parser.add_argument(
        "--max_lr",
        type=float,
        default=1e-3,
        help="Peak learning rate after warmup (default: 1e-3)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        help="Minimum learning rate at end of cosine decay (default: 1e-5)",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=100,
        help="Number of warmup steps for linear ramp to max_lr (default: 100)",
    )
    parser.add_argument(
        "--cosine_iters",
        type=int,
        default=None,
        help="Total steps for cosine decay (default: same as train_steps)",
    )

    args = parser.parse_args()
    d_ff = args.d_ff or (4 * args.d_model // 3)
    cosine_iters = args.cosine_iters if args.cosine_iters is not None else args.train_steps

    # Logging: console with timestamp and elapsed time
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("train")
    run_start_time = time.perf_counter()
    run_start_iso = datetime.now().isoformat(timespec="seconds")
    logger.info("Run started at %s", run_start_iso)

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
    if args.device == "mps":
        model = torch.compile(model, backend="aot_eager")
    optimizer = AdamW(
        model.parameters(),
        (args.b1, args.b2),
        args.weight_decay,
        args.eps,
        args.max_lr,
    )

    step = 0
    if args.load_checkpoint == 1:
        step = load_checkpoint(args.checkpoint_dir, model, optimizer)

    steps_log = []
    losses_log = []
    step_details = []  # [{step, loss, elapsed_sec, timestamp_iso}]

    for _ in range(args.train_steps):
        for param_group in optimizer.param_groups:
            param_group["lr"] = get_lr_learning_rate(
                step, args.max_lr, args.min_lr, args.warmup_iters, cosine_iters
            )
        step_start = time.perf_counter()
        model.zero_grad()
        step += 1
        outputs = model(inputs)
        loss = cross_entropy(outputs, labels)
        loss_val = loss.item()
        steps_log.append(step)
        losses_log.append(loss_val)
        elapsed = time.perf_counter() - step_start
        step_details.append({
            "step": step,
            "loss": loss_val,
            "elapsed_sec": round(elapsed, 4),
            "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
        })
        total_elapsed = time.perf_counter() - run_start_time
        logger.info(
            "step=%d loss=%.4f step_sec=%.3f total_sec=%.1f",
            step, loss_val, elapsed, total_elapsed,
        )
        loss.backward()
        optimizer.step()
        inputs, labels = get_batch(
            data, args.batch_size, args.context_length, args.device
        )

    run_end_time = time.perf_counter()
    run_end_iso = datetime.now().isoformat(timespec="seconds")
    total_seconds = round(run_end_time - run_start_time, 2)
    logger.info("Run finished at %s, total time %.1f s", run_end_iso, total_seconds)

    ckpt_dir = os.path.dirname(args.checkpoint_dir)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    save_checkpoint(model, optimizer, step, args.checkpoint_dir)
    logger.info("Checkpoint saved at step %d", step)

    # Serialize run params for logging (skip non-JSON-serializable)
    run_params = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))}

    # Save loss log and plot (with run metadata and per-step details)
    note_safe = re.sub(r"[^\w\-]", "_", args.run_note)[:32] if args.run_note else ""
    log_name = f"train_loss_{note_safe}.json" if note_safe else "train_loss.json"
    plot_name = f"loss_{note_safe}.png" if note_safe else "loss.png"
    log_path = os.path.join(ckpt_dir, log_name) if ckpt_dir else log_name
    plot_path = os.path.join(ckpt_dir, plot_name) if ckpt_dir else plot_name

    loss_log = {
        "run_note": args.run_note or "(no note)",
        "run_params": run_params,
        "start_time_iso": run_start_iso,
        "end_time_iso": run_end_iso,
        "total_seconds": total_seconds,
        "checkpoint_dir": args.checkpoint_dir,
        "steps": steps_log,
        "losses": losses_log,
        "step_details": step_details,
    }
    with open(log_path, "w") as f:
        json.dump(loss_log, f, indent=2)
    logger.info("Loss log saved to %s", log_path)

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
            logger.info("Loss plot saved to %s", plot_path)
        except Exception as e:
            logger.warning("Could not save loss plot: %s", e)


if __name__ == "__main__":
    main()
