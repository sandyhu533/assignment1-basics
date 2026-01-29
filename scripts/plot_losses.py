#!/usr/bin/env python3
"""
Plot training loss from one or more train_loss*.json files for comparison.
Usage:
  python scripts/plot_losses.py checkpoints/tinystories_small/train_loss.json
  python scripts/plot_losses.py checkpoints/tinystories_small/train_loss_*.json
  python scripts/plot_losses.py checkpoints/tinystories_small/  # all train_loss*.json in dir
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_log(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Plot training loss (single or multiple runs for comparison)"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to train_loss*.json files or a directory containing them",
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output plot path")
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    args = parser.parse_args()

    files = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("train_loss*.json")))
        elif path.is_file():
            files.append(path)
        else:
            # glob: e.g. checkpoints/tinystories_small/train_loss_*.json
            parent = path.parent
            found = list(parent.glob(path.name))
            files.extend(sorted(found))
            if not found:
                sys.exit(f"Not found: {p}")

    if not files:
        sys.exit("No train_loss*.json files found.")

    logs = []
    for f in files:
        try:
            logs.append((f, load_log(f)))
        except Exception as e:
            print(f"Skip {f}: {e}", file=sys.stderr)

    if not logs:
        sys.exit("No valid log files loaded.")

    plt.figure(figsize=(10, 6))
    for i, (path, data) in enumerate(logs):
        steps = data.get("steps", [])
        losses = data.get("losses", [])
        note = data.get("run_note", path.stem)
        if not steps or not losses:
            print(f"Empty log: {path}", file=sys.stderr)
            continue
        label = f"{note}" if note and note != "(no note)" else path.stem
        plt.plot(steps, losses, label=label, linewidth=1, alpha=0.9)

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(args.title or "Training Loss Comparison")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = args.output
    if not out and len(logs) == 1:
        out = str(logs[0][0].with_suffix(".png"))
    elif not out:
        out = "loss_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
