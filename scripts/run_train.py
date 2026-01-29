#!/usr/bin/env python3
"""
Training runner. Reads config from YAML and invokes train.
Usage:
  python scripts/run_train.py --config configs/train_tiny.yaml
  python scripts/run_train.py tinystories   # shorthand for train_tiny.yaml
  python scripts/run_train.py owt          # shorthand for train_owt.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Train Transformer LM (reads config from YAML)")
    parser.add_argument("config", nargs="?", help="Config name (tinystories|owt) or path to YAML")
    parser.add_argument("--config", dest="config_path", type=str, help="Path to config YAML")
    parser.add_argument("--load_checkpoint", type=int, default=None, choices=[0, 1], help="Override load_checkpoint")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    config_path = args.config_path
    if not config_path and args.config:
        if args.config in ("tinystories", "tiny"):
            config_path = str(root / "configs" / "train_tiny.yaml")
        elif args.config in ("owt",):
            config_path = str(root / "configs" / "train_owt.yaml")
        else:
            config_path = args.config
    if not config_path:
        parser.error("Provide config path or name (tinystories|owt)")
        return

    if not Path(config_path).exists():
        if args.config and "tinytories" in args.config.lower():
            parser.error(
                f"Unknown config '{args.config}'. Did you mean 'tinystories'? "
                "Use: ./run.sh train tinystories"
            )
        else:
            parser.error(f"Config file not found: {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    input_file = paths["input_file"]
    checkpoint_dir = paths["checkpoint_dir"]
    load_checkpoint = args.load_checkpoint if args.load_checkpoint is not None else config.get("load_checkpoint", 0)

    # Ensure checkpoint dir exists
    Path(checkpoint_dir).parent.mkdir(parents=True, exist_ok=True)

    # Build train args
    vocab_size = config.get("vocab_size")
    train_args = [
        "train",
        "--input_file", input_file,
        "--checkpoint_dir", checkpoint_dir,
        "--load_checkpoint", str(load_checkpoint),
        "--batch_size", str(config.get("batch_size", 32)),
        "--context_length", str(config.get("context_length", 128)),
        "--device", config.get("device", "cpu"),
        "--d_model", str(config.get("d_model", 256)),
        "--num_layers", str(config.get("num_layers", 4)),
        "--num_heads", str(config.get("num_heads", 4)),
        "--rope_theta", str(config.get("rope_theta", 10000.0)),
        "--train_steps", str(config.get("train_steps", 1000)),
        "--lr", str(config.get("lr", 1e-3)),
        "--b1", str(config.get("b1", 0.9)),
        "--b2", str(config.get("b2", 0.99)),
        "--weight_decay", str(config.get("weight_decay", 0.01)),
        "--eps", str(config.get("eps", 1e-8)),
    ]
    if vocab_size is not None:
        train_args.extend(["--vocab_size", str(vocab_size)])
    if config.get("d_ff"):
        train_args.extend(["--d_ff", str(config["d_ff"])])

    sys.argv = train_args
    from cs336_basics.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
