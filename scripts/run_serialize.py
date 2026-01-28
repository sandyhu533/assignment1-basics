#!/usr/bin/env python3
"""
Serialize text to token IDs. Reads config from YAML or CLI args.
Usage:
  python scripts/run_serialize.py <dataset> [split]
    dataset: tinystories | owt
    split: train | valid (default: train)
  Or with explicit paths:
  python scripts/run_serialize.py --tokenizer <pkl> --input <txt> --output <npy>
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_paths() -> dict:
    """Load paths.yaml."""
    config_path = Path(__file__).parent.parent / "configs" / "paths.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Serialize text to token IDs")
    parser.add_argument("dataset", nargs="?", help="Dataset name (tinystories|owt)")
    parser.add_argument("split", nargs="?", default="train", help="train or valid")
    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer .pkl")
    parser.add_argument("--input", type=str, help="Path to input .txt")
    parser.add_argument("--output", type=str, help="Path to output .npy")
    args = parser.parse_args()

    if args.tokenizer and args.input and args.output:
        tokenizer_path = args.tokenizer
        input_path = args.input
        output_path = args.output
    elif args.dataset and args.dataset in ("tinystories", "owt"):
        paths_config = load_paths()
        ds = paths_config["datasets"][args.dataset]
        tokenizer_path = ds["tokenizer_pkl"]
        input_path = ds["raw_train"] if args.split == "train" else ds["raw_valid"]
        output_path = ds["tokenized_train"] if args.split == "train" else ds["tokenized_valid"]
    else:
        parser.error("Either provide (dataset [split]) or (--tokenizer --input --output)")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    sys.argv = [
        "serialize",
        "--tokenizer_model", tokenizer_path,
        "--input_file", input_path,
        "--output_file", output_path,
    ]
    from cs336_basics.tokenizer.bpe_tokenizer import main as serialize_main

    serialize_main()


if __name__ == "__main__":
    main()
