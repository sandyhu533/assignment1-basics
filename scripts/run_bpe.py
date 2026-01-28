#!/usr/bin/env python3
"""
BPE training runner. Reads config from YAML and invokes bpe_trainer.
Usage: python scripts/run_bpe.py <dataset>
  dataset: tinystories | owt
"""

import argparse
import sys
from pathlib import Path

import yaml

from cs336_basics.tokenizer.bpe_trainer import BPETrainer, save_tokenizer_json, save_tokenizer_pickle


def load_config(dataset: str) -> dict:
    """Load BPE config for dataset."""
    config_path = Path(__file__).parent.parent / "configs" / f"bpe_{dataset}.yaml"
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer (reads config from YAML)")
    parser.add_argument("dataset", choices=["tinystories", "owt"], help="Dataset name")
    parser.add_argument("--raw_path", type=str, default=None, help="Override raw input path")
    args = parser.parse_args()

    config = load_config(args.dataset)
    paths = config["paths"]
    raw_input = args.raw_path or paths["raw_input"]
    output_dir = paths["tokenizer_output_dir"]
    vocab_size = config["vocab_size"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    bpe = BPETrainer()
    print(f"Training on {raw_input} with vocab size {vocab_size}...")
    token2byte, merges = bpe.train(
        raw_input,
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
    )

    # Save as model.pkl and model.json
    save_path = Path(output_dir) / "model"
    save_tokenizer_json(token2byte, merges, str(save_path) + ".json")
    save_tokenizer_pickle(token2byte, merges, str(save_path) + ".pkl")
    print(f"Successfully saved to {save_path}.json and {save_path}.pkl")


if __name__ == "__main__":
    main()
