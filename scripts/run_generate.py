#!/usr/bin/env python3
"""
Generation runner. Reads config from YAML and invokes gen.
Usage:
  python scripts/run_generate.py --config configs/generate_tiny.yaml
  python scripts/run_generate.py tinystories [--prompt "..." --max_tokens 100]
  Or with explicit args (same as cs336_basics.gen)
"""

import argparse
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Generate text (reads config from YAML)")
    parser.add_argument("config", nargs="?", help="Config name (tinystories|owt) or path to YAML")
    parser.add_argument("--config", dest="config_path", type=str, help="Path to config YAML")
    parser.add_argument("--prompt", type=str, help="Override prompt")
    parser.add_argument("--max_tokens", type=int, help="Override max_tokens")
    parser.add_argument("--temperature", type=float, help="Override temperature")
    parser.add_argument("--use_kv_cache", action="store_true", help="Use KV cache for generation")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    config_path = args.config_path
    if not config_path and args.config:
        if args.config in ("tinystories", "tiny"):
            config_path = str(root / "configs" / "generate_tiny.yaml")
        elif args.config in ("owt",):
            config_path = str(root / "configs" / "generate_owt.yaml")
        else:
            config_path = args.config
    if not config_path:
        parser.error("Provide config path or name (tinystories|owt)")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    model_config = config["model"]
    gen_config = config.get("generation", {})

    gen_args = [
        "gen",
        "--checkpoint", paths["checkpoint"],
        "--tokenizer", paths["tokenizer"],
        "--vocab_size", str(model_config["vocab_size"]),
        "--context_length", str(model_config["context_length"]),
        "--d_model", str(model_config.get("d_model", 256)),
        "--num_layers", str(model_config.get("num_layers", 4)),
        "--num_heads", str(model_config.get("num_heads", 4)),
        "--rope_theta", str(model_config.get("rope_theta", 10000.0)),
        "--prompt", args.prompt or gen_config.get("prompt", "Once upon a time"),
        "--max_tokens", str(args.max_tokens or gen_config.get("max_tokens", 100)),
        "--temperature", str(args.temperature or gen_config.get("temperature", 0.8)),
        "--device", config.get("device", "cpu"),
    ]
    if model_config.get("d_ff"):
        gen_args.extend(["--d_ff", str(model_config["d_ff"])])
    if gen_config.get("top_k") is not None:
        gen_args.extend(["--top_k", str(gen_config["top_k"])])
    if args.use_kv_cache or gen_config.get("use_kv_cache", False):
        gen_args.append("--use_kv_cache")

    sys.argv = gen_args
    from cs336_basics.gen import main as gen_main

    gen_main()


if __name__ == "__main__":
    main()
