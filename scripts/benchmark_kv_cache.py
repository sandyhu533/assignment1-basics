"""
Benchmark naive vs KV-cached generation.

Measures wall-clock time and tokens/second for both methods across
different generation lengths.  Prints a summary table and optionally
saves a speedup plot.

Usage:
    python scripts/benchmark_kv_cache.py [--device cpu] [--plot speedup.png]
"""

import argparse
import time

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.generation import generate, generate_with_kv_cache


def build_model(device: str) -> TransformerLM:
    """Build a small model for benchmarking."""
    model = TransformerLM(
        vocab_size=512,
        context_length=1024,
        d_model=128,
        num_layer=4,
        num_heads=4,
        d_ff=256,
        rope_theta=10000.0,
        device=device,
    )
    model.eval()
    return model


def benchmark_generate(model, prompt, max_new_tokens, method, n_warmup=1, n_runs=3):
    """Time a generation method over multiple runs, return avg seconds."""
    fn = generate if method == "naive" else generate_with_kv_cache
    for _ in range(n_warmup):
        fn(model, prompt.clone(), max_new_tokens=max_new_tokens, temperature=0.0)
    if torch.cuda.is_available() and prompt.is_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        if torch.cuda.is_available() and prompt.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(model, prompt.clone(), max_new_tokens=max_new_tokens, temperature=0.0)
        if torch.cuda.is_available() and prompt.is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV cache generation speedup")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gen-lengths", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--plot", type=str, default=None, help="Save speedup plot to this path")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")
    print(f"Prompt length: {args.prompt_len}")
    print(f"Warmup runs: {args.n_warmup}, Timed runs: {args.n_runs}")
    print()

    torch.manual_seed(42)
    model = build_model(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print()

    prompt = torch.randint(0, 512, (1, args.prompt_len), device=device)

    header = f"{'Gen Len':>8} | {'Naive (s)':>10} | {'KV Cache (s)':>12} | {'Speedup':>8} | {'Naive tok/s':>12} | {'Cache tok/s':>12}"
    print(header)
    print("-" * len(header))

    results = []
    for gen_len in args.gen_lengths:
        t_naive = benchmark_generate(model, prompt, gen_len, "naive", args.n_warmup, args.n_runs)
        t_cache = benchmark_generate(model, prompt, gen_len, "kv_cache", args.n_warmup, args.n_runs)
        speedup = t_naive / t_cache if t_cache > 0 else float("inf")
        tok_naive = gen_len / t_naive
        tok_cache = gen_len / t_cache
        results.append((gen_len, t_naive, t_cache, speedup))
        print(f"{gen_len:>8} | {t_naive:>10.4f} | {t_cache:>12.4f} | {speedup:>7.2f}x | {tok_naive:>12.1f} | {tok_cache:>12.1f}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            gen_lens = [r[0] for r in results]
            speedups = [r[3] for r in results]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.plot(gen_lens, [r[1] for r in results], "o-", label="Naive")
            ax1.plot(gen_lens, [r[2] for r in results], "s-", label="KV Cache")
            ax1.set_xlabel("Generation Length (tokens)")
            ax1.set_ylabel("Time (seconds)")
            ax1.set_title("Generation Time")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(gen_lens, speedups, "D-", color="green")
            ax2.set_xlabel("Generation Length (tokens)")
            ax2.set_ylabel("Speedup (x)")
            ax2.set_title("KV Cache Speedup")
            ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.plot, dpi=150)
            print(f"\nPlot saved to {args.plot}")
        except ImportError:
            print("\nmatplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()
