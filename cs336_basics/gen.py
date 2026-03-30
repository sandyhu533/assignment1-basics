"""
Text generation CLI.

Loads a trained checkpoint and tokenizer, then generates text from a prompt.
"""

import argparse
import pickle

import torch

from cs336_basics.model import TransformerLM, AdamW
from cs336_basics.data import load_checkpoint
from cs336_basics.generation import generate, generate_with_kv_cache


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer LM.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer .pkl (vocab, merges)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Text prompt to complete",
    )
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling (optional)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--vocab_size", type=int, required=True, help="Tokenizer vocab size")
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--use_kv_cache", action="store_true", help="Use KV cache for generation")

    args = parser.parse_args()
    d_ff = args.d_ff or (4 * args.d_model // 3)

    with open(args.tokenizer, "rb") as f:
        tokenizer_data = pickle.load(f)
    vocab = tokenizer_data["vocab"]  # id -> bytes
    merges = tokenizer_data["merges"]

    from cs336_basics.tokenizer import BPETokenizer

    tokenizer = BPETokenizer(vocab, merges, ['<|endoftext|>'])
    prompt_ids = tokenizer.encode(args.prompt)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layer=args.num_layers,
        num_heads=args.num_heads,
        d_ff=d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
        dtype=torch.float32,
    )
    optimizer = AdamW(model.parameters(), (0.9, 0.99), 0.01, 1e-8, 1e-3)
    load_checkpoint(args.checkpoint, model, optimizer)

    prompt_tensor = torch.tensor(
        [prompt_ids], dtype=torch.long, device=args.device
    )
    if prompt_tensor.size(-1) > args.context_length:
        prompt_tensor = prompt_tensor[:, -args.context_length:]
    
    eos_token_id = 256
    gen_fn = generate_with_kv_cache if args.use_kv_cache else generate
    output_ids = gen_fn(
        model,
        prompt_tensor,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=eos_token_id
    )
    output_ids = output_ids.cpu().tolist()
    if isinstance(output_ids[0], list):
        output_ids = output_ids[0]

    def decode(ids):
        tokens = []
        for i in ids:
            if i in vocab:
                tokens.append(vocab[i].decode("utf-8", errors="replace"))
            else:
                tokens.append(f"[{i}]")
        return "".join(tokens)

    print(decode(output_ids))


if __name__ == "__main__":
    main()
