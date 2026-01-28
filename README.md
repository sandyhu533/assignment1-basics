# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

---

## Project Structure

```
assignment1-basics/
├── cs336_basics/           # Main library
│   ├── data/               # Data utilities
│   │   ├── batch.py        # get_batch - sample LM sequences
│   │   └── checkpoint.py   # save/load_checkpoint
│   ├── model/              # Model components
│   │   ├── layers.py       # Linear, Embedding, RMSNorm, Swiglu, RoPE, Attention
│   │   ├── transformer.py  # TransformerBlock, TransformerLM
│   │   ├── optimizer.py    # SGD, AdamW, gradient_clipping
│   │   └── loss.py         # cross_entropy
│   ├── tokenizer/          # BPE tokenizer
│   │   ├── bpe_tokenizer.py
│   │   ├── bpe_trainer.py
│   │   └── linkedlist.py
│   ├── generation.py      # Text generation (sample_next_token, generate)
│   ├── train.py            # Training logic + CLI
│   └── gen.py              # Generation CLI
├── configs/                # Training configs (optional)
│   ├── train_tiny.yaml
│   └── train_owt.yaml
├── scripts/                # CLI entry points
│   ├── train.py
│   ├── generate.py
│   ├── train_bpe.py
│   └── serialize.py
├── tests/                  # Unit tests
├── data/                   # Data and model outputs
│   ├── _model/             # BPE tokenizers
│   └── _serialized/        # Tokenized .npy files
├── run.sh                  # Unified CLI for all commands
└── pyproject.toml
```

---

## Setup

### Environment

We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv` / `brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies).

You can run any code in the repo using:

```sh
uv run <python_file_path>
```

and the environment will be automatically solved and activated when necessary.

### Run Unit Tests

```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the functions in [./tests/adapters.py](./tests/adapters.py).

---

## Usage

### Quick Start with `run.sh`

The unified `run.sh` script provides commands for the full pipeline:

```sh
chmod +x run.sh
./run.sh help
```

### 1. Train BPE Tokenizer

Downloads data (if needed) and trains a BPE tokenizer:

```sh
# TinyStories (default, vocab_size=10000)
./run.sh bpe tinystories

# OpenWebText (vocab_size=32000)
./run.sh bpe owt
```

Output: `data/_model/<dataset>.result.pkl` and `.result.json`

### 2. Serialize Text to Token IDs

Convert raw text to a NumPy array of token IDs:

```sh
./run.sh serialize \
  data/_model/TinyStoriesV2-GPT4-train.txt.result.pkl \
  data/TinyStoriesV2-GPT4-train.txt \
  data/_serialized/tinystories_train.npy
```

### 3. Train Transformer LM

Train a causal language model on tokenized data:

```sh
# From scratch
uv run python -m cs336_basics.train \
  --input_file data/_serialized/tinystories_train.npy \
  --checkpoint_dir ckpt \
  --load_checkpoint 0 \
  --batch_size 32 \
  --context_length 128 \
  --d_model 256 \
  --num_layers 4 \
  --train_steps 1000

# Resume from checkpoint
uv run python -m cs336_basics.train \
  --input_file data/_serialized/tinystories_train.npy \
  --checkpoint_dir ckpt \
  --load_checkpoint 1
```

**Training arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_file` | required | Path to `.npy` tokenized data |
| `--checkpoint_dir` | required | Path to save/load checkpoint |
| `--load_checkpoint` | 0 | 1 to resume, 0 to train from scratch |
| `--batch_size` | 32 | Batch size |
| `--context_length` | 128 | Context length |
| `--device` | cpu | Device (cpu, cuda, mps) |
| `--d_model` | 256 | Model dimension |
| `--num_layers` | 4 | Number of transformer layers |
| `--num_heads` | 4 | Number of attention heads |
| `--d_ff` | 4*d_model/3 | FFN dimension |
| `--rope_theta` | 10000 | RoPE theta |
| `--train_steps` | 1000 | Number of training steps |
| `--lr` | 1e-3 | Learning rate |
| `--b1`, `--b2` | 0.9, 0.99 | Adam betas |
| `--weight_decay` | 0.01 | Weight decay |
| `--eps` | 1e-8 | Adam epsilon |

### 4. Generate Text

Generate text from a trained checkpoint:

```sh
uv run python -m cs336_basics.gen \
  --checkpoint ckpt \
  --tokenizer data/_model/TinyStoriesV2-GPT4-train.txt.result.pkl \
  --vocab_size 10000 \
  --context_length 128 \
  --prompt "Once upon a time" \
  --max_tokens 100 \
  --temperature 0.8
```

**Generation arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | required | Path to checkpoint |
| `--tokenizer` | required | Path to tokenizer `.pkl` |
| `--prompt` | "Once upon a time" | Text prompt |
| `--max_tokens` | 100 | Max tokens to generate |
| `--temperature` | 0.8 | Sampling temperature |
| `--top_k` | None | Top-k sampling (optional) |
| `--vocab_size` | required | Must match training |
| `--context_length` | required | Must match training |
| `--d_model`, `--num_layers`, etc. | - | Must match training config |

### Console Scripts (after `uv sync`)

```sh
uv run cs336-train --input_file data.npy --checkpoint_dir ckpt --load_checkpoint 0
uv run cs336-generate --checkpoint ckpt --tokenizer tok.pkl --vocab_size 10000 --context_length 128
```

---

## API Overview

### Data (`cs336_basics.data`)

- **`get_batch(dataset, batch_size, context_length, device)`** – Sample random (input, label) sequences for causal LM.
- **`save_checkpoint(model, optimizer, iteration, out)`** – Save model, optimizer, and step.
- **`load_checkpoint(src, model, optimizer)`** – Load checkpoint; returns iteration.

### Model (`cs336_basics.model`)

- **Layers:** `Linear`, `Embedding`, `RMSNorm`, `Swiglu`, `RotaryPositionalEmbedding`, `CausalMultiHeadSelfAttention`
- **Transformer:** `TransformerBlock`, `TransformerLM`
- **Optimizer:** `SGD`, `AdamW`, `get_lr_learning_rate`, `gradient_clipping`
- **Loss:** `cross_entropy`

### Tokenizer (`cs336_basics.tokenizer`)

- **`BPETokenizer(vocab, merges, special_tokens)`** – BPE tokenizer.
- **`BPETrainer()`** – BPE training.
- **`load_tokenizer_pickle(filename)`** – Load vocab and merges from `.pkl`.

### Generation (`cs336_basics.generation`)

- **`sample_next_token(logits, temperature, top_k)`** – Sample from logits.
- **`generate(model, prompt, max_new_tokens, temperature, top_k, eos_token_id)`** – Autoregressive generation.

---

## Download Data Manually

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

---

## Full Pipeline Example

```sh
# 1. Train BPE on TinyStories
./run.sh bpe tinystories

# 2. Serialize train split
./run.sh serialize \
  data/_model/TinyStoriesV2-GPT4-train.txt.result.pkl \
  data/TinyStoriesV2-GPT4-train.txt \
  data/_serialized/tinystories_train.npy

# 3. Train model (small config for quick test)
uv run python -m cs336_basics.train \
  --input_file data/_serialized/tinystories_train.npy \
  --checkpoint_dir ckpt \
  --load_checkpoint 0 \
  --batch_size 16 \
  --context_length 64 \
  --d_model 128 \
  --num_layers 2 \
  --train_steps 100

# 4. Generate
uv run python -m cs336_basics.gen \
  --checkpoint ckpt \
  --tokenizer data/_model/TinyStoriesV2-GPT4-train.txt.result.pkl \
  --vocab_size 10000 \
  --context_length 64 \
  --d_model 128 \
  --num_layers 2 \
  --prompt "Once upon a time" \
  --max_tokens 50
```

---

## License

See [LICENSE](./LICENSE).
