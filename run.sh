#!/bin/bash
# Unified run script for BPE training, serialization, model training, and generation.
# Paths are managed via configs/paths.yaml and configs/bpe_*.yaml, train_*.yaml.
# Usage: ./run.sh <command> [args...]
# Commands: bpe | serialize | train | generate | plot

set -e
export PYTHONPATH=.

case "${1:-help}" in
    bpe)
        # Train BPE tokenizer (reads config from configs/bpe_<dataset>.yaml)
        # Usage: ./run.sh bpe [tinystories|owt]
        DATASET="${2:-tinystories}"
        case "$DATASET" in
            tinystories)
                RAW_PATH="data/tinystories/raw/train.txt"
                URL="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
                if [ ! -f "$RAW_PATH" ]; then
                    mkdir -p "$(dirname "$RAW_PATH")"
                    echo "Downloading TinyStories..."
                    wget -O "$RAW_PATH" "$URL"
                fi
                ;;
            owt)
                RAW_PATH="data/owt/raw/train.txt"
                URL="https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"
                if [ ! -f "$RAW_PATH" ]; then
                    mkdir -p "$(dirname "$RAW_PATH")"
                    if [ ! -f "$RAW_PATH.gz" ]; then
                        echo "Downloading OpenWebText..."
                        wget -O "$RAW_PATH.gz" "$URL"
                    fi
                    echo "Decompressing..."
                    gunzip -f "$RAW_PATH.gz"
                fi
                ;;
            *)
                echo "Unknown dataset: $DATASET. Use 'tinystories' or 'owt'."
                exit 1
                ;;
        esac
        echo "Training BPE on $RAW_PATH (config: configs/bpe_${DATASET}.yaml)..."
        python scripts/run_bpe.py "$DATASET" --raw_path "$RAW_PATH"
        echo "Done. Output in data/${DATASET}/tokenizer/"
        ;;
    serialize)
        # Serialize text to .npy (reads paths from configs/paths.yaml)
        # Usage: ./run.sh serialize <dataset> [train|valid]
        #   Or: ./run.sh serialize -- --tokenizer <pkl> --input <txt> --output <npy>
        shift
        if [ "$1" = "--" ]; then
            shift
            python scripts/run_serialize.py "$@"
        elif [ -n "$1" ]; then
            python scripts/run_serialize.py "$1" "${2:-train}"
        else
            echo "Usage: ./run.sh serialize <dataset> [train|valid]"
            echo "   Or: ./run.sh serialize -- --tokenizer <pkl> --input <txt> --output <npy>"
            exit 1
        fi
        echo "Done."
        ;;
    train)
        # Train Transformer LM (reads config from YAML)
        # Usage: ./run.sh train [tinystories|owt|path/to/config.yaml]
        shift
        if [ $# -eq 0 ]; then
            python scripts/run_train.py tinystories
        else
            python scripts/run_train.py "$@"
        fi
        ;;
    generate)
        # Generate text (reads config from YAML)
        # Usage: ./run.sh generate [tinystories|owt|path/to/config.yaml] [--prompt "..." --max_tokens 100]
        shift
        if [ $# -eq 0 ]; then
            python scripts/run_generate.py tinystories
        else
            python scripts/run_generate.py "$@"
        fi
        ;;
    plot)
        # Plot training loss (single run or compare multiple runs)
        # Usage: ./run.sh plot checkpoints/tinystories_small/
        #   or: ./run.sh plot checkpoints/tinystories_small/train_loss_baseline.json checkpoints/tinystories_small/train_loss_v2.json
        shift
        if [ $# -eq 0 ]; then
            echo "Usage: ./run.sh plot <dir_or_json> [dir_or_json ...]"
            echo "  e.g. ./run.sh plot checkpoints/tinystories_small/"
            exit 1
        fi
        python scripts/plot_losses.py "$@"
        ;;
    help|*)
        echo "Usage: ./run.sh <command> [args...]"
        echo ""
        echo "Commands:"
        echo "  bpe [tinystories|owt]       Train BPE tokenizer (downloads data if needed)"
        echo "  serialize <dataset> [split] Serialize text to token IDs (dataset: tinystories|owt, split: train|valid)"
        echo "  train [config]              Train model (config: tinystories|owt or path to YAML)"
        echo "  generate [config]           Generate text (config: tinystories|owt or path to YAML)"
        echo "  plot [dir|json...]          Plot loss (single or compare multiple runs)"
        echo ""
        echo "Examples:"
        echo "  ./run.sh bpe tinystories"
        echo "  ./run.sh serialize tinystories train"
        echo "  ./run.sh train tinystories"
        echo "  ./run.sh train tinystories --note baseline"
        echo "  ./run.sh plot checkpoints/tinystories_small/"
        echo "  ./run.sh generate tinystories --prompt 'Once upon a time' --max_tokens 100"
        echo ""
        echo "Path structure (configs/paths.yaml):"
        echo "  data/<dataset>/raw/         Raw text"
        echo "  data/<dataset>/tokenized/   Tokenized .npy"
        echo "  data/<dataset>/tokenizer/   BPE model (model.pkl, model.json)"
        echo "  checkpoints/<run_name>/     Model checkpoints"
        ;;
esac
