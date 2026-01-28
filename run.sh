#!/bin/bash
# Unified run script for BPE training, serialization, model training, and generation.
# Usage: ./run.sh <command> [args...]
# Commands: bpe | serialize | train | generate

set -e
export PYTHONPATH=.

DATA_DIR="${DATA_DIR:-data}"
MODEL_DIR="${MODEL_DIR:-data/_model}"
OUT_DIR="${OUT_DIR:-data/_serialized}"

case "${1:-help}" in
    bpe)
        # Train BPE tokenizer
        # Usage: ./run.sh bpe [tinystories|owt]
        DATASET="${2:-tinystories}"
        mkdir -p "$DATA_DIR" "$MODEL_DIR"
        case "$DATASET" in
            tinystories)
                URL="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
                DATA_PATH="$DATA_DIR/TinyStoriesV2-GPT4-train.txt"
                VOCAB_SIZE=10000
                if [ ! -f "$DATA_PATH" ]; then
                    echo "Downloading TinyStories..."
                    wget -O "$DATA_PATH" "$URL"
                fi
                ;;
            owt)
                URL="https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"
                DATA_PATH="$DATA_DIR/owt_train.txt"
                VOCAB_SIZE=32000
                if [ ! -f "$DATA_PATH" ]; then
                    if [ ! -f "$DATA_PATH.gz" ]; then
                        echo "Downloading OpenWebText..."
                        wget -O "$DATA_PATH.gz" "$URL"
                    fi
                    echo "Decompressing..."
                    gunzip -f "$DATA_PATH.gz"
                fi
                ;;
            *)
                echo "Unknown dataset: $DATASET. Use 'tinystories' or 'owt'."
                exit 1
                ;;
        esac
        echo "Training BPE on $DATA_PATH (vocab_size=$VOCAB_SIZE)..."
        python scripts/train_bpe.py --dataset "$DATA_PATH" --vocab_size $VOCAB_SIZE --output_dir "$MODEL_DIR"
        echo "Done. Output in $MODEL_DIR"
        ;;
    serialize)
        # Serialize text to .npy using tokenizer
        # Usage: ./run.sh serialize <tokenizer.pkl> <input.txt> <output.npy>
        if [ $# -lt 4 ]; then
            echo "Usage: ./run.sh serialize <tokenizer.pkl> <input.txt> <output.npy>"
            exit 1
        fi
        TOKENIZER="$2"
        INPUT="$3"
        OUTPUT="$4"
        mkdir -p "$(dirname "$OUTPUT")"
        python scripts/serialize.py \
            --tokenizer_model "$TOKENIZER" \
            --input_file "$INPUT" \
            --output_file "$OUTPUT"
        echo "Serialized to $OUTPUT"
        ;;
    train)
        # Train Transformer LM
        # Usage: ./run.sh train --input_file <data.npy> --checkpoint_dir <ckpt> [--load_checkpoint 0|1] ...
        shift
        python -m cs336_basics.train "$@"
        ;;
    generate)
        # Generate text from checkpoint
        # Usage: ./run.sh generate --checkpoint <ckpt> --tokenizer <tokenizer.pkl> --prompt "..." ...
        shift
        python -m cs336_basics.gen "$@"
        ;;
    help|*)
        echo "Usage: ./run.sh <command> [args...]"
        echo ""
        echo "Commands:"
        echo "  bpe [tinystories|owt]     Train BPE tokenizer (downloads data if needed)"
        echo "  serialize <tok.pkl> <in.txt> <out.npy>  Serialize text to token IDs"
        echo "  train [args...]           Train model (pass --input_file, --checkpoint_dir, etc.)"
        echo "  generate [args...]        Generate text (pass --checkpoint, --tokenizer, etc.)"
        echo ""
        echo "Examples:"
        echo "  ./run.sh bpe tinystories"
        echo "  ./run.sh serialize data/_model/TinyStoriesV2-GPT4-train.txt.result.pkl data/TinyStoriesV2-GPT4-train.txt data/_serialized/tinystories_train.npy"
        echo "  ./run.sh train --input_file data/_serialized/tinystories_train.npy --checkpoint_dir ckpt --load_checkpoint 0"
        echo "  ./run.sh generate --checkpoint ckpt --tokenizer data/_model/TinyStoriesV2-GPT4-train.txt.result.pkl --vocab_size 10000 --context_length 128 --prompt 'Once upon a time'"
        ;;
esac
