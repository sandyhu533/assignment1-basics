#!/bin/bash

# 确保在根目录执行，以便找到 cs336_basics 包
export PYTHONPATH=.

# --- 配置路径 ---
MODEL_DIR="data/_model"
DATA_DIR="data"
OUT_DIR="data/_serialized"

mkdir -p $OUT_DIR

# --- 任务 1: TinyStories ---
echo "--- Serializing TinyStories (train)---"
python cs336_basics/bpe_tokenizer.py \
    --tokenizer_model "$MODEL_DIR/TinyStoriesV2-GPT4-train.txt.result.pkl" \
    --input_file "$DATA_DIR/TinyStoriesV2-GPT4-train.txt" \
    --output_file "$OUT_DIR/tinystories_train.npy"

echo "--- Serializing TinyStories (valid) ---"
python cs336_basics/bpe_tokenizer.py \
    --tokenizer_model "$MODEL_DIR/TinyStoriesV2-GPT4-train.txt.result.pkl" \
    --input_file "$DATA_DIR/TinyStoriesV2-GPT4-valid.txt" \
    --output_file "$OUT_DIR/tinystories_valid.npy"

# --- 任务 2: OpenWebText (OWT) ---
echo "--- Serializing OpenWebText (train) ---"
python cs336_basics/bpe_tokenizer.py \
    --tokenizer_model "$MODEL_DIR/owt_train.txt.result.pkl" \
    --input_file "$DATA_DIR/owt_train.txt" \
    --output_file "$OUT_DIR/owt_train.npy"

echo "--- Serializing OpenWebText (valid) ---"
python cs336_basics/bpe_tokenizer.py \
    --tokenizer_model "$MODEL_DIR/owt_train.txt.result.pkl" \
    --input_file "$DATA_DIR/owt_valid.txt" \
    --output_file "$OUT_DIR/owt_valid.npy"