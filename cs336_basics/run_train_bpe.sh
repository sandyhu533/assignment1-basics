#!/bin/bash

# --- 1. 配置参数 ---
DATA_SET=${1:-"tinystories"} # 默认运行 tinystories，或者传参 ./run.sh owt
OUTPUT_DIR="data/_model"
INPUT_DIR="data"

# --- 2. 定义数据源 ---
case $DATA_SET in
    "tinystories")
        URL="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
        DATA_PATH="$INPUT_DIR/TinyStoriesV2-GPT4-train.txt"
        IS_GZ=false
        VOCAB_SIZE=10000
        ;;
    "owt")
        URL="https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"
        DATA_PATH="$INPUT_DIR/owt_train.txt"
        IS_GZ=true
        VOCAB_SIZE=32000
        ;;
    *)
        echo "Unknown dataset: $DATA_SET. Use 'owt' or 'tinystories'."
        exit 1
        ;;
esac

# --- 3. 自动化下载与解压逻辑 ---
echo "Checking dataset: $DATA_PATH ..."

if [ ! -f "$DATA_PATH" ]; then
    if [ "$IS_GZ" = true ]; then
        if [ ! -f "$DATA_PATH.gz" ]; then
            echo "Downloading $DATA_PATH.gz ..."
            wget -O "$DATA_PATH.gz" "$URL"
        fi
        echo "Decompressing $DATA_PATH.gz ..."
        gunzip -f "$DATA_PATH.gz"
    else
        echo "Downloading $DATA_PATH ..."
        wget -O "$DATA_PATH" "$URL"
    fi
else
    echo "Dataset already exists, skipping download."
fi

# --- 4. 运行训练脚本 ---
echo "Starting BPE training..."
# 使用 PYTHONPATH=. 确保模块导入正确
PYTHONPATH=. python cs336_basics/bpe_trainer.py \
    --dataset "$DATA_PATH" \
    --vocab_size $VOCAB_SIZE \
    --output_dir "$OUTPUT_DIR"

echo "Done!"