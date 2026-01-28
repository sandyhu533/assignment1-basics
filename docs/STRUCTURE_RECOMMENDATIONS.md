# cs336_basics 项目结构与脚本建议

基于当前 `cs336_basics` 目录，参考常见开源 ML 项目（如 Hugging Face、minGPT、nanoGPT）的惯例，给出以下建议。

---

## 一、当前状态简要

| 文件 | 职责 | 问题 |
|------|------|------|
| `model.py` | 模型定义 + 可能含 `__main__` | 单文件过大，职责混合 |
| `train.py` | 训练循环 + `get_batch`/checkpoint + CLI | 数据、训练、配置、入口混在一起 |
| `bpe_trainer.py` | BPE 训练 + CLI | 入口与库逻辑可分离 |
| `bpe_tokenizer.py` | 分词器 + 序列化 CLI | 同上 |
| `gen.py` | 空 | 待实现推理/生成入口 |
| `run_*.sh` | 下载 + 调 Python | 可收敛到统一入口或 Makefile |

---

## 二、推荐目录结构

```
assignment1-basics/
├── cs336_basics/           # 库代码（可被 import）
│   ├── __init__.py
│   ├── data/               # 数据相关（可选子包）
│   │   __init__.py
│   │   batch.py            # get_batch, 数据加载工具
│   │   checkpoint.py       # save/load_checkpoint
│   ├── model/
│   │   __init__.py         # 导出 TransformerLM, AdamW, cross_entropy 等
│   │   layers.py           # Linear, Embedding, RMSNorm, Swiglu, RoPE 等
│   │   transformer.py     # TransformerBlock, TransformerLM
│   │   optimizer.py        # AdamW（若与作业一致可保留在 model 或单独）
│   ├── bpe_tokenizer.py    # 保持，或迁到 tokenizer/
│   ├── bpe_trainer.py      # 保持，或迁到 tokenizer/
│   └── linkedlist.py      # 保持
├── scripts/                # 可执行入口（推荐）
│   ├── train.py            # 训练 CLI：解析参数 → 调 cs336_basics
│   ├── train_bpe.py        # BPE 训练 CLI（或保留在 cs336_basics 用 -m 调用）
│   ├── serialize.py       # 文本 → .npy 序列化 CLI
│   └── generate.py         # 推理/生成 CLI（对应 gen.py 的职责）
├── configs/                # 配置（可选）
│   ├── train_tiny.yaml
│   └── train_owt.yaml
├── data/                   # 已有
├── tests/
└── pyproject.toml          # 可增加 [project.scripts] 入口
```

要点：

- **库**：`cs336_basics` 只放「可复用、可测试」的逻辑，尽量少 `argparse`/`if __name__`。
- **脚本**：`scripts/` 下用薄层 CLI 调用库；或继续用 `python -m cs336_basics.xxx`，但入口与核心逻辑分离。
- **配置**：超参、路径等集中到 `configs/` 或环境变量，避免在 `train.py` 里写死。

---

## 三、文件拆分建议

### 1. 训练相关（train）

- **从 `train.py` 抽离**：
  - `get_batch` → 独立模块，如 `cs336_basics/data/batch.py` 或 `cs336_basics/data/__init__.py`。
  - `save_checkpoint` / `load_checkpoint` → `cs336_basics/data/checkpoint.py`（或 `training/checkpoint.py`）。
  - 超参（batch_size、context_length、d_model、lr 等）→ 配置文件或 `argparse` 默认值，不要散落在循环附近。
- **保留在“训练入口”里**：
  - 解析参数、读配置、构造 DataLoader/循环、打印、保存 checkpoint 路径等。

这样测试可以只 `from cs336_basics.data.batch import get_batch`，无需起 CLI。

### 2. 模型相关（model）

- **按职责拆分**（在保持与 `tests/` 兼容的前提下）：
  - `model/layers.py`：Linear, Embedding, RMSNorm, Swiglu, RoPE, Attention, FFN 等。
  - `model/transformer.py`：TransformerBlock, TransformerLM。
  - `model/optimizer.py`：AdamW（若作业要求单独文件可保留）。
  - `model/__init__.py`：对外暴露 `TransformerLM`, `AdamW`, `cross_entropy` 等，保证 `from cs336_basics.model import TransformerLM` 仍可用。
- **`model.py`**：可保留为“兼容层”，内部 `from .model.layers import ...` 再 re-export，或逐步把测试改为从 `model` 子包导入。

### 3. BPE

- **维持现状也可**：`bpe_trainer.py` / `bpe_tokenizer.py` 作为库 + `if __name__` 入口。
- **若希望统一入口**：在 `scripts/` 下增加 `train_bpe.py`、`serialize.py`，内部只做 `argparse` 并调用 `cs336_basics.bpe_trainer` / `bpe_tokenizer`。

### 4. 生成（gen）

- **`gen.py` 的职责**：推理/解码入口（加载 checkpoint、读配置、逐 token 生成）。
- **建议**：
  - 生成逻辑（temperature、top_p、采样循环）放在 `cs336_basics/` 里（如 `generation.py` 或 `model/generate.py`）。
  - `scripts/generate.py` 或 `cs336_basics/gen.py` 只做 CLI：解析参数 → 调库 → 写结果。

---

## 四、脚本与入口约定

### 方式 A：保持 `python -m cs336_basics.xxx`（最小改动）

- 训练：`python -m cs336_basics.train`
- BPE 训练：`python -m cs336_basics.bpe_trainer`
- 序列化：`python -m cs336_basics.bpe_tokenizer`（或现有脚本）
- 生成：`python -m cs336_basics.gen`（在 gen 里实现 `main()`）

各模块内保持 `def main()` + `if __name__ == "__main__": main()`，参数用 `argparse`。

### 方式 B：统一放到 `scripts/` + 控制台入口（更常见）

在 `pyproject.toml` 中增加：

```toml
[project.scripts]
cs336-train = "scripts.train:main"
cs336-train-bpe = "scripts.train_bpe:main"
cs336-serialize = "scripts.serialize:main"
cs336-generate = "scripts.generate:main"
```

安装后可直接在终端执行 `cs336-train ...`、`cs336-generate ...`。  
若不想改 pyproject，也可用裸脚本：

```bash
PYTHONPATH=. python scripts/train.py --input_file data.npy --checkpoint_dir ckpt
```

### 方式 C：Shell 收敛

- 将 `run_train_bpe.sh`、`run_seralize.sh` 等统一为一个 `run.sh`，用子命令区分，例如：
  - `./run.sh bpe tinystories`
  - `./run.sh train --config configs/train_tiny.yaml`
- 或使用 `Makefile` 的 target（e.g. `make train-bpe`, `make train`），内部再调上述 Python 命令。

---

## 五、配置管理建议

- **训练超参**：用 YAML/JSON 或 `argparse` 的 `default` 集中管理，避免在 `train.py` 中间写死 `batch_size=1`、`context_length=10` 等。
- **路径**：`data/`、`checkpoint_dir`、`output_dir` 可通过环境变量或配置文件指定，便于本地/集群一致。
- **可选**：使用 `omegaconf` 或 `tyro` 做 CLI+配置文件一体化（与作业无冲突即可）。

---

## 六、实施优先级（建议顺序）

1. **数据与 checkpoint 抽离**：`get_batch`、`save/load_checkpoint` 移到独立模块，训练脚本只做调用。  
2. **训练入口参数化**：所有超参和路径通过 argparse 或配置文件传入，去掉硬编码。  
3. **gen 实现**：在库中实现生成逻辑，`gen.py` 或 `scripts/generate.py` 做薄 CLI。  
4. **model 子包化**：在保证 tests 通过的前提下，将 `model.py` 拆成 `model/layers.py`、`model/transformer.py` 等。  
5. **脚本与 Shell 统一**：按需引入 `scripts/` 与 `pyproject.toml` 的 `[project.scripts]`，或统一 `run.sh`/Makefile。

按上述方式拆分后，库与脚本边界清晰，测试可只针对库函数，配置与运行方式也更接近常见开源项目。
