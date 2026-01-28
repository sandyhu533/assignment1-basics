# 路径统一管理执行计划

## 一、目标

1. 将 BPE trainer/tokenizer 涉及的路径统一放入 YAML 配置管理
2. 参考开源项目（nanoGPT 等）的目录结构
3. 迁移已有生成文件到新路径

---

## 二、开源项目参考（nanoGPT）

```
nanoGPT/
├── data/
│   └── {dataset}/           # 如 openwebtext, shakespeare
│       ├── train.bin        # tokenized train
│       ├── val.bin         # tokenized val
│       └── meta.pkl        # vocab_size, tokenizer info
├── out/                     # 或 checkpoints/
│   └── ckpt.pt             # 模型 checkpoint
└── config/
    └── *.yaml
```

**要点**：按 dataset 分目录，每个 dataset 下包含 raw、tokenized、tokenizer；checkpoint 单独目录。

---

## 三、新目录结构设计

```
assignment1-basics/
├── data/
│   ├── tinystories/
│   │   ├── raw/             # 原始文本
│   │   │   ├── train.txt
│   │   │   └── valid.txt
│   │   ├── tokenized/       # 序列化后的 .npy
│   │   │   ├── train.npy
│   │   │   └── valid.npy
│   │   └── tokenizer/       # BPE 模型
│   │       ├── model.pkl
│   │       └── model.json
│   └── owt/
│       ├── raw/
│       │   ├── train.txt
│       │   └── valid.txt
│       ├── tokenized/
│       │   ├── train.npy
│       │   └── valid.npy
│       └── tokenizer/
│           ├── model.pkl
│           └── model.json
├── checkpoints/              # 模型 checkpoint（按 run 名）
│   ├── tinystories_small/
│   │   └── checkpoint.pt
│   └── owt_medium/
│       └── checkpoint.pt
├── configs/
│   ├── paths.yaml           # 全局路径配置（新建）
│   ├── bpe_tinystories.yaml # BPE 训练配置（新建）
│   ├── bpe_owt.yaml
│   ├── train_tiny.yaml
│   └── train_owt.yaml
└── ...
```

---

## 四、配置文件设计

### 4.1 `configs/paths.yaml`（新建）

```yaml
# 全局路径配置，可被环境变量覆盖
base:
  data_dir: data
  checkpoint_dir: checkpoints

datasets:
  tinystories:
    raw_dir: data/tinystories/raw
    tokenized_dir: data/tinystories/tokenized
    tokenizer_dir: data/tinystories/tokenizer
    raw_train: data/tinystories/raw/train.txt
    raw_valid: data/tinystories/raw/valid.txt
    tokenized_train: data/tinystories/tokenized/train.npy
    tokenized_valid: data/tinystories/tokenized/valid.npy
    tokenizer_pkl: data/tinystories/tokenizer/model.pkl
    tokenizer_json: data/tinystories/tokenizer/model.json
  owt:
    raw_dir: data/owt/raw
    tokenized_dir: data/owt/tokenized
    tokenizer_dir: data/owt/tokenizer
    raw_train: data/owt/raw/train.txt
    raw_valid: data/owt/raw/valid.txt
    tokenized_train: data/owt/tokenized/train.npy
    tokenized_valid: data/owt/tokenized/valid.npy
    tokenizer_pkl: data/owt/tokenizer/model.pkl
    tokenizer_json: data/owt/tokenizer/model.json
```

### 4.2 `configs/bpe_tinystories.yaml`（新建）

```yaml
dataset: tinystories
vocab_size: 10000
# 路径从 paths.yaml 或内联
paths:
  raw_input: data/tinystories/raw/train.txt
  tokenizer_output_dir: data/tinystories/tokenizer
```

### 4.3 更新 `configs/train_tiny.yaml`

```yaml
dataset: tinystories
run_name: tinystories_small

paths:
  input_file: data/tinystories/tokenized/train.npy
  checkpoint_dir: checkpoints/tinystories_small

load_checkpoint: 0
batch_size: 32
# ... 其他超参
```

---

## 五、文件迁移映射

| 当前路径 | 新路径 |
|----------|--------|
| `data/_model/TinyStoriesV2-GPT4-train.txt.result.pkl` | `data/tinystories/tokenizer/model.pkl` |
| `data/_model/TinyStoriesV2-GPT4-train.txt.result.json` | `data/tinystories/tokenizer/model.json` |
| `data/_model/owt_train.txt.result.pkl` | `data/owt/tokenizer/model.pkl` |
| `data/_model/owt_train.txt.result.json` | `data/owt/tokenizer/model.json` |
| `data/TinyStoriesV2-GPT4-train.txt`（若存在） | `data/tinystories/raw/train.txt` |
| `data/TinyStoriesV2-GPT4-valid.txt`（若存在） | `data/tinystories/raw/valid.txt` |
| `data/owt_train.txt`（若存在） | `data/owt/raw/train.txt` |
| `data/owt_valid.txt`（若存在） | `data/owt/raw/valid.txt` |
| `data/_serialized/tinystories_train.npy`（若存在） | `data/tinystories/tokenized/train.npy` |
| `data/_serialized/owt_train.npy`（若存在） | `data/owt/tokenized/train.npy` |
| `ckpt/`（若存在） | `checkpoints/tinystories_small/` |
| `ckpt_owt/`（若存在） | `checkpoints/owt_medium/` |

---

## 六、执行步骤

### 阶段 0：准备工作

| 步骤 | 操作 | 验证 |
|------|------|------|
| 0.1 | 备份或提交当前状态 | `git status` |
| 0.2 | 确认 Python 可解析 YAML（pyproject 已有依赖或需加 pyyaml） | `python -c "import yaml"` |

### 阶段 1：创建配置与目录结构

| 步骤 | 操作 |
|------|------|
| 1.1 | 新建 `configs/paths.yaml` |
| 1.2 | 新建 `configs/bpe_tinystories.yaml`、`configs/bpe_owt.yaml` |
| 1.3 | 创建目录：`data/tinystories/{raw,tokenized,tokenizer}`、`data/owt/{raw,tokenized,tokenizer}`、`checkpoints/` |
| 1.4 | 更新 `configs/train_tiny.yaml`、`configs/train_owt.yaml` 使用新路径 |

### 阶段 2：迁移已有文件

| 步骤 | 操作 |
|------|------|
| 2.1 | 迁移 tokenizer：`data/_model/*.result.pkl` → `data/{dataset}/tokenizer/model.pkl` |
| 2.2 | 迁移 tokenizer：`data/_model/*.result.json` → `data/{dataset}/tokenizer/model.json` |
| 2.3 | 若存在 raw 文件，迁移到 `data/{dataset}/raw/` |
| 2.4 | 若存在 `data/_serialized/*.npy`，迁移到 `data/{dataset}/tokenized/` |
| 2.5 | 若存在 `ckpt/`、`ckpt_owt/`，迁移到 `checkpoints/` |
| 2.6 | 删除空目录 `data/_model`、`data/_serialized` |

### 阶段 3：修改 run.sh 读取配置

| 步骤 | 操作 |
|------|------|
| 3.1 | 添加 YAML 解析（bash 可用 `yq` 或 Python 脚本） |
| 3.2 | `bpe` 子命令：从 `configs/bpe_{dataset}.yaml` 读取路径与 vocab_size |
| 3.3 | `serialize` 子命令：从 config 或 paths.yaml 读取 tokenizer/input/output 路径 |
| 3.4 | `train` 子命令：支持 `--config configs/train_tiny.yaml` 从 YAML 加载 |
| 3.5 | `generate` 子命令：支持 `--config` 或沿用现有参数 |

### 阶段 4：修改 BPE trainer 输出命名

| 步骤 | 操作 |
|------|------|
| 4.1 | `bpe_trainer`：支持 `--output_path` 或 `--output_dir` + `--output_name`，输出 `model.pkl`/`model.json` |
| 4.2 | 或保持 `output_dir`，由 run.sh 在调用后 `mv` 到目标路径（简单但不够优雅） |

### 阶段 5：更新文档与示例

| 步骤 | 操作 |
|------|------|
| 5.1 | 更新 `README.md` 中的路径示例 |
| 5.2 | 更新 `run.sh help` 中的示例 |
| 5.3 | 更新 `docs/REFACTORING_PLAN.md` 或本文档的执行记录 |

---

## 七、run.sh 读取 YAML 的方案

**方案 A**：使用 `yq`（需安装）
```bash
TOKENIZER_PKL=$(yq '.datasets.tinystories.tokenizer_pkl' configs/paths.yaml)
```

**方案 B**：使用 Python 单行
```bash
TOKENIZER_PKL=$(python -c "
import yaml
with open('configs/paths.yaml') as f:
    p = yaml.safe_load(f)
print(p['datasets']['tinystories']['tokenizer_pkl'])
")
```

**方案 C**：run.sh 调用 `scripts/run_bpe.py` 等，由 Python 读取 YAML 并调用子模块（更清晰，推荐）

---

## 八、依赖

- 若使用 YAML：需 `pyyaml` 或 `ruamel.yaml`，检查 `pyproject.toml` 是否已有；若无则添加。

---

## 九、回滚

若迁移后有问题，可从 git 恢复，或保留 `data/_model` 的软链接指向新路径以兼容旧脚本。

---

*文档生成时间：2025-01-28*
