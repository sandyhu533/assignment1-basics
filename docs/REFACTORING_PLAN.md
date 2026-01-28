# 项目目录结构重构执行计划

## 一、当前项目结构梳理

### 1.1 目录树概览

```
assignment1-basics/
├── cs336_basics/              # 核心库包
│   ├── __init__.py
│   ├── bpe_tokenizer.py       # BPE 分词器
│   ├── bpe_trainer.py         # BPE 训练器
│   ├── data/                  # 数据子包
│   │   ├── __init__.py
│   │   ├── batch.py           # get_batch
│   │   └── checkpoint.py      # save/load_checkpoint
│   ├── gen.py                 # 生成 CLI 入口
│   ├── generation.py          # 生成逻辑（库）
│   ├── linkedlist.py          # BPE 训练依赖
│   ├── model/                 # 模型子包
│   │   ├── __init__.py
│   │   ├── layers.py
│   │   ├── loss.py
│   │   ├── optimizer.py
│   │   └── transformer.py
│   ├── pretokenization_example.py
│   ├── pytorch_learning.ipynb # 学习笔记
│   ├── run_seralize.sh        # ⚠️ 冗余：run.sh 已覆盖
│   ├── run_train_bpe.sh      # ⚠️ 冗余：run.sh 已覆盖
│   └── train.py              # 训练 CLI 入口
├── data/                      # 数据目录（非包）
│   └── _model/                # BPE 模型输出
├── docs/
│   ├── STRUCTURE_RECOMMENDATIONS.md
│   └── REFACTORING_PLAN.md   # 本文档
├── scripts/                   # 薄层 CLI 入口
│   ├── generate.py            # → cs336_basics.gen.main
│   └── train.py               # → cs336_basics.train.main
├── tests/
├── run.sh                     # 统一运行脚本
├── pyproject.toml
├── setup.py
└── ...
```

### 1.2 模块职责与依赖关系

| 模块 | 职责 | 依赖 | 问题 |
|------|------|------|------|
| `cs336_basics.train` | 训练 CLI + 训练循环 | data, model | 库与 CLI 混合 |
| `cs336_basics.gen` | 生成 CLI | data, model, generation, bpe_tokenizer | 同上 |
| `cs336_basics.generation` | 生成逻辑（纯库） | model | ✅ 职责清晰 |
| `cs336_basics.data` | get_batch, checkpoint | - | ✅ 已抽离 |
| `cs336_basics.model` | 模型、优化器、损失 | - | ✅ 已子包化 |
| `cs336_basics.bpe_*` | BPE 训练与分词 | linkedlist | 含 CLI 入口 |
| `scripts/train.py` | 薄层入口 | cs336_basics.train | 与 run.sh 重复调用路径 |
| `scripts/generate.py` | 薄层入口 | cs336_basics.gen | 同上 |

### 1.3 入口调用链

```
run.sh bpe        → python -m cs336_basics.bpe_trainer
run.sh serialize  → python -m cs336_basics.bpe_tokenizer
run.sh train      → python -m cs336_basics.train
run.sh generate   → python -m cs336_basics.gen

pyproject.scripts:
  cs336-train     → cs336_basics.train:main
  cs336-generate  → cs336_basics.gen:main
```

### 1.4 冗余与不一致

1. **冗余 Shell 脚本**：`cs336_basics/run_train_bpe.sh`、`cs336_basics/run_seralize.sh` 与根目录 `run.sh` 功能重叠
2. **scripts/ 与 run.sh 双入口**：`scripts/train.py` 仅转发到 `cs336_basics.train`，run.sh 直接调 `cs336_basics.train`
3. **data 目录歧义**：根目录 `data/` 为数据存储，`cs336_basics/data/` 为 Python 子包，命名易混淆
4. **拼写错误**：`run_seralize.sh` 应为 `serialize`

---

## 二、目标目录结构（推荐）

```
assignment1-basics/
├── cs336_basics/              # 纯库代码（可 import，无 argparse 混入）
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── batch.py
│   │   └── checkpoint.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── layers.py
│   │   ├── loss.py
│   │   ├── optimizer.py
│   │   └── transformer.py
│   ├── tokenizer/             # 新建：BPE 相关集中
│   │   ├── __init__.py
│   │   ├── bpe_tokenizer.py
│   │   ├── bpe_trainer.py
│   │   └── linkedlist.py
│   ├── generation.py
│   └── pretokenization_example.py
├── scripts/                   # 所有 CLI 入口
│   ├── train.py
│   ├── generate.py
│   ├── train_bpe.py           # 新建：从 bpe_trainer 抽离 CLI
│   └── serialize.py           # 新建：从 bpe_tokenizer 抽离 CLI
├── configs/                   # 可选：YAML 配置
│   ├── train_tiny.yaml
│   └── train_owt.yaml
├── data/                      # 数据目录（保持不变）
│   ├── _model/
│   └── _serialized/
├── docs/
├── tests/
├── notebooks/                 # 可选：迁移 .ipynb
│   └── pytorch_learning.ipynb
├── run.sh                     # 统一入口
└── pyproject.toml
```

---

## 三、执行计划（分阶段）

### 阶段 0：准备工作（低风险）

| 步骤 | 操作 | 验证 |
|------|------|------|
| 0.1 | 运行 `uv run pytest -v ./tests` 确保当前全部通过 | 测试通过 |
| 0.2 | 备份或提交当前状态 | git status 干净 |

### 阶段 1：清理冗余（低风险）

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1.1 | 删除 `cs336_basics/run_train_bpe.sh` | 功能已由 run.sh bpe 覆盖 |
| 1.2 | 删除 `cs336_basics/run_seralize.sh` | 功能已由 run.sh serialize 覆盖 |
| 1.3 | 运行测试 | 确认无影响 |

### 阶段 2：统一 CLI 入口（中风险）

| 步骤 | 操作 | 说明 |
|------|------|------|
| 2.1 | 将 `scripts/train.py`、`scripts/generate.py` 作为**唯一** CLI 入口 | 或保持 run.sh 直接调 cs336_basics，二选一 |
| 2.2 | 更新 `pyproject.toml` 的 `[project.scripts]` 指向 scripts | `cs336-train = "scripts.train:main"` |
| 2.3 | 更新 `run.sh` 中 train/generate 调用为 `python -m scripts.train` 等 | 若采用 scripts 统一入口 |
| 2.4 | 从 `cs336_basics.train`、`cs336_basics.gen` 中抽离 CLI 逻辑到 scripts | train.py、gen.py 仅保留 `main()` 或删除，由 scripts 实现 |

**建议**：保持现状（run.sh 直接调 cs336_basics）可减少改动；若希望「库与 CLI 完全分离」，则执行 2.1–2.4。

### 阶段 3：tokenizer 子包化（中风险）

| 步骤 | 操作 | 说明 |
|------|------|------|
| 3.1 | 新建 `cs336_basics/tokenizer/` 目录 | |
| 3.2 | 移动 `bpe_tokenizer.py`、`bpe_trainer.py`、`linkedlist.py` 到 tokenizer/ | |
| 3.3 | 创建 `tokenizer/__init__.py`，导出 BPETokenizer、BPETrainer | |
| 3.4 | 更新所有导入：`cs336_basics.bpe_tokenizer` → `cs336_basics.tokenizer.bpe_tokenizer` | 涉及 tests/adapters.py, gen.py 等 |
| 3.5 | 运行测试 | 尤其 test_tokenizer.py, test_train_bpe.py |

### 阶段 4：训练/生成逻辑抽离（可选，中风险）

| 步骤 | 操作 | 说明 |
|------|------|------|
| 4.1 | 新建 `cs336_basics/training.py` | 纯训练循环函数 `train_loop(...)` |
| 4.2 | 从 `train.py` 抽离训练循环到 `training.train_loop` | train.py 仅做 argparse + 调用 |
| 4.3 | 新建 `cs336_basics/cli/` 或保持 train.py、gen.py 为薄 CLI | 视阶段 2 决策 |

### 阶段 5：配置与文档（低风险）

| 步骤 | 操作 | 说明 |
|------|------|------|
| 5.1 | 新建 `configs/`，添加 `train_tiny.yaml`、`train_owt.yaml` | 超参集中管理 |
| 5.2 | 迁移 `pytorch_learning.ipynb` 到 `notebooks/` | 可选 |
| 5.3 | 更新 README.md 中的目录说明与运行示例 | |

---

## 四、依赖与测试影响

### 4.1 需更新的导入（若执行 tokenizer 子包化）

| 文件 | 当前导入 | 更新后 |
|------|----------|--------|
| tests/adapters.py | `from cs336_basics.bpe_trainer import BPETrainer` | `from cs336_basics.tokenizer import BPETrainer` |
| tests/adapters.py | `from cs336_basics.bpe_tokenizer import BPETokenizer` | `from cs336_basics.tokenizer import BPETokenizer` |
| cs336_basics/gen.py | `from cs336_basics.bpe_tokenizer import BPETokenizer` | `from cs336_basics.tokenizer import BPETokenizer` |
| cs336_basics/bpe_trainer.py | `from cs336_basics.linkedlist import Node` | `from cs336_basics.tokenizer.linkedlist import Node` |

### 4.2 tests/adapters.py 兼容性

`adapters.py` 通过 `from cs336_basics.train import *` 获取 `get_batch`、`save_checkpoint`、`load_checkpoint`。  
若将 train 逻辑抽离，需保证这些仍可从 `cs336_basics.data` 导出，或 adapters 改为 `from cs336_basics.data import ...`。

---

## 五、推荐执行顺序（最小改动版）

若希望**最小改动、快速见效**，建议仅执行：

1. **阶段 1**：删除冗余 Shell 脚本
2. **阶段 5.3**：更新 README 中的目录说明

若希望**结构更清晰、便于后续扩展**，建议：

1. 阶段 1
2. 阶段 3（tokenizer 子包化）
3. 阶段 5.1（configs）
4. 阶段 2（视需求决定是否统一到 scripts）

---

## 六、检查清单

- [ ] 所有测试通过：`uv run pytest -v ./tests`
- [ ] run.sh 各子命令可用：bpe、serialize、train、generate
- [ ] make_submission.sh 可正常打包
- [ ] pyproject.toml scripts 入口可用（若使用）：`cs336-train --help`

---

## 七、执行记录

| 日期 | 阶段 | 操作 | 状态 |
|------|------|------|------|
| 2025-01-28 | 1 | 删除 `run_train_bpe.sh`、`run_seralize.sh` | ✅ 完成 |
| 2025-01-28 | 3 | 新建 `tokenizer/` 子包，迁移 bpe_tokenizer、bpe_trainer、linkedlist | ✅ 完成 |
| 2025-01-28 | 5 | 新建 `configs/train_tiny.yaml`、`configs/train_owt.yaml`，更新 README | ✅ 完成 |

---

*文档生成时间：2025-01-28*
