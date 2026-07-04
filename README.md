# Mapping Network 用户手册

本仓库复现论文 **Mapping Networks**（arXiv:2602.19134v1）在 MNIST 上的核心实验：用一个低维可训练隐向量 `z` 生成目标 CNN 的全部参数，实现 50–500× 的可训练参数量压缩。

---

## 环境说明

- **Python**：3.13
- **包管理器**：`uv`（使用项目虚拟环境 `.venv`，依赖见 `pyproject.toml` / `uv.lock`）
- **深度学习框架**：PyTorch 2.11+（CUDA 12.8 wheel）

所有命令统一使用 `uv run python3 ...`，确保运行在项目锁定的环境中，不会影响系统或其他项目使用的 Python 3.13 安装。

---

## 快速开始

### 1. 同步依赖

```bash
uv sync
```

### 2. 训练基线目标网络

```bash
# CNN2（LeNet 风格，约 108K 参数）
uv run python3 -m mapping_network.scripts.train_baseline --target cnn2 --epochs 30

# CNN1（AlexNet 风格，约 538K 参数）
uv run python3 -m mapping_network.scripts.train_baseline --target cnn1 --epochs 30

# CNN1_3Conv（三卷积实验版，约 32K 参数）
uv run python3 -m mapping_network.scripts.train_baseline --target cnn1_3conv --epochs 30
```

基线权重保存为 `{target}_baseline.pth`（例如 `cnn2_baseline.pth`）。

### 3. 训练 Mapping Network

#### SLVT（Single Latent Vector Training）

```bash
uv run python3 -m mapping_network.scripts.train --config configs/cnn2_slvt.yaml
```

#### LWT（Layer-wise Training）

```bash
uv run python3 -m mapping_network.scripts.train --config configs/cnn2_lwt.yaml
```

可用的配置文件：

- `configs/cnn1_slvt.yaml`
- `configs/cnn1_lwt.yaml`
- `configs/cnn1_3conv_slvt.yaml`
- `configs/cnn2_slvt.yaml`
- `configs/cnn2_lwt.yaml`

### 4. 评估 Checkpoint

```bash
# SLVT
uv run python3 -m mapping_network.scripts.evaluate \
  --checkpoint checkpoints/cnn2_slvt_final.pth \
  --config configs/cnn2_slvt.yaml

# LWT
uv run python3 -m mapping_network.scripts.evaluate \
  --checkpoint checkpoints/cnn2_lwt_final.pth \
  --config configs/cnn2_lwt.yaml
```

### 5. 运行测试

```bash
uv run python3 -m pytest tests/ -v
```

测试默认在 CUDA 可用时运行在 GPU 上，可通过 `--device cpu` 强制使用 CPU：

```bash
uv run python3 -m pytest tests/ -v --device cpu
```

---

## 常用命令速查

| 任务 | 命令 |
|------|------|
| 同步依赖 | `uv sync` |
| 运行全部测试 | `uv run python3 -m pytest tests/ -v` |
| 训练 SLVT | `uv run python3 -m mapping_network.scripts.train --config configs/cnn2_slvt.yaml` |
| 训练 LWT | `uv run python3 -m mapping_network.scripts.train --config configs/cnn2_lwt.yaml` |
| 训练基线 | `uv run python3 -m mapping_network.scripts.train_baseline --target cnn2 --epochs 30` |
| 评估 | `uv run python3 -m mapping_network.scripts.evaluate --checkpoint checkpoints/cnn2_slvt_final.pth --config configs/cnn2_slvt.yaml` |
| 代码检查 | `uv run ruff check .` |
| 代码格式化 | `uv run ruff format .` |

---

## 命令行覆盖选项

训练脚本支持通过命令行覆盖配置：

```bash
uv run python3 -m mapping_network.scripts.train \
  --config configs/cnn2_slvt.yaml \
  --device cpu \
  --epochs 1
```

支持的覆盖项：

- `--device cuda|cpu`
- `--epochs N`
- `--seed N`

---

## 训练产物

- **Checkpoint**：`checkpoints/{target}_{strategy}_final.pth`
- **结果 JSON**：`checkpoints/{target}_{strategy}_results.json`

这些文件已通过 `.gitignore` 排除，不会提交到版本控制。

---

## 注意事项

1. **所有计算默认在 GPU 上进行**。CUDA 不可用时自动回退到 CPU。
2. `L_smooth` 使用 `torch.func.jacfwd` 计算 Jacobian，在 `latent_dim=2048` 时需要较大显存；如遇到 OOM，可降低 `batch_size` 或 `latent_dim`。
3. 首次训练会自动下载 MNIST 数据集到 `./data`，请勿将该目录提交。
