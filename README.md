# Mapping Network 用户手册（小白版）

本仓库复现论文 **Mapping Networks**（arXiv:2602.19134v1）在 MNIST 手写数字识别上的核心实验。

**一句话理解**：传统神经网络是直接学习“每个参数该是多少”，而 Mapping Network 只学习一个很短的可训练向量 `z`，再用 `z` 生成出整个神经网络的参数。这样可以只用几十到几千个可训练参数，去控制原本几万到几十万个参数的神经网络。

---

## 一、环境准备

### 1.1 需要安装什么？

- **Python**：3.13
- **包管理器**：`uv`（自动创建 `.venv` 虚拟环境，不会污染系统 Python）
- **深度学习框架**：PyTorch 2.11+
- **数据集**：MNIST（第一次运行会自动下载到 `./data`）

### 1.2 同步依赖

在项目根目录执行：

```bash
uv sync
```

> 以后所有命令都要加 `uv run python3 ...`，这样才会使用项目自己的虚拟环境，不会影响你其他项目。

---

## 二、本项目能跑哪些网络？

### 2.1 三种目标网络（Target Network）

目标网络就是被压缩的那个“大网络”，它的参数由 `z` 生成。

| 网络名 | 中文说明 | 参数量 | 结构特点 |
|--------|---------|--------|----------|
| `CNN2` | 小型 CNN，类似 LeNet | 约 10.8 万 | 2 个卷积层 + 2 个全连接层，适合快速实验 |
| `CNN1` | 中型 CNN，类似 AlexNet | 约 53.8 万 | 2 个卷积层 + 2 个全连接层，通道数比 CNN2 多 |
| `CNN1_3Conv` | CNN1 的三卷积实验版 | 约 3.2 万 | 3 个卷积层 + 2 个全连接层，参数量最小 |

### 2.2 三种训练方式

| 方式 | 中文说明 | 训练什么 | 可训练参数量 |
|------|---------|----------|--------------|
| **Baseline** | 直接训练目标网络 | 目标网络本身的全部权重 | CNN2 约 10.8 万，CNN1 约 53.8 万 |
| **SLVT** | 单隐向量训练（论文 `Ours*`） | 只训练**一个**短向量 `z`，生成整个目标网络 | 等于 `z` 的长度（如 2048）+ 3 个损失系数 |
| **LWT** | 逐层隐向量训练（论文 `Ours†`） | 为每一层单独训练一个 `z`，每层独立生成该层参数 | 各层 `z` 长度之和 + 3 个损失系数 |

**举个例子**：
- Baseline 像直接雇 10 万个工人（参数）。
- SLVT 像只雇 2048 个设计师（`z`），让他们画出一套 10 万人施工的图纸。
- LWT 像给每一层楼单独雇一组设计师，每层画自己的图纸。

---

## 三、所有可运行的命令（完整清单）

### 3.1 训练基线目标网络

基线训练也支持 YAML 配置文件，和 SLVT/LWT 用法统一。

**推荐的配置文件方式**：

```bash
# CNN2 基线
uv run python3 -m mapping_network.scripts.train_baseline --config configs/cnn2_baseline.yaml

# CNN1 基线
uv run python3 -m mapping_network.scripts.train_baseline --config configs/cnn1_baseline.yaml

# CNN1_3Conv 基线
uv run python3 -m mapping_network.scripts.train_baseline --config configs/cnn1_3conv_baseline.yaml
```

**基线配置文件示例**（`configs/cnn2_baseline.yaml`）：

```yaml
target: cnn2          # 目标网络：cnn1 / cnn2 / cnn1_3conv
epochs: 30            # 训练轮数
batch_size: 64        # 每批样本数
lr: 0.001             # 学习率
seed: 42              # 随机种子
device: cuda          # cuda 或 cpu
```

**也可以直接用命令行**（适合临时测试）：

```bash
uv run python3 -m mapping_network.scripts.train_baseline --target cnn2 --epochs 1 --device cpu
```

**基线脚本支持的参数**：

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | 否 | 无 | YAML 配置文件路径 |
| `--target` | 否* | 无 | 选哪个目标网络：`cnn1` / `cnn2` / `cnn1_3conv` |
| `--epochs` | 否 | 30 | 训练多少轮 |
| `--batch-size` | 否 | 64 | 每批用多少张图 |
| `--lr` | 否 | 0.001 | 学习率 |
| `--seed` | 否 | 42 | 随机种子，保证可复现 |
| `--device` | 否 | `cuda` | 用 GPU 还是 CPU，可写 `cuda` 或 `cpu` |

> *如果不使用 `--config`，则 `--target` 必填。命令行参数优先级高于配置文件。*

### 3.2 训练 Mapping Network（SLVT / LWT）

```bash
# SLVT：单个 z 生成 CNN2 全部参数
uv run python3 -m mapping_network.scripts.train --config configs/cnn2_slvt.yaml

# LWT：每层独立 z 生成 CNN2 参数
uv run python3 -m mapping_network.scripts.train --config configs/cnn2_lwt.yaml
```

**所有可用的配置文件**：

```
configs/cnn1_baseline.yaml       # CNN1 + 基线训练
configs/cnn1_lwt.yaml            # CNN1 + LWT
configs/cnn1_slvt.yaml           # CNN1 + SLVT
configs/cnn1_3conv_baseline.yaml # CNN1_3Conv + 基线训练
configs/cnn1_3conv_slvt.yaml     # CNN1_3Conv + SLVT
configs/cnn2_baseline.yaml       # CNN2 + 基线训练
configs/cnn2_lwt.yaml            # CNN2 + LWT
configs/cnn2_slvt.yaml           # CNN2 + SLVT
```

### 3.3 评估已保存的模型

```bash
# 评估 SLVT 训练结果
uv run python3 -m mapping_network.scripts.evaluate \
  --checkpoint checkpoints/slvt/cnn2_slvt_final.pth \
  --config configs/cnn2_slvt.yaml

# 评估 LWT 训练结果
uv run python3 -m mapping_network.scripts.evaluate \
  --checkpoint checkpoints/lwt/cnn2_lwt_final.pth \
  --config configs/cnn2_lwt.yaml
```

> 新版本 checkpoint 已经自带 metadata，`--config` 只是作为备份信息。

### 3.4 运行测试

```bash
# 全部测试
uv run python3 -m pytest tests/ -v

# 指定用 CPU 跑测试
uv run python3 -m pytest tests/ -v --device cpu

# 指定用 GPU 跑测试
uv run python3 -m pytest tests/ -v --device cuda
```

---

## 四、配置文件每个参数是什么意思？

配置文件在 `configs/` 目录下，是 YAML 格式，可以直接用文本编辑器修改。

### 4.1 公共参数（SLVT 和 LWT 都有）

```yaml
target_net: cnn2              # 目标网络：cnn1 / cnn2 / cnn1_3conv
training_strategy: slvt       # 训练策略：slvt / lwt
batch_size: 64                # 每批训练样本数，越大越快但越占显存
epochs: 30                    # 训练轮数，整个数据集过多少遍
seed: 42                      # 随机种子，固定后每次结果可复现
lr: 0.001                     # 学习率，控制参数更新步长
weight_decay: 0.0001          # 权重衰减，防止过拟合
min_lr: 0.00001               # 余弦退火的最小学习率
alpha: 0.01                   # z 对映射权重的调制强度
sigma_noise: 0.01             # L_stab 里给 z 加噪声的标准差
device: cuda                  # 训练设备：cuda 或 cpu
log_interval: 100             # 每隔多少 batch 在进度条更新一次信息
checkpoint_dir: checkpoints   # 模型保存目录
save_interval: 1              # 每隔多少 epoch 保存一次中间模型，1 表示每轮都存
```

### 4.2 SLVT 特有参数

```yaml
latent_dim: 2048              # 隐向量 z 的长度，越短参数越少
```

- `latent_dim` 越大，表达能力越强，但 `L_smooth` 计算 Jacobian 越慢、越占显存。
- 如果显存不够，可以改成 `512` 或 `256`。

### 4.3 LWT 特有参数

```yaml
layer_latent_dims:            # 每一层隐向量的长度
  conv1: 256
  conv2: 256
  fc1: 256
  fc2: 256

layer_alphas:                 # 每一层的调制强度（可选，默认用上面的 alpha）
  conv1: 0.01
  conv2: 0.01
  fc1: 0.01
  fc2: 0.01
```

- `layer_latent_dims` 的键名（`conv1`、`conv2`、`fc1`、`fc2`）必须和目标网络参数名前缀一致。
- 对 CNN1_3Conv，需要写成 `conv1`、`conv2`、`conv3`、`fc1`、`fc2`。

### 4.4 如何修改配置？

**方法 1：直接改 YAML 文件**

用文本编辑器打开 `configs/cnn2_slvt.yaml`，例如把 `latent_dim: 2048` 改成 `latent_dim: 512`，保存即可。

**方法 2：命令行临时覆盖**

训练脚本支持覆盖 `device`、`epochs`、`seed`：

```bash
uv run python3 -m mapping_network.scripts.train \
  --config configs/cnn2_slvt.yaml \
  --device cpu \
  --epochs 1 \
  --seed 123
```

> 注意：`latent_dim`、`layer_latent_dims`、`batch_size` 等参数只能通过改 YAML 文件来修改。

---

## 五、训练产物说明

### 5.1 Mapping Network 产物

```
checkpoints/slvt/cnn2_slvt_final.pth       # 最后一轮模型权重（包含 metadata 和 state_dict）
checkpoints/slvt/cnn2_slvt_best.pth        # 测试准确率最高那轮的模型权重
checkpoints/slvt/cnn2_slvt_epoch1.pth      # 第 1 轮的中间权重（save_interval=1 时每轮都有）
checkpoints/slvt/cnn2_slvt_results.json    # 每轮训练 loss / acc / 学习率记录
checkpoints/slvt/cnn2_slvt.log             # 训练日志文本，和终端输出一致
```

### 5.2 LWT 产物

```
checkpoints/lwt/cnn2_lwt_final.pth        # 每层 MappingNetwork 的权重字典 + metadata
checkpoints/lwt/cnn2_lwt_best.pth         # 测试准确率最高的权重
checkpoints/lwt/cnn2_lwt_epoch1.pth       # 第 1 轮中间权重
checkpoints/lwt/cnn2_lwt_results.json     # 训练记录
checkpoints/lwt/cnn2_lwt.log              # 训练日志文本
```

### 5.3 基线产物

```
checkpoints/baseline/cnn2_baseline_final.pth   # 最后一轮基线目标网络权重 + metadata
checkpoints/baseline/cnn2_baseline_best.pth    # 测试准确率最高的基线权重
checkpoints/baseline/cnn2_baseline_epoch1.pth  # 第 1 轮中间权重
checkpoints/baseline/cnn2_baseline_results.json # 训练记录
checkpoints/baseline/cnn2_baseline.log          # 训练日志文本
```

加载基线时需要先取 `state_dict`：

```python
import torch
ckpt = torch.load('checkpoints/baseline/cnn2_baseline_final.pth')
model.load_state_dict(ckpt['state_dict'])
```

### 5.4 这些文件会被提交到 git 吗？

不会。`.gitignore` 已经排除了 `*.pth`、`*.json`、`data/`、`checkpoints/` 等，训练产物只保存在本地。

---

## 六、常见问题

### 6.1 默认用 GPU 还是 CPU？

- `train.py` 和 `train_baseline.py` 默认用 `cuda`。
- 如果电脑没有 NVIDIA 显卡或 CUDA 不可用，脚本会报错，需要手动加 `--device cpu`。

### 6.2 训练时显存溢出（OOM）怎么办？

通常是 `latent_dim` 太大或 `batch_size` 太大导致。尝试：

1. 改小 `latent_dim`（如从 2048 改为 512）
2. 改小 `batch_size`（如从 64 改为 32）
3. 使用 CPU：`--device cpu`（会慢很多）

### 6.3 训练多久？

- Baseline CNN2 30 轮：几分钟到十几分钟（GPU）
- SLVT 30 轮：比 baseline 慢，因为每个 batch 要计算 Jacobian
- LWT 30 轮：通常比 SLVT 更快，因为每层 `z` 维度小

### 6.4 准确率大概是多少？

论文参考值（MNIST）：

| 方法 | CNN2 | CNN1 |
|------|------|------|
| Baseline | ~98.69% | ~99.41% |
| SLVT (2048 维 z) | ~98.66% | ~99.38% |
| LWT | ~98.81% | ~99.43% |

实际值可能因随机种子、训练轮数略有波动。

---

## 七、命令速查表

| 想做什么 | 命令 |
|---------|------|
| 安装依赖 | `uv sync` |
| 跑全部测试 | `uv run python3 -m pytest tests/ -v` |
| 训练 CNN2 基线 | `uv run python3 -m mapping_network.scripts.train_baseline --config configs/cnn2_baseline.yaml` |
| 训练 CNN2 SLVT | `uv run python3 -m mapping_network.scripts.train --config configs/cnn2_slvt.yaml` |
| 训练 CNN2 LWT | `uv run python3 -m mapping_network.scripts.train --config configs/cnn2_lwt.yaml` |
| 评估 SLVT | `uv run python3 -m mapping_network.scripts.evaluate --checkpoint checkpoints/slvt/cnn2_slvt_final.pth --config configs/cnn2_slvt.yaml` |
| 评估 LWT | `uv run python3 -m mapping_network.scripts.evaluate --checkpoint checkpoints/lwt/cnn2_lwt_final.pth --config configs/cnn2_lwt.yaml` |
| CPU 快速测试 | `uv run python3 -m mapping_network.scripts.train --config configs/cnn2_slvt.yaml --device cpu --epochs 1` |
