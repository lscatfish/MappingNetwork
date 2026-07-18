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

| 网络名 | 中文说明 | 原始参数量 | 结构特点 |
|--------|---------|-----------|----------|
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

### 2.3 参数生成网络与 LRD

- **参数生成网络（ParameterGenerator）**：只负责把可学习向量 `z` 变成目标网络的参数向量 `theta_hat`。内置三种生成器，通过 `type` 字段选择：
  - `linear`（默认，`LinearMappingNetwork`）：固定正交矩阵 `W_fixed` + 可学习 `z`，`theta = tanh(W_fixed @ z + α·||z||² + b_fixed)`，参数效率最高。
  - `multilayer_linear`（`MultiLayerLinearMappingNetwork`）：MLP 风格，`z -> Linear -> ReLU -> ... -> Linear -> tanh`，表达能力更强但可训练参数更多。
  - `cnn`（`CNNMappingNetwork`）：卷积风格，把 `z` 投影到小特征图后卷积再展平，适合捕捉空间结构。
- **扩展自己的生成器**：新建文件继承 `ParameterGenerator`，加 `@register_generator('名字')` 装饰器，并在 `mapping_network/generators/__init__.py` 里 import 即可被工厂识别，无需改 trainer / factory。
- **LRD（Low-Rank Decomposition，低秩分解）**：当目标网络太大时，把全连接层的权重拆成 `U @ V.T` 两个小矩阵，显著减少生成网络需要输出的参数数量，避免显存爆炸。默认对超过 20 万参数的网络自动开启。

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
epochs: 16            # 训练轮数
batch_size: 256       # 每批样本数
lr: 0.001             # 学习率
seed: 42              # 随机种子
device: cuda          # cuda 或 cpu
checkpoint_dir: checkpoints
save_interval: 2      # 每隔多少 epoch 保存中间模型
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
| `--device` | 否 | 自动 | CUDA 可用时用 `cuda`，否则 `cpu`；可显式指定 |

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
configs/cnn1_3conv_lwt.yaml      # CNN1_3Conv + LWT
configs/cnn1_3conv_slvt.yaml     # CNN1_3Conv + SLVT
configs/cnn2_baseline.yaml       # CNN2 + 基线训练
configs/cnn2_lwt.yaml            # CNN2 + LWT
configs/cnn2_slvt.yaml           # CNN2 + SLVT
```

### 3.3 评估已保存的模型

```bash
# 评估 SLVT 训练结果
uv run python3 -m mapping_network.scripts.evaluate \
  --checkpoint checkpoints/cnn2_slvt/cnn2_slvt_final.pth \
  --config configs/cnn2_slvt.yaml

# 评估 LWT 训练结果
uv run python3 -m mapping_network.scripts.evaluate \
  --checkpoint checkpoints/cnn2_lwt/cnn2_lwt_final.pth \
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
batch_size: 32                # 每批训练样本数，越大越快但越占显存
epochs: 16                    # 训练轮数，整个数据集过多少遍
seed: 42                      # 随机种子，固定后每次结果可复现
lr: 0.001                     # 学习率，控制参数更新步长
weight_decay: 0.0001          # 权重衰减，防止过拟合
min_lr: 0.00001               # 余弦退火的最小学习率
optimizer: adamw              # 优化器：adamw / adam / sgd
scheduler: cosine_annealing   # 学习率调度：cosine_annealing / step
alpha: 0.01                   # z 对映射权重的调制强度
sigma_noise: 0.01             # L_stab 里给 z 加噪声的标准差
device: cuda                  # 训练设备：cuda 或 cpu
log_interval: 100             # 每隔多少 batch 在进度条更新一次信息
checkpoint_dir: checkpoints   # 模型保存目录
save_interval: 1              # 每隔多少 epoch 保存一次中间模型，1 表示每轮都存
```

### 4.2 SLVT 特有参数

```yaml
generator_type: linear        # 参数生成网络类型：linear / multilayer_linear / cnn
latent_dim: 2048              # 隐向量 z 的长度，越短参数越少
```

`generator_type` 可选：
- `linear`（默认）：固定正交矩阵，可训练参数 = `latent_dim`。
- `multilayer_linear`：MLP 风格，额外接受 `hidden_dim`（默认 64）、`num_hidden`（默认 1）。
- `cnn`：卷积风格，额外接受 `feature_size`（默认 4）、`channels`（默认 `(16, 8)`）。

- `latent_dim` 越大，表达能力越强，但 `L_smooth` 计算越慢、越占显存。
- 如果显存不够，可以改成 `512` 或 `256`。

### 4.3 LWT 特有参数

LWT 不再使用全局的 `layer_latent_dims`，而是使用 `layer_generators`，为每一层单独指定生成网络配置：

```yaml
layer_generators:
  conv1:
    type: linear
    latent_dim: 256
    alpha: 0.01
  conv2:
    type: linear
    latent_dim: 256
    alpha: 0.01
  fc1:
    type: linear
    latent_dim: 256
    alpha: 0.01
    lrd_rank: 10              # 可选：单独给 fc1 指定低秩分解的秩
  fc2:
    type: linear
    latent_dim: 64
    alpha: 0.01
```

- `layer_generators` 的键名（`conv1`、`conv2`、`fc1`、`fc2`）必须和目标网络参数名前缀一致。
- 对 CNN1_3Conv，需要写成 `conv1`、`conv2`、`conv3`、`fc1`、`fc2`。
- 每层的 `latent_dim` 可以不同；`alpha` 可以省略，默认用全局 `alpha`。
- 每层的 `type` 也可以不同，可选值同 SLVT 的 `generator_type`（`linear` / `multilayer_linear` / `cnn`）；非 `linear` 类型需补充对应参数（如 `hidden_dim`、`channels`）。

### 4.4 LRD（低秩分解）配置

```yaml
lrd:
  enabled: auto               # 是否开启 LRD：true / false / auto
  default_rank: 10            # 默认低秩秩
  auto_enable_threshold: 200000  # auto 模式下，参数超过该阈值才开启
  layer_ranks:                # 可选：单独为某层指定秩
    fc1: 10
```

- `enabled: auto` 表示：目标网络总参数超过 `auto_enable_threshold` 时自动开启。
- 开启 LRD 后，全连接层的权重会被拆成 `U @ V.T`，生成网络只需要生成 `U`、`V` 和 bias，参数量大幅减少。
- 当前 LRD 只对 `nn.Linear` 模块生效，卷积层保持完整参数。

### 4.5 如何修改配置？

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

> 注意：`latent_dim`、`layer_generators`、`batch_size`、`lrd` 等参数只能通过改 YAML 文件来修改。

---

## 五、训练产物说明

### 5.1 Mapping Network 产物（以 CNN2 SLVT 为例）

```
checkpoints/cnn2_slvt/cnn2_slvt_final.pth       # 最后一轮模型权重（包含 metadata 和 state_dict）
checkpoints/cnn2_slvt/cnn2_slvt_best.pth        # 测试准确率最高那轮的模型权重
checkpoints/cnn2_slvt/cnn2_slvt_epoch1.pth      # 第 1 轮的中间权重（save_interval=1 时每轮都有）
checkpoints/cnn2_slvt/cnn2_slvt_results.json    # 每轮训练 loss / acc / 学习率记录
checkpoints/cnn2_slvt/cnn2_slvt.log             # 训练日志文本，和终端输出一致
```

### 5.2 LWT 产物（以 CNN2 LWT 为例）

```
checkpoints/cnn2_lwt/cnn2_lwt_final.pth        # 每层 MappingNetwork 的权重字典 + metadata
checkpoints/cnn2_lwt/cnn2_lwt_best.pth         # 测试准确率最高的权重
checkpoints/cnn2_lwt/cnn2_lwt_epoch1.pth       # 第 1 轮中间权重
checkpoints/cnn2_lwt/cnn2_lwt_results.json     # 训练记录
checkpoints/cnn2_lwt/cnn2_lwt.log              # 训练日志文本
```

### 5.3 基线产物（以 CNN2 Baseline 为例）

```
checkpoints/cnn2_baseline/cnn2_baseline_final.pth     # 最后一轮基线目标网络权重 + metadata
checkpoints/cnn2_baseline/cnn2_baseline_best.pth      # 测试准确率最高的基线权重
checkpoints/cnn2_baseline/cnn2_baseline_epoch1.pth    # 第 1 轮中间权重
checkpoints/cnn2_baseline/cnn2_baseline_results.json  # 训练记录
checkpoints/cnn2_baseline/cnn2_baseline.log           # 训练日志文本
```

加载基线时需要先取 `state_dict`：

```python
import torch
ckpt = torch.load('checkpoints/cnn2_baseline/cnn2_baseline_final.pth')
model.load_state_dict(ckpt['state_dict'])
```

### 5.4 这些文件会被提交到 git 吗？

不会。`.gitignore` 已经排除了 `*.pth`、`*.json`、`data/`、`checkpoints/` 等，训练产物只保存在本地。

---

## 六、常见问题

### 6.1 默认用 GPU 还是 CPU？

- `train.py` 和 `train_baseline.py` 默认自动检测：CUDA 可用时用 `cuda`，否则用 `cpu`。
- 也可以用 `--device cpu` / `--device cuda` 显式指定。
- 测试通过 `conftest.py` 的 `device` fixture 控制，默认同样自动检测；可用 `--device cpu` / `--device cuda` 覆盖。

### 6.2 训练时显存溢出（OOM）怎么办？

通常是 `latent_dim` 太大、`batch_size` 太大，或者没有开启 LRD 导致。尝试：

1. 确保 LRD 开启（大网络如 CNN1 会自动开启）。
2. 改小 `latent_dim`（如从 2048 改为 512）。
3. 改小 `batch_size`（如从 64 改为 32）。
4. 使用 CPU：`--device cpu`（会慢很多）。

### 6.3 训练多久？

- Baseline CNN2 16 轮：几分钟到十几分钟（GPU）。
- SLVT 16 轮：比 baseline 慢，因为每个 batch 要计算 Jacobian。
- LWT 16 轮：通常比 SLVT 更快，因为每层 `z` 维度小。

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
| 评估 SLVT | `uv run python3 -m mapping_network.scripts.evaluate --checkpoint checkpoints/cnn2_slvt/cnn2_slvt_final.pth --config configs/cnn2_slvt.yaml` |
| 评估 LWT | `uv run python3 -m mapping_network.scripts.evaluate --checkpoint checkpoints/cnn2_lwt/cnn2_lwt_final.pth --config configs/cnn2_lwt.yaml` |
| CPU 快速测试 | `uv run python3 -m mapping_network.scripts.train --config configs/cnn2_slvt.yaml --device cpu --epochs 1` |
| 代码检查 | `uv run ruff check .` |
| 代码格式化 | `uv run ruff format .` |
