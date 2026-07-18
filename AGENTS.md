# Mapping Network — AI Agent 协作指南

本文件面向不熟悉本项目的 AI 编程助手，汇总项目架构、技术栈、构建/测试命令、开发约定与安全注意事项。信息基于当前代码仓库实际内容整理。

---

## 1. 项目概述

本项目复现论文 **Mapping Networks**（arXiv:2602.19134v1）在 MNIST 上的核心实验。核心思想是：

- 用一个低维、可训练的隐向量 `z` 生成目标 CNN 的全部参数。
- 通过**函数式前向**（functional forward）保持梯度链完整，使损失反向传播到 `z`。
- 相比直接训练目标网络，实现 50–500× 的可训练参数量压缩。

当前仓库实现了两套训练策略：

- **SLVT**（Single Latent Vector Training，`Ours*`）：单个 `z` 生成整个目标网络参数。
- **LWT**（Layer-wise Training，`Ours†`）：为目标网络每一层（按参数名前缀分组）使用独立的 `z^(l)` 和 MappingNetwork。

支持的目标网络：

- `CNN2`：LeNet 风格，约 108,610 参数。
- `CNN1`：AlexNet 风格（2 卷积），约 537,960 参数。
- `CNN1_3Conv`：CNN1 的三卷积实验变体，约 32,394 参数。

---

## 2. 技术栈

- **Python**：3.13（`.python-version`、`requires-python = ">=3.13"`）
- **包管理器**：`uv`（`pyproject.toml` + `uv.lock`）
- **深度学习框架**：PyTorch 2.11+（CUDA 12.8 wheel）
- **其他依赖**：torchvision、torchaudio、numpy、scipy、pandas、tqdm、PyYAML、scikit-learn、matplotlib
- **测试框架**：pytest
- **代码检查**：Ruff（E/F/I 规则，行宽 100，单引号，忽略 E501）

所有可执行脚本统一使用 `uv run python3 ...` 运行，确保使用项目锁定的虚拟环境（`.venv`）。

---

## 3. 项目结构

```
/root/MyProj/MappingNetwork
├── mapping_network/           # 主包
│   ├── target_nets/           # 目标网络实现
│   │   ├── base.py            # TargetNet 基类（functional_forward + LRD 支持）
│   │   ├── cnn1.py            # CNN1
│   │   ├── cnn1_3conv.py      # CNN1 三卷积实验版
│   │   ├── cnn2.py            # CNN2
│   │   └── lrd_config.py      # LRDConfig 配置 dataclass
│   ├── generators/            # 参数生成网络（@register_generator 注册）
│   │   ├── base.py            # ParameterGenerator 抽象基类 + GENERATOR_REGISTRY
│   │   ├── linear.py          # LinearMappingNetwork（'linear'）
│   │   ├── multilayer_linear.py  # MultiLayerLinearMappingNetwork（'multilayer_linear'）
│   │   └── cnn.py             # CNNMappingNetwork（'cnn'）
│   ├── mapping/               # Mapping Network 核心
│   │   └── loss.py            # MappingLoss：任务/稳定/光滑/对齐损失
│   ├── trainer/               # 训练器
│   │   ├── base.py            # BaseTrainer 基类（训练循环/checkpoint/日志）
│   │   ├── slvt.py            # SLVT 训练器
│   │   ├── lwt.py             # LWT 训练器
│   │   └── optim_utils.py     # 优化器/调度器工厂
│   ├── factory.py             # build_target_net / build_generator 工厂
│   └── scripts/               # 命令行入口
│       ├── train.py           # 统一训练入口（读取 YAML 配置）
│       ├── train_baseline.py  # 基线目标网络训练
│       └── evaluate.py        # 评估 checkpoint
├── configs/                   # YAML 训练配置
│   ├── cnn1_baseline.yaml
│   ├── cnn1_slvt.yaml
│   ├── cnn1_lwt.yaml
│   ├── cnn1_3conv_baseline.yaml
│   ├── cnn1_3conv_slvt.yaml
│   ├── cnn1_3conv_lwt.yaml
│   ├── cnn2_baseline.yaml
│   ├── cnn2_slvt.yaml
│   └── cnn2_lwt.yaml
├── tests/                     # pytest 测试
│   ├── conftest.py            # 全局 device fixture
│   ├── test_checkpoint.py     # checkpoint 重建测试
│   ├── test_configs.py        # 全配置冒烟测试
│   ├── test_extensibility.py  # 生成器可扩展性端到端验收
│   ├── test_factory.py        # 工厂函数测试
│   ├── test_generators.py     # 参数生成网络测试
│   ├── test_loss.py           # MappingLoss 测试
│   ├── test_lrd_config.py     # LRDConfig 测试
│   ├── test_lwt.py            # LWT 训练器测试
│   ├── test_slvt.py           # SLVT 训练器测试
│   └── test_target_nets.py    # 目标网络测试
├── checkpoints/               # 训练产物（.pth + .json），默认不提交
├── data/                      # MNIST 数据集下载目录，默认不提交
├── docs/superpowers/          # 开发计划与设计文档
├── pyproject.toml             # 项目配置与依赖
├── uv.lock                    # uv 锁定依赖
└── README.md
```

---

## 4. 核心架构说明

### 4.1 目标网络（TargetNet）

- 所有目标网络继承自 `mapping_network.target_nets.base.TargetNet`。
- 提供两套前向接口：
  - `forward(x)`：标准模块前向，用于基线训练。
  - `functional_forward(x, theta_hat)`：从 `theta_hat` 切片 reshape 出权重，使用 `F.conv2d` / `F.linear` 做函数式前向；支持 LRD 时自动重组 `U @ V.T`。
- **关键约束**：禁止使用 `.data.copy_()` 注入参数，必须通过函数式前向保持 `theta_hat → z` 的梯度链完整。
- 子类需在 `__init__` 末尾调用 `self._build_param_slices()`，以建立参数切分映射表。
- 支持 LRD（Low-Rank Decomposition）：对 `nn.Linear` 大权重自动拆分为 `U @ V.T`，由 `LRDConfig` 控制。

### 4.2 参数生成网络（ParameterGenerator）

- 抽象基类位于 `mapping_network.generators.base.ParameterGenerator`；子类用 `@register_generator('name')` 装饰器自动注册到 `GENERATOR_REGISTRY`，`factory.build_generator` 据 `config['type']` 查找。
- 子类必须实现：
  - `forward()`：返回 `theta_hat [P]`。
  - `noisy_forward(sigma)`：返回加噪后的 `theta_noisy [P]`，用于 `L_stab`。
  - `smooth_loss()`：返回 `L_smooth`。
  - `align_loss()`：返回 `L_align`。
- 可选重写 `persistent_state_dict()` / `load_persistent_state_dict()` / `_rebuild_buffers()` 控制大 buffer 的 checkpoint 持久化与重建。
- 当前注册的实现：
  - `linear`（`LinearMappingNetwork`）：固定权重 `W_fixed`（正交初始化）与偏置 `b_fixed` 注册为 buffer，`requires_grad=False`；唯一可训练参数是 `z`；前向 `theta_hat = tanh(W_fixed @ z + alpha * ||z||² + b_fixed)`。
  - `multilayer_linear`（`MultiLayerLinearMappingNetwork`）：MLP 风格，`z -> Linear -> ReLU -> ... -> Linear -> tanh`，额外参数 `hidden_dim`、`num_hidden`。
  - `cnn`（`CNNMappingNetwork`）：卷积风格，`z` 投影到小特征图后卷积再展平，额外参数 `feature_size`、`channels`。
- 新增生成器只需：新建文件继承 `ParameterGenerator` + `@register_generator('name')`，在 `mapping_network/generators/__init__.py` import，无需改 trainer / factory / evaluate。

### 4.3 MappingLoss

```
L_map = L_task + sigmoid(lambda_st) * L_stab
          + sigmoid(lambda_sm) * L_smooth
          + sigmoid(lambda_al) * L_align
```

- `L_task`：交叉熵分类损失。
- `L_stab`：对 `z` 加高斯噪声后前向输出的 MSE，衡量稳定性。
- `L_smooth`：`||∇_z M(z)||²_F / (P * d)`，由 `ParameterGenerator.smooth_loss()` 提供。
- `L_align`：`1 - cos(z, mean(W_mod, dim=0))`，由 `ParameterGenerator.align_loss()` 提供。
- `lambda_*` 为可训练的 `nn.Parameter`，通过 `sigmoid` 门控后参与加权。

---

## 5. 构建与运行命令

### 5.1 安装依赖

项目使用 `uv` 管理环境，通常无需手动安装：

```bash
uv sync
```

### 5.2 训练 Mapping Network

```bash
# SLVT 示例
uv run python3 -m mapping_network.scripts.train --config configs/cnn2_slvt.yaml

# LWT 示例
uv run python3 -m mapping_network.scripts.train --config configs/cnn2_lwt.yaml

# 覆盖部分配置（常用于快速冒烟测试）
uv run python3 -m mapping_network.scripts.train --config configs/cnn2_slvt.yaml --device cpu --epochs 1
```

### 5.3 训练基线目标网络

```bash
# 使用配置文件（推荐）
uv run python3 -m mapping_network.scripts.train_baseline --config configs/cnn2_baseline.yaml

# 使用命令行参数
uv run python3 -m mapping_network.scripts.train_baseline --target cnn2 --epochs 30
uv run python3 -m mapping_network.scripts.train_baseline --target cnn1 --device cpu --epochs 1
```

每种训练任务（target + strategy 组合）有独立目录，例如 `checkpoints/cnn2_baseline/`、`checkpoints/cnn2_slvt/`。最终权重保存为 `{target}_{strategy}_final.pth`，并同时生成 `_best.pth`、`_epochN.pth`、`_results.json`、`.log`。

### 5.4 评估 Checkpoint

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

---

## 6. 测试命令

所有测试默认在 CUDA 可用时运行在 GPU 上，可通过 `--device` 显式指定：

```bash
# 运行全部测试
uv run python3 -m pytest tests/ -v

# 指定设备
uv run python3 -m pytest tests/ -v --device cpu

# 单独运行某个测试文件
uv run python3 -m pytest tests/test_target_nets.py -v
uv run python3 -m pytest tests/test_generators.py -v
uv run python3 -m pytest tests/test_loss.py -v
uv run python3 -m pytest tests/test_slvt.py -v
uv run python3 -m pytest tests/test_lwt.py -v
uv run python3 -m pytest tests/test_extensibility.py -v
```

当前测试集共 55 个用例，覆盖：

- 各目标网络参数量与前向输出。
- `functional_forward` 输出与模块前向一致且梯度可回传。
- LRD 减少参数量且功能等价。
- ParameterGenerator 抽象性、LinearMappingNetwork 输出形状、可训练参数、辅助方法。
- MappingLoss 前向与梯度回传。
- SLVT / LWT 训练一个 epoch 后 `z` 被更新。
- LWT 稳定性损失的梯度不跨层泄漏。
- checkpoint 保存后能够重建并复现前向输出。
- 所有 YAML 配置均可完成一个 batch 的前向 + 反向。
- 非 `linear` 生成器（`multilayer_linear`）端到端跑通 factory -> trainer -> checkpoint -> evaluate，验证可扩展性。

---

## 7. 代码风格与开发约定

### 7.1 格式化与检查

- 使用 **Ruff** 进行 lint 与 format，配置在 `pyproject.toml`：
  - `line-length = 100`
  - 引号风格：单引号
  - lint 规则：`E`, `F`, `I`
  - 忽略 `E501`（行超长由 formatter 处理）
  - `__init__.py` 忽略 `F401`（允许未使用 import 暴露公共 API）

检查与格式化命令：

```bash
uv run ruff check .
uv run ruff check . --fix
uv run ruff format .
```

### 7.2 代码约定

- 所有网络类继承自 `torch.nn.Module`。
- 代码注释与 docstring 主要使用中文；类名、函数名、变量名为英文。
- `LinearMappingNetwork` 的 `W_fixed`、`b_fixed` 必须注册为 buffer，不可训练。
- 参数生成后**不注入**目标网络模块，而是调用 `target_net.functional_forward(x, theta_hat)`。
- LWT 中每层损失独立计算后再聚合，不得混用跨层隐向量；`L_stab` 计算时必须 detach 未扰动层的 `theta_hat` 切片。
- 配置项统一从 YAML 读取；脚本支持通过命令行覆盖 `device`、`epochs`、`seed`。
- 训练结束必须保存 checkpoint。每个 target + strategy 组合使用独立目录：
  - SLVT 保存到 `checkpoints/{target}_slvt/{target}_slvt_final.pth`。
  - LWT 保存到 `checkpoints/{target}_lwt/{target}_lwt_final.pth`。
  - 基线保存到 `checkpoints/{target}_baseline/{target}_baseline_final.pth`。
  - 三种训练都额外保存 `_best.pth`、`_epochN.pth`（由 `save_interval` 控制）、`_results.json`、`.log`。

### 7.3 配置文件约定

`configs/*.yaml` 必须包含：

```yaml
target_net: cnn1 | cnn2 | cnn1_3conv
training_strategy: slvt | lwt
batch_size: int
epochs: int
seed: int
lr: float
weight_decay: float
min_lr: float
optimizer: adamw | adam | sgd       # 默认 adamw
scheduler: cosine_annealing | step  # 默认 cosine_annealing
alpha: float
sigma_noise: float
device: cuda | cpu
log_interval: int
checkpoint_dir: str
save_interval: int                  # 每隔多少 epoch 保存中间模型，1 表示每轮都存

# SLVT 特有
generator_type: linear              # linear / multilayer_linear / cnn
latent_dim: int

# LWT 特有
layer_generators:                   # 每层独立配置
  conv1:
    type: linear
    latent_dim: int
    alpha: float                    # 可选，默认用全局 alpha
    lrd_rank: int                   # 可选，单独指定该层低秩秩
  ...

# LRD 可选（SLVT / LWT 均可）
lrd:
  enabled: true | false | auto      # 默认 auto
  default_rank: int                 # 默认 10
  auto_enable_threshold: int        # 默认 200000
  layer_ranks:
    fc1: int
    ...

# 基线特有（无 SLVT/LWT 特有字段）
target: cnn1 | cnn2 | cnn1_3conv
```

---

## 8. Git 与分支策略

- 主分支：`main`
- 复现分支：`feat/mapping-network-reproduction`（当前活跃分支）
- 开发时应从 `main` 切出功能分支，通过 PR 或 review 后合并。
- 历史提交遵循 `feat:`、`fix:` 等前缀约定。

---

## 9. 安全与部署注意事项

- **不要提交模型权重**：`.gitignore` 已排除 `*.pth`、`*.pt`、`*.onnx`、`data/`、`.venv/` 等。
- **不要提交凭据**：项目无 API key 或数据库连接，但请勿在代码中硬编码任何密钥。
- **CUDA 依赖**：PyTorch 从 `https://download.pytorch.org/whl/cu128` 显式索引安装。若在其他 CUDA 版本环境运行，需调整 `pyproject.toml` 中的 index 与源。
- **训练产物较大**：checkpoint 文件可能达数 MB 到数十 MB，请确保 `checkpoints/` 始终被 `.gitignore` 忽略。
- **测试写文件**：部分测试会写入 `/tmp/test_*` 目录，运行后通常无需清理。
- **MNIST 数据**：首次训练时会自动下载到 `./data`，请勿将该目录提交到版本控制。

---

## 10. 常见任务速查

| 任务 | 命令 |
|------|------|
| 同步依赖 | `uv sync` |
| 运行全部测试 | `uv run python3 -m pytest tests/ -v` |
| 训练 SLVT | `uv run python3 -m mapping_network.scripts.train --config configs/cnn2_slvt.yaml` |
| 训练 LWT | `uv run python3 -m mapping_network.scripts.train --config configs/cnn2_lwt.yaml` |
| 训练基线 | `uv run python3 -m mapping_network.scripts.train_baseline --config configs/cnn2_baseline.yaml` |
| 评估 SLVT | `uv run python3 -m mapping_network.scripts.evaluate --checkpoint checkpoints/cnn2_slvt/cnn2_slvt_final.pth --config configs/cnn2_slvt.yaml` |
| 评估 LWT | `uv run python3 -m mapping_network.scripts.evaluate --checkpoint checkpoints/cnn2_lwt/cnn2_lwt_final.pth --config configs/cnn2_lwt.yaml` |
| 代码检查 | `uv run ruff check .` |
| 代码格式化 | `uv run ruff format .` |

---

## 11. 参考文献

- 论文：*Mapping Networks*（arXiv:2602.19134v1）
- 设计文档：`docs/superpowers/specs/2026-07-04-mapping-networks-design.md`
- 实现计划：`docs/superpowers/plans/2026-07-04-mapping-networks-implementation.md`
