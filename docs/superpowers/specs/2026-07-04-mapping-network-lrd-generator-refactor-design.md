# Mapping Network 生成器抽象与 LRD 重构设计文档

## 1. 背景与目标

当前 `MappingNetwork` 是单一的线性映射实现。对 CNN1（P≈538k）做 SLVT 时，固定的正交映射矩阵 `W_fixed [P, d]` 占用约 4.5 GB 显存，无法在普通 GPU 上训练。

本设计目标：

1. 抽象出 `ParameterGenerator` 基类，支持未来扩展线性/卷积/转置卷积/MLP 等多种参数生成器。
2. 为所有生成器统一提供 **LRD（Low Rank Decomposition）** 支持，显式降低大网络的显存占用。
3. LWT 模式下每层可独立配置生成器类型、latent 维度、α、LRD 等参数。
4. 保持 SLVT / LWT 训练逻辑和测试接口稳定；不兼容旧配置格式与旧 checkpoint。

## 2. 核心设计决策

### 2.1 生成器与目标网络解耦

- **ParameterGenerator**：只负责从可学习隐变量生成 `theta_hat`。**不**知道 LRD、不知道目标网络结构。
- **TargetNet**：负责把 `theta_hat` 解析成权重，并做函数式前向。**知道**自己是否启用 LRD、每段 `theta_hat` 对应完整权重 `W` 还是低秩因子 `(U, V)`。
- **MappingLoss / Trainer**：通过 `ParameterGenerator` 接口使用生成器，不依赖具体实现。

这样任何生成器都能自动享受 LRD 带来的显存收益，LRD 是目标网络的参数化方式，不是生成器类型。

### 2.2 LRD 默认策略

LRD 配置三态：

- `enabled: true`：强制对所有 FC 层启用 LRD。
- `enabled: false`：强制关闭 LRD。
- `enabled: auto`：按 `auto_enable_threshold` 自动判断，目标网络总参数 `P > threshold` 时启用，否则关闭。

默认 `auto_enable_threshold = 200,000`。CNN1（P≈538k）默认启用；CNN2（P≈109k）和 CNN1_3Conv（P≈32k）默认关闭，但可通过 `enabled: true` 显式开启。

LRD 当前只对 **Linear 层**生效，卷积层保持完整权重。

### 2.3 默认 rank 与按层覆盖

- 全局默认 `default_rank = 10`。
- 全局 `layer_ranks` 可按层覆盖，如 `fc1: 20`。
- LWT 模式下，每层配置中的 `lrd_rank` 会合并到全局 `layer_ranks`，优先级高于全局默认值。
- LRD 的开关由全局 `lrd.enabled` 控制，不支持每层单独开关。

## 3. 新的项目结构

```
mapping_network/
├── generators/
│   ├── __init__.py
│   ├── base.py              # ParameterGenerator 抽象基类
│   └── linear.py            # LinearMappingNetwork（原 MappingNetwork 改名继承）
├── target_nets/
│   ├── base.py              # TargetNet：增加 LRDConfig 支持、低秩前向
│   ├── lrd_config.py        # LRDConfig 数据类
│   ├── cnn1.py
│   ├── cnn1_3conv.py
│   └── cnn2.py
├── mapping/
│   └── loss.py              # 依赖 ParameterGenerator 接口
├── trainer/
│   ├── slvt.py              # 接收 ParameterGenerator
│   └── lwt.py               # 接收 per-layer ParameterGenerator dict
├── scripts/
│   ├── train.py             # 用工厂函数创建生成器
│   └── evaluate.py          # 根据 checkpoint 元数据重建生成器
├── factory.py               # build_generator / build_target_net
└── configs/
    ├── cnn1_slvt.yaml       # 增加 lrd 字段
    ├── cnn1_lwt.yaml        # 增加 layer_generators 详细配置
    └── ...
```

## 4. ParameterGenerator 抽象基类

```python
# mapping_network/generators/base.py
class ParameterGenerator(nn.Module, ABC):
    """参数生成网络基类。只负责生成 theta_hat。"""

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """返回 theta_hat [P']，P' 是目标网络压缩后的总参数数。"""
        pass

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

## 5. LinearMappingNetwork

当前 `MappingNetwork` 改名为 `LinearMappingNetwork`，继承 `ParameterGenerator`，逻辑保持不变：

```python
class LinearMappingNetwork(ParameterGenerator):
    def __init__(self, target_total_params: int, latent_dim: int,
                 alpha: float = 0.01, device: str = 'cpu'):
        super().__init__()
        self.P = target_total_params
        self.d = latent_dim
        self.alpha = alpha

        W = torch.empty(self.P, self.d, device=device)
        init.orthogonal_(W)
        self.register_buffer('W_fixed', W)
        self.register_buffer('W_fixed_mean', W.mean(dim=0))
        self.register_buffer('b_fixed', torch.zeros(self.P, device=device))
        self.z = nn.Parameter(torch.randn(self.d, device=device) * 0.1)

    def forward(self) -> torch.Tensor:
        return torch.tanh(
            self.W_fixed @ self.z
            + self.alpha * (self.z * self.z).sum()
            + self.b_fixed
        )
```

## 6. LRD 在 TargetNet 中的实现

### 6.1 LRDConfig

```python
# mapping_network/target_nets/lrd_config.py
from dataclasses import dataclass, field

@dataclass
class LRDConfig:
    enabled: bool | str = 'auto'      # true / false / 'auto'
    default_rank: int = 10
    layer_ranks: dict = field(default_factory=dict)
    auto_enable_threshold: int = 200_000
```

### 6.2 TargetNet 基类改造

`TargetNet.__init__` 增加可选参数 `lrd_config: LRDConfig | None = None`。

`_build_param_slices` 需要区分：

- 普通层：`(start, end, shape, name, is_bias, 'full')`
- LRD 层：拆成三段 `(U_start, U_end, U_shape, 'U')`、`(V_start, V_end, V_shape, 'V')`、`(b_start, b_end, b_shape, name, is_bias, 'bias')`

`functional_forward` 对 LRD 层先计算 `W = U @ V.T`，再做正常前向。

```python
class TargetNet(nn.Module):
    def __init__(self, lrd_config: LRDConfig | None = None):
        super().__init__()
        self._param_slices = []
        self._lrd_config = lrd_config or LRDConfig()

    def _should_use_lrd(self, layer_name: str, layer_params: int, total_params: int) -> bool:
        enabled = self._lrd_config.enabled
        if enabled is True:
            return True
        if enabled is False:
            return False
        # auto
        return total_params > self._lrd_config.auto_enable_threshold

    def _build_param_slices(self):
        ...

    def functional_forward(self, x, theta_hat):
        params = {}
        for slice_info in self._param_slices:
            if slice_info.kind == 'full':
                params[slice_info.name] = theta_hat[slice_info.start:slice_info.end].reshape(slice_info.shape)
            elif slice_info.kind == 'lrd':
                U = theta_hat[slice_info.u_start:slice_info.u_end].reshape(slice_info.u_shape)
                V = theta_hat[slice_info.v_start:slice_info.v_end].reshape(slice_info.v_shape)
                params[slice_info.weight_name] = U @ V.T
                params[slice_info.bias_name] = theta_hat[slice_info.b_start:slice_info.b_end].reshape(slice_info.b_shape)
        return self._functional_forward(x, params)
```

### 6.3 显存收益估算

以 CNN1 SLVT（d=2072）为例：

| 配置 | P'（压缩后参数数） | W_fixed 显存 |
|------|-------------------|--------------|
| 无 LRD | 537,960 | 4.46 GB |
| fc1: r=10 | 537,960 - 381,114 + 10*(2048+186) ≈ 179,186 | 1.49 GB |
| fc1: r=20 | 537,960 - 381,114 + 20*(2048+186) ≈ 201,566 | 1.67 GB |

## 7. 工厂函数

```python
# mapping_network/factory.py
from mapping_network.generators.linear import LinearMappingNetwork
from mapping_network.target_nets import CNN1, CNN2, CNN1_3Conv
from mapping_network.target_nets.lrd_config import LRDConfig

TARGET_NET_MAP = {
    'cnn1': CNN1,
    'cnn2': CNN2,
    'cnn1_3conv': CNN1_3Conv,
}

def build_target_net(target_name: str, lrd_config: dict | None = None):
    cls = TARGET_NET_MAP[target_name]
    cfg = LRDConfig(**lrd_config) if lrd_config else None
    return cls(lrd_config=cfg)

def build_generator(generator_type: str, target_total_params: int,
                    latent_dim: int, alpha: float, device: str):
    if generator_type == 'linear':
        return LinearMappingNetwork(target_total_params, latent_dim, alpha, device)
    raise ValueError(f'Unknown generator type: {generator_type}')
```

**注意**：`target_total_params` 必须是 **LRD 压缩后的参数数 P'**。工厂函数调用顺序为：先 `build_target_net`（内部根据 LRD 配置决定哪些层用低秩），再 `target_net.get_total_params()` 得到压缩后的 P'，最后 `build_generator`。

## 8. 训练器适配

### 8.1 SLVTTrainer

- `mapping_net` 参数类型从 `MappingNetwork` 改为 `ParameterGenerator`。
- 收集可训练参数时不再硬编码 `.z`，改为遍历 `mapping_net.parameters()`。
- checkpoint 保存 `generator_type` 和 `lrd_config`。

### 8.2 LWTTrainer

- `layer_mappings` 改为 `nn.ModuleDict[str, ParameterGenerator]`。
- 每层从配置读取 `latent_dim`、`alpha`、`generator_type`、`lrd`（可选覆盖全局）。
- 每层的 `P_l` 是压缩后的参数数（已考虑 LRD）。
- checkpoint 保存每层生成器的配置。

### 8.3 Checkpoint 目录结构

保持现有机制不变：每个实验使用独立文件夹 `{checkpoint_dir}/{experiment_name}/`，保存 `{experiment_name}_final.pth`、`_best.pth`、`_epochN.pth`、`_results.json`、`.log`。

SLVT 的 `experiment_name` 形如 `cnn1_slvt`，LWT 的 `experiment_name` 形如 `cnn1_lwt`，与当前一致。

### 8.4 MappingLoss

当前 `MappingLoss` 的实现需要访问 `mapping_net.W_fixed`、`mapping_net.alpha`、`mapping_net.z`、`mapping_net.P`、`mapping_net.d` 等属性来计算 `L_smooth` 和 `L_align`。

由于本次只实现 `LinearMappingNetwork`，这些属性自然存在。因此暂时保留现有访问方式，`LinearMappingNetwork` 需暴露相同接口。

**未来扩展注意**：当加入卷积/MLP 等非线性生成器时，`L_smooth` 和 `L_align` 的解析公式会不同，届时需要把这两个损失的计算下沉到各 `ParameterGenerator` 子类，或改为数值方法（如 `torch.func.jacfwd`）。本次设计为这一迁移预留了接口，但不实现。

## 9. 配置文件格式

### 9.1 SLVT 配置

```yaml
# configs/cnn1_slvt.yaml
target_net: cnn1
training_strategy: slvt
latent_dim: 2072
batch_size: 32
epochs: 16
seed: 42

optimizer: adamw
lr: 0.001
weight_decay: 0.0001
scheduler: cosine_annealing
min_lr: 0.00001

alpha: 0.01
lambda_st_init: 0.1
lambda_sm_init: 0.1
lambda_al_init: 0.1
sigma_noise: 0.01

# LRD 配置（新增）
lrd:
  enabled: auto          # true / false / auto
  default_rank: 10
  layer_ranks:
    fc1: 20

device: cuda
log_interval: 100
checkpoint_dir: checkpoints
save_interval: 1
```

### 9.2 LWT 配置

```yaml
# configs/cnn1_lwt.yaml
target_net: cnn1
training_strategy: lwt
batch_size: 32
epochs: 16
seed: 42

optimizer: adamw
lr: 0.001
weight_decay: 0.0001
scheduler: cosine_annealing
min_lr: 0.00001

# 全局 LRD 默认（可被每层覆盖）
lrd:
  enabled: auto
  default_rank: 10

# 每层独立生成器配置（新增）
# lrd_rank 可选，覆盖全局 default_rank；LRD 开关由全局 lrd.enabled 控制
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
    lrd_rank: 10
  fc2:
    type: linear
    latent_dim: 64
    alpha: 0.01

lambda_st_init: 0.1
lambda_sm_init: 0.1
lambda_al_init: 0.1
sigma_noise: 0.01

device: cuda
log_interval: 100
checkpoint_dir: checkpoints
save_interval: 1
```

## 10. Checkpoint 格式

### 10.1 SLVT

```python
{
    'target_net': 'cnn1',
    'training_strategy': 'slvt',
    'generator_type': 'linear',
    'latent_dim': 2072,
    'alpha': 0.01,
    'lrd_config': {'enabled': 'auto', 'default_rank': 10, 'layer_ranks': {'fc1': 20}},
    'sigma_noise': 0.01,
    'state_dict': <LinearMappingNetwork state_dict>,
    'results': [...],
    'epoch': 16,
    'is_best': False,
}
```

### 10.2 LWT

```python
{
    'target_net': 'cnn1',
    'training_strategy': 'lwt',
    'layer_generator_configs': {
        'conv1': {'type': 'linear', 'latent_dim': 256, 'alpha': 0.01},
        'fc1': {'type': 'linear', 'latent_dim': 256, 'alpha': 0.01, 'lrd_rank': 10},
        ...
    },
    'layer_group_order': ['conv1', 'conv2', 'fc1', 'fc2'],
    'lrd_config': {...},
    'sigma_noise': 0.01,
    'state_dict': {'conv1': <state_dict>, 'fc1': <state_dict>, ...},
    'results': [...],
    'epoch': 16,
    'is_best': False,
}
```

## 11. 测试计划

旧测试用例按新接口和新配置格式重写，不保留旧兼容。

1. **目标网络测试**：
   - 无 LRD 时，`functional_forward` 输出与模块前向一致，梯度可回传。
   - 有 LRD 时，`functional_forward` 输出与模块前向一致，且 `W_fixed` 大小按 rank 减小。
2. **LinearMappingNetwork 测试**：
   - 输出形状 `[P']` 正确。
   - 可训练参数只有 `z`。
   - `W_fixed`、`b_fixed` 为 buffer，不训练。
   - 梯度可回传到 `z`。
3. **MappingLoss 测试**：
   - 前向与反向正常，不 OOM。
   - `L_smooth` 不物化 `[P', d]` 中间张量。
4. **SLVT / LWT 一轮训练测试**：
   - batch_size=1 跑一个 batch 的前向+反向。
   - 验证 `z` 被更新。
   - 可训练参数 = `latent_dim + 3`（SLVT）或 `sum(layer_latent_dims) + 3`（LWT）。
5. **配置冒烟测试**：
   - 每个 mapping 配置（含 LRD 开启）都能跑通一轮前向+反向。
   - 所有张量在 GPU 上。
6. **Checkpoint 重建测试**：
   - 保存新格式 checkpoint。
   - 仅通过 checkpoint 元数据重建生成器和目标网络。
   - 重建后前向输出一致。

## 12. 回退与兼容

**本次重构不保留对旧代码、旧配置、旧 checkpoint 的兼容。**

- 删除 `mapping_network/mapping/mapping_net.py` 文件，相关导入改到 `mapping_network.generators.linear`。
- `LinearMappingNetwork` 是新入口。
- `evaluate.py` 只识别新 checkpoint 格式（含 `generator_type` / `lrd_config` / `layer_generator_configs`）。
- 配置文件统一使用新格式，旧 YAML 需要手动迁移或删除。
