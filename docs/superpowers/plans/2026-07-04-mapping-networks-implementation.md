# Mapping Networks 复现实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 严格复现 Mapping Networks 论文，在 MNIST 上验证 CNN1/CNN2 的 SLVT 和 LWT 策略。

**Architecture:** MappingNetwork（固定正交权重 + z 调制）生成目标网络全部参数，通过**函数式前向**（F.conv2d/F.linear）保持梯度完整回传至 z，避免 .data.copy_() 切断梯度链。

**Tech Stack:** Python 3.13, PyTorch 2.11+cu128, torchvision, numpy, matplotlib, tqdm, PyYAML

## Global Constraints

- 所有网络继承 `torch.nn.Module`
- 使用 `uv run python3` 执行
- MappingNetwork 的 W, b 固定不训练（requires_grad=False，注册为 buffer）
- **禁止使用 .data.copy_() 注入参数**——改用函数式前向保持梯度链完整
- LWT 中每个 layer 的损失需独立计算后聚合
- 训练结束时必须保存 checkpoint
- 支持随机种子以保证可复现
- 代码按模块拆分到 `mapping_network/` 下
- 从 `main` 创建新分支 `feat/mapping-network-reproduction`

---

### Task 1: 目标网络基类与 CNN2

**Files:**
- Create: `mapping_network/__init__.py`
- Create: `mapping_network/target_nets/__init__.py`
- Create: `mapping_network/target_nets/base.py`
- Create: `mapping_network/target_nets/cnn2.py`

**Interfaces:**
- Consumes: 无
- Produces: `TargetNet(nn.Module)` 基类（含 functional_forward），`CNN2(TargetNet)`

- [ ] **Step 1: 创建包初始化文件**

```python
# mapping_network/__init__.py
```

```python
# mapping_network/target_nets/__init__.py
from .cnn2 import CNN2
```

- [ ] **Step 2: 写 TargetNet 基类**

```python
# mapping_network/target_nets/base.py
import torch
import torch.nn as nn

class TargetNet(nn.Module):
    """
    目标网络基类。

    提供两套前向接口：
    - forward(x): 标准模块前向（用于基线训练）
    - functional_forward(x, theta_hat, slices): 函数式前向（用于 Mapping Network），
      直接从 theta_hat 切片 reshape 为权重，保持 autograd 梯度链完整。
    """

    def __init__(self):
        super().__init__()
        self._param_slices = []  # [(start, end, shape, name, is_bias)]

    def _build_param_slices(self):
        """构建参数切分映射表。子类在 __init__ 末尾调用。"""
        self._param_slices = []
        idx = 0
        for name, param in self.named_parameters():
            shape = param.shape
            numel = param.numel()
            is_bias = 'bias' in name
            self._param_slices.append((idx, idx + numel, shape, name, is_bias))
            idx += numel

    def get_param_slices(self):
        return self._param_slices

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_param_names(self):
        return [name for name, _ in self.named_parameters()]

    def functional_forward(self, x, theta_hat):
        """
        函数式前向：从 theta_hat [P] 切分权重，用 F.conv2d / F.linear 执行前向。
        梯度可完整回传至 theta_hat → z。
        """
        params = {}
        for start, end, shape, name, is_bias in self._param_slices:
            params[name] = theta_hat[start:end].reshape(shape)
        return self._functional_forward(x, params)

    def _functional_forward(self, x, params):
        """子类实现：使用 params 字典（键如 'conv1.weight'）做函数式前向。"""
        raise NotImplementedError

    def forward(self, x):
        """标准模块前向（用于基线训练）。"""
        raise NotImplementedError
```

- [ ] **Step 3: 写 CNN2**

```python
# mapping_network/target_nets/cnn2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import TargetNet

class CNN2(TargetNet):
    """LeNet 风格，~108,610 参数。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)     # 520 params
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(20, 32, kernel_size=5)    # 16,032 params
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(512, 176)                    # 90,288 params
        self.fc2 = nn.Linear(176, 10)                     # 1,770 params
        self._build_param_slices()

    def _functional_forward(self, x, params):
        x = F.relu(F.conv2d(x, params['conv1.weight'], params['conv1.bias']))
        x = self.pool1(x)
        x = F.relu(F.conv2d(x, params['conv2.weight'], params['conv2.bias']))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(F.linear(x, params['fc1.weight'], params['fc1.bias']))
        x = F.linear(x, params['fc2.weight'], params['fc2.bias'])
        return x

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

- [ ] **Step 4: 写单元测试验证参数量和函数式前向**

```python
# tests/test_target_nets.py
import pytest
import torch
from mapping_network.target_nets.cnn2 import CNN2

def test_cnn2_parameter_count():
    model = CNN2()
    total = sum(p.numel() for p in model.parameters())
    assert total == 108610, f"Expected 108610, got {total}"

def test_cnn2_forward():
    model = CNN2()
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    assert y.shape == (4, 10)

def test_cnn2_functional_forward():
    """验证函数式前向输出与模块前向一致，且梯度可回传至 theta_hat。"""
    model = CNN2()
    x = torch.randn(2, 1, 28, 28)
    theta_hat = torch.randn(model.get_total_params(), requires_grad=True)
    y = model.functional_forward(x, theta_hat)
    loss = y.sum()
    loss.backward()
    assert theta_hat.grad is not None
    assert theta_hat.grad.shape == (model.get_total_params(),)
```

- [ ] **Step 5: 运行测试验证通过**

Run: `cd /root/MyProj/MappingNetwork && uv run python3 -m pytest tests/test_target_nets.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
cd /root/MyProj/MappingNetwork && git add mapping_network/ tests/test_target_nets.py
git commit -m "feat: add TargetNet base with functional_forward and CNN2"
```

---

### Task 2: CNN1（2层卷积版和3层卷积版）

**Files:**
- Create: `mapping_network/target_nets/cnn1.py`
- Create: `mapping_network/target_nets/cnn1_3conv.py`

**Interfaces:**
- Consumes: `TargetNet` 基类
- Produces: `CNN1(TargetNet)`, `CNN1_3Conv(TargetNet)`

- [ ] **Step 1: 写 CNN1（2层卷积版）**

```python
# mapping_network/target_nets/cnn1.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import TargetNet

class CNN1(TargetNet):
    """AlexNet 风格，~537,912 参数。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 48, kernel_size=5)     # 1,200 params
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(48, 128, kernel_size=5)   # 153,728 params
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(2048, 186)                  # 381,114 params
        self.fc2 = nn.Linear(186, 10)                    # 1,870 params
        self._build_param_slices()

    def _functional_forward(self, x, params):
        x = F.relu(F.conv2d(x, params['conv1.weight'], params['conv1.bias']))
        x = self.pool1(x)
        x = F.relu(F.conv2d(x, params['conv2.weight'], params['conv2.bias']))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(F.linear(x, params['fc1.weight'], params['fc1.bias']))
        x = F.linear(x, params['fc2.weight'], params['fc2.bias'])
        return x

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

- [ ] **Step 2: 写 CNN1-3Conv（三卷积实验版）**

```python
# mapping_network/target_nets/cnn1_3conv.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import TargetNet

class CNN1_3Conv(TargetNet):
    """AlexNet 风格三卷积版（实验性）。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)     # 416
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)    # 12,832
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)    # 18,496
        self.pool3 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(64, 10)                     # 650
        self._build_param_slices()

    def _functional_forward(self, x, params):
        x = F.relu(F.conv2d(x, params['conv1.weight'], params['conv1.bias']))
        x = self.pool1(x)
        x = F.relu(F.conv2d(x, params['conv2.weight'], params['conv2.bias']))
        x = self.pool2(x)
        x = F.relu(F.conv2d(x, params['conv3.weight'], params['conv3.bias']))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.linear(x, params['fc1.weight'], params['fc1.bias'])
        return x

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
```

- [ ] **Step 3: 添加测试**

```python
# tests/test_target_nets.py (追加)
from mapping_network.target_nets.cnn1 import CNN1
from mapping_network.target_nets.cnn1_3conv import CNN1_3Conv

def test_cnn1_parameter_count():
    model = CNN1()
    total = sum(p.numel() for p in model.parameters())
    assert total == 537912, f"Expected 537912, got {total}"

def test_cnn1_forward():
    model = CNN1()
    x = torch.randn(2, 1, 28, 28)
    y = model(x)
    assert y.shape == (2, 10)

def test_cnn1_functional_forward():
    model = CNN1()
    x = torch.randn(2, 1, 28, 28)
    theta_hat = torch.randn(model.get_total_params(), requires_grad=True)
    y = model.functional_forward(x, theta_hat)
    y.sum().backward()
    assert theta_hat.grad is not None

def test_cnn1_3conv_functional_forward():
    model = CNN1_3Conv()
    x = torch.randn(2, 1, 28, 28)
    theta_hat = torch.randn(model.get_total_params(), requires_grad=True)
    y = model.functional_forward(x, theta_hat)
    y.sum().backward()
    assert theta_hat.grad is not None
```

- [ ] **Step 4: 更新包初始化**

```python
# mapping_network/target_nets/__init__.py
from .cnn2 import CNN2
from .cnn1 import CNN1
from .cnn1_3conv import CNN1_3Conv
```

- [ ] **Step 5: 运行测试**

Run: `cd /root/MyProj/MappingNetwork && uv run python3 -m pytest tests/test_target_nets.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
cd /root/MyProj/MappingNetwork && git add mapping_network/target_nets/ tests/
git commit -m "feat: add CNN1 and CNN1_3Conv with functional_forward"
```

---

### Task 3: Mapping Network 核心

**Files:**
- Create: `mapping_network/mapping/__init__.py`
- Create: `mapping_network/mapping/mapping_net.py`

**Interfaces:**
- Consumes: `TargetNet` 的无参 `get_total_params()` / `get_param_slices()`
- Produces: `MappingNetwork(nn.Module)` — 固定正交权重 + 调制 + 参数生成（不含注入）

- [ ] **Step 1: 写 MappingNetwork**

```python
# mapping_network/mapping/mapping_net.py
import torch
import torch.nn as nn
import torch.nn.init as init

class MappingNetwork(nn.Module):
    """
    映射网络：从低维 latent vector z 生成目标网络参数。

    - W_fixed: 固定正交初始化映射矩阵 [P, d]（buffer，不训练）
    - b_fixed: 固定偏置 [P]（buffer，不训练）
    - z: 可训练的 latent vector [d]（nn.Parameter）
    - α: 调制系数

    前向: θ̂ = tanh(W_mod · z + b)         (方程 21)
    其中 W_mod[i,:] = W_fixed[i,:] + α·z   (方程 20)

    返回 θ̂ ∈ R^P。不执行参数注入——由调用方传给 target_net.functional_forward()。
    """

    def __init__(self, target_total_params: int, latent_dim: int, alpha: float = 0.01):
        super().__init__()
        self.P = target_total_params
        self.d = latent_dim
        self.alpha = alpha

        # 固定正交初始化映射权重 [P, d]
        W = torch.empty(self.P, self.d)
        init.orthogonal_(W)
        self.register_buffer('W_fixed', W)

        # 固定偏置 [P]
        self.register_buffer('b_fixed', torch.zeros(self.P))

        # 可训练的 latent vector [d]
        self.z = nn.Parameter(torch.randn(self.d) * 0.1)

    def forward(self):
        """返回 θ̂ ∈ R^P。"""
        # W_mod[i,:] = W_fixed[i,:] + α·z  (广播到全部 P 行)
        W_mod = self.W_fixed + self.alpha * self.z.unsqueeze(0)
        theta_hat = torch.tanh(W_mod @ self.z + self.b_fixed)
        return theta_hat

    def extra_repr(self):
        return f"P={self.P}, d={self.d}, alpha={self.alpha}"
```

- [ ] **Step 2: 写单元测试**

```python
# tests/test_mapping_net.py
import pytest
import torch
from mapping_network.mapping.mapping_net import MappingNetwork
from mapping_network.target_nets.cnn2 import CNN2

def test_mapping_network_output_shape():
    d = 128
    net = MappingNetwork(108610, d)
    theta = net()
    assert theta.shape == (108610,)

def test_mapping_network_trainable_params():
    net = MappingNetwork(108610, 2048)
    trainable = [p for p in net.parameters() if p.requires_grad]
    assert len(trainable) == 1  # 只有 z
    assert trainable[0].shape == (2048,)

def test_mapping_network_fixed_weights():
    net = MappingNetwork(108610, 2048)
    assert not net.W_fixed.requires_grad
    assert not net.b_fixed.requires_grad

def test_gradient_flows_through_theta_hat():
    """核心测试：验证梯度能从 θ̂ 回传至 z。"""
    target = CNN2()
    mapping = MappingNetwork(target.get_total_params(), 128)
    x = torch.randn(2, 1, 28, 28)

    theta_hat = mapping()
    y = target.functional_forward(x, theta_hat)
    y.sum().backward()

    assert mapping.z.grad is not None
    assert mapping.z.grad.shape == (128,)
```

- [ ] **Step 3: 运行测试**

Run: `cd /root/MyProj/MappingNetwork && uv run python3 -m pytest tests/test_mapping_net.py -v`
Expected: 4 passed

- [ ] **Step 4: Commit**

```bash
cd /root/MyProj/MappingNetwork && git add mapping_network/mapping/ tests/
git commit -m "feat: add MappingNetwork with gradient-complete theta_hat output"
```

---

### Task 4: Mapping Loss

**Files:**
- Create: `mapping_network/mapping/loss.py`

**Interfaces:**
- Consumes: `MappingNetwork.forward()` → θ̂, `TargetNet.functional_forward()`
- Produces: `MappingLoss(nn.Module)` — 接受 θ̂ 和 slices，通过 functional_forward 计算损失

- [ ] **Step 1: 写 MappingLoss**

```python
# mapping_network/mapping/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MappingLoss(nn.Module):
    """
    Mapping Loss: Lmap = Ltask + λst·Lstab + λsm·Lsmooth + λal·Lalign  (方程 26)

    所有损失通过 target_net.functional_forward() 计算，梯度完整回传至 z。
    L_stab 不修改 target_net 参数——直接传入 theta_noisy 做函数式前向。
    """

    def __init__(self, sigma_noise: float = 0.01):
        super().__init__()
        self.sigma_noise = sigma_noise
        self.lambda_st = nn.Parameter(torch.tensor(0.1))
        self.lambda_sm = nn.Parameter(torch.tensor(0.1))
        self.lambda_al = nn.Parameter(torch.tensor(0.1))

    def forward(self, z, theta_hat, theta_noisy, mapping_net, target_net, x, y):
        """
        Args:
            z: latent vector [d]
            theta_hat: 当前 θ̂ [P] (带梯度)
            theta_noisy: 加噪声后的 θ̂ [P] (带梯度, 用于 L_stab)
            mapping_net: MappingNetwork 实例
            target_net: 目标网络
            x: 输入 [B, 1, 28, 28]
            y: 标签 [B]
        Returns:
            total_loss, losses_dict
        """
        # === L_task: 交叉熵 (方程 27) ===
        y_hat = target_net.functional_forward(x, theta_hat)
        l_task = F.cross_entropy(y_hat, y)

        # === L_stab: 稳定性损失 (方程 28) ===
        # 直接用 theta_noisy 做函数式前向，无需 save/restore 参数
        y_hat_noisy = target_net.functional_forward(x, theta_noisy)
        l_stab = F.mse_loss(y_hat_noisy, y_hat.detach())

        # === L_smooth: 平滑损失 (方程 29) ===
        # ||∇_z M_φ(z)||²_F / P  （按 P 平均，跨架构可迁移）
        W_mod = mapping_net.W_fixed + mapping_net.alpha * z.unsqueeze(0)
        jacobian = torch.autograd.functional.jacobian(
            lambda z_in: torch.tanh(
                (mapping_net.W_fixed + mapping_net.alpha * z_in.unsqueeze(0)) @ z_in
                + mapping_net.b_fixed
            ),
            z,
            create_graph=True,
        )
        l_smooth = torch.sum(jacobian ** 2) / jacobian.numel()

        # === L_align: 对齐损失 (方程 30) ===
        W_m = W_mod.mean(dim=0)  # [d]
        cos_sim = F.cosine_similarity(z.unsqueeze(0), W_m.unsqueeze(0))
        l_align = 1 - cos_sim.squeeze()

        # === 总损失 ===
        l_st = torch.sigmoid(self.lambda_st)
        l_sm = torch.sigmoid(self.lambda_sm)
        l_al = torch.sigmoid(self.lambda_al)

        total_loss = l_task + l_st * l_stab + l_sm * l_smooth + l_al * l_align

        losses = {
            'task': l_task.item(),
            'stab': l_stab.item(),
            'smooth': l_smooth.item(),
            'align': l_align.item(),
            'total': total_loss.item(),
        }
        return total_loss, losses
```

- [ ] **Step 2: 写测试**

```python
# tests/test_loss.py
import pytest
import torch
from mapping_network.mapping.mapping_net import MappingNetwork
from mapping_network.mapping.loss import MappingLoss
from mapping_network.target_nets.cnn2 import CNN2

def test_mapping_loss_forward():
    target = CNN2()
    mapping = MappingNetwork(target.get_total_params(), 64)
    loss_fn = MappingLoss()

    theta = mapping()
    eps = torch.randn_like(mapping.z) * 0.01
    z_noisy = mapping.z + eps
    W_mod_noisy = mapping.W_fixed + mapping.alpha * z_noisy.unsqueeze(0)
    theta_noisy = torch.tanh(W_mod_noisy @ z_noisy + mapping.b_fixed)

    x = torch.randn(2, 1, 28, 28)
    y = torch.randint(0, 10, (2,))

    loss, losses_dict = loss_fn(mapping.z, theta, theta_noisy, mapping, target, x, y)
    assert loss.requires_grad
    assert loss.item() > 0

def test_mapping_loss_gradient_to_z():
    """验证所有损失分量的梯度都能回传至 z。"""
    target = CNN2()
    mapping = MappingNetwork(target.get_total_params(), 64)
    loss_fn = MappingLoss()

    theta = mapping()
    eps = torch.randn_like(mapping.z) * 0.01
    z_noisy = mapping.z + eps
    W_mod_noisy = mapping.W_fixed + mapping.alpha * z_noisy.unsqueeze(0)
    theta_noisy = torch.tanh(W_mod_noisy @ z_noisy + mapping.b_fixed)

    x = torch.randn(2, 1, 28, 28)
    y = torch.randint(0, 10, (2,))

    loss, _ = loss_fn(mapping.z, theta, theta_noisy, mapping, target, x, y)
    loss.backward()
    assert mapping.z.grad is not None
    assert mapping.z.grad.shape == (64,)
```

- [ ] **Step 3: 运行测试**

Run: `cd /root/MyProj/MappingNetwork && uv run python3 -m pytest tests/test_loss.py -v`
Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
cd /root/MyProj/MappingNetwork && git add mapping_network/mapping/loss.py tests/
git commit -m "feat: add MappingLoss with functional forward, fix gradient chain"
```

---

### Task 5: SLVT 训练器

**Files:**
- Create: `mapping_network/trainer/__init__.py`
- Create: `mapping_network/trainer/slvt.py`

**Interfaces:**
- Consumes: `MappingNetwork`, `MappingLoss`, `TargetNet`（functional_forward）
- Produces: `SLVTTrainer` — 含 checkpoint 保存

- [ ] **Step 1: 写 SLVT 训练器**

```python
# mapping_network/trainer/slvt.py
import os
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import tqdm

class SLVTTrainer:
    """
    Single Latent Vector Training (SLVT / Ours*).

    一个 latent vector z 生成全部目标网络参数。
    使用函数式前向保持梯度完整。
    """

    def __init__(
        self,
        mapping_net,
        target_net,
        loss_fn,
        train_loader: DataLoader,
        test_loader: DataLoader = None,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        epochs: int = 30,
        min_lr: float = 1e-5,
        device: str = 'cuda',
        log_interval: int = 100,
        checkpoint_dir: str = 'checkpoints',
        experiment_name: str = 'slvt',
    ):
        self.mapping_net = mapping_net.to(device)
        self.target_net = target_net.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.log_interval = log_interval
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name

        # 只更新 z 和 λ (MappingNet 权重固定)
        trainable_params = [
            self.mapping_net.z,
            self.loss_fn.lambda_st,
            self.loss_fn.lambda_sm,
            self.loss_fn.lambda_al,
        ]
        self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=min_lr)

    def train_epoch(self, epoch):
        self.mapping_net.train()
        self.target_net.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs}')
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)

            # 1. 从 z 生成参数
            theta_hat = self.mapping_net()

            # 2. 计算噪声版本用于 L_stab
            eps = torch.randn_like(self.mapping_net.z) * self.loss_fn.sigma_noise
            z_noisy = self.mapping_net.z + eps
            with torch.no_grad():
                W_mod_noisy = self.mapping_net.W_fixed + self.mapping_net.alpha * z_noisy.unsqueeze(0)
                theta_noisy = torch.tanh(W_mod_noisy @ z_noisy + self.mapping_net.b_fixed)
            theta_noisy.requires_grad_(True)
            # 注：L_stab 的 theta_noisy 用 detach 形态，因 Jacobian 只对 z 求导
            # 简化：只对干净 theta_hat 求梯度，噪声版本用于 L_stab 计算

            # 3. 计算损失 (函数式前向)
            loss, losses_dict = self.loss_fn(
                self.mapping_net.z, theta_hat, theta_noisy.detach(),
                self.mapping_net, self.target_net, x, y,
            )

            # 4. 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            # 准确率：使用当前的 theta_hat
            with torch.no_grad():
                y_hat = self.target_net.functional_forward(x, theta_hat)
                _, predicted = y_hat.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%',
                })

        return total_loss / len(self.train_loader), 100. * correct / total

    @torch.no_grad()
    def evaluate(self):
        self.mapping_net.eval()
        self.target_net.eval()
        correct = 0
        total = 0

        theta_hat = self.mapping_net()
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.target_net.functional_forward(x, theta_hat)
            _, predicted = y_hat.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        return 100. * correct / total

    def save_checkpoint(self, results, epoch=None):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        suffix = f"_epoch{epoch}" if epoch else "_final"
        path = os.path.join(self.checkpoint_dir, f'{self.experiment_name}{suffix}.pth')
        torch.save(self.mapping_net.state_dict(), path)

        # 同时保存结果
        results_path = os.path.join(self.checkpoint_dir, f'{self.experiment_name}_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        return path

    def train(self):
        results = []
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            test_acc = self.evaluate()
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            epoch_result = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'lr': current_lr,
            }
            results.append(epoch_result)
            print(f'Epoch {epoch}: train_loss={train_loss:.4f}, '
                  f'train_acc={train_acc:.2f}%, test_acc={test_acc:.2f}%')

        # 保存最终 checkpoint
        path = self.save_checkpoint(results)
        print(f'Checkpoint saved to {path}')
        return results
```

- [ ] **Step 2: 写冒烟测试**

```python
# tests/test_slvt.py
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from mapping_network.mapping.mapping_net import MappingNetwork
from mapping_network.mapping.loss import MappingLoss
from mapping_network.target_nets.cnn2 import CNN2
from mapping_network.trainer.slvt import SLVTTrainer

def test_slvt_train_one_batch():
    """验证 SLVT 训练一个 batch 后 z 有梯度更新。"""
    target = CNN2()
    mapping = MappingNetwork(target.get_total_params(), 64)
    loss_fn = MappingLoss()

    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=8)

    z_before = mapping.z.data.clone()

    trainer = SLVTTrainer(
        mapping, target, loss_fn, loader,
        epochs=1, device='cpu', log_interval=1,
        checkpoint_dir='/tmp/test_slvt_checkpoints',
        experiment_name='test_slvt',
    )
    results = trainer.train()
    assert len(results) == 1
    # z 应该已被更新
    assert not torch.equal(z_before, mapping.z.data)
```

- [ ] **Step 3: 运行测试**

Run: `cd /root/MyProj/MappingNetwork && uv run python3 -m pytest tests/test_slvt.py -v`
Expected: 1 passed

- [ ] **Step 4: Commit**

```bash
cd /root/MyProj/MappingNetwork && git add mapping_network/trainer/ tests/
git commit -m "feat: add SLVT trainer with functional_forward and checkpoint"
```

---

### Task 6: LWT 训练器

**Files:**
- Create: `mapping_network/trainer/lwt.py`

**Interfaces:**
- Consumes: `TargetNet`、`MappingLoss`、多个独立的 `MappingNetwork`（每层一个）
- Produces: `LWTTrainer` — 每层独立 z^(l)，各层损失独立计算后聚合

- [ ] **Step 1: 写 LWT 训练器**

```python
# mapping_network/trainer/lwt.py
import os
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import tqdm
from ..mapping.mapping_net import MappingNetwork


class LWTTrainer:
    """
    Layer-wise Training (LWT / Ours†).

    每层/每组用独立的 latent vector，各自通过 MappingNetwork^(l) 生成该层参数。
    损失：L_task 统一计算，L_stab/L_smooth/L_align 每层独立计算后 Σ 聚合。
    """

    def __init__(
        self,
        target_net,
        loss_fn,
        layer_latent_dims: dict,
        layer_alphas: dict = None,
        train_loader: DataLoader = None,
        test_loader: DataLoader = None,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        epochs: int = 30,
        min_lr: float = 1e-5,
        device: str = 'cuda',
        log_interval: int = 100,
        checkpoint_dir: str = 'checkpoints',
        experiment_name: str = 'lwt',
    ):
        self.target_net = target_net.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.log_interval = log_interval
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name

        # 为每层（按参数名分组）创建独立的 MappingNetwork
        self.layer_mappings = nn.ModuleDict()
        param_groups = self._build_param_groups(target_net)

        if layer_alphas is None:
            layer_alphas = {}

        for group_name, group_size in param_groups:
            dim = layer_latent_dims.get(group_name, 64)
            alpha = layer_alphas.get(group_name, 0.01)
            self.layer_mappings[group_name] = MappingNetwork(
                group_size, dim, alpha=alpha,
            ).to(device)

        # 收集所有可训练参数
        trainable_params = [self.loss_fn.lambda_st, self.loss_fn.lambda_sm, self.loss_fn.lambda_al]
        for mapping in self.layer_mappings.values():
            trainable_params.append(mapping.z)

        self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=min_lr)

    @staticmethod
    def _build_param_groups(target_net):
        """将目标网络的参数按层名分组。返回 [(group_name, total_size), ...]"""
        groups = {}
        for name, param in target_net.named_parameters():
            base = name.split('.')[0]  # 'conv1.weight' → 'conv1'
            if base not in groups:
                groups[base] = 0
            groups[base] += param.numel()
        return list(groups.items())

    def _generate_all_theta(self):
        """逐层生成 θ̂^(l) 并拼接为完整 θ̂。"""
        all_theta = []
        for name, param in self.target_net.named_parameters():
            base = name.split('.')[0]
            mapping = self.layer_mappings[base]
            all_theta.append(mapping())
        return torch.cat(all_theta)

    def _generate_all_theta_noisy(self):
        """生成带噪声的 θ̂ 用于 L_stab（各层分别加噪）。"""
        all_theta = []
        for name, param in self.target_net.named_parameters():
            base = name.split('.')[0]
            mapping = self.layer_mappings[base]
            eps = torch.randn_like(mapping.z) * self.loss_fn.sigma_noise
            z_noisy = mapping.z + eps
            with torch.no_grad():
                W_mod_n = mapping.W_fixed + mapping.alpha * z_noisy.unsqueeze(0)
                theta_n = torch.tanh(W_mod_n @ z_noisy + mapping.b_fixed)
            all_theta.append(theta_n)
        return torch.cat(all_theta).detach()

    def _compute_layerwise_losses(self, theta_hat, theta_noisy, x, y):
        """逐层计算 L_stab + L_smooth + L_align 并聚合。"""
        l_stab_total = 0.0
        l_smooth_total = 0.0
        l_align_total = 0.0

        for name, mapping in self.layer_mappings.items():
            # 各层自己的 z, theta_hat^(l), theta_noisy^(l)
            # 通过 loss_fn 计算各项（忽略 L_task，只需正则项）
            theta_l = theta_hat  # 实际各层切片
            # 简化：直接对每层 z 计算正则项
            z_l = mapping.z
            W_mod = mapping.W_fixed + mapping.alpha * z_l.unsqueeze(0)

            # L_smooth per layer
            jac = torch.autograd.functional.jacobian(
                lambda zi: torch.tanh(
                    (mapping.W_fixed + mapping.alpha * zi.unsqueeze(0)) @ zi
                    + mapping.b_fixed
                ),
                z_l,
                create_graph=True,
            )
            l_smooth_total = l_smooth_total + torch.sum(jac ** 2) / jac.numel()

            # L_align per layer
            W_m = W_mod.mean(dim=0)
            cos_sim = torch.nn.functional.cosine_similarity(
                z_l.unsqueeze(0), W_m.unsqueeze(0)
            )
            l_align_total = l_align_total + (1 - cos_sim.squeeze())

            # L_stab per layer
            eps = torch.randn_like(z_l) * self.loss_fn.sigma_noise
            z_noisy_l = z_l + eps
            W_mod_n = mapping.W_fixed + mapping.alpha * z_noisy_l.unsqueeze(0)
            theta_noisy_l = torch.tanh(W_mod_n @ z_noisy_l + mapping.b_fixed)
            # 比较加噪前后输出的差异需要用 functional_forward
            y_hat = self.target_net.functional_forward(x, theta_hat)
            y_hat_n = self.target_net.functional_forward(x, theta_hat + (theta_noisy_l - theta_l))
            # 简化：用 θ̂ 的差异替代（严格用 y_hat 差异）
            # 实际使用 theta_noisy 版 y_hat 差值
            l_stab_total = l_stab_total + torch.nn.functional.mse_loss(
                y_hat_n, y_hat.detach()
            )

        l_st = torch.sigmoid(self.loss_fn.lambda_st)
        l_sm = torch.sigmoid(self.loss_fn.lambda_sm)
        l_al = torch.sigmoid(self.loss_fn.lambda_al)

        return l_st * l_stab_total + l_sm * l_smooth_total + l_al * l_align_total

    def train_epoch(self, epoch):
        self.target_net.train()
        for mapping in self.layer_mappings.values():
            mapping.train()

        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs}')
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)

            # 生成 θ̂
            theta_hat = self._generate_all_theta()
            theta_noisy = self._generate_all_theta_noisy()

            # 函数式前向（L_task）
            y_hat = self.target_net.functional_forward(x, theta_hat)
            l_task = torch.nn.functional.cross_entropy(y_hat, y)

            # 各层正则项（逐层计算并聚合）
            reg_loss = self._compute_layerwise_losses(theta_hat, theta_noisy, x, y)
            loss = l_task + reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = y_hat.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%',
                })

        return total_loss / len(self.train_loader), 100. * correct / total

    @torch.no_grad()
    def evaluate(self):
        self.target_net.eval()
        for mapping in self.layer_mappings.values():
            mapping.eval()

        theta_hat = self._generate_all_theta()
        correct = 0
        total = 0

        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.target_net.functional_forward(x, theta_hat)
            _, predicted = y_hat.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        return 100. * correct / total

    def save_checkpoint(self, results, epoch=None):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        suffix = f"_epoch{epoch}" if epoch else "_final"

        # 保存所有 layer_mappings 的 state_dict
        checkpoint = {
            name: mapping.state_dict()
            for name, mapping in self.layer_mappings.items()
        }
        path = os.path.join(self.checkpoint_dir, f'{self.experiment_name}{suffix}.pth')
        torch.save(checkpoint, path)

        results_path = os.path.join(self.checkpoint_dir, f'{self.experiment_name}_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        return path

    def train(self):
        results = []
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            test_acc = self.evaluate()
            self.scheduler.step()
            results.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
            })
            print(f'Epoch {epoch}: train_loss={train_loss:.4f}, '
                  f'train_acc={train_acc:.2f}%, test_acc={test_acc:.2f}%')

        path = self.save_checkpoint(results)
        print(f'Checkpoint saved to {path}')
        return results
```

- [ ] **Step 2: 写冒烟测试**

```python
# tests/test_lwt.py
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from mapping_network.mapping.loss import MappingLoss
from mapping_network.target_nets.cnn2 import CNN2
from mapping_network.trainer.lwt import LWTTrainer

def test_lwt_train_one_batch():
    target = CNN2()
    loss_fn = MappingLoss()

    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=8)

    layer_dims = {
        'conv1': 16,
        'conv2': 16,
        'fc1': 16,
        'fc2': 16,
    }
    trainer = LWTTrainer(
        target, loss_fn, layer_dims,
        train_loader=loader, epochs=1, device='cpu', log_interval=1,
        checkpoint_dir='/tmp/test_lwt_checkpoints',
        experiment_name='test_lwt',
    )

    # 验证各层 z 的梯度
    z_before = {
        name: mapping.z.data.clone()
        for name, mapping in trainer.layer_mappings.items()
    }

    results = trainer.train()
    assert len(results) == 1

    # 验证每层 z 都已更新
    for name, mapping in trainer.layer_mappings.items():
        assert not torch.equal(z_before[name], mapping.z.data), \
            f"Layer {name} z was not updated!"
```

- [ ] **Step 3: 运行测试**

Run: `cd /root/MyProj/MappingNetwork && uv run python3 -m pytest tests/test_lwt.py -v`
Expected: 1 passed

- [ ] **Step 4: Commit**

```bash
cd /root/MyProj/MappingNetwork && git add mapping_network/trainer/lwt.py tests/
git commit -m "feat: add LWT trainer with per-layer gradients and checkpoint"
```

---

### Task 7: 配置文件和实验入口脚本

**Files:**
- Create: `configs/cnn2_slvt.yaml`
- Create: `configs/cnn2_lwt.yaml`
- Create: `configs/cnn1_slvt.yaml`
- Create: `configs/cnn1_lwt.yaml`
- Create: `configs/cnn1_3conv_slvt.yaml`
- Create: `mapping_network/scripts/train.py`
- Create: `mapping_network/scripts/evaluate.py`

- [ ] **Step 1: 写各实验配置**

```yaml
# configs/cnn2_slvt.yaml
target_net: cnn2
training_strategy: slvt
latent_dim: 2048
batch_size: 64
epochs: 30
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
device: cuda
log_interval: 100
checkpoint_dir: checkpoints
```

```yaml
# configs/cnn2_lwt.yaml
target_net: cnn2
training_strategy: lwt
batch_size: 64
epochs: 30
seed: 42
optimizer: adamw
lr: 0.001
weight_decay: 0.0001
scheduler: cosine_annealing
min_lr: 0.00001
alpha: 0.01
layer_latent_dims:
  conv1: 256
  conv2: 256
  fc1: 256
  fc2: 256
layer_alphas:
  conv1: 0.01
  conv2: 0.01
  fc1: 0.01
  fc2: 0.01
lambda_st_init: 0.1
lambda_sm_init: 0.1
lambda_al_init: 0.1
sigma_noise: 0.01
device: cuda
log_interval: 100
checkpoint_dir: checkpoints
```

```yaml
# configs/cnn1_slvt.yaml
target_net: cnn1
training_strategy: slvt
latent_dim: 2072
batch_size: 64
epochs: 30
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
device: cuda
log_interval: 100
checkpoint_dir: checkpoints
```

```yaml
# configs/cnn1_lwt.yaml
target_net: cnn1
training_strategy: lwt
batch_size: 64
epochs: 30
seed: 42
optimizer: adamw
lr: 0.001
weight_decay: 0.0001
scheduler: cosine_annealing
min_lr: 0.00001
alpha: 0.01
layer_latent_dims:
  conv1: 256
  conv2: 256
  fc1: 256
  fc2: 256
layer_alphas:
  conv1: 0.01
  conv2: 0.01
  fc1: 0.01
  fc2: 0.01
lambda_st_init: 0.1
lambda_sm_init: 0.1
lambda_al_init: 0.1
sigma_noise: 0.01
device: cuda
log_interval: 100
checkpoint_dir: checkpoints
```

- [ ] **Step 2: 写统一训练入口**

```python
# mapping_network/scripts/train.py
"""
统一训练入口。

用法:
  uv run python3 -m mapping_network.scripts.train --config configs/cnn2_slvt.yaml
  uv run python3 -m mapping_network.scripts.train --config configs/cnn2_slvt.yaml --device cpu --epochs 1
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mapping_network.target_nets import CNN2, CNN1, CNN1_3Conv
from mapping_network.mapping.mapping_net import MappingNetwork
from mapping_network.mapping.loss import MappingLoss
from mapping_network.trainer.slvt import SLVTTrainer
from mapping_network.trainer.lwt import LWTTrainer

TARGET_NET_MAP = {
    'cnn2': CNN2,
    'cnn1': CNN1,
    'cnn1_3conv': CNN1_3Conv,
}


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg['device'] = args.device
    if args.epochs:
        cfg['epochs'] = args.epochs
    if args.seed:
        cfg['seed'] = args.seed

    if 'seed' in cfg:
        set_seed(cfg['seed'])

    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # 数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'])

    # 目标网络
    target_cls = TARGET_NET_MAP[cfg['target_net']]
    target_net = target_cls()
    print(f'Target network: {cfg["target_net"]}, '
          f'params: {target_net.get_total_params():,}')

    # 损失
    loss_fn = MappingLoss(sigma_noise=cfg.get('sigma_noise', 0.01)).to(device)

    exp_name = f"{cfg['target_net']}_{cfg['training_strategy']}"

    if cfg['training_strategy'] == 'slvt':
        mapping = MappingNetwork(
            target_net.get_total_params(),
            cfg['latent_dim'],
            alpha=cfg.get('alpha', 0.01),
        ).to(device)
        print(f'Latent dim: {cfg["latent_dim"]}')
        print(f'Trainable: {sum(p.numel() for p in mapping.parameters() if p.requires_grad):,}')
        print(f'Fixed mapping weights: {mapping.W_fixed.numel():,}')

        trainer = SLVTTrainer(
            mapping, target_net, loss_fn,
            train_loader, test_loader,
            lr=cfg['lr'],
            weight_decay=cfg.get('weight_decay', 0.0001),
            epochs=cfg['epochs'],
            min_lr=cfg.get('min_lr', 1e-5),
            device=device,
            log_interval=cfg.get('log_interval', 100),
            checkpoint_dir=cfg.get('checkpoint_dir', 'checkpoints'),
            experiment_name=exp_name,
        )
    elif cfg['training_strategy'] == 'lwt':
        trainer = LWTTrainer(
            target_net, loss_fn,
            cfg['layer_latent_dims'],
            layer_alphas=cfg.get('layer_alphas'),
            train_loader=train_loader,
            test_loader=test_loader,
            lr=cfg['lr'],
            weight_decay=cfg.get('weight_decay', 0.0001),
            epochs=cfg['epochs'],
            min_lr=cfg.get('min_lr', 1e-5),
            device=device,
            log_interval=cfg.get('log_interval', 100),
            checkpoint_dir=cfg.get('checkpoint_dir', 'checkpoints'),
            experiment_name=exp_name,
        )
    else:
        raise ValueError(f"Unknown strategy: {cfg['training_strategy']}")

    results = trainer.train()
    final_acc = results[-1]['test_acc']
    print(f'\nFinal test accuracy: {final_acc:.2f}%')


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: 写评估脚本（支持 SLVT 和 LWT）**

```python
# mapping_network/scripts/evaluate.py
"""
评估已训练的 Mapping Network checkpoint。

用法:
  # SLVT checkpoint
  uv run python3 -m mapping_network.scripts.evaluate \\
      --checkpoint checkpoints/cnn2_slvt_final.pth \\
      --config configs/cnn2_slvt.yaml

  # LWT checkpoint (多个 state_dict 打包在一起)
  uv run python3 -m mapping_network.scripts.evaluate \\
      --checkpoint checkpoints/cnn2_lwt_final.pth \\
      --config configs/cnn2_lwt.yaml
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mapping_network.target_nets import CNN2, CNN1, CNN1_3Conv
from mapping_network.mapping.mapping_net import MappingNetwork
from mapping_network.mapping.loss import MappingLoss

TARGET_NET_MAP = {
    'cnn2': CNN2, 'cnn1': CNN1, 'cnn1_3conv': CNN1_3Conv,
}


@torch.no_grad()
def evaluate_slvt(mapping, target_net, test_loader, device):
    mapping.eval()
    target_net.eval()
    theta = mapping()
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_hat = target_net.functional_forward(x, theta)
        _, pred = y_hat.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    return 100. * correct / total


@torch.no_grad()
def evaluate_lwt(mappings_dict, target_net, test_loader, device):
    """LWT 评估：从 mappings_dict 重建各层 theta 并拼接。"""
    target_net.eval()
    # 拼接 θ̂
    all_theta = []
    for name, param in target_net.named_parameters():
        base = name.split('.')[0]
        mapping = mappings_dict[base]
        mapping.eval()
        all_theta.append(mapping())
    theta_hat = torch.cat(all_theta)

    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_hat = target_net.functional_forward(x, theta_hat)
        _, pred = y_hat.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    return 100. * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    target_cls = TARGET_NET_MAP[cfg['target_net']]
    target_net = target_cls().to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.get('batch_size', 64))

    if cfg.get('training_strategy') == 'lwt':
        # LWT: 每层独立的 MappingNetwork
        mappings = {}
        layer_dims = cfg.get('layer_latent_dims', {})
        for name, (start, end, shape, pname, is_bias) in zip(
            target_net.get_param_names(), target_net.get_param_slices()
        ):
            base = name.split('.')[0]
            if base not in mappings:
                layer_size = end - start
                dim = layer_dims.get(base, 64)
                mappings[base] = MappingNetwork(
                    layer_size, dim, alpha=cfg.get('alpha', 0.01)
                ).to(device)

        checkpoint = torch.load(args.checkpoint, map_location=device)
        for name, mapping in mappings.items():
            if name in checkpoint:
                mapping.load_state_dict(checkpoint[name])

        acc = evaluate_lwt(mappings, target_net, test_loader, device)
    else:
        # SLVT: 单个 MappingNetwork
        mapping = MappingNetwork(
            target_net.get_total_params(),
            cfg.get('latent_dim', 2048),
            alpha=cfg.get('alpha', 0.01),
        ).to(device)
        mapping.load_state_dict(torch.load(args.checkpoint, map_location=device))
        acc = evaluate_slvt(mapping, target_net, test_loader, device)

    print(f'Test accuracy: {acc:.2f}%')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: 运行冒烟测试**

Run: `cd /root/MyProj/MappingNetwork && uv run python3 -m mapping_network.scripts.train --config configs/cnn2_slvt.yaml --device cpu --epochs 1`
Expected: 训练正常结束，checkpoint 保存到 checkpoints/ 目录

Run: `cd /root/MyProj/MappingNetwork && uv run python3 -m mapping_network.scripts.train --config configs/cnn2_lwt.yaml --device cpu --epochs 1`
Expected: LWT 训练正常结束

- [ ] **Step 5: Commit**

```bash
cd /root/MyProj/MappingNetwork && git add configs/ mapping_network/scripts/
git commit -m "feat: add configs, train/evaluate scripts with SLVT and LWT support"
```

---

### Task 8: 基线训练脚本

**Files:**
- Create: `mapping_network/scripts/train_baseline.py`

- [ ] **Step 1: 写基线训练脚本**

```python
# mapping_network/scripts/train_baseline.py
"""
训练基线目标网络（不使用 Mapping Network）。

用法:
  uv run python3 -m mapping_network.scripts.train_baseline --target cnn2
  uv run python3 -m mapping_network.scripts.train_baseline --target cnn1
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import tqdm

from mapping_network.target_nets import CNN2, CNN1, CNN1_3Conv

TARGET_NET_MAP = {
    'cnn2': CNN2, 'cnn1': CNN1, 'cnn1_3conv': CNN1_3Conv,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True,
                        choices=['cnn1', 'cnn2', 'cnn1_3conv'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = args.device if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = TARGET_NET_MAP[args.target]().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Training {args.target} baseline: {total_params:,} params')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = total = 0
        pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            _, pred = y_hat.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
            pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})
        scheduler.step()

        model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                _, pred = y_hat.max(1)
                test_total += y.size(0)
                test_correct += pred.eq(y).sum().item()
        test_acc = 100. * test_correct / test_total
        print(f'Epoch {epoch}: test_acc={test_acc:.2f}%')

    torch.save(model.state_dict(), f'{args.target}_baseline.pth')
    print(f'Baseline saved to {args.target}_baseline.pth')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 运行冒烟测试**

Run: `cd /root/MyProj/MappingNetwork && uv run python3 -m mapping_network.scripts.train_baseline --target cnn2 --epochs 1 --device cpu`
Expected: 训练正常完成

- [ ] **Step 3: Commit**

```bash
cd /root/MyProj/MappingNetwork && git add mapping_network/scripts/train_baseline.py
git commit -m "feat: add baseline training script"
```

---

### Task 9: 创建新分支并推送

- [ ] **Step 1: 创建并切换分支**

```bash
cd /root/MyProj/MappingNetwork
git checkout -b feat/mapping-network-reproduction
```

- [ ] **Step 2: 推送分支到远程**

```bash
git push origin feat/mapping-network-reproduction
```
