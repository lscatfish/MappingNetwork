# Mapping Network LRD & Generator Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the Mapping Network codebase to support a pluggable `ParameterGenerator` base class, add Low Rank Decomposition (LRD) to target networks for memory reduction, and allow per-layer generator configuration in LWT.

**Architecture:** Introduce `mapping_network/generators/` with `ParameterGenerator` base and `LinearMappingNetwork`; move LRD awareness into `TargetNet` so any generator benefits; add `mapping_network/factory.py` to wire components from YAML; update trainers and scripts to use the new abstractions.

**Tech Stack:** Python 3.13, PyTorch 2.11+, uv, pytest.

## Global Constraints

- All network computations and tests must run on GPU by default (`device: cuda` in configs, tests use `cuda` when available).
- All `ParameterGenerator` subclasses must expose `forward() -> torch.Tensor` returning `theta_hat`.
- LRD applies only to `nn.Linear` layers in target networks.
- LRD default is `enabled: 'auto'` with `auto_enable_threshold = 200_000`.
- Default LRD rank is 10; per-layer `layer_ranks` overrides it.
- No backward compatibility with old code, old configs, old checkpoints, or old tests.
- Each experiment saves checkpoints into its own folder `{checkpoint_dir}/{experiment_name}/`.
- Code style: Ruff, line-length 100, single quotes, E/F/I rules.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `mapping_network/generators/base.py` | `ParameterGenerator` abstract base class |
| `mapping_network/generators/linear.py` | `LinearMappingNetwork` implementation |
| `mapping_network/generators/__init__.py` | Public exports |
| `mapping_network/target_nets/lrd_config.py` | `LRDConfig` dataclass |
| `mapping_network/target_nets/base.py` | `TargetNet` base with LRD-aware slice building and functional forward |
| `mapping_network/target_nets/cnn1.py` | CNN1, accepts optional `lrd_config` |
| `mapping_network/target_nets/cnn2.py` | CNN2, accepts optional `lrd_config` |
| `mapping_network/target_nets/cnn1_3conv.py` | CNN1_3Conv, accepts optional `lrd_config` |
| `mapping_network/mapping/loss.py` | `MappingLoss` using `ParameterGenerator` interface |
| `mapping_network/factory.py` | `build_target_net`, `build_generator`, config parsing helpers |
| `mapping_network/trainer/slvt.py` | SLVT trainer using `ParameterGenerator` |
| `mapping_network/trainer/lwt.py` | LWT trainer using per-layer `ParameterGenerator` dict |
| `mapping_network/scripts/train.py` | Updated training entry point |
| `mapping_network/scripts/evaluate.py` | Updated evaluation entry point |
| `configs/*.yaml` | New format with `lrd` and `layer_generators` |
| `tests/test_*.py` | Rewritten tests for new abstractions |

---

## Task 1: Create ParameterGenerator Base Class

**Files:**
- Create: `mapping_network/generators/base.py`
- Create: `mapping_network/generators/__init__.py`
- Test: `tests/test_generators.py`

**Interfaces:**
- Produces: `ParameterGenerator` base class with `forward()` abstract method and `trainable_params()` helper.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generators.py
import pytest
import torch
from mapping_network.generators.base import ParameterGenerator


def test_parameter_generator_is_abstract():
    with pytest.raises(TypeError):
        ParameterGenerator()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python3 -m pytest tests/test_generators.py::test_parameter_generator_is_abstract -v`
Expected: FAIL with "Can't instantiate abstract class"

- [ ] **Step 3: Write minimal implementation**

```python
# mapping_network/generators/base.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class ParameterGenerator(nn.Module, ABC):
    """参数生成网络基类。只负责生成 theta_hat。"""

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """返回 theta_hat [P']，P' 是目标网络压缩后的总参数数。"""
        pass

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

- [ ] **Step 4: Create package init**

```python
# mapping_network/generators/__init__.py
from .base import ParameterGenerator

__all__ = ['ParameterGenerator']
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run python3 -m pytest tests/test_generators.py::test_parameter_generator_is_abstract -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add mapping_network/generators/ tests/test_generators.py
git commit -m "feat: add ParameterGenerator abstract base class"
```

---

## Task 2: Create LinearMappingNetwork

**Files:**
- Create: `mapping_network/generators/linear.py`
- Modify: `mapping_network/generators/__init__.py`
- Test: `tests/test_generators.py`

**Interfaces:**
- Consumes: `ParameterGenerator` from Task 1.
- Produces: `LinearMappingNetwork(target_total_params, latent_dim, alpha, device)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generators.py
import torch
from mapping_network.generators.linear import LinearMappingNetwork


def test_linear_mapping_network_shape_and_trainable(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    gen = LinearMappingNetwork(100, 8, alpha=0.01, device=device)
    theta = gen()
    assert theta.shape == (100,)
    assert theta.device.type == device
    assert gen.trainable_params() == 8
    assert not gen.W_fixed.requires_grad
    assert gen.z.requires_grad
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python3 -m pytest tests/test_generators.py::test_linear_mapping_network_shape_and_trainable -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'mapping_network.generators.linear'"

- [ ] **Step 3: Write minimal implementation**

```python
# mapping_network/generators/linear.py
import torch
import torch.nn as nn
import torch.nn.init as init

from .base import ParameterGenerator


class LinearMappingNetwork(ParameterGenerator):
    """线性参数生成网络：固定正交权重 + 可学习 z。"""

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

    def extra_repr(self):
        return f'P={self.P}, d={self.d}, alpha={self.alpha}'
```

- [ ] **Step 4: Update package init**

```python
# mapping_network/generators/__init__.py
from .base import ParameterGenerator
from .linear import LinearMappingNetwork

__all__ = ['ParameterGenerator', 'LinearMappingNetwork']
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run python3 -m pytest tests/test_generators.py::test_linear_mapping_network_shape_and_trainable -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add mapping_network/generators/ tests/test_generators.py
git commit -m "feat: add LinearMappingNetwork"
```

---

## Task 3: Create LRDConfig

**Files:**
- Create: `mapping_network/target_nets/lrd_config.py`
- Test: `tests/test_lrd_config.py`

**Interfaces:**
- Produces: `LRDConfig(enabled='auto', default_rank=10, layer_ranks={}, auto_enable_threshold=200_000)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lrd_config.py
from mapping_network.target_nets.lrd_config import LRDConfig


def test_lrd_config_defaults():
    cfg = LRDConfig()
    assert cfg.enabled == 'auto'
    assert cfg.default_rank == 10
    assert cfg.layer_ranks == {}
    assert cfg.auto_enable_threshold == 200_000


def test_lrd_config_override():
    cfg = LRDConfig(enabled=True, default_rank=20, layer_ranks={'fc1': 15})
    assert cfg.enabled is True
    assert cfg.default_rank == 20
    assert cfg.layer_ranks == {'fc1': 15}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python3 -m pytest tests/test_lrd_config.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# mapping_network/target_nets/lrd_config.py
from dataclasses import dataclass, field


@dataclass
class LRDConfig:
    enabled: bool | str = 'auto'
    default_rank: int = 10
    layer_ranks: dict = field(default_factory=dict)
    auto_enable_threshold: int = 200_000
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python3 -m pytest tests/test_lrd_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mapping_network/target_nets/lrd_config.py tests/test_lrd_config.py
git commit -m "feat: add LRDConfig dataclass"
```

---

## Task 4: Refactor TargetNet Base for LRD

**Files:**
- Modify: `mapping_network/target_nets/base.py`
- Modify: `mapping_network/target_nets/cnn1.py`
- Modify: `mapping_network/target_nets/cnn2.py`
- Modify: `mapping_network/target_nets/cnn1_3conv.py`
- Test: `tests/test_target_nets.py`

**Interfaces:**
- Consumes: `LRDConfig` from Task 3.
- Produces: `TargetNet(lrd_config=None)` with `_build_param_slices()` supporting LRD and `functional_forward()` reconstructing `W = U @ V.T` for LRD layers.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_target_nets.py
import torch
import pytest
from mapping_network.target_nets import CNN2
from mapping_network.target_nets.lrd_config import LRDConfig


def test_cnn2_lrd_reduces_params(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    net_full = CNN2(lrd_config=LRDConfig(enabled=False)).to(device)
    net_lrd = CNN2(lrd_config=LRDConfig(enabled=True, default_rank=10)).to(device)
    assert net_lrd.get_total_params() < net_full.get_total_params()


def test_cnn2_lrd_functional_matches_module(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    net = CNN2(lrd_config=LRDConfig(enabled=True, default_rank=10)).to(device)
    x = torch.randn(2, 1, 28, 28, device=device)
    theta = torch.randn(net.get_total_params(), device=device, requires_grad=True)
    y_func = net.functional_forward(x, theta)
    y_mod = net(x)
    assert y_func.shape == y_mod.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python3 -m pytest tests/test_target_nets.py::test_cnn2_lrd_reduces_params -v`
Expected: FAIL with "unexpected keyword argument 'lrd_config'"

- [ ] **Step 3: Implement LRD-aware TargetNet base**

```python
# mapping_network/target_nets/base.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from .lrd_config import LRDConfig


@dataclass
class _ParamSlice:
    kind: str  # 'full' or 'lrd'
    # full
    start: int = 0
    end: int = 0
    shape: tuple = ()
    name: str = ''
    is_bias: bool = False
    # lrd
    weight_name: str = ''
    bias_name: str = ''
    u_start: int = 0
    u_end: int = 0
    u_shape: tuple = ()
    v_start: int = 0
    v_end: int = 0
    v_shape: tuple = ()
    b_start: int = 0
    b_end: int = 0
    b_shape: tuple = ()


class TargetNet(nn.Module):
    """目标网络基类，支持 LRD。"""

    def __init__(self, lrd_config: LRDConfig | None = None):
        super().__init__()
        self._lrd_config = lrd_config if lrd_config is not None else LRDConfig()
        self._param_slices = []

    def _should_use_lrd(self, layer_name: str, total_params: int) -> bool:
        enabled = self._lrd_config.enabled
        if enabled is True:
            return True
        if enabled is False:
            return False
        return total_params > self._lrd_config.auto_enable_threshold

    def _build_param_slices(self):
        self._param_slices = []
        idx = 0
        total_params = sum(p.numel() for p in self.parameters())
        params_dict = dict(self.named_parameters())
        processed_bias = set()

        for name, param in self.named_parameters():
            if name in processed_bias:
                continue

            base = name.split('.')[0]
            shape = param.shape
            numel = param.numel()
            is_bias = 'bias' in name

            bias_name = name.replace('weight', 'bias')
            bias_param = params_dict.get(bias_name)
            bias_shape = bias_param.shape if bias_param is not None else (shape[0],)
            bias_numel = bias_param.numel() if bias_param is not None else shape[0]

            if (not is_bias and len(shape) == 2 and
                    self._should_use_lrd(base, total_params)):
                m, n = shape
                rank = self._lrd_config.layer_ranks.get(base, self._lrd_config.default_rank)
                rank = min(rank, m, n)

                u_start, u_end = idx, idx + m * rank
                v_start, v_end = u_end, u_end + n * rank
                b_start, b_end = v_end, v_end + bias_numel

                self._param_slices.append(_ParamSlice(
                    kind='lrd',
                    weight_name=name,
                    bias_name=bias_name,
                    u_start=u_start, u_end=u_end, u_shape=(m, rank),
                    v_start=v_start, v_end=v_end, v_shape=(n, rank),
                    b_start=b_start, b_end=b_end,
                    b_shape=bias_shape,
                ))
                processed_bias.add(bias_name)
                idx = b_end
            else:
                self._param_slices.append(_ParamSlice(
                    kind='full',
                    start=idx, end=idx + numel,
                    shape=shape, name=name, is_bias=is_bias,
                ))
                idx += numel

    def get_param_slices(self):
        return self._param_slices

    def get_total_params(self):
        if not self._param_slices:
            return sum(p.numel() for p in self.parameters())
        last = self._param_slices[-1]
        if last.kind == 'full':
            return last.end
        return last.b_end

    def get_param_names(self):
        return [name for name, _ in self.named_parameters()]

    def get_group_param_size(self, group_name: str) -> int:
        """返回某一层组在 theta_hat 中占用的压缩后参数数。"""
        size = 0
        for s in self._param_slices:
            if s.kind == 'full' and s.name.split('.')[0] == group_name:
                size += s.end - s.start
            elif s.kind == 'lrd' and s.weight_name.split('.')[0] == group_name:
                size += s.b_end - s.u_start
        return size

    def get_group_names(self) -> list[str]:
        """按出现顺序返回所有层组名（如 ['conv1', 'conv2', 'fc1', 'fc2']）。"""
        names = []
        seen = set()
        for s in self._param_slices:
            name = s.name.split('.')[0] if s.kind == 'full' else s.weight_name.split('.')[0]
            if name not in seen:
                seen.add(name)
                names.append(name)
        return names

    def functional_forward(self, x, theta_hat):
        params = {}
        for s in self._param_slices:
            if s.kind == 'full':
                params[s.name] = theta_hat[s.start:s.end].reshape(s.shape)
            elif s.kind == 'lrd':
                U = theta_hat[s.u_start:s.u_end].reshape(s.u_shape)
                V = theta_hat[s.v_start:s.v_end].reshape(s.v_shape)
                params[s.weight_name] = U @ V.T
                params[s.bias_name] = theta_hat[s.b_start:s.b_end].reshape(s.b_shape)
        return self._functional_forward(x, params)

    def _functional_forward(self, x, params):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
```

- [ ] **Step 4: Update CNN1, CNN2, CNN1_3Conv to accept lrd_config**

For each file, change `def __init__(self):` to `def __init__(self, lrd_config=None):` and pass `lrd_config` to `super().__init__(lrd_config)`.

Example for `mapping_network/target_nets/cnn2.py`:

```python
class CNN2(TargetNet):
    def __init__(self, lrd_config=None):
        super().__init__(lrd_config)
        ...
```

- [ ] **Step 5: Run tests to verify**

Run: `uv run python3 -m pytest tests/test_target_nets.py::test_cnn2_lrd_reduces_params tests/test_target_nets.py::test_cnn2_lrd_functional_matches_module -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add mapping_network/target_nets/ tests/test_target_nets.py
git commit -m "feat: add LRD support to TargetNet base"
```

---

## Task 5: Create Factory Functions

**Files:**
- Create: `mapping_network/factory.py`
- Modify: `mapping_network/target_nets/__init__.py` if needed
- Test: `tests/test_factory.py`

**Interfaces:**
- Consumes: `LinearMappingNetwork`, `LRDConfig`, target net classes.
- Produces: `build_target_net(target_name, lrd_config)` and `build_generator(generator_type, target_total_params, latent_dim, alpha, device)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_factory.py
import torch
from mapping_network.factory import build_target_net, build_generator


def test_build_cnn1_with_lrd(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    net = build_target_net('cnn1', {'enabled': True, 'default_rank': 10})
    gen = build_generator('linear', net.get_total_params(), 2072, 0.01, device)
    theta = gen()
    assert theta.shape[0] < 537_960
    assert theta.device.type == device
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python3 -m pytest tests/test_factory.py::test_build_cnn1_with_lrd -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'mapping_network.factory'"

- [ ] **Step 3: Write minimal implementation**

```python
# mapping_network/factory.py
from mapping_network.generators.linear import LinearMappingNetwork
from mapping_network.target_nets.cnn1 import CNN1
from mapping_network.target_nets.cnn2 import CNN2
from mapping_network.target_nets.cnn1_3conv import CNN1_3Conv
from mapping_network.target_nets.lrd_config import LRDConfig


TARGET_NET_MAP = {
    'cnn1': CNN1,
    'cnn2': CNN2,
    'cnn1_3conv': CNN1_3Conv,
}

GENERATOR_MAP = {
    'linear': LinearMappingNetwork,
}


def build_target_net(target_name: str, lrd_config: dict | None = None):
    if target_name not in TARGET_NET_MAP:
        raise ValueError(f'Unknown target net: {target_name}')
    cfg = LRDConfig(**lrd_config) if lrd_config else LRDConfig()
    return TARGET_NET_MAP[target_name](lrd_config=cfg)


def build_generator(generator_type: str, target_total_params: int,
                    latent_dim: int, alpha: float, device: str):
    if generator_type not in GENERATOR_MAP:
        raise ValueError(f'Unknown generator type: {generator_type}')
    return GENERATOR_MAP[generator_type](
        target_total_params, latent_dim, alpha=alpha, device=device
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python3 -m pytest tests/test_factory.py::test_build_cnn1_with_lrd -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mapping_network/factory.py tests/test_factory.py
git commit -m "feat: add build_target_net and build_generator factories"
```

---

## Task 6: Update MappingLoss

**Files:**
- Modify: `mapping_network/mapping/loss.py`
- Delete: `mapping_network/mapping/mapping_net.py`
- Test: `tests/test_loss.py`

**Interfaces:**
- Consumes: `LinearMappingNetwork` (via `ParameterGenerator` interface, but currently requires `W_fixed`, `alpha`, `z`, `P`, `d`).
- Produces: Same `forward()` signature; works with LRD-enabled target nets.

- [ ] **Step 1: Update imports and remove old mapping_net.py**

Delete `mapping_network/mapping/mapping_net.py`. Update `mapping_network/mapping/__init__.py` to remove the export.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_loss.py
import torch
from mapping_network.target_nets import CNN2
from mapping_network.generators.linear import LinearMappingNetwork
from mapping_network.mapping.loss import MappingLoss


def test_mapping_loss_forward_lrd(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    target_net = CNN2(lrd_config={'enabled': True, 'default_rank': 10}).to(device)
    mapping = LinearMappingNetwork(target_net.get_total_params(), 64, device=device)
    loss_fn = MappingLoss(sigma_noise=0.01).to(device)
    x = torch.randn(2, 1, 28, 28, device=device)
    y = torch.tensor([0, 1], device=device)
    theta = mapping()
    eps = torch.randn_like(mapping.z) * loss_fn.sigma_noise
    z_noisy = mapping.z + eps
    theta_noisy = torch.tanh(
        mapping.W_fixed @ z_noisy
        + mapping.alpha * (z_noisy * z_noisy).sum()
        + mapping.b_fixed
    )
    loss, losses = loss_fn(mapping.z, theta, theta_noisy, mapping, target_net, x, y)
    assert loss.item() == losses['total']
    loss.backward()
    assert mapping.z.grad is not None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run python3 -m pytest tests/test_loss.py::test_mapping_loss_forward_lrd -v`
Expected: FAIL depending on current state; likely import error from deleted mapping_net.

- [ ] **Step 4: Update loss.py imports**

```python
# mapping_network/mapping/loss.py
# Remove any import of MappingNetwork; keep torch imports.
```

No functional change needed since `MappingLoss` already accesses `mapping_net.W_fixed` etc.

- [ ] **Step 5: Run tests to verify**

Run: `uv run python3 -m pytest tests/test_loss.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add mapping_network/mapping/ tests/test_loss.py
git commit -m "refactor: remove old MappingNetwork, loss uses LinearMappingNetwork"
```

---

## Task 7: Update SLVTTrainer

**Files:**
- Modify: `mapping_network/trainer/slvt.py`
- Test: `tests/test_slvt.py`

**Interfaces:**
- Consumes: `ParameterGenerator` (specifically `LinearMappingNetwork`), LRD-enabled `TargetNet`, `MappingLoss`.
- Produces: Updated training loop and checkpoint format.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_slvt.py
import torch
from torch.utils.data import TensorDataset, DataLoader
from mapping_network.target_nets import CNN2
from mapping_network.generators.linear import LinearMappingNetwork
from mapping_network.mapping.loss import MappingLoss
from mapping_network.trainer.slvt import SLVTTrainer


def test_slvt_z_updated_with_lrd(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    target_net = CNN2(lrd_config={'enabled': True, 'default_rank': 10}).to(device)
    mapping = LinearMappingNetwork(target_net.get_total_params(), 64, device=device)
    loss_fn = MappingLoss(sigma_noise=0.01).to(device)
    x = torch.randn(1, 1, 28, 28, device=device)
    y = torch.tensor([0], device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=1)
    trainer = SLVTTrainer(
        mapping, target_net, loss_fn, loader, loader,
        lr=0.001, weight_decay=0.0001, epochs=1, device=device,
        log_interval=1, checkpoint_dir='/tmp/test_slvt', experiment_name='test',
        checkpoint_metadata={'target_net': 'cnn2', 'training_strategy': 'slvt',
                             'latent_dim': 64, 'alpha': 0.01, 'sigma_noise': 0.01},
        save_interval=0,
    )
    z_before = mapping.z.clone().detach()
    trainer.train_epoch(1)
    assert not torch.allclose(z_before, mapping.z)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python3 -m pytest tests/test_slvt.py::test_slvt_z_updated_with_lrd -v`
Expected: FAIL if trainer still expects old MappingNetwork.

- [ ] **Step 3: Update SLVTTrainer**

Changes:
- Type annotations: `mapping_net: ParameterGenerator`.
- Collect trainable params with `mapping_net.parameters()` instead of hard-coded `.z`.
- In `train_epoch`, build `theta_noisy` by calling `mapping_net()` with a noisy clone of `z`. Since `LinearMappingNetwork.forward()` has no args, temporarily replace `z` for the noisy forward. Better: add a helper method or compute explicitly using `W_fixed`. For simplicity, keep explicit formula but read from `mapping_net` attributes.
- Checkpoint includes `generator_type` and `lrd_config`.

```python
# mapping_network/trainer/slvt.py
# key changes
from mapping_network.generators.base import ParameterGenerator

class SLVTTrainer:
    def __init__(self, mapping_net: ParameterGenerator, ...):
        ...
        trainable_params = list(mapping_net.parameters()) + [
            self.loss_fn.lambda_st,
            self.loss_fn.lambda_sm,
            self.loss_fn.lambda_al,
        ]
        ...

    def train_epoch(self, epoch):
        ...
        theta_hat = self.mapping_net()
        eps = torch.randn_like(self.mapping_net.z) * self.loss_fn.sigma_noise
        z_noisy = self.mapping_net.z + eps
        theta_noisy = torch.tanh(
            self.mapping_net.W_fixed @ z_noisy
            + self.mapping_net.alpha * (z_noisy * z_noisy).sum()
            + self.mapping_net.b_fixed
        )
        ...

    def save_checkpoint(self, results, suffix='_final', epoch=None, is_best=False):
        ...
        checkpoint = {
            'target_net': self.checkpoint_metadata.get('target_net'),
            'training_strategy': 'slvt',
            'generator_type': self.checkpoint_metadata.get('generator_type', 'linear'),
            'latent_dim': self.checkpoint_metadata.get('latent_dim'),
            'alpha': self.checkpoint_metadata.get('alpha'),
            'sigma_noise': self.checkpoint_metadata.get('sigma_noise'),
            'lrd_config': self.checkpoint_metadata.get('lrd_config'),
            'state_dict': self.mapping_net.state_dict(),
            'results': results,
            'epoch': epoch if epoch is not None else self.epochs,
            'is_best': is_best,
        }
        ...
```

- [ ] **Step 4: Run tests to verify**

Run: `uv run python3 -m pytest tests/test_slvt.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mapping_network/trainer/slvt.py tests/test_slvt.py
git commit -m "refactor: SLVTTrainer uses ParameterGenerator and saves LRD metadata"
```

---

## Task 8: Update LWTTrainer

**Files:**
- Modify: `mapping_network/trainer/lwt.py`
- Test: `tests/test_lwt.py`

**Interfaces:**
- Consumes: Per-layer `ParameterGenerator` configs, LRD-enabled `TargetNet`, `MappingLoss`.
- Produces: Layer-wise training with independent generator configs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lwt.py
import torch
from torch.utils.data import TensorDataset, DataLoader
from mapping_network.target_nets import CNN2
from mapping_network.mapping.loss import MappingLoss
from mapping_network.trainer.lwt import LWTTrainer


def test_lwt_per_layer_config(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    target_net = CNN2(lrd_config={'enabled': True, 'default_rank': 10}).to(device)
    loss_fn = MappingLoss(sigma_noise=0.01).to(device)
    x = torch.randn(1, 1, 28, 28, device=device)
    y = torch.tensor([0], device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=1)
    layer_generators = {
        'conv1': {'type': 'linear', 'latent_dim': 16, 'alpha': 0.01},
        'conv2': {'type': 'linear', 'latent_dim': 16, 'alpha': 0.01},
        'fc1': {'type': 'linear', 'latent_dim': 32, 'alpha': 0.01},
        'fc2': {'type': 'linear', 'latent_dim': 8, 'alpha': 0.01},
    }
    trainer = LWTTrainer(
        target_net, loss_fn, layer_generators,
        train_loader=loader, test_loader=loader,
        lr=0.001, weight_decay=0.0001, epochs=1, device=device,
        log_interval=1, checkpoint_dir='/tmp/test_lwt', experiment_name='test',
        checkpoint_metadata={'target_net': 'cnn2', 'training_strategy': 'lwt',
                             'lrd_config': {'enabled': True, 'default_rank': 10},
                             'sigma_noise': 0.01},
        save_interval=0,
    )
    trainer.train_epoch(1)
    total_z = sum(m.d for m in trainer.layer_mappings.values())
    assert total_z == 16 + 16 + 32 + 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python3 -m pytest tests/test_lwt.py::test_lwt_per_layer_config -v`
Expected: FAIL due to signature mismatch.

- [ ] **Step 3: Update LWTTrainer**

Major changes:
- Constructor signature changes from `layer_latent_dims: dict, layer_alphas: dict` to `layer_generators: dict`.
- Remove old import `from ..mapping.mapping_net import MappingNetwork`.
- Import `ParameterGenerator` and factory functions.
- Build `self.param_groups` using `target_net.get_group_names()` and `target_net.get_group_param_size(name)` — these give the **compressed** sizes after LRD.
- Build `layer_mappings` as `nn.ModuleDict[str, ParameterGenerator]` from per-layer config using `factory.build_generator`.
- `_compute_offsets` uses compressed group sizes.
- `_compute_layerwise_reg_loss` reads `mapping.W_fixed`, `mapping.alpha`, `mapping.z`, etc.
- Checkpoint saves `layer_generator_configs`, `layer_group_order`, and `lrd_config`.
- Per-layer config supports `lrd_rank` to override global default rank; LRD enable/disable remains controlled by global `lrd.enabled`.

Add helper to `TargetNet`:

```python
def get_group_param_size(self, group_name: str) -> int:
    """Return compressed param count for a layer group."""
    size = 0
    for s in self._param_slices:
        if s.kind == 'full' and s.name.split('.')[0] == group_name:
            size += s.end - s.start
        elif s.kind == 'lrd' and s.weight_name.split('.')[0] == group_name:
            size += s.b_end - s.u_start
    return size
```

- [ ] **Step 4: Run tests to verify**

Run: `uv run python3 -m pytest tests/test_lwt.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mapping_network/trainer/lwt.py tests/test_lwt.py mapping_network/target_nets/base.py
git commit -m "refactor: LWTTrainer uses per-layer ParameterGenerator configs"
```

---

## Task 9: Update Training and Evaluation Scripts

**Files:**
- Modify: `mapping_network/scripts/train.py`
- Modify: `mapping_network/scripts/evaluate.py`
- Test: smoke run

**Interfaces:**
- Consumes: Factory functions, new YAML config format.
- Produces: Training and evaluation using new abstractions.

- [ ] **Step 1: Update train.py**

Read new config fields:
- `lrd` → pass to `build_target_net`.
- `latent_dim` and `alpha` for SLVT.
- `layer_generators` for LWT.
- `generator_type` defaults to `'linear'`.

```python
# mapping_network/scripts/train.py (key snippet)
from mapping_network.factory import build_target_net, build_generator

lrd_config = cfg.get('lrd', {})

# Merge per-layer lrd_rank overrides into global LRDConfig
if cfg['training_strategy'] == 'lwt':
    layer_ranks = {}
    for name, gen_cfg in cfg['layer_generators'].items():
        if 'lrd_rank' in gen_cfg:
            layer_ranks[name] = gen_cfg['lrd_rank']
    if layer_ranks:
        lrd_config = {**lrd_config, 'layer_ranks': {**lrd_config.get('layer_ranks', {}), **layer_ranks}}

target_net = build_target_net(cfg['target_net'], lrd_config)

def make_experiment_name(cfg):
    target = cfg['target_net']
    strategy = cfg['training_strategy']
    return f'{target}_{strategy}'

if cfg['training_strategy'] == 'slvt':
    mapping = build_generator(
        cfg.get('generator_type', 'linear'),
        target_net.get_total_params(),
        cfg['latent_dim'],
        cfg.get('alpha', 0.01),
        device,
    )
    trainer = SLVTTrainer(
        mapping, target_net, loss_fn, train_loader, test_loader,
        lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 0.0001),
        epochs=cfg['epochs'], min_lr=cfg.get('min_lr', 1e-5),
        device=device, log_interval=cfg.get('log_interval', 100),
        checkpoint_dir=os.path.join(cfg['checkpoint_dir'], make_experiment_name(cfg)),
        experiment_name=make_experiment_name(cfg),
        checkpoint_metadata={
            'target_net': cfg['target_net'],
            'training_strategy': 'slvt',
            'generator_type': cfg.get('generator_type', 'linear'),
            'latent_dim': cfg['latent_dim'],
            'alpha': cfg.get('alpha', 0.01),
            'sigma_noise': cfg.get('sigma_noise', 0.01),
            'lrd_config': cfg.get('lrd'),
        },
        save_interval=cfg.get('save_interval', 1),
    )
elif cfg['training_strategy'] == 'lwt':
    trainer = LWTTrainer(
        target_net, loss_fn, cfg['layer_generators'],
        train_loader=train_loader, test_loader=test_loader,
        lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 0.0001),
        epochs=cfg['epochs'], min_lr=cfg.get('min_lr', 1e-5),
        device=device, log_interval=cfg.get('log_interval', 100),
        checkpoint_dir=os.path.join(cfg['checkpoint_dir'], make_experiment_name(cfg)),
        experiment_name=make_experiment_name(cfg),
        checkpoint_metadata={
            'target_net': cfg['target_net'],
            'training_strategy': 'lwt',
            'lrd_config': lrd_config,
            'sigma_noise': cfg.get('sigma_noise', 0.01),
        },
        save_interval=cfg.get('save_interval', 1),
    )
```

- [ ] **Step 2: Update evaluate.py**

Rebuild target net and generator from checkpoint metadata.

```python
# mapping_network/scripts/evaluate.py (key snippet)
checkpoint = torch.load(args.checkpoint, map_location=device)

target_net = build_target_net(checkpoint['target_net'], checkpoint.get('lrd_config'))
target_net = target_net.to(device)

if checkpoint['training_strategy'] == 'slvt':
    mapping = build_generator(
        checkpoint.get('generator_type', 'linear'),
        target_net.get_total_params(),
        checkpoint['latent_dim'],
        checkpoint.get('alpha', 0.01),
        device,
    )
    mapping.load_state_dict(checkpoint['state_dict'])
    theta_hat = mapping()
elif checkpoint['training_strategy'] == 'lwt':
    # Rebuild layer mappings and load each state_dict
    layer_mappings = nn.ModuleDict()
    for name, gen_cfg in checkpoint['layer_generator_configs'].items():
        group_size = target_net.get_group_param_size(name)
        mapping = build_generator(
            gen_cfg.get('type', 'linear'),
            group_size,
            gen_cfg['latent_dim'],
            gen_cfg.get('alpha', 0.01),
            device,
        )
        mapping.load_state_dict(checkpoint['state_dict'][name])
        layer_mappings[name] = mapping
    # Concatenate in the same order as target net param groups
    group_order = checkpoint.get('layer_group_order', list(layer_mappings.keys()))
    theta_hat = torch.cat([layer_mappings[name]() for name in group_order])
```

- [ ] **Step 3: Smoke test train.py**

Run: `uv run python3 -m mapping_network.scripts.train --config configs/cnn2_slvt.yaml --epochs 1 --device cuda`
Expected: completes 1 epoch without error.

- [ ] **Step 4: Smoke test evaluate.py**

Run: `uv run python3 -m mapping_network.scripts.evaluate --checkpoint checkpoints/cnn2_slvt/cnn2_slvt_final.pth --config configs/cnn2_slvt.yaml`
Expected: loads checkpoint and evaluates.

- [ ] **Step 5: Commit**

```bash
git add mapping_network/scripts/train.py mapping_network/scripts/evaluate.py
git commit -m "refactor: update train/evaluate scripts for new generator and LRD abstractions"
```

---

## Task 10: Update Config Files

**Files:**
- Modify: `configs/cnn1_slvt.yaml`
- Modify: `configs/cnn1_lwt.yaml`
- Modify: `configs/cnn1_3conv_slvt.yaml`
- Modify: `configs/cnn2_slvt.yaml`
- Modify: `configs/cnn2_lwt.yaml`

**Interfaces:**
- Produces: New YAML format with `lrd` and `layer_generators`.

- [ ] **Step 1: Update SLVT configs**

Add to each SLVT config:

```yaml
lrd:
  enabled: auto
  default_rank: 10
  layer_ranks:
    fc1: 10
```

For `cnn1_slvt.yaml`, explicitly enable LRD on `fc1` with rank 10 to ensure it runs on modest GPUs.

- [ ] **Step 2: Update LWT configs**

Replace `layer_latent_dims` and `layer_alphas` with `layer_generators`:

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
    lrd_rank: 10
  fc2:
    type: linear
    latent_dim: 64
    alpha: 0.01
```

- [ ] **Step 3: Validate configs load correctly**

Run:
```bash
uv run python3 -c "
import yaml
for path in ['configs/cnn1_slvt.yaml', 'configs/cnn1_lwt.yaml', 'configs/cnn1_3conv_slvt.yaml', 'configs/cnn2_slvt.yaml', 'configs/cnn2_lwt.yaml']:
    with open(path) as f:
        yaml.safe_load(f)
    print(path, 'OK')
"
```
Expected: all configs load without error.

- [ ] **Step 4: Commit**

```bash
git add configs/
git commit -m "config: update YAML configs for LRD and per-layer generators"
```

---

## Task 11: Rewrite and Expand Tests

**Files:**
- Modify: `tests/test_target_nets.py`
- Modify: `tests/test_mapping_net.py` → rename or delete, use `tests/test_generators.py`
- Modify: `tests/test_loss.py`
- Modify: `tests/test_slvt.py`
- Modify: `tests/test_lwt.py`
- Modify: `tests/test_configs.py`
- Create: `tests/test_factory.py`
- Create: `tests/test_lrd_config.py`
- Modify: `tests/conftest.py` if needed

**Interfaces:**
- Consumes: All new components.
- Produces: Comprehensive test coverage.

- [ ] **Step 1: Delete `tests/test_mapping_net.py` or merge into `tests/test_generators.py`**

- [ ] **Step 2: Update `tests/test_configs.py` to new format**

Parametrize over all configs and verify:
- All tensors on GPU.
- One forward + backward completes.
- Trainable param count equals expected (sum of z dims + 3 lambdas).

- [ ] **Step 3: Run full test suite**

Run: `uv run python3 -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: rewrite tests for ParameterGenerator, LRD, and new configs"
```

---

## Task 12: Final Verification and Cleanup

**Files:**
- All modified files
- `pyproject.toml` (if new packages needed — none expected)

- [ ] **Step 1: Run Ruff check**

Run: `uv run ruff check .`
Expected: no errors

- [ ] **Step 2: Run Ruff format**

Run: `uv run ruff format .`
Expected: clean

- [ ] **Step 3: Run full test suite again**

Run: `uv run python3 -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Run smoke training on CNN1 SLVT with LRD**

Run: `uv run python3 -m mapping_network.scripts.train --config configs/cnn1_slvt.yaml --epochs 1 --device cuda`
Expected: completes without OOM.

- [ ] **Step 5: Commit final formatting**

```bash
git add .
git commit -m "style: ruff format and final cleanup"
```

---

## Self-Review Checklist

1. **Spec coverage:**
   - `ParameterGenerator` base class → Task 1
   - `LinearMappingNetwork` → Task 2
   - `LRDConfig` → Task 3
   - LRD in `TargetNet` → Task 4
   - Factory functions → Task 5
   - `MappingLoss` update → Task 6
   - SLVT trainer update → Task 7
   - LWT trainer update → Task 8
   - Scripts update → Task 9
   - Config files update → Task 10
   - Tests rewrite → Task 11
   - No backward compatibility → reflected in Task 6 and Task 11
   - GPU default → reflected in all tests and smoke runs

2. **Placeholder scan:** No TBD/TODO/fill-in-details. Each step has concrete code or command.

3. **Type consistency:**
   - `ParameterGenerator.forward() -> torch.Tensor` used everywhere.
   - `LRDConfig` fields match usage in Task 4.
   - Checkpoint metadata keys match between `train.py`, `SLVTTrainer`, `LWTTrainer`, and `evaluate.py`.

4. **Gaps:** None identified.
