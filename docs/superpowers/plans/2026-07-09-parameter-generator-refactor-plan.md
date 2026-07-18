# ParameterGenerator Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `ParameterGenerator` to support standard `torch.nn.Module` inheritance, decouple trainers/evaluate/train script from `LinearMappingNetwork` internals, and add `MultiLayerLinearMappingNetwork` plus `CNNMappingNetwork` as extension proofs.

**Architecture:** Add `persistent_state_dict` / `load_persistent_state_dict` to the base class; change `factory.build_generator` to accept a `generator_config` dict; update trainers/scripts to use these generic interfaces; introduce two new flat-output generators that internally use `nn.Linear` / `nn.Conv2d`; add `TargetNet.assemble_params` to centralize LWT parameter concatenation.

**Tech Stack:** Python 3.13, PyTorch 2.11+, pytest, uv-managed environment at `/root/MyProj/MappingNetwork/.venv`.

## Global Constraints

- Work in the existing worktree `/root/MyProj/MappingNetwork/.claude/worktrees/issue12-refactor`.
- Run tests with the main project venv: `/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/... -v`.
- Do not reference `W_fixed`, `W_fixed_mean`, `b_fixed`, or `w_seed` outside `LinearMappingNetwork`.
- Old checkpoint format does not need to be supported.
- All new generators must output a 1-D `theta_hat` compatible with `target_net.functional_forward`.
- Follow existing ruff formatting; run `uv run ruff format .` if available, otherwise keep style consistent.

## File Structure

| File | Responsibility |
|------|----------------|
| `mapping_network/generators/base.py` | Base class with persistent state hooks |
| `mapping_network/generators/linear.py` | Existing linear generator; absorbs `w_seed` |
| `mapping_network/generators/multilayer_linear.py` | New MLP-style generator |
| `mapping_network/generators/cnn.py` | New Conv2d-based generator |
| `mapping_network/factory.py` | Generic `build_generator(generator_type, generator_config, device)` |
| `mapping_network/target_nets/base.py` | `assemble_params` helper for LWT |
| `mapping_network/trainer/slvt.py` | Save/load via persistent state hooks |
| `mapping_network/trainer/lwt.py` | Build generators via factory config; use `assemble_params` |
| `mapping_network/scripts/evaluate.py` | Rebuild generators via `generator_config`; use `assemble_params` |
| `mapping_network/scripts/train.py` | Build `generator_config` from YAML; remove `W_fixed` access |
| `tests/test_factory.py` | Update factory call signature |
| `tests/test_generators.py` | Add new generator unit tests |
| `tests/test_checkpoint.py` | Update checkpoint reconstruction tests |
| `tests/test_extensibility.py` | Integration test with `multilayer_linear` |

---

### Task 1: Base class persistent state + factory config dict

**Files:**
- Modify: `mapping_network/generators/base.py`
- Modify: `mapping_network/factory.py`
- Test: `tests/test_generators.py`, `tests/test_factory.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `ParameterGenerator.persistent_state_dict()`, `ParameterGenerator.load_persistent_state_dict(state)`, `build_generator(generator_type: str, generator_config: dict, device: str)`

- [ ] **Step 1: Write failing test for persistent state**

Append to `tests/test_generators.py`:

```python
def test_persistent_state_dict_only_trainable():
    gen = LinearMappingNetwork(20, 4, alpha=0.01, device='cpu')
    state = gen.persistent_state_dict()
    assert 'z' in state
    assert 'W_fixed' not in state
    assert all(v.requires_grad for v in state.values())


def test_load_persistent_state_dict_restores_trainable():
    gen = LinearMappingNetwork(20, 4, alpha=0.01, device='cpu')
    original_z = gen.z.detach().clone()
    gen.z.data.fill_(0.0)
    missing, unexpected = gen.load_persistent_state_dict({'z': original_z})
    assert torch.allclose(gen.z, original_z)
```

Append to `tests/test_factory.py`:

```python
def test_build_generator_with_config_dict(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    gen = build_generator('linear', {'target_total_params': 100, 'latent_dim': 8, 'alpha': 0.01}, device)
    assert isinstance(gen, LinearMappingNetwork)
    assert gen().shape == (100,)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generators.py::test_persistent_state_dict_only_trainable tests/test_generators.py::test_load_persistent_state_dict_restores_trainable tests/test_factory.py::test_build_generator_with_config_dict -v
```

Expected: FAIL (methods not defined / factory signature mismatch)

- [ ] **Step 3: Implement base class and factory changes**

Modify `mapping_network/generators/base.py`:

```python
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ParameterGenerator(nn.Module, ABC):
    """参数生成网络基类。负责生成 theta_hat 以及相关的辅助量。"""

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """返回 theta_hat [P']，P' 是目标网络压缩后的总参数数。"""
        pass

    @abstractmethod
    def noisy_forward(self, sigma: float) -> torch.Tensor:
        """对隐变量加高斯噪声后前向，返回 theta_noisy（用于 L_stab）。"""
        pass

    @abstractmethod
    def smooth_loss(self) -> torch.Tensor:
        """返回 L_smooth = ||nabla_z M(z)||^2_F / (P * d)。"""
        pass

    @abstractmethod
    def align_loss(self) -> torch.Tensor:
        """返回 L_align = 1 - cos(z, mean(W_mod, dim=0))。"""
        pass

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def persistent_state_dict(self) -> dict:
        """默认只保存可学习参数；固定 buffer 由 __init__ 重建。"""
        return {k: v for k, v in self.state_dict().items() if v.requires_grad}

    def load_persistent_state_dict(self, state: dict):
        """从 checkpoint 恢复可学习参数。"""
        missing, unexpected = self.load_state_dict(state, strict=False)
        return missing, unexpected
```

Modify `mapping_network/factory.py`:

```python
from mapping_network.generators.linear import LinearMappingNetwork
from mapping_network.target_nets.cnn1 import CNN1
from mapping_network.target_nets.cnn1_3conv import CNN1_3Conv
from mapping_network.target_nets.cnn2 import CNN2
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


def build_generator(generator_type: str, generator_config: dict, device: str):
    if generator_type not in GENERATOR_MAP:
        raise ValueError(f'Unknown generator type: {generator_type}')
    return GENERATOR_MAP[generator_type](**generator_config, device=device)
```

New generator map entries are added in Task 3 and Task 4.

- [ ] **Step 4: Run tests**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generators.py::test_persistent_state_dict_only_trainable tests/test_generators.py::test_load_persistent_state_dict_restores_trainable tests/test_factory.py::test_build_generator_with_config_dict -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mapping_network/generators/base.py mapping_network/factory.py tests/test_generators.py tests/test_factory.py
git commit -m "feat: add ParameterGenerator persistent state and factory config dict"
```

---

### Task 2: LinearMappingNetwork internalizes w_seed

**Files:**
- Modify: `mapping_network/generators/linear.py`
- Test: `tests/test_generators.py`

**Interfaces:**
- Consumes: `build_generator` now passes config dict
- Produces: `LinearMappingNetwork(..., w_seed=None)`

- [ ] **Step 1: Write failing test**

Append to `tests/test_generators.py`:

```python
def test_linear_mapping_network_w_seed_reproducible():
    gen1 = LinearMappingNetwork(20, 4, alpha=0.01, device='cpu', w_seed=123)
    gen2 = LinearMappingNetwork(20, 4, alpha=0.01, device='cpu', w_seed=123)
    assert torch.allclose(gen1.W_fixed, gen2.W_fixed)
    gen3 = LinearMappingNetwork(20, 4, alpha=0.01, device='cpu', w_seed=456)
    assert not torch.allclose(gen1.W_fixed, gen3.W_fixed)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generators.py::test_linear_mapping_network_w_seed_reproducible -v
```

Expected: FAIL (`w_seed` unexpected)

- [ ] **Step 3: Implement**

Modify `mapping_network/generators/linear.py` `__init__` signature and first lines:

```python
    def __init__(
        self,
        target_total_params: int,
        latent_dim: int,
        alpha: float = 0.01,
        device: str = 'cpu',
        w_seed: int | None = None,
    ):
        super().__init__()
        self.P = target_total_params
        self.d = latent_dim
        self.alpha = alpha

        if w_seed is not None:
            torch.manual_seed(w_seed)
        W = torch.empty(self.P, self.d, device=device)
        init.orthogonal_(W)
        self.register_buffer('W_fixed', W)
        self.register_buffer('W_fixed_mean', W.mean(dim=0))
        self.register_buffer('b_fixed', torch.zeros(self.P, device=device))
        self.z = nn.Parameter(torch.randn(self.d, device=device) * 0.1)
```

- [ ] **Step 4: Run test**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generators.py::test_linear_mapping_network_w_seed_reproducible -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mapping_network/generators/linear.py tests/test_generators.py
git commit -m "feat: internalize w_seed into LinearMappingNetwork"
```

---

### Task 3: MultiLayerLinearMappingNetwork

**Files:**
- Create: `mapping_network/generators/multilayer_linear.py`
- Test: `tests/test_generators.py`

**Interfaces:**
- Consumes: `ParameterGenerator` base
- Produces: `MultiLayerLinearMappingNetwork(target_total_params, latent_dim, alpha=0.01, hidden_dim=64, num_hidden=1, device='cpu')`

- [ ] **Step 1: Write failing test**

Append to `tests/test_generators.py`:

```python
from mapping_network.generators.multilayer_linear import MultiLayerLinearMappingNetwork


def test_multilayer_linear_mapping_network(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    gen = MultiLayerLinearMappingNetwork(50, 8, alpha=0.01, hidden_dim=16, num_hidden=2, device=device)
    theta = gen()
    assert theta.shape == (50,)
    assert theta.device.type == device
    assert gen.trainable_params() > 0

    theta_noisy = gen.noisy_forward(0.01)
    assert theta_noisy.shape == (50,)
    assert theta_noisy.requires_grad

    l_smooth = gen.smooth_loss()
    assert l_smooth.shape == ()
    assert l_smooth.requires_grad

    l_align = gen.align_loss()
    assert l_align.shape == ()
    assert l_align.requires_grad
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generators.py::test_multilayer_linear_mapping_network -v
```

Expected: FAIL (module missing)

- [ ] **Step 3: Implement**

Create `mapping_network/generators/multilayer_linear.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ParameterGenerator


class MultiLayerLinearMappingNetwork(ParameterGenerator):
    """MLP-style generator: z -> Linear -> ReLU -> ... -> Linear -> theta_hat."""

    def __init__(
        self,
        target_total_params: int,
        latent_dim: int,
        alpha: float = 0.01,
        hidden_dim: int = 64,
        num_hidden: int = 1,
        device: str = 'cpu',
    ):
        super().__init__()
        self.P = target_total_params
        self.d = latent_dim
        self.alpha = alpha

        self.z = nn.Parameter(torch.randn(self.d, device=device) * 0.1)

        layers = []
        in_dim = latent_dim
        for _ in range(num_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim, device=device))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, target_total_params, device=device))
        self.net = nn.Sequential(*layers)

    def _forward_z(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(z))

    def forward(self) -> torch.Tensor:
        return self._forward_z(self.z)

    def noisy_forward(self, sigma: float) -> torch.Tensor:
        eps = torch.randn_like(self.z) * sigma
        return self._forward_z(self.z + eps)

    def smooth_loss(self, chunk_size: int = 1024) -> torch.Tensor:
        theta = self.forward()
        P = theta.numel()
        total = torch.zeros((), device=theta.device, dtype=theta.dtype)
        for start in range(0, P, chunk_size):
            end = min(start + chunk_size, P)
            mask = torch.zeros(P, device=theta.device, dtype=theta.dtype)
            mask[start:end] = 1.0
            grads, = torch.autograd.grad(
                theta, self.z, grad_outputs=mask, retain_graph=True, create_graph=True
            )
            total = total + (grads * grads).sum()
        return total / (P * self.d)

    def align_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.z.device, requires_grad=True)

    def extra_repr(self):
        return f'P={self.P}, d={self.d}, alpha={self.alpha}, hidden_dim={self.net[0].out_features}'
```

- [ ] **Step 4: Run test**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generators.py::test_multilayer_linear_mapping_network -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mapping_network/generators/multilayer_linear.py tests/test_generators.py
git commit -m "feat: add MultiLayerLinearMappingNetwork generator"
```

Also register it in `mapping_network/factory.py`:

```python
from mapping_network.generators.multilayer_linear import MultiLayerLinearMappingNetwork

GENERATOR_MAP = {
    'linear': LinearMappingNetwork,
    'multilayer_linear': MultiLayerLinearMappingNetwork,
}
```

---

### Task 4: CNNMappingNetwork

**Files:**
- Create: `mapping_network/generators/cnn.py`
- Test: `tests/test_generators.py`

**Interfaces:**
- Consumes: `ParameterGenerator` base
- Produces: `CNNMappingNetwork(target_total_params, latent_dim, alpha=0.01, feature_size=4, channels=(16, 8), device='cpu')`

- [ ] **Step 1: Write failing test**

Append to `tests/test_generators.py`:

```python
from mapping_network.generators.cnn import CNNMappingNetwork


def test_cnn_mapping_network(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    gen = CNNMappingNetwork(50, 8, alpha=0.01, feature_size=4, channels=(8, 4), device=device)
    theta = gen()
    assert theta.shape == (50,)
    assert theta.device.type == device
    assert gen.trainable_params() > 0

    theta_noisy = gen.noisy_forward(0.01)
    assert theta_noisy.shape == (50,)
    assert theta_noisy.requires_grad

    l_smooth = gen.smooth_loss()
    assert l_smooth.shape == ()
    assert l_smooth.requires_grad

    l_align = gen.align_loss()
    assert l_align.shape == ()
    assert l_align.requires_grad
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generators.py::test_cnn_mapping_network -v
```

Expected: FAIL (module missing)

- [ ] **Step 3: Implement**

Create `mapping_network/generators/cnn.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ParameterGenerator


class CNNMappingNetwork(ParameterGenerator):
    """Conv2d-based generator: project z to a small feature map, convolve, flatten, project."""

    def __init__(
        self,
        target_total_params: int,
        latent_dim: int,
        alpha: float = 0.01,
        feature_size: int = 4,
        channels: tuple[int, ...] = (16, 8),
        device: str = 'cpu',
    ):
        super().__init__()
        self.P = target_total_params
        self.d = latent_dim
        self.alpha = alpha
        self.feature_size = feature_size
        self.channels = channels

        self.z = nn.Parameter(torch.randn(self.d, device=device) * 0.1)

        first_ch = channels[0]
        self.fc = nn.Linear(latent_dim, first_ch * feature_size * feature_size, device=device)

        conv_layers = []
        in_ch = first_ch
        for out_ch in channels[1:]:
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, device=device))
            conv_layers.append(nn.ReLU())
            in_ch = out_ch
        self.conv_net = nn.Sequential(*conv_layers)

        self.final = nn.Linear(in_ch * feature_size * feature_size, target_total_params, device=device)

    def _features(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc(z))
        x = x.view(-1, self.channels[0], self.feature_size, self.feature_size)
        x = self.conv_net(x)
        return x.view(1, -1)

    def _forward_z(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.final(self._features(z)).squeeze(0))

    def forward(self) -> torch.Tensor:
        return self._forward_z(self.z)

    def noisy_forward(self, sigma: float) -> torch.Tensor:
        eps = torch.randn_like(self.z) * sigma
        return self._forward_z(self.z + eps)

    def smooth_loss(self, chunk_size: int = 1024) -> torch.Tensor:
        theta = self.forward()
        P = theta.numel()
        total = torch.zeros((), device=theta.device, dtype=theta.dtype)
        for start in range(0, P, chunk_size):
            end = min(start + chunk_size, P)
            mask = torch.zeros(P, device=theta.device, dtype=theta.dtype)
            mask[start:end] = 1.0
            grads, = torch.autograd.grad(
                theta, self.z, grad_outputs=mask, retain_graph=True, create_graph=True
            )
            total = total + (grads * grads).sum()
        return total / (P * self.d)

    def align_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.z.device, requires_grad=True)

    def extra_repr(self):
        return f'P={self.P}, d={self.d}, alpha={self.alpha}, channels={self.channels}'
```

- [ ] **Step 4: Run test**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_generators.py::test_cnn_mapping_network -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mapping_network/generators/cnn.py tests/test_generators.py
git commit -m "feat: add CNNMappingNetwork generator"
```

Also register it in `mapping_network/factory.py`:

```python
from mapping_network.generators.cnn import CNNMappingNetwork

GENERATOR_MAP = {
    'linear': LinearMappingNetwork,
    'multilayer_linear': MultiLayerLinearMappingNetwork,
    'cnn': CNNMappingNetwork,
}
```

---

### Task 5: TargetNet.assemble_params

**Files:**
- Modify: `mapping_network/target_nets/base.py`
- Test: `tests/test_target_nets.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `TargetNet.assemble_params(group_outputs)`

- [ ] **Step 1: Write failing test**

Append to `tests/test_target_nets.py`:

```python
def test_target_net_assemble_params():
    net = build_target_net('cnn2')
    group_names = net.get_group_names()
    group_sizes = [net.get_group_param_size(name) for name in group_names]
    outputs = [torch.randn(size) for size in group_sizes]
    theta = net.assemble_params(outputs)
    assert theta.shape == (sum(group_sizes),)

    # dict input
    theta2 = net.assemble_params({name: out for name, out in zip(group_names, outputs)})
    assert torch.allclose(theta, theta2)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_target_nets.py::test_target_net_assemble_params -v
```

Expected: FAIL (`assemble_params` missing)

- [ ] **Step 3: Implement**

Ensure `mapping_network/target_nets/base.py` imports `torch` at the top (add `import torch` if missing). Then add to `TargetNet`:

```python
    def assemble_params(self, group_outputs: list[torch.Tensor] | dict[str, torch.Tensor]) -> torch.Tensor:
        """按 group_order 拼接每层的输出得到完整 theta_hat。"""
        if isinstance(group_outputs, dict):
            outputs = [group_outputs[name] for name in self.get_group_names()]
        else:
            outputs = group_outputs
        return torch.cat(outputs)
```

- [ ] **Step 4: Run test**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_target_nets.py::test_target_net_assemble_params -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mapping_network/target_nets/base.py tests/test_target_nets.py
git commit -m "feat: add TargetNet.assemble_params for LWT"
```

---

### Task 6: SLVTTrainer checkpoint persistence

**Files:**
- Modify: `mapping_network/trainer/slvt.py`
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: `mapping.persistent_state_dict()`, `mapping.load_persistent_state_dict(state)`
- Produces: checkpoint with `'generator_state_dict'` and `'generator_config'`

- [ ] **Step 1: Update test to expect new checkpoint keys**

In `tests/test_checkpoint.py`, update `test_slvt_checkpoint_reconstruction`:

- Change `build_generator` call in reconstruction to:

```python
    mapping_rebuilt = build_generator(
        ckpt.get('generator_type', 'linear'),
        ckpt['generator_config'],
        device,
    )
    mapping_rebuilt.load_persistent_state_dict(ckpt['generator_state_dict'])
```

- Keep the rest unchanged.

- [ ] **Step 2: Run test to verify it fails**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_checkpoint.py::test_slvt_checkpoint_reconstruction -v
```

Expected: FAIL (`generator_config` / `generator_state_dict` missing)

- [ ] **Step 3: Implement**

Modify `mapping_network/trainer/slvt.py`:

In `save_checkpoint`, replace `'state_dict': self.mapping_net.state_dict()` with:

```python
            'generator_config': self.checkpoint_metadata.get('generator_config'),
            'generator_state_dict': self.mapping_net.persistent_state_dict(),
```

In `load_checkpoint`, replace `self.mapping_net.load_state_dict(checkpoint['state_dict'])` with:

```python
        self.mapping_net.load_persistent_state_dict(checkpoint['generator_state_dict'])
```

- [ ] **Step 4: Run test**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_checkpoint.py::test_slvt_checkpoint_reconstruction -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mapping_network/trainer/slvt.py tests/test_checkpoint.py
git commit -m "refactor: SLVTTrainer uses persistent state hooks"
```

---

### Task 7: LWTTrainer uses factory config and assemble_params

**Files:**
- Modify: `mapping_network/trainer/lwt.py`
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: `build_generator(type, config, device)`, `target_net.assemble_params`
- Produces: LWT checkpoint with per-layer `generator_state_dict`

- [ ] **Step 1: Update checkpoint test**

In `tests/test_checkpoint.py` `test_lwt_checkpoint_reconstruction`, change reconstruction loop to:

```python
    for name, gen_cfg in ckpt['layer_generator_configs'].items():
        group_size = target_rebuilt.get_group_param_size(name)
        gen_type = gen_cfg.get('type', 'linear')
        config = {k: v for k, v in gen_cfg.items() if k != 'type'}
        config['target_total_params'] = group_size
        mapping = build_generator(gen_type, config, device)
        mapping.load_persistent_state_dict(ckpt['state_dict'][name])
        layer_mappings[name] = mapping
```

Also replace `theta_rebuilt = torch.cat(...)` with:

```python
    theta_rebuilt = target_rebuilt.assemble_params({name: layer_mappings[name]() for name in group_order})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_checkpoint.py::test_lwt_checkpoint_reconstruction -v
```

Expected: FAIL (factory signature mismatch / `assemble_params` not used in trainer)

- [ ] **Step 3: Implement**

Modify `mapping_network/trainer/lwt.py`:

1. In `__init__`, replace generator construction with:

```python
        for group_name, group_size in self.param_groups:
            if group_name not in layer_generators:
                raise ValueError(f'Missing generator config for layer group: {group_name}')
            config = layer_generators[group_name].copy()
            gen_type = config.pop('type')
            config['target_total_params'] = group_size
            self.layer_mappings[group_name] = build_generator(gen_type, config, device)
```

2. Replace `_generate_all_theta` with:

```python
    def _generate_all_theta(self):
        outputs = [self.layer_mappings[name]() for name, _ in self.param_groups]
        return self.target_net.assemble_params(outputs)
```

3. In `save_checkpoint`, replace `'state_dict': {name: mapping.state_dict() ...}` with:

```python
            'state_dict': {
                name: mapping.persistent_state_dict() for name, mapping in self.layer_mappings.items()
            },
```

4. In `load_checkpoint`, replace mapping.load_state_dict with:

```python
            self.layer_mappings[name].load_persistent_state_dict(state)
```

- [ ] **Step 4: Run test**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_checkpoint.py::test_lwt_checkpoint_reconstruction -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mapping_network/trainer/lwt.py tests/test_checkpoint.py
git commit -m "refactor: LWTTrainer builds generators via config dict and uses assemble_params"
```

---

### Task 8: evaluate.py rebuilds generators generically

**Files:**
- Modify: `mapping_network/scripts/evaluate.py`

**Interfaces:**
- Consumes: `build_generator`, `load_persistent_state_dict`, `assemble_params`
- Produces: no new public API

- [ ] **Step 1: Manual smoke test**

Train a tiny SLVT checkpoint first (from Task 10 or a quick manual run). For now, verify by running existing checkpoint tests; evaluate script does not have direct unit tests.

- [ ] **Step 2: Implement**

Modify `mapping_network/scripts/evaluate.py`:

SLVT branch:

```python
    if checkpoint['training_strategy'] == 'slvt':
        mapping = build_generator(
            checkpoint.get('generator_type', 'linear'),
            checkpoint['generator_config'],
            device,
        )
        mapping.load_persistent_state_dict(checkpoint['generator_state_dict'])
        theta_hat = mapping()
```

LWT branch:

```python
    elif checkpoint['training_strategy'] == 'lwt':
        layer_mappings = nn.ModuleDict()
        for name, gen_cfg in checkpoint['layer_generator_configs'].items():
            group_size = target_net.get_group_param_size(name)
            gen_type = gen_cfg.get('type', 'linear')
            config = {k: v for k, v in gen_cfg.items() if k != 'type'}
            config['target_total_params'] = group_size
            mapping = build_generator(gen_type, config, device)
            mapping.load_persistent_state_dict(checkpoint['state_dict'][name])
            layer_mappings[name] = mapping
        group_order = checkpoint.get('layer_group_order', list(layer_mappings.keys()))
        theta_hat = target_net.assemble_params({name: layer_mappings[name]() for name in group_order})
```

- [ ] **Step 3: Verify with a real checkpoint**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m mapping_network.scripts.train --config configs/cnn2_slvt.yaml --device cpu --epochs 1
/root/MyProj/MappingNetwork/.venv/bin/python -m mapping_network.scripts.evaluate --checkpoint checkpoints/cnn2_slvt/cnn2_slvt_final.pth --config configs/cnn2_slvt.yaml --device cpu
```

Expected: evaluation prints accuracy.

- [ ] **Step 4: Commit**

```bash
git add mapping_network/scripts/evaluate.py
git commit -m "refactor: evaluate.py uses generic generator config and assemble_params"
```

---

### Task 9: train.py builds generator_config and removes W_fixed access

**Files:**
- Modify: `mapping_network/scripts/train.py`

**Interfaces:**
- Consumes: `build_generator` with config dict
- Produces: `checkpoint_metadata['generator_config']`

- [ ] **Step 1: Implement**

Modify `mapping_network/scripts/train.py` SLVT branch:

Replace:

```python
        mapping = build_generator(
            cfg.get('generator_type', 'linear'),
            target_net.get_total_params(),
            cfg['latent_dim'],
            cfg.get('alpha', 0.01),
            device,
        )
        print(f'Latent dim: {cfg["latent_dim"]}')
        print(f'Trainable: {mapping.trainable_params():,}')
        print(f'Fixed mapping weights: {mapping.W_fixed.numel():,}')
```

with:

```python
        generator_config = cfg.get('generator_config') or {
            'target_total_params': target_net.get_total_params(),
            'latent_dim': cfg['latent_dim'],
            'alpha': cfg.get('alpha', 0.01),
        }
        if 'w_seed' in cfg:
            generator_config['w_seed'] = cfg['w_seed']
        mapping = build_generator(
            cfg.get('generator_type', 'linear'),
            generator_config,
            device,
        )
        total_params = sum(p.numel() for p in mapping.parameters()) + sum(
            b.numel() for b in mapping.buffers()
        )
        print(f'Latent dim: {generator_config["latent_dim"]}')
        print(f'Trainable: {mapping.trainable_params():,}')
        print(f'Total mapping params (fixed+trainable): {total_params:,}')
```

And in `checkpoint_metadata` replace `latent_dim`, `alpha`, `generator_type` with:

```python
                'generator_type': cfg.get('generator_type', 'linear'),
                'generator_config': generator_config,
```

- [ ] **Step 2: Verify training still works**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m mapping_network.scripts.train --config configs/cnn2_slvt.yaml --device cpu --epochs 1
```

Expected: completes without error.

- [ ] **Step 3: Commit**

```bash
git add mapping_network/scripts/train.py
git commit -m "refactor: train.py builds generator_config and avoids W_fixed access"
```

---

### Task 10: Extensibility integration test

**Files:**
- Create: `tests/test_extensibility.py`

**Interfaces:**
- Consumes: `build_generator('multilayer_linear', ...)`, trainers, evaluate-like flow
- Produces: integration test proving trainer/evaluate are generator-agnostic

- [ ] **Step 1: Write test**

Create `tests/test_extensibility.py`:

```python
"""Integration test: trainer and evaluate work with a non-Linear generator."""

import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from mapping_network.factory import build_generator, build_target_net
from mapping_network.mapping.loss import MappingLoss
from mapping_network.scripts.evaluate import evaluate_model
from mapping_network.trainer.lwt import LWTTrainer
from mapping_network.trainer.slvt import SLVTTrainer


def make_one_batch_loader(device):
    x = torch.randn(2, 1, 28, 28, device=device)
    y = torch.tensor([0, 1], device=device)
    return DataLoader(TensorDataset(x.cpu(), y.cpu()), batch_size=2)


def test_slvt_with_multilayer_linear(device='cpu'):
    target_net = build_target_net('cnn2').to(device)
    loss_fn = MappingLoss(sigma_noise=0.01).to(device)
    loader = make_one_batch_loader(device)

    generator_config = {
        'target_total_params': target_net.get_total_params(),
        'latent_dim': 32,
        'alpha': 0.01,
        'hidden_dim': 16,
        'num_hidden': 1,
    }
    mapping = build_generator('multilayer_linear', generator_config, device)

    trainer = SLVTTrainer(
        mapping,
        target_net,
        loss_fn,
        loader,
        loader,
        epochs=1,
        device=device,
        log_interval=1,
        checkpoint_dir='/tmp/test_ext_slvt',
        experiment_name='test_ext_slvt',
        checkpoint_metadata={
            'target_net': 'cnn2',
            'training_strategy': 'slvt',
            'generator_type': 'multilayer_linear',
            'generator_config': generator_config,
            'sigma_noise': 0.01,
            'lrd_config': None,
        },
        save_interval=0,
    )
    trainer.train()

    # Evaluate from checkpoint
    ckpt = torch.load('/tmp/test_ext_slvt/test_ext_slvt_final.pth', map_location=device)
    target_rebuilt = build_target_net(ckpt['target_net'], ckpt.get('lrd_config')).to(device)
    mapping_rebuilt = build_generator(ckpt['generator_type'], ckpt['generator_config'], device)
    mapping_rebuilt.load_persistent_state_dict(ckpt['generator_state_dict'])
    theta_hat = mapping_rebuilt()
    acc = evaluate_model(target_rebuilt, theta_hat, loader, device)
    assert 0 <= acc <= 100


def test_lwt_with_multilayer_linear(device='cpu'):
    target_net = build_target_net('cnn2').to(device)
    loss_fn = MappingLoss(sigma_noise=0.01).to(device)
    loader = make_one_batch_loader(device)
    layer_generators = {
        name: {'type': 'multilayer_linear', 'latent_dim': 8, 'alpha': 0.01, 'hidden_dim': 8}
        for name in target_net.get_group_names()
    }

    trainer = LWTTrainer(
        target_net,
        loss_fn,
        layer_generators,
        train_loader=loader,
        test_loader=loader,
        epochs=1,
        device=device,
        log_interval=1,
        checkpoint_dir='/tmp/test_ext_lwt',
        experiment_name='test_ext_lwt',
        checkpoint_metadata={
            'target_net': 'cnn2',
            'training_strategy': 'lwt',
            'lrd_config': None,
            'sigma_noise': 0.01,
        },
        save_interval=0,
    )
    trainer.train()

    ckpt = torch.load('/tmp/test_ext_lwt/test_ext_lwt_final.pth', map_location=device)
    target_rebuilt = build_target_net(ckpt['target_net'], ckpt.get('lrd_config')).to(device)
    layer_mappings = torch.nn.ModuleDict()
    for name, gen_cfg in ckpt['layer_generator_configs'].items():
        group_size = target_rebuilt.get_group_param_size(name)
        gen_type = gen_cfg.get('type', 'linear')
        config = {k: v for k, v in gen_cfg.items() if k != 'type'}
        config['target_total_params'] = group_size
        mapping = build_generator(gen_type, config, device)
        mapping.load_persistent_state_dict(ckpt['state_dict'][name])
        layer_mappings[name] = mapping
    group_order = ckpt.get('layer_group_order', list(layer_mappings.keys()))
    theta_hat = target_rebuilt.assemble_params({name: layer_mappings[name]() for name in group_order})
    acc = evaluate_model(target_rebuilt, theta_hat, loader, device)
    assert 0 <= acc <= 100
```

- [ ] **Step 2: Run test**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/test_extensibility.py -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_extensibility.py
git commit -m "test: add extensibility integration test with multilayer_linear"
```

---

### Task 11: Full test suite and final cleanup

- [ ] **Step 1: Run all tests**

```bash
/root/MyProj/MappingNetwork/.venv/bin/python -m pytest tests/ -v --device cpu
```

Expected: all tests pass.

- [ ] **Step 2: Run ruff format / check**

```bash
cd /root/MyProj/MappingNetwork/.claude/worktrees/issue12-refactor
uv run ruff format .
uv run ruff check .
```

Fix any issues.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "style: apply ruff formatting"
```

- [ ] **Step 4: Report**

Summarize changed files, test count, and any known limitations.

---

## Self-Review

1. **Spec coverage:**
   - Base persistent state: Task 1
   - Factory config dict: Task 1
   - `w_seed` internalized: Task 2
   - New MLP generator: Task 3
   - New CNN generator: Task 4
   - `TargetNet.assemble_params`: Task 5
   - Trainer/evaluate/train decoupling: Tasks 6-9
   - Extensibility test: Task 10
2. **Placeholder scan:** No TBD/TODO; all code blocks contain real code.
3. **Type consistency:** `build_generator` signature is `(str, dict, str)` everywhere; checkpoint keys `generator_config`/`generator_state_dict` used consistently; `assemble_params` accepts list or dict.
