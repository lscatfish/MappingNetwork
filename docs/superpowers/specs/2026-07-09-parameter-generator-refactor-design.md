# ParameterGenerator 标准 nn.Module 继承与解耦设计文档

## 背景

当前 `ParameterGenerator` 的实现把 `LinearMappingNetwork` 的私有细节（`W_fixed`、`W_fixed_mean`、`b_fixed`、`w_seed`）泄漏到了 `factory.py`、trainer、`scripts/evaluate.py`、`scripts/train.py` 以及测试文件中。Issue #12 要求把基类改成标准 `torch.nn.Module` 式继承，使未来新增 CNN、MLP 等 generator 时不需要修改训练/评估逻辑。

## 目标

1. `ParameterGenerator` 支持标准 `nn.Module` 继承。
2. 子类在 `__init__` 中自由定义 `nn.Linear` / `nn.Conv2d` 等结构；固定参数通过 `requires_grad=False` 或 `register_buffer` 管理。
3. `factory.build_generator` 接收 `generator_config: dict`。
4. trainer / evaluate / train script 不再硬编码 `W_fixed`、`W_fixed_mean`、`b_fixed`、`w_seed`。
5. checkpoint 的保存/恢复由 generator 自己控制。
6. 新增 `MultiLayerLinearMappingNetwork` 与 `CNNMappingNetwork` 作为扩展性验证，并补充集成测试。

## 非目标

- 不兼容旧格式 checkpoint（已确认不需要）。
- 本次不引入真实的大型 CNN hypernetwork，只提供结构示例与可运行测试。

## 设计

### 1. 基类接口

文件：`mapping_network/generators/base.py`

保留：

- `forward() -> torch.Tensor`
- `noisy_forward(sigma: float) -> torch.Tensor`
- `smooth_loss() -> torch.Tensor`
- `align_loss() -> torch.Tensor`
- `trainable_params() -> int`

新增：

```python
def persistent_state_dict(self) -> dict:
    """默认保存所有 requires_grad=True 的参数。"""
    return {k: v for k, v in self.state_dict().items() if v.requires_grad}

def load_persistent_state_dict(self, state: dict):
    """默认使用 strict=False 加载可学习参数；固定 buffer 由 __init__ 重建。"""
    missing, unexpected = self.load_state_dict(state, strict=False)
    return missing, unexpected
```

### 2. Factory

文件：`mapping_network/factory.py`

```python
def build_generator(generator_type: str, generator_config: dict, device: str):
    if generator_type not in GENERATOR_MAP:
        raise ValueError(f'Unknown generator type: {generator_type}')
    return GENERATOR_MAP[generator_type](**generator_config, device=device)
```

`GENERATOR_MAP` 新增 `multilayer_linear` 与 `cnn`。

### 3. LinearMappingNetwork

文件：`mapping_network/generators/linear.py`

- `__init__` 增加可选 `w_seed: int | None = None`，在内部生成固定正交矩阵。
- 保留 `W_fixed`、`W_fixed_mean`、`b_fixed`、`z` 作为实现细节，外部不再直接引用。
- `smooth_loss` 保持现有高效实现。

### 4. 新增 generator

#### MultiLayerLinearMappingNetwork

文件：`mapping_network/generators/multilayer_linear.py`

结构：`z -> Linear(hidden_dim) -> ReLU -> ... -> Linear(target_total_params)`，最终经 `tanh`。

#### CNNMappingNetwork

文件：`mapping_network/generators/cnn.py`

结构：把 `z` 投影为小特征图，经若干 `Conv2d` 层后展平、再经 `Linear` 投影到 `target_total_params`，最终经 `tanh`。

两者都输出与 `target_net.functional_forward` 兼容的一维 `theta_hat`，因此目标网络与训练循环无需改动。

### 5. TargetNet 增加 assemble_params

文件：`mapping_network/target_nets/base.py`

```python
def assemble_params(self, group_outputs: list[Tensor] | dict[str, Tensor]) -> Tensor:
    if isinstance(group_outputs, dict):
        outputs = [group_outputs[name] for name in self.get_group_names()]
    else:
        outputs = group_outputs
    return torch.cat(outputs)
```

仅用于把 LWT 的每层输出按 `group_order` 拼接，替代 `evaluate.py` 与 `LWTTrainer` 中的直接 `torch.cat`。

### 6. trainer / evaluate / train script

#### SLVTTrainer

- `save_checkpoint` 保存 `mapping.persistent_state_dict()`。
- `load_checkpoint` 调用 `mapping.load_persistent_state_dict(...)`。
- checkpoint 中存储 `generator_config`。

#### LWTTrainer

- 构建 generator 时使用 `build_generator(type, {target_total_params: group_size, ...}, device)`。
- `_generate_all_theta` 改为调用 `self.target_net.assemble_params([...])`。
- save/load 使用 `persistent_state_dict` / `load_persistent_state_dict`。

#### scripts/evaluate.py

- SLVT：从 checkpoint 读取 `generator_config`，`build_generator` 后调用 `load_persistent_state_dict`。
- LWT：使用 `target_net.assemble_params({name: mapping() for name, mapping in layer_mappings.items()})`。

#### scripts/train.py

- SLVT：从配置顶层字段推导 `generator_config`，不再访问 `mapping.W_fixed.numel()`。
- 打印 `trainable_params` 与总参数估算。

### 7. 测试

- 更新 `tests/test_factory.py` 使用新的 `build_generator` 签名。
- 更新 `tests/test_checkpoint.py` 使用 `persistent_state_dict` / `generator_config`。
- 在 `tests/test_generators.py` 中新增 `MultiLayerLinearMappingNetwork` 与 `CNNMappingNetwork` 的形状、设备、可训练参数、辅助损失测试。
- 新增 `tests/test_extensibility.py`：用 `multilayer_linear` 作为 generator 跑通 SLVT 与 LWT 的 train -> save -> evaluate -> resume，验证 trainer/evaluate 不依赖具体 generator 内部。

## 风险与回退

- `persistent_state_dict` 默认只保存可学习参数。如果未来某个 generator 的固定 buffer 也需要保存（例如太大无法重建），该 generator 可覆盖此方法。
- 通用 `smooth_loss` 使用 `autograd.grad` 分块计算，对于大 `target_total_params` 较慢；生产 generator 应像 `LinearMappingNetwork` 一样提供自定义实现。
- 旧 checkpoint 不再加载，用户需要重新训练。

## 验收标准

- [ ] `ParameterGenerator` 新增 `persistent_state_dict` / `load_persistent_state_dict`。
- [ ] `factory.build_generator` 接收 `generator_config: dict`。
- [ ] trainer / evaluate / train script 不出现 `W_fixed`、`W_fixed_mean`、`b_fixed`、`w_seed`。
- [ ] `MultiLayerLinearMappingNetwork` 与 `CNNMappingNetwork` 实现并通过单元测试。
- [ ] 新增扩展性集成测试，验证 trainer/evaluate 对 mock generator 无硬编码依赖。
- [ ] 所有现有测试通过。
