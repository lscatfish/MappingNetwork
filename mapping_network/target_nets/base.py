from dataclasses import dataclass

import torch
import torch.nn as nn

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

    def __init__(self, lrd_config: LRDConfig | dict | None = None):
        super().__init__()
        if isinstance(lrd_config, dict):
            lrd_config = LRDConfig(**lrd_config)
        self._lrd_config = lrd_config if lrd_config is not None else LRDConfig()
        self._param_slices = []

    def _should_use_lrd(self, layer_name: str, total_params: int) -> bool:
        layer_enabled = self._lrd_config.layer_enabled.get(layer_name)
        if layer_enabled is not None:
            if layer_enabled is True:
                return True
            if layer_enabled is False:
                return False
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

            bias_name = name.replace('.weight', '.bias') if name.endswith('.weight') else name
            bias_param = params_dict.get(bias_name)
            bias_shape = bias_param.shape if bias_param is not None else (shape[0],)
            bias_numel = bias_param.numel() if bias_param is not None else shape[0]

            module = self.get_submodule(base)
            if (
                not is_bias
                and isinstance(module, nn.Linear)
                and self._should_use_lrd(base, total_params)
            ):
                m, n = shape
                rank = self._lrd_config.layer_ranks.get(base, self._lrd_config.default_rank)
                rank = min(rank, m, n)

                u_start, u_end = idx, idx + m * rank
                v_start, v_end = u_end, u_end + n * rank
                b_start, b_end = v_end, v_end + bias_numel

                self._param_slices.append(
                    _ParamSlice(
                        kind='lrd',
                        weight_name=name,
                        bias_name=bias_name,
                        u_start=u_start,
                        u_end=u_end,
                        u_shape=(m, rank),
                        v_start=v_start,
                        v_end=v_end,
                        v_shape=(n, rank),
                        b_start=b_start,
                        b_end=b_end,
                        b_shape=bias_shape,
                    )
                )
                processed_bias.add(bias_name)
                idx = b_end
            else:
                self._param_slices.append(
                    _ParamSlice(
                        kind='full',
                        start=idx,
                        end=idx + numel,
                        shape=shape,
                        name=name,
                        is_bias=is_bias,
                    )
                )
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

    def assemble_params(self, group_outputs: list[torch.Tensor] | dict[str, torch.Tensor]) -> torch.Tensor:
        """按 group_order 拼接每层的输出得到完整 theta_hat。"""
        if isinstance(group_outputs, dict):
            outputs = [group_outputs[name] for name in self.get_group_names()]
        else:
            outputs = group_outputs
        return torch.cat(outputs)

    def functional_forward(self, x, theta_hat):
        params = {}
        for s in self._param_slices:
            if s.kind == 'full':
                params[s.name] = theta_hat[s.start : s.end].reshape(s.shape)
            elif s.kind == 'lrd':
                U = theta_hat[s.u_start : s.u_end].reshape(s.u_shape)
                V = theta_hat[s.v_start : s.v_end].reshape(s.v_shape)
                params[s.weight_name] = U @ V.T
                params[s.bias_name] = theta_hat[s.b_start : s.b_end].reshape(s.b_shape)
        return self._functional_forward(x, params)

    def _functional_forward(self, x, params):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
