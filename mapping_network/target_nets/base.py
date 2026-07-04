import torch
import torch.nn as nn


class TargetNet(nn.Module):
    """
    目标网络基类。

    提供两套前向接口：
    - forward(x): 标准模块前向（用于基线训练）
    - functional_forward(x, theta_hat): 函数式前向（用于 Mapping Network），
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
