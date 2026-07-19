"""Mapping 框架基类：Generator 和 MappingLayer。"""

from abc import ABC, abstractmethod
from functools import reduce
from operator import mul

import torch
import torch.nn as nn


def _prod(iterable) -> int:
    return reduce(mul, iterable, 1)


class Generator(nn.Module, ABC):
    """参数生成网络基类。

    基类自动从 param_spec 派生便利属性，用户无需手动处理 param_spec 字典。

    Args:
        param_spec: 目标参数规格，由 MappingLayer 自动传入。
            格式: {'weight': (C_out, C_in, kh, kw), 'bias': (C_out,)}
            当 bias=False 时，不含 'bias' 键。
        z_dim: 隐变量 z 的维度，必须显式声明。
        **kwargs: 用户自定义参数（如隐藏层大小等）。

    自动派生属性:
        self.w_shape  (tuple):  weight 目标形状
        self.b_shape  (tuple | None): bias 目标形状，或 None
        self.w_size   (int):   weight 总元素数
        self.b_size   (int):   bias 总元素数，或 0
    """

    def __init__(self, param_spec: dict, z_dim: int, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.z = nn.Parameter(torch.randn(z_dim))

        self.w_shape = param_spec['weight']
        self.b_shape = param_spec.get('bias')
        self.w_size = _prod(self.w_shape)
        self.b_size = _prod(self.b_shape) if self.b_shape else 0

    @abstractmethod
    def forward(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """返回生成的参数张量。

        Returns:
            tuple: (weight, bias)
                - weight: 形状为 self.w_shape 的张量，或 1D flat
                - bias:   形状为 self.b_shape 的张量，或 1D flat（bias=False 时为 None）
        """
        raise NotImplementedError
