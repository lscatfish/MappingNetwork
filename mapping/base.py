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


class MappingLayer(nn.Module):
    """主干网络层基类。

    子类需实现:
        - _functional(x, w, b) -> Tensor: 用参数执行函数式前向
    """

    def _set_generator(
        self,
        generator_cls: type[Generator] | None,
        generator_instance: Generator | None,
        generator_kwargs: dict,
    ) -> None:
        """根据互斥规则设置 self.generator。

        - generator_cls 与 generator_instance 互斥，同传抛 ValueError
        - generator_instance 必须是 Generator 实例，否则 TypeError
        - generator_instance 的 w_shape/b_shape 必须与 self.param_spec 一致，
          否则 ValueError
        - 都不传则为纯形状层（SLVT，由 Sequential 供参）
        """
        if generator_cls is not None and generator_instance is not None:
            raise ValueError(
                'generator_cls 与 generator_instance 互斥，只能传其中一个'
            )
        if generator_instance is not None:
            if generator_kwargs:
                raise ValueError(
                    'generator_instance 与 generator_kwargs 不能同时使用（如 z_dim）'
                )
            if not isinstance(generator_instance, Generator):
                raise TypeError(
                    f'generator_instance 必须是 Generator 实例，'
                    f'得到 {type(generator_instance).__name__}'
                )
            expected_w = self.param_spec['weight']
            expected_b = self.param_spec.get('bias')
            if (
                generator_instance.w_shape != expected_w
                or generator_instance.b_shape != expected_b
            ):
                raise ValueError(
                    f'generator_instance 的 param_spec 与层推导的不一致: '
                    f'期望 weight={expected_w}, bias={expected_b}，'
                    f'实际 weight={generator_instance.w_shape}, '
                    f'bias={generator_instance.b_shape}'
                )
            self.generator = generator_instance
        elif generator_cls is not None:
            self.generator = generator_cls(self.param_spec, **generator_kwargs)

    def _resolve(self, t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """解析张量形状：shaped 直通，flat 则 reshape。"""
        return t if t.shape == target_shape else t.reshape(target_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LWT 入口：调用自己的 generator -> _functional。"""
        w, b = self.generator()
        return self._functional(x, w, b)

    def forward_with_params(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None
    ) -> torch.Tensor:
        """SLVT 入口：接收外部参数 tuple -> _functional。"""
        return self._functional(x, w, b)

    def _functional(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None
    ) -> torch.Tensor:
        raise NotImplementedError
