"""Mapping 主干网络层：Conv2d, Linear。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapping.base import Generator, MappingLayer


class Conv2d(MappingLayer):
    """2D 卷积映射层。

    init 签名对齐 torch.nn.Conv2d。param_spec 自动推导。

    Args:
        in_channels  (int): 输入通道数 C_in
        out_channels (int): 输出通道数 C_out
        kernel_size  (int | tuple): 卷积核尺寸 (kh, kw)
        stride       (int | tuple): 步长 (默认 1)
        padding      (int | tuple): 填充 (默认 0)
        dilation     (int | tuple): 膨胀 (默认 1)
        groups       (int): 分组卷积数 (默认 1)
        bias         (bool): 是否使用偏置 (默认 True)
        generator_cls (type[Generator] | None): Generator 子类 (LWT 用)
        generator_instance (Generator | None): 已实例化的 Generator（权重捆绑用），
            与 generator_cls 互斥；param_spec 必须与层推导一致
        **generator_kwargs: 透传给 generator 构造函数的参数

    param_spec:
        weight: (C_out, C_in, kh, kw)
            总元素数 = C_out * C_in * kh * kw
        bias:   (C_out,)
            总元素数 = C_out  (仅 bias=True 时)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        generator_cls: type[Generator] | None = None,
        generator_instance: Generator | None = None,
        **generator_kwargs,
    ):
        super().__init__()
        kh, kw = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )

        self.param_spec = {'weight': (out_channels, in_channels, kh, kw)}
        if bias:
            self.param_spec['bias'] = (out_channels,)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias

        self._set_generator(generator_cls, generator_instance, generator_kwargs)

    def _functional(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None
    ) -> torch.Tensor:
        w = self._resolve(w, self.param_spec['weight'])
        if self.has_bias and b is not None:
            b = self._resolve(b, self.param_spec['bias'])
        return F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )


class Linear(MappingLayer):
    """线性映射层。

    init 签名对齐 torch.nn.Linear。param_spec 自动推导。

    Args:
        in_features  (int): 输入特征数 N_in
        out_features (int): 输出特征数 N_out
        bias         (bool): 是否使用偏置 (默认 True)
        generator_cls (type[Generator] | None): Generator 子类 (LWT 用)
        generator_instance (Generator | None): 已实例化的 Generator（权重捆绑用），
            与 generator_cls 互斥；param_spec 必须与层推导一致
        **generator_kwargs: 透传给 generator 构造函数的参数

    param_spec:
        weight: (N_out, N_in)
            总元素数 = N_out * N_in
        bias:   (N_out,)
            总元素数 = N_out  (仅 bias=True 时)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        generator_cls: type[Generator] | None = None,
        generator_instance: Generator | None = None,
        **generator_kwargs,
    ):
        super().__init__()
        self.param_spec = {'weight': (out_features, in_features)}
        if bias:
            self.param_spec['bias'] = (out_features,)

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        self._set_generator(generator_cls, generator_instance, generator_kwargs)

    def _functional(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None
    ) -> torch.Tensor:
        w = self._resolve(w, self.param_spec['weight'])
        if self.has_bias and b is not None:
            b = self._resolve(b, self.param_spec['bias'])
        return F.linear(x, w, b)
