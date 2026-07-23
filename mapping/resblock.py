"""Trunk 级残差块容器。"""

import torch
import torch.nn.functional as F

from mapping.base import MappingLayer, _prod
from mapping.layers import Conv2d


class ResBlock(MappingLayer):
    """主干级残差块：两个 Conv2d + 跳连。

    双模式：
    - LWT：传 generator_cls，内部各层各自带 generator，forward() 直接调用
    - SLVT：不传 generator_cls，纯形状层，param_spec 为聚合 flat，
      forward_with_params 收到整段切片后内部按边界二次切片

    通道数或空间尺寸变化时自动启用 1x1 shortcut 卷积。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = True,
        generator_cls=None,
        **generator_kwargs,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.use_shortcut = (in_channels != out_channels) or (stride != 1)
        self.has_bias = bias

        gen_kw = (
            dict(generator_cls=generator_cls, **generator_kwargs) if generator_cls else {}
        )

        self.conv1 = Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias, **gen_kw,
        )
        self.conv2 = Conv2d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, bias=bias, **gen_kw,
        )
        if self.use_shortcut:
            self.shortcut = Conv2d(
                in_channels, out_channels, 1,
                stride=stride, bias=bias, **gen_kw,
            )

        if generator_cls is None:
            self._build_aggregated_spec(bias)

    def _build_aggregated_spec(self, bias: bool) -> None:
        layers = [self.conv1, self.conv2]
        if self.use_shortcut:
            layers.append(self.shortcut)

        w_total = sum(_prod(layer.param_spec['weight']) for layer in layers)
        self.param_spec = {'weight': (w_total,)}

        if bias:
            b_total = sum(_prod(layer.param_spec['bias']) for layer in layers)
            self.param_spec['bias'] = (b_total,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LWT 入口：内部各层用自己的 generator。"""
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if self.use_shortcut:
            identity = self.shortcut(x)
        return out + identity

    def forward_with_params(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None
    ) -> torch.Tensor:
        """SLVT 入口：接收聚合 flat 参数，内部二次切片。"""
        offset_w = 0
        offset_b = 0

        def _slice(layer):
            nonlocal offset_w, offset_b
            ws = _prod(layer.param_spec['weight'])
            w_slice = w[offset_w:offset_w + ws]
            offset_w += ws
            b_slice = None
            if self.has_bias and b is not None:
                bs = _prod(layer.param_spec['bias'])
                b_slice = b[offset_b:offset_b + bs]
                offset_b += bs
            return w_slice, b_slice

        identity = x
        w1, b1 = _slice(self.conv1)
        out = F.relu(self.conv1.forward_with_params(x, w1, b1))

        w2, b2 = _slice(self.conv2)
        out = self.conv2.forward_with_params(out, w2, b2)

        if self.use_shortcut:
            ws, bs = _slice(self.shortcut)
            identity = self.shortcut.forward_with_params(x, ws, bs)

        return out + identity
