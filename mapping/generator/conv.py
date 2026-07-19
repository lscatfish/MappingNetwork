"""固定随机参数 Conv1d / Conv2d 子块。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvNd(nn.Module):
    """Conv 子块基类，共享 init_weights 逻辑。"""

    def __init__(self):
        super().__init__()

    def init_weights(self):
        """默认论文初始化方法：kaiming uniform。

        子类可重载此方法自定义初始化。
        """
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.weight.size(1)
            for s in self.weight.shape[2:]:
                fan_in *= s
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


class Conv1d(_ConvNd):
    """固定随机参数的一维卷积子块。

    init 签名对齐 torch.nn.Conv1d。内部参数在构造时随机初始化
    并设为 requires_grad=False。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核尺寸
        stride: 步长 (默认 1)
        padding: 填充 (默认 0)
        dilation: 膨胀 (默认 1)
        groups: 分组卷积数 (默认 1)
        bias: 是否使用偏置 (默认 True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )


class Conv2d(_ConvNd):
    """固定随机参数的二维卷积子块。

    init 签名对齐 torch.nn.Conv2d。内部参数在构造时随机初始化
    并设为 requires_grad=False。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核尺寸 (int 或 tuple)
        stride: 步长 (默认 1)
        padding: 填充 (默认 0)
        dilation: 膨胀 (默认 1)
        groups: 分组卷积数 (默认 1)
        bias: 是否使用偏置 (默认 True)
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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )
