"""固定随机参数 Linear 子块。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """固定随机参数的线性层子块。

    init 签名对齐 torch.nn.Linear。内部参数在构造时随机初始化
    并设为 requires_grad=False。默认采用论文方法初始化，
    用户可重载 init_weights() 自定义。

    Args:
        in_features: 输入特征数
        out_features: 输出特征数
        bias: 是否使用偏置 (默认 True)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features), requires_grad=False
            )
        else:
            self.register_parameter('bias', None)

        self.init_weights()

    def init_weights(self):
        """默认论文初始化方法：kaiming uniform。

        子类可重载此方法自定义初始化。
        """
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
