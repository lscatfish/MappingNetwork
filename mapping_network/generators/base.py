import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class ParameterGenerator(nn.Module, ABC):
    """参数生成网络基类。只负责生成 theta_hat。"""

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """返回 theta_hat [P']，P' 是目标网络压缩后的总参数数。"""
        pass

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
