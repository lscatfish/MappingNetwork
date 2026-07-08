from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ParameterGenerator(nn.Module, ABC):
    """参数生成网络基类。负责生成 theta_hat 以及相关的辅助量。"""

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """返回 theta_hat [P']，P' 是目标网络压缩后的总参数数。"""
        pass

    @abstractmethod
    def noisy_forward(self, sigma: float) -> torch.Tensor:
        """对隐变量加高斯噪声后前向，返回 theta_noisy（用于 L_stab）。"""
        pass

    @abstractmethod
    def smooth_loss(self) -> torch.Tensor:
        """返回 L_smooth = ||nabla_z M(z)||^2_F / (P * d)。"""
        pass

    @abstractmethod
    def align_loss(self) -> torch.Tensor:
        """返回 L_align = 1 - cos(z, mean(W_mod, dim=0))。"""
        pass

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def persistent_state_dict(self) -> dict:
        """默认只保存可学习参数；固定 buffer 由 __init__ 重建。"""
        return {k: v for k, v in self.named_parameters() if v.requires_grad}

    def load_persistent_state_dict(self, state: dict):
        """从 checkpoint 恢复可学习参数。"""
        missing, unexpected = self.load_state_dict(state, strict=False)
        return missing, unexpected
