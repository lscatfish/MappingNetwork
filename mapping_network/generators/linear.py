import torch
import torch.nn as nn
import torch.nn.init as init

from .base import ParameterGenerator


class LinearMappingNetwork(ParameterGenerator):
    """线性参数生成网络：固定正交权重 + 可学习 z。"""

    def __init__(self, target_total_params: int, latent_dim: int,
                 alpha: float = 0.01, device: str = 'cpu'):
        super().__init__()
        self.P = target_total_params
        self.d = latent_dim
        self.alpha = alpha

        W = torch.empty(self.P, self.d, device=device)
        init.orthogonal_(W)
        self.register_buffer('W_fixed', W)
        self.register_buffer('W_fixed_mean', W.mean(dim=0))
        self.register_buffer('b_fixed', torch.zeros(self.P, device=device))
        self.z = nn.Parameter(torch.randn(self.d, device=device) * 0.1)

    def forward(self) -> torch.Tensor:
        return torch.tanh(
            self.W_fixed @ self.z
            + self.alpha * (self.z * self.z).sum()
            + self.b_fixed
        )

    def extra_repr(self):
        return f'P={self.P}, d={self.d}, alpha={self.alpha}'
