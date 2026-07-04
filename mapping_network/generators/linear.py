import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .base import ParameterGenerator


class LinearMappingNetwork(ParameterGenerator):
    """线性参数生成网络：固定正交权重 + 可学习 z。"""

    def __init__(
        self, target_total_params: int, latent_dim: int, alpha: float = 0.01, device: str = 'cpu'
    ):
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
            self.W_fixed @ self.z + self.alpha * (self.z * self.z).sum() + self.b_fixed
        )

    def noisy_forward(self, sigma: float) -> torch.Tensor:
        """对 z 加噪声后前向，返回 theta_noisy。"""
        eps = torch.randn_like(self.z) * sigma
        z_noisy = self.z + eps
        return torch.tanh(
            self.W_fixed @ z_noisy + self.alpha * (z_noisy * z_noisy).sum() + self.b_fixed
        )

    def smooth_loss(self) -> torch.Tensor:
        """L_smooth = ||nabla_z M(z)||^2_F / (P * d)。

        M(z) = tanh(W_fixed @ z + alpha * ||z||^2 + b)
        nabla_z M_i = tanh'(a_i) * (W_fixed[i, :] + 2 * alpha * z)
        分项计算，避免产生 [P, d] 中间张量导致显存翻倍。
        """
        a = self.W_fixed @ self.z + self.alpha * (self.z * self.z).sum() + self.b_fixed
        tanh_derivative_sq = (1 - torch.tanh(a) ** 2) ** 2

        # term1 = sum_i tanh'(a_i)^2 * ||W_fixed[i, :]||^2
        row_norms_sq = torch.zeros(self.P, device=self.z.device, dtype=self.z.dtype)
        chunk_size = 10000
        for start in range(0, self.P, chunk_size):
            end = min(start + chunk_size, self.P)
            row_norms_sq[start:end] = (self.W_fixed[start:end] ** 2).sum(dim=1)
        term1 = (tanh_derivative_sq * row_norms_sq).sum()

        # term2 = sum_i tanh'(a_i)^2 * 4*alpha*W_fixed[i, :]@z
        #       = 4*alpha * z @ (W_fixed.T @ tanh_derivative_sq)
        term2 = 4 * self.alpha * (self.z * (self.W_fixed.T @ tanh_derivative_sq)).sum()

        # term3 = sum_i tanh'(a_i)^2 * 4*alpha^2*||z||^2
        term3 = 4 * self.alpha * self.alpha * (self.z * self.z).sum() * tanh_derivative_sq.sum()

        return (term1 + term2 + term3) / (self.P * self.d)

    def align_loss(self) -> torch.Tensor:
        """L_align = 1 - cos(z, mean(W_mod, dim=0))。"""
        W_m = self.W_fixed_mean + self.alpha * self.z
        cos_sim = F.cosine_similarity(self.z.unsqueeze(0), W_m.unsqueeze(0))
        return 1 - cos_sim.squeeze()

    def extra_repr(self):
        return f'P={self.P}, d={self.d}, alpha={self.alpha}'
