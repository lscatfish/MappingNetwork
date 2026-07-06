import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ParameterGenerator


class LinearMappingNetwork(ParameterGenerator):
    """线性参数生成网络：固定行归一化权重 + 可学习 z。

    初始化策略（修复 theta_hat 方差坍缩问题）：
    - W_fixed 行归一化（每行 L2 范数 = 1），使 W@z 各分量方差 ~ O(1)
      （对比 orthogonal_ 列正交时行范数 ~ sqrt(d/P) ≈ 0.14，方差过小）
    - z 初始化 std=1.0（配合行归一化 W，使 a_i ~ N(0,1)）
    - alpha 除以 latent_dim，避免 ||z||^2 项在 d 较大时主导并使 tanh 饱和
    - b_fixed=0，方差完全由 W@z 提供
    """

    def __init__(
        self, target_total_params: int, latent_dim: int, alpha: float = 0.01, device: str = 'cpu',
        w_seed: int | None = None,
    ):
        super().__init__()
        self.P = target_total_params
        self.d = latent_dim
        # 对 latent_dim 归一化，避免 ||z||^2 随 d 增大而主导激活值导致 tanh 饱和
        self.alpha = alpha / self.d

        # 固定种子生成 W_fixed，便于 checkpoint 重建（不存大矩阵）
        if w_seed is not None:
            torch.manual_seed(int(w_seed))
        # 行归一化：每行 L2 范数 = 1，保证 W@z 各分量方差 ~ z 的方差
        W = torch.randn(self.P, self.d)
        W = W / W.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.register_buffer('W_fixed', W.to(device))
        self.register_buffer('W_fixed_mean', W.mean(dim=0).to(device))
        self.register_buffer('b_fixed', torch.zeros(self.P, device=device))
        self.w_seed = w_seed  # 保存种子（非 buffer/parameter，不进 state_dict）
        self.z = nn.Parameter(torch.randn(self.d, device=device))

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
        # 行归一化后每行 ||W_i||^2 = 1，term1 简化为 tanh'(a)^2 的求和
        term1 = tanh_derivative_sq.sum()

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
