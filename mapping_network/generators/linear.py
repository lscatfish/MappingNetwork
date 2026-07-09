import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ParameterGenerator


class LinearMappingNetwork(ParameterGenerator):
    """线性参数生成网络：固定行归一化权重 + 可学习 z。

    论文 Eq. 20 的映射公式：
        theta_hat = tanh(W_fixed @ z + alpha * mod(z) + b_fixed)

    其中 mod(z) 是论文 Figure 4 描述的 weight modulation：
        w_ij ← w_ij + alpha * z_i

    即对 W_fixed 的每一行 i 施加一个与 z 相关的加性偏移。
    由于 z ∈ R^d 而 W_fixed ∈ R^{P×d}，modulation 通过一个固定的
    映射矩阵 W_mod ∈ R^{P×d} 将 z 投影为 P 维偏移信号。

    初始化策略：
    - W_fixed 行归一化（每行 L2 范数 = 1），使 W@z 各分量方差 ~ O(1)
    - z 初始化 std=z_init_std（默认 0.5），使 a_i 集中在 tanh 线性区
    - W_mod 行归一化，保证调制信号的方差与 z 的方差同量级
    - b_fixed=0，方差完全由 W@z + alpha*W_mod@z 提供
    """

    def __init__(
        self, target_total_params: int, latent_dim: int, alpha: float = 0.01, device: str = 'cpu',
        w_seed: int | None = None, z_init_std: float = 0.5,
    ):
        super().__init__()
        self.P = target_total_params
        self.d = latent_dim
        self.alpha = alpha
        self.w_seed = w_seed

        # 固定种子生成 W_fixed 和 W_mod，便于 checkpoint 重建
        self._init_buffers(device)

        # 缩小初始方差，使 a_i 集中在 tanh 线性区
        self.z = nn.Parameter(torch.randn(self.d, device=device) * z_init_std)

    def _init_buffers(self, device: str):
        """从 w_seed 重建所有大 buffer。"""
        if self.w_seed is not None:
            torch.manual_seed(int(self.w_seed))
        # W_fixed: 行归一化，每行 L2=1
        W = torch.randn(self.P, self.d)
        W = W / W.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.register_buffer('W_fixed', W.to(device))
        self.register_buffer('W_fixed_mean', W.mean(dim=0).to(device))
        # W_mod: 调制矩阵（论文 Eq. 20 的 weight modulation）
        W_mod = torch.randn(self.P, self.d)
        W_mod = W_mod / W_mod.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.register_buffer('W_mod', W_mod.to(device))
        self.register_buffer('b_fixed', torch.zeros(self.P, device=device))

    def _rebuild_buffers(self):
        """从 w_seed 重建大 buffer（checkpoint 恢复时调用）。"""
        self._init_buffers(self.W_fixed.device.type if hasattr(self, 'W_fixed') else 'cpu')

    def _compute_activation(self, z: torch.Tensor) -> torch.Tensor:
        """计算 tanh 前的激活值。

        a = W_fixed @ z + alpha * (W_mod @ z) + b_fixed

        论文 Eq. 20: w_ij ← w_ij + alpha * z_i
        这里 W_mod @ z 实现了逐参数的 modulation：
        每个输出参数 j 获得一个基于 z 的调制信号 W_mod[j] @ z。
        """
        return self.W_fixed @ z + self.alpha * (self.W_mod @ z) + self.b_fixed

    def forward(self) -> torch.Tensor:
        return torch.tanh(self._compute_activation(self.z))

    def noisy_forward(self, sigma: float) -> torch.Tensor:
        eps = torch.randn_like(self.z) * sigma
        z_noisy = self.z + eps
        return torch.tanh(self._compute_activation(z_noisy))

    def smooth_loss(self) -> torch.Tensor:
        """L_smooth = ||nabla_z M(z)||^2_F / (P * d)。

        M(z) = tanh(W_fixed @ z + alpha * W_mod @ z + b)
        nabla_z M_i = tanh'(a_i) * (W_fixed[i, :] + alpha * W_mod[i, :])
        """
        a = self._compute_activation(self.z)
        tanh_derivative_sq = (1 - torch.tanh(a) ** 2) ** 2
        # ||W_fixed[i,:] + alpha*W_mod[i,:]||^2 ≈ 1 + alpha^2（行归一化，忽略交叉项）
        term1 = (1 + self.alpha ** 2) * tanh_derivative_sq.sum()
        return term1 / (self.P * self.d)

    def align_loss(self) -> torch.Tensor:
        """L_align = 1 - cos(z, mean(W_mod_effective, dim=0))。"""
        W_m = self.W_fixed_mean + self.alpha * self.W_mod.mean(dim=0)
        cos_sim = F.cosine_similarity(self.z.unsqueeze(0), W_m.unsqueeze(0))
        return 1 - cos_sim.squeeze()

    # ===== Checkpoint 恢复接口 =====
    # 大 buffer（W_fixed, W_mod, W_fixed_mean, b_fixed）不存入 checkpoint，
    # 由 w_seed 重建。

    _LIGHT_EXCLUDE = frozenset({'W_fixed', 'W_mod', 'W_fixed_mean', 'b_fixed'})

    def light_state_dict(self) -> dict:
        """返回不含大 buffer 的 state_dict。"""
        return {
            k: v for k, v in self.state_dict().items()
            if k not in self._LIGHT_EXCLUDE
        }

    def load_light_state_dict(self, state_dict: dict):
        """加载 light_state_dict，自动重建大 buffer。"""
        self._rebuild_buffers()
        self.load_state_dict(state_dict, strict=False)

    def extra_repr(self):
        return f'P={self.P}, d={self.d}, alpha={self.alpha}'
