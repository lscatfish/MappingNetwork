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

    # 默认 w_seed。当用户未指定时，LinearMappingNetwork 用 (P, d) 派生一个
    # 确定性的 seed，保证不同 (P, d) 组合得到不同 W_fixed / W_mod。
    _DEFAULT_W_SEED = 0x4C4D4E54  # 'LMNT' ascii

    def __init__(
        self, target_total_params: int, latent_dim: int, alpha: float = 0.01, device: str = 'cpu',
        w_seed: int | None = None, z_init_std: float = 0.5, layer_name: str | None = None,
    ):
        """线性参数生成网络。

        Args:
            target_total_params: 目标网络压缩后总参数 P。
            latent_dim: 隐向量维度 d。
            alpha: modulation 系数。
            device: 设备。
            w_seed: 重建大 buffer 的种子。**由 generator 内部管理**。
                优先级：用户显式指定 > 基于 layer_name 派生 > 基于 (P, d) 派生。
            z_init_std: z 初始化标准差。
            layer_name: 可选的层名（LWT 场景下由 trainer 注入）。
                如果提供，会与 w_seed 联合派生一个唯一 seed，使各层 W_fixed 不同。
        """
        super().__init__()
        self.P = target_total_params
        self.d = latent_dim
        self.alpha = alpha

        # w_seed 完全是 generator 私有实现细节，外部不应读写。
        # 当外部提供 layer_name 时，自动派生该层唯一 seed；否则基于 (P, d) 派生。
        self.w_seed = self._derive_seed(
            target_total_params, latent_dim, w_seed, layer_name
        )

        # 固定种子生成 W_fixed 和 W_mod，便于 checkpoint 重建
        self._init_buffers(device)

        # 缩小初始方差，使 a_i 集中在 tanh 线性区
        self.z = nn.Parameter(torch.randn(self.d, device=device) * z_init_std)

    @classmethod
    def _derive_seed(
        cls, target_total_params: int, latent_dim: int,
        w_seed: int | None, layer_name: str | None,
    ) -> int:
        """根据用户输入、layer_name 或 (P, d) 派生一个确定性 seed。

        优先级：
        1. 用户显式指定 w_seed 且未提供 layer_name -> 直接使用 w_seed。
        2. 用户显式指定 w_seed 且提供 layer_name -> 基于 (w_seed, layer_name) hash。
        3. 未指定 w_seed 但提供 layer_name -> 基于 (DEFAULT, layer_name) hash。
        4. 否则 -> 基于 (P, d) hash。

        注意：此方法**不**依赖外部 idx。LWT 各层通过传入不同 layer_name
        自动获得不同 seed，且 generator 内部可自行演化 seed 策略。
        """
        if w_seed is not None:
            base = int(w_seed)
            if layer_name is not None:
                return hash((base, str(layer_name))) & 0x7FFFFFFF
            return base
        if layer_name is not None:
            return hash((cls._DEFAULT_W_SEED, str(layer_name))) & 0x7FFFFFFF
        return hash((cls._DEFAULT_W_SEED, target_total_params, latent_dim)) & 0x7FFFFFFF

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

        精确计算每行的梯度范数平方：
        ||W_fixed[i,:] + alpha*W_mod[i,:]||^2
        = ||W_fixed[i,:]||^2 + 2*alpha*<W_fixed[i,:], W_mod[i,:]> + alpha^2*||W_mod[i,:]||^2

        由于 W_fixed 和 W_mod 都行归一化（||W[i,:]||=1），简化为：
        = 1 + 2*alpha*<W_fixed[i,:], W_mod[i,:]> + alpha^2
        但此处不做近似，直接逐行计算。
        """
        a = self._compute_activation(self.z)
        tanh_derivative_sq = (1 - torch.tanh(a) ** 2) ** 2
        # 精确计算每行的 ||W_fixed[i,:] + alpha*W_mod[i,:]||^2
        grad_rows = self.W_fixed + self.alpha * self.W_mod
        grad_norm_sq = grad_rows.pow(2).sum(dim=1)
        term1 = (grad_norm_sq * tanh_derivative_sq).sum()
        return term1 / (self.P * self.d)

    def align_loss(self) -> torch.Tensor:
        """L_align = 1 - cos(z, mean(W_mod_effective, dim=0))。"""
        W_m = self.W_fixed_mean + self.alpha * self.W_mod.mean(dim=0)
        cos_sim = F.cosine_similarity(self.z.unsqueeze(0), W_m.unsqueeze(0))
        return 1 - cos_sim.squeeze()

    # ===== Checkpoint 恢复接口 =====
    # 大 buffer（W_fixed, W_mod, W_fixed_mean, b_fixed）不存入 checkpoint，
    # 由 w_seed 重建。

    _PERSISTENT_EXCLUDE = frozenset({'W_fixed', 'W_mod', 'W_fixed_mean', 'b_fixed'})

    def persistent_state_dict(self) -> dict:
        """返回不含大 buffer 的 state_dict。"""
        return {
            k: v for k, v in self.state_dict().items()
            if k not in self._PERSISTENT_EXCLUDE
        }

    def load_persistent_state_dict(self, state_dict: dict):
        """加载 persistent_state_dict，自动重建大 buffer。"""
        self._rebuild_buffers()
        self.load_state_dict(state_dict, strict=False)

    def extra_repr(self):
        return f'P={self.P}, d={self.d}, alpha={self.alpha}'
