import torch
import torch.nn as nn
import torch.nn.init as init


class MappingNetwork(nn.Module):
    """
    映射网络：从低维 latent vector z 生成目标网络参数。

    - W_fixed: 固定正交初始化映射矩阵 [P, d]（buffer，不训练）
    - b_fixed: 固定偏置 [P]（buffer，不训练）
    - z: 可训练的 latent vector [d]（nn.Parameter）
    - α: 调制系数

    前向: θ̂ = tanh(W_mod · z + b)         (方程 21)
    其中 W_mod[i,:] = W_fixed[i,:] + α·z   (方程 20)

    返回 θ̂ ∈ R^P。不执行参数注入——由调用方传给 target_net.functional_forward()。
    """

    def __init__(self, target_total_params: int, latent_dim: int, alpha: float = 0.01):
        super().__init__()
        self.P = target_total_params
        self.d = latent_dim
        self.alpha = alpha

        # 固定正交初始化映射权重 [P, d]
        W = torch.empty(self.P, self.d)
        init.orthogonal_(W)
        self.register_buffer('W_fixed', W)

        # 固定偏置 [P]
        self.register_buffer('b_fixed', torch.zeros(self.P))

        # 可训练的 latent vector [d]
        self.z = nn.Parameter(torch.randn(self.d) * 0.1)

    def forward(self):
        """返回 θ̂ ∈ R^P。"""
        # W_mod[i,:] = W_fixed[i,:] + α·z  (广播到全部 P 行)
        W_mod = self.W_fixed + self.alpha * self.z.unsqueeze(0)
        theta_hat = torch.tanh(W_mod @ self.z + self.b_fixed)
        return theta_hat

    def extra_repr(self):
        return f"P={self.P}, d={self.d}, alpha={self.alpha}"
