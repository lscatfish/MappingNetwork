"""Mapping 框架通用损失：MappingLoss。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapping.base import Generator


class MappingLoss(nn.Module):
    """通用 Mapping Loss，只依赖 generator.z 与 generator.forward()。

    L_map = L_task + sigmoid(λ_st)·L_stab + sigmoid(λ_sm)·L_smooth + sigmoid(λ_al)·L_align

    支持双模式：
    - SLVT：generators 为单个 Generator
    - LWT：generators 为 Generator 列表，正则损失逐层计算后取均值
    """

    def __init__(
        self,
        sigma_noise: float = 1e-4,
        n_stab_samples: int = 5,
        lambda_st_init: float = 0.1,
        lambda_sm_init: float = 0.1,
        lambda_al_init: float = 0.1,
    ):
        super().__init__()
        self.sigma_noise = sigma_noise
        self.n_stab_samples = n_stab_samples
        self.lambda_st = nn.Parameter(torch.tensor(lambda_st_init))
        self.lambda_sm = nn.Parameter(torch.tensor(lambda_sm_init))
        self.lambda_al = nn.Parameter(torch.tensor(lambda_al_init))

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        generators: 'Generator | list[Generator]',
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if isinstance(generators, Generator):
            generators = [generators]

        l_task = F.cross_entropy(logits, target)

        l_stab = torch.stack([self._stab_loss(g) for g in generators]).mean()
        l_smooth = torch.stack([self._smooth_loss(g) for g in generators]).mean()
        l_align = torch.stack([self._align_loss(g) for g in generators]).mean()

        total = (
            l_task
            + torch.sigmoid(self.lambda_st) * l_stab
            + torch.sigmoid(self.lambda_sm) * l_smooth
            + torch.sigmoid(self.lambda_al) * l_align
        )

        losses = {
            'task': l_task.item(),
            'stab': l_stab.item(),
            'smooth': l_smooth.item(),
            'align': l_align.item(),
            'total': total.item(),
        }
        return total, losses

    def _stab_loss(self, generator: Generator) -> torch.Tensor:
        """L_stab：z 加噪后重新前向，与无噪输出求 MSE。"""
        with torch.no_grad():
            w_clean, b_clean = generator()
            clean_flat = self._flat(w_clean, b_clean)

        l_stab = torch.tensor(0.0, device=generator.z.device)
        for _ in range(self.n_stab_samples):
            noise = torch.randn_like(generator.z) * self.sigma_noise
            z_noisy = generator.z + noise
            w_noisy, b_noisy = self._forward_with_z(generator, z_noisy)
            noisy_flat = self._flat(w_noisy, b_noisy)
            l_stab = l_stab + F.mse_loss(noisy_flat, clean_flat)
        return l_stab / self.n_stab_samples

    def _smooth_loss(self, generator: Generator) -> torch.Tensor:
        """L_smooth：‖J‖²_F / (P·d)，J = ∂(w_flat)/∂z。"""
        z = generator.z
        d = z.shape[0]

        def mapping_fn(z_val):
            w, b = self._forward_with_z(generator, z_val)
            return self._flat(w, b)

        J = torch.autograd.functional.jacobian(mapping_fn, z, create_graph=True)
        p = J.shape[0]
        return (J ** 2).sum() / (p * d)

    def _align_loss(self, generator: Generator) -> torch.Tensor:
        """L_align：1 - cos(z, mean(w_flat 按列))。"""
        z = generator.z

        def mapping_fn(z_val):
            w, b = self._forward_with_z(generator, z_val)
            return self._flat(w, b)

        J = torch.autograd.functional.jacobian(mapping_fn, z, create_graph=True)
        mean_direction = J.mean(dim=0)
        return 1.0 - F.cosine_similarity(z.unsqueeze(0), mean_direction.unsqueeze(0)).squeeze()

    def _forward_with_z(
        self, generator: Generator, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """用指定 z 执行 generator 前向（临时替换）。"""
        object.__setattr__(generator, 'z', z)
        try:
            w, b = generator()
        finally:
            object.__delattr__(generator, 'z')
        return w, b

    @staticmethod
    def _flat(w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
        if b is not None:
            return torch.cat([w.reshape(-1), b.reshape(-1)])
        return w.reshape(-1)
