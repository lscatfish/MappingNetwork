import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ParameterGenerator


class CNNMappingNetwork(ParameterGenerator):
    """Conv2d-based generator: project z to a small feature map, convolve, flatten, project."""

    def __init__(
        self,
        target_total_params: int,
        latent_dim: int,
        alpha: float = 0.01,
        feature_size: int = 4,
        channels: tuple[int, ...] = (16, 8),
        device: str = 'cpu',
    ):
        super().__init__()
        self.P = target_total_params
        self.d = latent_dim
        self.alpha = alpha
        self.feature_size = feature_size
        self.channels = channels

        self.z = nn.Parameter(torch.randn(self.d, device=device) * 0.1)

        first_ch = channels[0]
        self.fc = nn.Linear(latent_dim, first_ch * feature_size * feature_size, device=device)

        conv_layers = []
        in_ch = first_ch
        for out_ch in channels[1:]:
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, device=device))
            conv_layers.append(nn.ReLU())
            in_ch = out_ch
        self.conv_net = nn.Sequential(*conv_layers)

        self.final = nn.Linear(in_ch * feature_size * feature_size, target_total_params, device=device)

    def _features(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc(z))
        x = x.view(-1, self.channels[0], self.feature_size, self.feature_size)
        x = self.conv_net(x)
        return x.view(1, -1)

    def _forward_z(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.final(self._features(z)).squeeze(0))

    def forward(self) -> torch.Tensor:
        return self._forward_z(self.z)

    def noisy_forward(self, sigma: float) -> torch.Tensor:
        eps = torch.randn_like(self.z) * sigma
        return self._forward_z(self.z + eps)

    def smooth_loss(self, chunk_size: int = 1024) -> torch.Tensor:
        theta = self.forward()
        P = theta.numel()
        total = torch.zeros((), device=theta.device, dtype=theta.dtype)
        for start in range(0, P, chunk_size):
            end = min(start + chunk_size, P)
            mask = torch.zeros(P, device=theta.device, dtype=theta.dtype)
            mask[start:end] = 1.0
            grads, = torch.autograd.grad(
                theta, self.z, grad_outputs=mask, retain_graph=True, create_graph=True
            )
            total = total + (grads * grads).sum()
        return total / (P * self.d)

    def align_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.z.device, requires_grad=True)

    def extra_repr(self):
        return f'P={self.P}, d={self.d}, alpha={self.alpha}, channels={self.channels}'
