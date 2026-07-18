import torch
import torch.nn as nn

from .base import ParameterGenerator, register_generator


@register_generator('multilayer_linear')
class MultiLayerLinearMappingNetwork(ParameterGenerator):
    """MLP-style generator: z -> Linear -> ReLU -> ... -> Linear -> theta_hat."""

    def __init__(
        self,
        target_total_params: int,
        latent_dim: int,
        alpha: float = 0.01,
        hidden_dim: int = 64,
        num_hidden: int = 1,
        device: str = 'cpu',
    ):
        super().__init__()
        self.P = target_total_params
        self.d = latent_dim
        self.alpha = alpha

        self.z = nn.Parameter(torch.randn(self.d, device=device) * 0.1)

        layers = []
        in_dim = latent_dim
        for _ in range(num_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim, device=device))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, target_total_params, device=device))
        self.net = nn.Sequential(*layers)

    def _forward_z(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(z))

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
            (grads,) = torch.autograd.grad(
                theta, self.z, grad_outputs=mask, retain_graph=True, create_graph=True
            )
            total = total + (grads * grads).sum()
        return total / (P * self.d)

    def align_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.z.device, requires_grad=True)

    def extra_repr(self):
        return f'P={self.P}, d={self.d}, alpha={self.alpha}, hidden_dim={self.net[0].out_features}'
