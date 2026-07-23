import torch
import torch.nn.functional as F

from mapping.base import Generator
from mapping.loss import MappingLoss


class TinyGen(Generator):
    def __init__(self, param_spec, z_dim=8, **kwargs):
        super().__init__(param_spec, z_dim=z_dim)
        self.head = torch.nn.Linear(z_dim, self.w_size + self.b_size)

    def forward(self):
        h = self.head(self.z)
        w = h[:self.w_size].reshape(self.w_shape)
        b = h[self.w_size:].reshape(self.b_shape) if self.b_size > 0 else None
        return w, b


class TestMappingLossSLVT:
    """SLVT 模式：单个 generator。"""

    def test_forward_returns_loss_and_dict(self, device):
        gen = TinyGen({'weight': (4, 3), 'bias': (4,)}, z_dim=8).to(device)
        loss_fn = MappingLoss(n_stab_samples=2)
        logits = torch.randn(4, 10, device=device)
        target = torch.randint(0, 10, (4,), device=device)
        total, losses = loss_fn(logits, target, gen)
        assert total.ndim == 0
        assert 'task' in losses
        assert 'stab' in losses
        assert 'smooth' in losses
        assert 'align' in losses
        assert 'total' in losses

    def test_task_loss_is_cross_entropy(self, device):
        gen = TinyGen({'weight': (4, 3), 'bias': (4,)}, z_dim=8).to(device)
        loss_fn = MappingLoss(n_stab_samples=1)
        logits = torch.randn(4, 10, device=device)
        target = torch.randint(0, 10, (4,), device=device)
        _, losses = loss_fn(logits, target, gen)
        expected_task = F.cross_entropy(logits, target).item()
        assert abs(losses['task'] - expected_task) < 1e-5

    def test_gradient_flows_to_z(self, device):
        gen = TinyGen({'weight': (4, 3), 'bias': (4,)}, z_dim=8).to(device)
        loss_fn = MappingLoss(n_stab_samples=2)
        logits = torch.randn(4, 10, device=device, requires_grad=True)
        target = torch.randint(0, 10, (4,), device=device)
        total, _ = loss_fn(logits, target, gen)
        total.backward()
        assert gen.z.grad is not None
        assert not torch.allclose(gen.z.grad, torch.zeros_like(gen.z.grad))

    def test_lambda_params_learnable(self, device):
        loss_fn = MappingLoss()
        assert loss_fn.lambda_st.requires_grad
        assert loss_fn.lambda_sm.requires_grad
        assert loss_fn.lambda_al.requires_grad

    def test_no_bias_generator(self, device):
        gen = TinyGen({'weight': (4, 3)}, z_dim=8).to(device)
        loss_fn = MappingLoss(n_stab_samples=2)
        logits = torch.randn(4, 10, device=device)
        target = torch.randint(0, 10, (4,), device=device)
        total, _ = loss_fn(logits, target, gen)
        assert total.ndim == 0


class TestMappingLossLWT:
    """LWT 模式：多个 generator 列表。"""

    def test_forward_with_generator_list(self, device):
        gen1 = TinyGen({'weight': (4, 3), 'bias': (4,)}, z_dim=8).to(device)
        gen2 = TinyGen({'weight': (5, 4), 'bias': (5,)}, z_dim=8).to(device)
        loss_fn = MappingLoss(n_stab_samples=2)
        logits = torch.randn(4, 10, device=device)
        target = torch.randint(0, 10, (4,), device=device)
        total, losses = loss_fn(logits, target, [gen1, gen2])
        assert total.ndim == 0

    def test_gradient_flows_to_all_z(self, device):
        gen1 = TinyGen({'weight': (4, 3), 'bias': (4,)}, z_dim=8).to(device)
        gen2 = TinyGen({'weight': (5, 4), 'bias': (5,)}, z_dim=8).to(device)
        loss_fn = MappingLoss(n_stab_samples=2)
        logits = torch.randn(4, 10, device=device)
        target = torch.randint(0, 10, (4,), device=device)
        total, _ = loss_fn(logits, target, [gen1, gen2])
        total.backward()
        assert gen1.z.grad is not None
        assert gen2.z.grad is not None

    def test_lwt_regularization_averaged(self, device):
        """LWT 正则损失是各 generator 的均值。"""
        gen1 = TinyGen({'weight': (4, 3), 'bias': (4,)}, z_dim=8).to(device)
        gen2 = TinyGen({'weight': (5, 4), 'bias': (5,)}, z_dim=8).to(device)
        loss_fn = MappingLoss(n_stab_samples=2)
        logits = torch.randn(4, 10, device=device)
        target = torch.randint(0, 10, (4,), device=device)
        _, losses = loss_fn(logits, target, [gen1, gen2])
        assert losses['smooth'] >= 0
        assert 0 <= losses['align'] <= 2


class TestMappingLossComponents:
    """各损失分量的行为验证。"""

    def test_stab_zero_when_sigma_zero(self, device):
        gen = TinyGen({'weight': (4, 3), 'bias': (4,)}, z_dim=8).to(device)
        loss_fn = MappingLoss(sigma_noise=0.0, n_stab_samples=3)
        logits = torch.randn(4, 10, device=device)
        target = torch.randint(0, 10, (4,), device=device)
        _, losses = loss_fn(logits, target, gen)
        assert losses['stab'] < 1e-10

    def test_smooth_nonnegative(self, device):
        gen = TinyGen({'weight': (4, 3), 'bias': (4,)}, z_dim=8).to(device)
        loss_fn = MappingLoss(n_stab_samples=1)
        logits = torch.randn(4, 10, device=device)
        target = torch.randint(0, 10, (4,), device=device)
        _, losses = loss_fn(logits, target, gen)
        assert losses['smooth'] >= 0

    def test_align_bounded(self, device):
        gen = TinyGen({'weight': (4, 3), 'bias': (4,)}, z_dim=8).to(device)
        loss_fn = MappingLoss(n_stab_samples=1)
        logits = torch.randn(4, 10, device=device)
        target = torch.randint(0, 10, (4,), device=device)
        _, losses = loss_fn(logits, target, gen)
        assert 0 <= losses['align'] <= 2
