"""Tests for MappingLoss — runs on both CPU and GPU."""

import torch

from mapping_network.generators.linear import LinearMappingNetwork
from mapping_network.mapping.loss import MappingLoss
from mapping_network.target_nets import CNN2


class TestMappingLoss:
    def test_mapping_loss_forward(self, device):
        target = CNN2().to(device)
        mapping = LinearMappingNetwork(target.get_total_params(), 64).to(device)
        loss_fn = MappingLoss().to(device)

        theta = mapping()

        x = torch.randn(2, 1, 28, 28, device=device)
        y = torch.randint(0, 10, (2,), device=device)

        loss, losses_dict = loss_fn(theta, mapping, target, x, y)
        assert loss.requires_grad
        assert loss.item() > 0
        assert theta.device.type == device
        assert losses_dict['total'] == loss.item()

    def test_mapping_loss_gradient_to_z(self, device):
        """Verify that all loss components backpropagate gradients to z."""
        target = CNN2().to(device)
        mapping = LinearMappingNetwork(target.get_total_params(), 64).to(device)
        loss_fn = MappingLoss().to(device)

        theta = mapping()

        x = torch.randn(2, 1, 28, 28, device=device)
        y = torch.randint(0, 10, (2,), device=device)

        loss, _ = loss_fn(theta, mapping, target, x, y)
        loss.backward()
        assert mapping.z.grad is not None
        assert mapping.z.grad.shape == (64,)
        assert mapping.z.grad.device.type == device


def test_mapping_loss_lambda_inits():
    import pytest

    loss_fn = MappingLoss(lambda_st_init=0.01, lambda_sm_init=0.02, lambda_al_init=0.03)
    assert loss_fn.lambda_st.item() == pytest.approx(0.01, abs=1e-6)
    assert loss_fn.lambda_sm.item() == pytest.approx(0.02, abs=1e-6)
    assert loss_fn.lambda_al.item() == pytest.approx(0.03, abs=1e-6)


def test_mapping_loss_forward_lrd(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    target_net = CNN2(lrd_config={'enabled': True, 'default_rank': 10}).to(device)
    mapping = LinearMappingNetwork(target_net.get_total_params(), 64, device=device)
    loss_fn = MappingLoss(sigma_noise=0.01).to(device)
    x = torch.randn(2, 1, 28, 28, device=device)
    y = torch.tensor([0, 1], device=device)
    theta = mapping()
    loss, losses = loss_fn(theta, mapping, target_net, x, y)
    assert loss.item() == losses['total']
    loss.backward()
    assert mapping.z.grad is not None
