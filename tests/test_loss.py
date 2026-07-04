"""Tests for MappingLoss — runs on both CPU and GPU."""

import pytest
import torch
from mapping_network.mapping.mapping_net import MappingNetwork
from mapping_network.mapping.loss import MappingLoss
from mapping_network.target_nets.cnn2 import CNN2


class TestMappingLoss:
    def test_mapping_loss_forward(self, device):
        target = CNN2().to(device)
        mapping = MappingNetwork(target.get_total_params(), 64).to(device)
        loss_fn = MappingLoss().to(device)

        theta = mapping()
        eps = torch.randn_like(mapping.z) * 0.01
        z_noisy = mapping.z + eps
        W_mod_noisy = mapping.W_fixed + mapping.alpha * z_noisy.unsqueeze(0)
        theta_noisy = torch.tanh(W_mod_noisy @ z_noisy + mapping.b_fixed)

        x = torch.randn(2, 1, 28, 28, device=device)
        y = torch.randint(0, 10, (2,), device=device)

        loss, losses_dict = loss_fn(mapping.z, theta, theta_noisy, mapping, target, x, y)
        assert loss.requires_grad
        assert loss.item() > 0
        assert theta.device.type == device

    def test_mapping_loss_gradient_to_z(self, device):
        """Verify that all loss components backpropagate gradients to z on GPU."""
        target = CNN2().to(device)
        mapping = MappingNetwork(target.get_total_params(), 64).to(device)
        loss_fn = MappingLoss().to(device)

        theta = mapping()
        eps = torch.randn_like(mapping.z) * 0.01
        z_noisy = mapping.z + eps
        W_mod_noisy = mapping.W_fixed + mapping.alpha * z_noisy.unsqueeze(0)
        theta_noisy = torch.tanh(W_mod_noisy @ z_noisy + mapping.b_fixed)

        x = torch.randn(2, 1, 28, 28, device=device)
        y = torch.randint(0, 10, (2,), device=device)

        loss, _ = loss_fn(mapping.z, theta, theta_noisy, mapping, target, x, y)
        loss.backward()
        assert mapping.z.grad is not None
        assert mapping.z.grad.shape == (64,)
        assert mapping.z.grad.device.type == device
