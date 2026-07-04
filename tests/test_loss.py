import pytest
import torch
from mapping_network.mapping.mapping_net import MappingNetwork
from mapping_network.mapping.loss import MappingLoss
from mapping_network.target_nets.cnn2 import CNN2


def test_mapping_loss_forward():
    target = CNN2()
    mapping = MappingNetwork(target.get_total_params(), 64)
    loss_fn = MappingLoss()

    theta = mapping()
    eps = torch.randn_like(mapping.z) * 0.01
    z_noisy = mapping.z + eps
    W_mod_noisy = mapping.W_fixed + mapping.alpha * z_noisy.unsqueeze(0)
    theta_noisy = torch.tanh(W_mod_noisy @ z_noisy + mapping.b_fixed)

    x = torch.randn(2, 1, 28, 28)
    y = torch.randint(0, 10, (2,))

    loss, losses_dict = loss_fn(mapping.z, theta, theta_noisy, mapping, target, x, y)
    assert loss.requires_grad
    assert loss.item() > 0


def test_mapping_loss_gradient_to_z():
    """Verify that all loss components backpropagate gradients to z."""
    target = CNN2()
    mapping = MappingNetwork(target.get_total_params(), 64)
    loss_fn = MappingLoss()

    theta = mapping()
    eps = torch.randn_like(mapping.z) * 0.01
    z_noisy = mapping.z + eps
    W_mod_noisy = mapping.W_fixed + mapping.alpha * z_noisy.unsqueeze(0)
    theta_noisy = torch.tanh(W_mod_noisy @ z_noisy + mapping.b_fixed)

    x = torch.randn(2, 1, 28, 28)
    y = torch.randint(0, 10, (2,))

    loss, _ = loss_fn(mapping.z, theta, theta_noisy, mapping, target, x, y)
    loss.backward()
    assert mapping.z.grad is not None
    assert mapping.z.grad.shape == (64,)
