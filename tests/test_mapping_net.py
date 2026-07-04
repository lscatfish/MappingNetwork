"""Tests for MappingNetwork — runs on both CPU and GPU."""

import pytest
import torch
from mapping_network.mapping.mapping_net import MappingNetwork
from mapping_network.target_nets.cnn2 import CNN2


class TestMappingNetwork:
    def test_mapping_network_output_shape(self, device):
        d = 128
        net = MappingNetwork(108610, d).to(device)
        theta = net()
        assert theta.shape == (108610,)
        assert theta.device.type == device

    def test_mapping_network_trainable_params(self, device):
        net = MappingNetwork(108610, 2048).to(device)
        trainable = [p for p in net.parameters() if p.requires_grad]
        assert len(trainable) == 1  # 只有 z
        assert trainable[0].shape == (2048,)

    def test_mapping_network_fixed_weights(self, device):
        net = MappingNetwork(108610, 2048).to(device)
        assert not net.W_fixed.requires_grad
        assert not net.b_fixed.requires_grad
        assert net.W_fixed.device.type == device

    def test_gradient_flows_through_theta_hat(self, device):
        """核心测试：验证梯度能从 θ̂ 回传至 z，且全在 GPU 上。"""
        target = CNN2().to(device)
        mapping = MappingNetwork(target.get_total_params(), 128).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)

        theta_hat = mapping()
        y = target.functional_forward(x, theta_hat)
        y.sum().backward()

        assert mapping.z.grad is not None
        assert mapping.z.grad.shape == (128,)
        assert mapping.z.grad.device.type == device
