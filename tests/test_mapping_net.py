import pytest
import torch
from mapping_network.mapping.mapping_net import MappingNetwork
from mapping_network.target_nets.cnn2 import CNN2


def test_mapping_network_output_shape():
    d = 128
    net = MappingNetwork(108610, d)
    theta = net()
    assert theta.shape == (108610,)


def test_mapping_network_trainable_params():
    net = MappingNetwork(108610, 2048)
    trainable = [p for p in net.parameters() if p.requires_grad]
    assert len(trainable) == 1  # 只有 z
    assert trainable[0].shape == (2048,)


def test_mapping_network_fixed_weights():
    net = MappingNetwork(108610, 2048)
    assert not net.W_fixed.requires_grad
    assert not net.b_fixed.requires_grad


def test_gradient_flows_through_theta_hat():
    """核心测试：验证梯度能从 θ̂ 回传至 z。"""
    target = CNN2()
    mapping = MappingNetwork(target.get_total_params(), 128)
    x = torch.randn(2, 1, 28, 28)

    theta_hat = mapping()
    y = target.functional_forward(x, theta_hat)
    y.sum().backward()

    assert mapping.z.grad is not None
    assert mapping.z.grad.shape == (128,)
