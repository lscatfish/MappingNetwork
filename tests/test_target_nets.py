import pytest
import torch
from mapping_network.target_nets.cnn2 import CNN2


def test_cnn2_parameter_count():
    model = CNN2()
    total = sum(p.numel() for p in model.parameters())
    assert total == 108610, f'Expected 108610, got {total}'


def test_cnn2_forward():
    model = CNN2()
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    assert y.shape == (4, 10)


def test_cnn2_functional_forward():
    """验证函数式前向输出与模块前向一致，且梯度可回传至 theta_hat。"""
    model = CNN2()
    x = torch.randn(2, 1, 28, 28)
    theta_hat = torch.randn(model.get_total_params(), requires_grad=True)
    y = model.functional_forward(x, theta_hat)
    loss = y.sum()
    loss.backward()
    assert theta_hat.grad is not None
    assert theta_hat.grad.shape == (model.get_total_params(),)


from mapping_network.target_nets.cnn1 import CNN1
from mapping_network.target_nets.cnn1_3conv import CNN1_3Conv


def test_cnn1_parameter_count():
    model = CNN1()
    total = sum(p.numel() for p in model.parameters())
    assert total == 537960, f'Expected 537960, got {total}'


def test_cnn1_forward():
    model = CNN1()
    x = torch.randn(2, 1, 28, 28)
    y = model(x)
    assert y.shape == (2, 10)


def test_cnn1_functional_forward():
    model = CNN1()
    x = torch.randn(2, 1, 28, 28)
    theta_hat = torch.randn(model.get_total_params(), requires_grad=True)
    y = model.functional_forward(x, theta_hat)
    y.sum().backward()
    assert theta_hat.grad is not None


def test_cnn1_3conv_functional_forward():
    model = CNN1_3Conv()
    x = torch.randn(2, 1, 28, 28)
    theta_hat = torch.randn(model.get_total_params(), requires_grad=True)
    y = model.functional_forward(x, theta_hat)
    y.sum().backward()
    assert theta_hat.grad is not None
