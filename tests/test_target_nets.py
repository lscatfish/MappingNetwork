import torch

from mapping_network.factory import build_target_net
from mapping_network.target_nets.cnn1 import CNN1
from mapping_network.target_nets.cnn1_3conv import CNN1_3Conv
from mapping_network.target_nets.cnn2 import CNN2
from mapping_network.target_nets.lrd_config import LRDConfig


def test_cnn2_parameter_count(device):
    model = CNN2().to(device)
    total = sum(p.numel() for p in model.parameters())
    assert total == 108610, f'Expected 108610, got {total}'


def test_cnn2_forward(device):
    model = CNN2().to(device)
    x = torch.randn(4, 1, 28, 28, device=device)
    y = model(x)
    assert y.shape == (4, 10)


def test_cnn2_functional_forward(device):
    """验证函数式前向输出与模块前向一致，且梯度可回传至 theta_hat。"""
    model = CNN2().to(device)
    x = torch.randn(2, 1, 28, 28, device=device)
    theta_hat = torch.randn(model.get_total_params(), device=device, requires_grad=True)
    y = model.functional_forward(x, theta_hat)
    loss = y.sum()
    loss.backward()
    assert theta_hat.grad is not None
    assert theta_hat.grad.shape == (model.get_total_params(),)


def test_cnn1_parameter_count(device):
    model = CNN1().to(device)
    total = sum(p.numel() for p in model.parameters())
    assert total == 537960, f'Expected 537960, got {total}'


def test_cnn1_forward(device):
    model = CNN1().to(device)
    x = torch.randn(2, 1, 28, 28, device=device)
    y = model(x)
    assert y.shape == (2, 10)


def test_cnn1_functional_forward(device):
    model = CNN1().to(device)
    x = torch.randn(2, 1, 28, 28, device=device)
    theta_hat = torch.randn(model.get_total_params(), device=device, requires_grad=True)
    y = model.functional_forward(x, theta_hat)
    y.sum().backward()
    assert theta_hat.grad is not None


def test_cnn1_3conv_parameter_count(device):
    model = CNN1_3Conv().to(device)
    total = sum(p.numel() for p in model.parameters())
    # 16*25+16 + 16*32*25+32 + 32*64*9+64 + 64*10+10 = 416+12832+18496+650 = 32394
    assert total == 32394, f'Expected 32394, got {total}'


def test_cnn1_3conv_forward(device):
    model = CNN1_3Conv().to(device)
    x = torch.randn(2, 1, 28, 28, device=device)
    y = model(x)
    assert y.shape == (2, 10)


def test_cnn1_3conv_functional_forward(device):
    model = CNN1_3Conv().to(device)
    x = torch.randn(2, 1, 28, 28, device=device)
    theta_hat = torch.randn(model.get_total_params(), device=device, requires_grad=True)
    y = model.functional_forward(x, theta_hat)
    y.sum().backward()
    assert theta_hat.grad is not None


def test_cnn2_lrd_reduces_params(device):
    net_full = CNN2(lrd_config=LRDConfig(enabled=False)).to(device)
    net_lrd = CNN2(lrd_config=LRDConfig(enabled=True, default_rank=10)).to(device)
    assert net_lrd.get_total_params() < net_full.get_total_params()


def test_cnn2_per_layer_lrd_override(device):
    """layer_enabled 优先级高于全局 enabled=True。"""
    target = CNN2(lrd_config={'enabled': True, 'layer_enabled': {'fc2': False}}).to(device)
    for s in target.get_param_slices():
        name = s.name if s.kind == 'full' else s.weight_name
        if name.startswith('fc2'):
            assert s.kind == 'full', f'{name} should be full, got {s.kind}'


def test_cnn2_lrd_functional_matches_module(device):
    net = CNN2(lrd_config=LRDConfig(enabled=True, default_rank=10)).to(device)
    x = torch.randn(2, 1, 28, 28, device=device)
    theta = torch.randn(net.get_total_params(), device=device, requires_grad=True)
    y_func = net.functional_forward(x, theta)
    y_mod = net(x)
    assert y_func.shape == y_mod.shape


def test_target_net_assemble_params(device):
    net = build_target_net('cnn2').to(device)
    group_names = net.get_group_names()
    group_sizes = [net.get_group_param_size(name) for name in group_names]
    outputs = {
        name: torch.randn(size, device=device) for name, size in zip(group_names, group_sizes)
    }
    theta = net.assemble_params(outputs)
    assert theta.shape == (sum(group_sizes),)
