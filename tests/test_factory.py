import torch

from mapping_network.factory import build_generator, build_target_net
from mapping_network.generators.linear import LinearMappingNetwork


def test_build_cnn1_with_lrd(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    net = build_target_net('cnn1', {'enabled': True, 'default_rank': 10})
    gen = build_generator('linear', {'target_total_params': net.get_total_params(), 'latent_dim': 2072, 'alpha': 0.01}, device)
    theta = gen()
    assert theta.shape[0] < 537_960
    assert theta.device.type == device


def test_build_generator_with_config_dict(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    gen = build_generator('linear', {'target_total_params': 100, 'latent_dim': 8, 'alpha': 0.01}, device)
    assert isinstance(gen, LinearMappingNetwork)
    assert gen().shape == (100,)
