import torch

from mapping_network.factory import build_generator, build_target_net
from mapping_network.generators.cnn import CNNMappingNetwork
from mapping_network.generators.linear import LinearMappingNetwork
from mapping_network.generators.multilayer_linear import MultiLayerLinearMappingNetwork


def test_build_cnn1_with_lrd(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    net = build_target_net('cnn1', {'enabled': True, 'default_rank': 10})
    gen = build_generator(
        {'type': 'linear', 'latent_dim': 2072, 'alpha': 0.01},
        target_total_params=net.get_total_params(),
        device=device,
    )
    theta = gen()
    assert theta.shape[0] < 537_960
    assert theta.device.type == device


def test_build_generator_with_config_dict(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    gen = build_generator(
        {'type': 'linear', 'latent_dim': 8, 'alpha': 0.01},
        target_total_params=100,
        device=device,
    )
    assert isinstance(gen, LinearMappingNetwork)
    assert gen().shape == (100,)


def test_build_multilayer_linear_generator(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    gen = build_generator(
        {
            'type': 'multilayer_linear',
            'latent_dim': 8,
            'alpha': 0.01,
            'hidden_dim': 16,
            'num_hidden': 2,
        },
        target_total_params=50,
        device=device,
    )
    assert isinstance(gen, MultiLayerLinearMappingNetwork)
    assert gen().shape == (50,)


def test_build_cnn_generator(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    gen = build_generator(
        {'type': 'cnn', 'latent_dim': 8, 'alpha': 0.01, 'feature_size': 4, 'channels': (8, 4)},
        target_total_params=50,
        device=device,
    )
    assert isinstance(gen, CNNMappingNetwork)
    assert gen().shape == (50,)
