import pytest
import torch

from mapping_network.generators.base import ParameterGenerator
from mapping_network.generators.cnn import CNNMappingNetwork
from mapping_network.generators.linear import LinearMappingNetwork
from mapping_network.generators.multilayer_linear import MultiLayerLinearMappingNetwork


def test_parameter_generator_is_abstract():
    with pytest.raises(TypeError):
        ParameterGenerator()


def test_linear_mapping_network_shape_and_trainable(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    gen = LinearMappingNetwork(100, 8, alpha=0.01, device=device)
    theta = gen()
    assert theta.shape == (100,)
    assert theta.device.type == device
    assert gen.trainable_params() == 8
    assert not gen.W_fixed.requires_grad
    assert gen.z.requires_grad


def test_linear_mapping_network_aux_methods(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    gen = LinearMappingNetwork(100, 8, alpha=0.01, device=device)
    theta_noisy = gen.noisy_forward(0.01)
    assert theta_noisy.shape == (100,)
    assert theta_noisy.device.type == device
    assert theta_noisy.requires_grad

    l_smooth = gen.smooth_loss()
    assert l_smooth.shape == ()
    assert l_smooth.requires_grad

    l_align = gen.align_loss()
    assert l_align.shape == ()
    assert l_align.requires_grad


def test_persistent_state_dict_only_trainable():
    gen = LinearMappingNetwork(20, 4, alpha=0.01, device='cpu')
    state = gen.persistent_state_dict()
    assert 'z' in state
    assert 'W_fixed' not in state
    assert all(v.requires_grad for v in state.values())


def test_load_persistent_state_dict_restores_trainable():
    gen = LinearMappingNetwork(20, 4, alpha=0.01, device='cpu')
    original_z = gen.z.detach().clone()
    gen.z.data.fill_(0.0)
    missing, unexpected = gen.load_persistent_state_dict({'z': original_z})
    assert torch.allclose(gen.z, original_z)


def test_linear_mapping_network_w_seed_reproducible():
    gen1 = LinearMappingNetwork(20, 4, alpha=0.01, device='cpu', w_seed=123)
    gen2 = LinearMappingNetwork(20, 4, alpha=0.01, device='cpu', w_seed=123)
    assert torch.allclose(gen1.W_fixed, gen2.W_fixed)
    gen3 = LinearMappingNetwork(20, 4, alpha=0.01, device='cpu', w_seed=456)
    assert not torch.allclose(gen1.W_fixed, gen3.W_fixed)


def test_multilayer_linear_mapping_network(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    gen = MultiLayerLinearMappingNetwork(50, 8, alpha=0.01, hidden_dim=16, num_hidden=2, device=device)
    theta = gen()
    assert theta.shape == (50,)
    assert theta.device.type == device
    assert gen.trainable_params() > 0

    theta_noisy = gen.noisy_forward(0.01)
    assert theta_noisy.shape == (50,)
    assert theta_noisy.requires_grad

    l_smooth = gen.smooth_loss()
    assert l_smooth.shape == ()
    assert l_smooth.requires_grad

    l_align = gen.align_loss()
    assert l_align.shape == ()
    assert l_align.requires_grad


def test_cnn_mapping_network(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    gen = CNNMappingNetwork(50, 8, alpha=0.01, feature_size=4, channels=(8, 4), device=device)
    theta = gen()
    assert theta.shape == (50,)
    assert theta.device.type == device
    assert gen.trainable_params() > 0

    theta_noisy = gen.noisy_forward(0.01)
    assert theta_noisy.shape == (50,)
    assert theta_noisy.requires_grad

    l_smooth = gen.smooth_loss()
    assert l_smooth.shape == ()
    assert l_smooth.requires_grad

    l_align = gen.align_loss()
    assert l_align.shape == ()
    assert l_align.requires_grad
