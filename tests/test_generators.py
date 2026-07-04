import pytest
import torch

from mapping_network.generators.base import ParameterGenerator
from mapping_network.generators.linear import LinearMappingNetwork


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
