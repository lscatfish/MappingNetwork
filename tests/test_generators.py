import pytest
import torch
from mapping_network.generators.base import ParameterGenerator


def test_parameter_generator_is_abstract():
    with pytest.raises(TypeError):
        ParameterGenerator()
