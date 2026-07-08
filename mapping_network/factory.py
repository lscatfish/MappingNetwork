from mapping_network.generators.cnn import CNNMappingNetwork
from mapping_network.generators.linear import LinearMappingNetwork
from mapping_network.generators.multilayer_linear import MultiLayerLinearMappingNetwork
from mapping_network.target_nets.cnn1 import CNN1
from mapping_network.target_nets.cnn1_3conv import CNN1_3Conv
from mapping_network.target_nets.cnn2 import CNN2
from mapping_network.target_nets.lrd_config import LRDConfig

TARGET_NET_MAP = {
    'cnn1': CNN1,
    'cnn2': CNN2,
    'cnn1_3conv': CNN1_3Conv,
}

GENERATOR_MAP = {
    'linear': LinearMappingNetwork,
    'multilayer_linear': MultiLayerLinearMappingNetwork,
    'cnn': CNNMappingNetwork,
}


def build_target_net(target_name: str, lrd_config: dict | None = None):
    if target_name not in TARGET_NET_MAP:
        raise ValueError(f'Unknown target net: {target_name}')
    cfg = LRDConfig(**lrd_config) if lrd_config else LRDConfig()
    return TARGET_NET_MAP[target_name](lrd_config=cfg)


def build_generator(generator_type: str, generator_config: dict, device: str):
    if generator_type not in GENERATOR_MAP:
        raise ValueError(f'Unknown generator type: {generator_type}')
    config = dict(generator_config)
    config.pop('device', None)
    return GENERATOR_MAP[generator_type](**config, device=device)
