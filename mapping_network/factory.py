from mapping_network.generators.adaptive_dim import AdaptiveDimMappingNetwork
from mapping_network.generators.hadamard import HadamardMappingNetwork
from mapping_network.generators.kron_structured import KronStructuredMappingNetwork
from mapping_network.generators.kron_weight import KronWeightMappingNetwork
from mapping_network.generators.linear import LinearMappingNetwork
from mapping_network.generators.manifold_regularized import ManifoldRegularizedMappingNetwork
from mapping_network.generators.pca_linear import PCABasedMappingNetwork
from mapping_network.generators.superposition import SuperpositionMappingNetwork
from mapping_network.generators.tt_structured import TTStructuredMappingNetwork
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
    'hadamard': HadamardMappingNetwork,
    'kron_structured': KronStructuredMappingNetwork,
    'tt_structured': TTStructuredMappingNetwork,
    'kron_weight': KronWeightMappingNetwork,
    'pca': PCABasedMappingNetwork,
    'adaptive_dim': AdaptiveDimMappingNetwork,
    'manifold_reg': ManifoldRegularizedMappingNetwork,
    'superposition': SuperpositionMappingNetwork,
}


def build_target_net(target_name: str, lrd_config: dict | None = None):
    if target_name not in TARGET_NET_MAP:
        raise ValueError(f'Unknown target net: {target_name}')
    cfg = LRDConfig(**lrd_config) if lrd_config else LRDConfig()
    return TARGET_NET_MAP[target_name](lrd_config=cfg)


def build_generator(
    generator_config: dict,
    target_total_params: int,
    device: str = 'cpu',
):
    """Build a parameter generator from a config dict.

    Args:
        generator_config: Dict with keys:
            - 'type': generator type name from GENERATOR_MAP
            - 'latent_dim': latent dimension d
            - 'alpha': modulation coefficient (default 0.01)
            - 'w_seed': seed for W_fixed reconstruction (default 12345)
            - other generator-specific parameters
        target_total_params: Total number of target network parameters (compressed).
        device: Device string.

    Returns:
        ParameterGenerator instance.

    Example:
        >>> config = {'type': 'linear', 'latent_dim': 2048, 'alpha': 0.01}
        >>> gen = build_generator(config, target_total_params=108610, device='cuda')
    """
    gen_type = generator_config.get('type', 'linear')
    if gen_type not in GENERATOR_MAP:
        raise ValueError(f'Unknown generator type: {gen_type}')

    cls = GENERATOR_MAP[gen_type]
    # 提取通用参数
    kwargs = {
        'target_total_params': target_total_params,
        'latent_dim': generator_config['latent_dim'],
        'alpha': generator_config.get('alpha', 0.01),
        'device': device,
        'w_seed': generator_config.get('w_seed', 12345),
    }
    # 提取生成器特定参数（排除通用参数和 LWT 特有的 lrd_rank/lrd_enabled）
    skip_keys = {'type', 'latent_dim', 'alpha', 'w_seed', 'lrd_rank', 'lrd_enabled'}
    for k, v in generator_config.items():
        if k not in skip_keys:
            kwargs[k] = v

    return cls(**kwargs)
