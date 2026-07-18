from mapping_network.target_nets.cnn1 import CNN1
from mapping_network.target_nets.cnn1_3conv import CNN1_3Conv
from mapping_network.target_nets.cnn2 import CNN2
from mapping_network.target_nets.lrd_config import LRDConfig

# 触发所有 generator 子类的装饰器执行，填充 GENERATOR_REGISTRY
from mapping_network.generators import base as _generator_base
from mapping_network.generators import linear  # noqa: F401


TARGET_NET_MAP = {
    'cnn1': CNN1,
    'cnn2': CNN2,
    'cnn1_3conv': CNN1_3Conv,
}

GENERATOR_MAP = _generator_base.GENERATOR_REGISTRY


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

    Factory 只负责：
    1. 根据 generator_config['type'] 从注册表查找对应类。
    2. 将 target_total_params 和 device 注入 kwargs。
    3. 其余键值原样透传给具体 generator 类，由 generator 自行解析。

    新增 generator 类型时只需：
    - 创建新文件并继承 ParameterGenerator
    - 使用 @register_generator('name') 装饰器
    - 在 mapping_network/generators/__init__.py 中 import 该类（触发注册）
    无需修改 factory.py。

    Args:
        generator_config: Dict with keys:
            - 'type': generator type name from GENERATOR_REGISTRY
            - 'latent_dim': latent dimension d
            - 'alpha': modulation coefficient (default 0.01)
            - other generator-specific parameters (e.g. 'w_seed', 'layer_name')
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
    # 透传配置，仅排除 factory 职责内的键
    kwargs = dict(generator_config)
    kwargs.pop('type', None)
    kwargs['target_total_params'] = target_total_params
    kwargs['device'] = device
    # lrd_rank/lrd_enabled 是 LWT 目标网络配置，不属于 generator 参数
    kwargs.pop('lrd_rank', None)
    kwargs.pop('lrd_enabled', None)

    return cls(**kwargs)
