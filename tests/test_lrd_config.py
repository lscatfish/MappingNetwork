from mapping_network.target_nets.lrd_config import LRDConfig


def test_lrd_config_defaults():
    cfg = LRDConfig()
    assert cfg.enabled == 'auto'
    assert cfg.default_rank == 10
    assert cfg.layer_ranks == {}
    assert cfg.auto_enable_threshold == 200_000


def test_lrd_config_override():
    cfg = LRDConfig(enabled=True, default_rank=20, layer_ranks={'fc1': 15})
    assert cfg.enabled is True
    assert cfg.default_rank == 20
    assert cfg.layer_ranks == {'fc1': 15}
