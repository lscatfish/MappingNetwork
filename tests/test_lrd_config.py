import pytest

from mapping_network.target_nets.lrd_config import LRDConfig


def test_lrd_config_defaults():
    cfg = LRDConfig()
    assert cfg.enabled == 'auto'
    assert cfg.default_rank == 10
    assert cfg.layer_ranks == {}
    assert cfg.layer_enabled == {}
    assert cfg.auto_enable_threshold == 200_000


def test_lrd_config_override():
    cfg = LRDConfig(enabled=True, default_rank=20, layer_ranks={'fc1': 15})
    assert cfg.enabled is True
    assert cfg.default_rank == 20
    assert cfg.layer_ranks == {'fc1': 15}


def test_lrd_config_layer_enabled():
    cfg = LRDConfig(layer_enabled={'fc1': False, 'fc2': 'true', 'fc3': 'auto'})
    assert cfg.layer_enabled == {'fc1': False, 'fc2': True, 'fc3': 'auto'}


def test_lrd_config_invalid_layer_enabled():
    with pytest.raises(ValueError):
        LRDConfig(layer_enabled={'fc1': 'invalid'})
