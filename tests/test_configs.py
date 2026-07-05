"""
为每个 YAML 配置跑最小冒烟测试：batch_size=1 的一张图前向+反向，
验证所有张量都在目标设备、不 OOM、可训练参数数量正确。
"""

import os

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from mapping_network.factory import build_target_net
from mapping_network.generators.linear import LinearMappingNetwork
from mapping_network.mapping.loss import MappingLoss
from mapping_network.scripts.train import _merge_lwt_lrd_config
from mapping_network.trainer.lwt import LWTTrainer
from mapping_network.trainer.slvt import SLVTTrainer

# 包含 configs/ 下全部 YAML，按类型区分期望。
MAPPING_CONFIGS = [
    'configs/cnn1_slvt.yaml',
    'configs/cnn1_lwt.yaml',
    'configs/cnn1_3conv_slvt.yaml',
    'configs/cnn2_slvt.yaml',
    'configs/cnn2_lwt.yaml',
]

BASELINE_CONFIGS = [
    'configs/cnn1_baseline.yaml',
    'configs/cnn1_3conv_baseline.yaml',
    'configs/cnn2_baseline.yaml',
]

ALL_CONFIGS = MAPPING_CONFIGS + BASELINE_CONFIGS


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_one_batch_loader(device):
    x = torch.randn(1, 1, 28, 28, device=device)
    y = torch.tensor([0], device=device)
    return DataLoader(TensorDataset(x.cpu(), y.cpu()), batch_size=1)


def test_lwt_layer_lrd_config_merge():
    """Verify LWT per-layer lrd_rank/lrd_enabled merge into global LRDConfig."""
    cfg = load_cfg('configs/cnn1_lwt.yaml')
    assert cfg['training_strategy'] == 'lwt'

    lrd_config = _merge_lwt_lrd_config(cfg, cfg.get('lrd', {}))

    # Verify merge
    assert 'fc1' in lrd_config['layer_ranks']
    assert lrd_config['layer_ranks']['fc1'] == 10
    # No layer_enabled in cnn1_lwt.yaml; should be empty dict
    assert lrd_config.get('layer_enabled', {}) == {}

    target_net = build_target_net(cfg['target_net'], lrd_config)
    slices = target_net.get_param_slices()
    fc1_slices = [s for s in slices if (s.kind == 'lrd' and s.weight_name.split('.')[0] == 'fc1') or (s.kind == 'full' and s.name.split('.')[0] == 'fc1')]
    assert len(fc1_slices) == 1
    assert fc1_slices[0].kind == 'lrd', 'fc1 should use LRD with layer-specified rank'


def expected_mapping_trainable_params(cfg):
    """可训练参数 = 所有 z 维度之和 + 3 个 lambda。"""
    if cfg['training_strategy'] == 'slvt':
        z_dims = cfg['latent_dim']
    else:
        z_dims = sum(layer['latent_dim'] for layer in cfg['layer_generators'].values())
    return z_dims + 3  # lambda_st, lambda_sm, lambda_al


@pytest.mark.parametrize('cfg_path', MAPPING_CONFIGS)
def test_mapping_config_one_batch(cfg_path, device):
    cfg = load_cfg(cfg_path)

    lrd_config = _merge_lwt_lrd_config(cfg, cfg.get('lrd', {}))

    target_net = build_target_net(cfg['target_net'], lrd_config).to(device)
    loss_fn = MappingLoss(sigma_noise=cfg.get('sigma_noise', 0.01)).to(device)
    loader = make_one_batch_loader(device)

    if cfg['training_strategy'] == 'slvt':
        mapping = LinearMappingNetwork(
            target_net.get_total_params(),
            cfg['latent_dim'],
            alpha=cfg.get('alpha', 0.01),
            device=device,
        )
        trainer = SLVTTrainer(
            mapping,
            target_net,
            loss_fn,
            loader,
            loader,
            lr=cfg['lr'],
            weight_decay=cfg.get('weight_decay', 0.0001),
            epochs=1,
            min_lr=cfg.get('min_lr', 1e-5),
            device=device,
            log_interval=1,
            checkpoint_dir='/tmp/test_configs',
            experiment_name=f'test_config_{os.path.basename(cfg_path).replace(".yaml", "")}',
            save_interval=0,
            checkpoint_metadata={
                'target_net': cfg['target_net'],
                'training_strategy': 'slvt',
                'generator_type': 'linear',
                'latent_dim': cfg['latent_dim'],
                'alpha': cfg.get('alpha', 0.01),
                'sigma_noise': cfg.get('sigma_noise', 0.01),
                'lrd_config': lrd_config,
            },
        )
    else:
        trainer = LWTTrainer(
            target_net,
            loss_fn,
            cfg['layer_generators'],
            train_loader=loader,
            test_loader=loader,
            lr=cfg['lr'],
            weight_decay=cfg.get('weight_decay', 0.0001),
            epochs=1,
            min_lr=cfg.get('min_lr', 1e-5),
            device=device,
            log_interval=1,
            checkpoint_dir='/tmp/test_configs',
            experiment_name=f'test_config_{os.path.basename(cfg_path).replace(".yaml", "")}',
            save_interval=0,
            checkpoint_metadata={
                'target_net': cfg['target_net'],
                'training_strategy': 'lwt',
                'lrd_config': lrd_config,
                'sigma_noise': cfg.get('sigma_noise', 0.01),
            },
        )

    # 验证所有相关张量/参数在目标设备
    assert next(target_net.parameters()).device.type == device.split(':')[0]
    assert loss_fn.lambda_st.device.type == device.split(':')[0]
    assert loss_fn.lambda_sm.device.type == device.split(':')[0]
    assert loss_fn.lambda_al.device.type == device.split(':')[0]

    if cfg['training_strategy'] == 'slvt':
        assert mapping.z.device.type == device.split(':')[0]
        assert mapping.W_fixed.device.type == device.split(':')[0]
    else:
        for mapping in trainer.layer_mappings.values():
            assert mapping.z.device.type == device.split(':')[0]
            assert mapping.W_fixed.device.type == device.split(':')[0]

    # 跑一次 epoch（只有 1 个 batch）
    trainer.train_epoch(1)

    # 验证可训练参数数量
    trainable = sum(
        p.numel() for p in trainer.optimizer.param_groups[0]['params'] if p.requires_grad
    )
    assert trainable == expected_mapping_trainable_params(cfg), (
        f'{cfg_path}: trainable {trainable} != expected {expected_mapping_trainable_params(cfg)}'
    )

    # 验证 z 被更新（有梯度且不是 nan）
    if cfg['training_strategy'] == 'slvt':
        assert mapping.z.grad is not None
        assert not torch.isnan(mapping.z.grad).any()
    else:
        for mapping in trainer.layer_mappings.values():
            assert mapping.z.grad is not None
            assert not torch.isnan(mapping.z.grad).any()


@pytest.mark.parametrize('cfg_path', BASELINE_CONFIGS)
def test_baseline_config_one_batch(cfg_path, device):
    cfg = load_cfg(cfg_path)
    target_net = build_target_net(cfg['target']).to(device)
    loader = make_one_batch_loader(device)

    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    logits = target_net(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    assert next(target_net.parameters()).device.type == device.split(':')[0]
    assert loss.item() > 0
    assert any(p.grad is not None for p in target_net.parameters())
