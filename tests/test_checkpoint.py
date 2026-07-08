"""Checkpoint save/load reconstruction tests."""

import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from mapping_network.factory import build_generator, build_target_net
from mapping_network.generators.linear import LinearMappingNetwork
from mapping_network.mapping.loss import MappingLoss
from mapping_network.trainer.lwt import LWTTrainer
from mapping_network.trainer.slvt import SLVTTrainer


def make_one_batch_loader(device):
    x = torch.randn(1, 1, 28, 28, device=device)
    y = torch.tensor([0], device=device)
    return DataLoader(TensorDataset(x.cpu(), y.cpu()), batch_size=1)


def test_slvt_checkpoint_reconstruction(device):
    """SLVT checkpoint 保存后能重建并复现相同 logits。"""
    target_net = build_target_net('cnn2').to(device)
    mapping = LinearMappingNetwork(target_net.get_total_params(), 64, device=device, w_seed=42)
    loss_fn = MappingLoss().to(device)
    loader = make_one_batch_loader(device)

    trainer = SLVTTrainer(
        mapping,
        target_net,
        loss_fn,
        loader,
        loader,
        epochs=1,
        device=device,
        log_interval=1,
        checkpoint_dir='/tmp/test_ckpt_slvt',
        experiment_name='test_slvt',
        checkpoint_metadata={
            'target_net': 'cnn2',
            'training_strategy': 'slvt',
            'generator_type': 'linear',
            'latent_dim': 64,
            'alpha': 0.01,
            'sigma_noise': 0.01,
            'lrd_config': None,
            'generator_config': {
                'target_total_params': target_net.get_total_params(),
                'latent_dim': 64,
                'alpha': 0.01,
                'w_seed': 42,
            },
        },
        save_interval=0,
    )
    trainer.train()

    # 训练后前向得到参考 logits
    target_net.eval()
    mapping.eval()
    theta_ref = mapping()
    x, y = next(iter(loader))
    x = x.to(device)
    logits_ref = target_net.functional_forward(x, theta_ref)

    ckpt_path = os.path.join('/tmp/test_ckpt_slvt', 'test_slvt_final.pth')
    ckpt = torch.load(ckpt_path, map_location=device)

    # 重建
    target_rebuilt = build_target_net(ckpt['target_net'], ckpt.get('lrd_config')).to(device)
    mapping_rebuilt = build_generator(
        ckpt.get('generator_type', 'linear'),
        ckpt['generator_config'],
        device,
    )
    mapping_rebuilt.load_persistent_state_dict(ckpt['generator_state_dict'])
    mapping_rebuilt.eval()
    target_rebuilt.eval()

    theta_rebuilt = mapping_rebuilt()
    logits_rebuilt = target_rebuilt.functional_forward(x, theta_rebuilt)
    assert torch.allclose(logits_ref, logits_rebuilt, atol=1e-6)


def test_lwt_checkpoint_reconstruction(device):
    """LWT checkpoint 保存后能重建并复现相同 logits。"""
    target_net = build_target_net('cnn2').to(device)
    loss_fn = MappingLoss().to(device)
    loader = make_one_batch_loader(device)
    layer_generators = {
        'conv1': {'type': 'linear', 'latent_dim': 16, 'alpha': 0.01, 'w_seed': 1},
        'conv2': {'type': 'linear', 'latent_dim': 16, 'alpha': 0.01, 'w_seed': 2},
        'fc1': {'type': 'linear', 'latent_dim': 16, 'alpha': 0.01, 'w_seed': 3},
        'fc2': {'type': 'linear', 'latent_dim': 16, 'alpha': 0.01, 'w_seed': 4},
    }
    trainer = LWTTrainer(
        target_net,
        loss_fn,
        layer_generators,
        train_loader=loader,
        test_loader=loader,
        epochs=1,
        device=device,
        log_interval=1,
        checkpoint_dir='/tmp/test_ckpt_lwt',
        experiment_name='test_lwt',
        checkpoint_metadata={
            'target_net': 'cnn2',
            'training_strategy': 'lwt',
            'lrd_config': None,
            'sigma_noise': 0.01,
        },
        save_interval=0,
    )
    trainer.train()

    # 训练后前向得到参考 logits
    target_net.eval()
    for m in trainer.layer_mappings.values():
        m.eval()
    theta_ref = trainer._generate_all_theta()
    x, y = next(iter(loader))
    x = x.to(device)
    logits_ref = target_net.functional_forward(x, theta_ref)

    ckpt_path = os.path.join('/tmp/test_ckpt_lwt', 'test_lwt_final.pth')
    ckpt = torch.load(ckpt_path, map_location=device)

    # 重建
    target_rebuilt = build_target_net(ckpt['target_net'], ckpt.get('lrd_config')).to(device)
    layer_mappings = torch.nn.ModuleDict()
    for name, gen_cfg in ckpt['layer_generator_configs'].items():
        group_size = target_rebuilt.get_group_param_size(name)
        gen_type = gen_cfg.get('type', 'linear')
        config = {k: v for k, v in gen_cfg.items() if k != 'type'}
        config['target_total_params'] = group_size
        mapping = build_generator(gen_type, config, device)
        mapping.load_persistent_state_dict(ckpt['state_dict'][name])
        layer_mappings[name] = mapping

    group_order = ckpt.get('layer_group_order', list(layer_mappings.keys()))
    theta_rebuilt = target_rebuilt.assemble_params({name: layer_mappings[name]() for name in group_order})
    logits_rebuilt = target_rebuilt.functional_forward(x, theta_rebuilt)
    assert torch.allclose(logits_ref, logits_rebuilt, atol=1e-6)
