"""验证 trainer / evaluate 与新的 generator 类型（multilayer_linear）的互操作性。

证明：
1. SLVT + multilayer_linear 可训练一个 epoch 且梯度正确回传。
2. LWT + multilayer_linear 可训练一个 epoch 且梯度正确回传。
3. checkpoint 保存后可以 load_persistent_state_dict 重建。
"""

import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from mapping_network.factory import build_generator, build_target_net
from mapping_network.mapping.loss import MappingLoss
from mapping_network.trainer.lwt import LWTTrainer
from mapping_network.trainer.slvt import SLVTTrainer


def make_one_batch_loader(device):
    x = torch.randn(1, 1, 28, 28, device=device)
    y = torch.tensor([0], device=device)
    return DataLoader(TensorDataset(x.cpu(), y.cpu()), batch_size=1)


def test_slvt_multilayer_linear_trains(device):
    """SLVT + multilayer_linear 一个 epoch 后 z 被更新。"""
    target_net = build_target_net('cnn2').to(device)
    gen_config = {
        'type': 'multilayer_linear',
        'latent_dim': 32,
        'alpha': 0.01,
        'hidden_dim': 64,
        'num_layers': 2,
    }
    mapping = build_generator(gen_config, target_net.get_total_params(), device=device)
    z_before = mapping.z.clone().detach()

    loss_fn = MappingLoss().to(device)
    loader = make_one_batch_loader(device)

    trainer = SLVTTrainer(
        mapping, target_net, loss_fn, loader, loader,
        epochs=1, device=device, log_interval=1,
        checkpoint_dir='/tmp/test_ext_slvt',
        experiment_name='test_ext_slvt',
        checkpoint_metadata={
            'target_net': 'cnn2', 'training_strategy': 'slvt',
            'gen_config': gen_config,
            'latent_dim': 32, 'alpha': 0.01, 'sigma_noise': 0.01, 'lrd_config': None,
        },
        save_interval=0,
    )
    trainer.train()

    z_after = mapping.z.clone().detach()
    assert not torch.allclose(z_before, z_after), 'z should be updated after training'


def test_lwt_multilayer_linear_trains(device):
    """LWT + multilayer_linear 一个 epoch 后 z 被更新。"""
    target_net = build_target_net('cnn2').to(device)
    loss_fn = MappingLoss().to(device)
    loader = make_one_batch_loader(device)

    layer_generators = {
        'conv1': {'type': 'multilayer_linear', 'latent_dim': 16, 'alpha': 0.01, 'hidden_dim': 32, 'num_layers': 1},
        'conv2': {'type': 'multilayer_linear', 'latent_dim': 16, 'alpha': 0.01, 'hidden_dim': 32, 'num_layers': 1},
        'fc1': {'type': 'multilayer_linear', 'latent_dim': 16, 'alpha': 0.01, 'hidden_dim': 32, 'num_layers': 1},
        'fc2': {'type': 'multilayer_linear', 'latent_dim': 16, 'alpha': 0.01, 'hidden_dim': 32, 'num_layers': 1},
    }
    trainer = LWTTrainer(
        target_net, loss_fn, layer_generators,
        train_loader=loader, test_loader=loader,
        epochs=1, device=device, log_interval=1,
        checkpoint_dir='/tmp/test_ext_lwt',
        experiment_name='test_ext_lwt',
        checkpoint_metadata={
            'target_net': 'cnn2', 'training_strategy': 'lwt',
            'lrd_config': None, 'sigma_noise': 0.01,
        },
        save_interval=0,
    )

    z_before = {}
    for name, m in trainer.layer_mappings.items():
        z_before[name] = m.z.clone().detach()

    trainer.train()

    for name, m in trainer.layer_mappings.items():
        z_after = m.z.clone().detach()
        assert not torch.allclose(z_before[name], z_after), f'z for {name} should be updated'


def test_multilayer_linear_checkpoint_reconstruction(device):
    """multilayer_linear checkpoint 保存后能用 load_persistent_state_dict 重建。"""
    target_net = build_target_net('cnn2').to(device)
    gen_config = {
        'type': 'multilayer_linear',
        'latent_dim': 32,
        'alpha': 0.01,
        'hidden_dim': 64,
        'num_layers': 2,
    }
    mapping = build_generator(gen_config, target_net.get_total_params(), device=device)
    loss_fn = MappingLoss().to(device)
    loader = make_one_batch_loader(device)

    trainer = SLVTTrainer(
        mapping, target_net, loss_fn, loader, loader,
        epochs=1, device=device, log_interval=1,
        checkpoint_dir='/tmp/test_ext_ckpt',
        experiment_name='test_ext_ckpt',
        checkpoint_metadata={
            'target_net': 'cnn2', 'training_strategy': 'slvt',
            'gen_config': gen_config,
            'latent_dim': 32, 'alpha': 0.01, 'sigma_noise': 0.01, 'lrd_config': None,
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

    ckpt_path = os.path.join('/tmp/test_ext_ckpt', 'test_ext_ckpt_final.pth')
    ckpt = torch.load(ckpt_path, map_location=device)

    # 重建：使用 gen_config（不含 w_seed）
    target_rebuilt = build_target_net(ckpt['target_net'], ckpt.get('lrd_config')).to(device)
    mapping_rebuilt = build_generator(
        ckpt['gen_config'],
        target_total_params=target_rebuilt.get_total_params(),
        device=device,
    )
    mapping_rebuilt.load_persistent_state_dict(ckpt['state_dict'])
    mapping_rebuilt.eval()
    target_rebuilt.eval()

    theta_rebuilt = mapping_rebuilt()
    logits_rebuilt = target_rebuilt.functional_forward(x, theta_rebuilt)
    assert torch.allclose(logits_ref, logits_rebuilt, atol=1e-5), (
        'multilayer_linear checkpoint reconstruction failed'
    )


def test_lwt_multilayer_linear_checkpoint_reconstruction(device):
    """LWT + multilayer_linear checkpoint 保存后能用 load_persistent_state_dict 重建。"""
    target_net = build_target_net('cnn2').to(device)
    loss_fn = MappingLoss().to(device)
    loader = make_one_batch_loader(device)

    layer_generators = {
        'conv1': {'type': 'multilayer_linear', 'latent_dim': 16, 'alpha': 0.01, 'hidden_dim': 32, 'num_layers': 1},
        'conv2': {'type': 'multilayer_linear', 'latent_dim': 16, 'alpha': 0.01, 'hidden_dim': 32, 'num_layers': 1},
        'fc1': {'type': 'multilayer_linear', 'latent_dim': 16, 'alpha': 0.01, 'hidden_dim': 32, 'num_layers': 1},
        'fc2': {'type': 'multilayer_linear', 'latent_dim': 16, 'alpha': 0.01, 'hidden_dim': 32, 'num_layers': 1},
    }
    trainer = LWTTrainer(
        target_net, loss_fn, layer_generators,
        train_loader=loader, test_loader=loader,
        epochs=1, device=device, log_interval=1,
        checkpoint_dir='/tmp/test_ext_lwt_ckpt',
        experiment_name='test_ext_lwt_ckpt',
        checkpoint_metadata={
            'target_net': 'cnn2', 'training_strategy': 'lwt',
            'lrd_config': None, 'sigma_noise': 0.01,
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

    ckpt_path = os.path.join('/tmp/test_ext_lwt_ckpt', 'test_ext_lwt_ckpt_final.pth')
    ckpt = torch.load(ckpt_path, map_location=device)

    # 重建：使用 layer_generator_configs + layer_name
    target_rebuilt = build_target_net(ckpt['target_net'], ckpt.get('lrd_config')).to(device)
    layer_mappings = torch.nn.ModuleDict()
    for name, gen_cfg in ckpt['layer_generator_configs'].items():
        group_size = target_rebuilt.get_group_param_size(name)
        config = dict(gen_cfg)
        config['layer_name'] = name
        mapping = build_generator(config, target_total_params=group_size, device=device)
        mapping.load_persistent_state_dict(ckpt['state_dict'][name])
        layer_mappings[name] = mapping

    group_order = ckpt.get('layer_group_order', list(layer_mappings.keys()))
    group_theta = {name: layer_mappings[name]() for name in group_order}
    theta_rebuilt = target_rebuilt.assemble_params(group_theta)
    logits_rebuilt = target_rebuilt.functional_forward(x, theta_rebuilt)
    assert torch.allclose(logits_ref, logits_rebuilt, atol=1e-5), (
        'LWT multilayer_linear checkpoint reconstruction failed'
    )
