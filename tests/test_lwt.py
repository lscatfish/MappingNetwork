"""Tests for LWT trainer — runs on both CPU and GPU."""

import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from mapping_network.mapping.loss import MappingLoss
from mapping_network.target_nets.cnn2 import CNN2
from mapping_network.trainer.lwt import LWTTrainer


class TestLWT:
    def test_lwt_train_one_batch(self, device):
        """验证 LWT 每层独立训练一个 batch 后 z 有梯度更新，且全在指定设备上。"""
        target = CNN2().to(device)
        loss_fn = MappingLoss().to(device)

        x = torch.randn(8, 1, 28, 28, device=device)
        y = torch.randint(0, 10, (8,), device=device)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=8)

        layer_generators = {
            'conv1': {'type': 'linear', 'latent_dim': 16, 'alpha': 0.01},
            'conv2': {'type': 'linear', 'latent_dim': 16, 'alpha': 0.01},
            'fc1': {'type': 'linear', 'latent_dim': 16, 'alpha': 0.01},
            'fc2': {'type': 'linear', 'latent_dim': 16, 'alpha': 0.01},
        }

        trainer = LWTTrainer(
            target,
            loss_fn,
            layer_generators=layer_generators,
            train_loader=loader,
            epochs=1,
            device=device,
            log_interval=1,
            checkpoint_dir='/tmp/test_lwt',
            experiment_name='test_lwt',
        )

        # 记录各层 z 的初始值
        z_before = {
            name: mapping.z.data.clone() for name, mapping in trainer.layer_mappings.items()
        }

        results = trainer.train()
        assert len(results) == 1

        # 验证每层 z 都已更新，且所有参数在正确设备上
        for name, mapping in trainer.layer_mappings.items():
            assert not torch.equal(z_before[name], mapping.z.data), (
                f'Layer {name} z was not updated!'
            )
            assert next(mapping.parameters()).device.type == device

        # 验证 checkpoint 按新方法打包（含 metadata + state_dict）
        checkpoint_path = os.path.join('/tmp/test_lwt', 'test_lwt_final.pth')
        assert os.path.exists(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device)
        assert isinstance(ckpt, dict)
        assert 'state_dict' in ckpt
        assert 'target_net' in ckpt
        assert 'training_strategy' in ckpt
        assert 'layer_generator_configs' in ckpt
        assert 'layer_group_order' in ckpt
        assert ckpt['training_strategy'] == 'lwt'


def test_lwt_per_layer_config(device):
    target_net = CNN2(lrd_config={'enabled': True, 'default_rank': 10}).to(device)
    loss_fn = MappingLoss(sigma_noise=0.01).to(device)
    x = torch.randn(1, 1, 28, 28, device=device)
    y = torch.tensor([0], device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=1)
    layer_generators = {
        'conv1': {'type': 'linear', 'latent_dim': 16, 'alpha': 0.01},
        'conv2': {'type': 'linear', 'latent_dim': 16, 'alpha': 0.01},
        'fc1': {'type': 'linear', 'latent_dim': 32, 'alpha': 0.01},
        'fc2': {'type': 'linear', 'latent_dim': 8, 'alpha': 0.01},
    }
    trainer = LWTTrainer(
        target_net,
        loss_fn,
        layer_generators,
        train_loader=loader,
        test_loader=loader,
        lr=0.001,
        weight_decay=0.0001,
        epochs=1,
        device=device,
        log_interval=1,
        checkpoint_dir='/tmp/test_lwt',
        experiment_name='test',
        checkpoint_metadata={
            'target_net': 'cnn2',
            'training_strategy': 'lwt',
            'lrd_config': {'enabled': True, 'default_rank': 10},
            'sigma_noise': 0.01,
        },
        save_interval=0,
    )
    trainer.train_epoch(1)
    total_z = sum(m.d for m in trainer.layer_mappings.values())
    assert total_z == 16 + 16 + 32 + 8


def make_one_batch_loader(device):
    x = torch.randn(8, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (8,), device=device)
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_lwt_trainer_resume(tmp_path, device):
    target_net = CNN2().to(device)
    loss_fn = MappingLoss().to(device)
    loader = make_one_batch_loader(device)
    layer_gens = {
        name: {'type': 'linear', 'latent_dim': 16, 'alpha': 0.01}
        for name in target_net.get_group_names()
    }
    ckpt_dir = str(tmp_path)
    trainer = LWTTrainer(
        target_net,
        loss_fn,
        layer_gens,
        loader,
        loader,
        epochs=2,
        device=device,
        checkpoint_dir=ckpt_dir,
        experiment_name='test_lwt_resume',
        save_interval=0,
    )
    trainer.train()
    lambda_st_value = trainer.loss_fn.lambda_st.item()
    ckpt_path = tmp_path / 'test_lwt_resume_final.pth'
    trainer2 = LWTTrainer(
        target_net,
        loss_fn,
        layer_gens,
        loader,
        loader,
        epochs=2,
        device=device,
        checkpoint_dir=ckpt_dir,
        experiment_name='test_lwt_resume2',
        save_interval=0,
    )
    epoch = trainer2.load_checkpoint(str(ckpt_path))
    assert epoch == 2
    assert len(trainer2.results) == 2
    assert trainer2.loss_fn.lambda_st.item() == lambda_st_value


def test_lwt_stab_no_cross_layer_gradient(device):
    """验证 LWT 的 L_stab 只反向传播到被扰动那层的 z，不会泄漏到其他层。"""
    target_net = CNN2().to(device)
    loss_fn = MappingLoss(sigma_noise=0.01).to(device)
    x = torch.randn(1, 1, 28, 28, device=device)
    y = torch.tensor([0], device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=1)
    layer_generators = {
        'conv1': {'type': 'linear', 'latent_dim': 16, 'alpha': 0.01},
        'conv2': {'type': 'linear', 'latent_dim': 16, 'alpha': 0.01},
        'fc1': {'type': 'linear', 'latent_dim': 16, 'alpha': 0.01},
        'fc2': {'type': 'linear', 'latent_dim': 16, 'alpha': 0.01},
    }
    trainer = LWTTrainer(
        target_net,
        loss_fn,
        layer_generators,
        train_loader=loader,
        test_loader=loader,
        lr=0.001,
        weight_decay=0.0001,
        epochs=1,
        device=device,
        log_interval=1,
        checkpoint_dir='/tmp/test_lwt',
        experiment_name='test_no_grad_leak',
        save_interval=0,
    )

    # 只让 fc2 层参与 L_stab，其他层 detach
    theta_hat = trainer._generate_all_theta()
    y_hat = trainer.target_net.functional_forward(x, theta_hat)
    offsets = trainer._compute_offsets()
    start, end = offsets['fc2']
    theta_noisy = theta_hat.detach().clone()
    theta_noisy[start:end] = trainer.layer_mappings['fc2'].noisy_forward(loss_fn.sigma_noise)
    y_hat_noisy = trainer.target_net.functional_forward(x, theta_noisy)
    l_stab = torch.nn.functional.mse_loss(y_hat_noisy, y_hat.detach())
    l_stab.backward()

    # fc2 的 z 必须有梯度，其他层必须为 None 或全 0
    assert trainer.layer_mappings['fc2'].z.grad is not None
    assert trainer.layer_mappings['fc2'].z.grad.abs().sum() > 0
    for name in ('conv1', 'conv2', 'fc1'):
        grad = trainer.layer_mappings[name].z.grad
        assert grad is None or grad.abs().sum() == 0, (
            f'Layer {name} received cross-layer gradient from fc2 stab loss'
        )
