"""Tests for SLVT trainer — runs on both CPU and GPU."""

import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from mapping_network.generators.linear import LinearMappingNetwork
from mapping_network.mapping.loss import MappingLoss
from mapping_network.target_nets.cnn2 import CNN2
from mapping_network.trainer.slvt import SLVTTrainer


def make_one_batch_loader(device):
    x = torch.randn(1, 1, 28, 28, device=device)
    y = torch.tensor([0], device=device)
    return DataLoader(TensorDataset(x.cpu(), y.cpu()), batch_size=1)


class TestSLVT:
    def test_slvt_train_one_batch(self, device):
        """验证 SLVT 训练一个 batch 后 z 有梯度更新，且全在指定设备上。"""
        target = CNN2().to(device)
        mapping = LinearMappingNetwork(target.get_total_params(), 64, device=device).to(device)
        loss_fn = MappingLoss().to(device)

        x = torch.randn(8, 1, 28, 28, device=device)
        y = torch.randint(0, 10, (8,), device=device)
        dataset = TensorDataset(x.cpu(), y.cpu())  # DataLoader 需要 CPU 张量
        loader = DataLoader(dataset, batch_size=8)

        z_before = mapping.z.data.clone()

        trainer = SLVTTrainer(
            mapping,
            target,
            loss_fn,
            loader,
            epochs=1,
            device=device,
            log_interval=1,
            checkpoint_dir='/tmp/test_slvt_checkpoints',
            experiment_name='test_slvt',
        )
        results = trainer.train()
        assert len(results) == 1
        # z 应该已被更新
        assert not torch.equal(z_before.cpu(), mapping.z.data.cpu())
        # 确保所有模型参数在正确设备上
        assert next(mapping.parameters()).device.type == device

        # 验证 checkpoint 按新方法打包（含 metadata + state_dict）
        checkpoint_path = os.path.join('/tmp/test_slvt_checkpoints', 'test_slvt_final.pth')
        assert os.path.exists(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        assert isinstance(ckpt, dict)
        assert 'state_dict' in ckpt
        assert 'target_net' in ckpt
        assert 'training_strategy' in ckpt
        assert 'latent_dim' in ckpt
        assert 'generator_type' in ckpt
        assert 'lrd_config' in ckpt
        assert ckpt['training_strategy'] == 'slvt'


def test_slvt_z_updated_with_lrd(device):
    target_net = CNN2(lrd_config={'enabled': True, 'default_rank': 10}).to(device)
    mapping = LinearMappingNetwork(target_net.get_total_params(), 64, device=device).to(device)
    loss_fn = MappingLoss(sigma_noise=0.01).to(device)
    x = torch.randn(1, 1, 28, 28, device=device)
    y = torch.tensor([0], device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=1)
    trainer = SLVTTrainer(
        mapping,
        target_net,
        loss_fn,
        loader,
        loader,
        lr=0.001,
        weight_decay=0.0001,
        epochs=1,
        device=device,
        log_interval=1,
        checkpoint_dir='/tmp/test_slvt',
        experiment_name='test',
        checkpoint_metadata={
            'target_net': 'cnn2',
            'training_strategy': 'slvt',
            'generator_type': 'linear',
            'latent_dim': 64,
            'alpha': 0.01,
            'sigma_noise': 0.01,
            'lrd_config': {'enabled': True, 'default_rank': 10},
        },
        save_interval=0,
    )
    z_before = mapping.z.clone().detach()
    trainer.train_epoch(1)
    assert not torch.allclose(z_before, mapping.z)


def test_slvt_trainer_resume(tmp_path, device):
    target_net = CNN2().to(device)
    mapping = LinearMappingNetwork(target_net.get_total_params(), 64, device=device)
    loss_fn = MappingLoss().to(device)
    loader = make_one_batch_loader(device)
    trainer = SLVTTrainer(
        mapping,
        target_net,
        loss_fn,
        loader,
        loader,
        epochs=2,
        device=device,
        checkpoint_dir=str(tmp_path),
        experiment_name='test_resume',
        save_interval=0,
    )
    trainer.train()
    lambda_st_value = trainer.loss_fn.lambda_st.item()
    ckpt_path = str(tmp_path / 'test_resume_final.pth')
    trainer2 = SLVTTrainer(
        mapping,
        target_net,
        loss_fn,
        loader,
        loader,
        epochs=2,
        device=device,
        checkpoint_dir=str(tmp_path),
        experiment_name='test_resume2',
        save_interval=0,
    )
    epoch = trainer2.load_checkpoint(ckpt_path)
    assert epoch == 2
    assert len(trainer2.results) == 2
    assert trainer2.loss_fn.lambda_st.item() == lambda_st_value
