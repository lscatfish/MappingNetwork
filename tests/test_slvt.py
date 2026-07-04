"""Tests for SLVT trainer — runs on both CPU and GPU."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from mapping_network.mapping.mapping_net import MappingNetwork
from mapping_network.mapping.loss import MappingLoss
from mapping_network.target_nets.cnn2 import CNN2
from mapping_network.trainer.slvt import SLVTTrainer


class TestSLVT:
    def test_slvt_train_one_batch(self, device):
        """验证 SLVT 训练一个 batch 后 z 有梯度更新，且全在指定设备上。"""
        target = CNN2().to(device)
        mapping = MappingNetwork(target.get_total_params(), 64).to(device)
        loss_fn = MappingLoss().to(device)

        x = torch.randn(8, 1, 28, 28, device=device)
        y = torch.randint(0, 10, (8,), device=device)
        dataset = TensorDataset(x.cpu(), y.cpu())  # DataLoader 需要 CPU 张量
        loader = DataLoader(dataset, batch_size=8)

        z_before = mapping.z.data.clone()

        trainer = SLVTTrainer(
            mapping, target, loss_fn, loader,
            epochs=1, device=device, log_interval=1,
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
        import os
        checkpoint_path = os.path.join(
            '/tmp/test_slvt_checkpoints', 'test_slvt_final.pth'
        )
        assert os.path.exists(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        assert isinstance(ckpt, dict)
        assert 'state_dict' in ckpt
        assert 'target_net' in ckpt
        assert 'training_strategy' in ckpt
        assert 'latent_dim' in ckpt
        assert ckpt['training_strategy'] == 'slvt'
