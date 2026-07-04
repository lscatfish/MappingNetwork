"""Tests for LWT trainer — runs on both CPU and GPU."""

import pytest
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
        dataset = TensorDataset(x.cpu(), y.cpu())  # DataLoader 需要 CPU 张量
        loader = DataLoader(dataset, batch_size=8)

        layer_dims = {
            'conv1': 16,
            'conv2': 16,
            'fc1': 16,
            'fc2': 16,
        }

        trainer = LWTTrainer(
            target, loss_fn, layer_dims,
            train_loader=loader, epochs=1, device=device, log_interval=1,
            checkpoint_dir='/tmp/test_lwt', experiment_name='test_lwt',
        )

        # 记录各层 z 的初始值
        z_before = {
            name: mapping.z.data.clone()
            for name, mapping in trainer.layer_mappings.items()
        }

        results = trainer.train()
        assert len(results) == 1

        # 验证每层 z 都已更新，且所有参数在正确设备上
        for name, mapping in trainer.layer_mappings.items():
            assert not torch.equal(z_before[name].cpu(), mapping.z.data.cpu()), \
                f'Layer {name} z was not updated!'
            assert next(mapping.parameters()).device.type == device
