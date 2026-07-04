import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from mapping_network.mapping.mapping_net import MappingNetwork
from mapping_network.mapping.loss import MappingLoss
from mapping_network.target_nets.cnn2 import CNN2
from mapping_network.trainer.slvt import SLVTTrainer


def test_slvt_train_one_batch():
    """验证 SLVT 训练一个 batch 后 z 有梯度更新。"""
    target = CNN2()
    mapping = MappingNetwork(target.get_total_params(), 64)
    loss_fn = MappingLoss()

    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=8)

    z_before = mapping.z.data.clone()

    trainer = SLVTTrainer(
        mapping, target, loss_fn, loader,
        epochs=1, device='cpu', log_interval=1,
        checkpoint_dir='/tmp/test_slvt_checkpoints',
        experiment_name='test_slvt',
    )
    results = trainer.train()
    assert len(results) == 1
    # z 应该已被更新
    assert not torch.equal(z_before, mapping.z.data)
