"""扩展性集成测试：验证 trainer / evaluate 对非 LinearMappingNetwork 的 generator 无硬编码依赖。

用 MultiLayerLinearMappingNetwork 作为示例 generator（它没有 W_fixed / W_fixed_mean /
b_fixed / w_seed 等 LinearMappingNetwork 私有属性），跑通构造 -> 训练 -> checkpoint
save/load -> 评估。如果 SLVTTrainer / LWTTrainer / evaluate 中残留了对
LinearMappingNetwork 私有细节的硬编码引用，本测试会因 AttributeError 失败。

这是 issue #12 的核心验收点：新增 generator 不需要修改 trainer / evaluate。
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from mapping_network.factory import build_generator
from mapping_network.generators.multilayer_linear import MultiLayerLinearMappingNetwork
from mapping_network.mapping.loss import MappingLoss
from mapping_network.scripts.evaluate import evaluate_model
from mapping_network.target_nets.cnn2 import CNN2
from mapping_network.trainer.lwt import LWTTrainer
from mapping_network.trainer.slvt import SLVTTrainer

# 示例 generator 配置：multilayer_linear 没有 LinearMappingNetwork 的私有 buffer。
GEN_CONFIG = {
    'type': 'multilayer_linear',
    'latent_dim': 32,
    'alpha': 0.01,
    'hidden_dim': 16,
    'num_hidden': 1,
}


def _make_loader(device, batch_size=8):
    x = torch.randn(batch_size, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_slvt_train_save_load_with_multilayer_linear(device, tmp_path):
    """SLVT 用 multilayer_linear 训练 -> persistent save -> 重建 -> load -> z 一致。"""
    target = CNN2().to(device)
    P = target.get_total_params()
    loss_fn = MappingLoss().to(device)
    loader = _make_loader(device)

    gen = build_generator(GEN_CONFIG, target_total_params=P, device=device)
    assert isinstance(gen, MultiLayerLinearMappingNetwork)
    theta = gen()
    assert theta.shape == (P,)
    assert theta.device.type == device

    trainer = SLVTTrainer(
        gen,
        target,
        loss_fn,
        loader,
        epochs=1,
        device=device,
        log_interval=1,
        checkpoint_dir=str(tmp_path),
        experiment_name='ext_slvt',
        save_interval=0,
    )
    z_before = gen.z.detach().clone()
    trainer.train()
    # z 已更新，说明 forward / noisy_forward / smooth_loss / align_loss 在 trainer 中都被调通
    assert not torch.allclose(z_before, gen.z.detach())

    # persistent checkpoint: save -> 重建 generator -> load -> z 一致
    state = gen.persistent_state_dict()
    gen2 = build_generator(GEN_CONFIG, target_total_params=P, device=device)
    gen2.load_persistent_state_dict(state)
    assert torch.allclose(gen.z.detach(), gen2.z.detach())


def test_lwt_train_with_multilayer_linear(device, tmp_path):
    """LWT 用 multilayer_linear 训练（每层一个 generator）。"""
    target = CNN2().to(device)
    loss_fn = MappingLoss().to(device)
    loader = _make_loader(device)

    layer_generators = {
        name: {
            'type': 'multilayer_linear',
            'latent_dim': 16,
            'alpha': 0.01,
            'hidden_dim': 8,
            'num_hidden': 1,
        }
        for name in target.get_group_names()
    }
    trainer = LWTTrainer(
        target,
        loss_fn,
        layer_generators,
        train_loader=loader,
        test_loader=loader,
        epochs=1,
        device=device,
        log_interval=1,
        checkpoint_dir=str(tmp_path),
        experiment_name='ext_lwt',
        save_interval=0,
    )
    for mapping in trainer.layer_mappings.values():
        assert isinstance(mapping, MultiLayerLinearMappingNetwork)

    z_before = {n: m.z.detach().clone() for n, m in trainer.layer_mappings.items()}
    trainer.train()
    # 至少一层 z 更新
    assert any(
        not torch.allclose(z_before[n], m.z.detach()) for n, m in trainer.layer_mappings.items()
    )


def test_evaluate_model_with_multilayer_linear(device):
    """evaluate_model 接受 multilayer_linear 生成的 theta_hat 完成评估。"""
    target = CNN2().to(device)
    P = target.get_total_params()
    gen = build_generator(GEN_CONFIG, target_total_params=P, device=device)
    theta_hat = gen().detach()
    loader = _make_loader(device, batch_size=4)
    acc = evaluate_model(target, theta_hat, loader, device)
    assert 0.0 <= acc <= 100.0
