import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mapping import Conv2d, Generator, Linear, MappingLoss, Sequential
from mapping.trainer import LWTTrainer, SLVTTrainer, collect_generators


class TinyGen(Generator):
    def __init__(self, param_spec, z_dim=16, **kwargs):
        super().__init__(param_spec, z_dim=z_dim)
        self.head = nn.Linear(z_dim, self.w_size + self.b_size)

    def forward(self):
        h = self.head(self.z)
        w = h[:self.w_size].reshape(self.w_shape)
        b = h[self.w_size:].reshape(self.b_shape) if self.b_size > 0 else None
        return w, b


def make_loaders(device, n_train=32, n_test=16, num_classes=4):
    x_train = torch.randn(n_train, 1, 8, 8)
    y_train = torch.randint(0, num_classes, (n_train,))
    x_test = torch.randn(n_test, 1, 8, 8)
    y_test = torch.randint(0, num_classes, (n_test,))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=16)
    return train_loader, test_loader


class TestSLVTTrainer:
    def test_train_one_epoch_updates_z(self, device):
        net = Sequential(
            Conv2d(1, 4, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(1),
            Linear(4 * 8 * 8, 4),
            generator_cls=TinyGen,
            z_dim=16,
        ).to(device)
        loss_fn = MappingLoss(n_stab_samples=1)
        train_loader, test_loader = make_loaders(device)

        z_before = net.generator.z.data.clone()

        trainer = SLVTTrainer(
            net=net,
            loss_fn=loss_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=1,
            device=device,
            checkpoint_dir=tempfile.mkdtemp(),
            experiment_name='test_slvt',
            save_interval=0,
        )
        results = trainer.train()

        assert len(results) == 1
        assert results[0]['train_loss'] > 0
        assert not torch.equal(net.generator.z.data, z_before)

    def test_evaluate_returns_accuracy(self, device):
        net = Sequential(
            Conv2d(1, 4, 3, padding=1),
            nn.Flatten(1),
            Linear(4 * 8 * 8, 4),
            generator_cls=TinyGen,
            z_dim=16,
        ).to(device)
        loss_fn = MappingLoss(n_stab_samples=1)
        train_loader, test_loader = make_loaders(device)

        trainer = SLVTTrainer(
            net=net,
            loss_fn=loss_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=1,
            device=device,
            checkpoint_dir=tempfile.mkdtemp(),
            experiment_name='test_slvt_eval',
            save_interval=0,
        )
        acc = trainer.evaluate()
        assert acc is not None
        assert 0 <= acc <= 100

    def test_checkpoint_save_load(self, device):
        net = Sequential(
            Conv2d(1, 4, 3, padding=1),
            nn.Flatten(1),
            Linear(4 * 8 * 8, 4),
            generator_cls=TinyGen,
            z_dim=16,
        ).to(device)
        loss_fn = MappingLoss(n_stab_samples=1)
        train_loader, test_loader = make_loaders(device)
        ckpt_dir = tempfile.mkdtemp()

        trainer = SLVTTrainer(
            net=net,
            loss_fn=loss_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=1,
            device=device,
            checkpoint_dir=ckpt_dir,
            experiment_name='test_ckpt',
            save_interval=0,
        )
        trainer.train()

        x = torch.randn(2, 1, 8, 8, device=device)
        with torch.no_grad():
            out_before = net(x)

        net2 = Sequential(
            Conv2d(1, 4, 3, padding=1),
            nn.Flatten(1),
            Linear(4 * 8 * 8, 4),
            generator_cls=TinyGen,
            z_dim=16,
        ).to(device)
        loss_fn2 = MappingLoss(n_stab_samples=1)
        trainer2 = SLVTTrainer(
            net=net2,
            loss_fn=loss_fn2,
            train_loader=train_loader,
            device=device,
            epochs=1,
            checkpoint_dir=ckpt_dir,
            experiment_name='test_ckpt',
            save_interval=0,
        )
        import os
        ckpt_path = os.path.join(ckpt_dir, 'test_ckpt_final.pth')
        trainer2.load_checkpoint(ckpt_path)

        with torch.no_grad():
            out_after = net2(x)
        assert torch.allclose(out_before, out_after, atol=1e-6)


class TestLWTTrainer:
    def test_lwt_train_one_epoch(self, device):
        class LWTNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = Conv2d(1, 4, 3, padding=1, generator_cls=TinyGen, z_dim=16)
                self.fc = Linear(4 * 8 * 8, 4, generator_cls=TinyGen, z_dim=16)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = x.flatten(1)
                return self.fc(x)

        net = LWTNet().to(device)
        loss_fn = MappingLoss(n_stab_samples=1)
        train_loader, test_loader = make_loaders(device)

        z_before = net.conv1.generator.z.data.clone()

        trainer = LWTTrainer(
            net=net,
            loss_fn=loss_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=1,
            device=device,
            checkpoint_dir=tempfile.mkdtemp(),
            experiment_name='test_lwt',
            save_interval=0,
        )
        results = trainer.train()

        assert len(results) == 1
        assert not torch.equal(net.conv1.generator.z.data, z_before)

    def test_collect_generators(self, device):
        class MyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = Conv2d(1, 4, 3, generator_cls=TinyGen, z_dim=16)
                self.relu = nn.ReLU()
                self.fc = Linear(100, 4, generator_cls=TinyGen, z_dim=16)

            def forward(self, x):
                return self.fc(self.relu(self.conv1(x)).flatten(1))

        net = MyNet().to(device)
        gens = collect_generators(net)
        assert len(gens) == 2
        assert gens[0] is net.conv1.generator
        assert gens[1] is net.fc.generator

    def test_lwt_no_generators_raises(self, device):
        class PlainNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 4)

            def forward(self, x):
                return self.fc(x)

        net = PlainNet().to(device)
        loss_fn = MappingLoss(n_stab_samples=1)
        train_loader, _ = make_loaders(device)

        import pytest
        with pytest.raises(ValueError, match='未找到'):
            LWTTrainer(
                net=net,
                loss_fn=loss_fn,
                train_loader=train_loader,
                epochs=1,
                device=device,
                checkpoint_dir=tempfile.mkdtemp(),
            )
