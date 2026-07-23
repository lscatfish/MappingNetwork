"""Mapping 训练入口（SLVT / LWT）。

用法:
    uv run python3 examples/train.py --target cnn2 --strategy slvt --epochs 30
    uv run python3 examples/train.py --target cnn1_3conv --strategy lwt --epochs 50 --device cpu
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generators import MLPGenerator
from models.cnn1 import cnn1_lwt, cnn1_slvt
from models.cnn1_3conv import cnn1_3conv_lwt, cnn1_3conv_slvt
from models.cnn2 import cnn2_lwt, cnn2_slvt

from data import get_mnist_loaders
from mapping import MappingLoss
from mapping.trainer import LWTTrainer, SLVTTrainer

MODEL_FACTORIES = {
    'cnn2': {'slvt': cnn2_slvt, 'lwt': cnn2_lwt},
    'cnn1': {'slvt': cnn1_slvt, 'lwt': cnn1_lwt},
    'cnn1_3conv': {'slvt': cnn1_3conv_slvt, 'lwt': cnn1_3conv_lwt},
}


def main():
    parser = argparse.ArgumentParser(description='Mapping Network Training')
    parser.add_argument('--target', choices=['cnn1', 'cnn2', 'cnn1_3conv'], default='cnn2')
    parser.add_argument('--strategy', choices=['slvt', 'lwt'], default='slvt')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--z-dim', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--sigma-noise', type=float, default=1e-4)
    parser.add_argument('--n-stab-samples', type=int, default=5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--save-interval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_loader, test_loader = get_mnist_loaders(args.batch_size)
    experiment_name = f'{args.target}_{args.strategy}'
    ckpt_dir = str(Path(args.checkpoint_dir) / experiment_name)

    gen_kwargs = {'z_dim': args.z_dim, 'hidden_dim': args.hidden_dim}
    factory = MODEL_FACTORIES[args.target][args.strategy]
    net = factory(generator_cls=MLPGenerator, **gen_kwargs)

    loss_fn = MappingLoss(
        sigma_noise=args.sigma_noise,
        n_stab_samples=args.n_stab_samples,
    )

    trainer_cls = SLVTTrainer if args.strategy == 'slvt' else LWTTrainer
    trainer = trainer_cls(
        net=net,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=args.device,
        checkpoint_dir=ckpt_dir,
        experiment_name=experiment_name,
        save_interval=args.save_interval,
    )

    trainer.train()
    print(f'\nDone. Best test acc: {trainer.best_test_acc:.2f}%')


if __name__ == '__main__':
    main()
