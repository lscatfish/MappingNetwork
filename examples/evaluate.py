"""评估入口：加载 checkpoint 并评估。

用法:
    uv run python3 examples/evaluate.py --target cnn2 --strategy slvt --checkpoint checkpoints/cnn2_slvt/cnn2_slvt_final.pth
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
    parser = argparse.ArgumentParser(description='Evaluate Mapping Network')
    parser.add_argument('--target', choices=['cnn1', 'cnn2', 'cnn1_3conv'], default='cnn2')
    parser.add_argument('--strategy', choices=['slvt', 'lwt'], default='slvt')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--z-dim', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    _, test_loader = get_mnist_loaders(args.batch_size)

    gen_kwargs = {'z_dim': args.z_dim, 'hidden_dim': args.hidden_dim}
    factory = MODEL_FACTORIES[args.target][args.strategy]
    net = factory(generator_cls=MLPGenerator, **gen_kwargs)
    loss_fn = MappingLoss()

    trainer_cls = SLVTTrainer if args.strategy == 'slvt' else LWTTrainer
    trainer = trainer_cls(
        net=net,
        loss_fn=loss_fn,
        train_loader=test_loader,
        test_loader=test_loader,
        epochs=1,
        device=args.device,
        checkpoint_dir='/tmp/eval_tmp',
        experiment_name='eval',
        save_interval=0,
    )
    trainer.load_checkpoint(args.checkpoint)

    acc = trainer.evaluate()
    print(f'Test accuracy: {acc:.2f}%')


if __name__ == '__main__':
    main()
