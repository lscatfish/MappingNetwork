"""MNIST 数据加载。"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_MNIST_MEAN = (0.1307,)
_MNIST_STD = (0.3081,)


def get_mnist_loaders(batch_size: int = 64, root: str = './data'):
    """返回 (train_loader, test_loader)。"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MNIST_MEAN, _MNIST_STD),
    ])
    train_dataset = datasets.MNIST(root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root, train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader
