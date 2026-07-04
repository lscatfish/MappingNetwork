import torch

from mapping_network.factory import build_generator, build_target_net


def test_build_cnn1_with_lrd(device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    net = build_target_net('cnn1', {'enabled': True, 'default_rank': 10})
    gen = build_generator('linear', net.get_total_params(), 2072, 0.01, device)
    theta = gen()
    assert theta.shape[0] < 537_960
    assert theta.device.type == device
