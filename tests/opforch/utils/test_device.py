import torch

from opforch.utils.device import DeviceManager


def test_device_get_default():
    device = DeviceManager.get_default()

    assert isinstance(device, torch.device)


def test_device_get_all_gpus():
    gpus = DeviceManager.get_all_gpus()

    assert isinstance(gpus, list)


def test_device_resolve_none():
    device = DeviceManager.resolve(None)

    assert isinstance(device, torch.device)


def test_device_resolve_cpu():
    device = DeviceManager.resolve("cpu")

    assert device == torch.device("cpu")


def test_compute_distance_multi_gpu_cpu():
    X = torch.tensor([[0.0], [1.0]])
    Y = torch.tensor([[1.0]])
    distances = DeviceManager.compute_distance_multi_gpu(
        X,
        Y,
        lambda left, right: torch.cdist(left, right),
        [torch.device("cpu")],
    )

    assert distances.shape == (2, 1)
