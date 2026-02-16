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
