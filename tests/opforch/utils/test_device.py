# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import torch

from opforch.utils.device import DeviceManager


def test_device_get_default_returns_torch_device():
    device = DeviceManager.get_default()

    assert isinstance(device, torch.device)


def test_device_get_all_gpus_returns_list():
    gpus = DeviceManager.get_all_gpus()

    assert isinstance(gpus, list)


def test_device_resolve_none_returns_default_device():
    device = DeviceManager.resolve(None)

    assert isinstance(device, torch.device)


def test_device_resolve_cpu_returns_cpu_device():
    device = DeviceManager.resolve("cpu")

    assert device == torch.device("cpu")


def test_compute_distance_multi_gpu_supports_cpu_fallback():
    X = torch.tensor([[0.0], [1.0]])
    Y = torch.tensor([[1.0]])
    distances = DeviceManager.compute_distance_multi_gpu(
        X,
        Y,
        lambda left, right: torch.cdist(left, right),
        [torch.device("cpu")],
    )

    assert distances.shape == (2, 1)
    torch.testing.assert_close(distances, torch.tensor([[1.0], [0.0]]))
