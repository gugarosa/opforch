# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Manage devices for CPU, GPU, and multi-GPU calculations.

"""  # fmt: skip

from __future__ import annotations

from collections.abc import Callable

import torch


class DeviceManager:
    """Resolve devices and distribute distance calculations.

    """  # fmt: skip

    @staticmethod
    def get_default() -> torch.device:
        """Choose CUDA when available and CPU otherwise.

        Returns:
            An unindexed CUDA device when torch.cuda.is_available() is True, otherwise a CPU device.

        """

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_all_gpus() -> list[torch.device]:
        """List the CUDA devices visible to PyTorch.

        Returns:
            Indexed CUDA devices in ascending order, or an empty list when no GPUs are visible.

        """

        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

    @staticmethod
    def resolve(device: str | None = None) -> torch.device:
        """Resolve a device specification without allocating tensors.

        Explicit devices are normalized without checking availability. When device is None,
        selection follows get_default rather than PyTorch's configurable default allocation device.

        Args:
            device: PyTorch device specification, or None to choose CUDA when available and CPU otherwise.

        Returns:
            The explicit device or the result of get_default.

        Raises:
            TypeError: The device specification has an unsupported type.
            RuntimeError: PyTorch cannot parse the device specification.

        """

        return torch.device(device) if device is not None else DeviceManager.get_default()

    @staticmethod
    def compute_distance_multi_gpu(
        X: torch.Tensor,
        Y: torch.Tensor,
        distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        devices: list[torch.device] | None = None,
    ) -> torch.Tensor:
        """Compute pairwise distances using row chunks on the selected devices.

        When devices is None, all visible GPUs are selected. An empty list uses get_default.
        With one selected device, the callback is called once and its result is returned unchanged.
        With several devices, callbacks run sequentially on row chunks of X and all of Y, then
        their results are gathered on devices[0]. Trailing devices may be unused when X has fewer chunks.
        Device transfers preserve input dtypes and autograd history and do not force copies.
        Callback exceptions propagate unchanged, and result dtypes follow the callback and PyTorch concatenation.

        Args:
            X: Feature tensor of shape (N, D) with a dtype accepted by the distance callback.
            Y: Reference tensor of shape (M, D) with a dtype accepted by the distance callback.
            distance_fn: Callable accepting (K, D) and (M, D) tensors and returning (K, M) distances.
            devices: Ordered target devices, or None to discover all visible GPUs.

        Returns:
            An (N, M) callback result, gathered on devices[0] only for the multi-device path.

        Raises:
            RuntimeError: A device transfer or concatenation fails.

        """

        if devices is None:
            devices = DeviceManager.get_all_gpus()
        if len(devices) <= 1:
            target = devices[0] if devices else DeviceManager.get_default()
            return distance_fn(X.to(target), Y.to(target))

        chunks = X.chunk(len(devices), dim=0)
        results = [distance_fn(chunk.to(device), Y.to(device)).to(devices[0]) for chunk, device in zip(chunks, devices)]
        return torch.cat(results, dim=0)
