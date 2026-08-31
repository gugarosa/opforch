"""Device management for CPU, GPU, and multi-GPU environments."""

from __future__ import annotations

from collections.abc import Callable

import torch


class DeviceManager:
    """Resolve devices and distribute distance calculations."""

    @staticmethod
    def get_default() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_all_gpus() -> list[torch.device]:
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

    @staticmethod
    def resolve(device: str | None = None) -> torch.device:
        return (
            torch.device(device) if device is not None else DeviceManager.get_default()
        )

    @staticmethod
    def compute_distance_multi_gpu(
        X: torch.Tensor,
        Y: torch.Tensor,
        distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        devices: list[torch.device] | None = None,
    ) -> torch.Tensor:
        """Compute a batched ``(N, D) x (M, D) -> (N, M)`` distance by device."""

        if devices is None:
            devices = DeviceManager.get_all_gpus()
        if len(devices) <= 1:
            target = devices[0] if devices else DeviceManager.get_default()
            return distance_fn(X.to(target), Y.to(target))

        chunks = X.chunk(len(devices), dim=0)
        results = [
            distance_fn(chunk.to(device), Y.to(device)).to(devices[0])
            for chunk, device in zip(chunks, devices)
        ]
        return torch.cat(results, dim=0)
