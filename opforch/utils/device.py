"""Device management for CPU, GPU and Multi-GPU environments."""

from typing import List, Optional

import torch


class DeviceManager:
    """Centralized device detection and management."""

    @staticmethod
    def get_default() -> torch.device:
        """Auto-detect best available device.

        Returns:
            The best available torch device (CUDA if available, else CPU).

        """

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    @staticmethod
    def get_all_gpus() -> List[torch.device]:
        """Returns list of all available CUDA devices.

        Returns:
            A list of torch.device objects for each available GPU.

        """

        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

    @staticmethod
    def resolve(device: Optional[str] = None) -> torch.device:
        """Resolves a device string or None to a torch.device.

        Args:
            device: Device string ('cpu', 'cuda', 'cuda:0', etc.) or None for auto.

        Returns:
            A resolved torch.device.

        """

        if device is None:
            return DeviceManager.get_default()

        return torch.device(device)

    @staticmethod
    def compute_distance_multi_gpu(
        X: torch.Tensor,
        Y: torch.Tensor,
        distance_fn: callable,
        devices: Optional[List[torch.device]] = None,
    ) -> torch.Tensor:
        """Distribute distance computation across multiple GPUs.

        Splits X into chunks (one per GPU), computes partial distance matrices
        in parallel on each device, then concatenates results.

        Args:
            X: Source tensor of shape (N, D).
            Y: Target tensor of shape (M, D).
            distance_fn: Batched distance function (N,D)x(M,D) -> (N,M).
            devices: List of GPU devices. Auto-detected if None.

        Returns:
            Distance matrix of shape (N, M).

        """

        if devices is None:
            devices = DeviceManager.get_all_gpus()

        if len(devices) <= 1:
            target = devices[0] if devices else DeviceManager.get_default()
            return distance_fn(X.to(target), Y.to(target))

        chunks = X.chunk(len(devices), dim=0)
        Y_replicas = [Y.to(d) for d in devices]

        results = []
        for chunk, device, Y_d in zip(chunks, devices, Y_replicas):
            results.append(distance_fn(chunk.to(device), Y_d))

        return torch.cat([r.to(devices[0]) for r in results], dim=0)
