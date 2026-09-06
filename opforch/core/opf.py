# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Common distance, device, and persistence operations for OPF classifiers.

References:
    J. P. Papa, A. X. Falcão and C. T. N. Suzuki, LibOPF: A library for optimum-path forest classifiers (2015).

"""

from __future__ import annotations

from typing import Self

import numpy as np
import torch

import opforch.math.distance as d
import opforch.utils.exception as e
from opforch.core.subgraph import Subgraph
from opforch.utils.device import DeviceManager
from opforch.utils.logging import get_logger

logger = get_logger(__name__)


class OPF:
    """Share device, distance, and persistence behavior between OPF classifiers.

    """  # fmt: skip

    def __init__(
        self,
        distance: str = "log_squared_euclidean",
        pre_computed_distance: str | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize a classifier and load an optional distance file.

        Args:
            distance: Name of a registered distance function.
            pre_computed_distance: Path to a CSV, TXT, PT, or PTH distance file.
            device: Target device, or automatic CUDA/CPU selection when None.

        Raises:
            e.TypeError: The distance function name is not registered.
            e.ArgumentError: The distance file extension is unsupported.
            OSError: The distance file cannot be read.

        """

        logger.info("Creating OPF classifier.")

        self.device = DeviceManager.resolve(device)
        self.subgraph: Subgraph | None = None
        self.pre_distances: torch.Tensor | None = None

        if distance not in d.VALID_DISTANCES:
            raise e.TypeError(f"`distance` must name a registered function, but got {distance!r}.")

        self.distance = distance
        self.distance_fn = d.DISTANCES[distance]

        if pre_computed_distance:
            self.pre_computed_distance = True
            self._read_distances(pre_computed_distance)
        else:
            self.pre_computed_distance = False

        logger.debug(
            "Distance: %s | Pre-computed distance: %s | Device: %s.",
            self.distance,
            self.pre_computed_distance,
            self.device,
        )
        logger.info("Class created.")

    def _read_distances(self, file_name: str) -> None:
        logger.debug("Reading precomputed distances.")

        extension = file_name.split(".")[-1]

        if extension in ("pt", "pth"):
            distances = torch.load(file_name, map_location=self.device, weights_only=True)
        elif extension in ("csv", "txt"):
            data = np.loadtxt(file_name, delimiter="," if extension == "csv" else " ")
            distances = torch.from_numpy(data).to(dtype=torch.float64, device=self.device)
        else:
            raise e.ArgumentError(f"`file_name` must end in .csv, .txt, .pt, or .pth, but got {file_name!r}.")

        if distances is None:
            raise e.ValueError("`distances` is None after loading.")

        self.pre_distances = distances

    def get_distances(self, normalize: bool = False) -> torch.Tensor:
        """Return the training subgraph's distance matrix.

        Precomputed files are indexed by the subgraph's original sample indices.
        Min-max normalization maps a constant matrix to zeros.

        Args:
            normalize: Whether to apply min-max normalization.

        Returns:
            A distance tensor of shape (N, N) on the model device.

        Raises:
            e.BuildError: The subgraph or requested precomputed distances are unavailable.

        """

        distances = self._get_distances()

        if normalize:
            d_min = distances.min()
            d_max = distances.max()
            scale = d_max - d_min
            return (distances - d_min) / torch.where(scale == 0, 1, scale)

        return distances

    def _get_distances(
        self,
        X: torch.Tensor | None = None,
        I: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.subgraph is None:
            raise e.BuildError("`subgraph` is None.")
        if self.pre_computed_distance and self.pre_distances is None:
            raise e.BuildError("`pre_distances` is None while pre_computed_distance is True.")

        return self.subgraph._get_distances(
            self.distance_fn,
            self.pre_distances if self.pre_computed_distance else None,
            X,
            I,
        )

    def _check_fitted(self) -> None:
        if self.subgraph is None:
            raise e.BuildError("`subgraph` is None.")
        if not self.subgraph.trained:
            raise e.BuildError("`subgraph.trained` must be True before prediction.")

    def to(self, device: str | torch.device) -> Self:
        """Move the model's tensor state and device metadata in place.

        Args:
            device: Target device string or torch.device.

        Returns:
            This model for chaining.

        """

        self.device = torch.device(device) if isinstance(device, str) else device

        if self.subgraph is not None:
            self.subgraph.to(self.device)

        if self.pre_distances is not None:
            self.pre_distances = self.pre_distances.to(self.device)

        return self

    def save(self, file_name: str) -> None:
        """Serialize this model without relocating its live tensor state.

        The whole-object checkpoint retains the existing PyTorch serialization
        format and depends on the model's Python class definitions when loaded.

        Args:
            file_name: Destination checkpoint path, overwritten if it already exists.

        Raises:
            OSError: The checkpoint cannot be written.

        """

        logger.info("Saving model to file: %s ...", file_name)

        torch.save(self, file_name)

        logger.info("Model saved.")

    def load(self, file_name: str) -> None:
        """Replace this model's state from a trusted checkpoint.

        Tensor storage and graph metadata use this instance's requested device.
        Whole-object deserialization can execute Python code, so the file must
        come from a trusted source.

        Args:
            file_name: Path to a checkpoint produced by save().

        Raises:
            OSError: The checkpoint cannot be read.

        """

        logger.info("Loading model from file: %s ...", file_name)

        target_device = self.device
        loaded = torch.load(file_name, map_location=target_device, weights_only=False)

        self.__dict__.update(loaded.__dict__)
        self.to(target_device)

        logger.info("Model loaded.")

    def fit(self, X: torch.Tensor, Y: torch.Tensor, **kwargs) -> None:
        """Define the fitting interface implemented by concrete classifiers.

        Args:
            X: Feature tensor of shape (N, D).
            Y: Label tensor of shape (N,).
            **kwargs: Classifier-specific fitting arguments.

        Raises:
            NotImplementedError: A concrete classifier has not supplied fitting behavior.

        """

        raise NotImplementedError("`fit` must be implemented by a concrete classifier.")

    def predict(self, X: torch.Tensor, **kwargs) -> list[int]:
        """Define the prediction interface implemented by concrete classifiers.

        Args:
            X: Feature tensor of shape (M, D).
            **kwargs: Classifier-specific prediction arguments.

        Returns:
            A list of predicted labels.

        Raises:
            NotImplementedError: A concrete classifier has not supplied prediction behavior.

        """

        raise NotImplementedError("`predict` must be implemented by a concrete classifier.")
