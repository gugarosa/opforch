# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Store sample data and aligned OPF state as dense tensors.

Features use float32 storage. Labels, indices, predecessors, roots, and cluster
assignments use int64. Costs, densities, and radii use float64, while status and
relevance flags use int8.

"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import torch

import opforch.stream.parser as p
import opforch.utils.constants as c
import opforch.utils.exception as e
from opforch.math.distance import DistanceFn
from opforch.stream import loader
from opforch.utils.device import DeviceManager
from opforch.utils.logging import get_logger

logger = get_logger(__name__)


class Subgraph:
    """Represent samples and their aligned OPF graph state.

    """  # fmt: skip

    def __init__(
        self,
        X: torch.Tensor | None = None,
        Y: torch.Tensor | None = None,
        I: torch.Tensor | None = None,
        from_file: str | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize graph data and per-node state.

        Tensor or NumPy storage can be shared when conversion is unnecessary.
        Callers must not modify shared sample data while the graph is in use.

        Args:
            X: Feature array or tensor of shape (N, D), or None for an empty graph.
            Y: Label array or tensor of shape (N,), or None to initialize zero labels.
            I: Original indices of shape (N,), or None to use sample positions.
            from_file: CSV, TXT, or JSON data path that replaces X and Y when provided.
            device: Target device, or automatic CUDA/CPU selection when None.

        Raises:
            e.ArgumentError: The input file extension is unsupported.
            e.SizeError: Feature, label, or index dimensions do not align.
            e.TypeError: Sample indices are not integers.
            e.ValueError: Sample indices are negative.

        """

        self.device = DeviceManager.resolve(device)
        self.trained = False

        if from_file:
            X, Y = self._load(from_file)

        if X is not None:
            self._build(X, Y, I)
        else:
            self.features = torch.empty(0, device=self.device)
            self.labels = torch.empty(0, dtype=torch.int64, device=self.device)
            self.indices = torch.empty(0, dtype=torch.int64, device=self.device)
            self._n_features = 0
            self._init_state_tensors(0)
            logger.debug("Created an empty subgraph.")

    def _to_tensor(self, arr: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor | None:
        if arr is None:
            return None

        if isinstance(arr, np.ndarray):
            t = torch.from_numpy(arr)
        elif isinstance(arr, torch.Tensor):
            t = arr
        else:
            t = torch.tensor(arr)

        return t.to(dtype=dtype, device=self.device)

    def _to_indices(self, indices: Any, n: int) -> torch.Tensor:
        if indices is None:
            return torch.arange(n, dtype=torch.int64, device=self.device)

        indices = torch.as_tensor(indices, device=self.device)
        if indices.ndim != 1 or len(indices) != n:
            raise e.SizeError(f"`indices` must contain one entry per sample, but got shape {tuple(indices.shape)}.")
        if n == 0:
            return indices.to(dtype=torch.int64)
        if indices.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise e.TypeError(f"`indices` must contain integers, but got {indices.dtype}.")
        if (indices < 0).any():
            raise e.ValueError("`indices` must be non-negative.")

        return indices.to(dtype=torch.int64)

    def _get_distances(
        self,
        distance_fn: DistanceFn,
        pre_distances: torch.Tensor | None = None,
        X: torch.Tensor | None = None,
        I: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if X is None:
            X, I = self.features, self.indices
        if pre_distances is None:
            return distance_fn(self.features, X)

        if not isinstance(pre_distances, torch.Tensor):
            raise e.TypeError("`pre_distances` must be a tensor.")
        if pre_distances.ndim != 2 or pre_distances.shape[0] != pre_distances.shape[1]:
            raise e.SizeError("`pre_distances` must be a square matrix.")

        columns = self._to_indices(I, len(X))
        if (self.indices >= len(pre_distances)).any() or (columns >= len(pre_distances)).any():
            raise e.ValueError("`indices` must be within the precomputed distance matrix.")

        pre_distances = pre_distances.to(device=self.device)
        if pre_distances.is_complex():
            raise e.TypeError("`pre_distances` must be real-valued.")
        if not pre_distances.is_floating_point():
            pre_distances = pre_distances.to(dtype=torch.float64)

        return pre_distances[self.indices[:, None], columns]

    def _load(self, file_path: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        extension = file_path.split(".")[-1]

        if extension == "csv":
            data = loader.load_csv(file_path)
        elif extension == "txt":
            data = loader.load_txt(file_path)
        elif extension == "json":
            data = loader.load_json(file_path)
        else:
            raise e.ArgumentError(f"`file_path` must end in .csv, .json, or .txt, but got {file_path!r}.")

        X, Y = p.parse_loader(data)

        return X, Y

    def _init_state_tensors(self, n: int) -> None:
        dev = self.device

        self.pred_labels = torch.zeros(n, dtype=torch.int64, device=dev)
        self.cluster_labels = torch.zeros(n, dtype=torch.int64, device=dev)
        self.costs = torch.zeros(n, dtype=torch.float64, device=dev)
        self.densities = torch.zeros(n, dtype=torch.float64, device=dev)
        self.radii = torch.zeros(n, dtype=torch.float64, device=dev)
        self.n_plateaus = torch.zeros(n, dtype=torch.int64, device=dev)
        self.preds = torch.full((n,), c.NIL, dtype=torch.int64, device=dev)
        self.roots = torch.arange(n, dtype=torch.int64, device=dev)
        self.status = torch.full((n,), c.STANDARD, dtype=torch.int8, device=dev)
        self.relevant = torch.full((n,), c.IRRELEVANT, dtype=torch.int8, device=dev)

        self.idx_nodes = []
        self.adjacency = None

    def _build(
        self,
        X: torch.Tensor,
        Y: torch.Tensor | None,
        I: torch.Tensor | None,
    ) -> None:
        self.features = self._to_tensor(X, dtype=torch.float32)
        if self.features.ndim != 2:
            raise e.SizeError("`X` must have shape (n_samples, n_features).")

        n = self.features.shape[0]

        if Y is not None:
            self.labels = self._to_tensor(Y, dtype=torch.int64)
        else:
            self.labels = torch.zeros(n, dtype=torch.int64, device=self.device)
        if self.labels.ndim != 1 or len(self.labels) != n:
            raise e.SizeError("`Y` must contain one label per sample.")

        self.indices = self._to_indices(I, n)

        self._n_features = self.features.shape[1]

        self._init_state_tensors(n)

    @property
    def n_nodes(self) -> int:
        """Return the number of samples in the graph.

        Returns:
            Number of feature rows.

        """

        return self.features.shape[0]

    @property
    def n_features(self) -> int:
        """Return the dimensionality of each sample.

        Returns:
            Number of feature columns recorded when the graph was built.

        """

        return self._n_features

    def to(self, device: str | torch.device) -> Self:
        """Move all owned tensor references and device metadata in place.

        Args:
            device: Target torch device or string.

        Returns:
            This graph for chaining.

        """

        device = torch.device(device) if isinstance(device, str) else device
        self.device = device

        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.indices = self.indices.to(device)
        self.pred_labels = self.pred_labels.to(device)
        self.cluster_labels = self.cluster_labels.to(device)
        self.costs = self.costs.to(device)
        self.densities = self.densities.to(device)
        self.radii = self.radii.to(device)
        self.n_plateaus = self.n_plateaus.to(device)
        self.preds = self.preds.to(device)
        self.roots = self.roots.to(device)
        self.status = self.status.to(device)
        self.relevant = self.relevant.to(device)

        if self.adjacency is not None:
            self.adjacency = self.adjacency.to(device)

        return self

    def destroy_arcs(self) -> None:
        """Clear adjacency and plateau counts without changing sample data.

        """  # fmt: skip

        self.n_plateaus.zero_()
        self.adjacency = None

    def mark_nodes(self, i: int) -> None:
        """Mark a node and its predecessor chain as relevant.

        Args:
            i: Starting node index.

        Raises:
            e.ValueError: The starting node is outside the graph.

        """

        self._mark_predecessors([i])

    def _mark_predecessors(self, indices: list[int]) -> None:
        if not indices:
            return
        if any(not 0 <= i < self.n_nodes for i in indices):
            raise e.ValueError("`indices` must refer to nodes within the subgraph.")

        # Traverse in one CPU batch to avoid a device synchronization per predecessor
        preds = self.preds.tolist()
        relevant = self.relevant.tolist()
        marked = []
        for i in indices:
            while i != c.NIL and relevant[i] != c.RELEVANT:
                relevant[i] = c.RELEVANT
                marked.append(i)
                i = preds[i]

        if marked:
            self.relevant[marked] = c.RELEVANT

    def reset(self) -> None:
        """Reset predecessors, relevance flags, and arcs while retaining sample data.

        """  # fmt: skip

        self.preds.fill_(c.NIL)
        self.relevant.fill_(c.IRRELEVANT)
        self.destroy_arcs()
