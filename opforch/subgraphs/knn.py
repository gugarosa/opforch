# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Construct k-nearest-neighbour graphs and their density-based paths.

Arc and density operations consume distance matrices in local sample order.
Precomputed-file wrappers select that order through the graph's original indices.

"""

from __future__ import annotations

import torch

import opforch.utils.constants as c
import opforch.utils.exception as e
from opforch.core.subgraph import Subgraph
from opforch.math.distance import DistanceFn
from opforch.utils.logging import get_logger

logger = get_logger(__name__)


class KNNSubgraph(Subgraph):
    """Extend sample state with nearest-neighbour arcs and density information.

    """  # fmt: skip

    def __init__(
        self,
        X: torch.Tensor | None = None,
        Y: torch.Tensor | None = None,
        I: torch.Tensor | None = None,
        from_file: str | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize sample data and empty neighbour state.

        Args:
            X: Feature tensor of shape (N, D).
            Y: Label tensor of shape (N,).
            I: Index tensor of shape (N,).
            from_file: Path to load data from.
            device: Target device, or automatic CUDA/CPU selection when None.

        """

        super().__init__(X, Y, I, from_file, device)

        self.n_clusters = 0
        self.best_k = 0
        self.constant = 0.0

        self.density_val = 0.0
        self.min_density = 0.0
        self.max_density = 0.0

    @property
    def density(self) -> float:
        """Return the distance scale used for the density bandwidth.

        Returns:
            The graph-wide density_val scale.

        """

        return self.density_val

    @density.setter
    def density(self, value: float) -> None:
        """Set the distance scale used for the density bandwidth.

        Args:
            value: Graph-wide density_val scale.

        """

        self.density_val = value

    def create_arcs(
        self,
        k: int,
        distance_fn: DistanceFn,
        pre_computed_distance: bool = False,
        pre_distances: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Replace neighbour arcs using computed or precomputed distances.

        Use create_arcs_from_matrix() to reuse an existing matrix across k values.

        Args:
            k: Number of nearest neighbours.
            distance_fn: Batched distance function.
            pre_computed_distance: Whether to use pre-computed distances.
            pre_distances: Global distance matrix indexed by this graph's sample indices.

        Returns:
            Tensor of maximum distances per k value, shape (k,).

        Raises:
            e.BuildError: Precomputed distances were requested but not provided.
            e.ValueError: The neighbour count or an original sample index is invalid.
            e.SizeError: Distance or index dimensions are invalid.

        """

        if pre_computed_distance and pre_distances is None:
            raise e.BuildError("`pre_distances` is None while pre_computed_distance is True.")

        dist_matrix = self._get_distances(distance_fn, pre_distances if pre_computed_distance else None)

        return self.create_arcs_from_matrix(dist_matrix, k)

    def create_arcs_from_matrix(self, dist_matrix: torch.Tensor, k: int) -> torch.Tensor:
        """Replace neighbour arcs from a local distance matrix.

        Self-arcs are excluded without modifying the supplied matrix. The graph's
        adjacency, radii, distance scale, and plateau counts are replaced.

        Args:
            dist_matrix: Distance matrix of shape (N, N).
            k: Number of nearest neighbours, from 1 to N - 1.

        Returns:
            Tensor of maximum distances per k-position, shape (k,).

        Raises:
            e.ValueError: The neighbour count is outside the valid range.
            e.SizeError: The matrix does not describe every local node pair.

        """

        if not 1 <= k < self.n_nodes:
            raise e.ValueError(f"`k` must be between 1 and n_nodes - 1, but got {k}.")
        if dist_matrix.shape != (self.n_nodes, self.n_nodes):
            raise e.SizeError("`dist_matrix` must have shape (n_nodes, n_nodes).")

        dist_no_self = dist_matrix.clone()
        dist_no_self.fill_diagonal_(float("inf"))
        knn_dists, knn_idx = dist_no_self.topk(k, dim=1, largest=False)

        self.adjacency = knn_idx.to(dtype=torch.int64)
        finite_mask = torch.isfinite(knn_dists)
        self.radii = knn_dists[:, -1].to(dtype=torch.float64)

        finite_dists = knn_dists[finite_mask]
        if finite_dists.numel() > 0:
            self.density_val = finite_dists.max().item()
        else:
            self.density_val = 1.0
        if self.density_val < 1e-5:
            self.density_val = 1.0

        max_distances = knn_dists.max(dim=0).values
        max_distances = max_distances.clamp(max=self.density_val)
        self.n_plateaus.zero_()

        return max_distances.to(dtype=torch.float64)

    def calculate_pdf(
        self,
        n_neighbours: int,
        distance_fn: DistanceFn,
        pre_computed_distance: bool = False,
        pre_distances: torch.Tensor | None = None,
    ) -> None:
        """Replace node densities and initial costs using the current arcs.

        Use calculate_pdf_from_matrix() when a local distance matrix is already available.

        Args:
            n_neighbours: Number of neighbours.
            distance_fn: Batched distance function.
            pre_computed_distance: Whether to use pre-computed distances.
            pre_distances: Global distance matrix indexed by this graph's sample indices.

        Raises:
            e.BuildError: The graph has no arcs or the requested distance matrix is unavailable.

        """

        if pre_computed_distance and pre_distances is None:
            raise e.BuildError("`pre_distances` is None while pre_computed_distance is True.")

        dist_matrix = self._get_distances(distance_fn, pre_distances if pre_computed_distance else None)

        self.calculate_pdf_from_matrix(dist_matrix, n_neighbours)

    def calculate_pdf_from_matrix(self, dist_matrix: torch.Tensor, n_neighbours: int) -> None:
        """Replace node densities and initial costs from a local distance matrix.

        The training kernel sum is divided by n_neighbours + 1, matching the
        historical OPF convention. Densities are then scaled to the configured
        density range, with uniform densities assigned MAX_DENSITY.

        Args:
            dist_matrix: Distance matrix of shape (N, N).
            n_neighbours: Number of neighbours to use.

        Raises:
            e.BuildError: The graph's arcs have not been created.

        """

        if self.adjacency is None:
            raise e.BuildError("`adjacency` is None before density calculation.")

        self.constant = 2 * self.density_val / 9

        adj = self.adjacency[:, :n_neighbours]
        knn_dists = dist_matrix.gather(1, adj)

        pdf = torch.exp(-knn_dists / self.constant).sum(dim=1) / (n_neighbours + 1)

        self.min_density = pdf.min().item()
        self.max_density = pdf.max().item()

        if abs(self.min_density - self.max_density) < 1e-10:
            self.densities.fill_(c.MAX_DENSITY)
            self.costs.fill_(c.MAX_DENSITY - 1)
        else:
            self.densities = (
                (c.MAX_DENSITY - 1) * (pdf - self.min_density) / (self.max_density - self.min_density)
            ).to(dtype=torch.float64) + 1
            self.costs = self.densities - 1

    def _conquerors(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        knn_dists, knn_idx = dist_matrix.topk(self.best_k, dim=0, largest=False)

        density = torch.exp(-knn_dists / self.constant).mean(dim=0)
        density = (
            (c.MAX_DENSITY - 1) * (density - self.min_density) / (self.max_density - self.min_density + c.EPSILON)
        ) + 1

        neighbour_costs = self.costs[knn_idx]
        compete_costs = torch.minimum(neighbour_costs, density.unsqueeze(0))
        best_k_idx = compete_costs.argmax(dim=0)

        return knn_idx.gather(0, best_k_idx.unsqueeze(0)).squeeze(0)

    def eliminate_maxima_height(self, height: float) -> None:
        """Lower initial path costs by a positive density height.

        Nonpositive heights leave the current costs unchanged.

        Args:
            height: Height threshold.

        """

        logger.debug("Lowering initial density costs by %s.", height)

        if height > 0:
            self.costs = torch.clamp(self.densities - height, min=0)

        logger.debug("Maxima eliminated.")

    def insert_plateaus(self, n_neighbours: int) -> None:
        """Insert reciprocal arcs between equal-density neighbours.

        For each node i and its k-NN neighbour j, if density[i] == density[j]
        and i is not already in j's adjacency, prepend i to j's adjacency.
        This matches the KNNSupervisedOPF plateau logic in opfython.

        Args:
            n_neighbours: Number of k-NN neighbours to scan.

        """

        if self.adjacency is None:
            return

        N = self.n_nodes
        # Queue reciprocal arcs before mutating adjacency so every row uses the same snapshot
        to_prepend = [[] for _ in range(N)]

        for i in range(N):
            adj_i = self.adjacency[i]
            k_scan = min(n_neighbours, adj_i.numel())
            for ki in range(k_scan):
                j = adj_i[ki].item()
                if j < 0:
                    continue

                if self.densities[i].item() == self.densities[j].item():
                    adj_j = self.adjacency[j]
                    if i not in adj_j and i not in to_prepend[j]:
                        to_prepend[j].append(i)

        max_extra = max(len(v) for v in to_prepend) if to_prepend else 0
        if max_extra > 0:
            k_orig = self.adjacency.shape[1]
            new_adj = torch.full(
                (N, max_extra + k_orig),
                -1,
                dtype=torch.int64,
                device=self.adjacency.device,
            )
            for i in range(N):
                extras = to_prepend[i]
                n_extra = len(extras)
                if n_extra > 0:
                    new_adj[i, :n_extra] = torch.tensor(extras, dtype=torch.int64, device=self.adjacency.device)
                new_adj[i, n_extra : n_extra + k_orig] = self.adjacency[i]
                self.n_plateaus[i] = len(extras)

            self.adjacency = new_adj
