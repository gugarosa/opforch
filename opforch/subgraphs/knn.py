"""KNN-based Subgraph with batched arc creation and PDF computation."""

from typing import Optional

import torch

import opforch.utils.constants as c
import opforch.utils.exception as e
from opforch.core.subgraph import Subgraph
from opforch.utils import logging

logger = logging.get_logger(__name__)


class KNNSubgraph(Subgraph):
    """A KNN Subgraph that extends Subgraph with k-nearest neighbour operations.

    All KNN operations (arc creation, PDF calculation) operate on pre-computed
    distance matrices via torch.topk for GPU-accelerated batch computation.
    """

    def __init__(
        self,
        X: Optional[torch.Tensor] = None,
        Y: Optional[torch.Tensor] = None,
        I: Optional[torch.Tensor] = None,
        from_file: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialization method.

        Args:
            X: Feature tensor of shape (N, D).
            Y: Label tensor of shape (N,).
            I: Index tensor of shape (N,).
            from_file: Path to load data from.
            device: Target device string.

        """

        super(KNNSubgraph, self).__init__(X, Y, I, from_file, device)

        self.n_clusters = 0
        self.best_k = 0
        self.constant = 0.0

        self.density_val = 0.0
        self.min_density = 0.0
        self.max_density = 0.0

    @property
    def density(self) -> float:
        """Alias for density_val for backward compatibility with opfython."""
        return self.density_val

    @density.setter
    def density(self, value: float) -> None:
        self.density_val = value

    def create_arcs(
        self,
        k: int,
        distance_fn: callable,
        pre_computed_distance: bool = False,
        pre_distances: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Creates KNN arcs by computing distances and using torch.topk.

        This is a convenience method that computes the distance matrix internally.
        For repeated calls with different k values, prefer create_arcs_from_matrix().

        Args:
            k: Number of nearest neighbours.
            distance_fn: Batched distance function.
            pre_computed_distance: Whether to use pre-computed distances.
            pre_distances: Pre-computed distance matrix.

        Returns:
            Tensor of maximum distances per k value, shape (k,).

        """

        if pre_computed_distance and pre_distances is not None:
            dist_matrix = pre_distances
        else:
            dist_matrix = distance_fn(self.features, self.features)

        return self.create_arcs_from_matrix(dist_matrix, k)

    def create_arcs_from_matrix(
        self, dist_matrix: torch.Tensor, k: int
    ) -> torch.Tensor:
        """Creates KNN arcs from a pre-computed distance matrix using torch.topk.

        Args:
            dist_matrix: Distance matrix of shape (N, N).
            k: Number of nearest neighbours.

        Returns:
            Tensor of maximum distances per k-position, shape (k,).

        """

        N = self.n_nodes

        # Exclude self-loops by setting diagonal to infinity
        dist_no_self = dist_matrix.clone()
        dist_no_self.fill_diagonal_(float("inf"))

        # Find k-nearest neighbours for all nodes at once
        knn_dists, knn_idx = dist_no_self.topk(k, dim=1, largest=False)

        # Store adjacency as (N, k) tensor
        self.adjacency = knn_idx.to(dtype=torch.int64)

        # Compute per-node radius (max distance among k-NN)
        finite_mask = torch.isfinite(knn_dists)
        self.radii = knn_dists[:, -1].to(dtype=torch.float64)

        # Track global density (maximum arc distance across all nodes and k positions)
        finite_dists = knn_dists[finite_mask]
        if finite_dists.numel() > 0:
            self.density_val = finite_dists.max().item()
        else:
            self.density_val = 1.0
        if self.density_val < 1e-5:
            self.density_val = 1.0

        # Per k-position maximum distances
        max_distances = knn_dists.max(dim=0).values
        max_distances = max_distances.clamp(max=self.density_val)

        # Reset plateaus
        self.n_plateaus.zero_()

        return max_distances.to(dtype=torch.float64)

    def calculate_pdf(
        self,
        n_neighbours: int,
        distance_fn: callable,
        pre_computed_distance: bool = False,
        pre_distances: Optional[torch.Tensor] = None,
    ) -> None:
        """Calculates PDF by computing distances internally.

        For repeated calls, prefer calculate_pdf_from_matrix().

        Args:
            n_neighbours: Number of neighbours.
            distance_fn: Batched distance function.
            pre_computed_distance: Whether to use pre-computed distances.
            pre_distances: Pre-computed distance matrix.

        """

        if pre_computed_distance and pre_distances is not None:
            dist_matrix = pre_distances
        else:
            dist_matrix = distance_fn(self.features, self.features)

        self.calculate_pdf_from_matrix(dist_matrix, n_neighbours)

    def calculate_pdf_from_matrix(
        self, dist_matrix: torch.Tensor, n_neighbours: int
    ) -> None:
        """Calculates the probability density function using vectorized operations.

        Args:
            dist_matrix: Distance matrix of shape (N, N).
            n_neighbours: Number of neighbours to use.

        """

        if self.adjacency is None:
            raise e.BuildError("Arcs must be created before calculating PDF")

        self.constant = 2 * self.density_val / 9

        # Gather distances for k-NN neighbours: (N, k)
        adj = self.adjacency[:, :n_neighbours]
        knn_dists = dist_matrix.gather(1, adj)

        # Compute PDF: Gaussian kernel over k-NN distances
        # opfython initializes n_pdf=1 then increments per neighbour, dividing by (k+1)
        pdf = torch.exp(-knn_dists / self.constant).sum(dim=1) / (n_neighbours + 1)  # (N,)

        self.min_density = pdf.min().item()
        self.max_density = pdf.max().item()

        if abs(self.min_density - self.max_density) < 1e-10:
            self.densities.fill_(c.MAX_DENSITY)
            self.costs.fill_(c.MAX_DENSITY - 1)
        else:
            self.densities = (
                (c.MAX_DENSITY - 1)
                * (pdf - self.min_density)
                / (self.max_density - self.min_density)
            ).to(dtype=torch.float64) + 1
            self.costs = self.densities - 1

    def eliminate_maxima_height(self, height: float) -> None:
        """Eliminates density maxima below the given height threshold.

        Args:
            height: Height threshold.

        """

        logger.debug("Eliminating maxima above height = %s ...", height)

        if height > 0:
            self.costs = torch.clamp(self.densities - height, min=0)

        logger.debug("Maxima eliminated.")

    def destroy_arcs(self) -> None:
        """Destroys all adjacency arcs and resets plateau counts."""

        self.adjacency = None
        self.n_plateaus.zero_()

    def insert_plateaus(self, n_neighbours: int) -> None:
        """Inserts reciprocal adjacency connections for equal-density nodes (plateau detection).

        For each node i and its k-NN neighbour j, if density[i] == density[j]
        and i is not already in j's adjacency, prepend i to j's adjacency.
        This matches the KNNSupervisedOPF plateau logic in opfython.

        Args:
            n_neighbours: Number of k-NN neighbours to scan.

        """

        if self.adjacency is None:
            return

        N = self.n_nodes
        # Collect plateau insertions: for each node j, a list of nodes to prepend
        to_prepend = [[] for _ in range(N)]

        for i in range(N):
            adj_i = self.adjacency[i]
            k_scan = min(n_neighbours, adj_i.numel())
            for ki in range(k_scan):
                j = adj_i[ki].item()
                if j < 0:
                    continue

                if self.densities[i].item() == self.densities[j].item():
                    # Check if i is already in j's adjacency
                    adj_j = self.adjacency[j]
                    already_present = False
                    for li in range(adj_j.numel()):
                        if adj_j[li].item() == i:
                            already_present = True
                            break
                    # Also check previously queued insertions
                    if not already_present and i not in to_prepend[j]:
                        to_prepend[j].append(i)

        # Expand adjacency tensor: prepend plateau connections
        max_extra = max(len(v) for v in to_prepend) if to_prepend else 0
        if max_extra > 0:
            k_orig = self.adjacency.shape[1]
            new_adj = torch.full(
                (N, max_extra + k_orig), -1,
                dtype=torch.int64, device=self.adjacency.device,
            )
            for i in range(N):
                extras = to_prepend[i]
                n_extra = len(extras)
                if n_extra > 0:
                    new_adj[i, :n_extra] = torch.tensor(
                        extras, dtype=torch.int64, device=self.adjacency.device
                    )
                new_adj[i, max_extra:max_extra + k_orig] = self.adjacency[i]
                self.n_plateaus[i] = len(extras)

            self.adjacency = new_adj
