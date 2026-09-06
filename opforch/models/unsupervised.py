# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Cluster samples using density-based optimum-path forests and normalized cuts.

References:
    L. M. Rocha, F. A. M. Cappabianco and A. X. Falcão, Data clustering as an optimum-path forest problem (2009).

"""

from __future__ import annotations

import time

import torch

import opforch.utils.constants as c
import opforch.utils.exception as e
from opforch.core.heap import Heap
from opforch.core.opf import OPF
from opforch.subgraphs.knn import KNNSubgraph
from opforch.utils.logging import get_logger

logger = get_logger(__name__)


class UnsupervisedOPF(OPF):
    """Cluster samples using density competition and normalized-cut selection.

    """  # fmt: skip

    def __init__(
        self,
        min_k: int = 1,
        max_k: int = 1,
        distance: str = "log_squared_euclidean",
        pre_computed_distance: str | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize the neighbour search range and distance policy.

        Args:
            min_k: Minimum k value for KNN.
            max_k: Maximum k value for KNN.
            distance: Distance metric name.
            pre_computed_distance: Path to pre-computed distance file.
            device: Target device, or automatic CUDA/CPU selection when None.

        Raises:
            e.ValueError: The neighbour range is invalid.

        """

        logger.info("Creating unsupervised OPF classifier.")

        super().__init__(distance, pre_computed_distance, device)

        if not isinstance(min_k, int) or min_k < 1:
            raise e.ValueError(f"`min_k` must be a positive integer, but got {min_k}.")
        if not isinstance(max_k, int) or max_k < 1:
            raise e.ValueError(f"`max_k` must be a positive integer, but got {max_k}.")
        if max_k < min_k:
            raise e.ValueError(f"`max_k` must be at least min_k, but got {max_k}.")

        self.min_k = min_k
        self.max_k = max_k

    def _clustering(self, n_neighbours: int) -> None:
        N = self.subgraph.n_nodes

        if self.subgraph.adjacency is not None:
            self.subgraph.insert_plateaus(n_neighbours)

        h = Heap(size=N, policy="max", device=str(self.device))

        for i in range(N):
            h.cost[i] = self.subgraph.costs[i]
            self.subgraph.preds[i] = c.NIL
            self.subgraph.roots[i] = i
            h.insert(i)

        self.subgraph.idx_nodes = []
        cluster_id = 0

        while not h.is_empty():
            p = h.remove()
            self.subgraph.idx_nodes.append(p)

            if self.subgraph.preds[p].item() == c.NIL:
                h.cost[p] = self.subgraph.densities[p]
                self.subgraph.cluster_labels[p] = cluster_id
                cluster_id += 1

            self.subgraph.costs[p] = h.cost[p]

            if self.subgraph.adjacency is not None:
                adj = self.subgraph.adjacency[p]
                n_adj = min(n_neighbours + self.subgraph.n_plateaus[p].item(), adj.numel())
                for ki in range(n_adj):
                    q = adj[ki].item()
                    if q < 0:
                        continue

                    if h.color[q] != c.BLACK:
                        current_cost = min(h.cost[p].item(), self.subgraph.densities[q].item())

                        if current_cost > h.cost[q].item():
                            self.subgraph.preds[q] = p
                            self.subgraph.roots[q] = self.subgraph.roots[p]
                            self.subgraph.cluster_labels[q] = self.subgraph.cluster_labels[p]
                            h.update(q, current_cost)

        self.subgraph.n_clusters = cluster_id

    def _normalized_cut(
        self,
        n_neighbours: int,
        dist_matrix: torch.Tensor,
    ) -> float:
        N = self.subgraph.n_nodes
        n_clusters = self.subgraph.n_clusters

        if n_clusters == 0:
            return c.FLOAT_MAX

        cluster_labels = self.subgraph.cluster_labels

        internal = torch.zeros(n_clusters, dtype=torch.float64, device=self.device)
        external = torch.zeros(n_clusters, dtype=torch.float64, device=self.device)

        for i in range(N):
            if self.subgraph.adjacency is None:
                continue
            adj = self.subgraph.adjacency[i]
            n_adj = min(n_neighbours + self.subgraph.n_plateaus[i].item(), adj.numel())

            for ki in range(n_adj):
                j = adj[ki].item()
                if j < 0:
                    continue

                dist_val = dist_matrix[i, j].item()
                if dist_val > 0.0:
                    cl_i = cluster_labels[i].item()
                    cl_j = cluster_labels[j].item()
                    if cl_i == cl_j:
                        internal[cl_i] += 1.0 / dist_val
                    else:
                        external[cl_i] += 1.0 / dist_val

        total = internal + external
        cut = 0.0
        for l_idx in range(n_clusters):
            if total[l_idx].item() > 0.0:
                cut += external[l_idx].item() / total[l_idx].item()

        return cut

    def _best_minimum_cut(
        self,
        min_k: int,
        max_k: int,
        dist_matrix: torch.Tensor,
    ) -> None:
        logger.debug("Calculating the best minimum cut within [%d, %d] ...", min_k, max_k)

        max_distances = self.subgraph.create_arcs_from_matrix(dist_matrix, max_k)
        neighbours = self.subgraph.adjacency

        min_cut = c.FLOAT_MAX
        best_k = min_k

        for k in range(min_k, max_k + 1):
            if min_cut != 0.0:
                # Each candidate needs its own neighbour graph, not the previous plateaus
                self.subgraph.adjacency = neighbours[:, :k].clone()
                self.subgraph.n_plateaus.zero_()
                self.subgraph.density_val = max_distances[k - 1].item()
                if self.subgraph.density_val < 1e-5:
                    self.subgraph.density_val = 1.0
                self.subgraph.best_k = k
                self.subgraph.calculate_pdf_from_matrix(dist_matrix, k)

                self._clustering(k)

                cut = self._normalized_cut(k, dist_matrix)
                if cut < min_cut:
                    min_cut = cut
                    best_k = k

        self.subgraph.destroy_arcs()
        self.subgraph.best_k = best_k

        self.subgraph.create_arcs_from_matrix(dist_matrix, best_k)
        self.subgraph.calculate_pdf_from_matrix(dist_matrix, best_k)

        logger.debug("Best: %d | Minimum cut: %s.", best_k, min_cut)

    def fit(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor | None = None,
        I_train: torch.Tensor | None = None,
    ) -> None:
        """Replace the graph with density-based clusters.

        Neighbour selection minimizes a normalized cut using reciprocal positive
        arc distances as affinity. Nonpositive distances contribute no cut affinity.
        Call propagate_labels() after fitting to assign class labels from cluster roots.

        Args:
            X_train: Training features of shape (N, D).
            Y_train: Optional class labels of shape (N,) for propagation or evaluation.
            I_train: Original matrix positions for training samples.

        Raises:
            e.SizeError: Feature, label, or index dimensions do not align.
            e.ValueError: A neighbour count or original sample index is invalid.

        """

        logger.info("Clustering with classifier ...")

        start = time.time()

        self.subgraph = KNNSubgraph(X_train, Y_train, I_train, device=str(self.device))

        dist_matrix = self._get_distances()

        self._best_minimum_cut(self.min_k, self.max_k, dist_matrix)
        self._clustering(self.subgraph.best_k)

        self.subgraph.trained = True

        end = time.time()
        logger.info("Classifier has been clustered with.")
        logger.info("Number of clusters: %d.", self.subgraph.n_clusters)
        logger.info("Clustering time: %s seconds.", end - start)

    def predict(
        self,
        X_val: torch.Tensor,
        I_val: torch.Tensor | None = None,
    ) -> tuple[list[int], list[int]]:
        """Predict class labels and cluster assignments for new samples.

        Class labels use the state populated by propagate_labels(). They remain
        zero if no root-label propagation has been performed.

        Args:
            X_val: Validation features of shape (M, D).
            I_val: Original matrix positions for validation samples.

        Returns:
            A pair of class-label and cluster-label lists in input order.

        Raises:
            e.BuildError: The classifier has not been fitted.
            e.SizeError: Precomputed distance or index dimensions are invalid.
            e.ValueError: Original sample indices are outside the distance matrix.

        """

        self._check_fitted()

        logger.info("Predicting data ...")

        start = time.time()

        if not isinstance(X_val, torch.Tensor):
            X_val = torch.tensor(X_val, dtype=torch.float32)
        X_val = X_val.to(dtype=torch.float32, device=self.device)

        dist_matrix = self._get_distances(X_val, I_val)

        best_neighbours = self.subgraph._conquerors(dist_matrix)
        pred_labels = self.subgraph.pred_labels[best_neighbours]
        cluster_labels = self.subgraph.cluster_labels[best_neighbours]

        end = time.time()
        logger.info("Data has been predicted.")
        logger.info("Prediction time: %s seconds.", end - start)

        return pred_labels.cpu().tolist(), cluster_labels.cpu().tolist()

    def propagate_labels(self) -> None:
        """Copy each cluster root's class label to its member nodes.

        Raises:
            e.BuildError: The classifier has not been fitted.

        """

        self._check_fitted()

        logger.info("Assigning predicted labels from clusters ...")

        self.subgraph.pred_labels.copy_(self.subgraph.labels[self.subgraph.roots])

        logger.info("Labels assigned.")
