"""Unsupervised Optimum-Path Forest (PyTorch)."""

import time
from typing import List, Optional, Tuple

import torch

import opforch.utils.constants as c
import opforch.utils.exception as e
from opforch.core import OPF, Heap
from opforch.subgraphs import KNNSubgraph
from opforch.utils import logging

logger = logging.get_logger(__name__)


class UnsupervisedOPF(OPF):
    """Unsupervised OPF for clustering with batched KNN and normalized cut.

    References:
        L. M. Rocha, F. A. M. Cappabianco, A. X. Falcão.
        Data clustering as an optimum-path forest problem with applications in image analysis.
        International Journal of Imaging Systems and Technology (2009).

    """

    def __init__(
        self,
        min_k: int = 1,
        max_k: int = 1,
        distance: str = "log_squared_euclidean",
        pre_computed_distance: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialization method.

        Args:
            min_k: Minimum k value for KNN.
            max_k: Maximum k value for KNN.
            distance: Distance metric name.
            pre_computed_distance: Path to pre-computed distance file.
            device: Target device string.

        """

        logger.info("Overriding class: OPF -> UnsupervisedOPF.")

        super(UnsupervisedOPF, self).__init__(distance, pre_computed_distance, device)

        if not isinstance(min_k, int) or min_k < 1:
            raise e.ValueError("`min_k` should be an integer >= 1")
        if not isinstance(max_k, int) or max_k < 1:
            raise e.ValueError("`max_k` should be an integer >= 1")
        if max_k < min_k:
            raise e.ValueError("`max_k` should be >= `min_k`")

        self.min_k = min_k
        self.max_k = max_k

        logger.info("Class overrided.")

    def _clustering(self, n_neighbours: int) -> None:
        """Clusters the subgraph using max-heap density competition.

        Args:
            n_neighbours: Number of neighbours to consider.

        """

        N = self.subgraph.n_nodes

        # Insert reciprocal adjacency for equal-density nodes (plateau detection)
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

            # Iterate over adjacency
            if self.subgraph.adjacency is not None:
                adj = self.subgraph.adjacency[p]
                n_adj = min(
                    n_neighbours + self.subgraph.n_plateaus[p].item(), adj.numel()
                )
                for ki in range(n_adj):
                    q = adj[ki].item()
                    if q < 0:
                        continue

                    if h.color[q] != c.BLACK:
                        current_cost = min(
                            h.cost[p].item(), self.subgraph.densities[q].item()
                        )

                        if current_cost > h.cost[q].item():
                            self.subgraph.preds[q] = p
                            self.subgraph.roots[q] = self.subgraph.roots[p]
                            self.subgraph.cluster_labels[q] = (
                                self.subgraph.cluster_labels[p]
                            )
                            h.update(q, current_cost)

        self.subgraph.n_clusters = cluster_id

    def _normalized_cut(
        self,
        n_neighbours: int,
        dist_matrix: torch.Tensor,
    ) -> float:
        """Computes normalized cut using tensor operations.

        Args:
            n_neighbours: Number of neighbours.
            dist_matrix: Pre-computed distance matrix (N, N).

        Returns:
            The normalized cut value.

        """

        N = self.subgraph.n_nodes
        n_clusters = self.subgraph.n_clusters

        if n_clusters == 0:
            return c.FLOAT_MAX

        cluster_labels = self.subgraph.cluster_labels  # (N,)

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
        """Finds the best k value that minimizes the normalized cut.

        Args:
            min_k: Minimum k.
            max_k: Maximum k.
            dist_matrix: Pre-computed (N, N) distance matrix.

        """

        logger.debug(
            "Calculating the best minimum cut within [%d, %d] ...", min_k, max_k
        )

        max_distances = self.subgraph.create_arcs_from_matrix(dist_matrix, max_k)

        min_cut = c.FLOAT_MAX
        best_k = min_k

        for k in range(min_k, max_k + 1):
            if min_cut != 0.0:
                self.subgraph.density_val = max_distances[k - 1].item()
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
        Y_train: Optional[torch.Tensor] = None,
        I_train: Optional[torch.Tensor] = None,
    ) -> None:
        """Fits (clusters) data in the classifier.

        Args:
            X_train: Training features of shape (N, D).
            Y_train: Training labels (optional, used for evaluation).
            I_train: Training indices.

        """

        logger.info("Clustering with classifier ...")

        start = time.time()

        self.subgraph = KNNSubgraph(X_train, Y_train, I_train, device=str(self.device))

        # Compute distance matrix once
        if self.pre_computed_distance:
            dist_matrix = self.pre_distances
        else:
            dist_matrix = self.distance_fn(
                self.subgraph.features, self.subgraph.features
            )

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
        I_val: Optional[torch.Tensor] = None,
    ) -> Tuple[List[int], List[int]]:
        """Predicts labels and cluster assignments for new data.

        Args:
            X_val: Validation features of shape (M, D).
            I_val: Validation indices.

        Returns:
            Tuple of (predicted_labels, cluster_labels) as lists.

        """

        if self.subgraph is None:
            raise e.BuildError("KNNSubgraph has not been properly created")
        if not self.subgraph.trained:
            raise e.BuildError("Classifier has not been properly clustered")

        logger.info("Predicting data ...")

        start = time.time()

        if not isinstance(X_val, torch.Tensor):
            X_val = torch.tensor(X_val, dtype=torch.float32)
        X_val = X_val.to(dtype=torch.float32, device=self.device)

        # Compute train→test distances: (N_train, M_test)
        if self.pre_computed_distance:
            dist_matrix = self.pre_distances
        else:
            dist_matrix = self.distance_fn(self.subgraph.features, X_val)

        best_k = self.subgraph.best_k
        # Find k-nearest training nodes for each test sample
        knn_dists, knn_idx = dist_matrix.topk(best_k, dim=0, largest=False)

        # Compute density for each test sample
        density = torch.exp(-knn_dists / self.subgraph.constant).mean(dim=0)
        density = (
            (c.MAX_DENSITY - 1)
            * (density - self.subgraph.min_density)
            / (self.subgraph.max_density - self.subgraph.min_density + c.EPSILON)
        ) + 1

        # Find best conqueror among k-NN
        neighbour_costs = self.subgraph.costs[knn_idx]
        compete_costs = torch.minimum(neighbour_costs, density.unsqueeze(0))
        best_k_idx = compete_costs.argmax(dim=0)

        best_neighbours = knn_idx.gather(0, best_k_idx.unsqueeze(0)).squeeze(0)
        pred_labels = self.subgraph.pred_labels[best_neighbours]
        cluster_labels = self.subgraph.cluster_labels[best_neighbours]

        end = time.time()
        logger.info("Data has been predicted.")
        logger.info("Prediction time: %s seconds.", end - start)

        return pred_labels.cpu().tolist(), cluster_labels.cpu().tolist()

    def propagate_labels(self) -> None:
        """Propagates root labels to all nodes in each cluster tree."""

        logger.info("Assigning predicted labels from clusters ...")

        for i in range(self.subgraph.n_nodes):
            root = self.subgraph.roots[i].item()

            if root == i:
                self.subgraph.pred_labels[i] = self.subgraph.labels[i]
            else:
                self.subgraph.pred_labels[i] = self.subgraph.labels[root]

        logger.info("Labels assigned.")
