"""KNN-Supervised Optimum-Path Forest (PyTorch)."""

import time
from typing import List, Optional

import torch

import opforch.math.general as g
import opforch.utils.constants as c
import opforch.utils.exception as e
from opforch.core import OPF, Heap
from opforch.subgraphs import KNNSubgraph
from opforch.utils import logging

logger = logging.get_logger(__name__)


class KNNSupervisedOPF(OPF):
    """KNN-Supervised OPF with batched KNN via torch.topk.

    References:
        J. P. Papa and A. X. Falcão.
        A Learning Algorithm for the Optimum-Path Forest Classifier.
        Graph-Based Representations in Pattern Recognition (2009).

    """

    def __init__(
        self,
        max_k: int = 1,
        distance: str = "log_squared_euclidean",
        pre_computed_distance: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialization method.

        Args:
            max_k: Maximum k value for KNN adjacency.
            distance: Distance metric name.
            pre_computed_distance: Path to pre-computed distance file.
            device: Target device string.

        """

        logger.info("Overriding class: OPF -> KNNSupervisedOPF.")

        super(KNNSupervisedOPF, self).__init__(distance, pre_computed_distance, device)

        if not isinstance(max_k, int) or max_k < 1:
            raise e.ValueError("`max_k` should be an integer >= 1")

        self.max_k = max_k

        logger.info("Class overrided.")

    def _clustering(self, force_prototype: bool = False) -> None:
        """Clusters the subgraph using density-based competition.

        Uses a max-heap where nodes compete via min(parent_cost, node_density).

        Args:
            force_prototype: Whether to force each class to have at least one prototype.

        """

        N = self.subgraph.n_nodes

        # Insert reciprocal adjacency for equal-density nodes (plateau detection)
        if self.subgraph.adjacency is not None:
            self.subgraph.insert_plateaus(self.subgraph.adjacency.shape[1])

        h = Heap(size=N, policy="max", device=str(self.device))

        for i in range(N):
            h.cost[i] = self.subgraph.costs[i]
            self.subgraph.preds[i] = c.NIL
            self.subgraph.roots[i] = i
            h.insert(i)

        self.subgraph.idx_nodes = []

        while not h.is_empty():
            p = h.remove()
            self.subgraph.idx_nodes.append(p)

            if self.subgraph.preds[p].item() == c.NIL:
                h.cost[p] = self.subgraph.densities[p]
                self.subgraph.pred_labels[p] = self.subgraph.labels[p]

            self.subgraph.costs[p] = h.cost[p]

            # Iterate over adjacency
            if self.subgraph.adjacency is not None:
                adj = self.subgraph.adjacency[p]
                for qi in range(adj.numel()):
                    q = adj[qi].item()
                    if q < 0:
                        continue

                    if h.color[q] != c.BLACK:
                        current_cost = min(
                            h.cost[p].item(), self.subgraph.densities[q].item()
                        )

                        if force_prototype:
                            if self.subgraph.labels[p] != self.subgraph.labels[q]:
                                current_cost = -c.FLOAT_MAX

                        if current_cost > h.cost[q].item():
                            self.subgraph.preds[q] = p
                            self.subgraph.roots[q] = self.subgraph.roots[p]
                            self.subgraph.pred_labels[q] = self.subgraph.pred_labels[p]
                            h.update(q, current_cost)

    def _learn(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        I_train: Optional[torch.Tensor],
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        I_val: Optional[torch.Tensor],
    ) -> None:
        """Learns the best k value over the validation set.

        Computes the distance matrix once, then iterates over k values.

        Args:
            X_train: Training features.
            Y_train: Training labels.
            I_train: Training indices.
            X_val: Validation features.
            Y_val: Validation labels.
            I_val: Validation indices.

        """

        logger.info("Learning best `k` value ...")

        self.subgraph = KNNSubgraph(X_train, Y_train, I_train, device=str(self.device))

        # Compute distance matrix once for all k values
        dist_matrix = self._get_distances()

        max_acc = 0.0
        best_k = 1

        for k in range(1, self.max_k + 1):
            self.subgraph.best_k = k

            self.subgraph.create_arcs_from_matrix(dist_matrix, k)
            self.subgraph.calculate_pdf_from_matrix(dist_matrix, k)

            self._clustering()

            preds = self.predict(X_val, I_val)

            acc = g.opf_accuracy(Y_val, preds)
            if acc > max_acc:
                max_acc = acc
                best_k = k

            logger.info("Accuracy over k = %d: %s", k, acc)

            self.subgraph.destroy_arcs()

        self.subgraph.best_k = best_k

    def fit(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        I_train: Optional[torch.Tensor] = None,
        I_val: Optional[torch.Tensor] = None,
    ) -> None:
        """Fits data in the classifier.

        Args:
            X_train: Training features.
            Y_train: Training labels.
            X_val: Validation features.
            Y_val: Validation labels.
            I_train: Training indices.
            I_val: Validation indices.

        """

        logger.info("Fitting classifier ...")

        start = time.time()

        self._learn(X_train, Y_train, I_train, X_val, Y_val, I_val)

        # Recompute distance matrix for final clustering
        dist_matrix = self._get_distances()

        self.subgraph.create_arcs_from_matrix(dist_matrix, self.subgraph.best_k)
        self.subgraph.calculate_pdf_from_matrix(dist_matrix, self.subgraph.best_k)

        self._clustering(force_prototype=True)

        self.subgraph.destroy_arcs()

        self.subgraph.trained = True

        end = time.time()
        logger.info("Classifier has been fitted with k = %d.", self.subgraph.best_k)
        logger.info("Training time: %s seconds.", end - start)

    def predict(
        self,
        X_test: torch.Tensor,
        I_test: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """Predicts new data using batched KNN + density competition.

        Args:
            X_test: Test features of shape (M, D).
            I_test: Test indices.

        Returns:
            A list of predicted labels.

        """

        logger.info("Predicting data ...")

        start = time.time()

        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test, dtype=torch.float32)
        X_test = X_test.to(dtype=torch.float32, device=self.device)

        # Compute train→test distances: (N_train, M_test)
        dist_matrix = self._get_distances(X_test, I_test)

        best_k = self.subgraph.best_k
        # Find k-nearest training nodes for each test sample
        knn_dists, knn_idx = dist_matrix.topk(best_k, dim=0, largest=False)
        # knn_dists: (k, M), knn_idx: (k, M)

        # Compute density for each test sample
        density = torch.exp(-knn_dists / self.subgraph.constant).mean(dim=0)  # (M,)
        density = (
            (c.MAX_DENSITY - 1)
            * (density - self.subgraph.min_density)
            / (self.subgraph.max_density - self.subgraph.min_density + c.EPSILON)
        ) + 1

        # Find best conqueror among k-NN
        neighbour_costs = self.subgraph.costs[knn_idx]  # (k, M)
        compete_costs = torch.minimum(neighbour_costs, density.unsqueeze(0))  # (k, M)
        best_k_idx = compete_costs.argmax(dim=0)  # (M,)

        # Gather predictions
        best_neighbours = knn_idx.gather(0, best_k_idx.unsqueeze(0)).squeeze(0)
        predictions = self.subgraph.pred_labels[best_neighbours]

        end = time.time()
        logger.info("Data has been predicted.")
        logger.info("Prediction time: %s seconds.", end - start)

        return predictions.cpu().tolist()
