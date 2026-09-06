# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Train density-based supervised forests over k-nearest-neighbour graphs.

References:
    J. P. Papa and A. X. Falcão, A Learning Algorithm for the Optimum-Path Forest Classifier (2009).

"""

from __future__ import annotations

import time

import torch

import opforch.math.general as g
import opforch.utils.constants as c
import opforch.utils.exception as e
from opforch.core.heap import Heap
from opforch.core.opf import OPF
from opforch.subgraphs.knn import KNNSubgraph
from opforch.utils.logging import get_logger

logger = get_logger(__name__)


class KNNSupervisedOPF(OPF):
    """Classify samples with validation-selected k-nearest-neighbour forests.

    """  # fmt: skip

    def __init__(
        self,
        max_k: int = 1,
        distance: str = "log_squared_euclidean",
        pre_computed_distance: str | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize a k-nearest-neighbour supervised classifier.

        Args:
            max_k: Maximum k value for KNN adjacency.
            distance: Distance metric name.
            pre_computed_distance: Path to pre-computed distance file.
            device: Target device, or automatic CUDA/CPU selection when None.

        Raises:
            e.ValueError: The maximum neighbour count is not a positive integer.

        """

        logger.info("Creating k-nearest-neighbour supervised OPF classifier.")

        super().__init__(distance, pre_computed_distance, device)

        if not isinstance(max_k, int) or max_k < 1:
            raise e.ValueError(f"`max_k` must be a positive integer, but got {max_k}.")

        self.max_k = max_k

    def _clustering(self, force_prototype: bool = False) -> None:
        N = self.subgraph.n_nodes

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

            if self.subgraph.adjacency is not None:
                adj = self.subgraph.adjacency[p]
                for qi in range(adj.numel()):
                    q = adj[qi].item()
                    if q < 0:
                        continue

                    if h.color[q] != c.BLACK:
                        current_cost = min(h.cost[p].item(), self.subgraph.densities[q].item())

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
        I_train: torch.Tensor | None,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        I_val: torch.Tensor | None,
    ) -> torch.Tensor:
        logger.info("Learning the best neighbour count.")

        self.subgraph = KNNSubgraph(X_train, Y_train, I_train, device=str(self.device))

        dist_matrix = self._get_distances()

        max_acc = 0.0
        best_k = 1

        for k in range(1, self.max_k + 1):
            self.subgraph.best_k = k

            self.subgraph.create_arcs_from_matrix(dist_matrix, k)
            self.subgraph.calculate_pdf_from_matrix(dist_matrix, k)

            self._clustering()

            preds = self._predict(X_val, I_val)

            acc = g.opf_accuracy(Y_val, preds)
            if acc > max_acc:
                max_acc = acc
                best_k = k

            logger.info("Accuracy over k = %d: %s", k, acc)

            self.subgraph.destroy_arcs()

        self.subgraph.best_k = best_k

        return dist_matrix

    def fit(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        I_train: torch.Tensor | None = None,
        I_val: torch.Tensor | None = None,
    ) -> None:
        """Replace the fitted graph using validation-selected neighbour counts.

        Each candidate competes through minimum parent-cost/node-density paths.
        The training distance matrix is reused for all candidates and the final
        graph. Final competition preserves a prototype for each class.

        Args:
            X_train: Training features of shape (N, D).
            Y_train: Training labels of shape (N,).
            X_val: Validation features of shape (M, D).
            Y_val: Validation labels of shape (M,).
            I_train: Original matrix positions for training samples.
            I_val: Original matrix positions for validation samples.

        Raises:
            e.SizeError: Feature, label, or index dimensions do not align.
            e.ValueError: A neighbour count or original sample index is invalid.

        """

        logger.info("Fitting classifier ...")

        start = time.time()

        dist_matrix = self._learn(X_train, Y_train, I_train, X_val, Y_val, I_val)

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
        I_test: torch.Tensor | None = None,
    ) -> list[int]:
        """Predict labels with the fitted neighbour count and density forest.

        Args:
            X_test: Test features of shape (M, D).
            I_test: Original matrix positions for test samples.

        Returns:
            One predicted class label per input row, in input order.

        Raises:
            e.BuildError: The classifier has not completed a successful fit.
            e.SizeError: Precomputed distance or index dimensions are invalid.
            e.ValueError: Original sample indices are outside the distance matrix.

        """

        self._check_fitted()

        return self._predict(X_test, I_test)

    def _predict(self, X_test: torch.Tensor, I_test: torch.Tensor | None = None) -> list[int]:
        logger.info("Predicting data.")

        start = time.time()

        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test, dtype=torch.float32)
        X_test = X_test.to(dtype=torch.float32, device=self.device)

        dist_matrix = self._get_distances(X_test, I_test)

        best_neighbours = self.subgraph._conquerors(dist_matrix)
        predictions = self.subgraph.pred_labels[best_neighbours]

        end = time.time()
        logger.info("Data has been predicted.")
        logger.info("Prediction time: %s seconds.", end - start)

        return predictions.cpu().tolist()
