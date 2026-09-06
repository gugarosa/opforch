# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Train supervised optimum-path forests with batched distance computation.

References:
    J. P. Papa, A. X. Falcão and C. T. N. Suzuki, Supervised Pattern Classification based on OPF (2009).

"""

from __future__ import annotations

import copy
import time

import torch

import opforch.math.general as g
import opforch.math.random as r
import opforch.utils.constants as c
import opforch.utils.exception as e
from opforch.core.opf import OPF
from opforch.core.subgraph import Subgraph
from opforch.utils.logging import get_logger

logger = get_logger(__name__)


class SupervisedOPF(OPF):
    """Classify samples using supervised minimax path competition.

    """  # fmt: skip

    def __init__(
        self,
        distance: str = "log_squared_euclidean",
        pre_computed_distance: str | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize a supervised classifier.

        Args:
            distance: Distance metric name.
            pre_computed_distance: Path to a pre-computed distance file.
            device: Target device, or automatic CUDA/CPU selection when None.

        """

        logger.info("Creating supervised OPF classifier.")

        super().__init__(distance, pre_computed_distance, device)

    def _find_prototypes(self, dist_matrix: torch.Tensor) -> None:
        logger.debug("Finding prototypes ...")

        N = self.subgraph.n_nodes
        if N == 0:
            raise e.SizeError("`X_train` must contain at least one sample.")
        if (self.subgraph.labels == self.subgraph.labels[0]).all():
            # Without a class boundary, one seed still defines a complete forest
            self.subgraph.status[0] = c.PROTOTYPE
            return

        in_tree = torch.zeros(N, dtype=torch.bool, device=self.device)
        costs = torch.full((N,), c.FLOAT_MAX, dtype=torch.float64, device=self.device)
        preds = torch.full((N,), c.NIL, dtype=torch.int64, device=self.device)

        costs[0] = 0.0

        prototypes = []
        for _ in range(N):
            candidates = costs.clone()
            candidates[in_tree] = c.FLOAT_MAX
            p = candidates.argmin().item()
            in_tree[p] = True

            pred_idx = preds[p].item()
            if pred_idx != c.NIL:
                if self.subgraph.labels[p] != self.subgraph.labels[pred_idx]:
                    if self.subgraph.status[p] != c.PROTOTYPE:
                        self.subgraph.status[p] = c.PROTOTYPE
                        prototypes.append(p)
                    if self.subgraph.status[pred_idx] != c.PROTOTYPE:
                        self.subgraph.status[pred_idx] = c.PROTOTYPE
                        prototypes.append(pred_idx)

            mask = ~in_tree
            new_costs = dist_matrix[p][mask]
            improved = new_costs < costs[mask]

            mask_indices = mask.nonzero(as_tuple=True)[0]
            improved_indices = mask_indices[improved]

            costs[improved_indices] = new_costs[improved].to(dtype=torch.float64)
            preds[improved_indices] = p

        self.subgraph.preds = preds

        logger.debug("Prototypes: %s.", prototypes)

    def _compete(self, dist_matrix: torch.Tensor) -> None:
        N = self.subgraph.n_nodes
        costs = torch.full((N,), c.FLOAT_MAX, dtype=torch.float64, device=self.device)
        processed = torch.zeros(N, dtype=torch.bool, device=self.device)

        proto_mask = self.subgraph.status == c.PROTOTYPE
        costs[proto_mask] = 0.0
        self.subgraph.preds[proto_mask] = c.NIL
        self.subgraph.pred_labels[proto_mask] = self.subgraph.labels[proto_mask]

        idx_nodes = []

        for _ in range(N):
            candidates = costs.clone()
            candidates[processed] = c.FLOAT_MAX
            p = candidates.argmin().item()
            processed[p] = True
            idx_nodes.append(p)
            self.subgraph.costs[p] = costs[p]

            mask = ~processed
            if not mask.any():
                break

            mask_indices = mask.nonzero(as_tuple=True)[0]

            better_mask = costs[p] < costs[mask_indices]
            if not better_mask.any():
                continue

            candidate_indices = mask_indices[better_mask]

            arc_weights = dist_matrix[p, candidate_indices].to(dtype=torch.float64)
            path_costs = torch.maximum(costs[p].expand_as(arc_weights), arc_weights)

            improved = path_costs < costs[candidate_indices]
            update_indices = candidate_indices[improved]

            if update_indices.numel() > 0:
                costs[update_indices] = path_costs[improved]
                self.subgraph.preds[update_indices] = p
                self.subgraph.pred_labels[update_indices] = self.subgraph.pred_labels[p].clone()

        self.subgraph.idx_nodes = idx_nodes

    def fit(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        I_train: torch.Tensor | None = None,
    ) -> None:
        """Replace the fitted graph using labeled training samples.

        Class-crossing edges in a minimum spanning tree identify prototypes.
        Minimax competition assigns every remaining node to a prototype using
        the maximum edge weight on its path. Single-class input uses one seed.
        Input storage can be shared with the graph and must not be mutated afterward.

        Args:
            X_train: Training features of shape (N, D).
            Y_train: Training labels of shape (N,).
            I_train: Training indices of shape (N,).

        Raises:
            e.SizeError: Training data are empty or sample dimensions do not align.
            e.TypeError: Original sample indices are not integers.
            e.ValueError: Original sample indices are outside the distance matrix.

        """

        logger.info("Fitting classifier ...")

        start = time.time()

        self.subgraph = Subgraph(X_train, Y_train, I=I_train, device=str(self.device))

        dist_matrix = self._get_distances()

        self._find_prototypes(dist_matrix)
        self._compete(dist_matrix)
        self.subgraph.trained = True

        end = time.time()
        logger.info("Classifier has been fitted.")
        logger.info("Training time: %s seconds.", end - start)

    def predict(
        self,
        X_val: torch.Tensor,
        I_val: torch.Tensor | None = None,
    ) -> list[int]:
        """Predict labels using batched minimax path costs.

        Winning nodes and their predecessor paths are marked relevant for pruning.

        Args:
            X_val: Validation/test features of shape (M, D).
            I_val: Original matrix positions for the validation or test samples.

        Returns:
            One predicted class label per input row, in input order.

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

        train_costs = self.subgraph.costs.unsqueeze(1)
        path_costs = torch.maximum(train_costs, dist_matrix.to(dtype=torch.float64))
        _, best_nodes = path_costs.min(dim=0)

        predictions = self.subgraph.pred_labels[best_nodes]
        self.subgraph._mark_predecessors(best_nodes.unique().tolist())

        end = time.time()
        logger.info("Data has been predicted.")
        logger.info("Prediction time: %s seconds.", end - start)

        return predictions.cpu().tolist()

    def learn(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        n_iterations: int = 10,
        I_train: torch.Tensor | None = None,
        I_val: torch.Tensor | None = None,
    ) -> None:
        """Select a supervised classifier by exchanging misclassified samples.

        Training and validation arrays are cloned before exchanges. The
        highest-scoring fitted model replaces this instance's state.

        Args:
            X_train: Training features.
            Y_train: Training labels.
            X_val: Validation features.
            Y_val: Validation labels.
            n_iterations: Maximum iterations, at least one.
            I_train: Original training indices in a pre-computed distance matrix.
            I_val: Original validation indices in a pre-computed distance matrix.

        Raises:
            e.ValueError: The iteration count or original sample indices are invalid.
            e.SizeError: Feature, label, or index dimensions do not align.

        """

        logger.info("Learning the best classifier ...")

        if not isinstance(n_iterations, int) or n_iterations < 1:
            raise e.ValueError(f"`n_iterations` must be an integer of at least 1, but got {n_iterations}.")

        X_train = torch.as_tensor(X_train, dtype=torch.float32, device=self.device).clone()
        Y_train = torch.as_tensor(Y_train, dtype=torch.int64, device=self.device).clone()
        X_val = torch.as_tensor(X_val, dtype=torch.float32, device=self.device).clone()
        Y_val = torch.as_tensor(Y_val, dtype=torch.int64, device=self.device).clone()
        I_train = (
            torch.arange(len(X_train), device=self.device)
            if I_train is None
            else torch.as_tensor(I_train, device=self.device).clone()
        )
        I_val = (
            torch.arange(len(X_val), device=self.device)
            if I_val is None
            else torch.as_tensor(I_val, device=self.device).clone()
        )

        max_acc = float("-inf")
        previous_acc = 0.0
        best_opf = None
        best_t = 0

        t = 0
        while True:
            logger.info("Running iteration %d/%d ...", t + 1, n_iterations)

            self.fit(X_train, Y_train, I_train)
            preds = self.predict(X_val, I_val)
            preds_tensor = torch.tensor(preds, dtype=torch.int64, device=self.device)

            acc = g.opf_accuracy(Y_val, preds_tensor)
            if acc > max_acc:
                max_acc = acc
                best_opf = copy.deepcopy(self)
                best_t = t

            errors = (Y_val != preds_tensor).nonzero(as_tuple=True)[0]

            non_proto_mask = self.subgraph.status != c.PROTOTYPE
            non_proto_indices = non_proto_mask.nonzero(as_tuple=True)[0]
            n_non_proto = non_proto_indices.numel()

            for err_idx in errors:
                ctr = n_non_proto
                while ctr > 0:
                    j = int(r.generate_uniform_random_number(0, len(X_train), 1).item())

                    if self.subgraph.status[j] != c.PROTOTYPE:
                        X_train[j], X_val[err_idx] = (
                            X_val[err_idx].clone(),
                            X_train[j].clone(),
                        )
                        Y_train[j], Y_val[err_idx] = (
                            Y_val[err_idx].clone(),
                            Y_train[j].clone(),
                        )
                        I_train[j], I_val[err_idx] = (
                            I_val[err_idx].clone(),
                            I_train[j].clone(),
                        )
                        n_non_proto -= 1
                        ctr = 0
                    else:
                        ctr -= 1

            delta = abs(acc - previous_acc)
            previous_acc = acc
            t += 1

            logger.info("Accuracy: %s | Delta: %s | Maximum Accuracy: %s", acc, delta, max_acc)

            if delta < 0.0001 or t == n_iterations:
                if best_opf is not None:
                    self.__dict__.update(best_opf.__dict__)

                logger.info("Best classifier has been learned over iteration %d.", best_t + 1)
                break

    def prune(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        n_iterations: int = 10,
        I_train: torch.Tensor | None = None,
        I_val: torch.Tensor | None = None,
    ) -> None:
        """Remove irrelevant training samples while retaining class prototypes.

        Each reduction refits the graph without modifying the supplied arrays.
        Pruning is validation-driven and does not guarantee unchanged predictions
        for other data.

        Args:
            X_train: Training features.
            Y_train: Training labels.
            X_val: Validation features.
            Y_val: Validation labels.
            n_iterations: Maximum iterations, zero to fit without removing nodes.
            I_train: Original training indices in a pre-computed distance matrix.
            I_val: Original validation indices in a pre-computed distance matrix.

        Raises:
            e.ValueError: The iteration count or original sample indices are invalid.
            e.SizeError: Feature, label, or index dimensions do not align.

        """

        logger.info("Pruning classifier ...")

        if not isinstance(n_iterations, int) or n_iterations < 0:
            raise e.ValueError(f"`n_iterations` must be a non-negative integer, but got {n_iterations}.")

        self.fit(X_train, Y_train, I_train)
        self.predict(X_val, I_val)

        initial_nodes = self.subgraph.n_nodes

        for t in range(n_iterations):
            logger.info("Running iteration %d/%d ...", t + 1, n_iterations)

            relevant_mask = (self.subgraph.relevant != c.IRRELEVANT) | (self.subgraph.status == c.PROTOTYPE)
            if relevant_mask.all():
                break

            X_train = self.subgraph.features[relevant_mask]
            Y_train = self.subgraph.labels[relevant_mask]
            I_train = self.subgraph.indices[relevant_mask]

            self.fit(X_train, Y_train, I_train)
            preds = self.predict(X_val, I_val)

            acc = g.opf_accuracy(Y_val, preds)
            logger.info("Current accuracy: %s.", acc)

        final_nodes = self.subgraph.n_nodes
        prune_ratio = 1 - final_nodes / initial_nodes

        logger.info("Prune ratio: %s.", prune_ratio)
