"""Supervised Optimum-Path Forest (PyTorch)."""

import copy
import time
from typing import List, Optional

import torch

import opforch.math.general as g
import opforch.math.random as r
import opforch.utils.constants as c
import opforch.utils.exception as e
from opforch.core import OPF, Subgraph
from opforch.utils import logging

logger = logging.get_logger(__name__)


class SupervisedOPF(OPF):
    """Supervised OPF classifier with batched distance computation.

    References:
        J. P. Papa, A. X. Falcão and C. T. N. Suzuki.
        Supervised Pattern Classification based on Optimum-Path Forest.
        International Journal of Imaging Systems and Technology (2009).

    """

    def __init__(
        self,
        distance: str = "log_squared_euclidean",
        pre_computed_distance: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialization method.

        Args:
            distance: Distance metric name.
            pre_computed_distance: Path to a pre-computed distance file.
            device: Target device string.

        """

        logger.info("Overriding class: OPF -> SupervisedOPF.")

        super(SupervisedOPF, self).__init__(distance, pre_computed_distance, device)

        logger.info("Class overrided.")

    def _find_prototypes(self, dist_matrix: torch.Tensor) -> None:
        """Finds prototype nodes via Prim's MST with vectorized inner updates.

        Instead of computing pairwise distances inside the loop, we index
        into the pre-computed dist_matrix. The inner O(N) scan per extraction
        is replaced with masked tensor operations.

        Args:
            dist_matrix: Pre-computed (N, N) distance matrix.

        """

        logger.debug("Finding prototypes ...")

        N = self.subgraph.n_nodes
        in_tree = torch.zeros(N, dtype=torch.bool, device=self.device)
        costs = torch.full((N,), c.FLOAT_MAX, dtype=torch.float64, device=self.device)
        preds = torch.full((N,), c.NIL, dtype=torch.int64, device=self.device)

        costs[0] = 0.0

        prototypes = []
        for _ in range(N):
            # Find cheapest node not in tree
            candidates = costs.clone()
            candidates[in_tree] = c.FLOAT_MAX
            p = candidates.argmin().item()
            in_tree[p] = True

            # Check for class boundary → mark prototypes
            pred_idx = preds[p].item()
            if pred_idx != c.NIL:
                if self.subgraph.labels[p] != self.subgraph.labels[pred_idx]:
                    if self.subgraph.status[p] != c.PROTOTYPE:
                        self.subgraph.status[p] = c.PROTOTYPE
                        prototypes.append(p)
                    if self.subgraph.status[pred_idx] != c.PROTOTYPE:
                        self.subgraph.status[pred_idx] = c.PROTOTYPE
                        prototypes.append(pred_idx)

            # Update all non-tree nodes in one vectorized operation
            mask = ~in_tree
            new_costs = dist_matrix[p][mask]
            improved = new_costs < costs[mask]

            # Use tensor indexing for batch update
            mask_indices = mask.nonzero(as_tuple=True)[0]
            improved_indices = mask_indices[improved]

            costs[improved_indices] = new_costs[improved].to(dtype=torch.float64)
            preds[improved_indices] = p

        self.subgraph.preds = preds

        logger.debug("Prototypes: %s.", prototypes)

    def _compete(self, dist_matrix: torch.Tensor) -> None:
        """Runs optimum-path competition with vectorized inner updates.

        Dijkstra-like algorithm with minimax path cost. The sequential heap
        extraction remains, but inner work per iteration is fully vectorized.

        Args:
            dist_matrix: Pre-computed (N, N) distance matrix.

        """

        N = self.subgraph.n_nodes
        costs = torch.full((N,), c.FLOAT_MAX, dtype=torch.float64, device=self.device)
        processed = torch.zeros(N, dtype=torch.bool, device=self.device)

        # Prototypes start with cost 0
        proto_mask = self.subgraph.status == c.PROTOTYPE
        costs[proto_mask] = 0.0
        self.subgraph.preds[proto_mask] = c.NIL
        self.subgraph.pred_labels[proto_mask] = self.subgraph.labels[proto_mask]

        idx_nodes = []

        for _ in range(N):
            # Find unprocessed node with minimum cost
            candidates = costs.clone()
            candidates[processed] = c.FLOAT_MAX
            p = candidates.argmin().item()
            processed[p] = True
            idx_nodes.append(p)
            self.subgraph.costs[p] = costs[p]

            # Update all unprocessed nodes where p could improve their cost
            mask = ~processed
            if not mask.any():
                break

            mask_indices = mask.nonzero(as_tuple=True)[0]

            # Only consider nodes whose current cost is worse than p's cost
            better_mask = costs[p] < costs[mask_indices]
            if not better_mask.any():
                continue

            candidate_indices = mask_indices[better_mask]

            # Compute minimax path costs
            arc_weights = dist_matrix[p, candidate_indices].to(dtype=torch.float64)
            path_costs = torch.maximum(costs[p].expand_as(arc_weights), arc_weights)

            # Find which candidates actually improve
            improved = path_costs < costs[candidate_indices]
            update_indices = candidate_indices[improved]

            if update_indices.numel() > 0:
                costs[update_indices] = path_costs[improved]
                self.subgraph.preds[update_indices] = p
                self.subgraph.pred_labels[update_indices] = self.subgraph.pred_labels[
                    p
                ].clone()

        self.subgraph.idx_nodes = idx_nodes

    def fit(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        I_train: Optional[torch.Tensor] = None,
    ) -> None:
        """Fits data in the classifier.

        Args:
            X_train: Training features of shape (N, D).
            Y_train: Training labels of shape (N,).
            I_train: Training indices of shape (N,).

        """

        logger.info("Fitting classifier ...")

        start = time.time()

        self.subgraph = Subgraph(X_train, Y_train, I=I_train, device=str(self.device))

        # Compute full distance matrix ONCE — the main GPU-accelerated operation
        if self.pre_computed_distance:
            dist_matrix = self.pre_distances
        else:
            dist_matrix = self.distance_fn(
                self.subgraph.features, self.subgraph.features
            )

        # Step 1: Find prototypes via MST
        self._find_prototypes(dist_matrix)

        # Step 2: Optimum-path competition
        self._compete(dist_matrix)

        self.subgraph.trained = True

        end = time.time()
        logger.info("Classifier has been fitted.")
        logger.info("Training time: %s seconds.", end - start)

    def predict(
        self,
        X_val: torch.Tensor,
        I_val: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """Predicts new data using fully batched tensor operations.

        Instead of looping over test samples, computes all train-test
        distances and minimax costs in a single tensor operation.

        Args:
            X_val: Validation/test features of shape (M, D).
            I_val: Validation/test indices.

        Returns:
            A list of predicted labels.

        """

        if self.subgraph is None:
            raise e.BuildError("Subgraph has not been properly created")
        if not self.subgraph.trained:
            raise e.BuildError("Classifier has not been properly fitted")

        logger.info("Predicting data ...")

        start = time.time()

        if not isinstance(X_val, torch.Tensor):
            X_val = torch.tensor(X_val, dtype=torch.float32)
        X_val = X_val.to(dtype=torch.float32, device=self.device)

        # Compute all train-test distances in one batch: (N_train, M_test)
        if self.pre_computed_distance:
            dist_matrix = self.pre_distances
        else:
            dist_matrix = self.distance_fn(self.subgraph.features, X_val)

        # Minimax path costs: max(train_node_cost, arc_weight)
        train_costs = self.subgraph.costs.unsqueeze(1)  # (N, 1)
        path_costs = torch.maximum(
            train_costs, dist_matrix.to(dtype=torch.float64)
        )  # (N, M)

        # Best training node for each test sample (minimum minimax cost)
        _, best_nodes = path_costs.min(dim=0)  # (M,)

        predictions = self.subgraph.pred_labels[best_nodes]

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
    ) -> None:
        """Learns the best classifier over a validation set.

        Iteratively swaps misclassified validation samples with non-prototype
        training samples to improve accuracy.

        Args:
            X_train: Training features.
            Y_train: Training labels.
            X_val: Validation features.
            Y_val: Validation labels.
            n_iterations: Maximum iterations.

        """

        logger.info("Learning the best classifier ...")

        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        if not isinstance(Y_train, torch.Tensor):
            Y_train = torch.tensor(Y_train, dtype=torch.int64)
        if not isinstance(X_val, torch.Tensor):
            X_val = torch.tensor(X_val, dtype=torch.float32)
        if not isinstance(Y_val, torch.Tensor):
            Y_val = torch.tensor(Y_val, dtype=torch.int64)

        X_train = X_train.clone()
        Y_train = Y_train.clone()
        X_val = X_val.clone()
        Y_val = Y_val.clone()

        max_acc = 0.0
        previous_acc = 0.0
        best_opf = None

        t = 0
        while True:
            logger.info("Running iteration %d/%d ...", t + 1, n_iterations)

            self.fit(X_train, Y_train)
            preds = self.predict(X_val)
            preds_tensor = torch.tensor(preds, dtype=torch.int64)

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
                        n_non_proto -= 1
                        ctr = 0
                    else:
                        ctr -= 1

            delta = abs(acc - previous_acc)
            previous_acc = acc
            t += 1

            logger.info(
                "Accuracy: %s | Delta: %s | Maximum Accuracy: %s", acc, delta, max_acc
            )

            if delta < 0.0001 or t == n_iterations:
                if best_opf is not None:
                    self.__dict__.update(best_opf.__dict__)

                logger.info(
                    "Best classifier has been learned over iteration %d.", best_t + 1
                )
                break

    def prune(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        n_iterations: int = 10,
    ) -> None:
        """Prunes the classifier by removing irrelevant nodes.

        Args:
            X_train: Training features.
            Y_train: Training labels.
            X_val: Validation features.
            Y_val: Validation labels.
            n_iterations: Maximum iterations.

        """

        logger.info("Pruning classifier ...")

        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        if not isinstance(Y_train, torch.Tensor):
            Y_train = torch.tensor(Y_train, dtype=torch.int64)

        self.fit(X_train, Y_train)
        self.predict(X_val)

        initial_nodes = self.subgraph.n_nodes

        for t in range(n_iterations):
            logger.info("Running iteration %d/%d ...", t + 1, n_iterations)

            # Keep only relevant nodes
            relevant_mask = self.subgraph.relevant != c.IRRELEVANT
            X_train = X_train[relevant_mask]
            Y_train = Y_train[relevant_mask]

            self.fit(X_train, Y_train)
            preds = self.predict(X_val)
            preds_tensor = torch.tensor(preds, dtype=torch.int64)

            acc = g.opf_accuracy(Y_val, preds_tensor)
            logger.info("Current accuracy: %s.", acc)

        final_nodes = self.subgraph.n_nodes
        prune_ratio = 1 - final_nodes / initial_nodes

        logger.info("Prune ratio: %s.", prune_ratio)
