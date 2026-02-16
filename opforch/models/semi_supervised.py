"""Semi-Supervised Optimum-Path Forest (PyTorch)."""

import time
from typing import Optional

import torch

import opforch.utils.constants as c
from opforch.core import Subgraph
from opforch.models.supervised import SupervisedOPF
from opforch.utils import logging

logger = logging.get_logger(__name__)


class SemiSupervisedOPF(SupervisedOPF):
    """Semi-supervised OPF that incorporates unlabeled data during training.

    References:
        W. P. Amorim, A. X. Falcão and M. H. Carvalho.
        Semi-supervised Pattern Classification Using Optimum-Path Forest.
        27th SIBGRAPI Conference on Graphics, Patterns and Images (2014).

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
            pre_computed_distance: Path to pre-computed distance file.
            device: Target device string.

        """

        logger.info("Overriding class: SupervisedOPF -> SemiSupervisedOPF.")

        super(SemiSupervisedOPF, self).__init__(
            distance, pre_computed_distance, device
        )

        logger.info("Class overrided.")

    def fit(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_unlabeled: torch.Tensor,
        I_train: Optional[torch.Tensor] = None,
    ) -> None:
        """Fits the semi-supervised classifier.

        Builds a subgraph from labeled data, finds prototypes via MST,
        appends unlabeled samples, and runs optimum-path competition.
        Unlabeled nodes inherit labels from their conquerors.

        Args:
            X_train: Labeled training features of shape (N_l, D).
            Y_train: Training labels of shape (N_l,).
            X_unlabeled: Unlabeled features of shape (N_u, D).
            I_train: Training indices.

        """

        logger.info("Fitting semi-supervised classifier ...")

        start = time.time()

        # Build subgraph from labeled data
        self.subgraph = Subgraph(X_train, Y_train, I=I_train, device=str(self.device))

        # Find prototypes from labeled data only
        if self.pre_computed_distance:
            labeled_dist = self.pre_distances
        else:
            labeled_dist = self.distance_fn(
                self.subgraph.features, self.subgraph.features
            )
        self._find_prototypes(labeled_dist)

        # Append unlabeled samples to subgraph tensors
        if not isinstance(X_unlabeled, torch.Tensor):
            X_unlabeled = torch.tensor(X_unlabeled, dtype=torch.float32)
        X_unlabeled = X_unlabeled.to(dtype=torch.float32, device=self.device)

        n_labeled = self.subgraph.n_nodes
        n_unlabeled = X_unlabeled.shape[0]
        n_total = n_labeled + n_unlabeled

        # Expand all tensors
        self.subgraph.features = torch.cat(
            [self.subgraph.features, X_unlabeled], dim=0
        )
        self.subgraph.labels = torch.cat(
            [self.subgraph.labels, torch.zeros(n_unlabeled, dtype=torch.int64, device=self.device)]
        )
        self.subgraph.indices = torch.cat(
            [self.subgraph.indices,
             torch.arange(n_labeled, n_total, dtype=torch.int64, device=self.device)]
        )
        self.subgraph.pred_labels = torch.cat(
            [self.subgraph.pred_labels, torch.zeros(n_unlabeled, dtype=torch.int64, device=self.device)]
        )
        self.subgraph.cluster_labels = torch.cat(
            [self.subgraph.cluster_labels, torch.zeros(n_unlabeled, dtype=torch.int64, device=self.device)]
        )
        self.subgraph.costs = torch.cat(
            [self.subgraph.costs, torch.zeros(n_unlabeled, dtype=torch.float64, device=self.device)]
        )
        self.subgraph.densities = torch.cat(
            [self.subgraph.densities, torch.zeros(n_unlabeled, dtype=torch.float64, device=self.device)]
        )
        self.subgraph.radii = torch.cat(
            [self.subgraph.radii, torch.zeros(n_unlabeled, dtype=torch.float64, device=self.device)]
        )
        self.subgraph.n_plateaus = torch.cat(
            [self.subgraph.n_plateaus, torch.zeros(n_unlabeled, dtype=torch.int64, device=self.device)]
        )
        self.subgraph.preds = torch.cat(
            [self.subgraph.preds, torch.full((n_unlabeled,), c.NIL, dtype=torch.int64, device=self.device)]
        )
        self.subgraph.roots = torch.cat(
            [self.subgraph.roots, torch.arange(n_labeled, n_total, dtype=torch.int64, device=self.device)]
        )
        self.subgraph.status = torch.cat(
            [self.subgraph.status, torch.full((n_unlabeled,), c.STANDARD, dtype=torch.int8, device=self.device)]
        )
        self.subgraph.relevant = torch.cat(
            [self.subgraph.relevant, torch.full((n_unlabeled,), c.IRRELEVANT, dtype=torch.int8, device=self.device)]
        )

        # Compute distance matrix for the combined data
        if self.pre_computed_distance:
            dist_matrix = self.pre_distances
        else:
            dist_matrix = self.distance_fn(
                self.subgraph.features, self.subgraph.features
            )

        # Run optimum-path competition on combined labeled + unlabeled data
        self._compete(dist_matrix)

        # Propagate predicted labels to unlabeled node's true labels
        unlabeled_mask = torch.zeros(n_total, dtype=torch.bool, device=self.device)
        unlabeled_mask[n_labeled:] = True
        self.subgraph.labels[unlabeled_mask] = self.subgraph.pred_labels[unlabeled_mask]

        self.subgraph.trained = True

        end = time.time()
        logger.info("Semi-supervised classifier has been fitted.")
        logger.info("Training time: %s seconds.", end - start)
