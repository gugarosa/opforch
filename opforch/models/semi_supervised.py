# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Train an optimum-path forest with labeled and unlabeled samples.

References:
    W. P. Amorim, A. X. Falcão and M. H. Carvalho, Semi-supervised Pattern Classification Using OPF (2014).

"""

from __future__ import annotations

import time

import torch

from opforch.core.subgraph import Subgraph
from opforch.models.supervised import SupervisedOPF
from opforch.utils.logging import get_logger

logger = get_logger(__name__)


class SemiSupervisedOPF(SupervisedOPF):
    """Incorporate unlabeled samples into supervised optimum-path competition.

    """  # fmt: skip

    def __init__(
        self,
        distance: str = "log_squared_euclidean",
        pre_computed_distance: str | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize a semi-supervised classifier.

        Args:
            distance: Distance metric name.
            pre_computed_distance: Path to pre-computed distance file.
            device: Target device, or automatic CUDA/CPU selection when None.

        """

        logger.info("Creating semi-supervised OPF classifier.")

        super().__init__(distance, pre_computed_distance, device)

    def fit(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_unlabeled: torch.Tensor,
        I_train: torch.Tensor | None = None,
        I_unlabeled: torch.Tensor | None = None,
    ) -> None:
        """Replace the fitted graph using labeled and unlabeled samples.

        Builds a subgraph from labeled data, finds prototypes via MST,
        appends unlabeled samples, and runs optimum-path competition.
        Unlabeled nodes inherit labels from their conquerors.

        Args:
            X_train: Labeled training features of shape (N_l, D).
            Y_train: Training labels of shape (N_l,).
            X_unlabeled: Unlabeled features of shape (N_u, D).
            I_train: Original labeled indices in a pre-computed distance matrix.
            I_unlabeled: Unlabeled matrix indices, or positions after the labeled samples when None.

        """

        logger.info("Fitting semi-supervised classifier ...")

        start = time.time()

        self.subgraph = Subgraph(X_train, Y_train, I=I_train, device=str(self.device))
        labeled_dist = self._get_distances()
        self._find_prototypes(labeled_dist)
        labeled = self.subgraph

        if not isinstance(X_unlabeled, torch.Tensor):
            X_unlabeled = torch.tensor(X_unlabeled, dtype=torch.float32)
        X_unlabeled = X_unlabeled.to(dtype=torch.float32, device=self.device)

        n_labeled = labeled.n_nodes
        n_unlabeled = X_unlabeled.shape[0]
        if I_unlabeled is None:
            I_unlabeled = torch.arange(n_labeled, n_labeled + n_unlabeled, device=self.device)
        I_unlabeled = labeled._to_indices(I_unlabeled, n_unlabeled)

        features = torch.cat((labeled.features, X_unlabeled))
        labels = torch.cat((labeled.labels, torch.zeros(n_unlabeled, dtype=torch.int64, device=self.device)))
        indices = torch.cat((labeled.indices, I_unlabeled))
        self.subgraph = Subgraph(features, labels, I=indices, device=self.device)

        # Only prototype selection and the labeled MST survive graph initialization
        self.subgraph.status[:n_labeled] = labeled.status
        self.subgraph.preds[:n_labeled] = labeled.preds

        dist_matrix = self._get_distances()
        self._compete(dist_matrix)

        self.subgraph.labels[n_labeled:] = self.subgraph.pred_labels[n_labeled:]
        self.subgraph.trained = True

        end = time.time()
        logger.info("Semi-supervised classifier has been fitted.")
        logger.info("Training time: %s seconds.", end - start)
