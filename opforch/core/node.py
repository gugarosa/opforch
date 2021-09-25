"""Node structure that composes the subgraph of Optimum-Path Forest.
"""

import torch

import opforch.utils.logging as l

logger = l.get_logger(__name__)


class Node:
    """Used to compose the most low-level structure level in OPF.

    """

    def __init__(self, idx=0, label=0, features=None):
        """Initialization method.

        Args:
            idx (int): Identifier.
            label (int): Label.
            features (torch.Tensor): Tensor holding the features.

        """

        # Identifier
        self.idx = idx

        # True label
        self.label = label

        # Tensor of features
        self.features = torch.Tensor(features)

