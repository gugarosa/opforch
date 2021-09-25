"""Node structure that composes the subgraph of the Optimum-Path Forest.
"""

from typing import Optional, Union

import torch

import opforch.utils.logging as l

logger = l.get_logger(__name__)


class Node:
    """Used to compose the most low-level structure level in OPF.

    """

    def __init__(self,
                 idx: Optional[int] = 0,
                 label: Optional[int] = 0,
                 features: Optional[Union[list, torch.Tensor]] = None) -> None:
        """Initialization method.

        Args:
            idx: Numeric identifier.
            label: True label.
            features: Features (holds any type of information).

        """

        # Identifier
        self.idx = idx

        # True label
        self.label = label

        # Tensor of features
        self.features = torch.Tensor(features)
