"""Heap structure that assists nodes' conquering in the Optimum-Path Forest.
"""

from typing import Optional

import torch

import opforch.utils.constants as c


class Heap:
    """PyTorch-based heap structure.

    """

    def __init__(self,
                 size: Optional[int] = 1,
                 policy: Optional[str] = 'min') -> None:
        """Initialization method.

        Args:
            size: Maximum size of the heap.
            policy: Heap's policy (`min` or `max`).

        """

        # Maximum size of the heap
        self.size = size

        # Policy to rule the heap
        self.policy = policy

        # Costs for nodes in the heap
        self.cost = torch.full((size,), c.FLOAT_MAX, dtype=torch.float64)
