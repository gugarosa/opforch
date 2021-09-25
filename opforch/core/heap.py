"""PyTorch-based heap implementation.
"""

import torch

import opforch.utils.constants as c


class Heap:
    """An alternate PyTorch-based heap structure.

    """

    def __init__(self, size=1, policy='min'):
        """Initialization method.

        Args:
            size (int): Maximum size of the heap.
            policy (str): Heap's policy (`min` or `max`).

        """

        # Maximum size of the heap
        self.size = size

        # Policy to rule the heap
        self.policy = policy

        # Costs for nodes in the heap
        self.cost = torch.full((size,), c.FLOAT_MAX, dtype=torch.float64)
