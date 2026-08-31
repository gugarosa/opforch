"""Tensor-backed Heap for the Optimum-Path Forest.

A binary heap (priority queue) supporting min and max policies.
Internal state is stored as tensors for consistency with the
rest of OPForch, though heap operations remain sequential.
"""

from typing import Optional

import torch

import opforch.utils.constants as c
from opforch.utils.device import DeviceManager


class Heap:
    """A binary heap with tensor-backed storage."""

    def __init__(
        self,
        size: int = 1,
        policy: str = "min",
        device: Optional[str] = None,
    ) -> None:
        """Initialization method.

        Args:
            size: Maximum size of the heap.
            policy: Heap's policy ('min' or 'max').
            device: Target device string.

        """

        if size < 1:
            raise ValueError("`size` should be > 0")
        if policy not in ("min", "max"):
            raise ValueError("`policy` should be 'min' or 'max'")

        self.size = size
        self.policy = policy
        self.device = DeviceManager.resolve(device)

        self.cost = torch.full(
            (size,), c.FLOAT_MAX, dtype=torch.float64, device=self.device
        )
        self.color = torch.full((size,), c.WHITE, dtype=torch.int8, device=self.device)
        self.p = torch.full((size,), -1, dtype=torch.int64, device=self.device)
        self.pos = torch.full((size,), -1, dtype=torch.int64, device=self.device)
        self.last = -1

    def is_full(self) -> bool:
        """Checks if the heap is full."""

        return self.last == (self.size - 1)

    def is_empty(self) -> bool:
        """Checks if the heap is empty."""

        return self.last == -1

    def dad(self, i: int) -> int:
        """Returns the position of the node's parent."""

        return (i - 1) // 2

    def left_son(self, i: int) -> int:
        """Returns the position of the node's left child."""

        return 2 * i + 1

    def right_son(self, i: int) -> int:
        """Returns the position of the node's right child."""

        return 2 * i + 2

    def _higher_priority(self, left: int, right: int) -> bool:
        if self.policy == "min":
            return self.cost[left] < self.cost[right]
        return self.cost[left] > self.cost[right]

    def go_up(self, i: int) -> None:
        """Sifts a node up to maintain heap property.

        Args:
            i: Position to sift up from.

        """

        j = self.dad(i)

        while i > 0 and self._higher_priority(self.p[i], self.p[j]):
            self.p[j], self.p[i] = self.p[i].clone(), self.p[j].clone()
            self.pos[self.p[i]] = i
            self.pos[self.p[j]] = j
            i = j
            j = self.dad(i)

    def go_down(self, i: int) -> None:
        """Sifts a node down to maintain heap property.

        Args:
            i: Position to sift down from.

        """

        left = self.left_son(i)
        right = self.right_son(i)
        j = i

        if left <= self.last and self._higher_priority(self.p[left], self.p[j]):
            j = left
        if right <= self.last and self._higher_priority(self.p[right], self.p[j]):
            j = right

        if j != i:
            self.p[j], self.p[i] = self.p[i].clone(), self.p[j].clone()
            self.pos[self.p[i]] = i
            self.pos[self.p[j]] = j
            self.go_down(j)

    def insert(self, p: int) -> bool:
        """Inserts a new node into the heap.

        Args:
            p: Node index to insert.

        Returns:
            True if insertion succeeded, False if heap is full.

        """

        if not self.is_full():
            self.last += 1
            self.p[self.last] = p
            self.color[p] = c.GRAY
            self.pos[p] = self.last
            self.go_up(self.last)
            return True

        return False

    def remove(self) -> int:
        """Removes and returns the root node from the heap.

        Returns:
            The removed node index, or -1 if heap is empty.

        """

        if not self.is_empty():
            p = self.p[0].item()

            self.pos[p] = -1
            self.color[p] = c.BLACK

            self.p[0] = self.p[self.last]
            self.pos[self.p[0]] = 0
            self.p[self.last] = -1

            self.last -= 1

            self.go_down(0)

            return p

        return -1

    def update(self, p: int, cost: float) -> None:
        """Updates a node's cost and adjusts its position.

        Args:
            p: Node index.
            cost: New cost value.

        """

        self.cost[p] = cost

        if self.color[p] == c.BLACK:
            return

        if self.color[p] == c.WHITE:
            self.insert(p)
        else:
            self.go_up(self.pos[p].item())
