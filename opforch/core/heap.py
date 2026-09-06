# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide the tensor-backed binary heap used during OPF competition.

A binary heap (priority queue) supporting min and max policies.
Internal state is stored as tensors for consistency with the
rest of OPForch, though heap operations remain sequential.

"""

from __future__ import annotations

import torch

import opforch.utils.constants as c
from opforch.utils.device import DeviceManager


class Heap:
    """Maintain a binary heap with tensor-backed state.

    """  # fmt: skip

    def __init__(
        self,
        size: int = 1,
        policy: str = "min",
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize an empty heap with a fixed capacity.

        Args:
            size: Maximum size of the heap.
            policy: Priority policy, either min or max.
            device: Target device, or automatic CUDA/CPU selection when None.

        Raises:
            ValueError: The capacity or priority policy is invalid.

        """

        if size < 1:
            raise ValueError(f"`size` must be greater than 0, but got {size}.")
        if policy not in ("min", "max"):
            raise ValueError(f"`policy` must be min or max, but got {policy!r}.")

        self.size = size
        self.policy = policy
        self.device = DeviceManager.resolve(device)

        self.cost = torch.full((size,), c.FLOAT_MAX, dtype=torch.float64, device=self.device)
        self.color = torch.full((size,), c.WHITE, dtype=torch.int8, device=self.device)
        self.p = torch.full((size,), -1, dtype=torch.int64, device=self.device)
        self.pos = torch.full((size,), -1, dtype=torch.int64, device=self.device)
        self.last = -1

    def is_full(self) -> bool:
        """Return whether the heap has reached its capacity.

        Returns:
            Whether another node would exceed the capacity.

        """

        return self.last == (self.size - 1)

    def is_empty(self) -> bool:
        """Return whether the heap has no queued nodes.

        Returns:
            Whether the heap is empty.

        """

        return self.last == -1

    def dad(self, i: int) -> int:
        """Return a heap position's parent position.

        Args:
            i: Position in the heap array.

        Returns:
            Parent position, or -1 for the root.

        """

        return (i - 1) // 2

    def left_son(self, i: int) -> int:
        """Return a heap position's left-child position.

        Args:
            i: Position in the heap array.

        Returns:
            Left-child position without checking the current heap size.

        """

        return 2 * i + 1

    def right_son(self, i: int) -> int:
        """Return a heap position's right-child position.

        Args:
            i: Position in the heap array.

        Returns:
            Right-child position without checking the current heap size.

        """

        return 2 * i + 2

    def _higher_priority(self, left: int | torch.Tensor, right: int | torch.Tensor) -> bool:
        if self.policy == "min":
            return bool(self.cost[left] < self.cost[right])

        return bool(self.cost[left] > self.cost[right])

    def go_up(self, i: int) -> None:
        """Sift a node upward to restore the priority ordering.

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
        """Sift a node downward to restore the priority ordering.

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
        """Insert an unqueued node into the heap.

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
        """Remove the highest-priority node from the heap.

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
        """Update a node's cost and promote or insert it when eligible.

        A queued node's new cost is expected to improve its priority. Removed
        nodes receive the new cost but are not reinserted.

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
