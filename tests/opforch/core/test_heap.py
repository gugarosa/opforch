# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import pytest

from opforch.core import heap


def test_heap_size_defaults_to_one():
    h = heap.Heap()

    assert h.size == 1


def test_heap_size_accepts_positive_capacity():
    h = heap.Heap(size=3)

    assert h.size == 3


def test_heap_size_rejects_zero_capacity():
    with pytest.raises(ValueError):
        heap.Heap(size=0)


def test_heap_policy_defaults_to_minimum():
    h = heap.Heap()

    assert h.policy == "min"


@pytest.mark.parametrize("policy", ["min", "max"])
def test_heap_policy_accepts_priority_order(policy):
    h = heap.Heap(policy=policy)

    assert h.policy == policy


def test_heap_policy_rejects_unknown_order():
    with pytest.raises(ValueError):
        heap.Heap(policy="a")


def test_heap_cost_matches_capacity():
    h = heap.Heap()

    assert len(h.cost) == 1


def test_heap_color_matches_capacity():
    h = heap.Heap()

    assert len(h.color) == 1


def test_heap_p_matches_capacity():
    h = heap.Heap()

    assert len(h.p) == 1


def test_heap_pos_matches_capacity():
    h = heap.Heap()

    assert len(h.pos) == 1


def test_heap_last_marks_empty_heap():
    h = heap.Heap()

    assert h.last == -1


def test_heap_is_full_after_reaching_capacity():
    h = heap.Heap()

    h.insert(0)

    status = h.is_full()

    assert status is True


def test_heap_is_empty_before_insertion():
    h = heap.Heap()

    status = h.is_empty()

    assert status is True


def test_heap_dad_returns_parent_position():
    h = heap.Heap(size=10)

    dad = h.dad(5)

    assert dad == 2


def test_heap_left_son_returns_left_child_position():
    h = heap.Heap(size=10)

    left_son = h.left_son(5)

    assert left_son == 11


def test_heap_right_son_returns_right_child_position():
    h = heap.Heap(size=10)

    right_son = h.right_son(5)

    assert right_son == 12


def test_heap_insert_rejects_full_heap():
    h = heap.Heap()

    h.insert(0)

    status = h.insert(1)

    assert status is False


def test_heap_remove_returns_sentinel_for_empty_heap():
    h = heap.Heap()

    status = h.remove()

    assert status == -1


def test_heap_remove_returns_lowest_cost_first():
    h = heap.Heap(size=3, policy="min")

    h.cost[0] = 5.0
    h.cost[1] = 3.0
    h.cost[2] = 7.0

    h.insert(0)
    h.insert(1)
    h.insert(2)

    p = h.remove()
    assert p == 1

    p = h.remove()
    assert p == 0


def test_heap_update_promotes_lower_cost():
    h = heap.Heap(size=3, policy="min")

    h.cost[0] = 5.0
    h.cost[1] = 3.0

    h.insert(0)
    h.insert(1)

    h.update(2, 1.0)

    p = h.remove()
    assert p == 2
