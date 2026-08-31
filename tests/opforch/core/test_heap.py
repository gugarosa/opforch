from opforch.core import heap


def test_heap_size():
    h = heap.Heap()

    assert h.size == 1


def test_heap_size_setter():
    try:
        h = heap.Heap(size=0)
    except:
        h = heap.Heap(size=1)

    assert h.size == 1


def test_heap_policy():
    h = heap.Heap()

    assert h.policy == "min"


def test_heap_policy_setter():
    try:
        h = heap.Heap(policy="a")
    except:
        h = heap.Heap(policy="min")

    assert h.policy == "min"

    h = heap.Heap(policy="max")

    assert h.policy == "max"


def test_heap_cost():
    h = heap.Heap()

    assert len(h.cost) == 1


def test_heap_color():
    h = heap.Heap()

    assert len(h.color) == 1


def test_heap_p():
    h = heap.Heap()

    assert len(h.p) == 1


def test_heap_pos():
    h = heap.Heap()

    assert len(h.pos) == 1


def test_heap_last():
    h = heap.Heap()

    assert h.last == -1


def test_heap_is_full():
    h = heap.Heap()

    h.insert(0)

    status = h.is_full()

    assert status is True


def test_heap_is_empty():
    h = heap.Heap()

    status = h.is_empty()

    assert status is True


def test_heap_dad():
    h = heap.Heap(size=10)

    dad = h.dad(5)

    assert dad == 2


def test_heap_left_son():
    h = heap.Heap(size=10)

    left_son = h.left_son(5)

    assert left_son == 11


def test_heap_right_son():
    h = heap.Heap(size=10)

    right_son = h.right_son(5)

    assert right_son == 12


def test_heap_insert():
    h = heap.Heap()

    h.insert(0)

    status = h.insert(1)

    assert status is False


def test_heap_remove():
    h = heap.Heap()

    status = h.remove()

    assert status == -1


def test_heap_insert_and_remove():
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


def test_heap_update():
    h = heap.Heap(size=3, policy="min")

    h.cost[0] = 5.0
    h.cost[1] = 3.0

    h.insert(0)
    h.insert(1)

    h.update(2, 1.0)

    p = h.remove()
    assert p == 2
