# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

import opforch.utils.exception as e
from opforch.stream import splitter


def test_split_preserves_sample_label_alignment():
    X = torch.arange(12).reshape(6, 2)
    Y = torch.arange(6)

    X_1, X_2, Y_1, Y_2 = splitter.split(X, Y, percentage=0.5, random_state=1)

    assert X_1.shape == (3, 2)
    assert X_2.shape == (3, 2)
    assert Y_1.shape == (3,)
    assert Y_2.shape == (3,)
    torch.testing.assert_close(X_1, X[Y_1])
    torch.testing.assert_close(X_2, X[Y_2])
    assert sorted(torch.cat((Y_1, Y_2)).tolist()) == list(range(6))


def test_split_with_index_preserves_original_positions():
    X = torch.arange(12).reshape(6, 2)
    Y = torch.tensor([0, 1, 0, 1, 0, 1])

    X_1, X_2, Y_1, Y_2, I_1, I_2 = splitter.split_with_index(X, Y, percentage=0.5, random_state=1)

    assert X_1.shape == (3, 2)
    assert X_2.shape == (3, 2)
    assert Y_1.shape == (3,)
    assert Y_2.shape == (3,)
    assert I_1.shape == (3,)
    assert I_2.shape == (3,)
    torch.testing.assert_close(X_1, X[I_1])
    torch.testing.assert_close(X_2, X[I_2])
    torch.testing.assert_close(Y_1, Y[I_1])
    torch.testing.assert_close(Y_2, Y[I_2])
    assert sorted(torch.cat((I_1, I_2)).tolist()) == list(range(6))


@pytest.mark.parametrize("split", [splitter.split, splitter.split_with_index])
def test_split_rejects_mismatched_sample_counts(split):
    with pytest.raises(e.SizeError):
        split(torch.ones(5, 2), torch.ones(6, dtype=torch.int64), percentage=0.5, random_state=1)


def test_merge_preserves_partition_order():
    X_1 = torch.zeros(3, 2)
    Y_1 = torch.zeros(3, dtype=torch.int64)
    X_2 = torch.ones(3, 2)
    Y_2 = torch.ones(3, dtype=torch.int64)

    X, Y = splitter.merge(X_1, X_2, Y_1, Y_2)

    assert X.shape == (6, 2)
    assert Y.shape == (6,)
    torch.testing.assert_close(X[:3], X_1)
    torch.testing.assert_close(X[3:], X_2)
    torch.testing.assert_close(Y, torch.tensor([0, 0, 0, 1, 1, 1]))


@pytest.mark.parametrize(("first_size", "second_size"), [(2, 3), (3, 2)])
def test_merge_rejects_mismatched_partition_lengths(first_size, second_size):
    with pytest.raises(e.SizeError):
        splitter.merge(
            torch.ones(first_size, 2),
            torch.ones(second_size, 2),
            torch.ones(3, dtype=torch.int64),
            torch.ones(3, dtype=torch.int64),
        )


def test_merge_rejects_cancelling_partition_mismatches():
    with pytest.raises(e.SizeError):
        splitter.merge(
            torch.zeros(1, 1),
            torch.ones(2, 1),
            torch.zeros(2, dtype=torch.int64),
            torch.ones(1, dtype=torch.int64),
        )
