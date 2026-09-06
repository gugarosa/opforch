# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from opforch.math import random


def test_generate_uniform_random_number_returns_requested_shape():
    uniform_array = random.generate_uniform_random_number(0, 1, 5)

    assert uniform_array.shape == (5,)


def test_generate_gaussian_random_number_returns_requested_shape():
    gaussian_array = random.generate_gaussian_random_number(0, 1, 3)

    assert gaussian_array.shape == (3,)
