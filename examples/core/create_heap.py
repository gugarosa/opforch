# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Create an empty fixed-capacity minimum-priority heap.

"""

from opforch.core.heap import Heap

h = Heap(size=5, policy="min")
