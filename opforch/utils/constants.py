# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Define numerical guards and graph-state sentinels.

Attributes:
    EPSILON: Small positive guard against division by zero and logarithms of zero.
    FLOAT_MAX: Maximum Python float used to initialize unreached path costs.
    WHITE: Node color for nodes not yet inserted into a heap.
    GRAY: Node color for nodes currently in a heap.
    BLACK: Node color for nodes already removed from a heap.
    NIL: Missing predecessor sentinel, distinct from every valid node index.
    STANDARD: Status of a node that is not a prototype.
    PROTOTYPE: Status of a node selected as a prototype.
    IRRELEVANT: Relevance flag for an unselected node.
    RELEVANT: Relevance flag for a selected node.
    MAX_ARC_WEIGHT: Scaling factor for logarithmic distance metrics.
    MAX_DENSITY: Upper bound for normalized node density.

"""

import sys

EPSILON = 1e-20
FLOAT_MAX = sys.float_info.max

WHITE = 0
GRAY = 1
BLACK = 2
NIL = -1

STANDARD = 0
PROTOTYPE = 1
IRRELEVANT = 0
RELEVANT = 1

MAX_ARC_WEIGHT = 100000
MAX_DENSITY = 1000
