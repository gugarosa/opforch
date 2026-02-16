"""Constants definitions for OPForch."""

import sys

# Small value to prevent division by zero and log(0)
EPSILON = 1e-20

# Maximum float value for cost initialization
FLOAT_MAX = sys.float_info.max

# Heap node colors
WHITE = 0
GRAY = 1
BLACK = 2

# Predecessor sentinel
NIL = -1

# Node status flags
STANDARD = 0
PROTOTYPE = 1

# Node relevance flags
IRRELEVANT = 0
RELEVANT = 1

# Scaling factor for log-based distance metrics
MAX_ARC_WEIGHT = 100000

# Upper bound for normalized density
MAX_DENSITY = 1000
