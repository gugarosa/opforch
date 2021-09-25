"""Constants definitions.
"""

import torch

# When the costs are initialized, their value are defined as
# the maximum float value possible
FLOAT_MAX = torch.finfo(torch.float64).max
