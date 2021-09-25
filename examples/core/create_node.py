import torch

from opforch.core import Node

# Defines index, label and features
idx = 0
label = 0
features = torch.Tensor([1, 1.5, 2, 2.5, 3])

# Creates a Node
n = Node(idx, label, features)
