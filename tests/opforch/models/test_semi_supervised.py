import torch

from opforch.models import semi_supervised
from opforch.stream import loader, parser

csv = loader.load_csv("data/boat.csv")
X, Y = parser.parse_loader(csv)


def test_semi_supervised_opf_fit():
    opf = semi_supervised.SemiSupervisedOPF(device="cpu")

    opf.fit(X, Y, X)

    assert opf.subgraph.trained is True
