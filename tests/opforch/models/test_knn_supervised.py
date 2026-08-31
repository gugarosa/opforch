import torch

from opforch.models import knn_supervised
from opforch.stream import loader, parser

csv = loader.load_csv("data/boat.csv")
X, Y = parser.parse_loader(csv)


def test_knn_supervised_opf_max_k():
    opf = knn_supervised.KNNSupervisedOPF(device="cpu")

    assert opf.max_k == 1


def test_knn_supervised_opf_max_k_setter():
    try:
        opf = knn_supervised.KNNSupervisedOPF(max_k=0, device="cpu")
    except:
        opf = knn_supervised.KNNSupervisedOPF(max_k=3, device="cpu")

    assert opf.max_k == 3


def test_knn_supervised_opf_fit():
    opf = knn_supervised.KNNSupervisedOPF(device="cpu")

    opf.fit(X, Y, X, Y)

    assert opf.subgraph.trained is True

    opf.pre_computed_distance = True
    try:
        opf.pre_distances = torch.ones((99, 99), dtype=torch.float64)
        opf.fit(X, Y, X, Y)
    except:
        opf.pre_distances = torch.ones((100, 100), dtype=torch.float64)
        opf.fit(X, Y, X, Y)

    assert opf.subgraph.trained is True


def test_knn_supervised_opf_predict():
    opf = knn_supervised.KNNSupervisedOPF(device="cpu")

    try:
        _ = opf.predict(X)
    except:
        opf.fit(X, Y, X, Y)
        preds = opf.predict(X)

    assert len(preds) == 100

    opf.pre_computed_distance = True
    opf.pre_distances = torch.ones((100, 100), dtype=torch.float64)

    opf.fit(X, Y, X, Y)
    preds = opf.predict(X)

    assert len(preds) == 100
