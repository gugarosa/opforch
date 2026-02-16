import torch

from opforch.models import unsupervised
from opforch.stream import loader, parser

csv = loader.load_csv("data/boat.csv")
X, Y = parser.parse_loader(csv)


def test_unsupervised_opf_min_k():
    opf = unsupervised.UnsupervisedOPF(device="cpu")

    assert opf.min_k == 1


def test_unsupervised_opf_min_k_setter():
    try:
        opf = unsupervised.UnsupervisedOPF(min_k=0, device="cpu")
    except:
        opf = unsupervised.UnsupervisedOPF(min_k=1, device="cpu")

    assert opf.min_k == 1


def test_unsupervised_opf_max_k():
    opf = unsupervised.UnsupervisedOPF(device="cpu")

    assert opf.max_k == 1


def test_unsupervised_opf_max_k_setter():
    try:
        opf = unsupervised.UnsupervisedOPF(max_k=0, device="cpu")
    except:
        opf = unsupervised.UnsupervisedOPF(max_k=1, device="cpu")

    assert opf.max_k == 1

    try:
        opf = unsupervised.UnsupervisedOPF(min_k=2, max_k=1, device="cpu")
    except:
        opf = unsupervised.UnsupervisedOPF(min_k=1, max_k=3, device="cpu")

    assert opf.max_k == 3


def test_unsupervised_opf_fit():
    opf = unsupervised.UnsupervisedOPF(device="cpu")

    opf.fit(X, Y)

    assert opf.subgraph.trained is True

    opf.pre_computed_distance = True
    try:
        opf.pre_distances = torch.ones((99, 99), dtype=torch.float64)
        opf.fit(X, Y)
    except:
        opf.pre_distances = torch.ones((100, 100), dtype=torch.float64)
        opf.fit(X, Y)

    assert opf.subgraph.trained is True


def test_unsupervised_opf_predict():
    opf = unsupervised.UnsupervisedOPF(device="cpu")

    try:
        _ = opf.predict(X)
    except:
        opf.fit(X, Y)
        preds, clusters = opf.predict(X)

    assert len(preds) == 100
    assert len(clusters) == 100

    try:
        opf.fit(X, Y)
        opf.subgraph.trained = False
        _, _ = opf.predict(X)
    except:
        opf.fit(X, Y)
        preds, clusters = opf.predict(X)

    assert len(preds) == 100
    assert len(clusters) == 100

    opf.pre_computed_distance = True
    opf.pre_distances = torch.ones((100, 100), dtype=torch.float64)

    opf.fit(X, Y)
    preds, clusters = opf.predict(X)

    assert len(preds) == 100
    assert len(clusters) == 100


def test_unsupervised_opf_propagate_labels():
    opf = unsupervised.UnsupervisedOPF(device="cpu")

    opf.fit(X, Y)

    opf.propagate_labels()

    assert opf.subgraph.pred_labels[0].item() >= 0
