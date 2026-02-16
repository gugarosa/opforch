import torch

from opforch.models import supervised
from opforch.stream import loader, parser, splitter

csv = loader.load_csv("data/boat.csv")
X, Y = parser.parse_loader(csv)


def test_supervised_opf_fit():
    opf = supervised.SupervisedOPF(device="cpu")

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


def test_supervised_opf_predict():
    opf = supervised.SupervisedOPF(device="cpu")

    try:
        _ = opf.predict(X)
    except:
        opf.fit(X, Y)
        preds = opf.predict(X)

    assert len(preds) == 100

    try:
        opf.fit(X, Y)
        opf.subgraph.trained = False
        _ = opf.predict(X)
    except:
        opf.fit(X, Y)
        preds = opf.predict(X)

    assert len(preds) == 100

    opf.pre_computed_distance = True
    opf.pre_distances = torch.ones((100, 100), dtype=torch.float64)

    opf.fit(X, Y)
    preds = opf.predict(X)

    assert len(preds) == 100


def test_supervised_opf_learn():
    opf = supervised.SupervisedOPF(device="cpu")

    X_train, X_val, Y_train, Y_val = splitter.split(
        X, Y, percentage=0.1, random_state=1
    )

    opf.learn(X_train, Y_train, X_val, Y_val, n_iterations=5)

    assert isinstance(opf, supervised.SupervisedOPF)


def test_supervised_opf_prune():
    opf = supervised.SupervisedOPF(device="cpu")

    X_train, X_val, Y_train, Y_val = splitter.split(
        X, Y, percentage=0.5, random_state=1
    )

    opf.fit(X_train, Y_train)
    preds = opf.predict(X_val)

    assert len(preds) == X_val.shape[0]
