import pytest
import torch

from opforch.models import SemiSupervisedOPF, supervised
from opforch.stream import loader, parser, splitter
from opforch.utils import constants

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

    opf.prune(X_train, Y_train, X_val, Y_val, n_iterations=2)
    preds = opf.predict(X_val)

    assert len(preds) == X_val.shape[0]
    assert 0 < opf.subgraph.n_nodes <= len(X_train)


@pytest.mark.parametrize("model_class", [supervised.SupervisedOPF, SemiSupervisedOPF])
@pytest.mark.parametrize("n_samples", [1, 3])
def test_single_class_training(model_class, n_samples):
    features = torch.arange(n_samples, dtype=torch.float32).reshape(-1, 1)
    labels = torch.full((n_samples,), 7)
    opf = model_class(distance="euclidean", device="cpu")

    if model_class is SemiSupervisedOPF:
        opf.fit(features, labels, torch.tensor([[5.0]]))
    else:
        opf.fit(features, labels)

    assert opf.predict(torch.tensor([[0.5], [20.0]])) == [7, 7]
    assert sorted(opf.subgraph.idx_nodes) == list(range(opf.subgraph.n_nodes))
    assert (opf.subgraph.costs < constants.FLOAT_MAX).all()


def test_prediction_marks_the_winner_and_its_predecessors():
    opf = supervised.SupervisedOPF(distance="euclidean", device="cpu")
    opf.fit(torch.tensor([[0.0], [1.0], [9.0], [10.0]]), torch.tensor([0, 0, 1, 1]))

    assert opf.predict(torch.tensor([[0.0]])) == [0]
    assert opf.subgraph.relevant[0] == constants.RELEVANT
    assert opf.subgraph.relevant[1] == constants.RELEVANT
    assert opf.predict(torch.tensor([[10.0]])) == [1]
    assert opf.subgraph.relevant.tolist() == [1, 1, 1, 0]


@pytest.mark.parametrize("validation_indices", [[0], [0, 3]])
def test_pruning_preserves_prototypes_and_predictions(validation_indices):
    features = torch.tensor([[0.0], [1.0], [9.0], [10.0]])
    labels = torch.tensor([0, 0, 1, 1])
    validation = features[validation_indices]
    opf = supervised.SupervisedOPF(distance="euclidean", device="cpu")

    opf.prune(features, labels, validation, labels[validation_indices], n_iterations=2)

    assert 0 < opf.subgraph.n_nodes < len(features)
    assert set(opf.subgraph.labels.tolist()) == {0, 1}
    assert opf.predict(validation) == labels[validation_indices].tolist()
    torch.testing.assert_close(features, torch.tensor([[0.0], [1.0], [9.0], [10.0]]))


def test_learn_can_select_a_zero_accuracy_model():
    features = torch.tensor([[0.0], [10.0]])
    labels = torch.tensor([0, 1])
    opf = supervised.SupervisedOPF(distance="euclidean", device="cpu")

    opf.learn(features, labels, features, labels.flip(0), n_iterations=1)

    assert opf.subgraph.trained
    assert opf.predict(features) == [0, 1]


def test_supervised_minimax_costs_match_a_known_forest():
    opf = supervised.SupervisedOPF(distance="euclidean", device="cpu")
    opf.fit(torch.tensor([[0.0], [1.0], [4.0], [5.0]]), torch.tensor([0, 0, 1, 1]))

    assert opf.subgraph.status.tolist() == [
        constants.STANDARD,
        constants.PROTOTYPE,
        constants.PROTOTYPE,
        constants.STANDARD,
    ]
    torch.testing.assert_close(
        opf.subgraph.costs, torch.tensor([1.0, 0.0, 0.0, 1.0]).double()
    )
    assert opf.predict(torch.tensor([[-1.0], [0.5], [2.0], [3.0], [6.0]])) == [
        0,
        0,
        0,
        1,
        1,
    ]
