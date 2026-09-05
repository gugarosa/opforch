import pytest
import torch

import opforch.utils.exception as e
from opforch.core import subgraph
from opforch.utils import constants


def test_subgraph_n_nodes():
    s = subgraph.Subgraph()

    assert s.n_nodes == 0


def test_subgraph_n_features():
    s = subgraph.Subgraph()

    assert s.n_features == 0


def test_subgraph_trained():
    s = subgraph.Subgraph()

    assert s.trained is False


def test_subgraph_load():
    s = subgraph.Subgraph()

    try:
        X, Y = s._load("data/boat")
    except:
        X, Y = s._load("data/boat.csv")
        X, Y = s._load("data/boat.json")
        X, Y = s._load("data/boat.txt")

    assert X.shape == (100, 2)
    assert Y.shape == (100,)


def test_subgraph_build():
    s = subgraph.Subgraph()

    X, Y = s._load("data/boat.txt")

    s._build(X, Y, None)

    assert s.n_nodes == 100
    assert s.n_features == 2


def test_subgraph_build_with_index():
    s = subgraph.Subgraph()

    X, Y = s._load("data/boat.txt")

    I = Y

    s._build(X, Y, I)

    assert s.n_nodes == 100
    assert s.n_features == 2


def test_subgraph_from_file():
    s = subgraph.Subgraph(from_file="data/boat.txt")

    assert s.n_nodes == 100
    assert s.n_features == 2


def test_subgraph_from_tensors():
    X = torch.randn(10, 3)
    Y = torch.zeros(10, dtype=torch.int64)

    s = subgraph.Subgraph(X, Y)

    assert s.n_nodes == 10
    assert s.n_features == 3


def test_subgraph_destroy_arcs():
    s = subgraph.Subgraph(from_file="data/boat.txt")

    s.destroy_arcs()

    assert s.adjacency is None


def test_subgraph_mark_nodes():
    s = subgraph.Subgraph(from_file="data/boat.txt")

    s.mark_nodes(0)

    assert s.relevant[0].item() == constants.RELEVANT


def test_subgraph_reset():
    s = subgraph.Subgraph(from_file="data/boat.txt")

    s.reset()

    assert s.preds[0].item() == constants.NIL
    assert s.relevant[0].item() == constants.IRRELEVANT


def test_subgraph_to():
    s = subgraph.Subgraph(from_file="data/boat.txt")

    s.to("cpu")

    assert s.device == torch.device("cpu")


def test_empty_subgraph_can_change_device():
    s = subgraph.Subgraph(device="cpu").to("cpu")

    assert s.indices.shape == (0,)
    assert s.indices.dtype == torch.int64


def test_mark_nodes_follows_the_complete_predecessor_chain():
    s = subgraph.Subgraph(torch.zeros(4, 1), device="cpu")
    s.preds = torch.tensor([constants.NIL, 0, 1, 2])

    s.mark_nodes(3)
    s.mark_nodes(3)

    assert s.relevant.tolist() == [constants.RELEVANT] * 4
    with pytest.raises(e.ValueError):
        s.mark_nodes(constants.NIL)


@pytest.mark.parametrize(
    ("features", "labels", "indices", "error"),
    [
        (torch.zeros(2), None, None, e.SizeError),
        (torch.zeros(2, 1), torch.zeros(1), None, e.SizeError),
        (torch.zeros(2, 1), None, [0], e.SizeError),
        (torch.zeros(2, 1), None, [-1, 0], e.ValueError),
        (torch.zeros(2, 1), None, [0, 0.5], e.TypeError),
    ],
)
def test_subgraph_validates_sample_alignment(features, labels, indices, error):
    with pytest.raises(error):
        subgraph.Subgraph(features, labels, indices, device="cpu")
