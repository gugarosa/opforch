import pickle

import pytest
import torch

from opforch.core import opf
from opforch.core.subgraph import Subgraph


def test_opf_subgraph():
    clf = opf.OPF()

    clf.subgraph = Subgraph()

    assert isinstance(clf.subgraph, Subgraph)


def test_opf_distance():
    clf = opf.OPF()

    assert clf.distance == "log_squared_euclidean"

    try:
        clf2 = opf.OPF(distance="a")
    except:
        clf2 = opf.OPF(distance="euclidean")

    assert clf2.distance == "euclidean"


def test_opf_distance_fn():
    clf = opf.OPF()

    assert callable(clf.distance_fn)


def test_opf_pre_computed_distance():
    clf = opf.OPF()

    assert clf.pre_computed_distance is False


def test_opf_pre_distances():
    clf = opf.OPF()

    assert clf.pre_distances is None


def test_opf_read_distances():
    try:
        clf = opf.OPF(pre_computed_distance="data/boat")
    except:
        clf = opf.OPF(pre_computed_distance="data/boat.txt")

    assert clf.pre_distances.shape == (100, 4)

    try:
        clf = opf.OPF(pre_computed_distance="data/boa.txt")
    except:
        clf = opf.OPF(pre_computed_distance="data/boat.csv")

    assert clf.pre_distances.shape == (100, 4)


def test_distance_files_do_not_load_arbitrary_python_objects(tmp_path):
    output = tmp_path / "not-a-distance-tensor.pt"
    torch.save(tmp_path, output)

    with pytest.raises(pickle.UnpicklingError):
        opf.OPF(pre_computed_distance=str(output), device="cpu")


def test_opf_save_and_load(tmp_path):
    clf = opf.OPF(distance="bray_curtis")
    output = tmp_path / "model.pt"

    clf.save(str(output))
    assert output.is_file()

    clf = opf.OPF()
    clf.load(str(output))
    assert clf.distance == "bray_curtis"


def test_opf_fit():
    clf = opf.OPF()

    with pytest.raises(NotImplementedError):
        clf.fit(None, None)


def test_opf_predict():
    clf = opf.OPF()

    with pytest.raises(NotImplementedError):
        clf.predict(None)


def test_opf_to():
    clf = opf.OPF()

    clf.to("cpu")

    assert clf.device == torch.device("cpu")


def test_normalized_constant_distances_are_zero():
    clf = opf.OPF(distance="euclidean", device="cpu")
    clf.subgraph = Subgraph(torch.ones(3, 2), device="cpu")

    torch.testing.assert_close(clf.get_distances(normalize=True), torch.zeros(3, 3))
