# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import pickle

import pytest
import torch

import opforch.utils.exception as e
from opforch.core.opf import OPF
from opforch.core.subgraph import Subgraph
from opforch.models.supervised import SupervisedOPF


@pytest.fixture
def fitted_opf():
    features = torch.tensor([[0.0, 0.0], [1.0, 0.0], [9.0, 0.0], [10.0, 0.0]])
    model = SupervisedOPF(distance="euclidean", device="cpu")
    model.pre_computed_distance = True
    model.pre_distances = torch.cdist(features, features)

    model.fit(features, torch.tensor([0, 0, 1, 1]))
    return model


def test_opf_subgraph_accepts_graph_state():
    model = OPF()

    model.subgraph = Subgraph()

    assert isinstance(model.subgraph, Subgraph)


def test_opf_distance_defaults_to_log_squared_euclidean():
    model = OPF()

    assert model.distance == "log_squared_euclidean"


def test_opf_distance_accepts_registered_metric():
    model = OPF(distance="euclidean")

    assert model.distance == "euclidean"


def test_opf_distance_rejects_unknown_metric():
    with pytest.raises(e.TypeError):
        OPF(distance="a")


def test_opf_distance_fn_is_callable():
    model = OPF()

    assert callable(model.distance_fn)


def test_opf_pre_computed_distance_defaults_to_false():
    model = OPF()

    assert model.pre_computed_distance is False


def test_opf_pre_distances_defaults_to_none():
    model = OPF()

    assert model.pre_distances is None


@pytest.mark.parametrize("suffix", [".txt", ".csv"])
def test_opf_read_distances_loads_text_table(data_dir, suffix):
    model = OPF(pre_computed_distance=str(data_dir / f"boat{suffix}"), device="cpu")

    assert model.pre_distances.shape == (100, 4)
    assert model.pre_distances.dtype == torch.float64
    assert model.pre_computed_distance is True


def test_opf_read_distances_rejects_unknown_extension(tmp_path):
    with pytest.raises(e.ArgumentError):
        OPF(pre_computed_distance=str(tmp_path / "distances"))


def test_opf_read_distances_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        OPF(pre_computed_distance=str(tmp_path / "missing.txt"))


def test_opf_read_distances_rejects_arbitrary_python_objects(tmp_path):
    output = tmp_path / "not-a-distance-tensor.pt"
    torch.save(tmp_path, output)

    with pytest.raises(pickle.UnpicklingError):
        OPF(pre_computed_distance=str(output), device="cpu")


def test_opf_save_load_preserves_distance_choice(tmp_path):
    model = OPF(distance="bray_curtis")
    output = tmp_path / "model.pt"

    model.save(str(output))

    assert output.is_file()

    restored = OPF()
    restored.load(str(output))

    assert restored.distance == "bray_curtis"


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_opf_save_preserves_live_model_state(tmp_path, fitted_opf, device):
    model = fitted_opf.to(device)
    model_state = vars(model).copy()
    graph_state = vars(model.subgraph).copy()
    graph_values = {name: value.clone() for name, value in graph_state.items() if isinstance(value, torch.Tensor)}
    pre_distances = model.pre_distances.clone()
    output = tmp_path / "model.pt"

    model.save(str(output))

    assert output.is_file()
    assert vars(model).keys() == model_state.keys()
    assert vars(model.subgraph).keys() == graph_state.keys()
    for name, value in model_state.items():
        assert getattr(model, name) is value
    for name, value in graph_state.items():
        assert getattr(model.subgraph, name) is value
    for name, value in graph_values.items():
        torch.testing.assert_close(getattr(model.subgraph, name), value)
    torch.testing.assert_close(model.pre_distances, pre_distances)

    checkpoint = torch.load(output, map_location=device, weights_only=False)

    assert isinstance(checkpoint, SupervisedOPF)
    assert checkpoint.device == torch.device(device)
    assert checkpoint.subgraph.device == torch.device(device)
    assert checkpoint.subgraph.features.device == torch.device(device)
    assert checkpoint.pre_distances.device == torch.device(device)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_opf_load_keeps_requested_device_coherent(tmp_path, fitted_opf, device):
    output = tmp_path / "model.pt"
    fitted_opf.save(str(output))
    restored = OPF(device=device)

    restored.load(str(output))

    assert restored.device == torch.device(device)
    assert restored.subgraph.device == torch.device(device)
    assert restored.pre_distances.device == torch.device(device)
    assert restored.pre_distances.shape == fitted_opf.pre_distances.shape
    assert restored.pre_distances.dtype == fitted_opf.pre_distances.dtype
    assert restored.distance == "euclidean"
    assert restored.pre_computed_distance is True
    assert restored.subgraph.trained is True
    assert restored.subgraph.n_nodes == 4
    assert restored.subgraph.n_features == 2
    assert restored.subgraph.idx_nodes == fitted_opf.subgraph.idx_nodes
    for name, value in vars(fitted_opf.subgraph).items():
        if isinstance(value, torch.Tensor):
            actual = getattr(restored.subgraph, name)
            assert actual.device == torch.device(device)
            torch.testing.assert_close(actual, value.to(device))


def test_opf_save_load_preserves_cpu_predictions(tmp_path, fitted_opf):
    output = tmp_path / "model.pt"
    fitted_opf.save(str(output))
    restored = SupervisedOPF(device="cpu")

    restored.load(str(output))

    assert restored.predict(fitted_opf.subgraph.features) == [0, 0, 1, 1]
    torch.testing.assert_close(restored.pre_distances, fitted_opf.pre_distances)


def test_opf_fit_requires_concrete_implementation():
    model = OPF()

    with pytest.raises(NotImplementedError):
        model.fit(None, None)


def test_opf_predict_requires_concrete_implementation():
    model = OPF()

    with pytest.raises(NotImplementedError):
        model.predict(None)


def test_opf_to_updates_device_metadata():
    model = OPF()

    result = model.to("cpu")

    assert result is model
    assert model.device == torch.device("cpu")


def test_opf_get_distances_normalizes_constant_values_to_zero():
    model = OPF(distance="euclidean", device="cpu")
    model.subgraph = Subgraph(torch.ones(3, 2), device="cpu")

    torch.testing.assert_close(model.get_distances(normalize=True), torch.zeros(3, 3))
