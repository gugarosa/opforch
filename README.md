# OPForch: PyTorch Optimum-Path Forest Classifiers

[![Latest release](https://img.shields.io/github/release/gugarosa/opforch.svg)](https://github.com/gugarosa/opforch/releases)
[![CI](https://github.com/gugarosa/opforch/actions/workflows/tests.yml/badge.svg)](https://github.com/gugarosa/opforch/actions/workflows/tests.yml)
[![License](https://img.shields.io/github/license/gugarosa/opforch.svg)](LICENSE)

OPForch implements Optimum-Path Forest classifiers with PyTorch tensors for
CPU and CUDA execution. It is a tensor-first successor to
[OPFython](https://github.com/gugarosa/opfython) and retains the familiar OPF
training, prediction, persistence, distance, streaming, and conversion APIs.

## Installation

OPForch requires Python 3.8 or newer and PyTorch 2.0 or newer.

```bash
pip install opforch
```

For development:

```bash
git clone https://github.com/gugarosa/opforch.git
cd opforch
uv sync --locked
```

## Quick start

```python
from opforch.models import SupervisedOPF
from opforch.stream import loader, parser, splitter

data = loader.load_txt("data/boat.txt")
X, Y = parser.parse_loader(data)
X_train, X_test, Y_train, Y_test = splitter.split(X, Y)

model = SupervisedOPF(distance="log_squared_euclidean")
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
```

Select a CUDA device through the existing `device` argument:

```python
model = SupervisedOPF(distance="euclidean", device="cuda:0")
```

## Classifiers

| Classifier | Description |
|---|---|
| `SupervisedOPF` | MST-based prototype detection and cost competition |
| `KNNSupervisedOPF` | k-NN density clustering with validation-driven k selection |
| `SemiSupervisedOPF` | Labeled and unlabeled sample propagation |
| `UnsupervisedOPF` | Density clustering with normalized-cut selection |

All classifiers retain `fit()`, `predict()`, `save()`, and `load()` support.
The package also includes 47 distance metrics, OPF data loaders and
converters, train/test split helpers, tensor-backed subgraphs, and the public
`DeviceManager` API.

See [`examples/applications`](examples/applications) for complete classifier
workflows.

## Development

```bash
uv run pytest
uv run pre-commit run --all-files
uv build
```

## Citation

If you use OPForch, please also cite the original LibOPF authors:

```text
J. P. Papa, A. X. Falcão and C. T. N. Suzuki.
Supervised Pattern Classification based on Optimum-Path Forest.
International Journal of Imaging Systems and Technology (2009).
```

## Support

Open an [issue](https://github.com/gugarosa/opforch/issues) for bug reports,
questions, and suggestions.
