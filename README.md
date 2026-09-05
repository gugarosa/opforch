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

Only load model checkpoints from trusted sources: whole-model `load()` uses
Python object serialization.

See [`examples/applications`](examples/applications) for complete classifier
workflows.

## Pre-computed distances

A distance file contains a square matrix for the complete dataset. When
splitting or reordering samples, pass their original **matrix row positions**
so training and prediction select the correct rows and columns:

```python
from opforch.math.general import pre_compute_distance

pre_compute_distance(X, "distances.pt", distance="euclidean")
X_train, X_test, Y_train, Y_test, I_train, I_test = splitter.split_with_index(X, Y)

model = SupervisedOPF(distance="euclidean", pre_computed_distance="distances.pt")
model.fit(X_train, Y_train, I_train=I_train)
predictions = model.predict(X_test, I_val=I_test)
```

The same indexing applies to all four classifiers and the `KNNSubgraph`
distance wrappers. Indices must be non-negative integers within the matrix.
`.pt` and `.pth` distance files use PyTorch's restricted tensor loader, not
whole-model deserialization.
Without explicit indices, each input is assumed to start at matrix position
zero; sample IDs are not inferred from feature values.

`SemiSupervisedOPF.fit()` accepts `I_unlabeled` for the unlabeled samples.
Its default is the positions immediately following the labeled samples.
`SupervisedOPF.learn()` and `prune()` accept `I_train` and `I_val`; indices
follow samples when they are exchanged or removed.

## Data and numerical contracts

Features have shape `(n_samples, n_features)`, with one label and, when
provided, one index per sample. Supervised and semi-supervised training
support a single labeled class. A k-NN training graph requires
`1 <= k < n_samples`, excluding self-neighbours.

Supervised prediction marks its winning nodes and their predecessor paths
as relevant. Pruning keeps these paths and class prototypes; it is not a
guarantee of unchanged accuracy on other data.

Label metrics require equally sized, non-empty vectors of non-negative
integer class IDs. Classes are numbered from zero; validation subsets may
omit classes. Confusion matrices include classes found only in predictions,
and purity supports more clusters than ground-truth classes. OPF accuracy
retains its false-positive/false-negative normalization, treating undefined
error-rate terms as zero. It is not ordinary fraction-correct accuracy.

Z-score normalization maps constant columns to zero. Min-max normalization
through `get_distances(normalize=True)` likewise returns zeros for a
constant distance matrix.

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
