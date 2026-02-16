# OPForch: A PyTorch-Powered Optimum-Path Forest Classifier

[![Latest release](https://img.shields.io/github/release/gugarosa/opforch.svg)](https://github.com/gugarosa/opforch/releases)
[![Open issues](https://img.shields.io/github/issues/gugarosa/opforch.svg)](https://github.com/gugarosa/opforch/issues)
[![License](https://img.shields.io/github/license/gugarosa/opforch.svg)](https://github.com/gugarosa/opforch/blob/master/LICENSE)

## Welcome to OPForch.

*Note that this implementation relies purely on the standard [LibOPF](https://github.com/jppbsi/LibOPF). Therefore, if one uses our package, please also cite the original LibOPF [authors](https://github.com/jppbsi/LibOPF/wiki/Additional-information).*

OPForch is a **PyTorch-based** implementation of the Optimum-Path Forest (OPF) classifier, migrated from the original [OPFython](https://github.com/gugarosa/opfython) package. By replacing per-node Python objects with dense tensors and scalar Numba loops with batched tensor operations, OPForch delivers **massive speedups** while maintaining **zero prediction mismatches** against the reference implementation.

### Key Highlights

| Metric | Result |
|--------|--------|
| **Accuracy Parity** | 0 prediction mismatches across all 4 classifiers |
| **Predict Speedup** | Up to **484×** faster at N=10,000 |
| **Fit Speedup** | Up to **19×** faster at N=10,000 |
| **Distance Matrix** | Up to **413×** faster (batched tensor vs N² scalar loop) |
| **GPU Acceleration** | **12.7×** additional speedup on RTX 4070 for distance computation |
| **Device Support** | CPU, CUDA, and Multi-GPU via `DeviceManager` |

### Use OPForch if you need:

* Graph-based classification without hyperparameter tuning
* Deterministic training with competitive accuracy
* GPU-accelerated distance computation and prediction
* A drop-in replacement for OPFython with orders-of-magnitude speedups

OPForch is compatible with: **Python 3.8+** and **PyTorch 2.0+**.

---

## Package Structure

```
opforch/
├── core/
│   ├── heap.py          # Tensor-backed binary heap
│   ├── subgraph.py      # Dense tensor columns (13 state tensors)
│   └── opf.py           # Abstract base (torch.save/load, device)
├── math/
│   ├── distance.py      # 47 batched (N,D)×(M,D)→(N,M) distance metrics
│   ├── general.py       # Accuracy, confusion matrix, normalize, purity
│   └── random.py        # Tensor-based random generators
├── models/
│   ├── supervised.py        # MST + competition + batched predict
│   ├── knn_supervised.py    # KNN density clustering + k-selection
│   ├── semi_supervised.py   # Labeled + unlabeled propagation
│   └── unsupervised.py      # Density clustering + normalized cut
├── stream/
│   ├── loader.py        # CSV/TXT/JSON → torch.Tensor
│   ├── parser.py        # Extract features + labels
│   └── splitter.py      # Train/test split
├── subgraphs/
│   └── knn.py           # KNNSubgraph (torch.topk, vectorized PDF)
├── utils/
│   ├── constants.py     # EPSILON, FLOAT_MAX, status codes
│   ├── converter.py     # Binary OPF format converters
│   ├── device.py        # DeviceManager (CPU/GPU/multi-GPU)
│   ├── exception.py     # Custom exception hierarchy
│   └── logging.py       # Timed rotating file logger
├── benchmarks/          # Performance comparison suite
├── examples/            # Usage scripts for all 4 classifiers
└── report/              # Full migration report with benchmark plots
```

---

## Installation

Install from source:

```bash
git clone https://github.com/gugarosa/opforch.git
cd opforch
pip install -e .
```

For GPU support, install PyTorch with CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## Quick Start

### Supervised Classification

```python
import torch
from opforch.models import SupervisedOPF
from opforch.stream import loader, parser, splitter

# Load data
data = loader.load_txt("data/boat.txt")
X, Y = parser.parse_loader(data)
X_train, X_test, Y_train, Y_test = splitter.split(X, Y, percentage=0.5)

# Train and predict (CPU)
opf = SupervisedOPF(distance="log_squared_euclidean")
opf.fit(X_train, Y_train)
predictions = opf.predict(X_test)

# GPU — just change the device
opf_gpu = SupervisedOPF(distance="euclidean", device="cuda:0")
opf_gpu.fit(X_train.cuda(), Y_train.cuda())
predictions = opf_gpu.predict(X_test.cuda())
```

### Available Classifiers

| Classifier | Description |
|-----------|-------------|
| `SupervisedOPF` | MST-based prototype detection + cost competition |
| `KNNSupervisedOPF` | k-NN density clustering with validation-driven k |
| `SemiSupervisedOPF` | Extends supervised with unlabeled data propagation |
| `UnsupervisedOPF` | Density-based clustering with normalized cut |

All classifiers support `fit()`, `predict()`, `save()`, and `load()`, and accept a `device` parameter for CPU/GPU execution.

---

## Benchmarks

Run the benchmark suite to compare performance on your hardware:

```bash
# Baseline benchmarks (47 metrics, 4 models, scaling)
python benchmarks/benchmark.py

# Extended benchmarks (up to N=10K, GPU, dimensionality)
python benchmarks/benchmark_extended.py

# Generate plots
python benchmarks/plot_benchmarks.py
python benchmarks/plot_extended.py
```

For the full migration report with detailed analysis, see [`report/REPORT.md`](report/REPORT.md).

---

## Architecture

The key architectural change from OPFython is the elimination of per-node Python objects in favor of dense tensor columns:

```
OPFython:  subgraph.nodes[i].cost = 5.0        # Python object attribute
OPForch:   subgraph.costs[i] = 5.0             # Tensor element (GPU-ready)
```

Prediction is fully batched — a single tensor operation replaces the O(N×M) Python loop:

```python
dist_matrix = distance_fn(train_features, test_features)      # (N, M)
path_costs = torch.maximum(train_costs[:, None], dist_matrix)  # (N, M)
predictions = train_labels[path_costs.argmin(dim=0)]           # (M,)
```

For the complete architecture documentation, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Citation

If you use OPForch to fulfill any of your needs, please cite us:

```
J. P. Papa, A. X. Falcão and C. T. N. Suzuki.
Supervised Pattern Classification based on Optimum-Path Forest.
International Journal of Imaging Systems and Technology (2009).
```

---

## Datasets

Looking for datasets? We have some pre-loaded into OPF file format in the `data/` directory. More are available at [recogna.tech](http://recogna.tech).

---

## Support

If you ever need to report a bug, talk to us, or suggest improvements, please open an issue. We will do our best to help.

---
