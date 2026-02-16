# OPForch — Architecture Guide: PyTorch Redesign

> **Version:** 2.0.0 (redesign) | **License:** Apache 2.0 | **Python:** 3.8+
> **Backend:** PyTorch (CPU, CUDA, Multi-GPU)

This document defines the complete architecture for **OPForch**, a ground-up redesign of [OPFython](https://github.com/gugarosa/opfython) that replaces NumPy + Numba with **PyTorch tensors** as the universal data representation. The goal is to eliminate Python-level loops, leverage GPU parallelism, and scale to large datasets — while preserving algorithmic accuracy with the original OPF classifiers.

---

## Table of Contents

1. [Design Motivation](#1-design-motivation)
2. [Performance Bottleneck Analysis](#2-performance-bottleneck-analysis)
3. [Architectural Principles](#3-architectural-principles)
4. [Package Structure](#4-package-structure)
5. [Module Deep-Dive](#5-module-deep-dive)
   - 5.1 [core — Tensor-First Data Structures](#51-core--tensor-first-data-structures)
   - 5.2 [math — Vectorized Distance & Metrics](#52-math--vectorized-distance--metrics)
   - 5.3 [models — GPU-Accelerated Classifiers](#53-models--gpu-accelerated-classifiers)
   - 5.4 [stream — Data I/O Pipeline](#54-stream--data-io-pipeline)
   - 5.5 [subgraphs — KNN with Batched Computation](#55-subgraphs--knn-with-batched-computation)
   - 5.6 [utils — Cross-Cutting Utilities](#56-utils--cross-cutting-utilities)
6. [Class Hierarchy](#6-class-hierarchy)
7. [Device Management Strategy](#7-device-management-strategy)
8. [Performance Optimization Map](#8-performance-optimization-map)
9. [Data Flow](#9-data-flow)
10. [Migration Guide from OPFython](#10-migration-guide-from-opfython)
11. [Dependencies](#11-dependencies)

---

## 1. Design Motivation

OPFython achieves correctness but suffers from fundamental performance limitations that prevent it from scaling:

| Problem | Root Cause | Impact |
|---------|-----------|--------|
| O(N²) Python loops in `fit()` and `predict()` | Per-node distance computed one-at-a-time in Python | Training time grows quadratically, unusable above ~10K samples |
| Per-node object allocation (`Node` class) | Each sample is a Python object with 15+ property-validated attributes | Massive memory overhead, GC pressure, no vectorization |
| Numba JIT on scalar pairs | Distance functions JIT-compiled but called N² times from Python | JIT overhead per call negates compilation benefit |
| No GPU path | NumPy arrays are CPU-only | Cannot leverage modern hardware |
| Sequential heap operations | Pure-Python heap processes one node at a time | Cannot be parallelized across cores or devices |

**OPForch's answer:** Replace the *node-centric, loop-driven* architecture with a *tensor-centric, batch-driven* architecture where:
- All N samples live in a single `(N, D)` tensor.
- All N×N distances are computed in one batched operation.
- Graph state (costs, labels, predecessors) are dense tensors operated on with masked indexing.
- The heap is eliminated where possible in favor of tensor-wide `min`/`max`/`topk` operations.

---

## 2. Performance Bottleneck Analysis

Below is a detailed mapping of every hot path in OPFython and the corresponding PyTorch optimization strategy:

### 2.1 Distance Computation (Critical — called O(N²) or O(N×M) times)

**OPFython:** Each distance is a Numba-JIT function called on two 1-D arrays from a Python `for` loop.

```python
# opfython: O(N²) Python iterations, each calling a JIT function on (D,) arrays
for q in range(self.subgraph.n_nodes):
    weight = self.distance_fn(nodes[p].features, nodes[q].features)
```

**OPForch:** Single batched call on `(N, D)` tensors, producing `(N, N)` or `(N, M)` distance matrix.

```python
# opforch: ONE call, entire matrix computed in parallel via BLAS/cuBLAS
dist_matrix = distance_fn(features_train, features_test)  # → (N, M)
```

**Key optimizations:**
- `torch.cdist(X, Y, p=2)` for Euclidean/Lp distances (uses optimized BLAS).
- Custom batch kernels for non-Lp metrics (Kullback-Leibler, etc.) using broadcasting: `X[:, None, :] op Y[None, :, :]` → `(N, M, D)` → reduced to `(N, M)`.
- For very large N where `(N, N)` doesn't fit in memory: **chunked computation** with configurable `chunk_size`.

### 2.2 MST / Prototype Discovery (`SupervisedOPF._find_prototypes`)

**OPFython:** Prim's algorithm via Python Heap + inner loop scanning all nodes per extraction.

```python
while not h.is_empty():        # N iterations
    p = h.remove()
    for q in range(n_nodes):    # N iterations per extraction → O(N²) total
        weight = distance(p, q)
```

**OPForch:** Pre-compute the full distance matrix once, then run Prim's using tensor operations:

```python
dist_matrix = distance_fn(features, features)    # (N, N), one shot
# Prim's with tensor masking:
while not_done.any():
    p = costs[in_tree.logical_not()].argmin()     # GPU-accelerated argmin
    in_tree[p] = True
    mask = ~in_tree
    updates = dist_matrix[p, mask] < costs[mask]  # vectorized comparison
    costs[mask] = torch.where(updates, dist_matrix[p, mask], costs[mask])
    preds[mask] = torch.where(updates, p, preds[mask])
```

This keeps the heap logic but replaces the O(N) inner scan with **masked tensor operations** that run on GPU.

### 2.3 Optimum-Path Competition (`SupervisedOPF.fit` — main loop)

**OPFython:** Dijkstra-like loop with O(N) inner scan per extraction.

**OPForch:** Pre-compute distance matrix, then use batch tensor operations:

```python
# For each extracted node p, update ALL remaining nodes at once:
mask = ~processed
arc_weights = dist_matrix[p, mask]
path_costs = torch.maximum(costs[p], arc_weights)    # minimax: vectorized
improved = path_costs < costs[mask]
costs[mask] = torch.where(improved, path_costs, costs[mask])
pred_labels[mask] = torch.where(improved, pred_labels[p], pred_labels[mask])
```

The sequential heap extraction (O(N) iterations) remains inherently serial, but the *inner work per iteration* becomes a vectorized GPU operation instead of a Python loop.

### 2.4 KNN Arc Creation (`KNNSubgraph.create_arcs`)

**OPFython:** O(N²) with insertion-sort per node to find k-NN.

**OPForch:** `torch.topk` on the distance matrix — one call, all nodes at once:

```python
dist_matrix = distance_fn(features, features)          # (N, N)
dist_matrix.fill_diagonal_(float('inf'))                # exclude self
knn_dists, knn_indices = dist_matrix.topk(k, largest=False)  # (N, k)
```

This replaces O(N² × k) insertion-sort with a single O(N² log k) parallel `topk`.

### 2.5 PDF Calculation (`KNNSubgraph.calculate_pdf`)

**OPFython:** Python loop over N nodes, inner loop over k neighbours, scalar exp calls.

**OPForch:** Fully vectorized with gathered distances:

```python
knn_distances = dist_matrix.gather(1, knn_indices)       # (N, k)
pdf = torch.exp(-knn_distances / constant).mean(dim=1)    # (N,)
density = (MAX_DENSITY - 1) * (pdf - pdf.min()) / (pdf.max() - pdf.min() + EPS) + 1
```

### 2.6 Prediction (`SupervisedOPF.predict`)

**OPFython:** For each of M test samples, iterates through N training samples sequentially.

**OPForch:** Batch all M test samples in parallel:

```python
# Compute all train-test distances at once: (N_train, M_test)
dist_matrix = distance_fn(train_features, test_features)

# For each test sample, find the training node offering minimum minimax cost
path_costs = torch.maximum(train_costs[:, None], dist_matrix)  # (N, M)
min_costs, best_nodes = path_costs.min(dim=0)                  # (M,)
predictions = train_labels[best_nodes]                          # (M,)
```

This replaces the O(M × N) Python loop with a single tensor operation.

### 2.7 Normalized Cut (`UnsupervisedOPF._normalized_cut`)

**OPFython:** Triple-nested Python loop (N × adjacency × distance).

**OPForch:** Vectorized using cluster label tensors:

```python
same_cluster = (cluster_labels[:, None] == cluster_labels[None, :])  # (N, N) bool
reciprocal = 1.0 / (dist_matrix + EPS)  # avoid div-by-zero
internal = (reciprocal * same_cluster * adjacency_mask).sum(dim=1)
external = (reciprocal * ~same_cluster * adjacency_mask).sum(dim=1)
# Scatter-add by cluster label for per-cluster totals
```

---

## 3. Architectural Principles

### 3.1 Tensor-First, Object-Last

| OPFython | OPForch |
|----------|---------|
| `List[Node]` — each node is a Python object | `Subgraph` holds dense tensors: `features: (N, D)`, `labels: (N,)`, `costs: (N,)`, etc. |
| `node.features`, `node.label`, `node.cost` | `subgraph.features[i]`, `subgraph.labels[i]`, `subgraph.costs[i]` |
| Property validation on every attribute set | Validation at construction time only; tensors enforce dtypes |

The `Node` class is **eliminated** as a per-sample container. All per-node state is stored as columns in the `Subgraph`'s tensor collection. This enables batch operations and GPU transfer of entire datasets at once.

### 3.2 Compute Once, Index Many

Distance matrices are computed **once** and stored as `(N, N)` or `(N, M)` tensors. All subsequent algorithms (MST, OPF competition, KNN, PDF, normalized cut) index into this pre-computed matrix rather than recomputing distances.

For datasets too large to fit an N×N matrix in memory, a **chunked** mode splits the computation into `(chunk, N)` blocks.

### 3.3 Device Transparency

All tensor operations go through a `DeviceManager` that:
- Auto-detects the best available device (Multi-GPU > single GPU > CPU).
- Provides `.to(device)` on all data structures.
- Supports `DataParallel` and `DistributedDataParallel` for multi-GPU distance computation.

### 3.4 API Compatibility

The public API (`fit`, `predict`, `learn`, `prune`, `save`, `load`) preserves the same signatures as OPFython, accepting both `np.ndarray` and `torch.Tensor` inputs. Return values are Python lists (predictions) for backward compatibility, with an option to return tensors.

---

## 4. Package Structure

```
opforch/                          # Root package (v2.0.0)
├── __init__.py                   # Version, public API re-exports
├── core/                         # Tensor-based data structures
│   ├── __init__.py
│   ├── heap.py                   # Tensor-backed priority queue
│   ├── opf.py                    # Abstract base classifier with device management
│   └── subgraph.py               # Dense tensor container (replaces Node + old Subgraph)
├── math/                         # Batched mathematical operations
│   ├── __init__.py
│   ├── distance.py               # Vectorized distance kernels (44 metrics)
│   ├── general.py                # Accuracy, confusion matrix, normalization
│   └── random.py                 # Tensor-based random generation
├── models/                       # GPU-accelerated classifiers
│   ├── __init__.py
│   ├── supervised.py             # SupervisedOPF (batched MST + competition)
│   ├── knn_supervised.py         # KNNSupervisedOPF (torch.topk KNN)
│   ├── semi_supervised.py        # SemiSupervisedOPF
│   └── unsupervised.py           # UnsupervisedOPF (batched clustering)
├── stream/                       # Data I/O pipeline
│   ├── __init__.py
│   ├── loader.py                 # CSV/TXT/JSON → Tensor loaders
│   ├── parser.py                 # OPF-format parsing
│   └── splitter.py               # Train/test splitting with tensor outputs
├── subgraphs/                    # Specialized subgraph variants
│   ├── __init__.py
│   └── knn.py                    # KNNSubgraph with batched arc creation + PDF
└── utils/                        # Cross-cutting utilities
    ├── __init__.py
    ├── constants.py              # Constants using torch dtypes
    ├── converter.py              # Binary OPF format converters
    ├── device.py                 # DeviceManager (CPU/GPU/Multi-GPU)
    ├── exception.py              # Custom exception hierarchy
    └── logging.py                # Logger factory
```

**Removed from OPFython:** `decorator.py` (the `@avoid_zero_division` pattern is replaced by adding epsilon inline in vectorized distance functions — no decorator overhead).

**Added:** `device.py` (centralized device management for CPU/GPU/Multi-GPU).

---

## 5. Module Deep-Dive

### 5.1 `core` — Tensor-First Data Structures

#### `Subgraph` (`core/subgraph.py`) — **Major Redesign**

The single most important architectural change. Instead of a `List[Node]` where each `Node` holds scalar attributes, the `Subgraph` stores **columnar tensors** on a target device.

```python
class Subgraph:
    def __init__(self, X, Y=None, I=None, device=None):
        self.device = device or DeviceManager.get_default()

        # Dense tensors — all (N,) or (N, D), on the target device
        self.features: torch.Tensor    # (N, D) float32
        self.labels: torch.Tensor      # (N,)   int64
        self.pred_labels: torch.Tensor # (N,)   int64
        self.cluster_labels: torch.Tensor # (N,) int64
        self.costs: torch.Tensor       # (N,)   float64
        self.densities: torch.Tensor   # (N,)   float64
        self.radii: torch.Tensor       # (N,)   float64
        self.n_plateaus: torch.Tensor  # (N,)   int64
        self.preds: torch.Tensor       # (N,)   int64  (-1 = NIL)
        self.roots: torch.Tensor       # (N,)   int64
        self.status: torch.Tensor      # (N,)   int8   (0=STANDARD, 1=PROTOTYPE)
        self.relevant: torch.Tensor    # (N,)   int8   (0=IRRELEVANT, 1=RELEVANT)

        # Sparse adjacency (for KNN variants)
        self.adjacency: Optional[torch.Tensor]  # (N, max_k) int64, -1 padded

        # Ordered index (filled during training)
        self.idx_nodes: torch.Tensor   # (N,) int64, filled incrementally

        # Metadata
        self.n_nodes: int
        self.n_features: int
        self.trained: bool
```

**Why columnar tensors?**
- **Batch slicing:** `self.costs[mask]` selects all costs matching a condition in one GPU operation.
- **Device transfer:** `self.to(device)` moves everything to GPU in one call.
- **Memory layout:** Contiguous tensors enable cache-friendly access and GPU coalescing.
- **No GC pressure:** One tensor allocation per column vs. N object allocations.

**Key methods:**

| Method | Description |
|--------|-------------|
| `to(device)` | Moves all tensors to the target device, returns `self` |
| `destroy_arcs()` | Zeros out `adjacency` and `n_plateaus` tensors |
| `mark_nodes(i)` | Walks predecessor chain, sets `relevant[i] = 1` (still sequential) |
| `reset()` | Resets `preds` to -1, `relevant` to 0, destroys arcs |
| `n_nodes` | Property returning `features.shape[0]` |

**Accepting both NumPy and Tensors:**
```python
def __init__(self, X, Y=None, I=None, device=None):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    self.features = X.to(dtype=torch.float32, device=self.device)
    ...
```

#### `Heap` (`core/heap.py`) — **Tensor-Backed**

The heap stores its state in tensors for consistency, but its operations remain sequential (heaps are inherently serial). However, the *work done per heap iteration* (distance lookups, cost comparisons) is now done via tensor indexing instead of Python object attribute access.

```python
class Heap:
    def __init__(self, size: int, policy: str = 'min', device=None):
        self.device = device or DeviceManager.get_default()
        self.cost = torch.full((size,), FLOAT_MAX, dtype=torch.float64, device=self.device)
        self.color = torch.zeros(size, dtype=torch.int8, device=self.device)   # WHITE=0
        self.p = torch.full((size,), -1, dtype=torch.int64, device=self.device)
        self.pos = torch.full((size,), -1, dtype=torch.int64, device=self.device)
        self.last = -1
```

**Note on heap elimination:** For `SupervisedOPF.predict()`, the heap is entirely replaced by a batched tensor operation (see §2.6). The heap is retained only for the training-phase graph algorithms (MST, OPF competition) where sequential extraction order matters.

#### `OPF` (`core/opf.py`) — **Abstract Base with Device Support**

```python
class OPF:
    def __init__(self, distance='log_squared_euclidean',
                 pre_computed_distance=None, device=None):
        self.device = device or DeviceManager.get_default()
        self.distance = distance
        self.distance_fn = DISTANCES[distance]  # now returns (N,M) tensor
        self.subgraph: Optional[Subgraph] = None
        ...

    def to(self, device) -> 'OPF':
        """Move entire model to a device."""
        self.device = device
        if self.subgraph:
            self.subgraph.to(device)
        return self

    def get_distances(self, normalize=False) -> torch.Tensor:
        """Compute full distance matrix using batched distance function."""
        return self.distance_fn(self.subgraph.features, self.subgraph.features)

    def save(self, path: str) -> None: ...   # torch.save (supports CUDA tensors)
    def load(self, path: str) -> None: ...   # torch.load with map_location

    def fit(self, X, Y, **kwargs): raise NotImplementedError
    def predict(self, X, **kwargs): raise NotImplementedError
```

---

### 5.2 `math` — Vectorized Distance & Metrics

#### `distance.py` — **44 Batched Distance Kernels**

Every distance function is redesigned to operate on **two 2-D tensors** and return a **2-D distance matrix**.

**Signature convention:**
```python
def euclidean_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Args:
        X: (N, D) tensor
        Y: (M, D) tensor
    Returns:
        (N, M) distance matrix
    """
```

**Implementation strategies by metric category:**

| Category | Strategy | Example |
|----------|----------|---------|
| Lp norms | `torch.cdist(X, Y, p=...)` | Euclidean, Manhattan, Chebyshev |
| Squared Euclidean | `‖X‖² + ‖Y‖² - 2XYᵀ` expansion trick | `squared_euclidean`, `log_squared_euclidean` |
| Pointwise formulas | Broadcasting `X[:, None, :] op Y[None, :, :]` → reduce | KL, Jensen, Chi-squared, etc. |
| Set-theoretic | Dot products via `X @ Y.T` | Cosine, Dice, Jaccard |

**Handling zero-division:** Instead of a decorator, epsilon is added inline:
```python
def chi_squared_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    Xe = X.unsqueeze(1) + EPS   # (N, 1, D)
    Ye = Y.unsqueeze(0) + EPS   # (1, M, D)
    return (0.5 * ((Xe - Ye) ** 2 / (Xe + Ye))).sum(dim=-1)  # (N, M)
```

**Chunked computation for large datasets:**
```python
def compute_distance_chunked(X, Y, distance_fn, chunk_size=4096):
    """Computes distance matrix in chunks to avoid OOM on large datasets."""
    N, M = X.shape[0], Y.shape[0]
    result = torch.empty(N, M, device=X.device)
    for i in range(0, N, chunk_size):
        result[i:i+chunk_size] = distance_fn(X[i:i+chunk_size], Y)
    return result
```

**The `DISTANCES` registry:**
```python
DISTANCES: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    'euclidean': euclidean_distance,
    'log_squared_euclidean': log_squared_euclidean_distance,
    ...  # all 44 metrics
}
```

#### `general.py` — **Tensor-Based Evaluation**

```python
def confusion_matrix(labels: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    """Vectorized confusion matrix using scatter_add."""
    n_class = labels.max() + 1
    indices = labels * n_class + preds
    return torch.zeros(n_class * n_class, device=labels.device).scatter_add_(
        0, indices, torch.ones_like(indices, dtype=torch.float32)
    ).reshape(n_class, n_class)

def opf_accuracy(labels, preds) -> float:
    """Same OPF accuracy formula, vectorized."""
    ...

def pre_compute_distance(X, output, distance='log_squared_euclidean'):
    """Compute full distance matrix and save. GPU-accelerated."""
    dist = DISTANCES[distance](X, X)  # (N, N) on GPU
    torch.save(dist.cpu(), output)
```

#### `random.py`
Thin wrappers around `torch.rand` and `torch.randn` with the same API as OPFython.

---

### 5.3 `models` — GPU-Accelerated Classifiers

#### 5.3.1 `SupervisedOPF` — Batched MST + Competition

```python
class SupervisedOPF(OPF):
    def fit(self, X_train, Y_train, I_train=None):
        self.subgraph = Subgraph(X_train, Y_train, I_train, device=self.device)

        # STEP 1: Compute full distance matrix ONCE
        dist_matrix = self.distance_fn(
            self.subgraph.features, self.subgraph.features
        )  # (N, N) — the single most expensive operation, fully parallelized

        # STEP 2: Find prototypes via batched Prim's MST
        self._find_prototypes(dist_matrix)

        # STEP 3: Optimum-path competition with tensor operations
        self._compete(dist_matrix)

        self.subgraph.trained = True

    def _find_prototypes(self, dist_matrix):
        """Prim's MST using tensor masking instead of Python Heap + inner loop."""
        N = self.subgraph.n_nodes
        in_tree = torch.zeros(N, dtype=torch.bool, device=self.device)
        costs = torch.full((N,), float('inf'), device=self.device)
        preds = torch.full((N,), -1, dtype=torch.int64, device=self.device)

        costs[0] = 0.0
        for _ in range(N):
            # Find cheapest node not yet in tree — single argmin on masked tensor
            candidates = costs.clone()
            candidates[in_tree] = float('inf')
            p = candidates.argmin().item()
            in_tree[p] = True

            # Check if p and its predecessor cross a class boundary → PROTOTYPE
            ...

            # Update costs for all non-tree nodes in ONE vectorized op
            mask = ~in_tree
            new_costs = dist_matrix[p][mask]
            improved = new_costs < costs[mask]
            costs[mask] = torch.where(improved, new_costs, costs[mask])
            preds[mask] = torch.where(improved, torch.tensor(p, device=self.device), preds[mask])

    def _compete(self, dist_matrix):
        """Optimum-path competition: Dijkstra with minimax cost, vectorized inner loop."""
        N = self.subgraph.n_nodes
        costs = torch.full((N,), float('inf'), dtype=torch.float64, device=self.device)
        processed = torch.zeros(N, dtype=torch.bool, device=self.device)

        # Prototypes start with cost 0
        proto_mask = self.subgraph.status == PROTOTYPE
        costs[proto_mask] = 0.0
        self.subgraph.pred_labels[proto_mask] = self.subgraph.labels[proto_mask]

        for _ in range(N):
            candidates = costs.clone()
            candidates[processed] = float('inf')
            p = candidates.argmin().item()
            processed[p] = True
            self.subgraph.costs[p] = costs[p]

            # Update all unprocessed nodes at once
            mask = ~processed & (costs[p] < costs)
            arc_weights = dist_matrix[p][mask]
            path_costs = torch.maximum(costs[p].expand_as(arc_weights), arc_weights)
            improved = path_costs < costs[mask]

            costs[mask] = torch.where(improved, path_costs, costs[mask])
            self.subgraph.preds[mask] = torch.where(
                improved, torch.tensor(p, device=self.device), self.subgraph.preds[mask]
            )
            self.subgraph.pred_labels[mask] = torch.where(
                improved, self.subgraph.pred_labels[p], self.subgraph.pred_labels[mask]
            )

    def predict(self, X_test, I_test=None):
        """Fully batched prediction — no loops over test samples."""
        test_features = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)

        # (N_train, M_test) distance matrix
        dist_matrix = self.distance_fn(self.subgraph.features, test_features)

        # Minimax cost from each training node to each test sample
        train_costs = self.subgraph.costs.unsqueeze(1)    # (N, 1)
        path_costs = torch.maximum(train_costs, dist_matrix)  # (N, M)

        # Best training node for each test sample (minimum minimax cost)
        _, best_nodes = path_costs.min(dim=0)  # (M,)

        predictions = self.subgraph.pred_labels[best_nodes]
        return predictions.cpu().tolist()
```

> **Accuracy note on predict():** The OPFython `predict()` uses an early-termination optimization where it stops iterating training nodes once `min_cost <= node.cost` (since nodes are sorted by cost). The batched version computes all pairs but produces *identical* results because `argmin` naturally selects the same winner. The early-termination is a constant-factor optimization that is dominated by the massive parallelism gain of the batched approach.

#### 5.3.2 `KNNSupervisedOPF` — Batched KNN + Density

```python
class KNNSupervisedOPF(OPF):
    def fit(self, X_train, Y_train, X_val, Y_val, ...):
        self.subgraph = KNNSubgraph(X_train, Y_train, device=self.device)

        # Distance matrix computed once for all k values
        dist_matrix = self.distance_fn(self.subgraph.features, self.subgraph.features)

        best_acc, best_k = 0.0, 1
        for k in range(1, self.max_k + 1):
            self.subgraph.create_arcs_from_matrix(dist_matrix, k)
            self.subgraph.calculate_pdf_from_matrix(dist_matrix, k)
            self._clustering()
            preds = self.predict(X_val)
            acc = opf_accuracy(Y_val, preds)
            if acc > best_acc:
                best_acc, best_k = acc, k

        # Final clustering with best_k
        ...

    def predict(self, X_test, ...):
        """KNN-based prediction fully batched."""
        test_features = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)

        # (N_train, M_test) distances
        dist_matrix = self.distance_fn(self.subgraph.features, test_features)

        # Top-k nearest training nodes for each test sample
        knn_dists, knn_idx = dist_matrix.topk(self.subgraph.best_k, dim=0, largest=False)
        # knn_dists: (k, M), knn_idx: (k, M)

        # Compute density for each test sample
        density = torch.exp(-knn_dists / self.subgraph.constant).mean(dim=0)  # (M,)
        density = ((MAX_DENSITY - 1) * (density - self.subgraph.min_density)
                   / (self.subgraph.max_density - self.subgraph.min_density + EPS)) + 1

        # Find best conqueror among k neighbours
        neighbour_costs = self.subgraph.costs[knn_idx]          # (k, M)
        compete_costs = torch.minimum(neighbour_costs, density.unsqueeze(0))  # (k, M)
        best_k_idx = compete_costs.argmax(dim=0)                # (M,)

        # Gather predictions from best neighbours
        best_neighbours = knn_idx.gather(0, best_k_idx.unsqueeze(0)).squeeze(0)  # (M,)
        predictions = self.subgraph.pred_labels[best_neighbours]
        return predictions.cpu().tolist()
```

#### 5.3.3 `SemiSupervisedOPF`

Extends `SupervisedOPF`. The key change is appending unlabeled features to the subgraph tensors before competition:

```python
def fit(self, X_train, Y_train, X_unlabeled, ...):
    # Build subgraph from labeled data, find prototypes
    ...

    # Append unlabeled samples to existing tensors
    unlabeled = torch.as_tensor(X_unlabeled, dtype=torch.float32, device=self.device)
    self.subgraph.features = torch.cat([self.subgraph.features, unlabeled], dim=0)
    self.subgraph.labels = torch.cat([self.subgraph.labels,
                                       torch.zeros(len(X_unlabeled), dtype=torch.int64, device=self.device)])
    # ... extend all other tensors ...

    # Recompute distance matrix with combined data
    dist_matrix = self.distance_fn(self.subgraph.features, self.subgraph.features)

    # Run competition (same as SupervisedOPF but also sets labels = pred_labels for unlabeled)
    self._compete(dist_matrix)
    mask_unlabeled = self.subgraph.labels == 0
    self.subgraph.labels[mask_unlabeled] = self.subgraph.pred_labels[mask_unlabeled]
```

#### 5.3.4 `UnsupervisedOPF` — Batched Clustering + Normalized Cut

```python
class UnsupervisedOPF(OPF):
    def fit(self, X_train, Y_train=None, ...):
        self.subgraph = KNNSubgraph(X_train, Y_train, device=self.device)

        # Single distance matrix for all k evaluations
        dist_matrix = self.distance_fn(self.subgraph.features, self.subgraph.features)

        # KNN arcs via topk (for max_k, then slice for smaller k)
        dist_no_self = dist_matrix.clone()
        dist_no_self.fill_diagonal_(float('inf'))
        all_knn_dists, all_knn_idx = dist_no_self.topk(self.max_k, largest=False)

        # Evaluate normalized cut for each k
        best_cut, best_k = float('inf'), self.min_k
        for k in range(self.min_k, self.max_k + 1):
            knn_dists_k = all_knn_dists[:, :k]
            knn_idx_k = all_knn_idx[:, :k]
            ...
            self._clustering(knn_idx_k, ...)
            cut = self._normalized_cut_batched(dist_matrix, knn_idx_k)
            if cut < best_cut:
                best_cut, best_k = cut, k

    def _normalized_cut_batched(self, dist_matrix, knn_idx):
        """Vectorized normalized cut computation."""
        N = self.subgraph.n_nodes
        cluster_labels = self.subgraph.cluster_labels  # (N,)

        # Build adjacency mask from knn_idx: (N, N) sparse bool
        adj = torch.zeros(N, N, dtype=torch.bool, device=self.device)
        rows = torch.arange(N, device=self.device).unsqueeze(1).expand_as(knn_idx)
        adj[rows, knn_idx] = True

        # Reciprocal distances for adjacent pairs
        reciprocal = 1.0 / (dist_matrix + EPS) * adj.float()

        # Same-cluster mask
        same = cluster_labels.unsqueeze(0) == cluster_labels.unsqueeze(1)  # (N, N)

        internal = (reciprocal * same.float()).sum()
        external = (reciprocal * (~same).float()).sum()

        # Per-cluster sums via scatter
        n_clusters = cluster_labels.max() + 1
        internal_per = torch.zeros(n_clusters, device=self.device)
        external_per = torch.zeros(n_clusters, device=self.device)
        for c in range(n_clusters):
            mask = cluster_labels == c
            internal_per[c] = (reciprocal[mask][:, mask]).sum()
            external_per[c] = (reciprocal[mask][:, ~mask]).sum()

        total = internal_per + external_per
        cut = (external_per / (total + EPS)).sum()
        return cut.item()
```

---

### 5.4 `stream` — Data I/O Pipeline

Largely the same as OPFython, but loaders now return `torch.Tensor` instead of `np.ndarray`:

```python
def load_csv(path: str, device=None) -> torch.Tensor:
    data = np.loadtxt(path, delimiter=',')
    return torch.from_numpy(data).to(device=device)
```

`splitter.py` uses `torch.randperm` instead of `np.random.permutation`:
```python
def split(X, Y, percentage=0.5, random_state=1):
    torch.manual_seed(random_state)
    idx = torch.randperm(X.shape[0])
    halt = int(len(X) * percentage)
    return X[idx[:halt]], X[idx[halt:]], Y[idx[:halt]], Y[idx[halt:]]
```

---

### 5.5 `subgraphs` — KNN with Batched Computation

#### `KNNSubgraph` (`subgraphs/knn.py`)

Extends `Subgraph` with KNN-specific tensors and methods that accept a pre-computed distance matrix.

```python
class KNNSubgraph(Subgraph):
    def __init__(self, ...):
        super().__init__(...)
        self.n_clusters: int = 0
        self.best_k: int = 0
        self.constant: float = 0.0
        self.min_density: float = 0.0
        self.max_density: float = 0.0

    def create_arcs_from_matrix(self, dist_matrix: torch.Tensor, k: int):
        """Batched KNN arc creation using torch.topk."""
        dist_no_self = dist_matrix.clone()
        dist_no_self.fill_diagonal_(float('inf'))

        knn_dists, knn_idx = dist_no_self.topk(k, largest=False)  # (N, k)

        self.adjacency = knn_idx              # (N, k) int64
        self.radii = knn_dists[:, -1]         # (N,) max distance among k-NN
        self.density_val = knn_dists.max().item()

    def calculate_pdf_from_matrix(self, dist_matrix: torch.Tensor, k: int):
        """Fully vectorized PDF computation."""
        self.constant = 2 * self.density_val / 9

        # Gather KNN distances
        knn_dists = dist_matrix.gather(1, self.adjacency)  # (N, k)

        pdf = torch.exp(-knn_dists / self.constant).mean(dim=1)  # (N,)
        self.min_density = pdf.min().item()
        self.max_density = pdf.max().item()

        if abs(self.min_density - self.max_density) < 1e-10:
            self.densities.fill_(MAX_DENSITY)
            self.costs.fill_(MAX_DENSITY - 1)
        else:
            self.densities = ((MAX_DENSITY - 1) * (pdf - self.min_density)
                              / (self.max_density - self.min_density)) + 1
            self.costs = self.densities - 1
```

---

### 5.6 `utils` — Cross-Cutting Utilities

#### `device.py` — **New Module**

```python
class DeviceManager:
    """Centralized device detection and management."""

    @staticmethod
    def get_default() -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    @staticmethod
    def get_all_gpus() -> List[torch.device]:
        """Returns list of all available CUDA devices."""
        return [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]

    @staticmethod
    def compute_distance_multi_gpu(X, Y, distance_fn, devices=None):
        """Distribute distance computation across multiple GPUs.

        Splits X into chunks (one per GPU), computes partial distance matrices
        in parallel, then concatenates the results.
        """
        if devices is None:
            devices = DeviceManager.get_all_gpus()
        if len(devices) <= 1:
            return distance_fn(X.to(devices[0]), Y.to(devices[0]))

        chunks = X.chunk(len(devices), dim=0)
        Y_replicas = [Y.to(d) for d in devices]

        results = []
        for chunk, device, Y_d in zip(chunks, devices, Y_replicas):
            results.append(distance_fn(chunk.to(device), Y_d))

        return torch.cat([r.to(devices[0]) for r in results], dim=0)
```

**Multi-GPU strategy:** The distance matrix computation (`(N, D) × (M, D) → (N, M)`) is embarrassingly parallel across rows. Split the source matrix `X` across GPUs, replicate `Y` on each, compute partial results, and gather back.

#### `constants.py`

```python
import torch

EPSILON = 1e-20
FLOAT_MAX = torch.finfo(torch.float64).max

# Heap colors
WHITE, GRAY, BLACK = 0, 1, 2

# Node state
NIL = -1
STANDARD, PROTOTYPE = 0, 1
IRRELEVANT, RELEVANT = 0, 1

# Scaling constants
MAX_ARC_WEIGHT = 100000
MAX_DENSITY = 1000
```

#### `exception.py`
Identical hierarchy to OPFython (`Error`, `ArgumentError`, `BuildError`, `SizeError`, `TypeError`, `ValueError`).

#### `converter.py`
Same binary-to-text converters, but output can optionally return `torch.Tensor` instead of saving to file.

#### `logging.py`
Identical to OPFython. Produces `opforch.log`.

---

## 6. Class Hierarchy

```
Subgraph                            (core/subgraph.py)
└── KNNSubgraph                     (subgraphs/knn.py)

OPF                                 (core/opf.py)
├── SupervisedOPF                   (models/supervised.py)
│   └── SemiSupervisedOPF           (models/semi_supervised.py)
├── KNNSupervisedOPF                (models/knn_supervised.py)
└── UnsupervisedOPF                 (models/unsupervised.py)

DeviceManager                       (utils/device.py)  [new]

Error                               (utils/exception.py)
├── ArgumentError
├── BuildError
├── SizeError
├── TypeError
└── ValueError
```

---

## 7. Device Management Strategy

### 7.1 CPU (Default)

PyTorch operations on CPU already benefit from:
- **Multi-threaded BLAS** (MKL/OpenBLAS) for matrix multiplications in distance computation.
- **Vectorized tensor operations** replacing Python loops.
- No Numba compilation warmup — PyTorch kernels are pre-compiled.

### 7.2 Single GPU

```python
opf = SupervisedOPF(distance='euclidean', device='cuda')
opf.fit(X_train, Y_train)
preds = opf.predict(X_test)
```

All tensors are allocated on GPU. Distance matrices are computed via cuBLAS. The sequential heap loop runs on CPU with tensor data accessed via `.item()`, avoiding GPU→CPU transfer overhead for the full tensors.

**Hybrid strategy for graph algorithms:**
- Distance matrix: computed and stored on **GPU**.
- Heap extraction loop: runs on **CPU** (inherently serial), but each iteration's inner work (masked updates on costs, labels, preds) is dispatched to **GPU**.
- Trade-off: N small GPU kernel launches vs. N² Python operations. The GPU wins even for modest N.

### 7.3 Multi-GPU

```python
opf = SupervisedOPF(distance='euclidean', device='cuda')
# Distance computation is automatically distributed across GPUs
# when DeviceManager detects multiple CUDA devices
opf.fit(X_train, Y_train)
```

The distance matrix is the only operation large enough to benefit from multi-GPU distribution. All subsequent graph operations run on a single GPU (they are sequential and memory-bound, not compute-bound).

### 7.4 Memory Management for Large Datasets

When `N × N × sizeof(float64)` exceeds device memory:

```python
opf = SupervisedOPF(distance='euclidean', device='cuda', chunk_size=8192)
```

The `chunk_size` parameter triggers **row-chunked** distance computation:
- Instead of materializing the full `(N, N)` matrix, process `(chunk, N)` blocks.
- Prim's MST and OPF competition are adapted to work with chunked access.
- Trade-off: more kernel launches, but fits arbitrarily large datasets.

---

## 8. Performance Optimization Map

A summary of every optimization, ordered by expected impact:

| # | Component | OPFython Complexity | OPForch Approach | Speedup Factor |
|---|-----------|-------------------|------------------|----------------|
| 1 | **Distance matrix** | O(N²) Python calls to Numba JIT | Single `torch.cdist` or batched broadcast | 100-1000× (GPU), 10-50× (CPU) |
| 2 | **Predict (supervised)** | O(M×N) Python loop | Batched `(N,M)` tensor min | 100-500× (GPU) |
| 3 | **KNN arc creation** | O(N²×k) insertion sort | `torch.topk` on `(N,N)` matrix | 50-200× (GPU) |
| 4 | **PDF calculation** | O(N×k) Python loop + scalar exp | Vectorized `gather` + `exp` + `mean` | 20-100× |
| 5 | **Predict (KNN)** | O(M×N) Python loop + insertion sort | `torch.topk` + batched density | 100-500× (GPU) |
| 6 | **Normalized cut** | O(N×adj) triple-nested loop | Tensor boolean masks + scatter | 20-100× |
| 7 | **MST inner loop** | O(N) Python loop per extraction | Masked tensor update per extraction | 5-20× |
| 8 | **Node attributes** | N Python objects, 15 properties each | Dense tensors, zero GC overhead | 2-10× memory |
| 9 | **Confusion matrix** | Python loop over N samples | `scatter_add_` | 10-50× |
| 10 | **Data loading** | NumPy → per-node copy | NumPy → single `torch.from_numpy` | 5-10× |

**Operations that remain sequential (cannot be parallelized):**
- Heap extraction order (Prim's, Dijkstra's) — inherent data dependency.
- Predecessor chain walking (`mark_nodes`) — sparse and small.
- Prototype identification after MST — O(N) scan.

These sequential parts are optimized by minimizing Python overhead: tensor indexing with `.item()` instead of object attribute access.

---

## 9. Data Flow

```
                        ┌─────────────────┐
                        │   Raw Files     │
                        │ .csv/.txt/.json │
                        └────────┬────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   stream/loader.py      │  → torch.Tensor (on target device)
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   stream/parser.py      │  parse_loader() → (X, Y) tensors
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   stream/splitter.py    │  split() → train/test tensors
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────▼──────────────────────────┐
        │         Subgraph / KNNSubgraph                     │
        │   Dense tensors: features(N,D), labels(N,), ...    │
        │   .to(device) → moves everything to GPU            │
        └────────────────────────┬──────────────────────────┘
                                 │
        ┌────────────────────────▼──────────────────────────┐
        │     Distance Matrix Computation                    │
        │  distance_fn(features, features) → (N,N) tensor    │
        │  [GPU-accelerated, multi-GPU capable, chunked]      │
        └────────────────────────┬──────────────────────────┘
                                 │
        ┌────────────────────────▼──────────────────────────┐
        │     Graph Algorithm (fit)                          │
        │  MST → Prototypes → Competition (heap loop)        │
        │  [Sequential heap, vectorized inner work per step] │
        └────────────────────────┬──────────────────────────┘
                                 │ trained model
        ┌────────────────────────▼──────────────────────────┐
        │     Batch Prediction (predict)                     │
        │  dist(train, test) → (N,M) → argmin → labels      │
        │  [Fully parallel, no loops]                        │
        └────────────────────────┬──────────────────────────┘
                                 │
        ┌────────────────────────▼──────────────────────────┐
        │     math/general.py                                │
        │  opf_accuracy(), purity() → scalar metrics         │
        └───────────────────────────────────────────────────┘
```

---

## 10. Migration Guide from OPFython

### 10.1 API Changes

| OPFython | OPForch | Notes |
|----------|---------|-------|
| `SupervisedOPF(distance=...)` | `SupervisedOPF(distance=..., device='cuda')` | New `device` parameter |
| `opf.fit(X, Y)` — accepts `np.ndarray` | Same — accepts both `np.ndarray` and `torch.Tensor` | Automatic conversion |
| `opf.predict(X)` → `List[int]` | Same → `List[int]` (default) | Add `return_tensor=True` for `torch.Tensor` |
| `loader.load_csv(path)` → `np.ndarray` | `loader.load_csv(path)` → `torch.Tensor` | Changed return type |
| `s.split(X, Y, ...)` — NumPy arrays | Same — torch tensors | Uses `torch.randperm` |
| `opf.save(path)` / `opf.load(path)` | Same — uses `torch.save`/`torch.load` | Supports cross-device loading |
| `Node(idx, label, features)` | **Removed** — use Subgraph tensor indexing | No per-sample objects |
| `Subgraph.nodes[i].cost` | `Subgraph.costs[i]` | Tensor column access |
| `from opfython.models import ...` | `from opforch.models import ...` | Package rename |

### 10.2 Behavioral Equivalence

All four classifiers (`SupervisedOPF`, `KNNSupervisedOPF`, `SemiSupervisedOPF`, `UnsupervisedOPF`) must produce **identical predictions** to OPFython given the same input data and distance metric. The only acceptable differences are floating-point rounding at ≤1e-10 magnitude due to different reduction orders in parallel operations.

**Validation strategy:** Run both OPFython and OPForch on the same datasets with the same random seeds and assert prediction equality in the test suite.

---

## 11. Dependencies

| Package | Min Version | Role |
|---------|-------------|------|
| `torch` | ≥ 2.0.0 | Tensor operations, GPU acceleration, serialization |
| `numpy` | ≥ 1.19.5 | Input compatibility (auto-converted to tensors) |

**Removed:** `numba` (replaced by PyTorch's built-in JIT and CUDA kernels).

**Development:**

| Package | Role |
|---------|------|
| `pytest` | Test framework |
| `coverage` | Coverage reporting |
| `pre-commit` | Git hooks |

**Optional:**

| Package | Role |
|---------|------|
| `torch` (CUDA build) | GPU acceleration |
