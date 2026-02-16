"""Comprehensive benchmark suite: opfython vs opforch.

Measures accuracy parity and wall-clock performance across:
  1. Distance metrics (all 47, pairwise scalar vs batched tensor)
  2. Model training (fit) and prediction (predict) for all 4 classifiers
  3. Scaling behaviour with synthetic datasets of increasing size

Results are saved as JSON for downstream plotting.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
OPFYTHON_ROOT = ROOT / "opfython"
OPFORCH_ROOT = ROOT / "opforch"

sys.path.insert(0, str(OPFYTHON_ROOT))
sys.path.insert(0, str(OPFORCH_ROOT))

import logging

logging.disable(logging.CRITICAL)

import torch

# ---------------------------------------------------------------------------
# Imports – opfython
# ---------------------------------------------------------------------------
from opfython.math.distance import DISTANCES as PY_DISTS
from opfython.math.general import opf_accuracy as py_opf_accuracy
from opfython.models import (
    KNNSupervisedOPF as PyKNN,
    SemiSupervisedOPF as PySemi,
    SupervisedOPF as PySup,
    UnsupervisedOPF as PyUnsup,
)
from opfython.stream import loader as py_loader
from opfython.stream import parser as py_parser

# ---------------------------------------------------------------------------
# Imports – opforch
# ---------------------------------------------------------------------------
from opforch.math.distance import DISTANCES as TORCH_DISTS
from opforch.math.general import opf_accuracy as torch_opf_accuracy
from opforch.models import (
    KNNSupervisedOPF as TorchKNN,
    SemiSupervisedOPF as TorchSemi,
    SupervisedOPF as TorchSup,
    UnsupervisedOPF as TorchUnsup,
)
from opforch.stream import loader as torch_loader
from opforch.stream import parser as torch_parser


# ===========================================================================
# Helpers
# ===========================================================================

def _make_split(X, Y, ratio, seed):
    """Deterministic train/test split using numpy."""
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    n = int(ratio * len(idx))
    return np.sort(idx[:n]), np.sort(idx[n:])


def _make_synthetic(n_samples, n_features, n_classes, seed=42):
    """Create a synthetic dataset with Gaussian clusters."""
    np.random.seed(seed)
    X, Y = [], []
    per_class = n_samples // n_classes
    for c in range(n_classes):
        center = np.random.randn(n_features) * 3
        pts = center + np.random.randn(per_class, n_features) * 0.5
        X.append(pts)
        Y.append(np.full(per_class, c))
    X = np.vstack(X).astype(np.float64)
    Y = np.concatenate(Y).astype(np.int64)
    return X, Y


# ===========================================================================
# 1. Distance Metric Benchmarks
# ===========================================================================

def benchmark_distances(n_samples=100, n_features=10, repeats=3):
    """Time all 47 distance metrics: N×N scalar loop vs single batched tensor call."""
    print(f"\n=== Distance Metric Benchmarks ({n_samples}×{n_samples} pairwise matrix) ===")
    np.random.seed(42)

    X_np = np.random.rand(n_samples, n_features).astype(np.float64) + 0.1
    X_t = torch.tensor(X_np, dtype=torch.float32)

    results = []
    metric_names = sorted(PY_DISTS.keys())

    for name in metric_names:
        if name not in TORCH_DISTS:
            continue

        # opfython: N×N scalar calls
        times_py = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for i in range(n_samples):
                for j in range(n_samples):
                    PY_DISTS[name](X_np[i], X_np[j])
            times_py.append(time.perf_counter() - t0)
        py_time = min(times_py)

        # opforch: single batched call → (N, N)
        times_t = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            TORCH_DISTS[name](X_t, X_t)
            times_t.append(time.perf_counter() - t0)
        torch_time = min(times_t)

        speedup = py_time / max(torch_time, 1e-9)
        results.append({
            "metric": name,
            "opfython_ms": round(py_time * 1000, 3),
            "opforch_ms": round(torch_time * 1000, 3),
            "speedup": round(speedup, 2),
        })
        print(f"  {name:30s}  py={py_time*1000:8.2f}ms  torch={torch_time*1000:8.2f}ms  speedup={speedup:6.1f}x")

    return results


# ===========================================================================
# 2. Model Accuracy + Speed on Real Data
# ===========================================================================

def benchmark_models_real():
    """Benchmark all 4 models on boat.txt with identical splits."""
    print("\n=== Model Benchmarks (boat.txt) ===")

    txt_py = py_loader.load_txt(str(OPFYTHON_ROOT / "data" / "boat.txt"))
    X_np, Y_np = py_parser.parse_loader(txt_py)
    X_t = torch.tensor(X_np, dtype=torch.float32)
    Y_t = torch.tensor(Y_np, dtype=torch.int64)

    results = []

    # --- SupervisedOPF ---
    ti, vi = _make_split(X_np, Y_np, 0.5, seed=1)
    m = PySup(distance="log_squared_euclidean")
    t0 = time.perf_counter(); m.fit(X_np[ti], Y_np[ti]); fit_py = time.perf_counter() - t0
    t0 = time.perf_counter(); preds_py = m.predict(X_np[vi]); pred_py = time.perf_counter() - t0
    acc_py = py_opf_accuracy(Y_np[vi], preds_py)

    m2 = TorchSup(distance="log_squared_euclidean", device="cpu")
    t0 = time.perf_counter(); m2.fit(X_t[ti], Y_t[ti]); fit_t = time.perf_counter() - t0
    t0 = time.perf_counter(); preds_t = m2.predict(X_t[vi]); pred_t = time.perf_counter() - t0
    acc_t = torch_opf_accuracy(Y_t[vi], preds_t)
    mismatches = sum(1 for a, b in zip(preds_py, preds_t) if a != b)
    results.append({
        "model": "SupervisedOPF", "dataset": "boat.txt",
        "n_train": len(ti), "n_test": len(vi),
        "acc_opfython": round(acc_py, 6), "acc_opforch": round(acc_t, 6),
        "mismatches": mismatches,
        "fit_opfython_ms": round(fit_py * 1000, 2), "fit_opforch_ms": round(fit_t * 1000, 2),
        "predict_opfython_ms": round(pred_py * 1000, 2), "predict_opforch_ms": round(pred_t * 1000, 2),
        "fit_speedup": round(fit_py / max(fit_t, 1e-9), 2),
        "predict_speedup": round(pred_py / max(pred_t, 1e-9), 2),
    })
    print(f"  SupervisedOPF:     acc_py={acc_py:.6f}  acc_t={acc_t:.6f}  "
          f"fit={fit_py*1000:.1f}/{fit_t*1000:.1f}ms  pred={pred_py*1000:.1f}/{pred_t*1000:.1f}ms  mm={mismatches}")

    # --- KNNSupervisedOPF ---
    ti, vi = _make_split(X_np, Y_np, 0.8, seed=1)
    sti, svi = _make_split(X_np[ti], Y_np[ti], 0.25, seed=2)
    sub_train = ti[sti]; sub_val = ti[svi]
    m = PyKNN(max_k=10, distance="log_squared_euclidean")
    t0 = time.perf_counter(); m.fit(X_np[sub_train], Y_np[sub_train], X_np[sub_val], Y_np[sub_val]); fit_py = time.perf_counter() - t0
    t0 = time.perf_counter(); preds_py = m.predict(X_np[vi]); pred_py = time.perf_counter() - t0
    acc_py = py_opf_accuracy(Y_np[vi], preds_py)

    m2 = TorchKNN(max_k=10, distance="log_squared_euclidean", device="cpu")
    t0 = time.perf_counter(); m2.fit(X_t[sub_train], Y_t[sub_train], X_t[sub_val], Y_t[sub_val]); fit_t = time.perf_counter() - t0
    t0 = time.perf_counter(); preds_t = m2.predict(X_t[vi]); pred_t = time.perf_counter() - t0
    acc_t = torch_opf_accuracy(Y_t[vi], preds_t)
    mismatches = sum(1 for a, b in zip(preds_py, preds_t) if a != b)
    results.append({
        "model": "KNNSupervisedOPF", "dataset": "boat.txt",
        "n_train": len(sub_train), "n_test": len(vi),
        "acc_opfython": round(acc_py, 6), "acc_opforch": round(acc_t, 6),
        "mismatches": mismatches,
        "fit_opfython_ms": round(fit_py * 1000, 2), "fit_opforch_ms": round(fit_t * 1000, 2),
        "predict_opfython_ms": round(pred_py * 1000, 2), "predict_opforch_ms": round(pred_t * 1000, 2),
        "fit_speedup": round(fit_py / max(fit_t, 1e-9), 2),
        "predict_speedup": round(pred_py / max(pred_t, 1e-9), 2),
    })
    print(f"  KNNSupervisedOPF:  acc_py={acc_py:.6f}  acc_t={acc_t:.6f}  "
          f"fit={fit_py*1000:.1f}/{fit_t*1000:.1f}ms  pred={pred_py*1000:.1f}/{pred_t*1000:.1f}ms  mm={mismatches}")

    # --- SemiSupervisedOPF ---
    ti, vi = _make_split(X_np, Y_np, 0.8, seed=1)
    sti, sui = _make_split(X_np[ti], Y_np[ti], 0.25, seed=2)
    labeled = ti[sti]; unlabeled = ti[sui]
    m = PySemi(distance="log_squared_euclidean")
    t0 = time.perf_counter(); m.fit(X_np[labeled], Y_np[labeled], X_np[unlabeled]); fit_py = time.perf_counter() - t0
    t0 = time.perf_counter(); preds_py = m.predict(X_np[vi]); pred_py = time.perf_counter() - t0
    acc_py = py_opf_accuracy(Y_np[vi], preds_py)

    m2 = TorchSemi(distance="log_squared_euclidean", device="cpu")
    t0 = time.perf_counter(); m2.fit(X_t[labeled], Y_t[labeled], X_t[unlabeled]); fit_t = time.perf_counter() - t0
    t0 = time.perf_counter(); preds_t = m2.predict(X_t[vi]); pred_t = time.perf_counter() - t0
    acc_t = torch_opf_accuracy(Y_t[vi], preds_t)
    mismatches = sum(1 for a, b in zip(preds_py, preds_t) if a != b)
    results.append({
        "model": "SemiSupervisedOPF", "dataset": "boat.txt",
        "n_train": len(labeled), "n_test": len(vi),
        "acc_opfython": round(acc_py, 6), "acc_opforch": round(acc_t, 6),
        "mismatches": mismatches,
        "fit_opfython_ms": round(fit_py * 1000, 2), "fit_opforch_ms": round(fit_t * 1000, 2),
        "predict_opfython_ms": round(pred_py * 1000, 2), "predict_opforch_ms": round(pred_t * 1000, 2),
        "fit_speedup": round(fit_py / max(fit_t, 1e-9), 2),
        "predict_speedup": round(pred_py / max(pred_t, 1e-9), 2),
    })
    print(f"  SemiSupervisedOPF: acc_py={acc_py:.6f}  acc_t={acc_t:.6f}  "
          f"fit={fit_py*1000:.1f}/{fit_t*1000:.1f}ms  pred={pred_py*1000:.1f}/{pred_t*1000:.1f}ms  mm={mismatches}")

    # --- UnsupervisedOPF ---
    ti, vi = _make_split(X_np, Y_np, 0.5, seed=1)
    m = PyUnsup(min_k=1, max_k=10, distance="log_squared_euclidean")
    t0 = time.perf_counter(); m.fit(X_np[ti], Y_np[ti]); m.propagate_labels(); fit_py = time.perf_counter() - t0
    t0 = time.perf_counter(); preds_py = m.predict(X_np[vi])[0]; pred_py = time.perf_counter() - t0
    acc_py = py_opf_accuracy(Y_np[vi], preds_py)

    m2 = TorchUnsup(min_k=1, max_k=10, distance="log_squared_euclidean", device="cpu")
    t0 = time.perf_counter(); m2.fit(X_t[ti], Y_t[ti]); m2.propagate_labels(); fit_t = time.perf_counter() - t0
    t0 = time.perf_counter(); preds_t, _ = m2.predict(X_t[vi]); pred_t = time.perf_counter() - t0
    acc_t = torch_opf_accuracy(Y_t[vi], preds_t)
    mismatches = sum(1 for a, b in zip(preds_py, preds_t) if a != b)
    results.append({
        "model": "UnsupervisedOPF", "dataset": "boat.txt",
        "n_train": len(ti), "n_test": len(vi),
        "acc_opfython": round(acc_py, 6), "acc_opforch": round(acc_t, 6),
        "mismatches": mismatches,
        "fit_opfython_ms": round(fit_py * 1000, 2), "fit_opforch_ms": round(fit_t * 1000, 2),
        "predict_opfython_ms": round(pred_py * 1000, 2), "predict_opforch_ms": round(pred_t * 1000, 2),
        "fit_speedup": round(fit_py / max(fit_t, 1e-9), 2),
        "predict_speedup": round(pred_py / max(pred_t, 1e-9), 2),
    })
    print(f"  UnsupervisedOPF:   acc_py={acc_py:.6f}  acc_t={acc_t:.6f}  "
          f"fit={fit_py*1000:.1f}/{fit_t*1000:.1f}ms  pred={pred_py*1000:.1f}/{pred_t*1000:.1f}ms  mm={mismatches}")

    return results


# ===========================================================================
# 3. Scaling Benchmarks (Synthetic Data)
# ===========================================================================

def benchmark_scaling(sizes=None, n_features=10, n_classes=3, repeats=2):
    """Measure fit+predict time vs dataset size for SupervisedOPF."""
    if sizes is None:
        sizes = [50, 100, 200, 400, 800, 1500]

    print("\n=== Scaling Benchmarks (SupervisedOPF, synthetic data) ===")
    results = []

    for n in sizes:
        X_np, Y_np = _make_synthetic(n, n_features, n_classes)
        X_t = torch.tensor(X_np, dtype=torch.float32)
        Y_t = torch.tensor(Y_np, dtype=torch.int64)

        ti, vi = _make_split(X_np, Y_np, 0.7, seed=1)

        # opfython – fit
        fit_times_py = []
        for _ in range(repeats):
            m = PySup(distance="log_squared_euclidean")
            t0 = time.perf_counter()
            m.fit(X_np[ti], Y_np[ti])
            fit_times_py.append(time.perf_counter() - t0)
        fit_py = min(fit_times_py)

        # opfython – predict
        pred_times_py = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            preds_py = m.predict(X_np[vi])
            pred_times_py.append(time.perf_counter() - t0)
        pred_py = min(pred_times_py)
        acc_py = py_opf_accuracy(Y_np[vi], preds_py)

        # opforch – fit
        fit_times_t = []
        for _ in range(repeats):
            m2 = TorchSup(distance="log_squared_euclidean", device="cpu")
            t0 = time.perf_counter()
            m2.fit(X_t[ti], Y_t[ti])
            fit_times_t.append(time.perf_counter() - t0)
        fit_t = min(fit_times_t)

        # opforch – predict
        pred_times_t = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            preds_t = m2.predict(X_t[vi])
            pred_times_t.append(time.perf_counter() - t0)
        pred_t = min(pred_times_t)
        acc_t = torch_opf_accuracy(Y_t[vi], preds_t)

        results.append({
            "n_samples": n,
            "n_train": len(ti), "n_test": len(vi),
            "fit_opfython_ms": round(fit_py * 1000, 2),
            "fit_opforch_ms": round(fit_t * 1000, 2),
            "predict_opfython_ms": round(pred_py * 1000, 2),
            "predict_opforch_ms": round(pred_t * 1000, 2),
            "fit_speedup": round(fit_py / max(fit_t, 1e-9), 2),
            "predict_speedup": round(pred_py / max(pred_t, 1e-9), 2),
            "acc_opfython": round(acc_py, 6),
            "acc_opforch": round(acc_t, 6),
        })
        print(f"  N={n:5d}  fit: {fit_py*1000:8.1f}/{fit_t*1000:8.1f}ms ({fit_py/max(fit_t,1e-9):5.1f}x)  "
              f"pred: {pred_py*1000:8.1f}/{pred_t*1000:8.1f}ms ({pred_py/max(pred_t,1e-9):5.1f}x)  "
              f"acc: {acc_py:.4f}/{acc_t:.4f}")

    return results


# ===========================================================================
# 4. Distance Scaling Benchmarks
# ===========================================================================

def benchmark_distance_scaling(metric="log_squared_euclidean",
                                sizes=None, n_features=10, repeats=3):
    """Time N×N pairwise distance matrix: O(N²) scalar loop vs single batched call."""
    if sizes is None:
        sizes = [50, 100, 200, 400, 800]

    print(f"\n=== Distance Matrix Scaling ({metric}, N×N pairwise) ===")
    results = []

    for n in sizes:
        np.random.seed(42)
        X_np = np.random.rand(n, n_features).astype(np.float64) + 0.1
        X_t = torch.tensor(X_np, dtype=torch.float32)

        # opfython: N×N scalar calls (as done in training)
        times_py = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for i in range(n):
                for j in range(n):
                    PY_DISTS[metric](X_np[i], X_np[j])
            times_py.append(time.perf_counter() - t0)
        py_time = min(times_py)

        # opforch: single batched call → (N, N) matrix
        times_t = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            TORCH_DISTS[metric](X_t, X_t)
            times_t.append(time.perf_counter() - t0)
        torch_time = min(times_t)

        speedup = py_time / max(torch_time, 1e-9)
        results.append({
            "n_samples": n,
            "n_pairs": n * n,
            "opfython_ms": round(py_time * 1000, 2),
            "opforch_ms": round(torch_time * 1000, 2),
            "speedup": round(speedup, 2),
        })
        print(f"  N={n:5d} ({n*n:7d} pairs)  py={py_time*1000:10.2f}ms  torch={torch_time*1000:8.2f}ms  speedup={speedup:7.1f}x")

    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    all_results = {}

    all_results["distances"] = benchmark_distances(n_samples=100, n_features=10)
    all_results["models_real"] = benchmark_models_real()
    all_results["scaling"] = benchmark_scaling(
        sizes=[50, 100, 200, 400, 800, 1500]
    )
    all_results["distance_scaling"] = benchmark_distance_scaling(
        metric="log_squared_euclidean",
        sizes=[50, 100, 200, 400, 800],
    )

    out_path = out_dir / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
