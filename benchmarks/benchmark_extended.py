"""Extended benchmark suite: larger datasets + GPU benchmarks.

Extends the baseline benchmarks with:
  1. Larger scaling datasets (up to N=10000)
  2. GPU benchmarks for distance, fit, and predict (when CUDA available)
  3. CPU vs GPU comparison across dataset sizes
  4. Multi-model scaling on larger data

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
# Imports
# ---------------------------------------------------------------------------
from opfython.math.distance import DISTANCES as PY_DISTS
from opfython.math.general import opf_accuracy as py_opf_accuracy
from opfython.models import SupervisedOPF as PySup

from opforch.math.distance import DISTANCES as TORCH_DISTS
from opforch.math.general import opf_accuracy as torch_opf_accuracy
from opforch.models import (
    KNNSupervisedOPF as TorchKNN,
    SemiSupervisedOPF as TorchSemi,
    SupervisedOPF as TorchSup,
    UnsupervisedOPF as TorchUnsup,
)

HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    GPU_NAME = torch.cuda.get_device_name(0)
    print(f"CUDA available: {GPU_NAME}")
else:
    GPU_NAME = "N/A"
    print("CUDA not available — GPU benchmarks will be skipped")

print(f"PyTorch version: {torch.__version__}")
print(f"CPU: torch uses {torch.get_num_threads()} threads")


# ===========================================================================
# Helpers
# ===========================================================================

def _make_split(X, Y, ratio, seed):
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    n = int(ratio * len(idx))
    return np.sort(idx[:n]), np.sort(idx[n:])


def _make_synthetic(n_samples, n_features, n_classes, seed=42):
    np.random.seed(seed)
    X, Y = [], []
    per_class = n_samples // n_classes
    for c_idx in range(n_classes):
        center = np.random.randn(n_features) * 3
        pts = center + np.random.randn(per_class, n_features) * 0.5
        X.append(pts)
        Y.append(np.full(per_class, c_idx))
    X = np.vstack(X).astype(np.float64)
    Y = np.concatenate(Y).astype(np.int64)
    return X, Y


def _warmup_gpu():
    """Run a small tensor op to initialize CUDA context."""
    if HAS_CUDA:
        x = torch.randn(100, 100, device="cuda")
        _ = x @ x.T
        torch.cuda.synchronize()


# ===========================================================================
# 1. Extended Scaling (CPU): up to N=10000
# ===========================================================================

def benchmark_extended_scaling(sizes=None, n_features=10, n_classes=5, repeats=2):
    """Measure fit+predict time for larger datasets on CPU."""
    if sizes is None:
        sizes = [100, 200, 500, 1000, 2000, 3000, 5000, 8000, 10000]

    print(f"\n{'='*70}")
    print(f"  Extended Scaling (CPU) — SupervisedOPF, {n_features}D, {n_classes} classes")
    print(f"{'='*70}")

    results = []
    for n in sizes:
        X_np, Y_np = _make_synthetic(n, n_features, n_classes)
        X_t = torch.tensor(X_np, dtype=torch.float32)
        Y_t = torch.tensor(Y_np, dtype=torch.int64)
        ti, vi = _make_split(X_np, Y_np, 0.7, seed=1)

        # opfython fit
        fit_py = float("inf")
        for _ in range(repeats):
            m = PySup(distance="log_squared_euclidean")
            t0 = time.perf_counter()
            m.fit(X_np[ti], Y_np[ti])
            fit_py = min(fit_py, time.perf_counter() - t0)

        # opfython predict
        pred_py = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            preds_py = m.predict(X_np[vi])
            pred_py = min(pred_py, time.perf_counter() - t0)
        acc_py = py_opf_accuracy(Y_np[vi], preds_py)

        # opforch CPU fit
        fit_t = float("inf")
        for _ in range(repeats):
            m2 = TorchSup(distance="log_squared_euclidean", device="cpu")
            t0 = time.perf_counter()
            m2.fit(X_t[ti], Y_t[ti])
            fit_t = min(fit_t, time.perf_counter() - t0)

        # opforch CPU predict
        pred_t = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            preds_t = m2.predict(X_t[vi])
            pred_t = min(pred_t, time.perf_counter() - t0)
        acc_t = torch_opf_accuracy(Y_t[vi], torch.tensor(preds_t, dtype=torch.int64) if not isinstance(preds_t, torch.Tensor) else preds_t)

        results.append({
            "n_samples": n,
            "n_train": len(ti), "n_test": len(vi),
            "fit_opfython_ms": round(fit_py * 1000, 2),
            "fit_opforch_cpu_ms": round(fit_t * 1000, 2),
            "predict_opfython_ms": round(pred_py * 1000, 2),
            "predict_opforch_cpu_ms": round(pred_t * 1000, 2),
            "fit_speedup": round(fit_py / max(fit_t, 1e-9), 2),
            "predict_speedup": round(pred_py / max(pred_t, 1e-9), 2),
            "acc_opfython": round(acc_py, 6),
            "acc_opforch": round(float(acc_t), 6),
        })

        print(f"  N={n:6d}  fit: {fit_py*1000:10.1f} / {fit_t*1000:10.1f} ms ({fit_py/max(fit_t,1e-9):6.1f}x)  "
              f"pred: {pred_py*1000:8.1f} / {pred_t*1000:8.1f} ms ({pred_py/max(pred_t,1e-9):7.1f}x)  "
              f"acc: {acc_py:.4f}/{float(acc_t):.4f}")

    return results


# ===========================================================================
# 2. Extended Distance Scaling
# ===========================================================================

def benchmark_extended_distance_scaling(metric="log_squared_euclidean",
                                         sizes=None, n_features=10, repeats=3):
    """Distance matrix scaling up to larger N."""
    if sizes is None:
        sizes = [50, 100, 200, 500, 1000, 2000]

    print(f"\n{'='*70}")
    print(f"  Extended Distance Scaling — {metric}")
    print(f"{'='*70}")

    results = []
    for n in sizes:
        np.random.seed(42)
        X_np = np.random.rand(n, n_features).astype(np.float64) + 0.1
        X_t = torch.tensor(X_np, dtype=torch.float32)

        # opfython: N×N scalar calls
        py_time = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            for i in range(n):
                for j in range(n):
                    PY_DISTS[metric](X_np[i], X_np[j])
            py_time = min(py_time, time.perf_counter() - t0)

        # opforch CPU: single batched call
        t_time = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            TORCH_DISTS[metric](X_t, X_t)
            t_time = min(t_time, time.perf_counter() - t0)

        speedup = py_time / max(t_time, 1e-9)
        results.append({
            "n_samples": n, "n_pairs": n * n,
            "opfython_ms": round(py_time * 1000, 2),
            "opforch_cpu_ms": round(t_time * 1000, 2),
            "speedup_cpu": round(speedup, 2),
        })

        print(f"  N={n:5d} ({n*n:10d} pairs)  py={py_time*1000:10.2f}ms  "
              f"torch_cpu={t_time*1000:8.2f}ms  speedup={speedup:8.1f}x")

    return results


# ===========================================================================
# 3. GPU Benchmarks (distance, fit, predict)
# ===========================================================================

def benchmark_gpu_distance(metric="log_squared_euclidean",
                           sizes=None, n_features=10, repeats=5):
    """CPU vs GPU distance computation."""
    if not HAS_CUDA:
        print("\n  [SKIP] GPU distance benchmark — no CUDA")
        return []

    if sizes is None:
        sizes = [100, 500, 1000, 2000, 5000, 10000]

    _warmup_gpu()

    print(f"\n{'='*70}")
    print(f"  GPU Distance Benchmark — {metric} (CPU vs CUDA)")
    print(f"{'='*70}")

    results = []
    for n in sizes:
        np.random.seed(42)
        X_np = np.random.rand(n, n_features).astype(np.float32) + 0.1
        X_cpu = torch.tensor(X_np)
        X_gpu = X_cpu.cuda()

        dist_fn = TORCH_DISTS[metric]

        # CPU timing
        cpu_time = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = dist_fn(X_cpu, X_cpu)
            cpu_time = min(cpu_time, time.perf_counter() - t0)

        # GPU timing (with sync)
        gpu_time = float("inf")
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = dist_fn(X_gpu, X_gpu)
            torch.cuda.synchronize()
            gpu_time = min(gpu_time, time.perf_counter() - t0)

        speedup = cpu_time / max(gpu_time, 1e-9)
        results.append({
            "n_samples": n,
            "cpu_ms": round(cpu_time * 1000, 3),
            "gpu_ms": round(gpu_time * 1000, 3),
            "gpu_speedup": round(speedup, 2),
        })

        print(f"  N={n:6d}  cpu={cpu_time*1000:10.3f}ms  gpu={gpu_time*1000:10.3f}ms  "
              f"GPU speedup={speedup:6.1f}x")

    return results


def benchmark_gpu_model(sizes=None, n_features=10, n_classes=5, repeats=2):
    """CPU vs GPU model fit+predict for SupervisedOPF."""
    if not HAS_CUDA:
        print("\n  [SKIP] GPU model benchmark — no CUDA")
        return []

    if sizes is None:
        sizes = [100, 500, 1000, 2000, 5000]

    _warmup_gpu()

    print(f"\n{'='*70}")
    print(f"  GPU Model Benchmark — SupervisedOPF (CPU vs CUDA)")
    print(f"{'='*70}")

    results = []
    for n in sizes:
        X_np, Y_np = _make_synthetic(n, n_features, n_classes)
        X_t = torch.tensor(X_np, dtype=torch.float32)
        Y_t = torch.tensor(Y_np, dtype=torch.int64)
        ti, vi = _make_split(X_np, Y_np, 0.7, seed=1)

        X_train_cpu, Y_train_cpu = X_t[ti], Y_t[ti]
        X_val_cpu = X_t[vi]
        X_train_gpu = X_train_cpu.cuda()
        Y_train_gpu = Y_train_cpu.cuda()
        X_val_gpu = X_val_cpu.cuda()

        # CPU fit
        fit_cpu = float("inf")
        for _ in range(repeats):
            m = TorchSup(distance="log_squared_euclidean", device="cpu")
            t0 = time.perf_counter()
            m.fit(X_train_cpu, Y_train_cpu)
            fit_cpu = min(fit_cpu, time.perf_counter() - t0)

        # CPU predict
        pred_cpu = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = m.predict(X_val_cpu)
            pred_cpu = min(pred_cpu, time.perf_counter() - t0)

        # GPU fit
        fit_gpu = float("inf")
        for _ in range(repeats):
            m2 = TorchSup(distance="log_squared_euclidean", device="cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            m2.fit(X_train_gpu, Y_train_gpu)
            torch.cuda.synchronize()
            fit_gpu = min(fit_gpu, time.perf_counter() - t0)

        # GPU predict
        pred_gpu = float("inf")
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = m2.predict(X_val_gpu)
            torch.cuda.synchronize()
            pred_gpu = min(pred_gpu, time.perf_counter() - t0)

        results.append({
            "n_samples": n,
            "fit_cpu_ms": round(fit_cpu * 1000, 2),
            "fit_gpu_ms": round(fit_gpu * 1000, 2),
            "fit_gpu_speedup": round(fit_cpu / max(fit_gpu, 1e-9), 2),
            "predict_cpu_ms": round(pred_cpu * 1000, 2),
            "predict_gpu_ms": round(pred_gpu * 1000, 2),
            "predict_gpu_speedup": round(pred_cpu / max(pred_gpu, 1e-9), 2),
        })

        print(f"  N={n:6d}  fit: {fit_cpu*1000:8.1f}/{fit_gpu*1000:8.1f}ms ({fit_cpu/max(fit_gpu,1e-9):5.1f}x)  "
              f"pred: {pred_cpu*1000:6.1f}/{pred_gpu*1000:6.1f}ms ({pred_cpu/max(pred_gpu,1e-9):5.1f}x)")

    return results


# ===========================================================================
# 4. Multi-model scaling on larger datasets
# ===========================================================================

def benchmark_multimodel_scaling(sizes=None, n_features=10, n_classes=5):
    """All 4 models at larger scale (CPU only)."""
    if sizes is None:
        sizes = [200, 500, 1000, 2000]

    print(f"\n{'='*70}")
    print(f"  Multi-Model Scaling — All 4 classifiers on CPU")
    print(f"{'='*70}")

    model_configs = [
        ("SupervisedOPF", lambda: TorchSup(distance="log_squared_euclidean", device="cpu"), "simple"),
        ("KNNSupervisedOPF", lambda: TorchKNN(max_k=10, distance="log_squared_euclidean", device="cpu"), "knn"),
        ("SemiSupervisedOPF", lambda: TorchSemi(distance="log_squared_euclidean", device="cpu"), "semi"),
        ("UnsupervisedOPF", lambda: TorchUnsup(min_k=1, max_k=10, distance="log_squared_euclidean", device="cpu"), "unsup"),
    ]

    results = []

    for n in sizes:
        X_np, Y_np = _make_synthetic(n, n_features, n_classes)
        X_t = torch.tensor(X_np, dtype=torch.float32)
        Y_t = torch.tensor(Y_np, dtype=torch.int64)

        for model_name, model_factory, model_type in model_configs:
            try:
                if model_type == "simple":
                    ti, vi = _make_split(X_np, Y_np, 0.5, seed=1)
                    m = model_factory()
                    t0 = time.perf_counter()
                    m.fit(X_t[ti], Y_t[ti])
                    fit_time = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    preds = m.predict(X_t[vi])
                    pred_time = time.perf_counter() - t0

                elif model_type == "knn":
                    ti, vi = _make_split(X_np, Y_np, 0.8, seed=1)
                    sti, svi = _make_split(X_np[ti], Y_np[ti], 0.25, seed=2)
                    sub_train = ti[sti]; sub_val = ti[svi]
                    m = model_factory()
                    t0 = time.perf_counter()
                    m.fit(X_t[sub_train], Y_t[sub_train], X_t[sub_val], Y_t[sub_val])
                    fit_time = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    preds = m.predict(X_t[vi])
                    pred_time = time.perf_counter() - t0

                elif model_type == "semi":
                    ti, vi = _make_split(X_np, Y_np, 0.8, seed=1)
                    sti, sui = _make_split(X_np[ti], Y_np[ti], 0.25, seed=2)
                    labeled = ti[sti]; unlabeled = ti[sui]
                    m = model_factory()
                    t0 = time.perf_counter()
                    m.fit(X_t[labeled], Y_t[labeled], X_t[unlabeled])
                    fit_time = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    preds = m.predict(X_t[vi])
                    pred_time = time.perf_counter() - t0

                elif model_type == "unsup":
                    ti, vi = _make_split(X_np, Y_np, 0.5, seed=1)
                    m = model_factory()
                    t0 = time.perf_counter()
                    m.fit(X_t[ti], Y_t[ti])
                    m.propagate_labels()
                    fit_time = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    preds, _ = m.predict(X_t[vi])
                    pred_time = time.perf_counter() - t0

                results.append({
                    "model": model_name,
                    "n_samples": n,
                    "fit_ms": round(fit_time * 1000, 2),
                    "predict_ms": round(pred_time * 1000, 2),
                })

                print(f"  {model_name:22s}  N={n:6d}  fit={fit_time*1000:10.1f}ms  pred={pred_time*1000:8.1f}ms")

            except Exception as ex:
                print(f"  {model_name:22s}  N={n:6d}  ERROR: {ex}")
                results.append({
                    "model": model_name,
                    "n_samples": n,
                    "fit_ms": -1,
                    "predict_ms": -1,
                    "error": str(ex),
                })

    return results


# ===========================================================================
# 5. Dimensionality scaling
# ===========================================================================

def benchmark_dimension_scaling(dims=None, n_samples=1000, n_classes=5, repeats=2):
    """Measure impact of feature dimensionality on CPU performance."""
    if dims is None:
        dims = [5, 10, 25, 50, 100, 200]

    print(f"\n{'='*70}")
    print(f"  Dimensionality Scaling — SupervisedOPF, N={n_samples}")
    print(f"{'='*70}")

    results = []
    for d in dims:
        X_np, Y_np = _make_synthetic(n_samples, d, n_classes)
        X_t = torch.tensor(X_np, dtype=torch.float32)
        Y_t = torch.tensor(Y_np, dtype=torch.int64)
        ti, vi = _make_split(X_np, Y_np, 0.7, seed=1)

        # opfython
        fit_py = float("inf")
        for _ in range(repeats):
            m = PySup(distance="log_squared_euclidean")
            t0 = time.perf_counter()
            m.fit(X_np[ti], Y_np[ti])
            fit_py = min(fit_py, time.perf_counter() - t0)

        pred_py = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = m.predict(X_np[vi])
            pred_py = min(pred_py, time.perf_counter() - t0)

        # opforch CPU
        fit_t = float("inf")
        for _ in range(repeats):
            m2 = TorchSup(distance="log_squared_euclidean", device="cpu")
            t0 = time.perf_counter()
            m2.fit(X_t[ti], Y_t[ti])
            fit_t = min(fit_t, time.perf_counter() - t0)

        pred_t = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = m2.predict(X_t[vi])
            pred_t = min(pred_t, time.perf_counter() - t0)

        results.append({
            "n_features": d,
            "fit_opfython_ms": round(fit_py * 1000, 2),
            "fit_opforch_ms": round(fit_t * 1000, 2),
            "fit_speedup": round(fit_py / max(fit_t, 1e-9), 2),
            "predict_opfython_ms": round(pred_py * 1000, 2),
            "predict_opforch_ms": round(pred_t * 1000, 2),
            "predict_speedup": round(pred_py / max(pred_t, 1e-9), 2),
        })

        print(f"  D={d:4d}  fit: {fit_py*1000:8.1f}/{fit_t*1000:8.1f}ms ({fit_py/max(fit_t,1e-9):5.1f}x)  "
              f"pred: {pred_py*1000:6.1f}/{pred_t*1000:6.1f}ms ({pred_py/max(pred_t,1e-9):5.1f}x)")

    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    results = {}

    # 1. Extended scaling (CPU) — up to N=10000
    results["extended_scaling"] = benchmark_extended_scaling(
        sizes=[100, 200, 500, 1000, 2000, 3000, 5000, 8000, 10000]
    )

    # 2. Extended distance scaling
    results["extended_distance_scaling"] = benchmark_extended_distance_scaling(
        sizes=[50, 100, 200, 500, 1000, 2000]
    )

    # 3. GPU distance benchmark
    results["gpu_distance"] = benchmark_gpu_distance(
        sizes=[100, 500, 1000, 2000, 5000, 10000]
    )

    # 4. GPU model benchmark
    results["gpu_model"] = benchmark_gpu_model(
        sizes=[100, 500, 1000, 2000, 5000]
    )

    # 5. Multi-model scaling
    results["multimodel_scaling"] = benchmark_multimodel_scaling(
        sizes=[200, 500, 1000, 2000]
    )

    # 6. Dimensionality scaling
    results["dimension_scaling"] = benchmark_dimension_scaling(
        dims=[5, 10, 25, 50, 100, 200]
    )

    # Metadata
    results["meta"] = {
        "torch_version": torch.__version__,
        "cuda_available": HAS_CUDA,
        "gpu_name": GPU_NAME,
        "cpu_threads": torch.get_num_threads(),
    }

    out_path = out_dir / "extended_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Results saved to {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
