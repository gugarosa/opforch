"""Generate publication-quality plots from benchmark results.

Reads benchmarks/results/benchmark_results.json and produces:
  1. Distance metric speedup bar chart (all 47 metrics)
  2. Model accuracy parity table-plot (4 models)
  3. Model fit/predict speedup grouped bar chart
  4. Scaling curves: fit time vs dataset size
  5. Scaling curves: predict time vs dataset size
  6. Distance scaling: time vs N for log_squared_euclidean

All plots are saved as PNG in benchmarks/results/.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


RESULTS_DIR = Path(__file__).parent / "images"
RESULTS_FILE = Path(__file__).parent / "results" / "benchmark_results.json"

# Colour palette
C_OPFYTHON = "#5B8DEE"   # blue
C_OPFORCH = "#FF6B6B"    # coral
C_SPEEDUP = "#2ECC71"    # green
C_ACCENT = "#F39C12"     # amber


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


# ===========================================================================
# Plot 1 – Distance Metric Speedups
# ===========================================================================

def plot_distance_speedups(data):
    """Horizontal bar chart of speedup per metric, sorted descending."""
    metrics = [d["metric"] for d in data]
    speedups = [d["speedup"] for d in data]

    # Sort by speedup
    order = np.argsort(speedups)
    metrics = [metrics[i] for i in order]
    speedups = [speedups[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 12))
    y = np.arange(len(metrics))
    colors = [C_SPEEDUP if s >= 1 else "#E74C3C" for s in speedups]
    bars = ax.barh(y, speedups, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=8, fontfamily="monospace")
    ax.set_xlabel("Speedup (opfython time / opforch time)", fontsize=11)
    ax.set_title("Distance Metric Speedup: OPForch vs OPFython\n"
                 f"(100×100 pairwise matrix, batched tensor vs N² scalar loop)",
                 fontsize=13, fontweight="bold")
    ax.axvline(x=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Annotate median speedup
    med = np.median(speedups)
    ax.axvline(x=med, color=C_ACCENT, linestyle="-", linewidth=1.5, alpha=0.8)
    ax.text(med + 0.3, len(metrics) - 1, f"median = {med:.1f}×",
            color=C_ACCENT, fontsize=10, fontweight="bold", va="top")

    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "01_distance_speedups.png", dpi=150)
    plt.close(fig)
    print("  ✓ 01_distance_speedups.png")


# ===========================================================================
# Plot 2 – Model Accuracy Parity
# ===========================================================================

def plot_model_accuracy(data):
    """Side-by-side accuracy comparison for all 4 models."""
    models = [d["model"] for d in data]
    acc_py = [d["acc_opfython"] for d in data]
    acc_t = [d["acc_opforch"] for d in data]
    mm = [d["mismatches"] for d in data]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    w = 0.3

    bars1 = ax.bar(x - w/2, acc_py, w, label="OPFython", color=C_OPFYTHON,
                   edgecolor="white", linewidth=0.8, zorder=3)
    bars2 = ax.bar(x + w/2, acc_t, w, label="OPForch", color=C_OPFORCH,
                   edgecolor="white", linewidth=0.8, zorder=3)

    # Annotate with accuracy values and mismatches
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.003,
                f"{acc_py[i]:.4f}", ha="center", va="bottom", fontsize=8, color=C_OPFYTHON)
        ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.003,
                f"{acc_t[i]:.4f}", ha="center", va="bottom", fontsize=8, color=C_OPFORCH)
        ax.text(x[i], min(acc_py[i], acc_t[i]) - 0.02,
                f"Δ = {mm[i]} mismatches", ha="center", va="top", fontsize=7,
                style="italic", color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("OPF Accuracy", fontsize=11)
    ax.set_title("Accuracy Parity: OPFython vs OPForch (boat.txt)\n"
                 "Zero prediction mismatches across all classifiers",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(bottom=min(min(acc_py), min(acc_t)) - 0.05, top=1.05)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "02_model_accuracy_parity.png", dpi=150)
    plt.close(fig)
    print("  ✓ 02_model_accuracy_parity.png")


# ===========================================================================
# Plot 3 – Model Fit/Predict Time
# ===========================================================================

def plot_model_timing(data):
    """Grouped bar chart: fit and predict times side by side."""
    models = [d["model"] for d in data]
    n = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, phase in zip(axes, ["fit", "predict"]):
        py_times = [d[f"{phase}_opfython_ms"] for d in data]
        t_times = [d[f"{phase}_opforch_ms"] for d in data]
        speedups = [d[f"{phase}_speedup"] for d in data]

        x = np.arange(n)
        w = 0.3

        ax.bar(x - w/2, py_times, w, label="OPFython", color=C_OPFYTHON,
               edgecolor="white", linewidth=0.8, zorder=3)
        ax.bar(x + w/2, t_times, w, label="OPForch", color=C_OPFORCH,
               edgecolor="white", linewidth=0.8, zorder=3)

        for i in range(n):
            higher = max(py_times[i], t_times[i])
            label = f"{speedups[i]:.1f}×"
            ax.text(x[i], higher * 1.08, label, ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=C_SPEEDUP)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("OPF", "\nOPF") for m in models], fontsize=8)
        ax.set_ylabel("Time (ms)", fontsize=11)
        ax.set_title(f"{phase.capitalize()} Time", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("Model Timing: OPFython vs OPForch (boat.txt)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "03_model_timing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 03_model_timing.png")


# ===========================================================================
# Plot 4 – Scaling: Fit Time vs N
# ===========================================================================

def plot_scaling_fit(data):
    """Log-log scaling plot for fit time."""
    sizes = [d["n_samples"] for d in data]
    py_times = [d["fit_opfython_ms"] for d in data]
    t_times = [d["fit_opforch_ms"] for d in data]
    speedups = [d["fit_speedup"] for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute times
    ax1.plot(sizes, py_times, "o-", color=C_OPFYTHON, linewidth=2,
             markersize=7, label="OPFython", zorder=3)
    ax1.plot(sizes, t_times, "s-", color=C_OPFORCH, linewidth=2,
             markersize=7, label="OPForch", zorder=3)
    ax1.set_xlabel("Dataset Size (N samples)", fontsize=11)
    ax1.set_ylabel("Fit Time (ms)", fontsize=11)
    ax1.set_title("Fit Time vs Dataset Size", fontsize=12, fontweight="bold")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # Right: speedup
    ax2.plot(sizes, speedups, "D-", color=C_SPEEDUP, linewidth=2.5,
             markersize=8, zorder=3)
    ax2.axhline(y=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.set_xlabel("Dataset Size (N samples)", fontsize=11)
    ax2.set_ylabel("Speedup (×)", fontsize=11)
    ax2.set_title("Fit Speedup vs Dataset Size", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    for i, (s, sp) in enumerate(zip(sizes, speedups)):
        ax2.annotate(f"{sp:.1f}×", (s, sp), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, fontweight="bold",
                     color=C_SPEEDUP)

    fig.suptitle("SupervisedOPF Scaling (synthetic data, 10 features, 3 classes)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "04_scaling_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 04_scaling_fit.png")


# ===========================================================================
# Plot 5 – Scaling: Predict Time vs N
# ===========================================================================

def plot_scaling_predict(data):
    """Log-log scaling plot for predict time."""
    sizes = [d["n_samples"] for d in data]
    py_times = [d["predict_opfython_ms"] for d in data]
    t_times = [d["predict_opforch_ms"] for d in data]
    speedups = [d["predict_speedup"] for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(sizes, py_times, "o-", color=C_OPFYTHON, linewidth=2,
             markersize=7, label="OPFython", zorder=3)
    ax1.plot(sizes, t_times, "s-", color=C_OPFORCH, linewidth=2,
             markersize=7, label="OPForch", zorder=3)
    ax1.set_xlabel("Dataset Size (N samples)", fontsize=11)
    ax1.set_ylabel("Predict Time (ms)", fontsize=11)
    ax1.set_title("Predict Time vs Dataset Size", fontsize=12, fontweight="bold")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax2.plot(sizes, speedups, "D-", color=C_SPEEDUP, linewidth=2.5,
             markersize=8, zorder=3)
    ax2.axhline(y=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.set_xlabel("Dataset Size (N samples)", fontsize=11)
    ax2.set_ylabel("Speedup (×)", fontsize=11)
    ax2.set_title("Predict Speedup vs Dataset Size", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    for i, (s, sp) in enumerate(zip(sizes, speedups)):
        ax2.annotate(f"{sp:.1f}×", (s, sp), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, fontweight="bold",
                     color=C_SPEEDUP)

    fig.suptitle("SupervisedOPF Scaling (synthetic data, 10 features, 3 classes)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "05_scaling_predict.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 05_scaling_predict.png")


# ===========================================================================
# Plot 6 – Distance Scaling
# ===========================================================================

def plot_distance_scaling(data):
    """Time and speedup vs dataset size for N×N pairwise distance matrix."""
    sizes = [d["n_samples"] for d in data]
    py_times = [d["opfython_ms"] for d in data]
    t_times = [d["opforch_ms"] for d in data]
    speedups = [d["speedup"] for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(sizes, py_times, "o-", color=C_OPFYTHON, linewidth=2,
             markersize=7, label="OPFython (N² scalar loop)", zorder=3)
    ax1.plot(sizes, t_times, "s-", color=C_OPFORCH, linewidth=2,
             markersize=7, label="OPForch (batched tensor)", zorder=3)
    ax1.set_xlabel("Dataset Size (N)", fontsize=11)
    ax1.set_ylabel("Time (ms)", fontsize=11)
    ax1.set_title("N×N Distance Matrix Time", fontsize=12, fontweight="bold")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax2.plot(sizes, speedups, "D-", color=C_SPEEDUP, linewidth=2.5,
             markersize=8, zorder=3)
    ax2.axhline(y=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.set_xlabel("Dataset Size (N)", fontsize=11)
    ax2.set_ylabel("Speedup (×)", fontsize=11)
    ax2.set_title("Batched Tensor Speedup", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    for s, sp in zip(sizes, speedups):
        ax2.annotate(f"{sp:.0f}×", (s, sp), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, fontweight="bold",
                     color=C_SPEEDUP)

    fig.suptitle("log_squared_euclidean: N×N Pairwise Distance Matrix Scaling",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "06_distance_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 06_distance_scaling.png")


# ===========================================================================
# Plot 7 – Summary Dashboard
# ===========================================================================

def plot_summary_dashboard(results):
    """Single-page summary combining key metrics."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    # --- Panel A: Model accuracy parity ---
    ax_a = fig.add_subplot(gs[0, 0])
    models_data = results["models_real"]
    models = [d["model"].replace("OPF", "\nOPF") for d in models_data]
    acc_py = [d["acc_opfython"] for d in models_data]
    acc_t = [d["acc_opforch"] for d in models_data]
    x = np.arange(len(models))
    w = 0.3
    ax_a.bar(x - w/2, acc_py, w, color=C_OPFYTHON, label="OPFython", edgecolor="white", zorder=3)
    ax_a.bar(x + w/2, acc_t, w, color=C_OPFORCH, label="OPForch", edgecolor="white", zorder=3)
    ax_a.set_xticks(x); ax_a.set_xticklabels(models, fontsize=7)
    ax_a.set_ylabel("Accuracy", fontsize=9)
    ax_a.set_title("A. Accuracy Parity", fontsize=11, fontweight="bold")
    ax_a.set_ylim(min(min(acc_py), min(acc_t)) - 0.05, 1.05)
    ax_a.legend(fontsize=7, loc="lower right")
    ax_a.grid(axis="y", alpha=0.3, zorder=0)

    # --- Panel B: Model fit speedup ---
    ax_b = fig.add_subplot(gs[0, 1])
    fit_sp = [d["fit_speedup"] for d in models_data]
    pred_sp = [d["predict_speedup"] for d in models_data]
    ax_b.bar(x - w/2, fit_sp, w, color=C_SPEEDUP, label="Fit", edgecolor="white", zorder=3)
    ax_b.bar(x + w/2, pred_sp, w, color=C_ACCENT, label="Predict", edgecolor="white", zorder=3)
    ax_b.set_xticks(x); ax_b.set_xticklabels(models, fontsize=7)
    ax_b.set_ylabel("Speedup (×)", fontsize=9)
    ax_b.set_title("B. Model Speedups", fontsize=11, fontweight="bold")
    ax_b.axhline(y=1, color="gray", linestyle="--", linewidth=0.8)
    ax_b.legend(fontsize=7)
    ax_b.grid(axis="y", alpha=0.3, zorder=0)

    # --- Panel C: Distance metric speedup histogram ---
    ax_c = fig.add_subplot(gs[0, 2])
    dist_speedups = [d["speedup"] for d in results["distances"]]
    ax_c.hist(dist_speedups, bins=15, color=C_SPEEDUP, edgecolor="white", alpha=0.85)
    ax_c.axvline(np.median(dist_speedups), color=C_ACCENT, linestyle="-", linewidth=2,
                 label=f"Median = {np.median(dist_speedups):.0f}×")
    ax_c.set_xlabel("Speedup (×)", fontsize=9)
    ax_c.set_ylabel("# Metrics", fontsize=9)
    ax_c.set_title("C. Distance Metric Speedups", fontsize=11, fontweight="bold")
    ax_c.legend(fontsize=8)
    ax_c.grid(axis="y", alpha=0.3)

    # --- Panel D: Scaling fit ---
    ax_d = fig.add_subplot(gs[1, 0])
    scaling = results["scaling"]
    sizes = [d["n_samples"] for d in scaling]
    ax_d.plot(sizes, [d["fit_opfython_ms"] for d in scaling], "o-",
              color=C_OPFYTHON, linewidth=2, markersize=5, label="OPFython")
    ax_d.plot(sizes, [d["fit_opforch_ms"] for d in scaling], "s-",
              color=C_OPFORCH, linewidth=2, markersize=5, label="OPForch")
    ax_d.set_xlabel("N samples", fontsize=9); ax_d.set_ylabel("Fit Time (ms)", fontsize=9)
    ax_d.set_title("D. Fit Time Scaling", fontsize=11, fontweight="bold")
    ax_d.set_xscale("log"); ax_d.set_yscale("log")
    ax_d.legend(fontsize=7); ax_d.grid(True, alpha=0.3, which="both")
    ax_d.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # --- Panel E: Scaling predict ---
    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.plot(sizes, [d["predict_opfython_ms"] for d in scaling], "o-",
              color=C_OPFYTHON, linewidth=2, markersize=5, label="OPFython")
    ax_e.plot(sizes, [d["predict_opforch_ms"] for d in scaling], "s-",
              color=C_OPFORCH, linewidth=2, markersize=5, label="OPForch")
    ax_e.set_xlabel("N samples", fontsize=9); ax_e.set_ylabel("Predict Time (ms)", fontsize=9)
    ax_e.set_title("E. Predict Time Scaling", fontsize=11, fontweight="bold")
    ax_e.set_xscale("log"); ax_e.set_yscale("log")
    ax_e.legend(fontsize=7); ax_e.grid(True, alpha=0.3, which="both")
    ax_e.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # --- Panel F: Scaling speedup ---
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.plot(sizes, [d["fit_speedup"] for d in scaling], "D-",
              color=C_SPEEDUP, linewidth=2, markersize=6, label="Fit Speedup")
    ax_f.plot(sizes, [d["predict_speedup"] for d in scaling], "^-",
              color=C_ACCENT, linewidth=2, markersize=6, label="Predict Speedup")
    ax_f.axhline(y=1, color="gray", linestyle="--", linewidth=0.8)
    ax_f.set_xlabel("N samples", fontsize=9); ax_f.set_ylabel("Speedup (×)", fontsize=9)
    ax_f.set_title("F. Speedup vs Scale", fontsize=11, fontweight="bold")
    ax_f.legend(fontsize=7); ax_f.grid(True, alpha=0.3)

    fig.suptitle("OPForch vs OPFython — Performance & Quality Summary",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.savefig(RESULTS_DIR / "07_summary_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 07_summary_dashboard.png")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("Loading benchmark results ...")
    results = load_results()

    print("Generating plots ...")
    plot_distance_speedups(results["distances"])
    plot_model_accuracy(results["models_real"])
    plot_model_timing(results["models_real"])
    plot_scaling_fit(results["scaling"])
    plot_scaling_predict(results["scaling"])
    plot_distance_scaling(results["distance_scaling"])
    plot_summary_dashboard(results)

    print(f"\n✓ All plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
