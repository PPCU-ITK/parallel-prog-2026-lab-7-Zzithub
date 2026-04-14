#!/usr/bin/env python3
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cfd-euler")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CSV = ROOT / "cfd_euler_runtime_results.csv"


def load_rows():
    rows = []
    with CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            row["Nx"] = int(row["Nx"])
            row["Ny"] = int(row["Ny"])
            row["nSteps"] = int(row["nSteps"])
            row["runtime_seconds"] = float(row["runtime_seconds"])
            row["total_kinetic"] = float(row["total_kinetic"])
            row["scale"] = row["Nx"] // 200
            rows.append(row)
    return rows


def main():
    rows = load_rows()
    scales = sorted({row["scale"] for row in rows})
    by_backend = {backend: [] for backend in ("cpu", "gpu")}
    for backend in by_backend:
        for scale in scales:
            match = next(row for row in rows if row["backend"] == backend and row["scale"] == scale)
            by_backend[backend].append(match["runtime_seconds"])

    labels = [f"{scale}x\n{200 * scale}x{100 * scale}" for scale in scales]
    x = range(len(scales))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], by_backend["cpu"], width, label="CPU OpenMP")
    ax.bar([i + width / 2 for i in x], by_backend["gpu"], width, label="GPU OpenMP offload")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("CFD Euler OpenMP CPU vs GPU Runtime")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "runtime_comparison.png", dpi=180)

    speedups = [cpu / gpu for cpu, gpu in zip(by_backend["cpu"], by_backend["gpu"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(scales, speedups, marker="o", linewidth=2)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(scales)
    ax.set_xlabel("Grid scale factor")
    ax.set_ylabel("Speedup (CPU runtime / GPU runtime)")
    ax.set_title("CFD Euler GPU Speedup")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "gpu_speedup.png", dpi=180)



if __name__ == "__main__":
    main()
