"""
Run a small hyperparameter grid (train + benchmark) for team meetings.

Usage (from repo root):
    python offline_learning/run_quick_experiments.py

Requires: same deps as train.py / benchmark.py (numpy, PyPokerEngine, matplotlib).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    out_dir = _ROOT / "offline_learning" / "models" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        {
            "id": "linear_lr001",
            "train": [
                "offline_learning/train.py",
                "--model",
                "linear",
                "--iterations",
                "2",
                "--games-per-iter",
                "3",
                "--rounds-per-game",
                "40",
                "--epochs",
                "5",
                "--lr",
                "0.01",
                "--output",
                str(out_dir / "exp_linear_lr001.json"),
            ],
            "bench": [
                "offline_learning/benchmark.py",
                "--model",
                str(out_dir / "exp_linear_lr001.json"),
                "--games",
                "6",
                "--rounds",
                "35",
                "--simulations",
                "40",
                "--save",
                str(out_dir / "bench_linear_lr001.json"),
                "--plot-save",
                str(out_dir / "bench_linear_lr001.png"),
            ],
        },
        {
            "id": "linear_lr0005",
            "train": [
                "offline_learning/train.py",
                "--model",
                "linear",
                "--iterations",
                "2",
                "--games-per-iter",
                "3",
                "--rounds-per-game",
                "40",
                "--epochs",
                "5",
                "--lr",
                "0.005",
                "--output",
                str(out_dir / "exp_linear_lr0005.json"),
            ],
            "bench": [
                "offline_learning/benchmark.py",
                "--model",
                str(out_dir / "exp_linear_lr0005.json"),
                "--games",
                "6",
                "--rounds",
                "35",
                "--simulations",
                "40",
                "--save",
                str(out_dir / "bench_linear_lr0005.json"),
                "--plot-save",
                str(out_dir / "bench_linear_lr0005.png"),
            ],
        },
        {
            "id": "mlp_lr0001",
            "train": [
                "offline_learning/train.py",
                "--model",
                "mlp",
                "--iterations",
                "2",
                "--games-per-iter",
                "3",
                "--rounds-per-game",
                "40",
                "--epochs",
                "8",
                "--lr",
                "0.001",
                "--output",
                str(out_dir / "exp_mlp_lr0001.json"),
            ],
            "bench": [
                "offline_learning/benchmark.py",
                "--model",
                str(out_dir / "exp_mlp_lr0001.json"),
                "--games",
                "6",
                "--rounds",
                "35",
                "--simulations",
                "40",
                "--save",
                str(out_dir / "bench_mlp_lr0001.json"),
                "--plot-save",
                str(out_dir / "bench_mlp_lr0001.png"),
            ],
        },
    ]

    for run in runs:
        print(f"\n=== {run['id']}: train ===", flush=True)
        r1 = subprocess.run([sys.executable] + run["train"], cwd=_ROOT)
        if r1.returncode != 0:
            return r1.returncode
        print(f"\n=== {run['id']}: benchmark ===", flush=True)
        r2 = subprocess.run([sys.executable] + run["bench"], cwd=_ROOT)
        if r2.returncode != 0:
            return r2.returncode

    print("\nDone. See offline_learning/EXPERIMENTS_LOG.md for how to interpret results.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
