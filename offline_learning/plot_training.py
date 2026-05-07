"""
Plot MSE training curves from a saved training_log.json.

Usage (from repo root):
    python offline_learning/plot_training.py
    python offline_learning/plot_training.py --log offline_learning/models/mlp/training_log.json
    python offline_learning/plot_training.py --save offline_learning/models/training_curve.png
    python offline_learning/plot_training.py --all   # plots every log found in models/
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def plot_training(log_path: str | Path, save_path: str | Path | None = None) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    log_path = Path(log_path)
    if not log_path.exists():
        print(f"[plot_training] No log found at {log_path}. Run train.py first.")
        return

    with open(log_path) as f:
        log = json.load(f)

    cfg = log.get("config", {})
    iterations_data = log.get("iterations", [])

    # Build flat x-axis (cumulative epoch index) and y values
    mse_vals: list[float] = []
    std_vals: list[float] = []
    x_vals: list[int] = []
    iter_boundaries: list[int] = []  # x positions where a new iteration starts

    cumulative_epoch = 0
    for iter_entry in iterations_data:
        iter_boundaries.append(cumulative_epoch)
        for m in iter_entry.get("epoch_metrics", []):
            mse_vals.append(m["mse"])
            std_vals.append(m["std"])
            x_vals.append(cumulative_epoch)
            cumulative_epoch += 1

    if not x_vals:
        print("[plot_training] Log contains no epoch data.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Iterative Self-Play Training", fontsize=14, fontweight="bold")

    # --- Left: MSE curve with shaded std band ---
    ax = axes[0]
    mse_arr = mse_vals
    std_arr = std_vals
    lower = [max(0.0, m - s) for m, s in zip(mse_arr, std_arr)]
    upper = [m + s for m, s in zip(mse_arr, std_arr)]

    ax.plot(x_vals, mse_arr, color="steelblue", linewidth=2, label="MSE (mean)")
    ax.fill_between(x_vals, lower, upper, alpha=0.25, color="steelblue", label="MSE +/- std")

    # Mark iteration boundaries
    for i, boundary in enumerate(iter_boundaries):
        ax.axvline(x=boundary, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.text(boundary + 0.1, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.95,
                f"iter {i+1}", fontsize=8, color="gray", va="top")

    ax.set_xlabel("Cumulative Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Training Loss (SGD on Self-Play Data)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Right: examples collected per iteration ---
    ax2 = axes[1]
    iter_nums = [e["iteration"] for e in iterations_data]
    n_examples = [e["n_examples"] for e in iterations_data]
    bars = ax2.bar(iter_nums, n_examples, color="steelblue", alpha=0.75, edgecolor="white")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Training Examples Collected")
    ax2.set_title("Self-Play Data Volume per Iteration")
    ax2.set_xticks(iter_nums)
    for bar, n in zip(bars, n_examples):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(n), ha="center", va="bottom", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    # Config annotation
    config_str = (
        f"model={cfg.get('model_type','-')}  "
        f"target={cfg.get('target_mode','-')}  "
        f"iters={cfg.get('iterations','-')}  "
        f"games/iter={cfg.get('games_per_iter','-')}  "
        f"rounds/game={cfg.get('rounds_per_game','-')}  "
        f"epochs/iter={cfg.get('epochs_per_iter','-')}  "
        f"lr={cfg.get('lr','-')}"
    )
    fig.text(0.5, 0.01, config_str, ha="center", fontsize=8, color="gray")

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    else:
        plt.show()


# ------------------------------------------------------------------
# --all helpers
# ------------------------------------------------------------------

def _find_all_logs(models_dir: Path) -> list[tuple[str, Path]]:
    """
    Scan models_dir for training_log.json files up to two levels deep.
    Handles both models/<model>/training_log.json and
    models/<model>/<target>/training_log.json layouts.
    """
    found: list[tuple[str, Path]] = []
    root_log = models_dir / "training_log.json"
    if root_log.exists():
        found.append(("root", root_log))
    for subdir in sorted(models_dir.iterdir()):
        if not subdir.is_dir():
            continue
        candidate = subdir / "training_log.json"
        if candidate.exists():
            found.append((subdir.name, candidate))
        else:
            for subsubdir in sorted(subdir.iterdir()):
                if subsubdir.is_dir():
                    candidate2 = subsubdir / "training_log.json"
                    if candidate2.exists():
                        found.append((f"{subdir.name}/{subsubdir.name}", candidate2))
    return found


def plot_all_training(models_dir: Path | None = None) -> None:
    """
    Discover every training_log.json under models_dir, save an individual
    curve for each, then produce a combined MSE overlay for comparison.
    """
    if models_dir is None:
        models_dir = _ROOT / "offline_learning" / "models"
    models_dir = Path(models_dir)

    entries = _find_all_logs(models_dir)
    if not entries:
        print(f"No training logs found under {models_dir}. Train first.")
        return

    print(f"Found {len(entries)} log(s): {[label for label, _ in entries]}")

    # Individual plots
    for label, log_path in entries:
        save = log_path.parent / "training_curve.png"
        print(f"  [{label}] plotting -> {save}")
        plot_training(log_path, save)

    # Combined MSE overlay
    _plot_mse_overlay(entries, models_dir / "training_comparison.png")


def _plot_mse_overlay(entries: list[tuple[str, Path]], save_path: Path) -> None:
    """Single axes with one MSE curve per model, coloured by label."""
    import json as _json
    import matplotlib.pyplot as plt

    palette = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Training MSE — All Models", fontsize=13, fontweight="bold")

    for i, (label, log_path) in enumerate(entries):
        with open(log_path) as f:
            log = _json.load(f)
        mse_vals, x_vals = [], []
        epoch = 0
        for iter_entry in log.get("iterations", []):
            for m in iter_entry.get("epoch_metrics", []):
                mse_vals.append(m["mse"])
                x_vals.append(epoch)
                epoch += 1
        if x_vals:
            ax.plot(x_vals, mse_vals, label=label, color=palette[i % 10], linewidth=2)

    ax.set_xlabel("Cumulative Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved -> {save_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot training curves from training_log.json")
    parser.add_argument(
        "--log",
        type=str,
        default=str(_ROOT / "offline_learning" / "models" / "training_log.json"),
        help="Path to a specific training_log.json",
    )
    parser.add_argument("--save", type=str, default=None, help="Save to file instead of showing")
    parser.add_argument("--all", action="store_true",
                        help="Plot every training_log.json found in models/ and save a comparison")
    args = parser.parse_args()

    if args.all:
        plot_all_training()
    elif Path(args.log).exists():
        plot_training(args.log, args.save)
    else:
        # Root log not found — fall back to plotting all discovered logs.
        print(f"[plot_training] No log at {args.log} — scanning for all logs.")
        plot_all_training()
