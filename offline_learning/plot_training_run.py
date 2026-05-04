"""
Plot loss curves and baseline evaluation metrics from an iterative RL training run.

Reads:
  - metrics/iterative_rl_summary.json
  - models/loss_history_iter_*.json

Default run root is offline_learning/; use --run-name or --run-dir for self_play.py run folders.

Usage (from repo root AI-Poker-Agent):
  python offline_learning/plot_training_run.py
  python offline_learning/plot_training_run.py --run-name exp_alpha
  python offline_learning/plot_training_run.py --run-dir offline_learning/runs/exp_alpha
  python offline_learning/plot_training_run.py --summary path/to/iterative_rl_summary.json --models-dir path/to/models
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from offline_learning.self_play import resolve_run_directory


def _offline_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_run_root(args: argparse.Namespace) -> Path:
    """Match self_play.py: --run-dir wins; else --run-name -> offline_learning/runs/<name>; else offline_learning/."""
    ol = _offline_root()
    if getattr(args, "run_dir", "").strip():
        return Path(args.run_dir).expanduser().resolve()
    if getattr(args, "run_name", "").strip():
        return resolve_run_directory(str(ol), args.run_name)
    return ol.resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_loss_iter(name: str) -> Optional[int]:
    m = re.match(r"loss_history_iter_(\d+)\.json$", name, re.I)
    return int(m.group(1)) if m else None


def _collect_loss_series(models_dir: Path) -> List[Tuple[int, List[float]]]:
    rows: List[Tuple[int, List[float]]] = []
    if not models_dir.is_dir():
        return rows
    for p in sorted(models_dir.glob("loss_history_iter_*.json")):
        it = _parse_loss_iter(p.name)
        if it is None:
            continue
        data = _load_json(p)
        hist = data.get("loss_history")
        if isinstance(hist, list) and hist:
            rows.append((it, [float(x) for x in hist]))
    rows.sort(key=lambda x: x[0])
    return rows


def _baseline_names(history: List[Dict[str, Any]]) -> List[str]:
    names: set[str] = set()
    for h in history:
        be = h.get("baseline_eval") or {}
        if isinstance(be, dict):
            names.update(be.keys())
    return sorted(names)


def _plot(
    summary_path: Path,
    models_dir: Path,
    out_path: Path,
    show: bool,
    run_root: Optional[Path] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    summary = _load_json(summary_path)
    history_raw = summary.get("history") or []
    history: List[Dict[str, Any]] = []
    for h in history_raw:
        if isinstance(h, dict):
            history.append(h)

    loss_series = _collect_loss_series(models_dir)
    baseline_keys = _baseline_names(history)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    title = summary.get("run_directory") or (str(run_root) if run_root is not None else str(summary_path.parent))
    fig.suptitle(f"{Path(title).name} · {summary_path.name}", fontsize=12)

    # (0,0) Per-iteration loss vs epoch
    ax0 = axes[0, 0]
    for it, losses in loss_series:
        epochs = list(range(1, len(losses) + 1))
        ax0.plot(epochs, losses, marker="o", linewidth=1.5, label=f"iter {it}")
    ax0.set_xlabel("Epoch (within iteration)")
    ax0.set_ylabel("MSE loss")
    ax0.set_title("Value fit: loss per epoch")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", fontsize=8)

    # (0,1) Final loss per iteration (from summary)
    ax1 = axes[0, 1]
    iters: List[int] = []
    finals: List[float] = []
    for h in history:
        tr = h.get("train") or {}
        if "final_loss" in tr and "iteration" in h:
            iters.append(int(float(h["iteration"])))
            finals.append(float(tr["final_loss"]))
    if iters:
        ax1.bar([str(i) for i in iters], finals, color="steelblue", edgecolor="black", linewidth=0.5)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Final MSE")
        ax1.set_title("Final training loss by iteration")
        ax1.grid(True, axis="y", alpha=0.3)

    # (1,0) Baseline: avg stack delta (hero - opp), higher is better for hero
    ax2 = axes[1, 0]
    for key in baseline_keys:
        ys: List[float] = []
        xs: List[int] = []
        for h in history:
            it = int(float(h["iteration"]))
            be = h.get("baseline_eval") or {}
            if not isinstance(be, dict) or key not in be:
                continue
            row = be[key]
            if isinstance(row, dict) and "avg_stack_delta" in row:
                xs.append(it)
                ys.append(float(row["avg_stack_delta"]))
        if xs:
            ax2.plot(xs, ys, marker="s", linewidth=1.5, label=key)
    ax2.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Avg stack delta (hero − opp)")
    ax2.set_title("Baseline eval: stack advantage")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=8)

    # (1,1) Baseline: hero win rate
    ax3 = axes[1, 1]
    for key in baseline_keys:
        ys: List[float] = []
        xs: List[int] = []
        for h in history:
            it = int(float(h["iteration"]))
            be = h.get("baseline_eval") or {}
            if not isinstance(be, dict) or key not in be:
                continue
            row = be[key]
            if isinstance(row, dict) and "hero_win_rate" in row:
                xs.append(it)
                ys.append(float(row["hero_win_rate"]))
        if xs:
            ax3.plot(xs, ys, marker="^", linewidth=1.5, label=key)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Hero win rate")
    ax3.set_title("Baseline eval: win rate vs baseline")
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training loss and baseline metrics.")
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Use offline_learning/runs/<run-name>/ (same as self_play.py --run-name).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Explicit run root containing metrics/ and models/ (overrides --run-name).",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="",
        help="iterative_rl_summary.json path; default <run-root>/metrics/iterative_rl_summary.json",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="",
        help="Directory with loss_history_iter_*.json; default <run-root>/models",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output PNG; default <run-root>/metrics/training_run_plots.png",
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive window")
    args = parser.parse_args()

    run_root = _resolve_run_root(args)
    summary_path = (
        Path(args.summary).expanduser().resolve()
        if args.summary.strip()
        else (run_root / "metrics" / "iterative_rl_summary.json").resolve()
    )
    models_dir = (
        Path(args.models_dir).expanduser().resolve()
        if args.models_dir.strip()
        else (run_root / "models").resolve()
    )
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out.strip()
        else (run_root / "metrics" / "training_run_plots.png").resolve()
    )

    if not summary_path.is_file():
        print(f"Summary not found: {summary_path}", file=sys.stderr)
        print(f"(run root resolved to: {run_root})", file=sys.stderr)
        sys.exit(1)

    _plot(
        summary_path=summary_path,
        models_dir=models_dir,
        out_path=out_path,
        show=args.show,
        run_root=run_root,
    )


if __name__ == "__main__":
    main()
