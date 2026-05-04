import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

# Support script execution from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline_learning.value_model import LinearValueModel


def _model_for_training(input_dim: int, warm_start_path: str) -> Tuple[LinearValueModel, bool, bool]:
    """
    Build initial weights: load checkpoint if path exists and feature dim matches, else zeros.
    Returns (model, warm_started, dim_mismatch_after_load_attempt).
    """
    if not warm_start_path.strip():
        return LinearValueModel.zeros(input_dim), False, False
    path = Path(warm_start_path)
    if not path.is_file():
        return LinearValueModel.zeros(input_dim), False, False
    try:
        loaded = LinearValueModel.load(str(path))
    except (json.JSONDecodeError, KeyError, OSError, ValueError):
        return LinearValueModel.zeros(input_dim), False, False
    if len(loaded.weights) != input_dim:
        return LinearValueModel.zeros(input_dim), False, True
    return loaded, True, False


def _load_records(path: str) -> List[Tuple[List[float], float]]:
    rows: List[Tuple[List[float], float]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        rows.append((list(item["features"]), float(item["terminal_reward"])))
    return rows


def _plot_loss_curve(loss_history: List[float], loss_plot_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped loss plot generation.")
        return

    target = Path(loss_plot_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(loss_history) + 1))
    fig = plt.figure(figsize=(6, 4))
    plt.plot(epochs, loss_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(target)
    plt.close(fig)


def train_value_model(
    dataset_path: str,
    output_model_path: str,
    loss_history_path: str,
    loss_plot_path: str,
    epochs: int = 5,
    lr: float = 0.01,
    warm_start_path: str = "",
) -> Dict[str, float]:
    rows = _load_records(dataset_path)
    if not rows:
        raise ValueError("Dataset is empty. Generate self-play data first.")
    input_dim = len(rows[0][0])
    model, warm_started, dim_mismatch = _model_for_training(input_dim, warm_start_path)
    loss_history = model.train(rows, epochs=epochs, lr=lr)
    model.save(output_model_path)

    history_payload = {
        "epochs": len(loss_history),
        "learning_rate": lr,
        "loss_history": loss_history,
        "warm_started": warm_started,
        "warm_start_path": warm_start_path or "",
        "warm_start_dim_mismatch": dim_mismatch,
    }
    history_target = Path(loss_history_path)
    history_target.parent.mkdir(parents=True, exist_ok=True)
    history_target.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    _plot_loss_curve(loss_history, loss_plot_path)
    return {
        "final_loss": float(loss_history[-1]),
        "epochs": float(len(loss_history)),
        "warm_started": 1.0 if warm_started else 0.0,
        "warm_start_dim_mismatch": 1.0 if dim_mismatch else 0.0,
    }


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="offline_learning/data/self_play.jsonl")
    parser.add_argument("--out-model", type=str, default="offline_learning/models/value_model.json")
    parser.add_argument("--out-loss-history", type=str, default="offline_learning/models/loss_history.json")
    parser.add_argument("--out-loss-plot", type=str, default="offline_learning/models/loss_curve.png")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--warm-start",
        type=str,
        default="",
        help="Optional path to value_model.json to initialize weights (same feature dim as dataset).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = train_value_model(
        dataset_path=args.dataset,
        output_model_path=args.out_model,
        loss_history_path=args.out_loss_history,
        loss_plot_path=args.out_loss_plot,
        epochs=args.epochs,
        lr=args.lr,
        warm_start_path=args.warm_start,
    )
    print("Saved model:", args.out_model)
    print("Saved loss history:", args.out_loss_history)
    print("Saved loss plot:", args.out_loss_plot)
    print("Final training loss:", summary["final_loss"])

