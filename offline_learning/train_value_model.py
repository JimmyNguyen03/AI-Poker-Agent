import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

# Allow `python offline_learning/train_value_model.py` from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline_learning.value_model import LinearValueModel


def _load_records(path: str) -> List[Tuple[List[float], float]]:
    rows: List[Tuple[List[float], float]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        rows.append((list(item["features"]), float(item["terminal_reward"])))
    return rows


def train_value_model(
    dataset_path: str,
    output_model_path: str,
    loss_history_path: str,
    loss_plot_path: str,
    epochs: int = 5,
    lr: float = 0.01,
) -> Dict[str, float]:
    rows = _load_records(dataset_path)
    if not rows:
        raise ValueError("Dataset is empty. Generate self-play data first.")
    model = LinearValueModel.zeros(input_dim=len(rows[0][0]))
    loss_history = model.train(rows, epochs=epochs, lr=lr)
    model.save(output_model_path)
    loss_payload = {
        "epochs": len(loss_history),
        "learning_rate": lr,
        "loss_history": loss_history,
    }
    loss_target = Path(loss_history_path)
    loss_target.parent.mkdir(parents=True, exist_ok=True)
    loss_target.write_text(json.dumps(loss_payload, indent=2), encoding="utf-8")
    _plot_loss_curve(loss_history, loss_plot_path)
    return {"final_loss": float(loss_history[-1]), "epochs": float(len(loss_history))}


def _plot_loss_curve(loss_history: List[float], loss_plot_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped loss plot generation.")
        return

    plot_target = Path(loss_plot_path)
    plot_target.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(loss_history) + 1))
    fig = plt.figure(figsize=(6, 4))
    plt.plot(epochs, loss_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(plot_target)
    plt.close(fig)


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="offline_learning/data/self_play.jsonl")
    parser.add_argument("--out-model", type=str, default="offline_learning/models/value_model.json")
    parser.add_argument("--out-loss-history", type=str, default="offline_learning/models/loss_history.json")
    parser.add_argument("--out-loss-plot", type=str, default="offline_learning/models/loss_curve.png")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
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
    )
    print("Saved model:", args.out_model)
    print("Saved loss history:", args.out_loss_history)
    print("Saved loss plot:", args.out_loss_plot)
    print("Final training loss:", summary["final_loss"])

