import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple, Union

# Support script execution from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline_learning.data import TARGET_SCHEMA_VERSION
from offline_learning.mlp_value_model import MLPValueModel
from offline_learning.value_model import LinearValueModel, load_value_model


def _model_for_training(
    input_dim: int,
    warm_start_path: str,
    model_kind: str = "linear",
    hidden_dim: int = 32,
) -> Tuple[Union[LinearValueModel, MLPValueModel], bool, bool]:
    """
    Build initial weights: load checkpoint if path, kind, and dimensions match; else fresh model.
    Returns (model, warm_started, dim_mismatch_after_load_attempt).
    """
    model_kind = (model_kind or "linear").lower()
    if not warm_start_path.strip():
        if model_kind == "mlp":
            return MLPValueModel.zeros(input_dim, hidden_dim=hidden_dim), False, False
        return LinearValueModel.zeros(input_dim), False, False
    path = Path(warm_start_path)
    if not path.is_file():
        if model_kind == "mlp":
            return MLPValueModel.zeros(input_dim, hidden_dim=hidden_dim), False, False
        return LinearValueModel.zeros(input_dim), False, False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        file_kind = str(payload.get("kind", "linear")).lower()
    except (json.JSONDecodeError, OSError, UnicodeError):
        if model_kind == "mlp":
            return MLPValueModel.zeros(input_dim, hidden_dim=hidden_dim), False, False
        return LinearValueModel.zeros(input_dim), False, False

    if file_kind != model_kind:
        if model_kind == "mlp":
            return MLPValueModel.zeros(input_dim, hidden_dim=hidden_dim), False, True
        return LinearValueModel.zeros(input_dim), False, True

    try:
        loaded = load_value_model(str(path))
    except (KeyError, ValueError, TypeError):
        if model_kind == "mlp":
            return MLPValueModel.zeros(input_dim, hidden_dim=hidden_dim), False, False
        return LinearValueModel.zeros(input_dim), False, False

    if model_kind == "mlp":
        m = loaded
        if not isinstance(m, MLPValueModel) or m.input_dim != input_dim or m.hidden_dim != hidden_dim:
            return MLPValueModel.zeros(input_dim, hidden_dim=hidden_dim), False, True
        return m, True, False

    if not isinstance(loaded, LinearValueModel) or len(loaded.weights) != input_dim:
        return LinearValueModel.zeros(input_dim), False, True
    return loaded, True, False


def _load_records(
    path: str,
    regression_target: str = "chips",
) -> Tuple[List[Tuple[List[float], float]], Dict[str, int]]:
    rows: List[Tuple[List[float], float]] = []
    skipped_legacy = 0
    skipped_missing_rollout = 0
    mode = (regression_target or "chips").lower()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        if int(item.get("target_schema", 1)) != TARGET_SCHEMA_VERSION:
            skipped_legacy += 1
            continue
        if mode == "rollout":
            if "rollout_aligned_target" not in item:
                skipped_missing_rollout += 1
                continue
            y = float(item["rollout_aligned_target"])
        else:
            y = float(item.get("training_target", item.get("terminal_reward", 0.0)))
        rows.append((list(item["features"]), y))
    stats = {
        "skipped_legacy_schema": skipped_legacy,
        "skipped_missing_rollout": skipped_missing_rollout,
    }
    return rows, stats


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
    model_kind: str = "linear",
    hidden_dim: int = 32,
    regression_target: str = "chips",
) -> Dict[str, float]:
    rows, load_stats = _load_records(dataset_path, regression_target=regression_target)
    if not rows:
        raise ValueError(
            "Dataset is empty after filtering (wrong target_schema, missing columns, or no file). "
            f"skipped_legacy_schema={load_stats['skipped_legacy_schema']} "
            f"skipped_missing_rollout={load_stats['skipped_missing_rollout']} "
            f"regression_target={regression_target!r} path={dataset_path!r}"
        )
    input_dim = len(rows[0][0])
    model, warm_started, dim_mismatch = _model_for_training(
        input_dim, warm_start_path, model_kind=model_kind, hidden_dim=hidden_dim
    )
    loss_history = model.train(rows, epochs=epochs, lr=lr)
    model.save(output_model_path)

    history_payload: Dict = {
        "epochs": len(loss_history),
        "learning_rate": lr,
        "loss_history": loss_history,
        "warm_started": warm_started,
        "warm_start_path": warm_start_path or "",
        "warm_start_dim_mismatch": dim_mismatch,
        "model_kind": (model_kind or "linear").lower(),
        "regression_target": (regression_target or "chips").lower(),
        "load_stats": load_stats,
        "rows_trained": len(rows),
    }
    if (model_kind or "linear").lower() == "mlp":
        history_payload["hidden_dim"] = hidden_dim
    history_target = Path(loss_history_path)
    history_target.parent.mkdir(parents=True, exist_ok=True)
    history_target.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    _plot_loss_curve(loss_history, loss_plot_path)
    return {
        "final_loss": float(loss_history[-1]),
        "epochs": float(len(loss_history)),
        "warm_started": 1.0 if warm_started else 0.0,
        "warm_start_dim_mismatch": 1.0 if dim_mismatch else 0.0,
        "value_arch": (model_kind or "linear").lower(),
        "rows_trained": float(len(rows)),
        "skipped_legacy_schema": float(load_stats["skipped_legacy_schema"]),
        "skipped_missing_rollout": float(load_stats["skipped_missing_rollout"]),
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
    parser.add_argument(
        "--model",
        type=str,
        choices=("linear", "mlp"),
        default="linear",
        help="Value function architecture: linear regressor or small MLP (one hidden layer).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="Hidden units for --model mlp.",
    )
    parser.add_argument(
        "--regression-target",
        type=str,
        choices=("chips", "rollout"),
        default="chips",
        help="Label for MSE: per-decision chips (training_target) or MCTS leaf heuristic (rollout_aligned_target).",
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
        model_kind=args.model,
        hidden_dim=args.hidden_dim,
        regression_target=args.regression_target,
    )
    print("Saved model:", args.out_model)
    print("Saved loss history:", args.out_loss_history)
    print("Saved loss plot:", args.out_loss_plot)
    print("Final training loss:", summary["final_loss"])

