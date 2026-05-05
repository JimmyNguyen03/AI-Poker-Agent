"""
Minimal inference-only value model loader for the tournament runtime.
Supports the linear model only (the best-performing checkpoint).
Kept in mcts/ to avoid any dependency on the offline_learning training package.
"""

from __future__ import annotations
import json
from pathlib import Path


class LinearValueModel:
    """Dot-product value estimator over state features."""

    def __init__(self, weights: dict):
        self.weights = weights

    def predict(self, features: dict) -> float:
        s = sum(self.weights.get(k, 0.0) * v for k, v in features.items())
        s += self.weights.get("bias", 0.0)
        return max(-1.0, min(1.0, s))


def load_value_model(path: str):
    """Load a value model checkpoint. Returns a LinearValueModel for linear checkpoints."""
    with open(path, "r") as f:
        data = json.load(f)
    model_type = data.get("model_type", "linear")
    if model_type == "linear":
        weights = {k: float(v) for k, v in data.items() if k != "model_type"}
        return LinearValueModel(weights)
    raise ValueError(
        f"Model type '{model_type}' is not supported in the tournament runtime. "
        "Re-save the checkpoint as a linear model."
    )
