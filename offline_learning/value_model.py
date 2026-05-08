
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Optional

FEATURE_KEYS: tuple[str, ...] = (
    "win_rate", "hand_strength_norm", "hand_group_norm",
    "pot_odds", "call_price_stack_fraction", "raise_price_stack_fraction",
    "stack_advantage", "hero_stack_ratio",
    "street_progress", "street_is_preflop", "street_is_flop",
    "street_is_turn", "street_is_river",
    "opp_raise_rate", "opp_fold_rate", "opp_call_rate", "opp_aggression",
    "hero_is_next",
)


def features_to_vector(features: dict[str, float]) -> np.ndarray:
    """Convert a feature dict to a fixed-order numpy vector."""
    return np.array([features.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float64)


DEFAULT_WEIGHTS: dict[str, float] = {
    "win_rate": 1.60,
    "hand_strength_norm": 0.45,
    "hand_group_norm": 0.35,
    "pot_odds": 0.40,
    "call_price_stack_fraction": -0.55,
    "raise_price_stack_fraction": -0.25,
    "stack_advantage": 0.55,
    "hero_stack_ratio": 0.35,
    "street_progress": 0.10,
    "street_is_preflop": 0.0,
    "street_is_flop": 0.0,
    "street_is_turn": 0.0,
    "street_is_river": 0.0,
    "opp_raise_rate": -0.20,
    "opp_fold_rate": 0.10,
    "opp_call_rate": 0.0,
    "opp_aggression": -0.20,
    "hero_is_next": 0.05,
    "bias": 0.0,
}


class ValueModel:

    def __init__(self, weights: Optional[dict[str, float]] = None):
        self.weights: dict[str, float] = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    def predict(self, features: dict[str, float]) -> float:
        s = sum(self.weights.get(k, 0.0) * v for k, v in features.items())
        s += self.weights.get("bias", 0.0)
        return max(-1.0, min(1.0, s))

    def update(self, features: dict[str, float], target: float, lr: float) -> float:
        """One SGD step on MSE. Returns squared error."""
        s = sum(self.weights.get(k, 0.0) * v for k, v in features.items())
        s += self.weights.get("bias", 0.0)
        pred = max(-1.0, min(1.0, s))
        error = pred - target
        clip_gate = 1.0 if -1.0 < s < 1.0 else 0.0
        scale = 2.0 * lr * error * clip_gate
        for key, val in features.items():
            if key in self.weights:
                self.weights[key] -= scale * val
        if "bias" in self.weights:
            self.weights["bias"] -= scale
        return error * error

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model_type": "linear"}
        payload.update(self.weights)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def __repr__(self) -> str:
        top = sorted(self.weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        return f"ValueModel(top_weights={dict(top)})"


def load_value_model(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    model_type = data.get("model_type", "linear")
    if model_type == "mlp":
        from offline_learning.mlp_value_model import MLPValueModel
        return MLPValueModel.load(path)
    if model_type == "transformer":
        from offline_learning.transformer_value_model import TransformerValueModel
        return TransformerValueModel.load(path)
    # Linear: strip model_type key before passing as weights dict.
    weights = {k: v for k, v in data.items() if k != "model_type"}
    return ValueModel(weights)
