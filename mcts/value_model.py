"""
Minimal inference-only value model loader for the tournament runtime.
Supports linear, mlp, and transformer checkpoints.
Kept in mcts/ to avoid any dependency on the offline_learning training package.
"""

from __future__ import annotations
import json
import math
from pathlib import Path

# Canonical feature order — must match offline_learning/value_model.py.
_FEATURE_KEYS = (
    "win_rate", "hand_strength_norm", "hand_group_norm",
    "pot_odds", "call_price_stack_fraction", "raise_price_stack_fraction",
    "stack_advantage", "hero_stack_ratio",
    "street_progress", "street_is_preflop", "street_is_flop",
    "street_is_turn", "street_is_river",
    "opp_raise_rate", "opp_fold_rate", "opp_call_rate", "opp_aggression",
    "hero_is_next",
)


def _to_vec(features: dict) -> list[float]:
    return [float(features.get(k, 0.0)) for k in _FEATURE_KEYS]


def _relu(x: list[float]) -> list[float]:
    return [v if v > 0.0 else 0.0 for v in x]


def _matmul_vec(W: list[list[float]], x: list[float]) -> list[float]:
    """W (rows × cols) @ x (cols,) → (rows,)."""
    return [sum(W[i][j] * x[j] for j in range(len(x))) for i in range(len(W))]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(len(a))]


def _clip(v: float) -> float:
    return max(-1.0, min(1.0, v))


class LinearValueModel:
    """Dot-product value estimator over state features."""

    def __init__(self, weights: dict):
        self.weights = weights

    def predict(self, features: dict) -> float:
        s = sum(self.weights.get(k, 0.0) * v for k, v in features.items())
        s += self.weights.get("bias", 0.0)
        return _clip(s)


class MLPValueModel:
    """Inference-only two-hidden-layer MLP (pure Python, no numpy required)."""

    def __init__(self, W1, b1, W2, b2, W3, b3):
        self.W1, self.b1 = W1, b1
        self.W2, self.b2 = W2, b2
        self.W3, self.b3 = W3, b3

    def predict(self, features: dict) -> float:
        x  = _to_vec(features)
        h1 = _relu(_add(_matmul_vec(self.W1, x),  self.b1))
        h2 = _relu(_add(_matmul_vec(self.W2, h1), self.b2))
        out = _add(_matmul_vec(self.W3, h2), self.b3)[0]
        return _clip(out)


class TransformerValueModel:
    """Inference-only single-layer self-attention transformer (pure Python)."""

    def __init__(self, E_scale, E_bias, W_Q, W_K, W_V, W_out, b_out):
        self.E_scale = E_scale  # (N, d)
        self.E_bias  = E_bias   # (N, d)
        self.W_Q     = W_Q      # (d, d)
        self.W_K     = W_K      # (d, d)
        self.W_V     = W_V      # (d, d)
        self.W_out   = W_out    # (1, d)
        self.b_out   = b_out    # (1,)

    def predict(self, features: dict) -> float:
        x = _to_vec(features)   # length N
        N = len(x)
        d = len(self.W_Q)

        # Token embeddings: H[i] = x[i] * E_scale[i] + E_bias[i]
        H = [[x[i] * self.E_scale[i][k] + self.E_bias[i][k]
              for k in range(d)] for i in range(N)]

        Q = [_matmul_vec(self.W_Q, H[i]) for i in range(N)]
        K = [_matmul_vec(self.W_K, H[i]) for i in range(N)]
        V = [_matmul_vec(self.W_V, H[i]) for i in range(N)]

        # Scaled dot-product attention with stable softmax
        scale = math.sqrt(d)
        scores = [[sum(Q[i][k] * K[j][k] for k in range(d)) / scale
                   for j in range(N)] for i in range(N)]
        attn = []
        for row in scores:
            mx = max(row)
            exps = [math.exp(v - mx) for v in row]
            s = sum(exps)
            attn.append([e / s for e in exps])

        # context[i] = sum_j attn[i][j] * V[j]
        context = [[sum(attn[i][j] * V[j][k] for j in range(N))
                    for k in range(d)] for i in range(N)]

        # Mean pool over tokens
        pooled = [sum(context[i][k] for i in range(N)) / N for k in range(d)]

        out = sum(self.W_out[0][k] * pooled[k] for k in range(d)) + self.b_out[0]
        return _clip(out)


def load_value_model(path: str):
    """Load a value model checkpoint. Dispatches on model_type field."""
    with open(path, "r") as f:
        data = json.load(f)
    model_type = data.get("model_type", "linear")

    if model_type == "linear":
        weights = {k: float(v) for k, v in data.items() if k != "model_type"}
        return LinearValueModel(weights)

    if model_type == "mlp":
        return MLPValueModel(
            W1=data["W1"], b1=data["b1"],
            W2=data["W2"], b2=data["b2"],
            W3=data["W3"], b3=data["b3"],
        )

    if model_type == "transformer":
        return TransformerValueModel(
            E_scale=data["E_scale"], E_bias=data["E_bias"],
            W_Q=data["W_Q"], W_K=data["W_K"], W_V=data["W_V"],
            W_out=data["W_out"], b_out=data["b_out"],
        )

    raise ValueError(
        f"Model type '{model_type}' is not supported in the tournament runtime."
    )
