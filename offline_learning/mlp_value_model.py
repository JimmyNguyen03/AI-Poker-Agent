"""
MLP value model: 18 → ReLU(32) → ReLU(16) → clip(1).

Pure numpy; no external ML library required.
Implements the same predict / update / save / load interface as ValueModel.
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path

from offline_learning.value_model import FEATURE_KEYS, features_to_vector

_PARAM_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")
_WEIGHT_NAMES = ("W1", "W2", "W3")  # biases excluded from L2


class MLPValueModel:
    """Two-hidden-layer MLP trained with Adam on MSE."""

    def __init__(self, hidden: tuple[int, int] = (32, 16), seed: int = 42):
        self._hidden = hidden
        D = len(FEATURE_KEYS)
        H1, H2 = hidden
        rng = np.random.default_rng(seed)

        # He initialisation (appropriate for ReLU activations).
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / D),  (H1, D)).astype(np.float64)
        self.b1 = np.zeros(H1, dtype=np.float64)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / H1), (H2, H1)).astype(np.float64)
        self.b2 = np.zeros(H2, dtype=np.float64)
        self.W3 = rng.normal(0.0, np.sqrt(2.0 / H2), (1,  H2)).astype(np.float64)
        self.b3 = np.zeros(1, dtype=np.float64)

        self._init_adam()

    def _init_adam(self) -> None:
        self._t = 0
        self._m = {p: np.zeros_like(getattr(self, p)) for p in _PARAM_NAMES}
        self._v = {p: np.zeros_like(getattr(self, p)) for p in _PARAM_NAMES}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, features: dict) -> float:
        x = features_to_vector(features)
        _, _, _, _, z3 = self._forward(x)
        return float(np.clip(z3, -1.0, 1.0))

    def update(self, features: dict, target: float, lr: float) -> float:
        """One Adam step on MSE (with gradient clipping + L2 reg). Returns squared error."""
        _BETA1, _BETA2, _EPS, _WD, _CLIP = 0.9, 0.999, 1e-8, 1e-4, 1.0

        x = features_to_vector(features)
        z1, h1, z2, h2, z3 = self._forward(x)
        pred = float(np.clip(z3, -1.0, 1.0))
        error = pred - target

        # Backprop
        clip_gate = 1.0 if -1.0 < z3 < 1.0 else 0.0
        dz3 = 2.0 * error * clip_gate                  # scalar

        dW3 = dz3 * h2[np.newaxis, :]                  # (1, H2)
        db3 = np.array([dz3])                           # (1,)
        dh2 = self.W3[0] * dz3                          # (H2,)

        dz2 = dh2 * (z2 > 0.0)                         # (H2,) ReLU gate
        dW2 = np.outer(dz2, h1)                        # (H2, H1)
        db2 = dz2
        dh1 = self.W2.T @ dz2                          # (H1,)

        dz1 = dh1 * (z1 > 0.0)                         # (H1,) ReLU gate
        dW1 = np.outer(dz1, x)                         # (H1, D)
        db1 = dz1

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

        # L2 regularization on weights only
        for p in _WEIGHT_NAMES:
            grads[p] = grads[p] + _WD * getattr(self, p)

        # Gradient clipping (global norm)
        total_norm = np.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values()))
        if total_norm > _CLIP:
            scale = _CLIP / total_norm
            grads = {p: g * scale for p, g in grads.items()}

        # Adam update
        self._t += 1
        t = self._t
        for p, g in grads.items():
            self._m[p] = _BETA1 * self._m[p] + (1.0 - _BETA1) * g
            self._v[p] = _BETA2 * self._v[p] + (1.0 - _BETA2) * g ** 2
            m_hat = self._m[p] / (1.0 - _BETA1 ** t)
            v_hat = self._v[p] / (1.0 - _BETA2 ** t)
            param = getattr(self, p)
            param -= lr * m_hat / (np.sqrt(v_hat) + _EPS)

        return error * error

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "model_type": "mlp",
                "hidden": list(self._hidden),
                "W1": self.W1.tolist(), "b1": self.b1.tolist(),
                "W2": self.W2.tolist(), "b2": self.b2.tolist(),
                "W3": self.W3.tolist(), "b3": self.b3.tolist(),
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> MLPValueModel:
        with open(path, "r") as f:
            d = json.load(f)
        m = cls(hidden=tuple(d["hidden"]))
        m.W1 = np.array(d["W1"]); m.b1 = np.array(d["b1"])
        m.W2 = np.array(d["W2"]); m.b2 = np.array(d["b2"])
        m.W3 = np.array(d["W3"]); m.b3 = np.array(d["b3"])
        m._init_adam()
        return m

    def __repr__(self) -> str:
        return f"MLPValueModel(hidden={self._hidden})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _forward(self, x: np.ndarray):
        """Returns (z1, h1, z2, h2, z3_scalar) for use in backprop."""
        z1 = self.W1 @ x + self.b1
        h1 = np.maximum(0.0, z1)
        z2 = self.W2 @ h1 + self.b2
        h2 = np.maximum(0.0, z2)
        z3 = float((self.W3 @ h2 + self.b3)[0])
        return z1, h1, z2, h2, z3
