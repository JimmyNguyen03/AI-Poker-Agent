"""
Minimal transformer value model.

Architecture (single layer, single head):
  - Each of the N=18 features becomes a d-dim token:
      H[i] = value[i] * E_scale[i] + E_bias[i]
  - Scaled dot-product self-attention (Q, K, V projections)
  - Mean-pool attended context
  - Linear output, clipped to [-1, 1]

Pure numpy; no external ML library required.
Implements the same predict / update / save / load interface as ValueModel.
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path

from offline_learning.value_model import FEATURE_KEYS, features_to_vector

_N = len(FEATURE_KEYS)  # 18 tokens
_PARAM_NAMES = ("E_scale", "E_bias", "W_Q", "W_K", "W_V", "W_out", "b_out")
_WEIGHT_NAMES = ("E_scale", "W_Q", "W_K", "W_V", "W_out")  # biases excluded from L2


class TransformerValueModel:
    """Single-layer, single-head transformer over feature tokens."""

    def __init__(self, d_model: int = 16, seed: int = 42):
        self._d_model = d_model
        d = d_model
        rng = np.random.default_rng(seed)
        s = 0.02  # small init keeps early attention near uniform

        # Per-feature learnable (scale, bias) embeddings
        self.E_scale = rng.normal(0.0, s, (_N, d)).astype(np.float64)
        self.E_bias  = rng.normal(0.0, s, (_N, d)).astype(np.float64)

        # Attention projections
        self.W_Q = rng.normal(0.0, s, (d, d)).astype(np.float64)
        self.W_K = rng.normal(0.0, s, (d, d)).astype(np.float64)
        self.W_V = rng.normal(0.0, s, (d, d)).astype(np.float64)

        # Output linear layer
        self.W_out = rng.normal(0.0, s, (1, d)).astype(np.float64)
        self.b_out  = np.zeros(1, dtype=np.float64)

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
        pred, _ = self._forward(x)
        return pred

    def update(self, features: dict, target: float, lr: float) -> float:
        """One Adam step on MSE (with gradient clipping + L2 reg). Returns squared error."""
        _BETA1, _BETA2, _EPS, _WD, _CLIP = 0.9, 0.999, 1e-8, 1e-4, 1.0

        x = features_to_vector(features)
        pred, cache = self._forward(x)
        error = pred - target

        x_v, H, Q, K, V, attn, context, pooled, out_scalar = cache
        d = self._d_model

        # Backward through clip
        clip_gate = 1.0 if -1.0 < out_scalar < 1.0 else 0.0
        dout = 2.0 * error * clip_gate                   # scalar

        # Output linear
        dW_out  = dout * pooled[np.newaxis, :]           # (1, d)
        db_out  = np.array([dout])                       # (1,)
        dpooled = self.W_out[0] * dout                   # (d,)

        # Mean pool: context → pooled
        dcontext = np.tile(dpooled, (_N, 1)) / _N        # (N, d)

        # context = attn @ V
        dV    = attn.T @ dcontext                        # (N, d)
        dattn = dcontext @ V.T                           # (N, N)

        # Softmax backward (row-wise): ds_i = s_i * (d_i - s_i·d_i)
        d_scores = np.empty((_N, _N), dtype=np.float64)
        for i in range(_N):
            s = attn[i]
            d_scores[i] = s * (dattn[i] - np.dot(dattn[i], s))
        d_scores /= np.sqrt(d)                           # undo scale

        # scores = Q @ K.T / sqrt(d)
        dQ = d_scores @ K                                # (N, d)
        dK = d_scores.T @ Q                             # (N, d)

        # Q/K/V = H @ W_*
        dW_Q  = H.T @ dQ                                # (d, d)
        dH_Q  = dQ @ self.W_Q.T                         # (N, d)
        dW_K  = H.T @ dK                                # (d, d)
        dH_K  = dK @ self.W_K.T                         # (N, d)
        dW_V  = H.T @ dV                                # (d, d)
        dH_V  = dV @ self.W_V.T                         # (N, d)

        dH = dH_Q + dH_K + dH_V                        # (N, d)

        # H[i] = value[i] * E_scale[i] + E_bias[i]
        dE_scale = dH * x_v[:, np.newaxis]              # (N, d)
        dE_bias  = dH                                   # (N, d)

        grads = {
            "E_scale": dE_scale, "E_bias": dE_bias,
            "W_Q": dW_Q, "W_K": dW_K, "W_V": dW_V,
            "W_out": dW_out, "b_out": db_out,
        }

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
                "model_type": "transformer",
                "d_model": self._d_model,
                "E_scale": self.E_scale.tolist(),
                "E_bias":  self.E_bias.tolist(),
                "W_Q":     self.W_Q.tolist(),
                "W_K":     self.W_K.tolist(),
                "W_V":     self.W_V.tolist(),
                "W_out":   self.W_out.tolist(),
                "b_out":   self.b_out.tolist(),
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> TransformerValueModel:
        with open(path, "r") as f:
            d = json.load(f)
        m = cls(d_model=d["d_model"])
        m.E_scale = np.array(d["E_scale"])
        m.E_bias  = np.array(d["E_bias"])
        m.W_Q     = np.array(d["W_Q"])
        m.W_K     = np.array(d["W_K"])
        m.W_V     = np.array(d["W_V"])
        m.W_out   = np.array(d["W_out"])
        m.b_out   = np.array(d["b_out"])
        m._init_adam()
        return m

    def __repr__(self) -> str:
        return f"TransformerValueModel(d_model={self._d_model})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _forward(self, x: np.ndarray):
        """
        Full forward pass. Returns (pred, cache) where cache holds all
        intermediates needed for backprop.
        """
        d = self._d_model

        # Token embeddings: each feature gets its own d-dim representation
        H = x[:, np.newaxis] * self.E_scale + self.E_bias   # (N, d)

        Q = H @ self.W_Q   # (N, d)
        K = H @ self.W_K   # (N, d)
        V = H @ self.W_V   # (N, d)

        # Scaled dot-product attention (numerically stable softmax)
        scores = Q @ K.T / np.sqrt(d)                        # (N, N)
        scores -= scores.max(axis=1, keepdims=True)
        exp_s  = np.exp(scores)
        attn   = exp_s / exp_s.sum(axis=1, keepdims=True)    # (N, N)

        context = attn @ V                                    # (N, d)
        pooled  = context.mean(axis=0)                       # (d,)

        out_scalar = float((self.W_out @ pooled + self.b_out)[0])
        pred = float(np.clip(out_scalar, -1.0, 1.0))

        cache = (x, H, Q, K, V, attn, context, pooled, out_scalar)
        return pred, cache
