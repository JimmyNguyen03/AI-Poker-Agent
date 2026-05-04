"""Small value MLP: input → ReLU hidden → scalar (PyTorch + Adam)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Windows: PyTorch + NumPy/MKL often load two OpenMP runtimes; avoid abort on import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn

from offline_learning.value_model import _clip


def _build_net(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


class MLPValueModel:
    """
    Clipped value in [-1, 1] at inference; trained with MSE on raw logits (batch Adam per epoch).
    """

    def __init__(self, input_dim: int, hidden_dim: int, net: Optional[nn.Sequential] = None):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = net if net is not None else _build_net(input_dim, hidden_dim)

    @classmethod
    def zeros(cls, input_dim: int, hidden_dim: int = 32, seed: Optional[int] = None) -> MLPValueModel:
        if seed is not None:
            torch.manual_seed(seed)
        m = cls(input_dim, hidden_dim)
        for mod in m.net.modules():
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_uniform_(mod.weight, nonlinearity="relu")
                nn.init.zeros_(mod.bias)
        return m

    @torch.inference_mode()
    def predict_raw(self, features: List[float]) -> float:
        self.net.eval()
        x = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
        return float(self.net(x).squeeze())

    def predict(self, features: List[float]) -> float:
        return _clip(self.predict_raw(features), -1.0, 1.0)

    def train(self, records: Iterable[Tuple[List[float], float]], epochs: int = 5, lr: float = 0.01) -> List[float]:
        rows = list(records)
        if not rows:
            return []
        x = torch.as_tensor([r[0] for r in rows], dtype=torch.float32)
        y = torch.as_tensor([[r[1]] for r in rows], dtype=torch.float32)

        self.net.train()
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        history: List[float] = []
        for _ in range(max(1, epochs)):
            opt.zero_grad(set_to_none=True)
            pred = self.net(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            history.append(float(loss.detach()))

        self.net.eval()
        return history

    def to_payload(self) -> Dict[str, Any]:
        sd = self.net.state_dict()
        return {
            "kind": "mlp",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "state_dict": {k: v.detach().cpu().tolist() for k, v in sd.items()},
        }

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_payload(), indent=2), encoding="utf-8")

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> MLPValueModel:
        input_dim = int(payload["input_dim"])
        hidden_dim = int(payload["hidden_dim"])
        net = _build_net(input_dim, hidden_dim)

        if "state_dict" in payload:
            tensors = {
                k: torch.as_tensor(v, dtype=torch.float32) for k, v in payload["state_dict"].items()
            }
            net.load_state_dict(tensors)
        else:
            # Legacy JSON: W1 [H,D], b1 [H], W2 [H], b2 scalar
            w1 = torch.as_tensor(payload["W1"], dtype=torch.float32)
            b1 = torch.as_tensor(payload["b1"], dtype=torch.float32)
            w2 = torch.as_tensor(payload["W2"], dtype=torch.float32).view(1, -1)
            b2 = torch.as_tensor([float(payload["b2"])], dtype=torch.float32)
            net.load_state_dict(
                {
                    "0.weight": w1,
                    "0.bias": b1,
                    "2.weight": w2,
                    "2.bias": b2,
                }
            )

        return cls(input_dim=input_dim, hidden_dim=hidden_dim, net=net)

    @classmethod
    def load(cls, path: str) -> MLPValueModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_payload(payload)
