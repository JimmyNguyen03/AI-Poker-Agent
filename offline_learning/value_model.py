import json
from pathlib import Path
from typing import Iterable, List, Tuple


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class LinearValueModel:
    """Minimal trainable placeholder model: y = tanh(w.x + b)."""

    def __init__(self, weights: List[float], bias: float = 0.0):
        self.weights = list(weights)
        self.bias = float(bias)

    @classmethod
    def zeros(cls, input_dim: int) -> "LinearValueModel":
        return cls(weights=[0.0] * input_dim, bias=0.0)

    def predict_raw(self, features: List[float]) -> float:
        return sum(w * x for w, x in zip(self.weights, features)) + self.bias

    def predict(self, features: List[float]) -> float:
        # Keep value scale aligned with MCTS reward range.
        raw = self.predict_raw(features)
        return _clip(raw, -1.0, 1.0)

    def train_step(self, features: List[float], target: float, lr: float = 0.01) -> float:
        pred = self.predict(features)
        err = pred - target
        for idx, x in enumerate(features):
            self.weights[idx] -= lr * err * x
        self.bias -= lr * err
        return err * err

    def train(
        self,
        records: Iterable[Tuple[List[float], float]],
        epochs: int = 5,
        lr: float = 0.01,
    ) -> List[float]:
        loss_history: List[float] = []
        for _ in range(max(1, epochs)):
            total = 0.0
            count = 0
            for features, target in records:
                total += self.train_step(features, target, lr=lr)
                count += 1
            loss_history.append(total / max(1, count))
        return loss_history

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"weights": self.weights, "bias": self.bias}
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "LinearValueModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(weights=list(payload["weights"]), bias=float(payload["bias"]))

