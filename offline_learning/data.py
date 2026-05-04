import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

# Bump when on-disk label semantics change so replay/training can ignore stale rows.
TARGET_SCHEMA_VERSION = 2


@dataclass
class DecisionSample:
    game_index: int
    round_index: int
    player_uuid: str
    action: str
    legal_actions: List[str]
    features: List[float]
    terminal_reward: float
    # Normalized chip change from this decision to round end: clip(delta / norm_stack, -1, 1).
    training_target: float
    # Same scale as mcts/search._rollout_value at this decision state (see features.mcts_rollout_leaf_value).
    rollout_aligned_target: float
    final_hero_stack: int
    final_opp_stack: int
    metadata: Dict[str, float]
    target_schema: int = TARGET_SCHEMA_VERSION


class JsonlDatasetWriter:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_path.open("w", encoding="utf-8")

    def write(self, sample: DecisionSample) -> None:
        self._fh.write(json.dumps(asdict(sample), sort_keys=True) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

