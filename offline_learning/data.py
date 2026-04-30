import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class DecisionSample:
    game_index: int
    round_index: int
    player_uuid: str
    action: str
    legal_actions: List[str]
    features: List[float]
    terminal_reward: float
    final_hero_stack: int
    final_opp_stack: int
    metadata: Dict[str, float]


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

