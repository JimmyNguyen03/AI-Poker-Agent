from dataclasses import dataclass
from typing import Dict


@dataclass
class OpponentBeliefModel:

    fold_count: int = 1
    call_count: int = 1
    raise_count: int = 1
    observed_actions: int = 0

    def observe(self, action_event: Dict, hero_uuid: str) -> None:
        actor_uuid = action_event.get("player_uuid")
        if actor_uuid is None or actor_uuid == hero_uuid:
            return
        action = str(action_event.get("action", "")).lower()
        if action == "fold":
            self.fold_count += 1
            self.observed_actions += 1
        elif action == "call":
            self.call_count += 1
            self.observed_actions += 1
        elif action in {"raise", "bet"}:
            self.raise_count += 1
            self.observed_actions += 1

    def snapshot(self) -> Dict[str, float]:
        total = float(self.fold_count + self.call_count + self.raise_count)
        fold_rate = self.fold_count / total
        call_rate = self.call_count / total
        raise_rate = self.raise_count / total
        strength_estimate = min(1.0, max(0.0, 0.25 + 0.75 * raise_rate - 0.25 * fold_rate))
        return {
            "opp_fold_rate": fold_rate,
            "opp_call_rate": call_rate,
            "opp_raise_rate": raise_rate,
            "opp_strength_estimate": strength_estimate,
            "observations": float(self.observed_actions),
        }
