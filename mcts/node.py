import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .state import PokerState


@dataclass
class Node:
    state: PokerState
    parent: Optional["Node"] = None
    action_from_parent: Optional[str] = None
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[str, "Node"] = field(default_factory=dict)
    untried_actions: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.untried_actions:
            self.untried_actions = list(self.state.legal_actions)

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child_uct(self, exploration: float) -> "Node":
        parent_log = math.log(max(self.visits, 1))
        # Hero maximizes; opp minimizes hero's value. Use sign to flip exploit term.
        sign = 1.0 if self.state.hero_is_next else -1.0
        def score(child: "Node") -> float:
            if child.visits == 0:
                return float("inf")
            exploit = sign * child.mean_value
            explore = exploration * math.sqrt(parent_log / child.visits)
            return exploit + explore
        return max(self.children.values(), key=score)

