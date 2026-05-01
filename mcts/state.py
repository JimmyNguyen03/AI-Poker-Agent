from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PlayerSnapshot:
    uuid: str
    stack: int
    state: str


@dataclass(frozen=True)
class PokerState:

    hero_uuid: str
    hole_card: Tuple[str, ...]
    street: str
    next_player: int
    small_blind_amount: int
    hero_is_next: bool
    community_card: Tuple[str, ...]
    pot_main: int
    hero_stack: int
    opp_stack: int
    legal_actions: Tuple[str, ...]
    opp_fold_rate: float
    opp_call_rate: float
    opp_raise_rate: float
    opp_strength_estimate: float
    round_state_raw: Dict[str, Any]

    def is_terminal(self) -> bool:
        return self.street == "showdown" or self.hero_stack <= 0 or self.opp_stack <= 0


def _extract_main_pot(round_state: Dict[str, Any]) -> int:
    pot_info = round_state.get("pot", {})
    if isinstance(pot_info, dict):
        main = pot_info.get("main", 0)
        if isinstance(main, dict):
            return int(main.get("amount", 0))
        return int(main or 0)
    return 0


def _find_player(seats: List[Dict[str, Any]], uuid: str) -> Optional[Dict[str, Any]]:
    for seat in seats:
        if seat.get("uuid") == uuid:
            return seat
    return None


def build_state(
    hero_uuid: str,
    valid_actions: List[Dict[str, Any]],
    hole_card: List[str],
    round_state: Dict[str, Any],
    belief_snapshot: Optional[Dict[str, float]] = None,
) -> PokerState:
    seats = round_state.get("seats", [])
    hero = _find_player(seats, hero_uuid) or {}
    opp = next((s for s in seats if s.get("uuid") != hero_uuid), {})
    hero_index = next((idx for idx, seat in enumerate(seats) if seat.get("uuid") == hero_uuid), 0)
    next_player = int(round_state.get("next_player", 0))
    action_names = tuple(a.get("action") for a in valid_actions if "action" in a)
    belief = belief_snapshot or {}
    return PokerState(
        hero_uuid=hero_uuid,
        hole_card=tuple(hole_card),
        street=round_state.get("street", "preflop"),
        next_player=next_player,
        small_blind_amount=int(round_state.get("small_blind_amount", 10)),
        hero_is_next=(next_player == hero_index),
        community_card=tuple(round_state.get("community_card", [])),
        pot_main=_extract_main_pot(round_state),
        hero_stack=int(hero.get("stack", 0)),
        opp_stack=int(opp.get("stack", 0)),
        legal_actions=action_names,
        opp_fold_rate=float(belief.get("opp_fold_rate", 1.0 / 3.0)),
        opp_call_rate=float(belief.get("opp_call_rate", 1.0 / 3.0)),
        opp_raise_rate=float(belief.get("opp_raise_rate", 1.0 / 3.0)),
        opp_strength_estimate=float(belief.get("opp_strength_estimate", 0.5)),
        round_state_raw=round_state,
    )

