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
    actual_call_amount: int = 0   # Real call cost from engine valid_actions (0 = use heuristic)
    actual_raise_amount: int = 0  # Real min-raise total from engine valid_actions (0 = use heuristic)
    hero_folded: bool = False     # True only when hero's own fold action produced this state

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


def _compute_bet_amounts(hero_uuid: str, round_state: Dict[str, Any]):
    """
    Derive real call cost and min-raise total from round_state['action_histories'].

    PyPokerEngine's legal_actions() strips amounts, so we recompute them:
      agree_amount  = highest total bet any player has committed this street
      hero_paid     = what hero has already committed this street
      call_amount   = agree_amount - hero_paid  (>= 0)
      raise_amount  = agree_amount + last_add_amount  (min-raise total)
    """
    street = round_state.get("street", "preflop")
    histories = round_state.get("action_histories", {})
    street_history = histories.get(street, [])

    sb = int(round_state.get("small_blind_amount", 10))
    default_inc = 2 * sb  # preflop/flop; turn/river uses 4*sb but sb unknown here

    agree_amount: int = 0
    last_add_amount: int = sb  # minimum raise increment default
    hero_committed: int = 0   # hero's total committed this street (the 'amount' field)

    for entry in street_history:
        action = entry.get("action", "")
        amt = int(entry.get("amount", 0) or 0)
        add = int(entry.get("add_amount", 0) or 0)

        if action in ("RAISE", "SMALLBLIND", "BIGBLIND"):
            if amt > agree_amount:
                agree_amount = amt
            if add > 0:
                last_add_amount = add

        if entry.get("uuid") == hero_uuid:
            # 'amount' is the player's cumulative total committed this street.
            if amt > hero_committed:
                hero_committed = amt

    actual_call_amount = max(0, agree_amount - hero_committed)
    actual_raise_amount = agree_amount + last_add_amount if agree_amount > 0 else default_inc
    return actual_call_amount, actual_raise_amount


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

    # Extract real call / raise costs from action_histories (valid_actions has no amounts).
    actual_call_amount, actual_raise_amount = _compute_bet_amounts(
        hero_uuid, round_state
    )

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
        actual_call_amount=actual_call_amount,
        actual_raise_amount=actual_raise_amount,
    )

