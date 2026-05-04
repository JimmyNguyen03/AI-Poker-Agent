from __future__ import annotations
from pypokerengine.engine.card import Card
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards
from typing import Any, Optional

from . import hand_features as hf

# Initialize the CutoffAbstraction data bundle object
def build_cutoff_abstraction(
        hero_uuid: str, hole_card: list[str], round_state: dict[str, Any],
        valid_actions: Optional[list[dict[str, Any]]], win_rate_simulations: Optional[int]) -> hf.CutoffAbstraction:
    
    call_amount, raise_amount = get_bet_amounts(hero_uuid, round_state, valid_actions)
    evaluator = hf.LinearCutoffEvaluator(None)
    hand_group = group_hand_strength(hole_card, round_state, win_rate_simulations)
    call_group = group_bet_size("call", call_amount, hero_uuid, round_state)
    raise_group = group_bet_size("raise", raise_amount, hero_uuid, round_state)

    features = get_cutoff_state_features(
        hero_uuid, hole_card, round_state, hand_group, call_group, raise_group
    )

    return hf.CutoffAbstraction(
        hand_group=hand_group,
        call_group=call_group,
        raise_group=raise_group,
        features=features,
        heuristic_value=evaluator.score(features),
    )

# Jimmy: one direct score for cuttoff evaluation
def get_evaluate_cutoff_state(
        hero_uuid: str, hole_card: list[str], round_state: dict[str, Any],
        valid_actions: Optional[list[dict[str, Any]]], weights: Optional[dict[str, float]],
        win_rate_simulations: Optional[int]) -> float:
    
    return build_cutoff_abstraction(
        hero_uuid=hero_uuid, 
        hole_card=hole_card, 
        round_state=round_state, 
        valid_actions=valid_actions, 
        win_rate_simulations=win_rate_simulations).heuristic_value

# Initialize HandStrengthGroup with hand score and win_rate to hand strength abstraction 
def group_hand_strength(
        hole_card: list[str], round_state: dict[str, Any], 
        win_rate_simulations: Optional[int]) -> hf.HandStrengthGroup:
    
    hole, community = _to_cards(hole_card), _to_cards(list(round_state.get("community_card", [])))
    default_win_rate = {"preflop": 120, "flop": 90, "turn": 70, "river": 50, "showdown": 50}.get(
        str(round_state.get("street", "preflop")), 50
    )
    win_rate = estimate_hole_card_win_rate(win_rate_simulations or default_win_rate, 2, hole, community)
    group_strength = next((g for t, g in [(0.80,5),(0.65,4),(0.50,3),(0.35,2)] if win_rate >= t), 1)
    hand = HandEvaluator.gen_hand_rank_info(hole, community)["hand"]["strength"]

    return hf.HandStrengthGroup(
        group_strength=group_strength, 
        win_rate=win_rate, 
        hand=hand, 
        hand_score=hf.HAND_TO_SCORE[hand]
    )

# Get the call/raise bets from valid_actions or build from the current fixed limit state
def get_bet_amounts(
        hero_uuid: str, round_state: dict[str, Any], 
        valid_actions: Optional[list[dict[str, Any]]]) -> tuple[int, int]:
    
    call_amount = _get_amount_from_valid_actions(valid_actions, "call")
    raise_call = _get_amount_from_valid_actions(valid_actions, "raise")
    
    if call_amount is None or raise_call is None:
        current_bet = max((int(h["amount"]) for h in _get_street_histories(round_state) if h.get("amount") is not None), default=0)
        hero_street_bet = _get_player_bet_in_street(hero_uuid, round_state)
        street = str(round_state.get("street", "preflop"))
        small_blind = int(round_state.get("small_blind_amount", 10))
        call_amount = max(0, current_bet - hero_street_bet)
        street_raise_increment = 2 * small_blind if street in ("preflop", "flop") else 4 * small_blind
        raise_call = max(0, current_bet + street_raise_increment - hero_street_bet)
    
    return call_amount, raise_call

# Initialize BetSizeGroup by grouping call/raise by relative size to the pot and hero stack
def group_bet_size(
        action: str, amount: int, hero_uuid: str, 
        round_state: dict[str, Any]) -> hf.BetSizeGroup:
    
    hero_seat = _find_seat(round_state, hero_uuid)
    amount_over_pot = float(amount) / max(_get_main_pot(round_state), 1)
    amount_over_stack = float(amount) / max(int(hero_seat.get("stack", 0)) if hero_seat else 0, 1)
    
    bet_size_group = (
        1 if amount <= 0
        else 2 if amount_over_pot < 0.25
        else 3 if amount_over_pot < 0.50
        else 4 if amount_over_stack < 0.50 and amount_over_pot < 1.00
        else 5
    )

    return hf.BetSizeGroup(
        action=action, 
        amount=int(amount), 
        bet_size_group=bet_size_group, 
        amount_over_pot=amount_over_pot, 
        amount_over_stack=amount_over_stack)

# Create the cutoff feature vector used by the linear evaluator
# Jimmy: for the raw feature vector inside
def get_cutoff_state_features(
        hero_uuid: str, hole_card: list[str], round_state: dict[str, Any], 
        hand_group: Optional[hf.HandStrengthGroup], call_group: Optional[hf.BetSizeGroup], 
        raise_group: Optional[hf.BetSizeGroup]) -> dict[str, float]:
    
    hero  = _find_seat(round_state, hero_uuid)
    opp = next((s for s in round_state.get("seats", []) if s.get("uuid") != hero_uuid), None)

    hand_group = hand_group or group_hand_strength(hole_card, round_state)
    call_amount, raise_amount = get_bet_amounts(hero_uuid, round_state, None)
    call_group = call_group or group_bet_size("call", call_amount, hero_uuid, round_state)
    raise_group = raise_group or group_bet_size("raise", raise_amount, hero_uuid, round_state)

    hero_stack, opp_stack = (int(x.get("stack", 0)) if x else 0 for x in (hero, opp))
    community = list(round_state.get("community_card", []))
    street = str(round_state.get("street"))

    # TODO add more features
    return {
        "win_rate": hand_group.win_rate,
        "hand_group_strength_norm": hand_group.group_strength / 4,
        "pot_odds": _division(call_amount, _get_main_pot(round_state) + call_amount),
        "stack_advantage": _division(hero_stack - opp_stack, hero_stack + opp_stack),
        f"street_is_preflop": float(street == "preflop"),
        f"street_is_flop": float(street == "flop"),
        f"street_is_turn": float(street == "turn"),
        f"street_is_river": float(street == "river"),
        #"board_danger": _board_danger(community),
        #"in_position": 1.0 if _is_in_position(hero_uuid, round_state) else 0,
        #"opp_aggression": _simple_history_stats(hero_uuid, round_state)["opp_aggression"],
    }

# -----------------------------------------------------------

# Safe division
def _division(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0

# Convert string cards into Card objects
def _to_cards(cards: list) -> list[Card]:
    return list(cards) if cards and isinstance(cards[0], Card) else (gen_cards(list(cards)) if cards else [])

# Find seat entry of the hero from the round_state seat list
def _find_seat(round_state: dict[str, Any], uuid: str) -> Optional[dict[str, Any]]:
    return next((seat for seat in round_state.get("seats", []) if seat.get("uuid") == uuid), None)

# Read the current main pot amount from round_state
def _get_main_pot(round_state: dict[str, Any]) -> int:
    main_Pot = round_state.get("pot", {}).get("main", 0)
    return int(main_Pot.get("amount", 0)) if isinstance(main_Pot, dict) else int(main_Pot or 0)

# Get a specific action amount from valid_actions
def _get_amount_from_valid_actions(
        valid_actions: Optional[list[dict[str, Any]]], action_name: str) -> Optional[int]:
    if valid_actions:
        for a in valid_actions:
            if a.get("action") == action_name:
                amount = a.get("amount")
                if amount is None:
                    return 0 if action_name == "fold" else None
                if isinstance(amount, dict):
                    if amount.get("max") is not None and amount.get("max") != -1:
                        return int(amount.get("max"))
                    if amount.get("min") is not None and amount.get("min") != -1:
                        return int(amount.get("min"))
                    return None
                return int(amount)
    return None

# Get the current street action history
def _get_street_histories(round_state: dict[str, Any]) -> list[dict[str, Any]]:
    h = round_state.get("action_histories", {})
    return list((h.get(str(round_state.get("street", "preflop")), []) if isinstance(h, dict) else []) or [])

# Get the current street bet of the hero from action history
def _get_player_bet_in_street(hero_uuid: str, round_state: dict[str, Any]) -> int:
    hist = _get_street_histories(round_state)
    return max((int(h["amount"]) for h in hist if h.get("uuid") == hero_uuid and h.get("amount") is not None), default=0)

# -----------------------------------------------------------

# TODO Approximate position by checking whether hero is currently on the dealer button
def _is_in_position(hero_uuid: str, round_state: dict[str, Any]) -> bool:
    pass

# TODO combine history stats for heuristic: raise counts, raise rates, opponent aggression
def _simple_history_stats(hero_uuid: str, round_state: dict[str, Any]) -> dict[str, float]:
    pass

# TODO Estimate score from card patterns in the community cards
def _board_danger(community_cards: list[str]) -> float:
    pass
