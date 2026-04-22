from typing import Dict, List, Optional, Tuple


def extract_action_names(valid_actions: List[Dict]) -> List[str]:
    return [entry["action"] for entry in valid_actions if "action" in entry]


def default_raise_amount(round_state: Dict) -> int:
    sb = int(round_state.get("small_blind_amount", 10))
    street = round_state.get("street", "preflop")
    # Match fixed-limit rules: preflop/flop raise is 2*SB, turn/river is 4*SB.
    if street in ("preflop", "flop"):
        return 2 * sb
    return 4 * sb


def select_amount_for_action(
    action: str,
    valid_actions: List[Dict],
    round_state: Dict,
) -> Optional[int]:
    """Return amount only when required by API consumers."""
    if action == "raise":
        for entry in valid_actions:
            if entry.get("action") != "raise":
                continue
            amount = entry.get("amount")
            # Compatible with both scalar and dict amount format.
            if isinstance(amount, dict):
                max_amount = amount.get("max")
                min_amount = amount.get("min")
                if max_amount is not None and max_amount != -1:
                    return int(max_amount)
                if min_amount is not None and min_amount != -1:
                    return int(min_amount)
            elif amount is not None:
                return int(amount)
        return default_raise_amount(round_state)
    if action == "call":
        for entry in valid_actions:
            if entry.get("action") == "call":
                amount = entry.get("amount")
                if amount is not None:
                    return int(amount)
    return None


def to_engine_action(
    action: str,
    valid_actions: List[Dict],
    round_state: Dict,
) -> Tuple[str, Optional[int]]:
    amount = select_amount_for_action(action, valid_actions, round_state)
    return action, amount

