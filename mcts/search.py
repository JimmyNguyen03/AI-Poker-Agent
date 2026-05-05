import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from .node import Node
from .state import PokerState


def _limit_raise_increment_chips(small_blind: int, street: str) -> int:
    """Match fixed-limit engine / abstraction_heuristics: preflop+flop 2*sb, turn+river 4*sb."""
    if street in ("preflop", "flop"):
        return 2 * small_blind
    if street in ("turn", "river"):
        return 4 * small_blind
    return 0


def _board_prefix_for_street(round_state_raw: Dict[str, Any], street: str) -> Tuple[str, ...]:
    """
    When the abstract model advances street, use board cards from the same engine snapshot
    (root round_state_raw) so rollouts do not play the turn with a preflop-only board.
    """
    raw_cards = list(round_state_raw.get("community_card", []))
    want = {"preflop": 0, "flop": 3, "turn": 4, "river": 5, "showdown": len(raw_cards)}.get(
        street, len(raw_cards)
    )
    if want > len(raw_cards):
        want = len(raw_cards)
    return tuple(raw_cards[:want])


def _apply_heuristic_transition(state: PokerState, action: str) -> PokerState:
    hero_stack = state.hero_stack
    opp_stack = state.opp_stack
    pot = state.pot_main
    sb = state.small_blind_amount
    street_order = ["preflop", "flop", "turn", "river", "showdown"]
    idx = street_order.index(state.street) if state.street in street_order else 0
    inc = _limit_raise_increment_chips(sb, state.street)

    # Use real engine amounts when available; fall back to fixed-limit heuristic.
    call_cost = state.actual_call_amount if state.actual_call_amount > 0 else max(1, inc // 2)
    raise_cost = state.actual_raise_amount if state.actual_raise_amount > 0 else (inc if inc else 2 * sb)

    # Child amounts reset to 0 (heuristic) unless propagated below.
    next_call_amount = 0
    next_raise_amount = 0

    if action == "fold":
        if state.hero_is_next:
            # Hero folds: no additional deduction (chips already committed are in pot_main).
            pass
        else:
            # Opponent folds: hero collects the pot.
            hero_stack = hero_stack + pot
            pot = 0
        street = "showdown"
        community = _board_prefix_for_street(state.round_state_raw, street)
    elif action == "call":
        # Asymmetric: only the acting player pays the call cost.
        if state.hero_is_next:
            hero_stack = max(0, hero_stack - call_cost)
        else:
            opp_stack = max(0, opp_stack - call_cost)
        pot += call_cost
        street = street_order[min(idx + 1, len(street_order) - 1)]
        community = _board_prefix_for_street(state.round_state_raw, street)
        # Street advances: revert to heuristic amounts for the new street.
    else:  # raise
        # Asymmetric: only the acting player commits the raise amount.
        if state.hero_is_next:
            hero_stack = max(0, hero_stack - raise_cost)
        else:
            opp_stack = max(0, opp_stack - raise_cost)
        pot += raise_cost
        street = street_order[min(idx + 1, len(street_order) - 1)]
        community = _board_prefix_for_street(state.round_state_raw, street)
        # Opponent now faces raise_cost as their call; cap re-raise at effective stack.
        next_call_amount = raise_cost
        next_raise_amount = min(raise_cost * 2, max(hero_stack, opp_stack))

    next_legal = ("fold", "call", "raise") if street != "showdown" else tuple()
    hero_folded = (action == "fold" and state.hero_is_next)
    return PokerState(
        hero_uuid=state.hero_uuid,
        hole_card=state.hole_card,
        street=street,
        next_player=1 - state.next_player,
        small_blind_amount=state.small_blind_amount,
        hero_is_next=not state.hero_is_next,
        community_card=community,
        pot_main=pot,
        hero_stack=hero_stack,
        opp_stack=opp_stack,
        legal_actions=tuple(next_legal),
        opp_fold_rate=state.opp_fold_rate,
        opp_call_rate=state.opp_call_rate,
        opp_raise_rate=state.opp_raise_rate,
        opp_strength_estimate=state.opp_strength_estimate,
        round_state_raw=state.round_state_raw,
        actual_call_amount=next_call_amount,
        actual_raise_amount=next_raise_amount,
        hero_folded=hero_folded,
    )


def _rollout_value(state: PokerState) -> float:
    """
    Value estimate in [-1, 1], from hero perspective.

    Incorporates hero's MC win probability so rollouts are card-aware:
    AA returns a clearly positive value, 72o a clearly negative one.
    This is the signal the value model (and pure-rollout MCTS) trains on.
    """
    if state.hole_card:
        from mcts.features import _hand_strength, _COMMUNITY_STREET
        wr_street = _COMMUNITY_STREET.get(len(state.community_card), "preflop")
        win_rate, _, _ = _hand_strength(state.hole_card, state.community_card, wr_street)
        hand_advantage = 2.0 * win_rate - 1.0
    else:
        hand_advantage = 0.0

    if state.street == "showdown":
        # Opp folded: pot was set to 0 by _apply_heuristic_transition.
        if state.pot_main == 0:
            return 1.0
        # Hero folded: hero gave up the hand — neutral outcome (saved remaining chips
        # but lost pot equity). Using 0.0 separates this from a winning call-down
        # (+hand_advantage) so MCTS correctly prefers calling good hands over folding.
        if state.hero_folded:
            return 0.0
        # Called down to abstract showdown — hand strength determines the result.
        return max(-1.0, min(1.0, hand_advantage))

    stack_delta = state.hero_stack - state.opp_stack
    norm = max(state.hero_stack + state.opp_stack, 1)
    base_value = stack_delta / norm

    belief_adjustment = 0.35 * (0.5 - state.opp_strength_estimate)
    return max(-1.0, min(1.0, base_value + 0.5 * hand_advantage + belief_adjustment))


def _sample_rollout_action(state: PokerState) -> str:
    legal = list(state.legal_actions)
    if len(legal) == 1:
        return legal[0]

    if state.hero_is_next:
        # Use hero's actual hand strength so rollouts are card-aware:
        # good hands play aggressively, weak hands fold.
        if state.hole_card:
            from mcts.features import _hand_strength, _COMMUNITY_STREET
            wr_street = _COMMUNITY_STREET.get(len(state.community_card), "preflop")
            win_rate, _, _ = _hand_strength(state.hole_card, state.community_card, wr_street)
        else:
            win_rate = 0.5
        raise_weight = 0.2 + 2.0 * win_rate + 0.5 * state.opp_fold_rate
        call_weight  = 0.5 + 1.0 * win_rate
        fold_weight  = max(0.05, 1.5 - 2.0 * win_rate)
    else:
        # Keep fold probability proportional to observed rate so rollouts reflect
        # the actual opponent (e.g. RaisedPlayer never folds → ~3% in rollout).
        raise_weight = 0.5 + 2.5 * state.opp_raise_rate
        call_weight  = 0.5 + 2.0 * state.opp_call_rate
        fold_weight  = max(0.05, 0.1 + 2.5 * state.opp_fold_rate)

    weights = []
    for action in legal:
        if action == "raise":
            weights.append(raise_weight)
        elif action == "call":
            weights.append(call_weight)
        else:
            weights.append(fold_weight)
    return random.choices(legal, weights=weights, k=1)[0]


def _simulate_from(
    node: Node,
    rollout_depth: int = 8,
    value_estimator: Optional[Callable[[PokerState], float]] = None,
) -> float:
    state = node.state
    if state.is_terminal():
        return _rollout_value(state)
    if value_estimator is not None:
        return max(-1.0, min(1.0, float(value_estimator(state))))
    # Pure rollout fallback (no value estimator).
    depth = 0
    while not state.is_terminal() and depth < rollout_depth and state.legal_actions:
        action = _sample_rollout_action(state)
        state = _apply_heuristic_transition(state, action)
        depth += 1
    return _rollout_value(state)


def _expand(node: Node) -> Node:
    action = node.untried_actions.pop(random.randrange(len(node.untried_actions)))
    child_state = _apply_heuristic_transition(node.state, action)
    child = Node(state=child_state, parent=node, action_from_parent=action)
    node.children[action] = child
    return child


def run_mcts(
    root_state: PokerState,
    num_simulations: int = 250,
    exploration: float = 1.4,
    rollout_depth: int = 8,
    value_estimator: Optional[Callable[[PokerState], float]] = None,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    root = Node(state=root_state)
    if not root.state.legal_actions:
        return "fold", {}

    for _ in range(num_simulations):
        node = root

        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child_uct(exploration)

        # Expansion
        if not node.state.is_terminal() and node.untried_actions:
            node = _expand(node)

        # Simulation
        reward = _simulate_from(node, rollout_depth=rollout_depth, value_estimator=value_estimator)

        # Backpropagation
        while node is not None:
            node.visits += 1
            node.value_sum += reward
            node = node.parent

    if not root.children:
        return random.choice(list(root.state.legal_actions)), {}

    # By lecture convention: pick action with highest visit count.
    best_action, best_child = max(root.children.items(), key=lambda kv: kv[1].visits)
    diagnostics = {
        action: {"visits": child.visits, "mean_value": child.mean_value}
        for action, child in root.children.items()
    }
    return best_action, diagnostics

