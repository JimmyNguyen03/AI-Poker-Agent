from mcts.legal_actions import extract_action_names, to_engine_action
from mcts.search import run_mcts
from mcts.state import build_state


def _sample_round_state():
    return {
        "street": "preflop",
        "next_player": 0,
        "small_blind_amount": 10,
        "community_card": [],
        "pot": {"main": {"amount": 30}},
        "seats": [
            {"uuid": "hero", "stack": 990, "state": "participating"},
            {"uuid": "opp", "stack": 980, "state": "participating"},
        ],
    }


def test_state_builder():
    valid_actions = [{"action": "fold"}, {"action": "call"}, {"action": "raise"}]
    state = build_state(
        hero_uuid="hero",
        valid_actions=valid_actions,
        hole_card=["SA", "HK"],
        round_state=_sample_round_state(),
    )
    assert state.hero_uuid == "hero"
    assert state.pot_main == 30
    assert state.legal_actions == ("fold", "call", "raise")


def test_legal_actions():
    valid_actions = [
        {"action": "fold"},
        {"action": "call", "amount": 20},
        {"action": "raise", "amount": {"min": 30, "max": 30}},
    ]
    names = extract_action_names(valid_actions)
    assert names == ["fold", "call", "raise"]
    action, amount = to_engine_action("raise", valid_actions, _sample_round_state())
    assert action == "raise"
    assert amount == 30


def test_mcts_output_action_is_legal():
    valid_actions = [{"action": "fold"}, {"action": "call"}, {"action": "raise"}]
    state = build_state(
        hero_uuid="hero",
        valid_actions=valid_actions,
        hole_card=["SA", "HK"],
        round_state=_sample_round_state(),
    )
    action, diagnostics = run_mcts(state, num_simulations=40)
    assert action in {"fold", "call", "raise"}
    assert isinstance(diagnostics, dict)


if __name__ == "__main__":
    test_state_builder()
    test_legal_actions()
    test_mcts_output_action_is_legal()
    print("MCTS starter tests passed.")

