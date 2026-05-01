from argparse import ArgumentParser
from collections import Counter
import random

from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer

from mcts_player import MCTSPlayer


def _valid_action_names(valid_actions):
    return [entry["action"] for entry in valid_actions if "action" in entry]


class ScriptedOpponent(BasePokerPlayer):

    def __init__(self, fold_p=0.2, call_p=0.5, raise_p=0.3, seed=42):
        super().__init__()
        total = fold_p + call_p + raise_p
        self.fold_p = fold_p / total
        self.call_p = call_p / total
        self.raise_p = raise_p / total
        self.rng = random.Random(seed)
        self.action_counter = Counter()

    def declare_action(self, valid_actions, hole_card, round_state):
        legal = _valid_action_names(valid_actions)
        weighted = []
        for action in legal:
            if action == "fold":
                weighted.append(self.fold_p)
            elif action == "call":
                weighted.append(self.call_p)
            elif action == "raise":
                weighted.append(self.raise_p)
            else:
                weighted.append(0.0)

        if sum(weighted) <= 0:
            chosen = legal[0]
        else:
            chosen = self.rng.choices(legal, weights=weighted, k=1)[0]
        self.action_counter[chosen] += 1
        return chosen

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def run_belief_eval(rounds=400, sims=100, fold_p=0.2, call_p=0.5, raise_p=0.3, seed=42):
    mcts = MCTSPlayer(simulations=sims)
    opponent = ScriptedOpponent(fold_p=fold_p, call_p=call_p, raise_p=raise_p, seed=seed)

    config = setup_config(
        max_round=rounds,
        initial_stack=10000,
        small_blind_amount=20,
    )
    config.register_player(name="mcts", algorithm=mcts)
    config.register_player(name="scripted", algorithm=opponent)
    start_poker(config, verbose=0)

    belief = mcts.get_belief_snapshot()
    counts = opponent.action_counter
    total = max(1, counts["fold"] + counts["call"] + counts["raise"])
    actual = {
        "opp_fold_rate": counts["fold"] / total,
        "opp_call_rate": counts["call"] / total,
        "opp_raise_rate": counts["raise"] / total,
        "observations": float(total),
    }

    print("Target tendency:")
    print({"opp_fold_rate": fold_p, "opp_call_rate": call_p, "opp_raise_rate": raise_p})
    print("Actual sampled tendency:")
    print(actual)
    print("Belief model estimate:")
    print(belief)
    print("Absolute errors:")
    print(
        {
            "fold_err": abs(belief["opp_fold_rate"] - actual["opp_fold_rate"]),
            "call_err": abs(belief["opp_call_rate"] - actual["opp_call_rate"]),
            "raise_err": abs(belief["opp_raise_rate"] - actual["opp_raise_rate"]),
        }
    )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--fold-p", type=float, default=0.2)
    parser.add_argument("--call-p", type=float, default=0.5)
    parser.add_argument("--raise-p", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_belief_eval(
        rounds=args.rounds,
        sims=args.sims,
        fold_p=args.fold_p,
        call_p=args.call_p,
        raise_p=args.raise_p,
        seed=args.seed,
    )
