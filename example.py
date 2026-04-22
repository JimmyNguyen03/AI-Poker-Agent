from pypokerengine.api.game import setup_config, start_poker

from mcts_player import MCTSPlayer
from randomplayer import RandomPlayer


def run_example():
    config = setup_config(max_round=50, initial_stack=1000, small_blind_amount=10)
    config.register_player(name="mcts_agent", algorithm=MCTSPlayer(simulations=250))
    config.register_player(name="random_agent", algorithm=RandomPlayer())
    game_result = start_poker(config, verbose=1)
    return game_result


if __name__ == "__main__":
    result = run_example()
    print("Final stacks:", [(p["name"], p["stack"]) for p in result["players"]])

