from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from mcts_player import MCTSPlayer


def run_example():
    # Keep starter setup style, but use MCTS player for testing.
    config = setup_config(max_round=10, initial_stack=10000, small_blind_amount=10)
    config.register_player(name="mcts_agent", algorithm=MCTSPlayer(simulations=250))
    config.register_player(name="random_agent", algorithm=RandomPlayer())
    # Alternative baseline from starter code:
    # config.register_player(name="raise_agent", algorithm=RaisedPlayer())
    game_result = start_poker(config, verbose=1)
    return game_result


if __name__ == "__main__":
    result = run_example()
    print("Final stacks:", [(p["name"], p["stack"]) for p in result["players"]])
