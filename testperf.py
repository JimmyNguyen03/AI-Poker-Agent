from argparse import ArgumentParser
from statistics import mean

from pypokerengine.api.game import setup_config, start_poker

from mcts_player import MCTSPlayer
from randomplayer import RandomPlayer


def play_match(
    rounds_per_game: int,
    num_games: int,
    mcts_simulations: int,
    initial_stack: int = 1000,
    small_blind_amount: int = 10,
):
    mcts_final_stacks = []
    random_final_stacks = []
    mcts_wins = 0

    for game_idx in range(num_games):
        config = setup_config(
            max_round=rounds_per_game,
            initial_stack=initial_stack,
            small_blind_amount=small_blind_amount,
        )
        config.register_player(name="mcts_agent", algorithm=MCTSPlayer(simulations=mcts_simulations))
        config.register_player(name="random_agent", algorithm=RandomPlayer())
        result = start_poker(config, verbose=0)
        p0, p1 = result["players"][0], result["players"][1]
        mcts_final_stacks.append(p0["stack"])
        random_final_stacks.append(p1["stack"])
        if p0["stack"] > p1["stack"]:
            mcts_wins += 1
        print(f"Game {game_idx + 1}/{num_games}: mcts={p0['stack']} random={p1['stack']}")

    return {
        "num_games": num_games,
        "mcts_win_rate": mcts_wins / num_games,
        "mcts_mean_stack": mean(mcts_final_stacks),
        "random_mean_stack": mean(random_final_stacks),
    }


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--sims", type=int, default=250)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = play_match(
        rounds_per_game=args.rounds,
        num_games=args.games,
        mcts_simulations=args.sims,
    )
    print("Summary:", summary)

