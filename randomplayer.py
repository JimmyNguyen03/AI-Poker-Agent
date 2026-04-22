import random

from pypokerengine.players import BasePokerPlayer


class RandomPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        action_info = random.choice(valid_actions)
        action = action_info["action"]
        amount = action_info.get("amount")
        if amount is None:
            return action
        if isinstance(amount, dict):
            max_amount = amount.get("max")
            min_amount = amount.get("min")
            if max_amount is not None and max_amount != -1:
                return action, int(max_amount)
            if min_amount is not None and min_amount != -1:
                return action, int(min_amount)
            return action
        return action, int(amount)

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


def setup_ai():
    return RandomPlayer()

