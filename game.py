import sys
from copy import deepcopy
from typing import List, Tuple

import players
from classes import Action, Hand, Tap


class GameState:
    def __init__(
        self,
        player_one: players.Player,
        player_two: players.Player,
        enable_announce=True,
    ):
        self.player_one = player_one
        self.player_two = player_two
        self.active_player = self.player_one
        self.inactive_player = self.player_two
        self.enable_announce = enable_announce

    def play(self):
        self.player_one.setup()
        self.player_two.setup()
        while True:
            winner = self.decide_winner()
            if self.enable_announce:
                self.announce_state()
            if winner is not None:
                self.player_one.process_results(self)
                self.player_two.process_results(self)
                return winner
            else:
                if self.enable_announce:
                    print(self.active_player.name + ", it is your turn.")
                action = self.active_player.decide_action(self)
                try:
                    self.active_player.take_action(self, action)
                except RuntimeError as err:
                    print(f"Invalid action: {err}")
                    print("Retrying")
                    raise err
                self.swap_active()

    def swap_active(self):
        if self.active_player == self.player_one:
            self.active_player = self.player_two
            self.inactive_player = self.player_one
        else:
            self.active_player = self.player_one
            self.inactive_player = self.player_two

    def decide_winner(self):
        if self.player_one.left_fingers == 0 and self.player_one.right_fingers == 0:
            return self.player_two
        if self.player_two.left_fingers == 0 and self.player_two.right_fingers == 0:
            return self.player_one
        return None

    def announce_state(self):
        print("\n")
        winner = self.decide_winner()
        if winner == self.player_two:
            print(self.player_two.name + " Wins!")
        if winner == self.player_one:
            print(self.player_one.name + " Wins!")
        print(
            self.player_one.name
            + ": Left Hand "
            + str(self.player_one.left_fingers)
            + " : Right Hand "
            + str(self.player_one.right_fingers)
        )
        print(
            self.player_two.name
            + ": Left Hand "
            + str(self.player_two.left_fingers)
            + " : Right Hand "
            + str(self.player_two.right_fingers)
        )

    def generate_child_sa_pairs(self) -> List[Tuple["GameState", Action]]:
        if self.decide_winner() is not None:
            copied_state = deepcopy(self)
            copied_state.swap_active()
            return [(copied_state, None)]
        child_states = []
        valid_actions = self.active_player.generate_valid_actions(self.inactive_player)
        for action in valid_actions:
            copied_state = deepcopy(self)
            if type(action) is Tap:
                active_player = copied_state.active_player
                opponent = copied_state.inactive_player
                active_player.tap(opponent, action)
            else:
                active_player.redistribute(action)
            copied_state.swap_active()
            child_states.append((copied_state, action))

        return child_states

    def create_copy_with_diff_fingers(self, active_fingers, inactive_fingers):
        copy = deepcopy(self)
        copy.active_player.left_fingers = active_fingers[0]
        copy.active_player.right_fingers = active_fingers[1]
        copy.inactive_player.left_fingers = inactive_fingers[0]
        copy.inactive_player.right_fingers = inactive_fingers[1]
        return copy

    def __str__(self):
        active_left = str(self.active_player.get_fingers(Hand.LEFT))
        active_right = str(self.active_player.get_fingers(Hand.RIGHT))
        active_str = f"Active: (Left: {active_left}, Right: {active_right})"
        inactive_left = str(self.inactive_player.get_fingers(Hand.LEFT))
        inactive_right = str(self.inactive_player.get_fingers(Hand.RIGHT))
        inactive_str = f"Inactive: (Left: {inactive_left}, Right: {inactive_right})"
        return active_str + " ~~ " + inactive_str


mode = sys.argv[1] if len(sys.argv) == 2 else "human"

if mode == "self-play":
    dyna_player_1 = players.DynaPlayer(name="dyna_1")
    dyna_player_2 = players.DynaPlayer(name="dyna_2")
    game = GameState(dyna_player_1, dyna_player_2, enable_announce=False)
    total_games = 0
    player_1_win_count = 0
    while True:
        if total_games % 100 == 0 and total_games != 0:
            dyna_player_1.long_term_agent.store_state()
            dyna_player_2.long_term_agent.store_state()
            print(total_games, player_1_win_count)
        res = game.play()
        if res == dyna_player_1:
            player_1_win_count += 1
        total_games += 1

if mode == "minimax":
    dyna_player = players.DynaPlayer(name="dyna_1")
    mm_player = players.MinimaxPlayer("Minimax", max_depth=16)
    game = GameState(dyna_player, mm_player, enable_announce=False)
    total_games = 0
    dyna_player_win_count = 0
    while total_games <= 50000:
        if total_games % 10 == 0 and total_games != 0:
            dyna_player.long_term_agent.store_state()
            print(total_games, dyna_player_win_count)
        res = game.play()
        if res == dyna_player:
            dyna_player_win_count += 1
        total_games += 1

if mode == "human":
    dyna_player = players.DynaPlayer(name="dyna_1")
    human_player = players.HumanPlayer("Human")
    game = GameState(dyna_player, human_player, enable_announce=True)
    game.play()
