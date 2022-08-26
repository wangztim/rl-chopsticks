from __future__ import annotations

import itertools
import math
import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, List, Tuple

from classes import Action, Hand, Redistribute, Tap

if TYPE_CHECKING:
    from game import GameState


class Player:
    possible_taps: List[Tap] = [
        Tap(Hand.LEFT, Hand.RIGHT),
        Tap(Hand.LEFT, Hand.LEFT),
        Tap(Hand.RIGHT, Hand.LEFT),
        Tap(Hand.RIGHT, Hand.RIGHT),
    ]

    def __init__(self, name: str):
        self.left_fingers = 1
        self.right_fingers = 1
        self.name = name

    def setup(self):
        self.left_fingers = 1
        self.right_fingers = 1

    def process_results(self, game_state: GameState):
        pass

    def take_action(self, game_state: GameState, action: Action):
        opponent = game_state.inactive_player
        if self.action_is_valid(action, opponent):
            if type(action) is Tap:
                return self.tap(opponent, action)
            else:
                return self.redistribute(action)
        else:
            raise RuntimeError("Invalid action detected")

    def decide_action(self, game_state: GameState) -> Action:
        pass

    def tap(self, opponent: Player, action: Tap):
        """
        Taps an opponent's *destination* hand with *source* hand
        """
        fingers_in_source_hand = self.get_fingers(action.src_hand)
        opponent.increment_hand_by(action.dest_hand, fingers_in_source_hand)

    def redistribute(self, action: Redistribute):
        """
        Redistributes fingers.
        """
        self.left_fingers = action.new_left_fingers
        self.right_fingers = action.new_right_fingers

    def action_is_valid(self, action: Action, opponent: Player = None) -> bool:
        if type(action) is Redistribute:
            current_fingers = sorted(
                [self.get_fingers(Hand.LEFT), self.get_fingers(Hand.RIGHT)]
            )
            new_fingers = sorted([action.new_left_fingers, action.new_right_fingers])

            same_unordered_pair = current_fingers == new_fingers
            finger_amount_same = sum(current_fingers) == sum(new_fingers)

            desired_left_valid = 0 < action.new_left_fingers < 5
            desired_right_valid = 0 < action.new_right_fingers < 5
            return (
                not same_unordered_pair
                and finger_amount_same
                and desired_left_valid
                and desired_right_valid
            )
        elif type(action) is Tap:
            fingers_in_source_hand = self.get_fingers(action.src_hand)
            fingers_in_dest_hand = opponent.get_fingers(action.dest_hand)

            return fingers_in_source_hand > 0 and fingers_in_dest_hand > 0
        else:
            return False

    def get_fingers(self, hand: Hand):
        if hand == Hand.LEFT:
            return self.left_fingers
        elif hand == Hand.RIGHT:
            return self.right_fingers

    def increment_hand_by(self, hand: Hand, amount: int):
        if amount >= 5 or amount <= 0:
            raise RuntimeError("Invalid increment amount")
        if hand == Hand.LEFT:
            self.left_fingers = (self.left_fingers + amount) % 5
        elif hand == Hand.RIGHT:
            self.right_fingers = (self.right_fingers + amount) % 5

    def generate_valid_actions(self, opponent: Player) -> List[Action]:
        valid_actions = []

        for tap in Player.possible_taps:
            if self.action_is_valid(tap, opponent):
                valid_actions.append(tap)

        total_fingers = self.get_fingers(Hand.LEFT) + self.get_fingers(Hand.RIGHT)
        possible_redistributions = [
            Redistribute(n, total_fingers - n) for n in range(total_fingers + 1)
        ]

        for redistribution in possible_redistributions:
            if self.action_is_valid(redistribution):
                valid_actions.append(redistribution)

        return valid_actions

    def parse_state(self, state: GameState) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        opponent = (
            state.active_player
            if state.active_player.name != self.name
            else state.inactive_player
        )
        me = (
            state.active_player
            if state.active_player.name == self.name
            else state.inactive_player
        )
        my_fingers = (me.left_fingers, me.right_fingers)
        opponent_fingers = (opponent.left_fingers, opponent.right_fingers)
        return (my_fingers, opponent_fingers)

    def __deepcopy__(self, memo):
        new_player = Player(self.name)
        new_player.left_fingers = self.left_fingers
        new_player.right_fingers = self.right_fingers
        return new_player


class HumanPlayer(Player):
    def __init__(self, name: str):
        super().__init__(name)

    def setup(self):
        super().setup()

    def decide_action(self, game: GameState) -> Action:
        action = None
        while action is None:
            c = input("Enter 1 to tap. Enter 2 to redistribute: ")
            if c == "1":
                source_hand = input("Which hand to use (L for left, R for right)? ")
                if source_hand != "L" and source_hand != "R":
                    print("Invalid input. Try again")
                    continue
                dest_hand = input("Which hand to tap (L for left, R for right)? ")
                if dest_hand != "L" and dest_hand != "R":
                    print("Invalid input. Try again")
                    continue
                action = Tap(Hand(source_hand), Hand(dest_hand))
            elif c == "2":
                left_hand = input("How many fingers would you like on your left hand? ")
                right_hand = input(
                    "How many fingers would you like on your right hand? "
                )
                action = Redistribute(int(left_hand), int(right_hand))
            else:
                print("Invalid input. Try again")
                continue
        return action


class MinimaxPlayer(Player):
    def __init__(self, name: str = "Minimax", max_depth=8):
        self.max_depth = max_depth
        super().__init__(name)

    def setup(self):
        super().setup()

    def decide_action(self, state: GameState) -> Action:
        _, action = self.minimax(deepcopy(state), 0, True, -float("inf"), float("inf"))
        return action

    def minimax(
        self, state: GameState, depth: int, maximizing: bool, alpha: int, beta: int
    ):
        if self._is_leaf(state, depth):
            return self._evaluate_state(state, depth), None

        child_sa_pairs = state.generate_child_sa_pairs()

        if maximizing:
            best_val = -float("inf")
            best_action = None
            for candidate_state, action in child_sa_pairs:
                value, _ = self.minimax(candidate_state, depth + 1, False, alpha, beta)
                if value > best_val:
                    best_val = value
                    best_action = action
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
            return best_val, best_action
        else:
            best_val = float("inf")
            best_action = None
            for candidate_state, action in child_sa_pairs:
                value, _ = self.minimax(candidate_state, depth + 1, True, alpha, beta)
                if value < best_val:
                    best_val = value
                    best_action = action
                beta = min(beta, best_val)
                if beta <= alpha:
                    break
            return best_val, best_action

    def _is_leaf(self, state: GameState, depth: int):
        if state.decide_winner() is not None:
            return True
        elif depth == self.max_depth:
            return True
        else:
            return False

    def _evaluate_state(self, state: GameState, iters_in_future: int):
        winner = state.decide_winner()
        if winner is None:
            return 0
        elif winner.name == self.name:
            return self.max_depth - iters_in_future + 2
        else:
            return -1 * (self.max_depth - iters_in_future + 2)


class TDPlayer(Player):
    neutral_reward = 0
    loss_reward = -1
    win_reward = 1

    def __init__(self, name: str, state_dir: str = "", gamma=0.9, lambd=0.3):
        self.gamma = gamma
        self.lambd = lambd
        self.state = None
        self.state_dir = state_dir
        self.latest_sa: Tuple[Tuple, Action] = None

        self.init_state()
        super().__init__(name)

    def setup(self):
        self.latest_sa = None
        self.set_episode_count(self.num_episodes_experienced + 1)
        self.reset_eligiblity_traces()
        super().setup()

    def process_results(self, game_state: GameState):
        winner = game_state.decide_winner()
        if winner.name == self.name:
            self.update_Q_E(TDPlayer.win_reward, game_state, None)
        else:
            self.update_Q_E(TDPlayer.loss_reward, game_state, None)
        super().process_results(game_state)

    def take_action(self, game_state: GameState, action: Action):
        self.update_state(game_state, action)
        super().take_action(game_state, action)

    def update_state(self, game_state: GameState, action: Action):
        parsed_state = self.parse_state(game_state)
        if self.latest_sa is not None:
            self.update_Q_E(TDPlayer.neutral_reward, parsed_state, action)
        self.latest_sa = (parsed_state, action)
        self.sa_visit_counts[parsed_state][action] += 1

    def decide_action(self, game_state: GameState) -> Action:
        parsed_state = self.parse_state(game_state)
        actions = self.Q[parsed_state].keys()
        return max(actions, key=lambda a: self.score_action(parsed_state, a))

    def update_Q_E(self, reward: int, new_game_state: Tuple, new_action: Action):
        old_state = self.latest_sa[0]
        old_action = self.latest_sa[1]
        old_q = self.get_q_value_of_sa(old_state, old_action)
        new_q = 0
        if new_action:
            new_q = self.get_q_value_of_sa(new_game_state, new_action)

        delta = reward + (self.gamma * new_q) - old_q
        self.E[old_state][old_action] += 1

        for state, actions_dict in self.Q.items():
            for action in actions_dict.keys():
                E = self.E[state][action]
                self.Q[state][action] += self.get_alpha(state, action) * delta * E
                self.E[state][action] = self.gamma * self.lambd * E

    def score_action(self, game_state: Tuple, action: Action) -> float:
        u = self.calculate_exploration_bonus(game_state, action)
        q = self.get_q_value_of_sa(game_state, action)
        return q + u

    def calculate_exploration_bonus(self, state: Tuple, action: Action) -> float:
        visit_count = self.sa_visit_counts[state][action]
        t = self.num_episodes_experienced
        return (
            float("inf")
            if visit_count == 0
            else math.sqrt(2 * math.log(t) / visit_count)
        )

    def get_q_value_of_sa(self, game_state: Tuple, action: Action) -> float:
        return self.Q[game_state][action]

    def store_state(self):
        state_path = self.state_dir + "/state.pkl"
        if not os.path.exists(self.state_dir):
            os.mkdir(self.state_dir)
            open(state_path, "a").close()
        with open(state_path, "wb") as state_file:
            pickle.dump(self.state, state_file)

    def init_state(self):
        if not os.path.exists(self.state_dir) or self.state_dir == "":
            self.state = TDPlayer.initialize_state()
        else:
            state_path = self.state_dir + "/state.pkl"
            with open(state_path, "rb") as state_file:
                self.state = pickle.load(state_file)

    def reset_eligiblity_traces(self):
        for state in self.E.keys():
            for action in self.E[state].keys():
                self.E[state][action] = 0

    def get_alpha(self, state, action):
        counts = max(self.sa_visit_counts[state][action], 1)
        return 1 / counts

    @staticmethod
    def initialize_state():
        """
        Initialize the agent's state variables
        """
        possible_finger_configs = list(itertools.product(range(5), range(5)))
        state = {}
        state["Q"] = {}
        state["E"] = {}
        state["sa_visit_counts"] = {}
        state["num_episodes_experienced"] = 0

        for game_state in itertools.product(
            possible_finger_configs, possible_finger_configs
        ):
            p1_fingers, p2_fingers = game_state
            p1 = Player("p1")
            p2 = Player("p2")

            p1.left_fingers = p1_fingers[0]
            p1.right_fingers = p1_fingers[1]
            p2.left_fingers = p2_fingers[0]
            p2.right_fingers = p2_fingers[1]

            valid_actions = p1.generate_valid_actions(p2)

            for action in valid_actions:
                if game_state not in state["E"]:
                    state["E"][game_state] = {}
                if game_state not in state["Q"]:
                    state["Q"][game_state] = {}
                if game_state not in state["sa_visit_counts"]:
                    state["sa_visit_counts"][game_state] = {}
                state["Q"][game_state][action] = 0
                state["E"][game_state][action] = 0
                state["sa_visit_counts"][game_state][action] = 0

        return state

    @property
    def Q(self):
        return self.state["Q"]

    @property
    def E(self):
        return self.state["E"]

    @property
    def sa_visit_counts(self):
        return self.state["sa_visit_counts"]

    @property
    def num_episodes_experienced(self):
        return self.state["num_episodes_experienced"]

    def set_episode_count(self, n):
        self.state["num_episodes_experienced"] = n


class DynaPlayer(Player):
    def __init__(self, name: str = "Dyna"):
        self.transition_counts = DynaPlayer.initialize_transition_counts()
        self.long_term_agent = TDPlayer(name, f"./{name}_long_term_agent_state")
        self.short_term_agent = SimTDPlayer(
            name, self.transition_counts, self.long_term_agent.Q
        )
        super().__init__(name)

    def setup(self):
        self.long_term_agent.setup()
        self.short_term_agent.setup()
        super().setup()

    def process_results(self, game_state: GameState):
        prev_state, prev_action = self.long_term_agent.latest_sa
        parsed_state = self.parse_state(game_state)
        self.transition_counts[prev_state][prev_action][parsed_state] += 1
        self.long_term_agent.process_results(game_state)
        super().process_results(game_state)

    def take_action(self, game_state: GameState, action: Action):
        parsed_state = self.parse_state(game_state)
        if self.long_term_agent.latest_sa:
            prev_state, prev_action = self.long_term_agent.latest_sa
            self.transition_counts[prev_state][prev_action][parsed_state] += 1
        self.long_term_agent.update_state(game_state, action)
        super().take_action(game_state, action)

    def decide_action(self, state: GameState) -> Action:
        self.short_term_agent.simulate_episodes(state, 50)
        actions = self.generate_valid_actions(state.inactive_player)
        parsed_state = self.parse_state(state)
        return max(actions, key=lambda a: self.score_action(parsed_state, a))

    def score_action(self, parsed_state: Tuple, action: Action):
        q = self.short_term_agent.get_q_value_of_sa(parsed_state, action)
        u = self.long_term_agent.calculate_exploration_bonus(parsed_state, action)
        return q + u

    @staticmethod
    def initialize_transition_counts():
        """
        Initialize the agent's transition counts
        """
        possible_finger_configs = list(itertools.product(range(5), range(5)))
        transition_counts = {}

        for state in itertools.product(
            possible_finger_configs, possible_finger_configs
        ):
            p1_fingers, p2_fingers = state
            p1 = Player("p1")
            p2 = Player("p2")

            p1.left_fingers = p1_fingers[0]
            p1.right_fingers = p1_fingers[1]
            p2.left_fingers = p2_fingers[0]
            p2.right_fingers = p2_fingers[1]

            valid_actions = p1.generate_valid_actions(p2)

            for action in valid_actions:
                if state not in transition_counts:
                    transition_counts[state] = {}
                transition_counts[state][action] = defaultdict(int)

        return transition_counts


class SimTDPlayer(TDPlayer):
    def __init__(
        self,
        name: str,
        transition_counts: dict,
        long_term_Q: dict,
        gamma=0.9,
        lambd=0.3,
    ):
        self.epsilon = 0.3
        self.transition_counts = transition_counts
        self.long_term_Q = long_term_Q
        return super().__init__(name, state_dir="", gamma=gamma, lambd=lambd)

    def setup(self):
        self.reset_state()
        return super().setup()

    def simulate_episodes(self, game_state: GameState, num_episodes: int):
        while num_episodes > 0:
            super().setup()
            latest_state = game_state
            latest_action = self.decide_action(game_state)
            while True:
                parsed_state = self.parse_state(latest_state)
                self.latest_sa = (parsed_state, latest_action)
                self.sa_visit_counts[parsed_state][latest_action] += 1
                next_state = self.sample_state(latest_state, latest_action)
                winner = next_state.decide_winner()
                if winner is not None:
                    if winner.name == self.name:
                        self.update_Q_E(TDPlayer.win_reward, parsed_state, None)
                    else:
                        self.update_Q_E(TDPlayer.loss_reward, parsed_state, None)
                    break
                next_action = self.decide_action(next_state)
                self.update_Q_E(
                    TDPlayer.neutral_reward, self.parse_state(next_state), next_action
                )
                latest_state, latest_action = next_state, next_action
            num_episodes -= 1

    def sample_state(self, game_state: GameState, action) -> GameState:
        parsed_state = self.parse_state(game_state)
        states = list(self.transition_counts[parsed_state][action].keys())
        if len(states) == 0:
            random_opponent_active_state = random.choice(
                game_state.generate_child_sa_pairs()
            )[0]
            return random.choice(
                random_opponent_active_state.generate_child_sa_pairs()
            )[0]
        counts = list(self.transition_counts[parsed_state][action].values())
        next_state_finger_tup = random.choices(states, weights=counts, k=1)[0]
        next_state = game_state.create_copy_with_diff_fingers(*next_state_finger_tup)
        return next_state

    def decide_action(self, game_state: GameState) -> Action:
        if random.uniform(0, 1) < self.epsilon:
            parsed_state = self.parse_state(game_state)
            return random.choice(list(self.Q[parsed_state].keys()))
        else:
            return super().decide_action(game_state)

    def calculate_exploration_bonus(self, state: Tuple, action: Action) -> float:
        return 0

    def get_q_value_of_sa(self, state, action) -> float:
        long_term = self.long_term_Q[state][action]
        short_term = self.Q[state][action]
        return long_term + short_term

    def reset_state(self):
        for state in self.Q.keys():
            for action in self.Q[state].keys():
                self.Q[state][action] = 0
                self.sa_visit_counts[state][action] = 0
        self.set_episode_count(0)
