from enum import Enum
from typing import NamedTuple, Union


class Hand(Enum):
    LEFT = "L"
    RIGHT = "R"


class Tap(NamedTuple):
    src_hand: Hand
    dest_hand: Hand


class Redistribute(NamedTuple):
    new_left_fingers: int
    new_right_fingers: int


Action = Union[Tap, Redistribute]
