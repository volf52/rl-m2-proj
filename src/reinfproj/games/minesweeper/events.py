from dataclasses import dataclass
from typing import TypeAlias


@dataclass
class Clicked:
    row: int
    col: int


@dataclass
class Flagged:
    row: int
    col: int


InputEvent: TypeAlias = Clicked | Flagged


@dataclass
class Win:
    num_moves: int


@dataclass
class Exploded:
    num_moves: int


OutputEvent: TypeAlias = Win | Exploded
