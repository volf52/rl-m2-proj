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


@dataclass
class Lost:
    num_moves: int


@dataclass
class Nothing:
    pass


@dataclass
class AlreadyFlagged:
    pass


@dataclass
class AlreadyRevealed:
    pass


OutputEvent: TypeAlias = (
    Win | Exploded | Lost | Nothing | AlreadyFlagged | AlreadyRevealed
)
