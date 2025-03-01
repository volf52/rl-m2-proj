import math
import random
from typing import Literal, TypeAlias
import numpy as np

Position: TypeAlias = tuple[int, int]
MsGrid: TypeAlias = np.ndarray[Position, np.dtype[np.uint16]]

MsDifficulty: TypeAlias = Literal["Easy", "Medium", "Hard"]


class MinesweeperState:
    size: int
    grid: MsGrid

    DIFF_TO_BD: dict[MsDifficulty, float] = {"Easy": 0.05, "Medium": 0.10, "Hard": 0.15}

    BOMB: int = 65535

    def __init__(self, size: int, *, difficulty: MsDifficulty = "Easy") -> None:
        self.size = size
        self.grid = MinesweeperState.init_grid(size, difficulty)

    @staticmethod
    def init_grid(size: int, difficulty: MsDifficulty) -> MsGrid:
        arr: MsGrid = np.zeros((size, size), dtype=np.uint16)

        # Place bombs based on difficulty
        possible_positions = [(i, j) for i in range(size) for j in range(size)]
        random.shuffle(possible_positions)

        k = math.ceil(size * size * MinesweeperState.DIFF_TO_BD[difficulty])
        bomb_positions = random.choices(possible_positions, k=k)

        for x, y in bomb_positions:
            arr[x, y] = MinesweeperState.BOMB

        return arr

    def reveal(self, pos: Position):
        pass
