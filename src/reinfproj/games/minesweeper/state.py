import math
from queue import Queue
import random
from typing import Literal, TypeAlias

from reinfproj.games.minesweeper import events
from reinfproj.games.minesweeper.events import InputEvent
from reinfproj.games.minesweeper.sprites import Tile
from reinfproj.games.minesweeper.types import Position
from reinfproj.games.minesweeper.config import MinesweeperCfg

MsGrid: TypeAlias = list[list[Tile]]

MsDifficulty: TypeAlias = Literal["Easy", "Medium", "Hard"]


class MinesweeperState:
    cfg: MinesweeperCfg
    grid: MsGrid
    difficulty: MsDifficulty
    is_over: bool
    num_moves: int
    events: Queue[InputEvent]

    DIFF_TO_BD: dict[MsDifficulty, float] = {"Easy": 0.05, "Medium": 0.08, "Hard": 0.13}

    BOMB: int = 65535

    def __init__(
        self, cfg: MinesweeperCfg, *, difficulty: MsDifficulty = "Easy"
    ) -> None:
        self.cfg = cfg
        self.difficulty = difficulty

        self.reset()

    def reset(self):
        self.is_over = False
        self.num_moves = 0
        self.grid = MinesweeperState.init_grid(self.cfg, self.difficulty)
        self.events = Queue()

    def tick(self):
        if self.is_over:
            return None

        while not self.events.empty():
            ev = self.events.get()
            exploded = self.process_event(ev)
            if exploded:
                return events.Exploded(self.num_moves)
            elif self.is_over:
                return events.Win(self.num_moves)

        return None

    @staticmethod
    def init_grid(cfg: MinesweeperCfg, difficulty: MsDifficulty) -> MsGrid:
        grid: MsGrid = [
            [Tile((col * cfg.TILESIZE, row * cfg.TILESIZE)) for row in range(cfg.WIDTH)]
            for col in range(cfg.HEIGHT)
        ]

        positions = [(j, i) for i in range(cfg.ROWS) for j in range(cfg.COLS)]
        random.shuffle(positions)

        num_bombs = math.ceil(MinesweeperState.DIFF_TO_BD[difficulty] * len(positions))
        bomb_positions = random.choices(positions, k=num_bombs)
        print("num bombs", num_bombs, len(positions))

        valid_ii_range = range(cfg.ROWS)
        valid_jj_range = range(cfg.COLS)

        for j, i in bomb_positions:
            grid[j][i].type_ = "mine"

            for jj in range(j - 1, j + 2):
                for ii in range(i - 1, i + 2):
                    if ii in valid_ii_range and jj in valid_jj_range:
                        grid[jj][ii].num_bombs += 1

        return grid

    def process_event(self, ev: InputEvent):
        match ev:
            case events.Clicked():
                exploded = self.click((ev.row, ev.col))

                return exploded

            case events.Flagged():
                _ = self.click((ev.row, ev.col), flag=True)

    def click(self, pos: Position, flag: bool = False):
        self.num_moves += 1
        tile = self.grid[pos[1]][pos[0]]
        has_exploded = False

        if flag:
            tile.flag()
        else:
            has_exploded = tile.reveal()

        if has_exploded:
            self.is_over = True

        return self.is_over

    def end_game_reveal(self):
        for col in self.grid:
            for tile in col:
                _ = tile.reveal(check_mine=False)
