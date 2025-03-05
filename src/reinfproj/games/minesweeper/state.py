import math
from queue import Queue
import random
from typing import Literal, TypeAlias

from reinfproj.games.minesweeper import events
from reinfproj.games.minesweeper.events import InputEvent, OutputEvent
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

    __dug: set[Position]
    __num_flags: int
    __num_bombs: int
    __first_click: bool

    DIFF_TO_BD: dict[MsDifficulty, float] = {"Easy": 0.08, "Medium": 0.11, "Hard": 0.14}

    def __init__(
        self, cfg: MinesweeperCfg, *, difficulty: MsDifficulty = "Easy"
    ) -> None:
        self.cfg = cfg
        self.difficulty = difficulty
        self.__first_click = True
        self.__num_flags = 0

        self.reset()

    def reset(self):
        self.is_over = False
        self.num_moves = 0
        self.grid, self.__num_bombs = MinesweeperState.init_grid(
            self.cfg, self.difficulty
        )
        self.events = Queue()
        self.__dug = set()

    def tick(self):
        if self.is_over:
            return events.Nothing()

        while not self.events.empty():
            ev = self.events.get()
            out_ev = self.process_event(ev)
            return out_ev

        return None

    @staticmethod
    def init_grid(cfg: MinesweeperCfg, difficulty: MsDifficulty) -> tuple[MsGrid, int]:
        grid: MsGrid = [
            [Tile((col * cfg.TILESIZE, row * cfg.TILESIZE)) for row in range(cfg.WIDTH)]
            for col in range(cfg.HEIGHT)
        ]

        positions = [(j, i) for i in range(cfg.ROWS) for j in range(cfg.COLS)]
        random.shuffle(positions)

        num_bombs = math.ceil(MinesweeperState.DIFF_TO_BD[difficulty] * len(positions))
        bomb_positions = random.sample(positions, k=num_bombs)
        # bomb_positions = [(1, 0), (1, 1), (1, 2), (10, 10)]

        valid_ii_range = range(cfg.COLS)
        valid_jj_range = range(cfg.ROWS)

        for j, i in bomb_positions:
            grid[j][i].type_ = "mine"

            for jj in range(j - 1, j + 2):
                for ii in range(i - 1, i + 2):
                    if (ii, jj) == (i, j) or grid[jj][ii].type_ == "mine":
                        pass
                    elif ii in valid_ii_range and jj in valid_jj_range:
                        grid[jj][ii].num_bombs += 1

        return grid, num_bombs

    def process_event(self, ev: InputEvent) -> OutputEvent:
        match ev:
            case events.Clicked():
                return self.click((ev.row, ev.col))

            case events.Flagged():
                return self.click((ev.row, ev.col), flag=True)

    def click(self, pos: Position, flag: bool = False) -> OutputEvent:
        # print("flags", self.__num_flags, "bombs", self.__num_bombs)
        tile = self.grid[pos[0]][pos[1]]
        if tile.revealed:
            return events.AlreadyRevealed()

        self.num_moves += 1
        ev: OutputEvent = events.Nothing()

        if flag:
            already_flagged = tile.flag()
            if tile.flagged:
                self.__num_flags += 1
            else:
                self.__num_flags -= 1

            self.is_over = self.check_ended()
            if not self.is_over and already_flagged:
                return events.AlreadyFlagged()

        else:
            has_exploded = not self.dig(pos)
            if has_exploded and self.__first_click:
                self.grid, self.__num_bombs = MinesweeperState.init_grid(
                    self.cfg, self.difficulty
                )
                has_exploded = not self.dig(pos)
                self.__first_click = False

                if has_exploded:
                    self.is_over = True
                    self.end_game_reveal()

                    return events.Exploded(self.num_moves)

            if not has_exploded:
                self.is_over = self.check_ended()

        if self.is_over:
            self.end_game_reveal()
            if self.check_win():
                ev = events.Win(self.num_moves)
            else:
                ev = events.Lost(self.num_moves)

        return ev

    def check_ended(self):
        if self.__num_flags >= self.__num_bombs:
            return True
        for col in self.grid:
            for tile in col:
                if tile.type_ == "normal" and not tile.flagged and not tile.revealed:
                    return False

        return True

    def check_win(self):
        for col in self.grid:
            for tile in col:
                if tile.type_ == "exploded":
                    return False
                if tile.type_ == "mine" and not tile.flagged:
                    return False

                if tile.type_ == "normal" and tile.flagged:
                    return False

        return True

    def dig(self, pos: Position):
        self.__dug.add(pos)
        r, c = pos

        tile = self.grid[r][c]

        if tile.type_ == "mine":
            tile.type_ = "exploded"
            tile.revealed = True
            return False

        if tile.type_ == "normal" and tile.num_bombs > 0:
            # found a clue
            tile.revealed = True
            return True

        tile.revealed = True
        for row in range(max(0, r - 1), min(self.cfg.ROWS - 1, r + 1) + 1):
            for col in range(max(0, c - 1), min(self.cfg.COLS - 1, c + 1) + 1):
                if (row, col) not in self.__dug:
                    _ = self.dig((row, col))

        return True

    def end_game_reveal(self):
        for col in self.grid:
            for tile in col:
                tile.end_game_reveal()
