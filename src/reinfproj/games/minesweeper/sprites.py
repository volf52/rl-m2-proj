from typing import Literal
from reinfproj.games.minesweeper.config import MinesweeperCfg
from reinfproj.games.minesweeper.types import Position
import pygame

TileType = Literal["normal", "mine", "exploded", "notmine"]


class Tile:
    pos: Position
    revealed: bool
    flagged: bool
    num_bombs: int
    type_: TileType

    __was_already_flagged: bool

    def __init__(self, pos: Position):
        self.pos = pos
        self.revealed = False
        self.flagged = False
        self.num_bombs = 0
        self.type_ = "normal"
        self.__was_already_flagged = False

    def render(self, surf: pygame.Surface, cfg: MinesweeperCfg):
        to_blit: pygame.Surface

        match (self.type_, self.num_bombs, self.revealed, self.flagged):
            case (_, _, False, True):
                to_blit = cfg.TileFlag
            case ("normal", _, False, _):
                to_blit = cfg.TileUnknown
            case ("normal", 1, True, _):
                to_blit = cfg.Tile1
            case ("normal", 1, True, _):
                to_blit = cfg.Tile1
            case ("normal", 2, True, _):
                to_blit = cfg.Tile2
            case ("normal", 3, True, _):
                to_blit = cfg.Tile3
            case ("normal", 4, True, _):
                to_blit = cfg.Tile4
            case ("normal", 5, True, _):
                to_blit = cfg.Tile5
            case ("normal", 6, True, _):
                to_blit = cfg.Tile6
            case ("normal", 7, True, _):
                to_blit = cfg.Tile7
            case ("normal", 8, True, _):
                to_blit = cfg.Tile8
            case ("normal", _, True, _):
                to_blit = cfg.TileEmpty

            case ("mine", _, True, _):
                to_blit = cfg.TileMine
            case ("mine", _, False, _):
                to_blit = cfg.TileUnknown
            case ("notmine", _, _, _):
                to_blit = cfg.TileNotMine

            case ("exploded", _, _, _):
                to_blit = cfg.TileExploded

        _ = surf.blit(to_blit, self.pos)

    def flag(self):
        self.flagged = not self.flagged
        old_was_already_flagged = self.__was_already_flagged
        self.__was_already_flagged = True

        return old_was_already_flagged

    def end_game_reveal(self):
        self.revealed = True
        if self.flagged and self.type_ != "mine":
            self.type_ = "notmine"
