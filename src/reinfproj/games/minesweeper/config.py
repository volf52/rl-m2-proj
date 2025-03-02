from dataclasses import dataclass
from pathlib import Path

import pygame

ASSETS_DIR: Path = Path(__file__).parent.resolve() / "assets"

Color = tuple[int, int, int]


def load_file(name: str, tilesize: int) -> pygame.Surface:
    return pygame.transform.scale(
        pygame.image.load(ASSETS_DIR / name),
        (tilesize, tilesize),
    )


@dataclass
class MinesweeperCfg:
    WHITE: Color = (255, 255, 255)
    BLACK: Color = (0, 0, 0)
    DARKGREY: Color = (40, 40, 40)
    LIGHTGREY: Color = (100, 100, 100)
    GREEN: Color = (0, 255, 0)
    DARKGREEN: Color = (0, 200, 0)
    BLUE: Color = (0, 0, 255)
    RED: Color = (255, 0, 0)
    YELLOW: Color = (255, 255, 0)

    BGCOLOR: Color = DARKGREY

    TILESIZE: int = 32
    COLS: int = 15
    ROWS: int = 15
    AMOUNT_MINES: int = 5

    WIDTH: int = TILESIZE * ROWS
    HEIGHT: int = TILESIZE * COLS

    FPS: int = 60

    Tile1: pygame.Surface = load_file("Tile1.png", TILESIZE)
    Tile2: pygame.Surface = load_file("Tile2.png", TILESIZE)
    Tile3: pygame.Surface = load_file("Tile3.png", TILESIZE)
    Tile4: pygame.Surface = load_file("Tile4.png", TILESIZE)
    Tile5: pygame.Surface = load_file("Tile5.png", TILESIZE)
    Tile6: pygame.Surface = load_file("Tile6.png", TILESIZE)
    Tile7: pygame.Surface = load_file("Tile7.png", TILESIZE)
    Tile8: pygame.Surface = load_file("Tile8.png", TILESIZE)

    TileEmpty: pygame.Surface = load_file("TileEmpty.png", TILESIZE)
    TileExploded: pygame.Surface = load_file("TileExploded.png", TILESIZE)
    TileFlag: pygame.Surface = load_file("TileFlag.png", TILESIZE)
    TileMine: pygame.Surface = load_file("TileMine.png", TILESIZE)
    TileUnknown: pygame.Surface = load_file("TileUnknown.png", TILESIZE)
    TileNotMine: pygame.Surface = load_file("TileNotMine.png", TILESIZE)
