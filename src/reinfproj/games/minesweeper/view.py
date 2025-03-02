import pygame

from reinfproj.games.minesweeper.config import MinesweeperCfg
from reinfproj.games.minesweeper.state import MinesweeperState


class MinesweeperView:
    screen: pygame.Surface
    font: pygame.font.Font
    clock: pygame.time.Clock

    cfg: MinesweeperCfg

    def __init__(self, cfg: MinesweeperCfg):
        pygame.display.set_caption("Minesweeper")

        self.screen = pygame.display.set_mode(
            (cfg.WIDTH, cfg.HEIGHT),
        )

        pygame.font.init()
        self.font = pygame.font.SysFont(None, size=32, bold=True)
        self.clock = pygame.time.Clock()
        self.cfg = cfg

    def open(self):
        if not pygame.get_init():
            _ = pygame.init()

    def close(self):
        pygame.quit()

    def render(self, state: MinesweeperState):
        _ = self.clock.tick(60)

        _ = self.screen.fill(self.cfg.BGCOLOR)

        # Draw grid
        for row in state.grid:
            for tile in row:
                tile.render(self.screen, state.cfg)

        # Draw open areas / closed areas
        # Draw mines
        # Draw numbers
        # Show score etc

        pygame.display.flip()
