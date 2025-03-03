import sys
import pygame
from reinfproj.games.minesweeper import events
from reinfproj.games.minesweeper.config import MinesweeperCfg
from reinfproj.games.minesweeper.state import MinesweeperState
from reinfproj.games.minesweeper.view import MinesweeperView


class Minesweeper:
    state: MinesweeperState
    view: MinesweeperView | None

    def __init__(self, show_window: bool = False):
        self.state = MinesweeperState(MinesweeperCfg())
        self.view = MinesweeperView(self.state.cfg) if show_window else None

    def open(self):
        if self.view is not None:
            self.view.open()

    def close(self):
        if self.view is not None:
            self.view.close()

    def tick(self):
        cond = self.state.tick()
        if self.view is not None:
            self.view.render(self.state)

        return cond

    def run_human_mode(self):
        assert self.view is not None, "requires show_window"

        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.close()
                    print(f"Score: {self.state.num_moves}")
                    sys.exit(0)

                if ev.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    print(mx, my)

                    mx //= self.state.cfg.TILESIZE
                    my //= self.state.cfg.TILESIZE

                    print(mx, my)

                    if ev.button == 1:  # click
                        self.state.events.put(events.Clicked(mx, my))
                    elif ev.button == 3:  # flag
                        self.state.events.put(events.Flagged(mx, my))

            out_ev = self.tick()
            match out_ev:
                case events.Win():
                    print(f"Score: {self.state.num_moves}")
                case events.Exploded():
                    print(f"Score: {self.state.num_moves}")
                    print("Exploded")
                case _:
                    pass
