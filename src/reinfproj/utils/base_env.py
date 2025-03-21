from typing import Protocol, TypeVar

from numpy.random import Generator
from reinfproj.utils.state_action import TAction, TState

State = TypeVar("State", bound=TState, covariant=True)
Action = TypeVar("Action", bound=TAction)


class BaseEnv(Protocol[State, Action]):
    def get_possible_actions(self) -> list[Action]: ...
    def get_random_action(self, rng: Generator | None = None) -> Action: ...
    def n_actions(self) -> int: ...
    def reset(self) -> State: ...
    def step(self, action: Action) -> tuple[State, float, bool, bool]: ...
