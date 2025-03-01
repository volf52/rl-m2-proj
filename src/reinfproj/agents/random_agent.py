from typing import TypeVar
from typing_extensions import override

from reinfproj.utils.agent import Agent
from reinfproj.utils.base_env import BaseEnv
from reinfproj.utils.state_action import TAction, TState

State = TypeVar("State", bound=TState)
Action = TypeVar("Action", bound=TAction)


class RandomAgent(Agent[None, None, State, Action]):
    env: BaseEnv[State, Action]

    def __init__(self, env: BaseEnv[State, Action]) -> None:
        self.env = env

    @override
    def get_action(self, state: TState) -> Action:
        return self.env.get_random_action()

    @override
    def train(self, params: None):
        pass

    @override
    def reset(self) -> None:
        pass
