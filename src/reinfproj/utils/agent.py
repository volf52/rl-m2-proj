from typing import Protocol, TypeVar
from reinfproj.utils.state_action import TAction, TState

State = TypeVar("State", bound=TState, contravariant=True)
Action = TypeVar("Action", bound=TAction, covariant=True)
TrainingParams = TypeVar("TrainingParams", contravariant=True)
TrainingResult = TypeVar("TrainingResult", covariant=True)


class Agent(Protocol[TrainingParams, TrainingResult, State, Action]):
    def get_action(self, state: State) -> Action: ...
    def train(self, params: TrainingParams) -> TrainingResult: ...
    def reset(self) -> None: ...
