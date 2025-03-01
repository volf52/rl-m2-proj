from typing import Protocol, Self, TypeAlias, TypeVar
from typing_extensions import override

from reinfproj.utils.base_env import BaseEnv
import gymnasium as gym
import numpy as np

from reinfproj.utils.state_action import TAction, TState

Int64Env = gym.Env[np.int64, np.int64]


class IntHolder:
    __val: int

    def __init__(self, val: int) -> None:
        self.__val = val

    def to_int(self) -> int:
        return self.__val

    @classmethod
    def from_int(cls, i: int) -> Self:
        return cls(i)

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self.__val}>"


class IntableState(TState, Protocol):
    def to_int(self) -> int: ...
    @classmethod
    def from_int(cls, i: int) -> Self: ...


class DiscreteState(IntableState, Protocol):
    @classmethod
    def total(cls) -> int: ...

    @classmethod
    @override
    def get_shape(cls, n_actions: int) -> tuple[int, ...]:
        return (cls.total(), n_actions)

    @override
    def get_slice(self, action: TAction | None) -> tuple[int, int | slice]:
        i = self.to_int()

        if action is None:
            return (i, slice(None))

        return (i, action.to_int())


def gym_to_discrete_state(cls: Int64Env):
    obs_space = cls.observation_space
    act_space = cls.action_space

    assert isinstance(obs_space, gym.spaces.Discrete), "Expected discrete state space"
    assert isinstance(act_space, gym.spaces.Discrete), "Expected discrete action space"

    class DiscreteGymState(IntHolder, DiscreteState):
        @override
        @classmethod
        def total(cls) -> int:
            return obs_space.n.item()

    class DiscreteGymAction(IntHolder, TAction):
        pass

    return DiscreteGymState, DiscreteGymAction


DEnv: TypeAlias = gym.Env[np.int64, np.int64]

State = TypeVar("State", bound=IntableState, covariant=True)
Action = TypeVar("Action", bound=TAction)


class DiscreteGymEnv(BaseEnv[State, Action]):
    env: DEnv
    t_action: type[Action]
    t_state: type[State]

    _possible_actions: list[Action]
    _n_actions: int

    def __init__(
        self,
        env: Int64Env,
        t_state: type[State],
        t_action: type[Action],
    ) -> None:
        self.env = env
        self.t_action = t_action
        self.t_state = t_state

        assert isinstance(env.action_space, gym.spaces.Discrete), (
            "require Discrete action space"
        )

        self._n_actions = env.action_space.n.item()
        self._possible_actions = [t_action.from_int(i) for i in range(self._n_actions)]

    @override
    def step(self, action: Action) -> tuple[State, float, bool, bool]:
        obs, reward, terminated, truncated, _ = self.env.step(np.int64(action.to_int()))
        self.t_state

        return self.t_state.from_int(int(obs)), float(reward), terminated, truncated

    @override
    def reset(self) -> State:
        obs, _ = self.env.reset()

        return self.t_state.from_int(int(obs))

    @override
    def get_possible_actions(self) -> list[Action]:
        return self._possible_actions

    @override
    def get_random_action(self, rng: np.random.Generator | None = None) -> Action:
        i: int
        if rng is None:
            i = self.env.action_space.sample().item()
        else:
            i = rng.choice(self._n_actions)

        return self.t_action.from_int(i)

    @override
    def n_actions(self) -> int:
        return self._n_actions

    def destroy(self):
        self.env.close()
