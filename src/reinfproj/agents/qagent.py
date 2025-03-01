from dataclasses import dataclass
from typing import TypeAlias, TypeVar
from typing_extensions import override
from reinfproj.utils.agent import Agent
from reinfproj.utils.base_env import BaseEnv
from reinfproj.utils.state_action import TAction, TState
from reinfproj.utils.types import FloatArr
import numpy as np
from tqdm.auto import trange


UIntArr: TypeAlias = np.ndarray[tuple[int], np.dtype[np.uint16]]
RewardArr: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float32]]


@dataclass
class QAgentTrainingParams:
    alpha: float
    gamma: float
    eps_decay_rate: float
    n_episodes: int


@dataclass
class QAgentTrainingResults:
    steps_per_episode: UIntArr
    total_reward_per_episode: RewardArr


State = TypeVar("State", bound=TState, contravariant=True)
Action = TypeVar("Action", bound=TAction, covariant=True)


class QAgent(Agent[QAgentTrainingParams, QAgentTrainingResults, State, Action]):
    qvals: FloatArr
    action_const: type[Action]
    env: BaseEnv[State, Action]

    def __init__(
        self,
        env: BaseEnv[State, Action],
        state_const: type[State] | State,
        act_const: type[Action],
    ):
        n_actions = env.n_actions()
        required_shape = state_const.get_shape(n_actions)

        self.env = env
        self.qvals = np.zeros(required_shape, dtype=np.float32)
        self.action_const = act_const

    @override
    def reset(self):
        self.qvals = np.zeros_like(self.qvals, dtype=np.float32)

    @override
    def get_action(self, state: State) -> Action:
        sl = state.get_slice(None)
        i = self.qvals[sl].argmax().item()
        act = self.action_const.from_int(i)

        return act

    @override
    def train(self, params: QAgentTrainingParams) -> QAgentTrainingResults:
        steps_pe: UIntArr = np.zeros(params.n_episodes, dtype=np.uint16)
        t_reward_pe: RewardArr = np.zeros(params.n_episodes, dtype=np.float32)

        epsilon = 1.0
        for i in trange(params.n_episodes):
            state = self.env.reset()

            done = False
            rng = np.random.default_rng()
            reward_for_ep = 0.0
            steps: int = 0

            while not done:
                act: Action
                if rng.random() < epsilon:
                    act = self.env.get_random_action(rng)
                else:
                    act = self.get_action(state)

                next_state, reward, terminated, truncated = self.env.step(act)

                match params.method:
                    case "qlearning":
                        curr_slice = state.get_slice(act)
                        next_slice = next_state.get_slice(None)
                        # print(curr_slice)
                        # print(next_slice)

                        self.qvals[curr_slice] += params.alpha * (
                            reward
                            + params.gamma * np.max(self.qvals[next_slice]).item()
                            - self.qvals[curr_slice]
                        )

                reward_for_ep += reward
                steps += 1
                done = terminated or truncated
                state = next_state

            epsilon = max(0, epsilon - params.eps_decay_rate)
            steps_pe[i] = steps
            t_reward_pe[i] = reward_for_ep

        return QAgentTrainingResults(steps_pe, t_reward_pe)
