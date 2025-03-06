from typing import Any, Literal
from typing_extensions import override
import gymnasium as gym
import numpy as np

from reinfproj.games.minesweeper import Minesweeper
from reinfproj.games.minesweeper.events import (
    OutputEvent,
    Nothing,
    Win,
    Lost,
    Exploded,
    AlreadyFlagged,
    AlreadyRevealed,
    Flagged,
    Clicked,
)
from reinfproj.utils.discrete_gym_env import (
    DiscreteGymEnv,
    gym_to_discrete_state,
)

RenderMode = Literal["human"] | None


REWARD_LIST = [0.0, 10.0, -10.0, -10.0, -2.0, -4.0]


class MinesweeperEnv(gym.Env[np.int64, np.int64]):
    def __init__(self, render_mode: RenderMode = "human"):
        self.game: Minesweeper = Minesweeper(show_window=render_mode == "human")

        self.action_space: gym.spaces.Space[np.int64] = gym.spaces.Discrete(
            15 * 15 * 2
        )  # Rows x Cols x right/left click
        self.observation_space: gym.spaces.Space[np.int64] = gym.spaces.Discrete(6)

    @staticmethod
    def _event_to_observation(ev: OutputEvent | None) -> int:
        match ev:
            case None:
                return 0
            case Nothing():
                return 0
            case Win():
                return 1
            case Lost():
                return 2
            case Exploded():
                return 3
            case AlreadyFlagged():
                return 4
            case AlreadyRevealed():
                return 5

    @override
    def reset(self) -> tuple[np.int64, dict]:
        ev = self.game.state.reset()
        obs = MinesweeperEnv._event_to_observation(ev)
        obs = np.int64(obs)

        return obs, {}

    @override
    def step(
        self, action: np.int64
    ) -> tuple[np.int64, float, bool, bool, dict[str, Any]]:
        act = int(action.item())

        should_flag = act >= 175
        if should_flag:
            act -= 175

        col, row = divmod(act, 15)
        # print(row, col, should_flag)

        if should_flag:
            self.game.state.events.put(Flagged(row, col))
        else:
            self.game.state.events.put(Clicked(row, col))

        out_ev = self.game.tick()
        obs = MinesweeperEnv._event_to_observation(out_ev)
        reward = REWARD_LIST[obs]
        terminated = self.game.state.is_over
        truncated = self.game.state.num_moves >= 200
        info = {}

        return np.int64(obs), reward, terminated, truncated, info


def get_minesweeper_env(render_mode: RenderMode = None):
    env = MinesweeperEnv(render_mode=render_mode)
    MineweeperEnvState, MinesweeperEnvAction = gym_to_discrete_state(env)

    env_up = DiscreteGymEnv(env, MineweeperEnvState, MinesweeperEnvAction)

    return env_up, MineweeperEnvState, MinesweeperEnvAction
