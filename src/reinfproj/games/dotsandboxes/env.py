from typing import TypeAlias
import numpy as np
import pyspiel

StateArr: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float32]]
FlatStateArr: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float32]]
MovePiecesArr: TypeAlias = np.ndarray[tuple[int], np.dtype[np.uint16]]
PiecesArr: TypeAlias = np.ndarray[tuple[int], np.dtype[np.uint16]]
EnemyPiecesArr: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.uint16]]

ZERO_STATE: FlatStateArr = np.zeros(441, dtype=np.float32)


class DotsAndBoxesEnv:
    def __init__(self, n_rows_cols: int = 6):
        game_name = f"dots_and_boxes(num_rows={n_rows_cols},num_cols={n_rows_cols})"
        self.game: pyspiel.Game = pyspiel.load_game(game_name)
        self.state: pyspiel.State = self.game.new_initial_state()

    def reset(self):
        self.state = self.game.new_initial_state()
        return self.get_observation()

    def get_observation(self):
        # The observation is given as a tensor by OpenSpiel.
        # We cast it to a flat float32 NumPy array.
        obs_tensor = np.array(self.state.observation_tensor(), dtype=np.float32)
        return obs_tensor.flatten()

    def step(self, action: int):
        # If the action is illegal, we can either choose to penalize or ignore it.
        if action not in self.state.legal_actions():
            # Here we assign a negative reward and do not change the state.
            reward = -1.0
            done = self.state.is_terminal()
            return self.get_observation(), reward, done

        self.state.apply_action(action)
        reward = self.state.rewards()[-1]  # reward is provided by the game
        done = self.state.is_terminal()
        if done:
            return ZERO_STATE, reward, done

        return self.get_observation(), reward, done

    def legal_actions(self) -> list[int]:
        return self.state.legal_actions()

    def current_player(self) -> int:
        return self.state.current_player()
