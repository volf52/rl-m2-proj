from reinfproj.envs.minesweeper import get_minesweeper_env
from reinfproj.agents.sarsa import SarsaAgent, SarsaTrainingParams
import numpy as np


env, MinesweeperEnvState, MinesweeperEnvAction = get_minesweeper_env(
    # render_mode="human"
    render_mode=None
)

agent = SarsaAgent(env, MinesweeperEnvState, MinesweeperEnvAction)

train_params = SarsaTrainingParams(1e-3, 0.95, 1.0 / 2000.0, 2500)
info = agent.train(train_params)

np.save("./sarsa_minesweeper.npy", agent.qvals)
np.save("./sarsa_steps_per_episode.npy", info.steps_per_episode)
np.save("./sarsa_rewards_per_episode.npy", info.total_reward_per_episode)

# obs = env.reset()
#
# done = False
#
# while not done:
#     action = env.get_random_action()
#     obs, reward, done, _ = env.step(action)
#
#
# while True:
#     pass
