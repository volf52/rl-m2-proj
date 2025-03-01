import gymnasium as gym
from reinfproj.agents.sarsa import SarsaAgent, SarsaTrainingParams
from reinfproj.utils.discrete_gym_env import (
    DiscreteGymEnv,
    Int64Env,
    gym_to_discrete_state,
)
import numpy as np

base_env: Int64Env = gym.make("Taxi-v3", max_episode_steps=500, render_mode=None)

TaxiState, TaxiAction = gym_to_discrete_state(base_env)


env = DiscreteGymEnv(base_env, TaxiState, TaxiAction)

done = False

state = env.reset()

agent = SarsaAgent(env, TaxiState, TaxiAction)

params = SarsaTrainingParams(0.8, 0.95, 1 / 800, 1500)

result = agent.train(params)

np.save("./taxi_sarsa_qvalues.npy", agent.qvals)
np.save("./taxi_sarsa_steps.npy", result.steps_per_episode)
np.save("./taxi_sarsa_rewards.npy", result.total_reward_per_episode)
