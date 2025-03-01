import gymnasium as gym
from reinfproj.agents.sarsa import SarsaAgent, SarsaTrainingParams
from reinfproj.utils.discrete_gym_env import (
    DiscreteGymEnv,
    Int64Env,
    gym_to_discrete_state,
)
import numpy as np
import signal
import sys

base_env: Int64Env = gym.make("Taxi-v3", max_episode_steps=500, render_mode="human")

TaxiState, TaxiAction = gym_to_discrete_state(base_env)


env = DiscreteGymEnv(base_env, TaxiState, TaxiAction)

done = False

state = env.reset()

agent = SarsaAgent(env, TaxiState, TaxiAction)

agent.qvals = np.load("./taxi_sarsa_qvalues.npy")

while not done:
    action = agent.get_action(state)
    state, reward, terminated, truncated = env.step(action)
    done = terminated or truncated

print("Completed")


def on_quit(_sig, _frame):
    print("Bye bye")
    env.env.close()

    sys.exit(0)


signal.signal(signal.SIGINT, on_quit)
signal.pause()
