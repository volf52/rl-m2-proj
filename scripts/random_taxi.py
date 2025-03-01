import gymnasium as gym
from reinfproj.agents.random_agent import RandomAgent
from reinfproj.utils.discrete_gym_env import (
    DiscreteGymEnv,
    Int64Env,
    gym_to_discrete_state,
)

base_env: Int64Env = gym.make("Taxi-v3", max_episode_steps=500, render_mode="human")

TaxiState, TaxiAction = gym_to_discrete_state(base_env)


env = DiscreteGymEnv(base_env, TaxiState, TaxiAction)

done = False

state = env.reset()

agent = RandomAgent(env)

while not done:
    action = agent.get_action(state)

    state, reward, terminated, truncated = env.step(action)

    print(action, reward)
