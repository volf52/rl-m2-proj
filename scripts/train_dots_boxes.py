import random
from reinfproj.games.dotsandboxes.env import DotsAndBoxesEnv
from reinfproj.games.dotsandboxes.dqn_agent import DQNAgent
import numpy as np
import torch
from tqdm import trange


env = DotsAndBoxesEnv()

state = env.reset()

possible_actions = list(range(env.game.num_distinct_actions()))


# Hyperparameters
NUM_EPISODES = 1000
BATCH_SIZE = 4
GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10
BUFFER_CAPACITY = 10000


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    env = DotsAndBoxesEnv()
    obs = env.reset()
    input_dim = obs.shape[0]
    # Use the number of distinct actions from the game.
    num_actions = len(possible_actions)
    agent = DQNAgent(
        input_dim, num_actions, DEVICE, lr=LR, buffer_capacity=BUFFER_CAPACITY
    )

    epsilon = EPSILON_START
    episode_rewards: list[float] = []
    episode_losses: list[float] = []
    num_wins = [0, 0]

    for ep in trange(NUM_EPISODES):
        obs = env.reset()  # flat state vector
        total_reward = 0.0
        ep_losses: list[float] = []

        done = False
        while not done:
            legal = env.legal_actions()
            if env.current_player() == 1:
                act = random.choice(legal)
                obs, _, done = env.step(act)
            else:
                action = agent.act(obs, legal, epsilon)
                next_obs, reward, done = env.step(action)
                total_reward += reward
                agent.replay_buffer.push(obs, action, reward, next_obs, done)
                obs = next_obs
                loss = agent.update(BATCH_SIZE, GAMMA)
                if loss is not None:
                    ep_losses.append(loss)

                if done and reward > -1:
                    num_wins += 1

        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        episode_rewards.append(total_reward)
        episode_losses.append(np.mean(ep_losses))
        # print(f"Episode {ep}: reward {total_reward}")

    return agent, episode_rewards, episode_losses, num_wins


agent, rewards, losses, num_wins = train()
torch.save(agent.network.state_dict(), "dots_and_boxes_dqn.pth")
np.save("dots_and_boxes_rewards.npy", rewards)
np.save("dots_and_boxes_losses.npy", losses)

print("Training complete!")
