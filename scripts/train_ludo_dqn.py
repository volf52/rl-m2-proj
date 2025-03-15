import torch
import numpy as np
from tqdm import trange

from reinfproj.games.ludo.agent import RandomAgent
from reinfproj.games.ludo.wrapper import Ludo
from reinfproj.games.ludo.macro_dqn_agent import DQNAgent as MacroDQNAgent
from reinfproj.games.ludo.ludo_obs import LudoObs

NUM_EPISODES = 1000
BATCH_SIZE = 8
GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10  # episodes
BUFFER_CAPACITY = 10_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_multi_agent():
    env = Ludo(render_mode=None)

    # Create 4 agents—one for each player.
    neural = MacroDQNAgent(
        input_dim=16,
        output_dim=9,
        lr=LR,
        buffer_capacity=BUFFER_CAPACITY,
        device=DEVICE,
    )
    _ = neural.network.to(DEVICE)
    _ = neural.target_network.to(DEVICE)

    agents = [
        neural,
        RandomAgent(),
        RandomAgent(),
        RandomAgent(),
    ]

    epsilon = EPSILON_START

    total_losses = np.zeros(NUM_EPISODES, dtype=np.float32)
    total_rewards = np.zeros(NUM_EPISODES, dtype=np.float32)
    total_wins = 0
    no_feasible = 0

    for episode in trange(NUM_EPISODES):
        ep_loss: list[float] = []
        ep_reward: list[float] = []

        env.reset()
        done = False
        while not done:
            obs: LudoObs = env.observe()
            # The environment's current player index (0..3).
            current_player = env.game.current_player

            if current_player != 0:
                agent = agents[current_player]
                act = agent.get_action(obs)
                next_obs, _ = env.step(act, obs)
                done = next_obs.is_over()
            else:
                act = neural.act(obs, epsilon)
                next_obs, reward = env.step(act, obs)
                state_macro = obs.encode_macro_state()
                next_state_macro = next_obs.encode_macro_state()
                done = next_obs.is_over()

                if act != -1:
                    neural.replay_mem.push(
                        state_macro, int(act), reward or 0.0, next_state_macro, done
                    )
                    ep_reward.append(reward)
                else:
                    no_feasible += 1

                loss = neural.update(BATCH_SIZE, GAMMA, device=DEVICE)
                if loss is not None:
                    ep_loss.append(loss)

                if episode % TARGET_UPDATE_FREQ == 0:
                    neural.update_target()

                epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
                done = next_obs.is_over()
                if done and env.game.first_winner_was == 0:
                    total_wins += 1

        total_losses[episode] = np.mean(ep_loss)
        # print(ep_reward)
        total_rewards[episode] = np.sum(ep_reward)
        # print(f"Episode {episode}: reward: {total_rewards[episode]:0.4f}")

    env.close()
    return neural, total_losses, total_rewards, total_wins, no_feasible


neural, total_losses, total_rewards, total_wins, no_feasible = train_multi_agent()

np.save("total_losses.npy", total_losses)
np.save("total_rewards.npy", total_rewards)
torch.save(neural.network.state_dict(), "ludo_dqn.pth")

print("No feasible actions:", no_feasible)
print(f"Total wins: {total_wins} {total_wins / NUM_EPISODES * 100:.2f}%")
