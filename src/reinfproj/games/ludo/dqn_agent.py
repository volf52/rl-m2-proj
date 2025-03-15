from reinfproj.games.ludo.agent import LudoAgent

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import numpy as np

from reinfproj.games.ludo.wrapper import LudoObs


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        """
        input_dim: number of features in your state representation
        hidden_dim: size of hidden layer(s)
        Outputs: Q-values for 4 piece moves (action = 0..3).
        """
        super().__init__()
        self.net: nn.Module = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # 4 possible pieces
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape [batch_size, input_dim]
        returns: shape [batch_size, 4]  (Q-values for each piece)
        """
        return self.net(x)


class ReplayMemory:
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int = 64):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class LudoDQN(LudoAgent):
    def __init__(
        self,
        state_dim: int = 4 * 59 + 1,  # 4*59 board positions + 1 for dice
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.gamma = gamma

        # Q-Network and target Q-network
        self.q_net = QNetwork(state_dim).to(device)
        self.target_q_net = QNetwork(state_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Simple replay memory
        self.memory = ReplayMemory()

        # Steps for scheduling
        self.learn_steps = 0
        self.target_update_interval = 1000  # how often to sync target net

    def get_state_tensor(self, dice: int, state_arr: np.ndarray) -> torch.Tensor:
        """
        Flatten the board state, append dice as last feature.
        Return shape [1, state_dim].
        """
        # state_arr.shape = (4,59) -> flatten -> length 236
        # plus 1 for dice => 237
        s = np.concatenate((state_arr.flatten(), [dice]))
        s_tensor = torch.tensor(s, dtype=torch.float32, device=self.device)
        return s_tensor.unsqueeze(0)  # shape [1, state_dim]

    def act(
        self, dice: int, state_arr: np.ndarray, move_pieces: np.ndarray, epsilon: float
    ) -> int:
        """
        Epsilon-greedy action selection.
        - If no valid move exists, return -1.
        - Otherwise produce Q-values for [0..3], mask out invalid moves,
          and pick argmax or random according to epsilon.
        """
        if len(move_pieces) == 0:
            return -1  # no valid move

        # Epsilon check
        if random.random() < epsilon:
            # random valid piece from move_pieces
            return int(np.random.choice(move_pieces))

        # Greedy from Q-network
        state_t = self.get_state_tensor(dice, state_arr)
        with torch.no_grad():
            q_vals = self.q_net(state_t)  # shape [1, 4]
            q_vals = q_vals.squeeze(0)  # shape [4]

        # Mask out invalid piece indices
        # We'll set Q-values for invalid moves to a large negative
        mask = torch.full_like(q_vals, float("-inf"))
        for piece_id in move_pieces:
            mask[piece_id] = q_vals[piece_id]

        best_action = torch.argmax(mask).item()
        return int(best_action)

    def store_transition(
        self, state, action: int, reward: float, next_state, done: bool
    ):
        """
        Save (s, a, r, s', done) into replay memory.
        """
        self.memory.push(state, action, reward, next_state, done)

    def update(self, batch_size: int = 32):
        """
        Sample from replay memory, compute DQN loss, backprop, update target net occasionally.
        """
        if len(self.memory) < batch_size:
            return

        self.learn_steps += 1

        # Sample
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Convert to Tensors
        states_t = torch.from_numpy(states).to(torch.float32)
        actions_t = torch.from_numpy(actions).to(torch.long).view(-1, 1)
        rewards_t = torch.from_numpy(rewards).to(torch.float32).view(-1, 1)
        next_states_t = torch.from_numpy(next_states).to(torch.float32)
        dones_t = torch.from_numpy(dones).to(torch.long).view(-1, 1)

        # Q(s,a)
        q_vals = self.q_net(states_t).gather(1, actions_t)

        # Max_{a'} Q'(s', a')
        with torch.no_grad():
            next_q_vals = self.target_q_net(next_states_t)
            max_next_q_vals = next_q_vals.max(dim=1, keepdim=True)[0]
            # if done -> target is just reward
            target = rewards_t + (1 - dones_t) * self.gamma * max_next_q_vals

        loss = F.mse_loss(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.learn_steps % self.target_update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    @torch.no_grad
    def get_action(self, obs: LudoObs) -> int:
        return self.act(obs.dice, obs.state, obs.move_pieces, epsilon=0.0)
