import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

from reinfproj.games.dotsandboxes.env import FlatStateArr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.fc1: nn.Module = nn.Linear(input_dim, 128)
        self.fc2: nn.Module = nn.Linear(128, 128)
        self.out: nn.Module = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


Transition = tuple[FlatStateArr, int, float, FlatStateArr, bool]


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: FlatStateArr,
        action: int,
        reward: float,
        next_state: FlatStateArr,
        done: bool,
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = map(list, zip(*batch))

        states = np.stack(states)
        actions = np.array(actions, dtype=np.long)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.stack(next_states)
        dones = np.array(dones, dtype=np.long)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: torch.device,
        lr: float = 1e-3,
        buffer_capacity: int = 10000,
    ):
        self.device = device
        self.network = DQN(input_dim, output_dim).to(self.device)
        self.target_network = DQN(input_dim, output_dim).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr, amsgrad=True)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.output_dim = output_dim

    def act(self, state: FlatStateArr, legal_actions: list[int], epsilon: float) -> int:
        """
        Given a state (flat NumPy array) and the list of legal actions,
        select an action using ε–greedy policy.
        """
        if np.random.rand() < epsilon:
            return random.choice(legal_actions)

        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        q_values = self.network(state_tensor).detach().cpu().numpy().flatten()
        # Mask illegal actions by setting their Q-values to a very negative number.
        mask = np.full(self.output_dim, -1e9, dtype=np.float32)
        for a in legal_actions:
            mask[a] = 0.0
        q_values = q_values + mask
        return int(np.argmax(q_values))

    def update(self, batch_size: int, gamma: float) -> float | None:
        if len(self.replay_buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )
        states_tensor = torch.from_numpy(states).float().to(self.device)
        actions_tensor = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards_tensor = torch.from_numpy(rewards).float().to(self.device)
        next_states_tensor = torch.from_numpy(next_states).float().to(self.device)
        dones_tensor = torch.from_numpy(dones).long().to(self.device)

        q_vals = self.network(states_tensor).gather(1, actions_tensor).squeeze(1)
        with torch.no_grad():
            next_q_vals = self.target_network(next_states_tensor).max(dim=1).values

        target_q = rewards_tensor + gamma * (1 - dones_tensor) * next_q_vals
        loss = F.mse_loss(q_vals, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())
