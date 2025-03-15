import sys
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from reinfproj.games.ludo.ludo_obs import FlatStateArr, LudoObs
from reinfproj.games.ludo.ludo_macro_action import (
    MacroAction,
    choose_piece_for_macro,
    get_feasible_macro_actions,
)
from reinfproj.games.ludo.agent import LudoAgent
from collections import deque

Transition = tuple[FlatStateArr, int, float, FlatStateArr, bool]


class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.memory: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: FlatStateArr,
        action: int,
        reward: float,
        next_state: FlatStateArr,
        done: bool,
    ):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int = 32):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class MacroDQN(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 9):
        super().__init__()

        self.fc1: nn.Module = nn.Linear(input_dim, 64)
        self.fc2: nn.Module = nn.Linear(64, 64)
        self.out: nn.Module = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x: torch.Tensor = self.out(x)

        return x


class DQNAgent(LudoAgent):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 9,
        lr: float = 1e-3,
        buffer_capacity: int = 10000,
        device: torch.device | None = None,
    ):
        device = torch.device("cpu") if device is None else device
        self.device: torch.device = device

        self.network: MacroDQN = MacroDQN(input_dim, output_dim).to(device)
        self.target_network: MacroDQN = MacroDQN(input_dim, output_dim).to(device)
        _ = self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer: optim.Optimizer = optim.AdamW(
            self.network.parameters(), lr=lr, amsgrad=True
        )
        self.replay_mem: ReplayMemory = ReplayMemory(buffer_capacity)

    @torch.no_grad
    def get_action(self, obs: LudoObs) -> int:
        return self.act(obs, 0.0)

    def act(self, obs: LudoObs, epsilon: float) -> int:
        """
        Given an observation (LudoObs), choose a macro action and then map it to a piece index.
        Returns (macro_action, piece_idx).
        """
        state = obs.encode_macro_state()  # shape (16,)
        feasible = get_feasible_macro_actions(obs)
        if len(feasible) == 0:
            chosen_macro = MacroAction.NORMAL
        else:
            if np.random.rand() < epsilon:
                chosen_macro = random.choice(feasible)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = (
                    self.network(state_tensor).detach().cpu().numpy().flatten()
                )  # shape (9,)
                # Mask out non-feasible actions.
                mask = np.full(9, -1e9, dtype=np.float32)
                for act in feasible:
                    mask[int(act)] = 0.0
                q_values = q_values + mask
                chosen_macro = MacroAction(int(np.argmax(q_values)))
        piece = choose_piece_for_macro(obs, chosen_macro)
        if piece == -1:
            piece = choose_piece_for_macro(obs, MacroAction.NORMAL)
        return piece

    def update(
        self, batch_size: int, gamma: float, device: torch.device
    ) -> float | None:
        if len(self.replay_mem) < batch_size:
            return None

        s, a, r, s2, d = self.replay_mem.sample(batch_size)

        s_tensor = torch.from_numpy(s).to(torch.float32).to(device)
        a_tensor = torch.from_numpy(a).long().unsqueeze(1).to(device)
        r_tensor = torch.from_numpy(r).to(torch.float32).to(device)
        s2_tensor = torch.from_numpy(s2).to(torch.float32).to(device)
        d_tensor = torch.from_numpy(d).to(torch.short).to(device)

        q_vals = self.network(s_tensor).gather(1, a_tensor).squeeze(1)
        # print(q_vals.shape, q_vals.dtype)
        # print(s2, type(s2), s2.shape, s2.dtype)
        # print(s2_tensor.cpu())

        with torch.no_grad():
            q_next = self.target_network(s2_tensor).max(1).values
            # print(q_next.shape, q_next.dtype)

        target_q = r_tensor + gamma * (1 - d_tensor) * q_next
        # print(q_next.shape, q_vals.shape, target_q.shape, r_tensor.shape)

        loss = F.mse_loss(q_vals, target_q)
        # print(loss.dtype)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.cpu().item()

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())
