from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(slots=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_mask: np.ndarray


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DQNAgent:
    def __init__(
        self,
        action_dim: int = 5,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 5000,
        target_update_interval: int = 100,
        memory_size: int = 20000,
    ):
        self.action_dim = action_dim
        self.state_dim = 10
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.target_update_interval = target_update_interval
        self.policy_net = QNetwork(self.state_dim, action_dim).to(DEVICE)
        self.target_net = QNetwork(self.state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory: deque[Transition] = deque(maxlen=memory_size)
        self.learn_steps = 0

    def encode_observation(self, observation: dict) -> np.ndarray:
        row, col = observation["position"]
        dst_row, dst_col = observation["destination"]
        adjacent = observation["adjacent_occupancy"]
        grid_size = max(observation["grid_size"] - 1, 1)
        return np.array(
            [
                row / grid_size,
                col / grid_size,
                dst_row / grid_size,
                dst_col / grid_size,
                float(observation["route_type"]),
                float(adjacent[0]),
                float(adjacent[1]),
                float(adjacent[2]),
                float(adjacent[3]),
                observation["num_agents"] / max(observation["grid_size"], 1),
            ],
            dtype=np.float32,
        )

    def select_action(self, observation: dict, action_mask: np.ndarray, *, evaluate: bool = False) -> int:
        valid_actions = np.flatnonzero(action_mask)
        if not evaluate and random.random() < self.epsilon:
            return int(random.choice(valid_actions))
        state = torch.tensor(self.encode_observation(observation), device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state).squeeze(0).cpu().numpy()
        q_values[action_mask == 0] = -np.inf
        return int(np.argmax(q_values))

    def remember(
        self,
        observation: dict,
        action: int,
        reward: float,
        next_observation: dict,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        self.memory.append(
            Transition(
                state=self.encode_observation(observation),
                action=action,
                reward=reward,
                next_state=self.encode_observation(next_observation),
                done=done,
                next_mask=next_mask.astype(np.float32),
            )
        )

    def train_step(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states = torch.tensor(np.stack([t.state for t in batch]), device=DEVICE)
        actions = torch.tensor([t.action for t in batch], device=DEVICE).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=DEVICE)
        next_states = torch.tensor(np.stack([t.next_state for t in batch]), device=DEVICE)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=DEVICE)
        next_masks = torch.tensor(np.stack([t.next_mask for t in batch]), device=DEVICE)

        current_q = self.policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states)
            next_q[next_masks == 0] = -torch.inf
            best_next_q = torch.max(next_q, dim=1).values
            best_next_q[~torch.isfinite(best_next_q)] = 0.0
            targets = rewards + (1 - dones) * self.gamma * best_next_q

        loss = nn.functional.smooth_l1_loss(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    def update_epsilon(self, step: int) -> float:
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-step / self.epsilon_decay)
        return self.epsilon
