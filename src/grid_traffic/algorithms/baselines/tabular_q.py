from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np


class TabularQAgent:
    def __init__(
        self,
        action_dim: int = 5,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 3000,
    ):
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.q_table: defaultdict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(action_dim, dtype=np.float32)
        )

    def encode_observation(self, observation: dict) -> tuple:
        position = observation["position"]
        destination = observation["destination"]
        adjacent = tuple(int(x) for x in observation["adjacent_occupancy"])
        return (
            observation["grid_size"],
            position[0],
            position[1],
            destination[0],
            destination[1],
            int(observation["route_type"]),
            *adjacent,
        )

    def select_action(self, observation: dict, action_mask: np.ndarray, *, evaluate: bool = False) -> int:
        state = self.encode_observation(observation)
        valid_actions = np.flatnonzero(action_mask)
        if not evaluate and random.random() < self.epsilon:
            return int(random.choice(valid_actions))
        q_values = self.q_table[state].copy()
        q_values[action_mask == 0] = -np.inf
        return int(np.argmax(q_values))

    def update(
        self,
        observation: dict,
        action: int,
        reward: float,
        next_observation: dict,
        done: bool,
        next_action_mask: np.ndarray,
    ) -> None:
        state = self.encode_observation(observation)
        next_state = self.encode_observation(next_observation)
        next_q = self.q_table[next_state].copy()
        next_q[next_action_mask == 0] = -np.inf
        target = reward if done else reward + self.gamma * np.max(next_q)
        if not np.isfinite(target):
            target = reward
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    def update_epsilon(self, step: int) -> float:
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-step / self.epsilon_decay)
        return self.epsilon
