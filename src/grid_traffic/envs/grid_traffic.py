from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np

from .config import EnvironmentConfig


ACTIONS: dict[int, tuple[int, int]] = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, 1),
    3: (0, -1),
    4: (0, 0),
}


@dataclass(slots=True)
class AgentState:
    position: tuple[int, int]
    goal: tuple[int, int]
    route_type: int


class GridTrafficEnv:
    """Turn-based multi-agent grid traffic environment."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.rng = random.Random()
        self.np_rng = np.random.default_rng()
        self.possible_agents = [f"agent_{idx}" for idx in range(config.num_agents)]
        self.reset(seed=0)

    def clone(self) -> "GridTrafficEnv":
        return copy.deepcopy(self)

    @property
    def current_agent(self) -> str:
        return self.agents[self.current_agent_index]

    def reset(
        self,
        seed: int | None = None,
        *,
        training: bool = False,
        num_active_agents: int | None = None,
    ) -> dict:
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)

        self.training = training
        active_agents = num_active_agents or (
            self.config.training_agents if training else self.config.num_agents
        )
        active_agents = max(1, min(active_agents, self.config.num_agents))
        self.agents = self.possible_agents[:active_agents]
        self.current_agent_index = 0
        self.step_count = 0
        self.collision_count = 0
        self.throughput_count = 0
        self.road_map = self._build_road_map()
        self.entry_points = self._boundary_road_cells()
        self.exit_points = self.entry_points.copy()

        self.agent_states: dict[str, AgentState] = {}
        for agent in self.agents:
            start, goal = self._sample_origin_goal_pair()
            self.agent_states[agent] = AgentState(
                position=start,
                goal=goal,
                route_type=int(start[0] == goal[0]),
            )

        self.done = False
        return self.observe(self.current_agent)

    def action_mask(self, agent: str) -> np.ndarray:
        row, col = self.agent_states[agent].position
        size = self.config.grid_size
        mask = np.ones(len(ACTIONS), dtype=np.int8)
        for action, (d_row, d_col) in ACTIONS.items():
            next_row, next_col = row + d_row, col + d_col
            if not (0 <= next_row < size and 0 <= next_col < size):
                mask[action] = 0
                continue
            if action != 4 and not self.road_map[next_row, next_col]:
                mask[action] = 0
        return mask

    def observe(self, agent: str) -> dict:
        state = self.agent_states[agent]
        adjacent = self._adjacent_occupancy(agent)
        return {
            "agent_id": agent,
            "position": state.position,
            "destination": state.goal,
            "route_type": state.route_type,
            "adjacent_occupancy": adjacent,
            "grid_size": self.config.grid_size,
            "num_agents": len(self.agents),
            "step_count": self.step_count,
            "local_density": float(np.sum(adjacent)),
        }

    def _adjacent_occupancy(self, agent: str) -> np.ndarray:
        row, col = self.agent_states[agent].position
        occupancies = np.zeros(4, dtype=np.int8)
        neighbors = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        for idx, (d_row, d_col) in enumerate(neighbors):
            next_row, next_col = row + d_row, col + d_col
            if not (0 <= next_row < self.config.grid_size and 0 <= next_col < self.config.grid_size):
                continue
            next_pos = (next_row, next_col)
            for other_agent, other_state in self.agent_states.items():
                if other_agent == agent:
                    continue
                if other_state.position == next_pos:
                    occupancies[idx] = 1
                    break
        return occupancies

    def _has_collision(self, agent: str) -> bool:
        state = self.agent_states[agent]
        for other_agent, other_state in self.agent_states.items():
            if other_agent == agent:
                continue
            if other_state.position == state.position:
                return True
        return False

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        if self.done:
            raise RuntimeError("environment is done, call reset() before stepping again")

        agent = self.current_agent
        mask = self.action_mask(agent)
        if mask[action] == 0:
            raise ValueError(f"invalid action {action} for {agent}")

        state = self.agent_states[agent]
        old_distance = self._manhattan(state.position, state.goal)
        d_row, d_col = ACTIONS[action]
        state.position = (state.position[0] + d_row, state.position[1] + d_col)

        new_distance = self._manhattan(state.position, state.goal)
        reward = self.config.reward.step_penalty
        reward += (old_distance - new_distance) * 0.6
        reward += float(np.sum(self._adjacent_occupancy(agent))) * self.config.reward.proximity_penalty
        if action == 4:
            reward += self.config.reward.wait_penalty

        if self._has_collision(agent):
            reward += self.config.reward.collision_penalty
            self.collision_count += 1

        if state.position == state.goal:
            reward += self.config.reward.goal_reward
            self.throughput_count += 1
            start, goal = self._sample_origin_goal_pair()
            state.position = start
            state.goal = goal
            state.route_type = int(start[0] == goal[0])

        self.step_count += 1
        self.done = self.step_count >= self.config.max_steps
        self.current_agent_index = (self.current_agent_index + 1) % len(self.agents)
        next_obs = self.observe(self.current_agent)
        info = {
            "action_mask": self.action_mask(self.current_agent),
            "collisions": self.collision_count,
            "throughput": self.throughput_count,
        }
        return next_obs, reward, self.done, info

    def last(self) -> tuple[dict, float, bool, dict]:
        obs = self.observe(self.current_agent)
        info = {"action_mask": self.action_mask(self.current_agent)}
        return obs, 0.0, self.done, info

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _build_road_map(self) -> np.ndarray:
        size = self.config.grid_size
        road_map = np.zeros((size, size), dtype=bool)
        mid = size // 2
        road_map[mid, :] = True
        road_map[:, mid] = True
        if size >= 7:
            road_map[1, 1:-1] = True
            road_map[1:-1, 1] = True
        return road_map

    def _boundary_road_cells(self) -> list[tuple[int, int]]:
        size = self.config.grid_size
        cells: list[tuple[int, int]] = []
        for row in range(size):
            for col in range(size):
                if not self.road_map[row, col]:
                    continue
                if row in (0, size - 1) or col in (0, size - 1):
                    cells.append((row, col))
        return cells

    def _sample_origin_goal_pair(self) -> tuple[tuple[int, int], tuple[int, int]]:
        origin = self.rng.choice(self.entry_points)
        candidates = [cell for cell in self.exit_points if cell != origin]
        goal = self.rng.choice(candidates)
        return origin, goal


GridTransportEnv = GridTrafficEnv
