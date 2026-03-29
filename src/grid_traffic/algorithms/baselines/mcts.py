from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from ...envs import GridTransportEnv


@dataclass
class Node:
    action: int | None = None
    parent: "Node | None" = None
    visits: int = 0
    value: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)

    def ucb_score(self, parent_visits: int, exploration_weight: float) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.value / self.visits
        explore = exploration_weight * math.sqrt(math.log(parent_visits) / self.visits)
        return exploit + explore


class MCTSAgent:
    def __init__(
        self,
        simulations: int = 60,
        rollout_depth: int = 12,
        exploration_weight: float = 1.2,
        seed: int = 7,
    ):
        self.simulations = simulations
        self.rollout_depth = rollout_depth
        self.exploration_weight = exploration_weight
        self.rng = random.Random(seed)

    def select_action(self, env: GridTransportEnv, observation: dict, action_mask: np.ndarray) -> int:
        valid_actions = [int(a) for a in np.flatnonzero(action_mask)]
        if len(valid_actions) == 1:
            return valid_actions[0]

        root = Node()
        for _ in range(self.simulations):
            sim_env = env.clone()
            node = root
            total_reward = 0.0
            depth = 0

            while depth < self.rollout_depth:
                obs = sim_env.observe(sim_env.current_agent)
                mask = sim_env.action_mask(sim_env.current_agent)
                actions = [int(a) for a in np.flatnonzero(mask)]

                if sim_env.current_agent == observation["agent_id"]:
                    if len(node.children) < len(actions):
                        unexplored = [a for a in actions if a not in node.children]
                        action = self.rng.choice(unexplored)
                        child = Node(action=action, parent=node)
                        node.children[action] = child
                        node = child
                    else:
                        parent_visits = max(node.visits, 1)
                        action, node = max(
                            node.children.items(),
                            key=lambda item: item[1].ucb_score(parent_visits, self.exploration_weight),
                        )
                else:
                    action = self._heuristic_action(obs, mask)

                _, reward, done, _ = sim_env.step(action)
                if sim_env.current_agent != observation["agent_id"]:
                    total_reward += 0.15 * reward
                else:
                    total_reward += reward
                depth += 1
                if done:
                    break

            total_reward += self._rollout(sim_env, observation["agent_id"], depth)
            self._backpropagate(node, total_reward)

        best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return int(best_action)

    def _rollout(self, env: GridTransportEnv, target_agent: str, start_depth: int) -> float:
        total_reward = 0.0
        for _ in range(start_depth, self.rollout_depth):
            if env.done:
                break
            obs = env.observe(env.current_agent)
            mask = env.action_mask(env.current_agent)
            action = self._heuristic_action(obs, mask)
            _, reward, done, _ = env.step(action)
            if obs["agent_id"] == target_agent:
                total_reward += reward
            else:
                total_reward += 0.1 * reward
            if done:
                break
        return total_reward

    def _heuristic_action(self, observation: dict, action_mask: np.ndarray) -> int:
        valid_actions = [int(a) for a in np.flatnonzero(action_mask)]
        best_actions: list[int] = []
        best_distance = None
        row, col = observation["position"]
        dst_row, dst_col = observation["destination"]
        adjacent = observation["adjacent_occupancy"]
        deltas = ACTIONS = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, 1),
            3: (0, -1),
            4: (0, 0),
        }
        for action in valid_actions:
            d_row, d_col = deltas[action]
            next_row, next_col = row + d_row, col + d_col
            distance = abs(next_row - dst_row) + abs(next_col - dst_col)
            penalty = adjacent[action] * 2 if action < 4 else 0
            score = distance + penalty
            if best_distance is None or score < best_distance:
                best_distance = score
                best_actions = [action]
            elif score == best_distance:
                best_actions.append(action)
        return self.rng.choice(best_actions)

    def _backpropagate(self, node: Node, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
