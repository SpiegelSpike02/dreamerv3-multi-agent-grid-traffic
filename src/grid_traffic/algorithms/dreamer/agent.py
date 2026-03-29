from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import DreamerV3Config


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(slots=True)
class DreamerTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_mask: np.ndarray


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


class DynamicsModel(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.SiLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, latent: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(torch.cat([latent, action_features], dim=-1)))


class Head(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DreamerV3Agent:
    def __init__(self, config: DreamerV3Config | None = None, action_dim: int = 5):
        self.config = config or DreamerV3Config()
        self.action_dim = action_dim
        self.obs_dim = 10
        self.latent_dim = self.config.deter_size

        self.encoder = Encoder(self.obs_dim, self.latent_dim).to(DEVICE)
        self.target_encoder = Encoder(self.obs_dim, self.latent_dim).to(DEVICE)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.dynamics = DynamicsModel(self.latent_dim, action_dim).to(DEVICE)
        self.reward_head = Head(self.latent_dim, 1).to(DEVICE)
        self.continue_head = Head(self.latent_dim, 1).to(DEVICE)
        self.policy_head = Head(self.latent_dim, action_dim).to(DEVICE)
        self.value_head = Head(self.latent_dim, 1).to(DEVICE)

        params = (
            list(self.encoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.reward_head.parameters())
            + list(self.continue_head.parameters())
            + list(self.policy_head.parameters())
            + list(self.value_head.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)
        self.memory: deque[DreamerTransition] = deque(maxlen=self.config.replay_size)
        self.update_steps = 0
        self.exploration_epsilon = 0.2

    def encode_observation(self, observation: dict) -> np.ndarray:
        row, col = observation["position"]
        dst_row, dst_col = observation["destination"]
        grid_size = max(observation["grid_size"] - 1, 1)
        adjacent = observation["adjacent_occupancy"]
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
                observation["local_density"] / 4.0,
            ],
            dtype=np.float32,
        )

    def select_action(self, observation: dict, action_mask: np.ndarray, *, evaluate: bool = False) -> int:
        valid_actions = np.flatnonzero(action_mask)
        if not evaluate and random.random() < self.exploration_epsilon:
            return int(random.choice(valid_actions))

        state = torch.tensor(self.encode_observation(observation), device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            latent = self.encoder(state)
            logits = self.policy_head(latent).squeeze(0).cpu().numpy()
        logits[action_mask == 0] = -np.inf
        return int(np.argmax(logits))

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
            DreamerTransition(
                state=self.encode_observation(observation),
                action=action,
                reward=reward,
                next_state=self.encode_observation(next_observation),
                done=done,
                next_mask=next_mask.astype(np.float32),
            )
        )

    def train_step(self) -> dict[str, float] | None:
        if len(self.memory) < self.config.batch_size:
            return None

        batch = random.sample(self.memory, self.config.batch_size)
        states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=DEVICE)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=DEVICE)
        next_states = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=DEVICE)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=DEVICE)

        latent = self.encoder(states)
        with torch.no_grad():
            target_next_latent = self.target_encoder(next_states)

        action_features = torch.nn.functional.one_hot(actions, num_classes=self.action_dim).float()
        pred_next_latent = self.dynamics(latent, action_features)
        pred_reward = self.reward_head(pred_next_latent).squeeze(-1)
        pred_continue = self.continue_head(pred_next_latent).squeeze(-1)

        world_loss = (
            nn.functional.mse_loss(pred_next_latent, target_next_latent)
            + nn.functional.mse_loss(pred_reward, rewards)
            + nn.functional.binary_cross_entropy_with_logits(pred_continue, 1.0 - dones)
        )

        with torch.no_grad():
            next_values_target = self.value_head(target_next_latent).squeeze(-1)
            td_target = rewards + (1.0 - dones) * self.config.discount * next_values_target
            td_target = torch.clamp(td_target, -50.0, 50.0)
        critic_loss = nn.functional.mse_loss(self.value_head(latent).squeeze(-1), td_target)

        imagined_latent = latent.detach()
        actor_objective = torch.zeros((), device=DEVICE)
        imagined_reward = torch.zeros((), device=DEVICE)
        for _ in range(self.config.horizon):
            logits = self.policy_head(imagined_latent)
            probs = torch.softmax(logits, dim=-1)
            imagined_latent = self.dynamics(imagined_latent, probs)
            reward_pred = torch.tanh(self.reward_head(imagined_latent).squeeze(-1))
            value_pred = torch.tanh(self.value_head(imagined_latent).squeeze(-1))
            actor_objective = actor_objective + (reward_pred + self.config.discount * value_pred).mean()
            imagined_reward = imagined_reward + reward_pred.mean()
        actor_loss = -actor_objective / self.config.horizon

        loss = world_loss + self.config.critic_coef * critic_loss + self.config.actor_coef * actor_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.reward_head.parameters())
            + list(self.continue_head.parameters())
            + list(self.policy_head.parameters())
            + list(self.value_head.parameters()),
            max_norm=5.0,
        )
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.config.target_update_interval == 0:
            self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.exploration_epsilon = max(0.05, self.exploration_epsilon * 0.999)

        return {
            "loss": float(loss.item()),
            "world_loss": float(world_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "imagined_reward": float(imagined_reward.item() / self.config.horizon),
        }
