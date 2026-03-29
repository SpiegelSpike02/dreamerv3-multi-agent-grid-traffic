from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(slots=True)
class RolloutTransition:
    state: np.ndarray
    action: int
    log_prob: float
    reward: float
    done: bool
    value: float
    action_mask: np.ndarray


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


class PPOAgent:
    def __init__(
        self,
        action_dim: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        learning_rate: float = 3e-4,
        rollout_size: int = 256,
        update_epochs: int = 4,
        minibatch_size: int = 64,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ):
        self.action_dim = action_dim
        self.state_dim = 10
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.rollout_size = rollout_size
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.model = ActorCritic(self.state_dim, action_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.rollout_buffer: list[RolloutTransition] = []

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
                observation["local_density"] / 4.0,
            ],
            dtype=np.float32,
        )

    def select_action(self, observation: dict, action_mask: np.ndarray, *, evaluate: bool = False) -> int:
        action, _, _ = self.act(observation, action_mask, evaluate=evaluate)
        return action

    def act(self, observation: dict, action_mask: np.ndarray, *, evaluate: bool = False) -> tuple[int, float, float]:
        state = torch.tensor(self.encode_observation(observation), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mask = torch.tensor(action_mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(state)
            logits = logits.masked_fill(~mask, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            if evaluate:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def remember(
        self,
        observation: dict,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        action_mask: np.ndarray,
    ) -> None:
        self.rollout_buffer.append(
            RolloutTransition(
                state=self.encode_observation(observation),
                action=action,
                log_prob=log_prob,
                reward=reward,
                done=done,
                value=value,
                action_mask=action_mask.astype(np.float32),
            )
        )

    def ready_to_update(self) -> bool:
        return len(self.rollout_buffer) >= self.rollout_size

    def finish_episode_and_update(self, last_observation: dict | None = None) -> dict[str, float] | None:
        if not self.rollout_buffer:
            return None

        next_value = 0.0
        if last_observation is not None:
            state = torch.tensor(self.encode_observation(last_observation), dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                _, value = self.model(state)
                next_value = float(value.item())

        rewards = [t.reward for t in self.rollout_buffer]
        values = [t.value for t in self.rollout_buffer] + [next_value]
        dones = [t.done for t in self.rollout_buffer]
        advantages = []
        gae = 0.0
        for step in reversed(range(len(self.rollout_buffer))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1.0 - float(dones[step])) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[step])) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]

        states = torch.tensor(np.stack([t.state for t in self.rollout_buffer]), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor([t.action for t in self.rollout_buffer], dtype=torch.int64, device=DEVICE)
        old_log_probs = torch.tensor([t.log_prob for t in self.rollout_buffer], dtype=torch.float32, device=DEVICE)
        action_masks = torch.tensor(np.stack([t.action_mask for t in self.rollout_buffer]), dtype=torch.bool, device=DEVICE)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        losses = []
        actor_losses = []
        critic_losses = []
        entropy_terms = []
        total_steps = len(self.rollout_buffer)

        for _ in range(self.update_epochs):
            indices = torch.randperm(total_steps, device=DEVICE)
            for start in range(0, total_steps, self.minibatch_size):
                batch_idx = indices[start : start + self.minibatch_size]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                batch_returns = returns_t[batch_idx]
                batch_masks = action_masks[batch_idx]

                logits, values_pred = self.model(batch_states)
                logits = logits.masked_fill(~batch_masks, -1e9)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                actor_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
                critic_loss = nn.functional.mse_loss(values_pred, batch_returns)
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                losses.append(float(loss.item()))
                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropy_terms.append(float(entropy.item()))

        self.rollout_buffer.clear()
        return {
            "loss": float(np.mean(losses)),
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropy_terms)),
        }
