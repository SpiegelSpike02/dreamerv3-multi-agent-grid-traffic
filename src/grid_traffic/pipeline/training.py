from __future__ import annotations

import numpy as np

from ..algorithms.baselines import DQNAgent, PPOAgent, TabularQAgent
from ..algorithms.dreamer import DreamerV3Agent, DreamerV3Config
from ..envs import EnvironmentConfig, GridTrafficEnv, TrainingConfig
from .evaluation import evaluate_policy


def train_tabular_q(env_config: EnvironmentConfig, train_config: TrainingConfig) -> tuple[TabularQAgent, list[dict]]:
    env = GridTrafficEnv(env_config)
    agent = TabularQAgent()
    history: list[dict] = []
    global_step = 0

    for episode in range(1, train_config.episodes + 1):
        obs = env.reset(seed=train_config.seed + episode, training=True)
        done = False
        episode_reward = 0.0
        while not done:
            mask = env.action_mask(env.current_agent)
            action = agent.select_action(obs, mask)
            next_obs, reward, done, info = env.step(action)
            agent.update(obs, action, reward, next_obs, done, info["action_mask"])
            episode_reward += reward
            obs = next_obs
            global_step += 1
            agent.update_epsilon(global_step)

        row = {
            "episode": episode,
            "reward": round(episode_reward, 2),
            "throughput": env.throughput_count,
            "collisions": env.collision_count,
            "epsilon": round(agent.epsilon, 4),
        }
        if episode % train_config.eval_interval == 0 or episode == train_config.episodes:
            row.update({f"eval_{k}": v for k, v in evaluate_policy(agent, env_config, scenarios=20, seed=train_config.seed + episode).items()})
        history.append(row)
    return agent, history


def train_dqn(env_config: EnvironmentConfig, train_config: TrainingConfig) -> tuple[DQNAgent, list[dict]]:
    env = GridTrafficEnv(env_config)
    agent = DQNAgent()
    history: list[dict] = []
    global_step = 0

    for episode in range(1, train_config.episodes + 1):
        obs = env.reset(seed=train_config.seed + episode, training=True)
        done = False
        losses: list[float] = []
        episode_reward = 0.0
        while not done:
            mask = env.action_mask(env.current_agent)
            action = agent.select_action(obs, mask)
            next_obs, reward, done, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, done, info["action_mask"])
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            obs = next_obs
            episode_reward += reward
            global_step += 1
            agent.update_epsilon(global_step)

        row = {
            "episode": episode,
            "reward": round(episode_reward, 2),
            "throughput": env.throughput_count,
            "collisions": env.collision_count,
            "epsilon": round(agent.epsilon, 4),
            "loss": round(float(np.mean(losses)), 4) if losses else None,
        }
        if episode % train_config.eval_interval == 0 or episode == train_config.episodes:
            row.update({f"eval_{k}": v for k, v in evaluate_policy(agent, env_config, scenarios=20, seed=train_config.seed + episode).items()})
        history.append(row)
    return agent, history


def train_ppo(env_config: EnvironmentConfig, train_config: TrainingConfig) -> tuple[PPOAgent, list[dict]]:
    env = GridTrafficEnv(env_config)
    agent = PPOAgent()
    history: list[dict] = []

    for episode in range(1, train_config.episodes + 1):
        obs = env.reset(seed=train_config.seed + episode, training=True)
        done = False
        updates: list[dict[str, float]] = []
        episode_reward = 0.0

        while not done:
            mask = env.action_mask(env.current_agent)
            action, log_prob, value = agent.act(obs, mask)
            next_obs, reward, done, _ = env.step(action)
            agent.remember(obs, action, log_prob, reward, done, value, mask)
            episode_reward += reward
            obs = next_obs

            if agent.ready_to_update():
                metrics = agent.finish_episode_and_update(last_observation=obs)
                if metrics is not None:
                    updates.append(metrics)

        metrics = agent.finish_episode_and_update()
        if metrics is not None:
            updates.append(metrics)

        row = {
            "episode": episode,
            "reward": round(episode_reward, 2),
            "throughput": env.throughput_count,
            "collisions": env.collision_count,
            "loss": round(float(np.mean([m["loss"] for m in updates])), 4) if updates else None,
            "actor_loss": round(float(np.mean([m["actor_loss"] for m in updates])), 4) if updates else None,
            "critic_loss": round(float(np.mean([m["critic_loss"] for m in updates])), 4) if updates else None,
            "entropy": round(float(np.mean([m["entropy"] for m in updates])), 4) if updates else None,
        }
        if episode % train_config.eval_interval == 0 or episode == train_config.episodes:
            row.update({f"eval_{k}": v for k, v in evaluate_policy(agent, env_config, scenarios=20, seed=train_config.seed + episode).items()})
        history.append(row)
    return agent, history


def train_dreamerv3(
    env_config: EnvironmentConfig,
    train_config: TrainingConfig,
    dreamer_config: DreamerV3Config | None = None,
) -> tuple[DreamerV3Agent, list[dict]]:
    env = GridTrafficEnv(env_config)
    agent = DreamerV3Agent(dreamer_config or DreamerV3Config())
    history: list[dict] = []

    for episode in range(1, train_config.episodes + 1):
        obs = env.reset(seed=train_config.seed + episode, training=True)
        done = False
        losses: list[dict[str, float]] = []
        episode_reward = 0.0

        while not done:
            mask = env.action_mask(env.current_agent)
            action = agent.select_action(obs, mask)
            next_obs, reward, done, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, done, info["action_mask"])
            metrics = agent.train_step()
            if metrics is not None:
                losses.append(metrics)
            obs = next_obs
            episode_reward += reward

        row = {
            "episode": episode,
            "reward": round(episode_reward, 2),
            "throughput": env.throughput_count,
            "collisions": env.collision_count,
            "loss": round(float(np.mean([m["loss"] for m in losses])), 4) if losses else None,
            "world_loss": round(float(np.mean([m["world_loss"] for m in losses])), 4) if losses else None,
            "actor_loss": round(float(np.mean([m["actor_loss"] for m in losses])), 4) if losses else None,
            "critic_loss": round(float(np.mean([m["critic_loss"] for m in losses])), 4) if losses else None,
        }
        if episode % train_config.eval_interval == 0 or episode == train_config.episodes:
            row.update({f"eval_{k}": v for k, v in evaluate_policy(agent, env_config, scenarios=20, seed=train_config.seed + episode).items()})
        history.append(row)

    return agent, history
