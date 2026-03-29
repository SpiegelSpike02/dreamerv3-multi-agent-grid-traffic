from __future__ import annotations

from ..algorithms.baselines import MCTSAgent
from ..envs import EnvironmentConfig, GridTrafficEnv


def evaluate_policy(agent, env_config: EnvironmentConfig, *, scenarios: int = 50, seed: int = 0) -> dict[str, float]:
    env = GridTrafficEnv(env_config)

    successes = 0
    total_throughput = 0
    total_collisions = 0
    total_steps = 0

    for scenario in range(scenarios):
        obs = env.reset(seed=seed + scenario, training=False, num_active_agents=env_config.num_agents)
        done = False
        while not done:
            mask = env.action_mask(env.current_agent)
            if isinstance(agent, MCTSAgent):
                action = agent.select_action(env, obs, mask)
            else:
                action = agent.select_action(obs, mask, evaluate=True)
            obs, _, done, _ = env.step(action)
        successes += int(env.throughput_count >= env_config.num_agents)
        total_throughput += env.throughput_count
        total_collisions += env.collision_count
        total_steps += env.step_count

    return {
        "success_rate": round(successes / scenarios, 3),
        "avg_throughput": round(total_throughput / scenarios, 2),
        "avg_collisions": round(total_collisions / scenarios, 2),
        "avg_steps": round(total_steps / scenarios, 2),
    }
