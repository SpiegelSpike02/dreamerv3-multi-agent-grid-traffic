from __future__ import annotations

import argparse
import json

from .algorithms.baselines import MCTSAgent
from .algorithms.dreamer import DreamerV3Config
from .envs import EnvironmentConfig, TrainingConfig
from .pipeline import evaluate_policy, train_dqn, train_dreamerv3, train_ppo, train_tabular_q


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DreamerV3-based multi-agent grid traffic benchmark")
    parser.add_argument("--agent", choices=["tabular_q", "dqn", "ppo", "mcts", "dreamerv3"], default="tabular_q")
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--eval-scenarios", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mcts-simulations", type=int, default=60)
    parser.add_argument("--mcts-depth", type=int, default=12)
    parser.add_argument("--dreamer-horizon", type=int, default=12)
    parser.add_argument("--dreamer-batch-size", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    env_config = EnvironmentConfig(
        grid_size=args.grid_size,
        num_agents=args.num_agents,
        max_steps=args.max_steps,
    )
    train_config = TrainingConfig(
        episodes=args.episodes,
        seed=args.seed,
        eval_interval=max(1, args.episodes // 5),
    )

    if args.agent == "tabular_q":
        agent, history = train_tabular_q(env_config, train_config)
    elif args.agent == "dqn":
        agent, history = train_dqn(env_config, train_config)
    elif args.agent == "ppo":
        agent, history = train_ppo(env_config, train_config)
    elif args.agent == "dreamerv3":
        dreamer_config = DreamerV3Config(
            horizon=args.dreamer_horizon,
            batch_size=args.dreamer_batch_size,
        )
        agent, history = train_dreamerv3(env_config, train_config, dreamer_config)
    else:
        agent = MCTSAgent(
            simulations=args.mcts_simulations,
            rollout_depth=args.mcts_depth,
            seed=args.seed,
        )
        history = []

    metrics = evaluate_policy(agent, env_config, scenarios=args.eval_scenarios, seed=args.seed)
    print(json.dumps({"config": vars(args), "metrics": metrics, "history_tail": history[-3:]}, indent=2))


if __name__ == "__main__":
    main()
