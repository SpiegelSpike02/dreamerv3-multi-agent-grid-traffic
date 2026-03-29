from grid_traffic.algorithms.baselines import DQNAgent, MCTSAgent, PPOAgent, TabularQAgent
from grid_traffic.envs import EnvironmentConfig, GridTrafficEnv


def test_tabular_agent_returns_valid_action():
    env = GridTrafficEnv(EnvironmentConfig())
    obs = env.reset(seed=3)
    agent = TabularQAgent()
    action = agent.select_action(obs, env.action_mask(env.current_agent))
    assert env.action_mask(env.current_agent)[action] == 1


def test_dqn_agent_returns_valid_action():
    env = GridTrafficEnv(EnvironmentConfig())
    obs = env.reset(seed=4)
    agent = DQNAgent()
    action = agent.select_action(obs, env.action_mask(env.current_agent), evaluate=True)
    assert env.action_mask(env.current_agent)[action] == 1


def test_mcts_agent_returns_valid_action():
    env = GridTrafficEnv(EnvironmentConfig())
    obs = env.reset(seed=5)
    agent = MCTSAgent(simulations=5, rollout_depth=4, seed=5)
    action = agent.select_action(env, obs, env.action_mask(env.current_agent))
    assert env.action_mask(env.current_agent)[action] == 1


def test_ppo_agent_returns_valid_action():
    env = GridTrafficEnv(EnvironmentConfig())
    obs = env.reset(seed=6)
    agent = PPOAgent()
    action = agent.select_action(obs, env.action_mask(env.current_agent), evaluate=True)
    assert env.action_mask(env.current_agent)[action] == 1
