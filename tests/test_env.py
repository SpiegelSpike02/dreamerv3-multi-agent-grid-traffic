from grid_traffic.envs import EnvironmentConfig, GridTrafficEnv


def test_env_supports_variable_grid_and_agent_count():
    env = GridTrafficEnv(EnvironmentConfig(grid_size=7, num_agents=6, max_steps=40))
    observation = env.reset(seed=1)
    assert observation["grid_size"] == 7
    assert len(env.agents) == 6
    assert len(env.entry_points) >= 4


def test_training_mode_uses_subset_of_agents():
    env = GridTrafficEnv(EnvironmentConfig(grid_size=5, num_agents=5, training_agents=2))
    env.reset(seed=2, training=True)
    assert len(env.agents) == 2
