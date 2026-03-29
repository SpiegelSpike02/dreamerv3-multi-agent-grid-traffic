from dataclasses import dataclass, field


@dataclass(slots=True)
class RewardConfig:
    step_penalty: float = -1.0
    collision_penalty: float = -40.0
    goal_reward: float = 30.0
    proximity_penalty: float = -4.0
    wait_penalty: float = -0.5


@dataclass(slots=True)
class EnvironmentConfig:
    grid_size: int = 5
    num_agents: int = 4
    max_steps: int = 80
    training_agents: int = 2
    road_width: int = 1
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass(slots=True)
class TrainingConfig:
    episodes: int = 300
    seed: int = 7
    eval_interval: int = 50
