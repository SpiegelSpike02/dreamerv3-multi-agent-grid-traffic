from dataclasses import dataclass


@dataclass(slots=True)
class DreamerV3Config:
    deter_size: int = 128
    stoch_size: int = 32
    hidden_size: int = 128
    horizon: int = 12
    batch_size: int = 32
    learning_rate: float = 3e-4
    replay_size: int = 50000
    discount: float = 0.99
    actor_coef: float = 1.0
    critic_coef: float = 0.5
    target_update_interval: int = 100
