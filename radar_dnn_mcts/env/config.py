from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class RewardConfig:
    """Versioned environment objective shared by training and evaluation."""

    env_mode: str = "pufferlib_service"
    search_action_reward: float = 0.0
    track_action_reward: float = 0.0
    drop_penalty: float = 8.0
    search_frame_overdue_weight: float = 0.5
    search_frame_desired_ms: float = 3000.0
    search_frame_deadline_ms: float = 4500.0
    search_frame_drop_penalty: float = 8.0
    search_frame_state_penalty_weight: float = 2.0
    search_frame_delta_reward_weight: float = 5.0
    service_pressure_delta_reward_weight: float = 0.30
    serviced_pressure_improvement_reward_weight: float = 0.15
    discovered_target_reward: float = 0.08
    tracked_count_delta_reward_weight: float = 0.0
    tracked_target_ms_reward_weight: float = 0.0
    penalize_hidden_targets: bool = False
    enable_priority: bool = False

    def as_dict(self) -> dict[str, float | str | bool]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkConfig:
    initial_targets: tuple[int, ...] = (20, 40, 60)
    arrival_rates: tuple[float, ...] = (2.0, 3.0, 4.0)
    seeds: tuple[int, ...] = (916,)
    windows: int = 100
    window_ms: float = 200.0
    max_targets: int = 100
    sensors: tuple[str, ...] = ("S",)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def cells(self):
        for initial in self.initial_targets:
            for rate in self.arrival_rates:
                for seed in self.seeds:
                    yield initial, rate, seed
