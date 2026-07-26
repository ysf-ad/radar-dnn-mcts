"""Full-window re-encode and autoregressive PUCT scheduler adapters."""

from __future__ import annotations

from radar_dnn_mcts.env.config import RewardConfig
from radar_dnn_mcts.env.features import FeatureBuilder
from radar_dnn_mcts.models.decoders import AutoregressiveDecoder
from radar_dnn_mcts.models.scheduler import RadarSchedulerModel
from radar_dnn_mcts.search import PUCT, PUCTConfig
from radar_dnn_mcts.training.radar_search import (
    RadarAREvaluator,
    RadarModelEvaluator,
    RadarWindowSearchState,
)


class FullWindowPUCTScheduler:
    """Plan one complete 200 ms schedule with PUCT over action prefixes."""

    def __init__(
        self,
        model: RadarSchedulerModel,
        config: PUCTConfig | None = None,
        features: FeatureBuilder | None = None,
        reward: RewardConfig | None = None,
    ):
        self.features = features or FeatureBuilder()
        self.reward = reward or RewardConfig()
        self.search = PUCT(RadarModelEvaluator(model), config or PUCTConfig())

    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        """Return the visit-count principal trajectory from full-window search."""
        state = RadarWindowSearchState(
            obs, self.features, self.reward, budget_ms=budget_ms
        )
        result = self.search.run(state, training=False)
        return [decision.action for decision in result.trajectory]


class AutoregressivePUCTScheduler:
    """Search one full window with the causal AR prefix evaluator."""

    def __init__(
        self,
        decoder: AutoregressiveDecoder,
        config: PUCTConfig,
        features: FeatureBuilder | None = None,
        reward: RewardConfig | None = None,
    ):
        self.features = features or FeatureBuilder()
        self.reward = reward or RewardConfig()
        self.evaluator = RadarAREvaluator(decoder)
        self.search = PUCT(self.evaluator, config)

    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        state = RadarWindowSearchState(
            obs,
            self.features,
            self.reward,
            budget_ms=budget_ms,
        )
        result = self.search.run(state, training=False)
        return [decision.action for decision in result.trajectory]
