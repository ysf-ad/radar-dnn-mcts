from __future__ import annotations

import torch

from radar_dnn_mcts.env.features import FeatureBuilder
from radar_dnn_mcts.env.transition import RadarObservationTransition
from radar_dnn_mcts.models.scheduler import RadarSchedulerModel
from radar_dnn_mcts.schedulers.base import choose_row, device_of, selected_mask, tensor, update_prefix


class FullReencodeScheduler:
    """Rebuild and re-encode a shadow state after every selected action."""

    def __init__(
        self,
        model: RadarSchedulerModel,
        features: FeatureBuilder | None = None,
        transition: RadarObservationTransition | None = None,
        policy_weight: float = 1.0,
        q_weight: float = 1.0,
        max_steps: int = 32,
    ):
        self.model = model.eval()
        self.features = features or FeatureBuilder()
        self.transition = transition or RadarObservationTransition()
        self.policy_weight = float(policy_weight)
        self.q_weight = float(q_weight)
        self.max_steps = int(max_steps)

    @torch.inference_mode()
    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        device = device_of(self.model)
        shadow = self.transition.clone(obs)
        plan: list[int] = []
        selected: set[int] = set()
        elapsed = 0.0
        searches = tracks = 0
        last = -1
        for _ in range(self.max_steps):
            tokens = tensor(self.features.tokens(shadow, selected, searches), device)
            context = tensor(self.features.context(shadow, elapsed, searches, tracks, last), device)
            chosen = selected_mask(tokens.shape[1], selected, device)
            output = self.model(tokens, context, chosen)
            row = choose_row(output.policy_logits, output.q_values, output.valid_rows, self.policy_weight, self.q_weight)
            if elapsed >= budget_ms:
                break
            plan.append(row)
            elapsed, searches, tracks = update_prefix(row, shadow, elapsed, searches, tracks, selected)
            shadow, _ = self.transition.step(shadow, row)
            last = row
        return plan
