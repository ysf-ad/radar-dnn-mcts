"""Predict the next boundary state while the current radar window executes."""

from __future__ import annotations

import torch

from radar_dnn_mcts.env.features import FeatureBuilder
from radar_dnn_mcts.models.backbone import EncodedState
from radar_dnn_mcts.models.boundary import BoundaryPredictor
from radar_dnn_mcts.models.scheduler import RadarSchedulerModel
from radar_dnn_mcts.schedulers.base import choose_row, device_of, selected_mask, tensor, update_prefix


def latency_aware_start_ms(
    window_ms: float,
    max_latency_ms: float,
    buffer_ms: float = 0.0,
) -> float:
    """Start planning at B - L_max - buffer, clipped to the window."""
    return max(
        0.0,
        min(float(window_ms), float(window_ms) - float(max_latency_ms) - float(buffer_ms)),
    )


class AsynchronousBoundaryScheduler:
    """Prepare the next schedule while the remaining current-window actions execute."""

    def __init__(
        self,
        model: RadarSchedulerModel,
        boundary: BoundaryPredictor,
        features: FeatureBuilder | None = None,
        policy_weight: float = 1.0,
        q_weight: float = 1.0,
        max_steps: int = 32,
        latency_buffer_ms: float = 0.0,
    ):
        self.model = model.eval()
        self.boundary = boundary.eval()
        self.features = features or FeatureBuilder()
        self.policy_weight = float(policy_weight)
        self.q_weight = float(q_weight)
        self.max_steps = int(max_steps)
        self.latency_buffer_ms = float(latency_buffer_ms)

    def planning_start_ms(
        self,
        max_latency_ms: float,
        window_ms: float = 200.0,
    ) -> float:
        """Return the latency-aware boundary-planning start time."""
        return latency_aware_start_ms(
            window_ms,
            max_latency_ms,
            self.latency_buffer_ms,
        )

    @torch.inference_mode()
    def plan_next(
        self,
        midpoint_obs: dict,
        remaining_rows: list[int],
        remaining_ms: float,
        budget_ms: float = 200.0,
    ) -> list[int]:
        device = device_of(self.model)
        tokens = tensor(self.features.tokens(midpoint_obs), device)
        root_context = tensor(self.features.context(midpoint_obs, 0.0, 0, 0, -1), device)
        state = self.model.encode(tokens, root_context)
        width = self.boundary.max_suffix
        suffix = torch.zeros(1, width, dtype=torch.long, device=device)
        mask = torch.zeros(1, width, dtype=torch.bool, device=device)
        count = min(width, len(remaining_rows))
        if count:
            suffix[0, :count] = torch.as_tensor(remaining_rows[:count], device=device)
            mask[0, :count] = True
        state = self.boundary(state, suffix, mask, torch.tensor([remaining_ms], device=device))
        return self._decode(state, midpoint_obs, budget_ms)

    def _decode(self, state: EncodedState, obs: dict, budget_ms: float) -> list[int]:
        device = state.global_state.device
        plan: list[int] = []
        selected: set[int] = set()
        elapsed = 0.0
        searches = tracks = 0
        last = -1
        for _ in range(self.max_steps):
            if elapsed >= budget_ms:
                break
            context = tensor(self.features.context(obs, elapsed, searches, tracks, last), device)
            output = self.model.predict(state, context, selected_mask(state.target_states.shape[1], selected, device))
            row = choose_row(output.policy_logits, output.q_values, output.valid_rows, self.policy_weight, self.q_weight)
            plan.append(row)
            elapsed, searches, tracks = update_prefix(row, obs, elapsed, searches, tracks, selected)
            last = row
        return plan
