from __future__ import annotations

import torch

from radar_dnn_mcts.env.features import FeatureBuilder
from radar_dnn_mcts.models.backbone import EncodedState
from radar_dnn_mcts.models.boundary import BoundaryPredictor
from radar_dnn_mcts.models.scheduler import RadarSchedulerModel
from radar_dnn_mcts.schedulers.base import choose_row, device_of, selected_mask, tensor, update_prefix


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
    ):
        self.model = model.eval()
        self.boundary = boundary.eval()
        self.features = features or FeatureBuilder()
        self.policy_weight = float(policy_weight)
        self.q_weight = float(q_weight)
        self.max_steps = int(max_steps)

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
