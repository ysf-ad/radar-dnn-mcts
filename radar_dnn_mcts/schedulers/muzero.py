from __future__ import annotations

import torch

from radar_dnn_mcts.env.features import FeatureBuilder
from radar_dnn_mcts.models.dynamics import LatentDynamics
from radar_dnn_mcts.models.scheduler import RadarSchedulerModel
from radar_dnn_mcts.schedulers.base import choose_row, device_of, selected_mask, tensor, update_prefix


class MuZeroScheduler:
    """Encode once and recurrently update the latent state with learned dynamics g."""

    def __init__(
        self,
        model: RadarSchedulerModel,
        dynamics: LatentDynamics,
        features: FeatureBuilder | None = None,
        policy_weight: float = 1.0,
        q_weight: float = 1.0,
        max_steps: int = 32,
    ):
        self.model = model.eval()
        self.dynamics = dynamics.eval()
        self.features = features or FeatureBuilder()
        self.policy_weight = float(policy_weight)
        self.q_weight = float(q_weight)
        self.max_steps = int(max_steps)

    @torch.inference_mode()
    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        """Decode greedily while g predicts each next latent radar state."""
        device = device_of(self.model)
        root_tokens = tensor(self.features.tokens(obs), device)
        context = tensor(self.features.context(obs, 0.0, 0, 0, -1), device)
        state = self.model.encode(root_tokens, context)
        plan: list[int] = []
        selected: set[int] = set()
        elapsed = 0.0
        searches = tracks = 0
        last = -1
        for _ in range(self.max_steps):
            if elapsed >= budget_ms:
                break
            context = tensor(self.features.context(obs, elapsed, searches, tracks, last), device)
            chosen = selected_mask(root_tokens.shape[1], selected, device, obs, elapsed)
            output = self.model.predict(state, context, chosen)
            row = choose_row(
                output.policy_logits,
                output.q_values,
                output.valid_rows,
                self.policy_weight,
                self.q_weight,
            )
            plan.append(row)
            state, _reward = self.dynamics(state, torch.tensor([row], device=device))
            elapsed, searches, tracks = update_prefix(row, obs, elapsed, searches, tracks, selected)
            last = row
        return plan
