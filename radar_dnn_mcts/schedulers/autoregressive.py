from __future__ import annotations

import torch

from radar_dnn_mcts.env.features import FeatureBuilder
from radar_dnn_mcts.models.decoders import AutoregressiveDecoder
from radar_dnn_mcts.schedulers.base import choose_row, device_of, selected_mask, tensor, update_prefix


class AutoregressiveScheduler:
    """Encode once and decode each action conditioned on the selected prefix."""

    def __init__(
        self,
        decoder: AutoregressiveDecoder,
        features: FeatureBuilder | None = None,
        policy_weight: float = 1.0,
        q_weight: float = 1.0,
        max_steps: int = 32,
    ):
        self.decoder = decoder.eval()
        self.features = features or FeatureBuilder()
        self.policy_weight = float(policy_weight)
        self.q_weight = float(q_weight)
        self.max_steps = int(max_steps)

    @torch.inference_mode()
    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        device = device_of(self.decoder)
        tokens = tensor(self.features.tokens(obs), device)
        root_context = tensor(self.features.context(obs, 0.0, 0, 0, -1), device)
        state, prefix = self.decoder.initial(tokens, root_context)
        plan: list[int] = []
        selected: set[int] = set()
        elapsed = 0.0
        searches = tracks = 0
        previous = torch.zeros(1, dtype=torch.long, device=device)
        last = -1
        for _ in range(self.max_steps):
            if elapsed >= budget_ms:
                break
            context = tensor(self.features.context(obs, elapsed, searches, tracks, last), device)
            output, prefix = self.decoder.step(
                state, prefix, context, previous, selected_mask(tokens.shape[1], selected, device)
            )
            row = choose_row(output.policy_logits, output.q_values, output.valid_rows, self.policy_weight, self.q_weight)
            plan.append(row)
            elapsed, searches, tracks = update_prefix(row, obs, elapsed, searches, tracks, selected)
            previous.fill_(row)
            last = row
        return plan
