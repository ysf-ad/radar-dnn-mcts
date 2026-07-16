from __future__ import annotations

import torch

from radar_dnn_mcts.env.actions import action_duration_ms
from radar_dnn_mcts.env.features import FeatureBuilder
from radar_dnn_mcts.models.decoders import BatchDecoder
from radar_dnn_mcts.schedulers.base import device_of, tensor


class BatchScheduler:
    """Score all window positions in one pass, then apply physical validity masks."""

    def __init__(
        self,
        decoder: BatchDecoder,
        features: FeatureBuilder | None = None,
        policy_weight: float = 1.0,
        q_weight: float = 1.0,
    ):
        self.decoder = decoder.eval()
        self.features = features or FeatureBuilder()
        self.policy_weight = float(policy_weight)
        self.q_weight = float(q_weight)

    @torch.inference_mode()
    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        device = device_of(self.decoder)
        tokens = tensor(self.features.tokens(obs), device)
        context = tensor(self.features.context(obs, 0.0, 0, 0, -1), device)
        output = self.decoder(tokens, context)
        utility = self.policy_weight * output.policy_logits[0] + self.q_weight * output.q_values[0]
        plan: list[int] = []
        selected: set[int] = set()
        elapsed = 0.0
        for step in range(utility.shape[0]):
            scores = utility[step].clone()
            if selected:
                scores[torch.as_tensor(sorted(selected), device=device)] = -1e9
            row = int(torch.argmax(scores).item())
            if elapsed >= budget_ms:
                break
            plan.append(row)
            elapsed += action_duration_ms(obs, row)
            if row > 0:
                selected.add(row)
        return plan
