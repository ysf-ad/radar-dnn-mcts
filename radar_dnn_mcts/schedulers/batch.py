from __future__ import annotations

import numpy as np
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
        policy_logits = output.policy_logits[0].cpu().numpy()
        q_values = output.q_values[0].cpu().numpy()
        valid = output.valid_rows[0].cpu().numpy()
        deadlines = np.asarray(obs["t_deadline"], dtype=np.float32)
        plan: list[int] = []
        selected: set[int] = set()
        elapsed = 0.0
        for step in range(policy_logits.shape[0]):
            if elapsed >= budget_ms:
                break
            feasible = valid[step].copy()
            feasible[1:] &= deadlines > elapsed
            if selected:
                feasible[np.asarray(sorted(selected), dtype=np.int64)] = False
            utility = (
                self.policy_weight * policy_logits[step]
                + self.q_weight * q_values[step]
            )
            row = int(np.argmax(np.where(feasible, utility, -np.inf)))
            plan.append(row)
            if row > 0:
                selected.add(row)
            elapsed += action_duration_ms(obs, row)
        return plan
