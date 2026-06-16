from __future__ import annotations

import numpy as np

from final_radar_campaign import MAXT
from pufferlib.ocean.radarxs.models.transformer_mcts_policy import PolicyOnlyMCTSPlanner, PolicyValueTransformer


def adapter():
    return PolicyOnlyMCTSPlanner(
        model=PolicyValueTransformer(num_tasks=MAXT + 1, q_head_use_tanh=False),
        max_trackers=MAXT,
        num_rollouts=1,
        device="cpu",
    )


def valid_mask(obs):
    mask = np.zeros(MAXT + 1, dtype=np.float32)
    mask[0] = 1.0
    active = np.asarray(obs["active_mask"]).astype(bool)
    dead = np.asarray(obs["t_deadline"])
    mask[np.where(active & (dead >= 0.0))[0] + 1] = 1.0
    return mask
