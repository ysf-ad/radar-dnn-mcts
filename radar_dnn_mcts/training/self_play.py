from __future__ import annotations

import numpy as np
import torch

from radar_dnn_mcts.search.puct import PUCT, SearchState
from radar_dnn_mcts.training.replay import TrainingSample


def collect_episode(search: PUCT, state: SearchState, max_steps: int = 256) -> list[TrainingSample]:
    """Collect planner-improved policy and tree action-Q targets."""

    records = []
    for _ in range(max_steps):
        legal = state.legal_actions()
        if not legal:
            break
        result = search.run(state)
        network_input = state.network_input()
        policy = np.zeros(network_input[0].shape[-2], dtype=np.float32)
        q_target = np.zeros_like(policy)
        q_mask = np.zeros_like(policy, dtype=bool)
        for action, probability in result.policy.items():
            policy[action] = probability
            q_target[action] = result.q_values[action]
            q_mask[action] = True
        _, terminal = state.step(result.action)
        records.append((network_input, policy, q_target, q_mask))
        if terminal:
            break

    return [
        TrainingSample(
            tokens=torch.as_tensor(inputs[0]),
            context=torch.as_tensor(inputs[1]),
            policy_target=torch.from_numpy(policy),
            q_target=torch.from_numpy(q_target),
            q_mask=torch.from_numpy(q_mask),
        )
        for inputs, policy, q_target, q_mask in records
    ]
