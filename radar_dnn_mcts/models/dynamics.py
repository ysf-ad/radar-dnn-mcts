from __future__ import annotations

import torch
from torch import nn

from radar_dnn_mcts.models.backbone import EncodedState


class LatentDynamics(nn.Module):
    """MuZero-style learned transition g(h, a) -> (h', reward)."""

    def __init__(self, max_rows: int = 101, d_model: int = 96):
        super().__init__()
        self.action_embedding = nn.Embedding(max_rows, d_model)
        self.global_update = nn.GRUCell(d_model, d_model)
        self.target_update = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.reward = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, state: EncodedState, action_rows: torch.Tensor) -> tuple[EncodedState, torch.Tensor]:
        rows = action_rows.long()
        gather = rows[:, None, None].expand(-1, 1, state.target_states.shape[-1])
        selected_action = state.target_states.gather(1, gather).squeeze(1)
        action = self.action_embedding((rows > 0).long()) + selected_action
        new_global = self.global_update(action, state.global_state)
        action_rows_expanded = action[:, None, :].expand_as(state.target_states)
        global_rows = new_global[:, None, :].expand_as(state.target_states)
        delta = self.target_update(torch.cat([state.target_states, global_rows, action_rows_expanded], dim=-1))
        new_targets = state.target_states + delta
        reward = self.reward(torch.cat([new_global, action], dim=-1)).squeeze(-1)
        return EncodedState(new_global, new_targets, state.valid_rows), reward
