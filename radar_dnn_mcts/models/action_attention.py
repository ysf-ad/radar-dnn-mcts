from __future__ import annotations

import torch
from torch import nn

from radar_dnn_mcts.models.backbone import EncodedState


class ActionTokenMixer(nn.Module):
    """Build one candidate token per search/track row and mix candidates by self-attention."""

    def __init__(self, d_model: int = 96, nhead: int = 4, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.search_embedding = nn.Parameter(torch.randn(d_model) * 0.02)
        self.track_embedding = nn.Parameter(torch.randn(d_model) * 0.02)
        self.builder = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mixer = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)

    def forward(self, state: EncodedState, context_state: torch.Tensor, valid_rows: torch.Tensor) -> torch.Tensor:
        rows = state.target_states.shape[1]
        action_type = self.track_embedding[None, None, :].expand(state.target_states.shape[0], rows, -1).clone()
        action_type[:, 0] = self.search_embedding
        global_rows = (state.global_state + context_state)[:, None, :].expand(-1, rows, -1)
        candidates = self.builder(torch.cat([state.target_states, global_rows, action_type], dim=-1))
        return self.mixer(candidates, src_key_padding_mask=~valid_rows)
