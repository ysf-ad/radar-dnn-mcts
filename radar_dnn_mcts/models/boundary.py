from __future__ import annotations

import torch
from torch import nn

from radar_dnn_mcts.models.backbone import EncodedState


class BoundaryPredictor(nn.Module):
    """Predict the next window's latent state from a midpoint state and remaining actions."""

    def __init__(self, max_rows: int = 101, max_suffix: int = 32, d_model: int = 96, nhead: int = 4, layers: int = 2):
        super().__init__()
        self.max_suffix = int(max_suffix)
        self.action = nn.Embedding(max_rows, d_model)
        self.position = nn.Parameter(torch.randn(max_suffix, d_model) * 0.02)
        self.time = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, 0.0, "gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.global_delta = nn.Linear(d_model, d_model)
        self.target_delta = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.global_delta.weight)
        nn.init.zeros_(self.global_delta.bias)
        nn.init.zeros_(self.target_delta.weight)
        nn.init.zeros_(self.target_delta.bias)

    def forward(
        self,
        state: EncodedState,
        suffix_rows: torch.Tensor,
        suffix_mask: torch.Tensor,
        remaining_ms: torch.Tensor,
    ) -> EncodedState:
        suffix = self.action(suffix_rows.long()) + self.position[None, : suffix_rows.shape[1]]
        time = self.time((remaining_ms / 200.0).unsqueeze(-1))[:, None, :]
        sequence = torch.cat([state.global_state[:, None, :], state.target_states, suffix, time], dim=1)
        padding = torch.cat(
            [
                torch.zeros(state.global_state.shape[0], 1, dtype=torch.bool, device=sequence.device),
                ~state.valid_rows,
                ~suffix_mask.bool(),
                torch.zeros(state.global_state.shape[0], 1, dtype=torch.bool, device=sequence.device),
            ],
            dim=1,
        )
        mixed = self.encoder(sequence, src_key_padding_mask=padding)
        global_state = state.global_state + self.global_delta(mixed[:, 0])
        target_states = state.target_states + self.target_delta(mixed[:, 1 : 1 + state.target_states.shape[1]])
        return EncodedState(global_state, target_states, state.valid_rows)
