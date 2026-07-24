from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from radar_dnn_mcts.env.features import CONTEXT_DIM, TOKEN_DIM


@dataclass
class EncodedState:
    """Global CLS representation, candidate rows, and their validity mask."""

    global_state: torch.Tensor
    target_states: torch.Tensor
    valid_rows: torch.Tensor

    def detach(self) -> "EncodedState":
        return EncodedState(self.global_state.detach(), self.target_states.detach(), self.valid_rows.detach())


class StateEncoder(nn.Module):
    """Encode search/target tokens into one global state and contextual target states."""

    def __init__(self, d_model: int = 96, nhead: int = 4, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.token_projection = nn.Linear(TOKEN_DIM, d_model)
        self.context_projection = nn.Sequential(nn.LayerNorm(CONTEXT_DIM), nn.Linear(CONTEXT_DIM, d_model), nn.GELU())
        self.cls = nn.Parameter(torch.randn(d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> EncodedState:
        """Contextualize every visible candidate together with a CLS token."""
        if tokens.ndim != 3 or context.ndim != 2:
            raise ValueError("tokens must be [batch, rows, features] and context [batch, features]")
        valid = tokens[:, :, 4] > 0.5
        # Row zero represents search and remains available independently of targets.
        valid = torch.cat([torch.ones_like(valid[:, :1]), valid[:, 1:]], dim=1)
        token_state = self.token_projection(tokens)
        # The learned CLS token carries window-level context through attention.
        cls = self.cls[None, None, :].expand(tokens.shape[0], 1, -1) + self.context_projection(context)[:, None, :]
        sequence = torch.cat([cls, token_state], dim=1)
        sequence_valid = torch.cat(
            [torch.ones(tokens.shape[0], 1, dtype=torch.bool, device=tokens.device), valid], dim=1
        )
        encoded = self.encoder(sequence, src_key_padding_mask=~sequence_valid)
        return EncodedState(encoded[:, 0], encoded[:, 1:], valid)
