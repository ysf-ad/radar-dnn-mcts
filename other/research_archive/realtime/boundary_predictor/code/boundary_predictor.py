"""Fixed-token boundary-state predictor."""

from __future__ import annotations

import math

import torch
from torch import nn


class BoundaryStatePredictor(nn.Module):
    """Predict boundary tokens from midpoint tokens and a known action suffix."""

    def __init__(
        self,
        token_dim: int,
        max_action_id: int,
        max_suffix: int,
        max_tokens: int = 101,
        use_row_embeddings: bool = True,
        d_model: int = 96,
        nhead: int = 4,
        layers: int = 2,
        max_time_ms: float = 200.0,
    ):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.token_dim = int(token_dim)
        self.max_suffix = int(max_suffix)
        self.max_tokens = int(max_tokens)
        self.use_row_embeddings = bool(use_row_embeddings)
        self.max_time_ms = float(max_time_ms)
        self.token_in = nn.Linear(token_dim, d_model)
        self.action_embedding = nn.Embedding(int(max_action_id) + 1, d_model)
        # S-only actions encode row r as 2*r. Sharing row identity between the
        # state tokens and suffix actions makes the transition target explicit.
        self.row_embedding = nn.Embedding(self.max_tokens, d_model) if self.use_row_embeddings else None
        self.action_position = nn.Parameter(torch.randn(max_suffix, d_model) * 0.02)
        self.time_encoder = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.delta_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, token_dim))
        # Start exactly at the causal carry-forward baseline. Training must earn
        # every deviation from the midpoint instead of injecting random drift.
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(
        self,
        midpoint_tokens: torch.Tensor,
        suffix_actions: torch.Tensor,
        suffix_mask: torch.Tensor,
        remaining_time_ms: torch.Tensor,
    ) -> torch.Tensor:
        if midpoint_tokens.ndim != 3:
            raise ValueError("midpoint_tokens must have shape [batch, tokens, features]")
        if suffix_actions.shape[1] != self.max_suffix:
            raise ValueError("suffix_actions has the wrong fixed width")
        if torch.any(suffix_actions < 0) or torch.any(suffix_actions >= self.action_embedding.num_embeddings):
            raise ValueError("suffix action ID is outside the configured vocabulary")
        token_count = midpoint_tokens.shape[1]
        if token_count > self.max_tokens:
            raise ValueError("midpoint token count exceeds max_tokens")
        token_state = self.token_in(midpoint_tokens)
        action_state = self.action_embedding(suffix_actions) + self.action_position.unsqueeze(0)
        if self.row_embedding is not None:
            token_rows = torch.arange(token_count, device=midpoint_tokens.device)
            action_rows = torch.div(suffix_actions, 2, rounding_mode="floor").clamp_max(self.max_tokens - 1)
            token_state = token_state + self.row_embedding(token_rows).unsqueeze(0)
            action_state = action_state + self.row_embedding(action_rows)
        scale = remaining_time_ms.float() / max(self.max_time_ms, 1e-6)
        time_features = torch.stack([scale, torch.log1p(scale.clamp_min(0.0))], dim=-1)
        time_state = self.time_encoder(time_features).unsqueeze(1)
        sequence = torch.cat([token_state, action_state, time_state], dim=1)
        padding = torch.cat(
            [
                torch.zeros(
                    (midpoint_tokens.shape[0], token_count),
                    dtype=torch.bool,
                    device=midpoint_tokens.device,
                ),
                ~suffix_mask.bool(),
                torch.zeros((midpoint_tokens.shape[0], 1), dtype=torch.bool, device=midpoint_tokens.device),
            ],
            dim=1,
        )
        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        return midpoint_tokens + self.delta_head(encoded[:, :token_count]) / math.sqrt(self.token_dim)
