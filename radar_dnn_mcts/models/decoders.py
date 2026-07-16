from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from radar_dnn_mcts.env.features import CONTEXT_DIM
from radar_dnn_mcts.models.backbone import EncodedState, StateEncoder
from radar_dnn_mcts.models.scheduler import PolicyQOutput, RadarSchedulerModel


@dataclass
class SequenceOutput:
    policy_logits: torch.Tensor
    q_values: torch.Tensor
    valid_rows: torch.Tensor


class AutoregressiveDecoder(nn.Module):
    """Encode once, then condition each action score on the previously decoded prefix."""

    def __init__(self, backbone: RadarSchedulerModel, max_rows: int = 101):
        super().__init__()
        self.backbone = backbone
        d_model = backbone.d_model
        self.previous_action = nn.Embedding(max_rows, d_model)
        self.prefix_cell = nn.GRUCell(d_model, d_model)
        self.context_delta = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, CONTEXT_DIM))

    def initial(self, tokens: torch.Tensor, context: torch.Tensor) -> tuple[EncodedState, torch.Tensor]:
        state = self.backbone.encode(tokens, context)
        return state, state.global_state

    def step(
        self,
        state: EncodedState,
        prefix_state: torch.Tensor,
        context: torch.Tensor,
        previous_row: torch.Tensor,
        selected_rows: torch.Tensor,
    ) -> tuple[PolicyQOutput, torch.Tensor]:
        prefix_state = self.prefix_cell(self.previous_action(previous_row.long()), prefix_state)
        conditioned_context = context + self.context_delta(prefix_state)
        conditioned = EncodedState(state.global_state + prefix_state, state.target_states, state.valid_rows)
        return self.backbone.predict(conditioned, conditioned_context, selected_rows), prefix_state


class BatchDecoder(nn.Module):
    """Encode once and score all schedule positions in one parallel decoder pass."""

    def __init__(self, max_steps: int = 32, d_model: int = 96, nhead: int = 4, encoder_layers: int = 2, decoder_layers: int = 2):
        super().__init__()
        self.max_steps = int(max_steps)
        self.encoder = StateEncoder(d_model, nhead, encoder_layers)
        self.position = nn.Parameter(torch.randn(max_steps, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=decoder_layers)
        self.query_projection = nn.Linear(d_model, d_model)
        self.target_projection = nn.Linear(d_model, d_model)
        self.q_head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> SequenceOutput:
        state = self.encoder(tokens, context)
        queries = state.global_state[:, None, :] + self.position[None, :, :]
        decoded = self.decoder(queries, state.target_states, memory_key_padding_mask=~state.valid_rows)
        logits = torch.einsum(
            "bsd,brd->bsr", self.query_projection(decoded), self.target_projection(state.target_states)
        ) / self.query_projection.out_features**0.5
        valid = state.valid_rows[:, None, :].expand(-1, self.max_steps, -1)
        logits = logits.masked_fill(~valid, -1e9)
        action_state = state.target_states[:, None, :, :].expand(-1, self.max_steps, -1, -1)
        slot_state = decoded[:, :, None, :].expand_as(action_state)
        q = self.q_head(torch.cat([slot_state, action_state], dim=-1)).squeeze(-1).masked_fill(~valid, 0.0)
        return SequenceOutput(logits, q, valid)
