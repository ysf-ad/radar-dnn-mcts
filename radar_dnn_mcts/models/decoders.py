from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from radar_dnn_mcts.env.features import CONTEXT_DIM
from radar_dnn_mcts.models.action_attention import ActionTokenMixer
from radar_dnn_mcts.models.backbone import EncodedState, StateEncoder
from radar_dnn_mcts.models.scheduler import PolicyQOutput, RadarSchedulerModel


@dataclass
class SequenceOutput:
    policy_logits: torch.Tensor
    q_values: torch.Tensor
    valid_rows: torch.Tensor
    type_logits: torch.Tensor | None = None
    target_logits: torch.Tensor | None = None
    slot_logits: torch.Tensor | None = None


class AutoregressiveDecoder(nn.Module):
    """Causal Transformer constructor conditioned on one encoded radar state."""

    def __init__(
        self,
        backbone: RadarSchedulerModel,
        max_rows: int = 101,
        max_steps: int = 32,
        nhead: int = 4,
        decoder_layers: int = 2,
    ):
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        self.max_steps = int(max_steps)
        d_model = backbone.d_model
        self.action_identity = nn.Embedding(max_rows, d_model)
        self.action_type = nn.Embedding(2, d_model)
        self.position = nn.Embedding(max_steps + 1, d_model)
        self.bos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
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
        self.context_delta = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, CONTEXT_DIM))

    def initial(self, tokens: torch.Tensor, context: torch.Tensor) -> tuple[EncodedState, torch.Tensor]:
        state = self.backbone.encode(tokens, context)
        prefix = torch.empty(tokens.shape[0], 0, dtype=torch.long, device=tokens.device)
        return state, prefix

    def decode_prefix_deltas(
        self,
        state: EncodedState,
        prefix_rows: torch.Tensor,
    ) -> torch.Tensor:
        batch, prefix_length = prefix_rows.shape
        sequence = self.bos.expand(batch, -1, -1) + self.position.weight[0][None, None, :]
        if prefix_length:
            gather = prefix_rows[:, :, None].expand(-1, -1, state.target_states.shape[-1])
            target = state.target_states.gather(1, gather)
            positions = torch.arange(1, prefix_length + 1, device=prefix_rows.device)
            action_tokens = (
                target
                + self.action_identity(prefix_rows)
                + self.action_type((prefix_rows > 0).long())
                + self.position(positions)[None, :, :]
            )
            sequence = torch.cat([sequence, action_tokens], dim=1)
        length = sequence.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=sequence.device), diagonal=1
        )
        decoded = self.decoder(
            sequence,
            state.target_states,
            tgt_mask=causal_mask,
            memory_key_padding_mask=~state.valid_rows,
            tgt_is_causal=True,
        )
        return decoded - decoded[:, :1]

    def step(
        self,
        state: EncodedState,
        prefix_rows: torch.Tensor,
        context: torch.Tensor,
        selected_rows: torch.Tensor,
    ) -> PolicyQOutput:
        prefix_delta = self.decode_prefix_deltas(state, prefix_rows)[:, -1]
        if prefix_rows.shape[1]:
            context = context + self.context_delta(prefix_delta)
            global_state = state.global_state + prefix_delta
        else:
            global_state = state.global_state
        conditioned = EncodedState(global_state, state.target_states, state.valid_rows)
        return self.backbone.predict(conditioned, context, selected_rows)


class BatchDecoder(nn.Module):
    """Mix candidate actions once and score all positions in one parallel pass."""

    def __init__(self, max_steps: int = 32, d_model: int = 96, nhead: int = 4, encoder_layers: int = 2, decoder_layers: int = 2):
        super().__init__()
        self.max_steps = int(max_steps)
        self.encoder = StateEncoder(d_model, nhead, encoder_layers)
        self.context_projection = nn.Sequential(
            nn.LayerNorm(CONTEXT_DIM), nn.Linear(CONTEXT_DIM, d_model), nn.GELU()
        )
        self.action_mixer = ActionTokenMixer(d_model, nhead, layers=1)
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
        self.query_norm = nn.LayerNorm(d_model)
        self.target_norm = nn.LayerNorm(d_model)
        self.query_projection = nn.Linear(d_model, d_model)
        self.target_projection = nn.Linear(d_model, d_model)
        self.type_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2)
        )
        self.slot_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2)
        )
        self.q_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> SequenceOutput:
        state = self.encoder(tokens, context)
        action_memory = self.action_mixer(
            state, self.context_projection(context), state.valid_rows
        )
        slot_context = context[:, None, :].expand(-1, self.max_steps, -1).clone()
        slot_context[:, :, 0] = (
            torch.arange(self.max_steps, device=context.device, dtype=context.dtype) / 20.0
        ).clamp_max(1.0)
        context_delta = (
            self.context_projection(slot_context)
            - self.context_projection(context)[:, None, :]
        )
        queries = state.global_state[:, None, :] + self.position[None, :, :] + context_delta
        decoded = self.decoder(
            queries, action_memory, memory_key_padding_mask=~state.valid_rows
        )
        target_logits = torch.einsum(
            "bsd,brd->bsr",
            self.query_projection(self.query_norm(decoded)),
            self.target_projection(self.target_norm(action_memory)),
        ) / self.query_projection.out_features**0.5
        valid = state.valid_rows[:, None, :].expand(-1, self.max_steps, -1)
        target_logits = target_logits.masked_fill(~valid, -1e9)
        type_logits = self.type_head(decoded)
        track_valid = torch.cat(
            [torch.zeros_like(valid[:, :, :1]), valid[:, :, 1:]], dim=2
        )
        target_log_prob = torch.log_softmax(
            target_logits.masked_fill(~track_valid, -1e9), dim=-1
        )
        type_log_prob = torch.log_softmax(type_logits, dim=-1)
        track_logits = target_log_prob + type_log_prob[:, :, 1:2]
        logits = torch.cat([type_log_prob[:, :, :1], track_logits[:, :, 1:]], dim=2)
        logits = logits.masked_fill(~valid, -1e9)
        value = self.q_head(decoded).squeeze(-1)
        q = value[:, :, None].expand_as(logits).masked_fill(~valid, 0.0)
        return SequenceOutput(logits, q, valid, type_logits, target_logits, None)
