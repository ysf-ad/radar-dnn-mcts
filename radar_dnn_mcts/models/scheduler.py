from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from radar_dnn_mcts.env.features import CONTEXT_DIM
from radar_dnn_mcts.models.action_attention import ActionTokenMixer
from radar_dnn_mcts.models.backbone import EncodedState, StateEncoder


@dataclass
class PolicyQOutput:
    policy_logits: torch.Tensor
    q_values: torch.Tensor
    valid_rows: torch.Tensor
    type_logits: torch.Tensor | None = None
    target_logits: torch.Tensor | None = None


class RadarSchedulerModel(nn.Module):
    """Shared state encoder, action attention, factorized policy, and complete-action Q."""

    def __init__(
        self,
        d_model: int = 96,
        nhead: int = 4,
        encoder_layers: int = 2,
        action_layers: int = 1,
        use_action_attention: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.encoder = StateEncoder(d_model, nhead, encoder_layers)
        self.context_projection = nn.Sequential(nn.LayerNorm(CONTEXT_DIM), nn.Linear(CONTEXT_DIM, d_model), nn.GELU())
        self.action_mixer = ActionTokenMixer(d_model, nhead, action_layers) if use_action_attention else None
        self.direct_action = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU())
        self.type_head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.target_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.q_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def encode(self, tokens: torch.Tensor, context: torch.Tensor) -> EncodedState:
        return self.encoder(tokens, context)

    def predict(
        self,
        state: EncodedState,
        context: torch.Tensor,
        selected_rows: torch.Tensor | None = None,
    ) -> PolicyQOutput:
        context_state = self.context_projection(context)
        valid = state.valid_rows.clone()
        if selected_rows is not None:
            valid = valid & ~selected_rows.bool()
            valid[:, 0] = True
        if self.action_mixer is not None:
            actions = self.action_mixer(state, context_state, valid)
        else:
            global_rows = (state.global_state + context_state)[:, None, :].expand_as(state.target_states)
            actions = self.direct_action(torch.cat([state.target_states, global_rows], dim=-1))

        global_context = torch.cat([state.global_state, context_state], dim=-1)
        type_logits = self.type_head(global_context)
        target_logits = self.target_head(actions).squeeze(-1)
        track_valid = valid.clone()
        track_valid[:, 0] = False
        target_log_prob = F.log_softmax(target_logits.masked_fill(~track_valid, -1e9), dim=-1)
        type_log_prob = F.log_softmax(type_logits, dim=-1)
        policy = target_log_prob + type_log_prob[:, 1:2]
        policy[:, 0] = type_log_prob[:, 0]
        policy = policy.masked_fill(~valid, -1e9)
        q = self.q_head(actions).squeeze(-1).masked_fill(~valid, 0.0)
        return PolicyQOutput(policy, q, valid, type_logits, target_logits)

    def forward(self, tokens: torch.Tensor, context: torch.Tensor, selected_rows: torch.Tensor | None = None) -> PolicyQOutput:
        return self.predict(self.encode(tokens, context), context, selected_rows)
