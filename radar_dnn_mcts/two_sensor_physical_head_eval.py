from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazero_orthodox import save_targets
from compare_action_heads_smoke import (
    TwoRowFactorizedNet,
    batch_tensors,
    flat_step,
    top_metrics,
    two_row_factorized_step,
    usable_targets,
)
from exact_env_mutual import (
    EDFPlanner,
    ESTPlanner,
    MAXT,
    attach_env_obs,
    engine_env_cfg,
    env_cfg_for,
    xs_decode_action,
    xs_s_search_action,
    xs_x_search_action,
    xs_s_track_action,
    xs_x_track_action,
)
from final_radar_campaign import get_obs, run_fixed, summarize_window_df
from mutual_features import slot_features, tokenize
from mutual_features import SLOT_DIM, TOKEN_DIM
from mutual_foundation import MutualRadarNet, SearchTarget
from penalty_window_quota_learner_eval import make_exact_args
from pufferlib.ocean.radarxs import binding
from realistic_reward_retrain import adapter
from repaired_campaign_tools import build_env, decode_sensor_action, execute_first_valid_action
from strict_window_report import sample_state_metrics


class BinaryTypeNet(nn.Module):
    def __init__(self, d_model: int = 48, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.token_proj = nn.Linear(TOKEN_DIM, d_model)
        self.slot_proj = nn.Sequential(nn.LayerNorm(SLOT_DIM), nn.Linear(SLOT_DIM, d_model), nn.GELU())
        self.cls = nn.Parameter(torch.randn(d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers, enable_nested_tensor=False, mask_check=False)
        self.head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.q_head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.value_head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, tokens: torch.Tensor, slot: torch.Tensor):
        active = tokens[:, :, 4] > 0.5
        active[:, 0] = True
        emb = self.token_proj(tokens)
        cls = self.cls[None, None, :].expand(tokens.shape[0], 1, -1)
        enc_in = torch.cat([cls, emb], dim=1)
        cls_valid = torch.ones((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device)
        enc = self.encoder(enc_in, src_key_padding_mask=~torch.cat([cls_valid, active], dim=1))
        z = torch.cat([enc[:, 0, :], self.slot_proj(slot)], dim=1)
        return self.head(z).squeeze(1), self.q_head(z), self.value_head(z).squeeze(1)


class TwoSensorTypeNet(nn.Module):
    def __init__(self, d_model: int = 48, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.token_proj = nn.Linear(TOKEN_DIM, d_model)
        self.slot_proj = nn.Sequential(nn.LayerNorm(SLOT_DIM), nn.Linear(SLOT_DIM, d_model), nn.GELU())
        self.cls = nn.Parameter(torch.randn(d_model) * 0.02)
        self.sensor_embed = nn.Parameter(torch.randn(2, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers, enable_nested_tensor=False, mask_check=False)
        self.head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.q_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.value_head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, tokens: torch.Tensor, slot: torch.Tensor):
        active = tokens[:, :, 4] > 0.5
        active[:, 0] = True
        emb = self.token_proj(tokens)
        cls = self.cls[None, None, :].expand(tokens.shape[0], 1, -1)
        enc_in = torch.cat([cls, emb], dim=1)
        cls_valid = torch.ones((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device)
        enc = self.encoder(enc_in, src_key_padding_mask=~torch.cat([cls_valid, active], dim=1))
        slot_emb = self.slot_proj(slot)
        sensor = self.sensor_embed[None, :, :].expand(tokens.shape[0], -1, -1)
        cls_s = enc[:, 0, None, :].expand(-1, 2, -1)
        slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
        ctx = torch.cat([cls_s, slot_s, sensor], dim=-1)
        value = self.value_head(torch.cat([enc[:, 0, :], slot_emb], dim=-1)).squeeze(1)
        return self.head(ctx).squeeze(-1), self.q_head(ctx), value


class SeparatedTypeTargetFactorizedNet(nn.Module):
    def __init__(self, d_model: int = 48, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.backbone = MutualRadarNet(d_model=d_model, nhead=nhead, nlayers=nlayers, head_arch="branch_context")
        self.sensor_embed = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.type_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.type_q_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.target_head = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.target_q_head = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.value_head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward_parts(self, tokens: torch.Tensor, slot: torch.Tensor):
        cls_out, tok_out, selected, token_active = self.backbone.encode_tokens(tokens)
        slot_emb = self.backbone.slot_proj(slot)
        bsz, rows, _d_model = tok_out.shape

        sensor = self.sensor_embed[None, :, :].expand(bsz, -1, -1)
        cls_s = cls_out[:, None, :].expand(-1, 2, -1)
        slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
        type_ctx = torch.cat([cls_s, slot_s, sensor], dim=-1)
        type_logits = self.type_head(type_ctx)
        type_q = self.type_q_head(type_ctx)

        tok_st = tok_out[:, :, None, :].expand(-1, -1, 2, -1)
        cls_st = cls_out[:, None, None, :].expand(-1, rows, 2, -1)
        slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
        sensor_st = self.sensor_embed[None, None, :, :].expand(bsz, rows, -1, -1)
        target_ctx = torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)
        target_logits = self.target_head(target_ctx).squeeze(-1)
        target_q = self.target_q_head(target_ctx).squeeze(-1)
        return type_logits, type_q, target_logits, target_q, selected, token_active, cls_out, slot_emb

    def forward_scores(self, tokens: torch.Tensor, slot: torch.Tensor):
        type_logits, type_q, target_logits, target_q, selected, token_active, _cls_out, _slot_emb = self.forward_parts(tokens, slot)
        bsz, rows, _sensors = target_logits.shape
        scores = tokens.new_full((bsz, rows, 2), -1e9)
        q = tokens.new_zeros((bsz, rows, 2))
        scores[:, 0, :] = type_logits[:, :, 0]
        q[:, 0, :] = type_q[:, :, 0]
        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        scores[:, 1:, :] = (type_logits[:, None, :, 1] + target_logits)[:, 1:, :]
        q[:, 1:, :] = (type_q[:, None, :, 1] + target_q)[:, 1:, :]
        row_is_search = torch.arange(rows, device=tokens.device)[None, :, None] == 0
        valid = track_mask[:, :, None] | row_is_search
        scores = scores.masked_fill(~valid, -1e9)
        q = q.masked_fill(~valid, 0.0)
        return scores, q

    def forward_value(self, tokens: torch.Tensor, slot: torch.Tensor):
        cls_out, _tok_out, _selected, _token_active = self.backbone.encode_tokens(tokens)
        slot_emb = self.backbone.slot_proj(slot)
        return self.value_head(torch.cat([cls_out, slot_emb], dim=-1)).squeeze(-1)


class MultiSearchFlatNet(nn.Module):
    def __init__(self, d_model: int = 48, nhead: int = 4, nlayers: int = 2, search_slots: int = 20):
        super().__init__()
        self.search_slots = int(search_slots)
        self.base = MutualRadarNet(d_model=d_model, nhead=nhead, nlayers=nlayers, head_arch="branch_context")
        self.search_slot_embed = nn.Parameter(torch.randn(self.search_slots, d_model) * 0.02)
        self.search_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.search_q_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))

    def forward_search20(self, tokens: torch.Tensor, slot: torch.Tensor):
        token_active = tokens[:, :, 4] > 0.5
        token_active[:, 0] = True
        selected = tokens[:, :, 8] > 0.5
        cls_out, tok_out, _, _ = self.base.encode_tokens(tokens)
        slot_emb = self.base.slot_proj(slot)
        cls_rep = cls_out.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
        slot_rep = slot_emb.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
        track_ctx = torch.cat([tok_out, cls_rep, slot_rep], dim=-1)
        track_logits = self.base.physical_flat_head(track_ctx)
        track_q = self.base.physical_flat_q_head(track_ctx)
        action_mask = token_active & ~selected
        action_mask[:, 0] = False
        track_logits = track_logits.masked_fill(~action_mask[:, :, None], -1e9)
        track_q = track_q.masked_fill(~action_mask[:, :, None], 0.0)

        search_rep = self.search_slot_embed.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        cls_s = cls_out.unsqueeze(1).expand(-1, self.search_slots, -1)
        slot_s = slot_emb.unsqueeze(1).expand(-1, self.search_slots, -1)
        search_ctx = torch.cat([search_rep, cls_s, slot_s], dim=-1)
        search_logits = self.search_head(search_ctx)
        search_q = self.search_q_head(search_ctx)
        value = self.base.value_head(torch.cat([cls_out, slot_emb], dim=-1)).squeeze(-1)
        return track_logits, track_q, search_logits, search_q, value


class FlatActionAttentionNet(nn.Module):
    def __init__(self, d_model: int = 48, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.base = MutualRadarNet(d_model=d_model, nhead=nhead, nlayers=nlayers, head_arch="branch_context")
        self.sensor_embed = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.action_proj = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, d_model), nn.GELU())
        action_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
        )
        self.action_coupler = nn.TransformerEncoder(action_layer, num_layers=1, enable_nested_tensor=False, mask_check=False)
        self.policy_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.q_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward_scores(self, tokens: torch.Tensor, slot: torch.Tensor):
        token_active = tokens[:, :, 4] > 0.5
        token_active[:, 0] = True
        selected = tokens[:, :, 8] > 0.5
        cls_out, tok_out, _, _ = self.base.encode_tokens(tokens)
        slot_emb = self.base.slot_proj(slot)
        bsz, rows, _d_model = tok_out.shape

        tok_st = tok_out[:, :, None, :].expand(-1, -1, 2, -1)
        cls_st = cls_out[:, None, None, :].expand(-1, rows, 2, -1)
        slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
        sensor_st = self.sensor_embed[None, None, :, :].expand(bsz, rows, -1, -1)
        action_ctx = self.action_proj(torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1))

        row_is_search = torch.arange(rows, device=tokens.device)[None, :, None] == 0
        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        valid = (track_mask[:, :, None] | row_is_search).expand(-1, -1, 2)
        mixed = self.action_coupler(action_ctx.reshape(bsz, rows * 2, -1), src_key_padding_mask=~valid.reshape(bsz, rows * 2))
        mixed = mixed.reshape(bsz, rows, 2, -1)
        logits = self.policy_head(mixed).squeeze(-1).masked_fill(~valid, -1e9)
        q = self.q_head(mixed).squeeze(-1).masked_fill(~valid, 0.0)
        return logits, q

    def forward_value(self, tokens: torch.Tensor, slot: torch.Tensor):
        cls_out, _tok_out, _selected, _token_active = self.base.encode_tokens(tokens)
        slot_emb = self.base.slot_proj(slot)
        return self.base.value_head(torch.cat([cls_out, slot_emb], dim=-1)).squeeze(-1)


def flat_search_slots(variant: str) -> int | None:
    m = re.fullmatch(r"flat_search(\d+)", str(variant))
    if not m:
        return None
    slots = int(m.group(1))
    if slots <= 0:
        raise ValueError(f"flat search slots must be positive: {variant}")
    return slots


class JointTwoRowFactorizedNet(TwoRowFactorizedNet):
    def __init__(self, d_model: int = 48, nhead: int = 4, nlayers: int = 2, head_arch: str = "branch_context"):
        super().__init__(d_model=d_model, nhead=nhead, nlayers=nlayers, head_arch=head_arch)
        self.joint_policy_head = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.joint_utility_head = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.joint_policy_scale = nn.Parameter(torch.tensor(-2.0))
        self.joint_utility_scale = nn.Parameter(torch.tensor(-2.0))

    def forward_scores(self, tokens: torch.Tensor, slot: torch.Tensor):
        cls_out, tok_out, selected, token_active = self.backbone.encode_tokens(tokens)
        slot_emb = self.backbone.slot_proj(slot)
        bsz, rows, _d_model = tok_out.shape

        sensor = self.sensor_embed[None, :, :].expand(bsz, -1, -1)
        cls_s = cls_out[:, None, :].expand(-1, 2, -1)
        slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
        type_ctx = torch.cat([cls_s, slot_s, sensor], dim=-1)
        type_logits = self.type_head(type_ctx)
        type_q = self.type_q_head(type_ctx)

        tok_st = tok_out[:, :, None, :].expand(-1, -1, 2, -1)
        cls_st = cls_out[:, None, None, :].expand(-1, rows, 2, -1)
        slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
        sensor_st = self.sensor_embed[None, None, :, :].expand(bsz, rows, -1, -1)
        target_ctx = torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)
        target_logits = self.target_head(target_ctx).squeeze(-1)
        target_q = self.target_q_head(target_ctx).squeeze(-1)
        joint_policy = self.joint_policy_head(target_ctx).squeeze(-1)
        joint_utility = self.joint_utility_head(target_ctx).squeeze(-1)

        p_scale = 0.25 * torch.sigmoid(self.joint_policy_scale)
        u_scale = 0.50 * torch.sigmoid(self.joint_utility_scale)
        scores = tokens.new_full((bsz, rows, 2), -1e9)
        q = tokens.new_zeros((bsz, rows, 2))
        utility = tokens.new_zeros((bsz, rows, 2))
        scores[:, 0, :] = type_logits[:, :, 0]
        q[:, 0, :] = type_q[:, :, 0]
        utility[:, 0, :] = type_q[:, :, 0]
        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        scores[:, 1:, :] = (type_logits[:, None, :, 1] + target_logits + p_scale * joint_policy)[:, 1:, :]
        q[:, 1:, :] = (type_q[:, None, :, 1] + target_q)[:, 1:, :]
        utility[:, 1:, :] = (type_q[:, None, :, 1] + target_q + u_scale * joint_utility)[:, 1:, :]
        row_is_search = torch.arange(rows, device=tokens.device)[None, :, None] == 0
        invalid = (~track_mask[:, :, None]) & (~row_is_search)
        scores = scores.masked_fill(invalid, -1e9)
        q = q.masked_fill(invalid, 0.0)
        utility = utility.masked_fill(invalid, 0.0)
        return scores, q, utility


class StrongJointTwoRowFactorizedNet(TwoRowFactorizedNet):
    def __init__(self, d_model: int = 48, nhead: int = 4, nlayers: int = 2, head_arch: str = "branch_context"):
        super().__init__(d_model=d_model, nhead=nhead, nlayers=nlayers, head_arch=head_arch)
        self.compat_head = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.compat_head[-1].weight)
        nn.init.zeros_(self.compat_head[-1].bias)

    def forward_scores(self, tokens: torch.Tensor, slot: torch.Tensor):
        cls_out, tok_out, selected, token_active = self.backbone.encode_tokens(tokens)
        slot_emb = self.backbone.slot_proj(slot)
        bsz, rows, _d_model = tok_out.shape

        sensor = self.sensor_embed[None, :, :].expand(bsz, -1, -1)
        cls_s = cls_out[:, None, :].expand(-1, 2, -1)
        slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
        type_ctx = torch.cat([cls_s, slot_s, sensor], dim=-1)
        type_logits = self.type_head(type_ctx)
        type_q = self.type_q_head(type_ctx)

        tok_st = tok_out[:, :, None, :].expand(-1, -1, 2, -1)
        cls_st = cls_out[:, None, None, :].expand(-1, rows, 2, -1)
        slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
        sensor_st = self.sensor_embed[None, None, :, :].expand(bsz, rows, -1, -1)
        target_ctx = torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)
        target_logits = self.target_head(target_ctx).squeeze(-1)
        target_q = self.target_q_head(target_ctx).squeeze(-1)
        compat = self.compat_head(target_ctx).squeeze(-1)

        scores = tokens.new_full((bsz, rows, 2), -1e9)
        q = tokens.new_zeros((bsz, rows, 2))
        scores[:, 0, :] = type_logits[:, :, 0]
        q[:, 0, :] = type_q[:, :, 0]
        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        scores[:, 1:, :] = (type_logits[:, None, :, 1] + target_logits + compat)[:, 1:, :]
        q[:, 1:, :] = (type_q[:, None, :, 1] + target_q)[:, 1:, :]
        row_is_search = torch.arange(rows, device=tokens.device)[None, :, None] == 0
        invalid = (~track_mask[:, :, None]) & (~row_is_search)
        scores = scores.masked_fill(invalid, -1e9)
        q = q.masked_fill(invalid, 0.0)
        return scores, q


class CoupledTwoRowFactorizedNet(TwoRowFactorizedNet):
    def __init__(self, d_model: int = 48, nhead: int = 4, nlayers: int = 2, head_arch: str = "branch_context"):
        super().__init__(d_model=d_model, nhead=nhead, nlayers=nlayers, head_arch=head_arch)
        self.sensor_state_proj = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
        )
        self.sensor_coupler = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False, mask_check=False)
        self.compat_head = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.compat_head[-1].weight)
        nn.init.zeros_(self.compat_head[-1].bias)

    def forward_scores(self, tokens: torch.Tensor, slot: torch.Tensor):
        cls_out, tok_out, selected, token_active = self.backbone.encode_tokens(tokens)
        slot_emb = self.backbone.slot_proj(slot)
        bsz, rows, _d_model = tok_out.shape

        sensor = self.sensor_embed[None, :, :].expand(bsz, -1, -1)
        cls_s = cls_out[:, None, :].expand(-1, 2, -1)
        slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
        sensor_state = self.sensor_state_proj(torch.cat([cls_s, slot_s, sensor], dim=-1))
        coupled_sensor = self.sensor_coupler(sensor_state)
        type_ctx = torch.cat([cls_s, slot_s, coupled_sensor], dim=-1)
        type_logits = self.type_head(type_ctx)
        type_q = self.type_q_head(type_ctx)

        tok_st = tok_out[:, :, None, :].expand(-1, -1, 2, -1)
        cls_st = cls_out[:, None, None, :].expand(-1, rows, 2, -1)
        slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
        sensor_st = coupled_sensor[:, None, :, :].expand(bsz, rows, -1, -1)
        target_ctx = torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)
        target_logits = self.target_head(target_ctx).squeeze(-1)
        target_q = self.target_q_head(target_ctx).squeeze(-1)
        compat = self.compat_head(target_ctx).squeeze(-1)

        scores = tokens.new_full((bsz, rows, 2), -1e9)
        q = tokens.new_zeros((bsz, rows, 2))
        scores[:, 0, :] = type_logits[:, :, 0]
        q[:, 0, :] = type_q[:, :, 0]
        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        scores[:, 1:, :] = (type_logits[:, None, :, 1] + target_logits + compat)[:, 1:, :]
        q[:, 1:, :] = (type_q[:, None, :, 1] + target_q)[:, 1:, :]
        row_is_search = torch.arange(rows, device=tokens.device)[None, :, None] == 0
        invalid = (~track_mask[:, :, None]) & (~row_is_search)
        scores = scores.masked_fill(invalid, -1e9)
        q = q.masked_fill(invalid, 0.0)
        return scores, q


class ActionAttentionFactorizedNet(TwoRowFactorizedNet):
    def __init__(self, d_model: int = 48, nhead: int = 4, nlayers: int = 2, head_arch: str = "branch_context"):
        super().__init__(d_model=d_model, nhead=nhead, nlayers=nlayers, head_arch=head_arch)
        self.sensor_state_proj = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU())
        sensor_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
        )
        self.sensor_coupler = nn.TransformerEncoder(sensor_layer, num_layers=1, enable_nested_tensor=False, mask_check=False)
        self.action_proj = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, d_model), nn.GELU())
        action_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
        )
        self.action_coupler = nn.TransformerEncoder(action_layer, num_layers=1, enable_nested_tensor=False, mask_check=False)
        self.action_policy_residual = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.action_q_residual = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        nn.init.zeros_(self.action_policy_residual[-1].weight)
        nn.init.zeros_(self.action_policy_residual[-1].bias)
        nn.init.zeros_(self.action_q_residual[-1].weight)
        nn.init.zeros_(self.action_q_residual[-1].bias)

    def forward_scores(self, tokens: torch.Tensor, slot: torch.Tensor):
        cls_out, tok_out, selected, token_active = self.backbone.encode_tokens(tokens)
        slot_emb = self.backbone.slot_proj(slot)
        bsz, rows, _d_model = tok_out.shape

        sensor = self.sensor_embed[None, :, :].expand(bsz, -1, -1)
        cls_s = cls_out[:, None, :].expand(-1, 2, -1)
        slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
        sensor_state = self.sensor_state_proj(torch.cat([cls_s, slot_s, sensor], dim=-1))
        coupled_sensor = self.sensor_coupler(sensor_state)
        type_ctx = torch.cat([cls_s, slot_s, coupled_sensor], dim=-1)
        type_logits = self.type_head(type_ctx)
        type_q = self.type_q_head(type_ctx)

        tok_st = tok_out[:, :, None, :].expand(-1, -1, 2, -1)
        cls_st = cls_out[:, None, None, :].expand(-1, rows, 2, -1)
        slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
        sensor_st = coupled_sensor[:, None, :, :].expand(bsz, rows, -1, -1)
        target_ctx = torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)
        target_logits = self.target_head(target_ctx).squeeze(-1)
        target_q = self.target_q_head(target_ctx).squeeze(-1)

        base_scores = tokens.new_full((bsz, rows, 2), -1e9)
        base_q = tokens.new_zeros((bsz, rows, 2))
        base_scores[:, 0, :] = type_logits[:, :, 0]
        base_q[:, 0, :] = type_q[:, :, 0]
        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        base_scores[:, 1:, :] = (type_logits[:, None, :, 1] + target_logits)[:, 1:, :]
        base_q[:, 1:, :] = (type_q[:, None, :, 1] + target_q)[:, 1:, :]

        row_is_search = torch.arange(rows, device=tokens.device)[None, :, None] == 0
        valid = (track_mask[:, :, None] | row_is_search).expand(-1, -1, 2)
        action_ctx = self.action_proj(target_ctx).reshape(bsz, rows * 2, -1)
        action_valid = valid.reshape(bsz, rows * 2)
        action_ctx = self.action_coupler(action_ctx, src_key_padding_mask=~action_valid)
        residual = self.action_policy_residual(action_ctx).reshape(bsz, rows, 2)
        q_residual = self.action_q_residual(action_ctx).reshape(bsz, rows, 2)
        scores = (base_scores + residual).masked_fill(~valid, -1e9)
        q = (base_q + q_residual).masked_fill(~valid, 0.0)
        return scores, q


class AutoregressiveActionAttentionFactorizedNet(ActionAttentionFactorizedNet):
    """AlphaStar-style ordered action head: S action first, then X conditioned on S."""

    def __init__(self, d_model: int = 48, nhead: int = 4, nlayers: int = 2, head_arch: str = "branch_context"):
        super().__init__(d_model=d_model, nhead=nhead, nlayers=nlayers, head_arch=head_arch)
        self.s_to_x_policy = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.s_to_x_q = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        nn.init.zeros_(self.s_to_x_policy[-1].weight)
        nn.init.zeros_(self.s_to_x_policy[-1].bias)
        nn.init.zeros_(self.s_to_x_q[-1].weight)
        nn.init.zeros_(self.s_to_x_q[-1].bias)

    def _base_action_context(self, tokens: torch.Tensor, slot: torch.Tensor):
        cls_out, tok_out, selected, token_active = self.backbone.encode_tokens(tokens)
        slot_emb = self.backbone.slot_proj(slot)
        bsz, rows, _d_model = tok_out.shape

        sensor = self.sensor_embed[None, :, :].expand(bsz, -1, -1)
        cls_s = cls_out[:, None, :].expand(-1, 2, -1)
        slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
        sensor_state = self.sensor_state_proj(torch.cat([cls_s, slot_s, sensor], dim=-1))
        coupled_sensor = self.sensor_coupler(sensor_state)
        type_ctx = torch.cat([cls_s, slot_s, coupled_sensor], dim=-1)
        type_logits = self.type_head(type_ctx)
        type_q = self.type_q_head(type_ctx)

        tok_st = tok_out[:, :, None, :].expand(-1, -1, 2, -1)
        cls_st = cls_out[:, None, None, :].expand(-1, rows, 2, -1)
        slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
        sensor_st = coupled_sensor[:, None, :, :].expand(bsz, rows, -1, -1)
        target_ctx = torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)
        target_logits = self.target_head(target_ctx).squeeze(-1)
        target_q = self.target_q_head(target_ctx).squeeze(-1)

        base_scores = tokens.new_full((bsz, rows, 2), -1e9)
        base_q = tokens.new_zeros((bsz, rows, 2))
        base_scores[:, 0, :] = type_logits[:, :, 0]
        base_q[:, 0, :] = type_q[:, :, 0]
        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        base_scores[:, 1:, :] = (type_logits[:, None, :, 1] + target_logits)[:, 1:, :]
        base_q[:, 1:, :] = (type_q[:, None, :, 1] + target_q)[:, 1:, :]

        row_is_search = torch.arange(rows, device=tokens.device)[None, :, None] == 0
        valid = (track_mask[:, :, None] | row_is_search).expand(-1, -1, 2)
        action_ctx = self.action_proj(target_ctx).reshape(bsz, rows * 2, -1)
        action_valid = valid.reshape(bsz, rows * 2)
        action_ctx = self.action_coupler(action_ctx, src_key_padding_mask=~action_valid).reshape(bsz, rows, 2, -1)
        residual = self.action_policy_residual(action_ctx).reshape(bsz, rows, 2)
        q_residual = self.action_q_residual(action_ctx).reshape(bsz, rows, 2)
        scores = (base_scores + residual).masked_fill(~valid, -1e9)
        q = (base_q + q_residual).masked_fill(~valid, 0.0)
        return scores, q, action_ctx, valid

    def forward_scores(self, tokens: torch.Tensor, slot: torch.Tensor):
        scores, q, action_ctx, valid = self._base_action_context(tokens, slot)
        s_log_probs = F.log_softmax(scores[:, :, 0], dim=1)
        s_context = torch.sum(s_log_probs.exp().unsqueeze(-1) * action_ctx[:, :, 0, :], dim=1)
        x_context = torch.cat([action_ctx[:, :, 1, :], s_context[:, None, :].expand(-1, action_ctx.shape[1], -1)], dim=-1)
        scores = scores.clone()
        q = q.clone()
        scores[:, :, 1] = scores[:, :, 1] + self.s_to_x_policy(x_context).squeeze(-1)
        q[:, :, 1] = q[:, :, 1] + self.s_to_x_q(x_context).squeeze(-1)
        return scores.masked_fill(~valid, -1e9), q.masked_fill(~valid, 0.0)

    def forward_x_conditioned_on_s(self, tokens: torch.Tensor, slot: torch.Tensor, s_rows: torch.Tensor):
        scores, q, action_ctx, valid = self._base_action_context(tokens, slot)
        bidx = torch.arange(tokens.shape[0], device=tokens.device)
        s_rows = s_rows.to(device=tokens.device, dtype=torch.long).clamp(0, scores.shape[1] - 1)
        s_context = action_ctx[bidx, s_rows, 0, :]
        x_context = torch.cat([action_ctx[:, :, 1, :], s_context[:, None, :].expand(-1, action_ctx.shape[1], -1)], dim=-1)
        x_scores = scores[:, :, 1] + self.s_to_x_policy(x_context).squeeze(-1)
        x_q = q[:, :, 1] + self.s_to_x_q(x_context).squeeze(-1)
        return scores[:, :, 0].masked_fill(~valid[:, :, 0], -1e9), q[:, :, 0], x_scores.masked_fill(~valid[:, :, 1], -1e9), x_q.masked_fill(~valid[:, :, 1], 0.0)

    def forward_scores_teacher(self, tokens: torch.Tensor, slot: torch.Tensor, sensor_pi: torch.Tensor):
        scores, q, action_ctx, valid = self._base_action_context(tokens, slot)
        s_rows = sensor_pi[:, :, 0].argmax(dim=1)
        s_context = action_ctx[torch.arange(tokens.shape[0], device=tokens.device), s_rows, 0, :]
        x_context = torch.cat([action_ctx[:, :, 1, :], s_context[:, None, :].expand(-1, action_ctx.shape[1], -1)], dim=-1)
        scores = scores.clone()
        q = q.clone()
        scores[:, :, 1] = scores[:, :, 1] + self.s_to_x_policy(x_context).squeeze(-1)
        q[:, :, 1] = q[:, :, 1] + self.s_to_x_q(x_context).squeeze(-1)
        return scores.masked_fill(~valid, -1e9), q.masked_fill(~valid, 0.0)


class AlphaStarFactorizedNet(ActionAttentionFactorizedNet):
    """Explicit action-type -> target-argument autoregressive head for the two sensors."""

    def __init__(self, d_model: int = 48, nhead: int = 4, nlayers: int = 2, head_arch: str = "branch_context"):
        super().__init__(d_model=d_model, nhead=nhead, nlayers=nlayers, head_arch=head_arch)
        self.arg_type_embed = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.x_type_cond = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.x_target_cond = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        nn.init.zeros_(self.x_type_cond[-1].weight)
        nn.init.zeros_(self.x_type_cond[-1].bias)
        nn.init.zeros_(self.x_target_cond[-1].weight)
        nn.init.zeros_(self.x_target_cond[-1].bias)

    def _parts(self, tokens: torch.Tensor, slot: torch.Tensor):
        cls_out, tok_out, selected, token_active = self.backbone.encode_tokens(tokens)
        slot_emb = self.backbone.slot_proj(slot)
        bsz, rows, _d_model = tok_out.shape

        sensor = self.sensor_embed[None, :, :].expand(bsz, -1, -1)
        cls_s = cls_out[:, None, :].expand(-1, 2, -1)
        slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
        sensor_state = self.sensor_state_proj(torch.cat([cls_s, slot_s, sensor], dim=-1))
        coupled_sensor = self.sensor_coupler(sensor_state)
        type_ctx = torch.cat([cls_s, slot_s, coupled_sensor], dim=-1)
        type_logits = self.type_head(type_ctx)
        type_q = self.type_q_head(type_ctx)

        tok_st = tok_out[:, :, None, :].expand(-1, -1, 2, -1)
        cls_st = cls_out[:, None, None, :].expand(-1, rows, 2, -1)
        slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
        sensor_st = coupled_sensor[:, None, :, :].expand(bsz, rows, -1, -1)
        target_ctx = torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)
        target_logits = self.target_head(target_ctx).squeeze(-1)
        target_q = self.target_q_head(target_ctx).squeeze(-1)
        action_ctx = self.action_proj(target_ctx).reshape(bsz, rows * 2, -1)

        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        row_is_search = torch.arange(rows, device=tokens.device)[None, :, None] == 0
        valid = (track_mask[:, :, None] | row_is_search).expand(-1, -1, 2)
        action_ctx = self.action_coupler(action_ctx, src_key_padding_mask=~valid.reshape(bsz, rows * 2)).reshape(bsz, rows, 2, -1)
        return type_ctx, type_logits, type_q, target_logits, target_q, action_ctx, valid

    def _selected_s_context(self, action_ctx: torch.Tensor, sensor_pi: torch.Tensor | None = None, s_rows: torch.Tensor | None = None):
        bsz = action_ctx.shape[0]
        if s_rows is None:
            if sensor_pi is None:
                s_rows = torch.zeros((bsz,), dtype=torch.long, device=action_ctx.device)
            else:
                s_rows = sensor_pi[:, :, 0].argmax(dim=1).to(torch.long)
        s_rows = s_rows.to(device=action_ctx.device, dtype=torch.long).clamp(0, action_ctx.shape[1] - 1)
        return action_ctx[torch.arange(bsz, device=action_ctx.device), s_rows, 0, :], s_rows

    def forward_scores_teacher(self, tokens: torch.Tensor, slot: torch.Tensor, sensor_pi: torch.Tensor):
        type_ctx, type_logits, type_q, target_logits, target_q, action_ctx, valid = self._parts(tokens, slot)
        s_ctx, _s_rows = self._selected_s_context(action_ctx, sensor_pi=sensor_pi)
        x_type_logits = type_logits[:, 1, :] + self.x_type_cond(torch.cat([type_ctx[:, 1, :], s_ctx], dim=-1))
        x_target_ctx = torch.cat(
            [
                action_ctx[:, :, 1, :],
                s_ctx[:, None, :].expand(-1, action_ctx.shape[1], -1),
                self.arg_type_embed[1][None, None, :].expand(action_ctx.shape[0], action_ctx.shape[1], -1),
            ],
            dim=-1,
        )
        x_target_logits = target_logits[:, :, 1] + self.x_target_cond(x_target_ctx).squeeze(-1)
        return self._compose_scores(type_logits[:, 0, :], x_type_logits, target_logits[:, :, 0], x_target_logits, type_q, target_q, valid)

    def forward_scores_conditioned(self, tokens: torch.Tensor, slot: torch.Tensor, s_rows: torch.Tensor):
        type_ctx, type_logits, type_q, target_logits, target_q, action_ctx, valid = self._parts(tokens, slot)
        s_ctx, _s_rows = self._selected_s_context(action_ctx, s_rows=s_rows)
        x_type_logits = type_logits[:, 1, :] + self.x_type_cond(torch.cat([type_ctx[:, 1, :], s_ctx], dim=-1))
        x_target_ctx = torch.cat(
            [
                action_ctx[:, :, 1, :],
                s_ctx[:, None, :].expand(-1, action_ctx.shape[1], -1),
                self.arg_type_embed[1][None, None, :].expand(action_ctx.shape[0], action_ctx.shape[1], -1),
            ],
            dim=-1,
        )
        x_target_logits = target_logits[:, :, 1] + self.x_target_cond(x_target_ctx).squeeze(-1)
        return self._compose_scores(type_logits[:, 0, :], x_type_logits, target_logits[:, :, 0], x_target_logits, type_q, target_q, valid)

    def _compose_scores(self, s_type_logits, x_type_logits, s_target_logits, x_target_logits, type_q, target_q, valid):
        bsz, rows = s_target_logits.shape
        scores = s_target_logits.new_full((bsz, rows, 2), -1e9)
        q = s_target_logits.new_zeros((bsz, rows, 2))
        scores[:, 0, 0] = s_type_logits[:, 0]
        scores[:, 1:, 0] = s_type_logits[:, 1, None] + s_target_logits[:, 1:]
        scores[:, 0, 1] = x_type_logits[:, 0]
        scores[:, 1:, 1] = x_type_logits[:, 1, None] + x_target_logits[:, 1:]
        q[:, 0, :] = type_q[:, :, 0]
        q[:, 1:, :] = (type_q[:, None, :, 1] + target_q)[:, 1:, :]
        return scores.masked_fill(~valid, -1e9), q.masked_fill(~valid, 0.0), {
            "s_type_logits": s_type_logits,
            "x_type_logits": x_type_logits,
            "s_target_logits": s_target_logits,
            "x_target_logits": x_target_logits,
            "valid": valid,
        }

    def forward_scores(self, tokens: torch.Tensor, slot: torch.Tensor):
        scores, q, _aux = self.forward_scores_conditioned(tokens, slot, torch.zeros((tokens.shape[0],), dtype=torch.long, device=tokens.device))
        return scores, q


class FullSharedActionEncoderFactorizedNet(ActionAttentionFactorizedNet):
    def __init__(self, d_model: int = 48, nhead: int = 4, nlayers: int = 2, head_arch: str = "branch_context"):
        super().__init__(d_model=d_model, nhead=nhead, nlayers=nlayers, head_arch=head_arch)
        full_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
        )
        self.full_action_state_encoder = nn.TransformerEncoder(full_layer, num_layers=1, enable_nested_tensor=False, mask_check=False)

    def forward_scores(self, tokens: torch.Tensor, slot: torch.Tensor):
        cls_out, tok_out, selected, token_active = self.backbone.encode_tokens(tokens)
        slot_emb = self.backbone.slot_proj(slot)
        bsz, rows, _d_model = tok_out.shape

        sensor = self.sensor_embed[None, :, :].expand(bsz, -1, -1)
        cls_s = cls_out[:, None, :].expand(-1, 2, -1)
        slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
        sensor_state = self.sensor_state_proj(torch.cat([cls_s, slot_s, sensor], dim=-1))
        coupled_sensor = self.sensor_coupler(sensor_state)
        type_ctx = torch.cat([cls_s, slot_s, coupled_sensor], dim=-1)
        type_logits = self.type_head(type_ctx)
        type_q = self.type_q_head(type_ctx)

        tok_st = tok_out[:, :, None, :].expand(-1, -1, 2, -1)
        cls_st = cls_out[:, None, None, :].expand(-1, rows, 2, -1)
        slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
        sensor_st = coupled_sensor[:, None, :, :].expand(bsz, rows, -1, -1)
        target_ctx = torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)
        target_logits = self.target_head(target_ctx).squeeze(-1)
        target_q = self.target_q_head(target_ctx).squeeze(-1)

        base_scores = tokens.new_full((bsz, rows, 2), -1e9)
        base_q = tokens.new_zeros((bsz, rows, 2))
        base_scores[:, 0, :] = type_logits[:, :, 0]
        base_q[:, 0, :] = type_q[:, :, 0]
        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        base_scores[:, 1:, :] = (type_logits[:, None, :, 1] + target_logits)[:, 1:, :]
        base_q[:, 1:, :] = (type_q[:, None, :, 1] + target_q)[:, 1:, :]

        row_is_search = torch.arange(rows, device=tokens.device)[None, :, None] == 0
        valid = (track_mask[:, :, None] | row_is_search).expand(-1, -1, 2)
        action_ctx = self.action_proj(target_ctx).reshape(bsz, rows * 2, -1)
        action_valid = valid.reshape(bsz, rows * 2)
        state_ctx = torch.cat([cls_out[:, None, :], tok_out], dim=1)
        state_valid = torch.cat(
            [torch.ones((bsz, 1), dtype=torch.bool, device=tokens.device), token_active],
            dim=1,
        )
        mixed = torch.cat([state_ctx, action_ctx], dim=1)
        mixed_valid = torch.cat([state_valid, action_valid], dim=1)
        mixed = self.full_action_state_encoder(mixed, src_key_padding_mask=~mixed_valid)
        action_out = mixed[:, state_ctx.shape[1] :, :]
        residual = self.action_policy_residual(action_out).reshape(bsz, rows, 2)
        q_residual = self.action_q_residual(action_out).reshape(bsz, rows, 2)
        scores = (base_scores + residual).masked_fill(~valid, -1e9)
        q = (base_q + q_residual).masked_fill(~valid, 0.0)
        return scores, q


class CalibratedActionAttentionFactorizedNet(ActionAttentionFactorizedNet):
    def __init__(self, d_model: int = 48, nhead: int = 4, nlayers: int = 2, head_arch: str = "branch_context"):
        super().__init__(d_model=d_model, nhead=nhead, nlayers=nlayers, head_arch=head_arch)
        self.branch_calibration = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        nn.init.zeros_(self.branch_calibration[-1].weight)
        nn.init.zeros_(self.branch_calibration[-1].bias)

    def forward_scores(self, tokens: torch.Tensor, slot: torch.Tensor):
        cls_out, tok_out, selected, token_active = self.backbone.encode_tokens(tokens)
        slot_emb = self.backbone.slot_proj(slot)
        bsz, rows, _d_model = tok_out.shape

        sensor = self.sensor_embed[None, :, :].expand(bsz, -1, -1)
        cls_s = cls_out[:, None, :].expand(-1, 2, -1)
        slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
        sensor_state = self.sensor_state_proj(torch.cat([cls_s, slot_s, sensor], dim=-1))
        coupled_sensor = self.sensor_coupler(sensor_state)
        type_ctx = torch.cat([cls_s, slot_s, coupled_sensor], dim=-1)
        type_logits = self.type_head(type_ctx)
        type_q = self.type_q_head(type_ctx)
        branch_delta = torch.tanh(self.branch_calibration(type_ctx))

        tok_st = tok_out[:, :, None, :].expand(-1, -1, 2, -1)
        cls_st = cls_out[:, None, None, :].expand(-1, rows, 2, -1)
        slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
        sensor_st = coupled_sensor[:, None, :, :].expand(bsz, rows, -1, -1)
        target_ctx = torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)
        target_logits = self.target_head(target_ctx).squeeze(-1)
        target_q = self.target_q_head(target_ctx).squeeze(-1)

        base_scores = tokens.new_full((bsz, rows, 2), -1e9)
        base_q = tokens.new_zeros((bsz, rows, 2))
        base_scores[:, 0, :] = type_logits[:, :, 0] + branch_delta[:, :, 0]
        base_q[:, 0, :] = type_q[:, :, 0]
        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        base_scores[:, 1:, :] = (type_logits[:, None, :, 1] + branch_delta[:, None, :, 1] + target_logits)[:, 1:, :]
        base_q[:, 1:, :] = (type_q[:, None, :, 1] + target_q)[:, 1:, :]

        row_is_search = torch.arange(rows, device=tokens.device)[None, :, None] == 0
        valid = (track_mask[:, :, None] | row_is_search).expand(-1, -1, 2)
        action_ctx = self.action_proj(target_ctx).reshape(bsz, rows * 2, -1)
        action_valid = valid.reshape(bsz, rows * 2)
        action_ctx = self.action_coupler(action_ctx, src_key_padding_mask=~action_valid)
        residual = self.action_policy_residual(action_ctx).reshape(bsz, rows, 2)
        q_residual = self.action_q_residual(action_ctx).reshape(bsz, rows, 2)
        scores = (base_scores + residual).masked_fill(~valid, -1e9)
        q = (base_q + q_residual).masked_fill(~valid, 0.0)
        return scores, q


class FlatResidualFactorizedNet(TwoRowFactorizedNet):
    def __init__(self, d_model: int = 48, nhead: int = 4, nlayers: int = 2, head_arch: str = "branch_context"):
        super().__init__(d_model=d_model, nhead=nhead, nlayers=nlayers, head_arch=head_arch)
        self.flat_residual_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.flat_residual_q_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        nn.init.zeros_(self.flat_residual_head[-1].weight)
        nn.init.zeros_(self.flat_residual_head[-1].bias)
        nn.init.zeros_(self.flat_residual_q_head[-1].weight)
        nn.init.zeros_(self.flat_residual_q_head[-1].bias)

    def forward_scores(self, tokens: torch.Tensor, slot: torch.Tensor):
        scores, q = super().forward_scores(tokens, slot)
        cls_out, tok_out, selected, token_active = self.backbone.encode_tokens(tokens)
        slot_emb = self.backbone.slot_proj(slot)
        cls_rep = cls_out.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
        slot_rep = slot_emb.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
        ctx = torch.cat([tok_out, cls_rep, slot_rep], dim=-1)
        residual = self.flat_residual_head(ctx)
        q_residual = self.flat_residual_q_head(ctx)
        action_mask = token_active & ~selected
        action_mask[:, 0] = True
        valid = action_mask[:, :, None].expand_as(scores)
        scores = (scores + residual).masked_fill(~valid, -1e9)
        q = (q + q_residual).masked_fill(~valid, 0.0)
        return scores, q


class CalibratedTwoRowFactorizedNet(TwoRowFactorizedNet):
    def __init__(self, d_model: int = 48, nhead: int = 4, nlayers: int = 2, head_arch: str = "branch_context"):
        super().__init__(d_model=d_model, nhead=nhead, nlayers=nlayers, head_arch=head_arch)
        self.branch_calibration = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward_scores(self, tokens: torch.Tensor, slot: torch.Tensor, return_abs_q: bool = False):
        cls_out, tok_out, selected, token_active = self.backbone.encode_tokens(tokens)
        slot_emb = self.backbone.slot_proj(slot)
        bsz, rows, _d_model = tok_out.shape

        sensor = self.sensor_embed[None, :, :].expand(bsz, -1, -1)
        cls_s = cls_out[:, None, :].expand(-1, 2, -1)
        slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
        type_ctx = torch.cat([cls_s, slot_s, sensor], dim=-1)
        type_logits = self.type_head(type_ctx)
        type_q = self.type_q_head(type_ctx)
        branch_delta = 0.5 * torch.tanh(self.branch_calibration(type_ctx))

        tok_st = tok_out[:, :, None, :].expand(-1, -1, 2, -1)
        cls_st = cls_out[:, None, None, :].expand(-1, rows, 2, -1)
        slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
        sensor_st = self.sensor_embed[None, None, :, :].expand(bsz, rows, -1, -1)
        target_ctx = torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)
        target_logits = self.target_head(target_ctx).squeeze(-1)
        target_q = self.target_q_head(target_ctx).squeeze(-1)

        scores = tokens.new_full((bsz, rows, 2), -1e9)
        q_abs = tokens.new_zeros((bsz, rows, 2))
        scores[:, 0, :] = type_logits[:, :, 0] + branch_delta[:, :, 0]
        q_abs[:, 0, :] = type_q[:, :, 0]
        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        scores[:, 1:, :] = (type_logits[:, None, :, 1] + branch_delta[:, None, :, 1] + target_logits)[:, 1:, :]
        q_abs[:, 1:, :] = (type_q[:, None, :, 1] + target_q)[:, 1:, :]

        row_is_search = torch.arange(rows, device=tokens.device)[None, :, None] == 0
        valid = track_mask[:, :, None] | row_is_search
        scores = scores.masked_fill(~valid, -1e9)
        q_abs = q_abs.masked_fill(~valid, 0.0)

        valid_f = valid.float()
        q_mean = (q_abs * valid_f).sum(dim=(1, 2), keepdim=True) / valid_f.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        q_adv = torch.where(valid, q_abs - q_mean, torch.zeros_like(q_abs))
        if return_abs_q:
            return scores, q_adv, q_abs
        return scores, q_adv


class LearnedBlendTwoRowFactorizedNet(CalibratedTwoRowFactorizedNet):
    def __init__(self, d_model: int = 48, nhead: int = 4, nlayers: int = 2, head_arch: str = "branch_context"):
        super().__init__(d_model=d_model, nhead=nhead, nlayers=nlayers, head_arch=head_arch)
        self.adv_mix_head = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward_scores(self, tokens: torch.Tensor, slot: torch.Tensor, return_abs_q: bool = False):
        cls_out, tok_out, selected, token_active = self.backbone.encode_tokens(tokens)
        slot_emb = self.backbone.slot_proj(slot)
        bsz, rows, _d_model = tok_out.shape

        sensor = self.sensor_embed[None, :, :].expand(bsz, -1, -1)
        cls_s = cls_out[:, None, :].expand(-1, 2, -1)
        slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
        type_ctx = torch.cat([cls_s, slot_s, sensor], dim=-1)
        type_logits = self.type_head(type_ctx)
        type_q = self.type_q_head(type_ctx)
        branch_delta = 0.5 * torch.tanh(self.branch_calibration(type_ctx))
        adv_mix = 0.5 * torch.sigmoid(self.adv_mix_head(type_ctx).squeeze(-1))

        tok_st = tok_out[:, :, None, :].expand(-1, -1, 2, -1)
        cls_st = cls_out[:, None, None, :].expand(-1, rows, 2, -1)
        slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
        sensor_st = self.sensor_embed[None, None, :, :].expand(bsz, rows, -1, -1)
        target_ctx = torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)
        target_logits = self.target_head(target_ctx).squeeze(-1)
        target_q = self.target_q_head(target_ctx).squeeze(-1)

        base_scores = tokens.new_full((bsz, rows, 2), -1e9)
        q_abs = tokens.new_zeros((bsz, rows, 2))
        base_scores[:, 0, :] = type_logits[:, :, 0] + branch_delta[:, :, 0]
        q_abs[:, 0, :] = type_q[:, :, 0]
        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        base_scores[:, 1:, :] = (type_logits[:, None, :, 1] + branch_delta[:, None, :, 1] + target_logits)[:, 1:, :]
        q_abs[:, 1:, :] = (type_q[:, None, :, 1] + target_q)[:, 1:, :]

        row_is_search = torch.arange(rows, device=tokens.device)[None, :, None] == 0
        valid = track_mask[:, :, None] | row_is_search
        base_scores = base_scores.masked_fill(~valid, -1e9)
        q_abs = q_abs.masked_fill(~valid, 0.0)
        valid_f = valid.float()
        q_mean = (q_abs * valid_f).sum(dim=(1, 2), keepdim=True) / valid_f.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        q_adv = torch.where(valid, q_abs - q_mean, torch.zeros_like(q_abs))
        scores = base_scores + adv_mix[:, None, :] * q_adv
        scores = scores.masked_fill(~valid, -1e9)
        if return_abs_q:
            return scores, q_adv, q_abs
        return scores, q_adv


class AuxFlatTwoRowFactorizedNet(TwoRowFactorizedNet):
    def __init__(self, d_model: int = 48, nhead: int = 4, nlayers: int = 2, head_arch: str = "branch_context"):
        super().__init__(d_model=d_model, nhead=nhead, nlayers=nlayers, head_arch=head_arch)
        self.aux_flat_head = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward_aux_flat(self, tokens: torch.Tensor, slot: torch.Tensor):
        cls_out, tok_out, selected, token_active = self.backbone.encode_tokens(tokens)
        slot_emb = self.backbone.slot_proj(slot)
        cls_rep = cls_out.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
        slot_rep = slot_emb.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
        ctx = torch.cat([tok_out, cls_rep, slot_rep], dim=-1)
        logits = self.aux_flat_head(ctx)
        action_mask = token_active & ~selected
        action_mask[:, 0] = True
        return logits.masked_fill(~action_mask[:, :, None], -1e9)


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def physical_candidates(obs: dict, top_k: int) -> list[int]:
    cands = [xs_s_search_action(MAXT)]
    if int(obs.get("enable_x_band", 0)) and float(obs.get("x_band_busy_ms", 0.0)) <= 0.0:
        cands.append(xs_x_search_action(MAXT))
    active = np.asarray(obs["active_mask"], dtype=bool)
    deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
    ranges = np.asarray(obs.get("target_range", np.zeros_like(deadline)), dtype=np.float32)
    ranked = []
    for i, ok in enumerate(active[:MAXT]):
        if not bool(ok) or i >= len(deadline) or float(deadline[i]) < 0.0:
            continue
        ranked.append((float(deadline[i]), i + 1))
    ranked.sort(key=lambda x: (x[0], x[1]))
    for _, base in ranked[: max(0, int(top_k))]:
        r = float(ranges[base - 1]) if base - 1 < len(ranges) else 50_000_000.0
        if float(obs.get("s_band_busy_ms", 0.0)) <= 0.0 and 10_000_000.0 < r < 184_000_000.0:
            cands.append(xs_s_track_action(base, MAXT))
        if int(obs.get("enable_x_band", 0)) and float(obs.get("x_band_busy_ms", 0.0)) <= 0.0 and 5_000_000.0 < r < 100_000_000.0:
            cands.append(xs_x_track_action(base, MAXT))
    return list(dict.fromkeys(int(a) for a in cands))


def make_behavior_planner(name: str):
    if str(name) == "est":
        return ESTPlanner(MAXT)
    if str(name) == "edf":
        return EDFPlanner(MAXT)
    raise ValueError(f"unknown behavior planner: {name}")


def make_physical_model(variant: str, args):
    if variant in {"two_row_factorized", "two_row_factorized_qnorm", "two_row_factorized_adaptive"}:
        return TwoRowFactorizedNet(args.d_model, args.nhead, args.nlayers)
    if variant == "two_row_factored_loss":
        return TwoRowFactorizedNet(args.d_model, args.nhead, args.nlayers)
    if variant == "two_row_auxflat_factorized":
        return AuxFlatTwoRowFactorizedNet(args.d_model, args.nhead, args.nlayers)
    if variant == "two_row_calibrated_factorized":
        return CalibratedTwoRowFactorizedNet(args.d_model, args.nhead, args.nlayers)
    if variant == "two_row_learned_blend_factorized":
        return LearnedBlendTwoRowFactorizedNet(args.d_model, args.nhead, args.nlayers)
    if variant == "two_row_joint_factorized":
        return JointTwoRowFactorizedNet(args.d_model, args.nhead, args.nlayers)
    if variant == "two_row_strong_joint_factorized":
        return StrongJointTwoRowFactorizedNet(args.d_model, args.nhead, args.nlayers)
    if variant in {"two_row_coupled_factorized", "two_row_coupled_factored_loss", "two_row_coupled_qpolicy", "two_row_coupled_qpolicy_factored_loss"}:
        return CoupledTwoRowFactorizedNet(args.d_model, args.nhead, args.nlayers)
    if variant in {"two_row_action_attention", "two_row_action_attention_factored_loss", "two_row_action_attention_factor_only", "two_row_action_attention_qpolicy", "two_row_action_attention_qpolicy_factored_loss", "two_row_action_attention_branchfair_qpolicy"}:
        return ActionAttentionFactorizedNet(args.d_model, args.nhead, args.nlayers)
    if variant == "two_row_action_attention_autoregressive":
        return AutoregressiveActionAttentionFactorizedNet(args.d_model, args.nhead, args.nlayers)
    if variant == "alphastar_factorized":
        return AlphaStarFactorizedNet(args.d_model, args.nhead, args.nlayers)
    if variant == "two_row_calibrated_action_attention_qpolicy_factored_loss":
        return CalibratedActionAttentionFactorizedNet(args.d_model, args.nhead, args.nlayers)
    if variant == "two_row_full_shared_action_qpolicy_factored_loss":
        return FullSharedActionEncoderFactorizedNet(args.d_model, args.nhead, args.nlayers)
    if variant in {"two_row_flat_residual", "two_row_flat_residual_factored_loss"}:
        return FlatResidualFactorizedNet(args.d_model, args.nhead, args.nlayers)
    if variant == "flat_action_attention":
        return FlatActionAttentionNet(args.d_model, args.nhead, args.nlayers)
    if variant == "binary_type":
        return BinaryTypeNet(args.d_model, args.nhead, args.nlayers)
    if variant in {"two_sensor_type", "two_sensor_type_qpolicy"}:
        return TwoSensorTypeNet(args.d_model, args.nhead, args.nlayers)
    if variant == "separated_type_target_factorized":
        return SeparatedTypeTargetFactorizedNet(args.d_model, args.nhead, args.nlayers)
    if flat_search_slots(variant) is not None:
        return MultiSearchFlatNet(args.d_model, args.nhead, args.nlayers, search_slots=int(flat_search_slots(variant)))
    return MutualRadarNet(d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers, head_arch="branch_context")


def load_bootstrap_model(args):
    state_path = str(getattr(args, "bootstrap_state", "")).strip()
    if not state_path:
        return None
    variant = str(getattr(args, "bootstrap_variant", "flat"))
    model = make_physical_model(variant, args)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def bootstrap_value(model, variant: str, adapt, obs: dict, elapsed: float, search_count: int, track_count: int, last: int) -> float:
    if model is None:
        return 0.0
    tok = tokenize(adapt, obs, selected=set(), search_count=int(search_count))
    slot = slot_features(obs, float(elapsed), int(search_count), int(track_count), int(last), 200.0)
    with torch.inference_mode():
        x = torch.from_numpy(tok).float().unsqueeze(0)
        s = torch.from_numpy(slot).float().unsqueeze(0)
        if hasattr(model, "forward_value"):
            value = model.forward_value(x, s)
        elif hasattr(model, "forward_physical_flat"):
            _scores, _q, value = model.forward_physical_flat(x, s)
        else:
            out = model(x, s)
            value = out[-1] if isinstance(out, tuple) else torch.zeros(1)
    return float(value.reshape(-1)[0].detach().cpu())


def state_potential(eng, debt: float) -> float:
    metrics = sample_state_metrics(eng, float(debt))
    dropped_pct = float(metrics.get("drop_pct_active", 0.0))
    mean_delay = float(metrics.get("mean_delay_active", 0.0))
    tracked = float(metrics.get("tracked_targets", 0.0))
    return 0.10 * tracked - 0.25 * dropped_pct - 0.20 * mean_delay


def rollout_budget(eng, planner, debt: float, budget_ms: float, env_cfg: dict) -> tuple[float, float]:
    local_budget = float(budget_ms)
    total = 0.0
    next_debt = float(debt)
    idle_action = int(eng.max_trackers) + 1
    while local_budget > 0.0 and not bool(eng.term_buf[0]):
        obs = attach_env_obs(get_obs(eng, next_debt), env_cfg, True, True)
        plan = list(planner.plan(obs, budget_ms=int(max(1.0, local_budget))))
        progressed = False
        for action in plan:
            if local_budget <= 0.0 or bool(eng.term_buf[0]):
                break
            reward, spent, ex = execute_first_valid_action(eng, [int(action)], local_budget)
            if ex is None or float(spent) <= 0.0:
                continue
            base, _sensor = xs_decode_action(int(ex), MAXT)
            next_debt = 0.0 if int(base) == 0 else next_debt + float(spent)
            total += float(reward)
            local_budget -= float(spent)
            progressed = True
        if not progressed:
            reward, spent, ex = execute_first_valid_action(eng, [idle_action], local_budget)
            if ex is None or float(spent) <= 0.0:
                break
            next_debt += float(spent)
            total += float(reward)
            local_budget -= float(spent)
    return float(total), float(next_debt)


def eval_candidate(
    eng,
    snapshot,
    action: int,
    debt: float,
    remaining_ms: float,
    tail_planner,
    env_cfg: dict,
    tail_windows: int,
    potential_weight: float,
    bootstrap_model=None,
    bootstrap_variant: str = "flat",
    bootstrap_weight: float = 0.0,
    adapt=None,
    elapsed_ms: float = 0.0,
    search_count: int = 0,
    track_count: int = 0,
    last_action: int = -1,
) -> float:
    binding.vec_restore(eng.env, snapshot)
    reward, dt, executed = execute_first_valid_action(eng, [int(action)], float(remaining_ms))
    if executed is None or float(dt) <= 0.0:
        binding.vec_restore(eng.env, snapshot)
        return -1e9
    base, _ = xs_decode_action(int(executed), MAXT)
    next_debt = 0.0 if int(base) == 0 else float(debt) + float(dt)
    total = float(reward)
    next_search_count = int(search_count) + (1 if int(base) == 0 else 0)
    next_track_count = int(track_count) + (1 if int(base) > 0 else 0)
    next_last = int(base)
    next_elapsed = float(elapsed_ms) + float(dt)
    budget = max(0.0, float(remaining_ms) - float(dt))
    for w in range(max(0, int(tail_windows))):
        local_budget = budget if w == 0 else 200.0
        r, next_debt = rollout_budget(eng, tail_planner, next_debt, local_budget, env_cfg)
        total += float(r)
        budget = 200.0
    total += float(potential_weight) * state_potential(eng, next_debt)
    if bootstrap_model is not None and float(bootstrap_weight) != 0.0:
        obs_next = attach_env_obs(get_obs(eng, next_debt), env_cfg, True, True)
        total += float(bootstrap_weight) * bootstrap_value(
            bootstrap_model,
            str(bootstrap_variant),
            adapt,
            obs_next,
            next_elapsed,
            next_search_count,
            next_track_count,
            next_last,
        )
    binding.vec_restore(eng.env, snapshot)
    return float(total)


def collect_targets(args, exact_args, out_path: Path, behavior_factory=None):
    adapt = adapter()
    bootstrap_model = load_bootstrap_model(args)
    bootstrap_variant = str(getattr(args, "bootstrap_variant", "flat"))
    bootstrap_weight = float(getattr(args, "bootstrap_value_weight", 0.0))
    targets: list[SearchTarget] = []
    rows = []
    for seed in parse_ints(args.train_seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                cell_start = len(targets)
                env_cfg = env_cfg_for(float(rate), exact_args)
                env_cfg["enable_x_band"] = 1
                behavior = behavior_factory(env_cfg) if behavior_factory is not None else make_behavior_planner(args.behavior_policy)
                tail = make_behavior_planner(args.tail_policy)
                eng = build_env(behavior, int(initial), MAXT, int(seed), 200, env_cfg)
                eng.reset(seed=int(seed))
                debt = 0.0
                try:
                    for window in range(int(args.windows)):
                        spent = 0.0
                        search_count = 0
                        track_count = 0
                        last = -1
                        while (
                            spent < 200.0
                            and not bool(eng.term_buf[0])
                            and len(targets) < int(args.max_targets)
                            and (
                                int(args.max_targets_per_cell) <= 0
                                or (len(targets) - cell_start) < int(args.max_targets_per_cell)
                            )
                        ):
                            obs = attach_env_obs(get_obs(eng, debt), env_cfg, True, True)
                            remaining = max(1.0, 200.0 - spent)
                            snapshot = binding.vec_snapshot(eng.env)
                            cands = physical_candidates(obs, int(args.top_k))
                            vals = []
                            for action in cands:
                                val = eval_candidate(
                                    eng,
                                    snapshot,
                                    int(action),
                                    debt,
                                    remaining,
                                    tail,
                                    env_cfg,
                                    int(args.tail_windows),
                                    float(args.potential_weight),
                                    bootstrap_model=bootstrap_model,
                                    bootstrap_variant=bootstrap_variant,
                                    bootstrap_weight=bootstrap_weight,
                                    adapt=adapt,
                                    elapsed_ms=spent,
                                    search_count=search_count,
                                    track_count=track_count,
                                    last_action=last,
                                )
                                vals.append((int(action), float(val)))
                            vals = [(a, v) for a, v in vals if np.isfinite(v) and v > -1e8]
                            if vals:
                                logits = np.asarray([v for _, v in vals], dtype=np.float64) / max(1e-6, float(args.policy_tau))
                                logits -= float(np.max(logits))
                                probs = np.exp(np.clip(logits, -60.0, 60.0))
                                probs /= max(float(probs.sum()), 1e-12)
                                pi = np.zeros((MAXT + 1,), dtype=np.float32)
                                q = np.zeros((MAXT + 1,), dtype=np.float32)
                                q_mask = np.zeros((MAXT + 1,), dtype=np.float32)
                                sensor_pi = np.zeros((MAXT + 1, 2), dtype=np.float32)
                                sensor_q = np.zeros((MAXT + 1, 2), dtype=np.float32)
                                sensor_q_mask = np.zeros((MAXT + 1, 2), dtype=np.float32)
                                for (action, val), p in zip(vals, probs):
                                    base, sensor = xs_decode_action(int(action), MAXT)
                                    if int(base) < 0:
                                        continue
                                    sidx = 0 if sensor is None else int(sensor)
                                    pi[int(base)] += float(p)
                                    sensor_pi[int(base), sidx] += float(p)
                                    if q_mask[int(base)] <= 0.5 or val > q[int(base)]:
                                        q[int(base)] = float(val)
                                        q_mask[int(base)] = 1.0
                                    sensor_q[int(base), sidx] = float(val)
                                    sensor_q_mask[int(base), sidx] = 1.0
                                tok = tokenize(adapt, obs, selected=set(), search_count=search_count).astype(np.float32)
                                slot = slot_features(obs, spent, search_count, track_count, last, 200.0).astype(np.float32)
                                root_value = float(max(v for _, v in vals))
                                targets.append(
                                    SearchTarget(
                                        tok,
                                        slot,
                                        pi,
                                        q,
                                        q_mask,
                                        search_count,
                                        track_count,
                                        reward=0.0,
                                        ret=root_value,
                                        sensor_pi=sensor_pi,
                                        sensor_q=sensor_q,
                                        sensor_q_mask=sensor_q_mask,
                                        initial=int(initial),
                                        rate=float(rate),
                                        seed=int(seed),
                                        window=int(window),
                                        action_index=len(targets),
                                    )
                                )
                                rows.append({"initial": initial, "rate": rate, "seed": seed, "window": window, "sensor1_mass": float(sensor_pi[:, 1].sum()), "search_mass": float(sensor_pi[0, :].sum())})
                            binding.vec_restore(eng.env, snapshot)
                            plan = list(behavior.plan(obs, budget_ms=int(remaining)))
                            if not plan:
                                break
                            reward, dt, executed = execute_first_valid_action(eng, plan, remaining)
                            if executed is None or float(dt) <= 0.0:
                                break
                            base, _ = xs_decode_action(int(executed), MAXT)
                            debt = 0.0 if int(base) == 0 else debt + float(dt)
                            spent += float(dt)
                            if int(base) == 0:
                                search_count += 1
                            elif int(base) > 0:
                                track_count += 1
                            last = int(base)
                        if len(targets) >= int(args.max_targets) or (
                            int(args.max_targets_per_cell) > 0
                            and (len(targets) - cell_start) >= int(args.max_targets_per_cell)
                        ):
                            break
                finally:
                    eng.close()
                print({"targets": len(targets), "seed": seed, "initial": initial, "rate": rate}, flush=True)
                if len(targets) >= int(args.max_targets):
                    break
            if len(targets) >= int(args.max_targets):
                break
        if len(targets) >= int(args.max_targets):
            break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_targets(out_path, targets)
    pd.DataFrame(rows).to_csv(out_path.with_suffix(".csv"), index=False)
    return targets


def factorized_marginal_supervision_loss(scores: torch.Tensor, sensor_pi: torch.Tensor) -> torch.Tensor:
    bsz, rows, _sensors = scores.shape
    search_mass = sensor_pi[:, 0, :]
    track_mass = sensor_pi[:, 1:, :].sum(dim=1)
    type_target = torch.stack([search_mass, track_mass], dim=-1)
    sensor_mass = type_target.sum(dim=-1)
    type_target = type_target / sensor_mass[:, :, None].clamp_min(1e-6)
    track_marginal = torch.logsumexp(scores[:, 1:, :], dim=1)
    type_logits = torch.stack([scores[:, 0, :], track_marginal], dim=-1)
    type_log_probs = F.log_softmax(type_logits, dim=-1)
    type_loss = -((type_target * type_log_probs).sum(dim=-1) * sensor_mass).sum(dim=1).mean()

    target_mass = sensor_pi[:, 1:, :]
    target_total = target_mass.sum(dim=1)
    target_dist = target_mass / target_total[:, None, :].clamp_min(1e-6)
    target_log_probs = F.log_softmax(scores[:, 1:, :], dim=1)
    target_loss = -((target_dist * target_log_probs).sum(dim=1) * target_total).sum(dim=1).mean()
    return type_loss + target_loss


def autoregressive_sensor_supervision_loss(scores: torch.Tensor, sensor_pi: torch.Tensor) -> torch.Tensor:
    """Ordered sensor policy loss: S action, then X action conditioned on S."""
    s_mass = sensor_pi[:, :, 0].sum(dim=1)
    x_mass = sensor_pi[:, :, 1].sum(dim=1)
    s_target = sensor_pi[:, :, 0] / s_mass[:, None].clamp_min(1e-6)
    x_target = sensor_pi[:, :, 1] / x_mass[:, None].clamp_min(1e-6)
    s_loss = -((s_target * F.log_softmax(scores[:, :, 0], dim=1)).sum(dim=1) * s_mass).mean()
    x_loss = -((x_target * F.log_softmax(scores[:, :, 1], dim=1)).sum(dim=1) * x_mass).mean()
    return s_loss + x_loss


def alphastar_factorized_loss(aux: dict, sensor_pi: torch.Tensor) -> torch.Tensor:
    losses = []
    for sensor_idx, prefix in ((0, "s"), (1, "x")):
        search_mass = sensor_pi[:, 0, sensor_idx]
        track_mass = sensor_pi[:, 1:, sensor_idx].sum(dim=1)
        type_mass = (search_mass + track_mass).clamp_min(1e-6)
        type_target = torch.stack([search_mass / type_mass, track_mass / type_mass], dim=-1)
        type_logits = aux[f"{prefix}_type_logits"]
        losses.append(-((type_target * F.log_softmax(type_logits, dim=-1)).sum(dim=-1) * type_mass).mean())

        target_logits = aux[f"{prefix}_target_logits"][:, 1:]
        target_dist = sensor_pi[:, 1:, sensor_idx] / track_mass[:, None].clamp_min(1e-6)
        target_loss = -((target_dist * F.log_softmax(target_logits, dim=1)).sum(dim=1) * track_mass).mean()
        losses.append(target_loss)
    return torch.stack(losses).sum()


def q_soft_policy_target(sensor_q: torch.Tensor | None, sensor_q_mask: torch.Tensor | None, value_scale: float, tau: float = 0.50) -> torch.Tensor | None:
    if sensor_q is None or sensor_q_mask is None or not bool((sensor_q_mask > 0.5).any()):
        return None
    valid = sensor_q_mask > 0.5
    logits = (sensor_q / float(value_scale)) / max(1e-6, float(tau))
    logits = logits.masked_fill(~valid, -1e9)
    flat = logits.reshape(logits.shape[0], -1)
    has_valid = valid.reshape(valid.shape[0], -1).any(dim=1)
    probs = torch.zeros_like(flat)
    if bool(has_valid.any()):
        probs[has_valid] = F.softmax(flat[has_valid], dim=1)
    return probs.reshape_as(sensor_q)


def q_policy_loss(scores: torch.Tensor, sensor_q: torch.Tensor | None, sensor_q_mask: torch.Tensor | None, value_scale: float, tau: float = 0.50) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    target = q_soft_policy_target(sensor_q, sensor_q_mask, value_scale, tau=tau)
    if target is None:
        return None, None
    log_probs = F.log_softmax(scores.reshape(scores.shape[0], -1), dim=1).reshape_as(scores)
    return -(target * log_probs).sum(dim=(1, 2)).mean(), target


def sensor_type_targets_from_q(sensor_q: torch.Tensor | None, sensor_q_mask: torch.Tensor | None, value_scale: float, tau: float = 0.50) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if sensor_q is None or sensor_q_mask is None:
        return None, None
    search_valid = sensor_q_mask[:, 0, :] > 0.5
    track_valid = sensor_q_mask[:, 1:, :] > 0.5
    search_q = sensor_q[:, 0, :] / float(value_scale)
    track_q = (sensor_q[:, 1:, :] / float(value_scale)).masked_fill(~track_valid, -1e9).max(dim=1).values
    valid = search_valid | track_valid.reshape(track_valid.shape[0], -1, 2).any(dim=1)
    logits = torch.stack([track_q, search_q], dim=-1) / max(1e-6, float(tau))
    probs = F.softmax(logits, dim=-1)
    return probs[:, :, 1].masked_fill(~valid, 0.0), valid.float()


def separated_type_target_step(model: SeparatedTypeTargetFactorizedNet, x, slot, sensor_pi, sensor_q, sensor_q_mask, value_scale):
    type_logits, type_q, target_logits, target_q, selected, token_active, _cls_out, _slot_emb = model.forward_parts(x, slot)
    scores, pred_q = model.forward_scores(x, slot)
    type_mass = sensor_pi[:, 0, :] + sensor_pi[:, 1:, :].sum(dim=1)
    type_target = torch.stack(
        [
            sensor_pi[:, 0, :] / type_mass.clamp_min(1e-6),
            sensor_pi[:, 1:, :].sum(dim=1) / type_mass.clamp_min(1e-6),
        ],
        dim=-1,
    )
    type_loss = -(
        type_mass
        * (type_target * F.log_softmax(type_logits, dim=-1)).sum(dim=-1)
    ).sum(dim=1).mean()

    track_mass = sensor_pi[:, 1:, :]
    target_total = track_mass.sum(dim=1)
    target_dist = track_mass / target_total[:, None, :].clamp_min(1e-6)
    track_mask = token_active[:, 1:] & ~selected[:, 1:]
    target_log_probs = F.log_softmax(target_logits[:, 1:, :].masked_fill(~track_mask[:, :, None], -1e9), dim=1)
    target_loss = -((target_dist * target_log_probs).sum(dim=1) * target_total).sum(dim=1).mean()

    joint_log_probs = F.log_softmax(scores.reshape(scores.shape[0], -1), dim=1).reshape_as(scores)
    joint_loss = -(sensor_pi * joint_log_probs).sum(dim=(1, 2)).mean()
    policy_loss = 0.40 * joint_loss + 0.35 * type_loss + 0.25 * target_loss

    q_loss = torch.zeros((), device=x.device)
    if sensor_q is not None and sensor_q_mask is not None and bool((sensor_q_mask > 0.5).any()):
        target_q_scaled = sensor_q / float(value_scale)
        joint_q_loss = F.smooth_l1_loss(pred_q[sensor_q_mask > 0.5], target_q_scaled[sensor_q_mask > 0.5])
        search_valid = sensor_q_mask[:, 0, :] > 0.5
        track_valid = sensor_q_mask[:, 1:, :] > 0.5
        search_q_t = sensor_q[:, 0, :] / float(value_scale)
        track_q_t = (sensor_q[:, 1:, :] / float(value_scale)).masked_fill(~track_valid, -1e9).max(dim=1).values
        has_track = track_valid.any(dim=1)
        type_q_target = torch.stack([track_q_t, search_q_t], dim=-1)
        type_q_valid = torch.stack([has_track.float(), search_valid.float()], dim=-1)
        type_q_loss = (F.smooth_l1_loss(type_q, type_q_target, reduction="none") * type_q_valid).sum() / type_q_valid.sum().clamp_min(1.0)
        q_loss = 0.7 * joint_q_loss + 0.3 * type_q_loss
    return policy_loss, q_loss, joint_log_probs


def branchfair_score_loss(scores: torch.Tensor, sensor_q: torch.Tensor | None, sensor_q_mask: torch.Tensor | None, value_scale: float, tau: float = 0.50):
    """Branch-fair policy target for factorized score heads."""
    if sensor_q is None or sensor_q_mask is None or not bool((sensor_q_mask > 0.5).any()):
        return None
    q = sensor_q / float(value_scale)
    valid = sensor_q_mask > 0.5
    search_valid = valid[:, 0, :]
    track_valid = valid[:, 1:, :]
    has_track = track_valid.any(dim=1)
    search_q = torch.where(search_valid, q[:, 0, :], torch.full_like(q[:, 0, :], -1e9))
    track_q = q[:, 1:, :].masked_fill(~track_valid, -1e9).max(dim=1).values
    branch_target_logits = torch.stack([search_q, track_q], dim=-1) / max(1e-6, float(tau))
    branch_valid = torch.stack([search_valid, has_track], dim=-1)
    branch_target = torch.softmax(branch_target_logits.masked_fill(~branch_valid, -1e9), dim=-1)
    branch_rows = branch_valid.any(dim=-1)
    if not bool(branch_rows.any()):
        return None

    pred_search = scores[:, 0, :]
    pred_track = scores[:, 1:, :].masked_fill(~track_valid, -1e9).max(dim=1).values
    pred_branch = torch.stack([pred_search, pred_track], dim=-1)
    type_loss = -(
        branch_target[branch_rows] * F.log_softmax(pred_branch[branch_rows], dim=-1)
    ).sum(dim=-1).mean()

    target_losses = []
    for sensor in range(scores.shape[2]):
        row_valid = has_track[:, sensor]
        if not bool(row_valid.any()):
            continue
        target_logits = q[:, 1:, sensor].masked_fill(~track_valid[:, :, sensor], -1e9)
        target = torch.softmax(target_logits[row_valid] / max(1e-6, float(tau)), dim=1)
        pred = scores[:, 1:, sensor][row_valid].masked_fill(~track_valid[:, :, sensor][row_valid], -1e9)
        target_losses.append(-(target * F.log_softmax(pred, dim=1)).sum(dim=1).mean())
    target_loss = torch.stack(target_losses).mean() if target_losses else torch.zeros((), device=scores.device)
    return 0.55 * type_loss + 0.45 * target_loss


def search_mass_calibration_loss(log_probs: torch.Tensor, target_pi: torch.Tensor) -> torch.Tensor:
    pred_search_mass = log_probs.exp()[:, 0, :].sum(dim=1).clamp(1e-6, 1.0 - 1e-6)
    target_search_mass = target_pi[:, 0, :].sum(dim=1).clamp(0.0, 1.0)
    return F.binary_cross_entropy(pred_search_mass, target_search_mass)


def train_head(variant: str, targets, args, device, model=None):
    if model is None:
        model = make_physical_model(variant, args)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(args.model_seed))
    cell_indices = {}
    if bool(args.cell_balanced_sampling):
        for i, t in enumerate(targets):
            key = (int(getattr(t, "initial", -1)), float(getattr(t, "rate", 0.0)))
            cell_indices.setdefault(key, []).append(int(i))
        cell_indices = {k: np.asarray(v, dtype=np.int64) for k, v in cell_indices.items() if len(v) > 0}
        cell_keys = list(cell_indices.keys())
    else:
        cell_keys = []
    abs_q = []
    for t in targets:
        mask = np.asarray(t.sensor_q_mask) > 0.5
        if np.any(mask):
            abs_q.extend(np.abs(np.asarray(t.sensor_q)[mask]).tolist())
        if abs(float(getattr(t, "ret", 0.0))) > 0.0:
            abs_q.append(abs(float(t.ret)))
    value_scale = max(1.0, float(np.percentile(abs_q, 90))) if abs_q else 10.0
    for step in range(int(args.train_steps)):
        if cell_keys:
            sampled_cells = rng.integers(0, len(cell_keys), size=int(args.batch_size))
            idx = np.asarray([rng.choice(cell_indices[cell_keys[int(c)]]) for c in sampled_cells], dtype=np.int64)
        else:
            idx = rng.integers(0, len(targets), size=int(args.batch_size))
        x, slot, sensor_pi, sensor_q, sensor_q_mask = batch_tensors(targets, idx, device)
        if bool(getattr(args, "hard_policy_target", False)):
            flat_idx = sensor_pi.reshape(sensor_pi.shape[0], -1).argmax(dim=1)
            hard = torch.zeros_like(sensor_pi.reshape(sensor_pi.shape[0], -1))
            hard[torch.arange(sensor_pi.shape[0], device=device), flat_idx] = 1.0
            sensor_pi = hard.reshape_as(sensor_pi)
        ret = torch.tensor([float(targets[int(i)].ret) / float(value_scale) for i in idx], dtype=torch.float32, device=device)
        value_pred = None
        if variant == "binary_type":
            search_target = sensor_pi[:, 0, :].sum(dim=1).clamp(0.0, 1.0)
            logit, pred_q, value_pred = model(x, slot)
            policy_loss = F.binary_cross_entropy_with_logits(logit, search_target)
            q_target = torch.zeros((sensor_pi.shape[0], 2), dtype=torch.float32, device=device)
            q_valid = torch.zeros((sensor_pi.shape[0], 2), dtype=torch.float32, device=device)
            if sensor_q is not None and sensor_q_mask is not None:
                search_valid = sensor_q_mask[:, 0, :] > 0.5
                track_valid = sensor_q_mask[:, 1:, :] > 0.5
                search_vals = sensor_q[:, 0, :].masked_fill(~search_valid, -1e9)
                track_vals = sensor_q[:, 1:, :].masked_fill(~track_valid, -1e9)
                has_search = search_valid.any(dim=1)
                has_track = track_valid.reshape(track_valid.shape[0], -1).any(dim=1)
                q_target[:, 1] = torch.where(has_search, search_vals.max(dim=1).values, torch.zeros_like(q_target[:, 1]))
                q_target[:, 0] = torch.where(has_track, track_vals.reshape(track_vals.shape[0], -1).max(dim=1).values, torch.zeros_like(q_target[:, 0]))
                q_valid[:, 1] = has_search.float()
                q_valid[:, 0] = has_track.float()
            q_loss = (F.smooth_l1_loss(pred_q, q_target / float(value_scale), reduction="none") * q_valid).sum() / q_valid.sum().clamp_min(1.0)
            log_probs = torch.stack([F.logsigmoid(-logit), F.logsigmoid(logit)], dim=1)
            top1 = ((torch.sigmoid(logit) >= 0.5).float() == (search_target >= 0.5).float()).float().mean().item()
            search_acc = top1
        elif variant in {"two_sensor_type", "two_sensor_type_qpolicy"}:
            if variant == "two_sensor_type_qpolicy":
                q_search_target, q_valid = sensor_type_targets_from_q(sensor_q, sensor_q_mask, value_scale)
                if q_search_target is None:
                    search_target = sensor_pi[:, 0, :].clamp(0.0, 1.0)
                    valid = torch.ones_like(search_target)
                else:
                    search_target = q_search_target
                    valid = q_valid
            else:
                type_mass = sensor_pi[:, 0, :] + sensor_pi[:, 1:, :].sum(dim=1)
                search_target = sensor_pi[:, 0, :] / type_mass.clamp_min(1e-6)
                valid = (type_mass > 1e-6).float()
            logit, pred_q, value_pred = model(x, slot)
            policy_loss = (F.binary_cross_entropy_with_logits(logit, search_target, reduction="none") * valid).sum() / valid.sum().clamp_min(1.0)
            q_target = torch.zeros((sensor_pi.shape[0], 2, 2), dtype=torch.float32, device=device)
            q_valid = torch.zeros((sensor_pi.shape[0], 2, 2), dtype=torch.float32, device=device)
            if sensor_q is not None and sensor_q_mask is not None:
                search_valid = sensor_q_mask[:, 0, :] > 0.5
                track_valid = sensor_q_mask[:, 1:, :] > 0.5
                search_vals = sensor_q[:, 0, :]
                track_vals = sensor_q[:, 1:, :].masked_fill(~track_valid, -1e9).max(dim=1).values
                has_track = track_valid.any(dim=1)
                q_target[:, :, 1] = search_vals
                q_target[:, :, 0] = torch.where(has_track, track_vals, torch.zeros_like(track_vals))
                q_valid[:, :, 1] = search_valid.float()
                q_valid[:, :, 0] = has_track.float()
            q_loss = (F.smooth_l1_loss(pred_q, q_target / float(value_scale), reduction="none") * q_valid).sum() / q_valid.sum().clamp_min(1.0)
            log_probs = torch.stack([F.logsigmoid(-logit), F.logsigmoid(logit)], dim=-1)
            top1 = (((torch.sigmoid(logit) >= 0.5).float() == (search_target >= 0.5).float()).float() * valid).sum().div(valid.sum().clamp_min(1.0)).item()
            search_acc = top1
        elif flat_search_slots(variant) is not None:
            track_logits, track_q, search_logits, search_q, value_pred = model.forward_search20(x, slot)
            bsz = int(x.shape[0])
            search_slots = int(model.search_slots)
            search_count_t = torch.tensor(
                [min(max(0, int(getattr(targets[int(i)], "search_count", 0))), search_slots - 1) for i in idx],
                dtype=torch.long,
                device=device,
            )
            search_target = torch.zeros((bsz, search_slots, 2), dtype=torch.float32, device=device)
            search_target[torch.arange(bsz, device=device), search_count_t, :] = sensor_pi[:, 0, :]
            track_target = sensor_pi.clone()
            track_target[:, 0, :] = 0.0
            target = torch.cat([search_target.reshape(bsz, search_slots * 2), track_target.reshape(bsz, -1)], dim=1)
            target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
            logits = torch.cat([search_logits.reshape(bsz, search_slots * 2), track_logits.reshape(bsz, -1)], dim=1)
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = -(target * log_probs).sum(dim=1).mean()
            q_loss = torch.zeros((), device=device)
            if sensor_q is not None and sensor_q_mask is not None and bool((sensor_q_mask > 0.5).any()):
                search_q_target = torch.zeros_like(search_q)
                search_q_valid = torch.zeros_like(search_q)
                search_q_target[torch.arange(bsz, device=device), search_count_t, :] = sensor_q[:, 0, :] / float(value_scale)
                search_q_valid[torch.arange(bsz, device=device), search_count_t, :] = sensor_q_mask[:, 0, :]
                track_q_target = sensor_q / float(value_scale)
                track_q_valid = sensor_q_mask.clone()
                track_q_valid[:, 0, :] = 0.0
                pred = torch.cat([search_q.reshape(bsz, search_slots * 2), track_q.reshape(bsz, -1)], dim=1)
                q_tgt = torch.cat([search_q_target.reshape(bsz, search_slots * 2), track_q_target.reshape(bsz, -1)], dim=1)
                q_valid = torch.cat([search_q_valid.reshape(bsz, search_slots * 2), track_q_valid.reshape(bsz, -1)], dim=1)
                q_loss = (F.smooth_l1_loss(pred, q_tgt, reduction="none") * q_valid).sum() / q_valid.sum().clamp_min(1.0)
            top = log_probs.argmax(dim=1)
            tgt = target.argmax(dim=1)
            top1 = (top == tgt).float().mean().item()
            pred_search = top < (search_slots * 2)
            tgt_search = tgt < (search_slots * 2)
            search_acc = (pred_search == tgt_search).float().mean().item()
        elif variant in {
            "two_row_factorized",
            "two_row_factorized_qnorm",
            "two_row_factorized_adaptive",
            "separated_type_target_factorized",
            "two_row_factored_loss",
            "two_row_auxflat_factorized",
            "two_row_calibrated_factorized",
            "two_row_learned_blend_factorized",
            "two_row_joint_factorized",
            "two_row_strong_joint_factorized",
            "two_row_coupled_factorized",
            "two_row_coupled_factored_loss",
            "two_row_coupled_qpolicy",
            "two_row_coupled_qpolicy_factored_loss",
            "two_row_action_attention",
            "two_row_action_attention_factored_loss",
            "two_row_action_attention_factor_only",
            "two_row_action_attention_qpolicy",
            "two_row_action_attention_qpolicy_factored_loss",
            "two_row_action_attention_branchfair_qpolicy",
            "two_row_action_attention_autoregressive",
            "alphastar_factorized",
            "two_row_calibrated_action_attention_qpolicy_factored_loss",
            "two_row_full_shared_action_qpolicy_factored_loss",
            "two_row_flat_residual",
            "two_row_flat_residual_factored_loss",
            "flat_action_attention",
        }:
            calibration_target_pi = sensor_pi
            if variant == "separated_type_target_factorized":
                policy_loss, q_loss, log_probs = separated_type_target_step(model, x, slot, sensor_pi, sensor_q, sensor_q_mask, value_scale)
            elif variant == "two_row_joint_factorized":
                scores, pred_q, utility_q = model.forward_scores(x, slot)
                log_probs = F.log_softmax(scores.reshape(scores.shape[0], -1), dim=1).reshape_as(scores)
                policy_loss = -(sensor_pi * log_probs).sum(dim=(1, 2)).mean()
                q_loss = torch.zeros((), device=device)
                if sensor_q is not None and sensor_q_mask is not None and bool((sensor_q_mask > 0.5).any()):
                    target_q = sensor_q / float(value_scale)
                    q_base = F.smooth_l1_loss(pred_q[sensor_q_mask > 0.5], target_q[sensor_q_mask > 0.5])
                    q_utility = F.smooth_l1_loss(utility_q[sensor_q_mask > 0.5], target_q[sensor_q_mask > 0.5])
                    q_loss = 0.5 * (q_base + q_utility)
            elif variant in {
                "two_row_strong_joint_factorized",
                "two_row_coupled_factorized",
                "two_row_coupled_factored_loss",
                "two_row_coupled_qpolicy",
                "two_row_coupled_qpolicy_factored_loss",
                "two_row_action_attention",
                "two_row_action_attention_factored_loss",
                "two_row_action_attention_factor_only",
                "two_row_action_attention_qpolicy",
                "two_row_action_attention_qpolicy_factored_loss",
                "two_row_action_attention_branchfair_qpolicy",
                "two_row_action_attention_autoregressive",
                "alphastar_factorized",
                "two_row_calibrated_action_attention_qpolicy_factored_loss",
                "two_row_full_shared_action_qpolicy_factored_loss",
                "two_row_flat_residual",
                "two_row_flat_residual_factored_loss",
                "flat_action_attention",
            }:
                if variant == "alphastar_factorized":
                    scores, pred_q, aux = model.forward_scores_teacher(x, slot, sensor_pi)
                elif variant == "two_row_action_attention_autoregressive":
                    scores, pred_q = model.forward_scores_teacher(x, slot, sensor_pi)
                    aux = None
                else:
                    scores, pred_q = model.forward_scores(x, slot)
                    aux = None
                log_probs = F.log_softmax(scores.reshape(scores.shape[0], -1), dim=1).reshape_as(scores)
                joint_loss = -(sensor_pi * log_probs).sum(dim=(1, 2)).mean()
                qpol_loss, qpol_target = q_policy_loss(scores, sensor_q, sensor_q_mask, value_scale)
                branchfair_loss = branchfair_score_loss(scores, sensor_q, sensor_q_mask, value_scale)
                if variant == "two_row_action_attention_branchfair_qpolicy" and branchfair_loss is not None:
                    policy_loss = branchfair_loss
                    calibration_target_pi = qpol_target if qpol_target is not None else sensor_pi
                elif variant in {"two_row_coupled_qpolicy", "two_row_action_attention_qpolicy"} and qpol_loss is not None:
                    policy_loss = qpol_loss
                    calibration_target_pi = qpol_target if qpol_target is not None else sensor_pi
                    log_probs = F.log_softmax(scores.reshape(scores.shape[0], -1), dim=1).reshape_as(scores)
                elif variant in {
                    "two_row_coupled_qpolicy_factored_loss",
                    "two_row_action_attention_qpolicy_factored_loss",
                    "two_row_calibrated_action_attention_qpolicy_factored_loss",
                } and qpol_loss is not None and qpol_target is not None:
                    policy_loss = 0.5 * qpol_loss + 0.5 * factorized_marginal_supervision_loss(scores, qpol_target)
                    calibration_target_pi = qpol_target
                    log_probs = F.log_softmax(scores.reshape(scores.shape[0], -1), dim=1).reshape_as(scores)
                elif variant == "two_row_full_shared_action_qpolicy_factored_loss" and qpol_loss is not None and qpol_target is not None:
                    policy_loss = 0.5 * qpol_loss + 0.5 * factorized_marginal_supervision_loss(scores, qpol_target)
                    calibration_target_pi = qpol_target
                    log_probs = F.log_softmax(scores.reshape(scores.shape[0], -1), dim=1).reshape_as(scores)
                elif variant in {"two_row_coupled_factorized", "two_row_action_attention", "two_row_flat_residual", "flat_action_attention"}:
                    policy_loss = joint_loss
                elif variant == "two_row_action_attention_factor_only":
                    policy_loss = factorized_marginal_supervision_loss(scores, sensor_pi)
                elif variant == "two_row_action_attention_autoregressive":
                    policy_loss = 0.25 * autoregressive_sensor_supervision_loss(scores, sensor_pi) + 0.75 * factorized_marginal_supervision_loss(scores, sensor_pi)
                elif variant == "alphastar_factorized":
                    policy_loss = alphastar_factorized_loss(aux, sensor_pi)
                else:
                    policy_loss = 0.5 * joint_loss + 0.5 * factorized_marginal_supervision_loss(scores, sensor_pi)
                q_loss = torch.zeros((), device=device)
                if sensor_q is not None and sensor_q_mask is not None and bool((sensor_q_mask > 0.5).any()):
                    target_q = sensor_q / float(value_scale)
                    q_loss = F.smooth_l1_loss(pred_q[sensor_q_mask > 0.5], target_q[sensor_q_mask > 0.5])
            elif variant in {"two_row_calibrated_factorized", "two_row_learned_blend_factorized"}:
                scores, _q_adv, pred_q_abs = model.forward_scores(x, slot, return_abs_q=True)
                log_probs = F.log_softmax(scores.reshape(scores.shape[0], -1), dim=1).reshape_as(scores)
                policy_loss = -(sensor_pi * log_probs).sum(dim=(1, 2)).mean()
                q_loss = torch.zeros((), device=device)
                if sensor_q is not None and sensor_q_mask is not None and bool((sensor_q_mask > 0.5).any()):
                    target_q = sensor_q / float(value_scale)
                    q_loss = F.smooth_l1_loss(pred_q_abs[sensor_q_mask > 0.5], target_q[sensor_q_mask > 0.5])
            else:
                policy_loss, q_loss, log_probs = two_row_factorized_step(model, x, slot, sensor_pi, sensor_q, sensor_q_mask, value_scale)
                if variant == "two_row_factored_loss":
                    scores, _pred_q = model.forward_scores(x, slot)
                    policy_loss = 0.5 * policy_loss + 0.5 * factorized_marginal_supervision_loss(scores, sensor_pi)
                if variant == "two_row_auxflat_factorized":
                    aux_logits = model.forward_aux_flat(x, slot)
                    aux_log_probs = F.log_softmax(aux_logits.reshape(aux_logits.shape[0], -1), dim=1).reshape_as(aux_logits)
                    aux_loss = -(sensor_pi * aux_log_probs).sum(dim=(1, 2)).mean()
                    policy_loss = 0.75 * policy_loss + 0.25 * aux_loss
            search_calibration_weight = float(getattr(args, "search_calibration_weight", 0.0))
            if search_calibration_weight > 0.0:
                policy_loss = policy_loss + search_calibration_weight * search_mass_calibration_loss(log_probs, calibration_target_pi)
            value_pred = model.forward_value(x, slot)
        else:
            policy_loss, q_loss, log_probs = flat_step(model, x, slot, sensor_pi, sensor_q, sensor_q_mask, value_scale)
            _scores, _q, value_pred = model.forward_physical_flat(x, slot)
        value_loss = F.smooth_l1_loss(value_pred, ret) if value_pred is not None else torch.zeros((), device=device)
        loss = policy_loss + float(args.q_loss_weight) * q_loss + float(args.value_loss_weight) * value_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, int(args.log_every)) == 0 or step == int(args.train_steps) - 1:
            if variant != "binary_type":
                top1, search_acc = top_metrics(log_probs.detach(), sensor_pi)
            print(
                {
                    "variant": variant,
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "q_loss": float(q_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "top1": top1,
                    "search_acc": search_acc,
                },
                flush=True,
            )
    return model.eval()


def train_value_critic(variant: str, targets, args, device):
    model = make_physical_model(variant, args).to(device)
    rng = np.random.default_rng(int(args.model_seed))
    order = rng.permutation(len(targets))
    split = max(1, int(0.8 * len(order)))
    train_idx = order[:split]
    val_idx = order[split:] if split < len(order) else order[:1]
    abs_ret = [abs(float(t.ret)) for t in targets]
    value_scale = max(1.0, float(np.percentile(abs_ret, 90))) if abs_ret else 1.0
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)

    def predict(batch_idx):
        x, slot, _sensor_pi, _sensor_q, _sensor_q_mask = batch_tensors(targets, batch_idx, device)
        if hasattr(model, "forward_value"):
            pred = model.forward_value(x, slot)
        elif hasattr(model, "forward_physical_flat"):
            _scores, _q, pred = model.forward_physical_flat(x, slot)
        else:
            out = model(x, slot)
            pred = out[-1] if isinstance(out, tuple) else torch.zeros((x.shape[0],), device=device)
        return pred

    for step in range(int(args.train_steps)):
        idx = rng.choice(train_idx, size=min(int(args.batch_size), len(train_idx)), replace=len(train_idx) < int(args.batch_size))
        pred = predict(idx)
        ret = torch.tensor([float(targets[int(i)].ret) / value_scale for i in idx], dtype=torch.float32, device=device)
        loss = F.smooth_l1_loss(pred.reshape_as(ret), ret)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, int(args.log_every)) == 0 or step == int(args.train_steps) - 1:
            with torch.no_grad():
                vi = rng.choice(val_idx, size=min(int(args.batch_size), len(val_idx)), replace=len(val_idx) < int(args.batch_size))
                vp = predict(vi).detach().cpu().numpy().reshape(-1) * value_scale
                vt = np.asarray([float(targets[int(i)].ret) for i in vi], dtype=np.float32)
                mae = float(np.mean(np.abs(vp - vt)))
                corr = float(np.corrcoef(vp, vt)[0, 1]) if len(vp) > 1 and np.std(vp) > 1e-8 and np.std(vt) > 1e-8 else 0.0
            print({"critic_variant": variant, "step": step, "loss": float(loss.detach().cpu()), "val_mae": mae, "val_corr": corr, "value_scale": value_scale}, flush=True)
    model.eval()
    state_out = Path(args.out).with_name(Path(args.out).stem + f"_{variant}_critic.pt")
    torch.save(model.state_dict(), state_out)
    print({"saved_critic": str(state_out), "variant": variant}, flush=True)
    return model


class PhysicalHeadPlanner:
    def __init__(
        self,
        model,
        variant: str,
        env_cfg: dict,
        policy_weight: float = 1.0,
        q_weight: float = 1.0,
        search_score_bias: float = 0.0,
        utility_weight: float = 0.0,
    ):
        self.model = model.eval()
        self.variant = str(variant)
        self.env_cfg = dict(env_cfg)
        self.policy_weight = float(policy_weight)
        self.q_weight = float(q_weight)
        self.search_score_bias = float(search_score_bias)
        self.utility_weight = float(utility_weight)
        self.adapt = adapter()

    def score_actions(self, obs, selected=None, elapsed: float = 0.0, search_count: int = 0, track_count: int = 0, last: int = -1) -> np.ndarray:
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        selected = set() if selected is None else set(selected)
        tok = tokenize(self.adapt, obs, selected=selected, search_count=int(search_count))
        slot = slot_features(obs, float(elapsed), int(search_count), int(track_count), int(last), 200.0)
        with torch.inference_mode():
            x = torch.from_numpy(tok).float().unsqueeze(0)
            s = torch.from_numpy(slot).float().unsqueeze(0)
            if self.variant in {
                "two_row_factorized",
                "two_row_factorized_qnorm",
                "two_row_factorized_adaptive",
                "separated_type_target_factorized",
                "two_row_factored_loss",
                "two_row_auxflat_factorized",
                "two_row_calibrated_factorized",
                "two_row_learned_blend_factorized",
                "two_row_strong_joint_factorized",
                "two_row_coupled_factorized",
                "two_row_coupled_factored_loss",
                "two_row_coupled_qpolicy",
                "two_row_coupled_qpolicy_factored_loss",
                "two_row_action_attention",
                "two_row_action_attention_factored_loss",
                "two_row_action_attention_factor_only",
                "two_row_action_attention_qpolicy",
                "two_row_action_attention_qpolicy_factored_loss",
                "two_row_action_attention_branchfair_qpolicy",
                "two_row_action_attention_autoregressive",
                "alphastar_factorized",
                "two_row_calibrated_action_attention_qpolicy_factored_loss",
                "two_row_full_shared_action_qpolicy_factored_loss",
                "two_row_flat_residual",
                "two_row_flat_residual_factored_loss",
                "flat_action_attention",
            }:
                scores, q = self.model.forward_scores(x, s)
                if self.variant == "two_row_factorized_qnorm":
                    finite = torch.isfinite(scores) & (scores > -1e8)
                    if bool(finite.any()):
                        q_valid = q[finite]
                        q_center = q_valid.mean()
                        q_scale = q_valid.std(unbiased=False).clamp_min(0.25)
                        q = torch.where(finite, (q - q_center) / q_scale, torch.zeros_like(q))
                    score = (self.policy_weight * scores + self.q_weight * q).squeeze(0).cpu().numpy()
                elif self.variant == "two_row_factorized_adaptive":
                    active_count = int(np.asarray(obs.get("active_mask", []), dtype=bool)[:MAXT].sum())
                    finite = torch.isfinite(scores) & (scores > -1e8)
                    q_norm = q
                    if bool(finite.any()):
                        q_valid = q[finite]
                        q_center = q_valid.mean()
                        q_scale = q_valid.std(unbiased=False).clamp_min(0.25)
                        q_norm = torch.where(finite, (q - q_center) / q_scale, torch.zeros_like(q))
                    if active_count <= 30:
                        score_t = scores + 0.5 * q_norm
                    elif active_count <= 50:
                        score_t = scores
                    else:
                        score_t = scores + 0.5 * q
                    score = score_t.squeeze(0).cpu().numpy()
                else:
                    score = (self.policy_weight * scores + self.q_weight * q).squeeze(0).cpu().numpy()
            elif self.variant == "two_row_joint_factorized":
                scores, q, utility = self.model.forward_scores(x, s)
                score = (
                    self.policy_weight * scores
                    + self.q_weight * q
                    + self.utility_weight * utility
                ).squeeze(0).cpu().numpy()
            elif flat_search_slots(self.variant) is not None:
                track_scores, track_q, search_scores, search_q, _ = self.model.forward_search20(x, s)
                search_slot = min(max(0, int(search_count)), int(self.model.search_slots) - 1)
                score = (self.policy_weight * track_scores + self.q_weight * track_q).squeeze(0).cpu().numpy()
                search_score = (
                    self.policy_weight * search_scores[:, search_slot, :]
                    + self.q_weight * search_q[:, search_slot, :]
                ).squeeze(0).cpu().numpy()
                score[0, :] = search_score
            else:
                scores, q, _ = self.model.forward_physical_flat(x, s)
                score = (self.policy_weight * scores + self.q_weight * q).squeeze(0).cpu().numpy()
        score = np.asarray(score, dtype=np.float32).copy()
        score[0, :] += self.search_score_bias
        return score

    def plan(self, obs, budget_ms=200):
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        selected = set()
        plan = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        while elapsed < float(budget_ms) and len(plan) < 64:
            score = self.score_actions(obs, selected=selected, elapsed=elapsed, search_count=search_count, track_count=track_count, last=last)
            cands = physical_candidates(obs, top_k=MAXT)
            if not cands:
                break
            best_action = None
            best_score = -np.inf
            if self.variant == "two_row_action_attention_branchfair_qpolicy":
                for sensor_idx in range(2):
                    search_actions = []
                    track_actions = []
                    track_scores = []
                    for action in cands:
                        base, sensor = xs_decode_action(int(action), MAXT)
                        if int(base) < 0:
                            continue
                        sidx = 0 if sensor is None else int(sensor)
                        if sidx != sensor_idx:
                            continue
                        val = float(score[int(base), sidx])
                        if int(base) == 0:
                            search_actions.append((int(action), val))
                        else:
                            track_actions.append((int(action), val))
                            track_scores.append(val)
                    if search_actions:
                        action, val = max(search_actions, key=lambda x: x[1])
                        if val > best_score:
                            best_action, best_score = int(action), float(val)
                    if track_actions:
                        action, _ = max(track_actions, key=lambda x: x[1])
                        branch_val = float(max(track_scores))
                        if branch_val > best_score:
                            best_action, best_score = int(action), branch_val
            else:
                for action in cands:
                    base, sensor = xs_decode_action(int(action), MAXT)
                    if int(base) < 0:
                        continue
                    sidx = 0 if sensor is None else int(sensor)
                    val = float(score[int(base), sidx])
                    if val > best_score:
                        best_action, best_score = int(action), val
            if best_action is None:
                break
            plan.append(best_action)
            base, _ = xs_decode_action(best_action, MAXT)
            if int(base) == 0:
                search_count += 1
                dt = 10.0
            else:
                selected.add(int(base))
                track_count += 1
                dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
                dt = float(dwell[int(base) - 1]) if int(base) - 1 < len(dwell) else 10.0
            elapsed += max(1.0, dt)
            last = int(base)
        return plan if plan else [xs_s_search_action(MAXT)]


JOINT_ACTION_BASE_LOCAL = 1_000_000
JOINT_ACTION_STRIDE_LOCAL = 1_000


def encode_joint_action_local(s_action: int, x_action: int) -> int:
    return int(JOINT_ACTION_BASE_LOCAL + int(s_action) * JOINT_ACTION_STRIDE_LOCAL + int(x_action))


class AutoregressiveBeamPlanner:
    """Faithful autoregressive decoder: top S actions, then X conditioned on each S."""

    def __init__(
        self,
        model: AutoregressiveActionAttentionFactorizedNet,
        env_cfg: dict,
        s_top_k: int = 6,
        x_top_k: int = 6,
        policy_weight: float = 1.0,
        q_weight: float = 1.0,
        search_score_bias: float = 0.0,
    ):
        self.model = model.eval()
        self.env_cfg = dict(env_cfg)
        self.s_top_k = max(1, int(s_top_k))
        self.x_top_k = max(1, int(x_top_k))
        self.policy_weight = float(policy_weight)
        self.q_weight = float(q_weight)
        self.search_score_bias = float(search_score_bias)
        self.adapt = adapter()

    @staticmethod
    def _action_duration(obs: dict, action: int) -> float:
        base, sensor = xs_decode_action(int(action), MAXT)
        if int(base) == 0:
            return 10.0
        dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        dt = float(dwell[int(base) - 1]) if int(base) > 0 and int(base) - 1 < len(dwell) else 10.0
        if sensor == 1:
            dt *= 0.5
        return max(1.0, dt)

    def _state_tensors(self, obs, selected, elapsed: float, search_count: int, track_count: int, last: int):
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        tok = tokenize(self.adapt, obs, selected=set(selected), search_count=int(search_count))
        slot = slot_features(obs, float(elapsed), int(search_count), int(track_count), int(last), 200.0)
        return obs, tok, slot

    def _rank_s_actions(self, obs, selected, elapsed, search_count, track_count, last):
        obs, tok, slot = self._state_tensors(obs, selected, elapsed, search_count, track_count, last)
        with torch.inference_mode():
            x = torch.from_numpy(tok).float().unsqueeze(0)
            s = torch.from_numpy(slot).float().unsqueeze(0)
            s_scores, s_q, _x_scores, _x_q = self.model.forward_x_conditioned_on_s(x, s, torch.zeros((1,), dtype=torch.long))
            total_s = (self.policy_weight * s_scores + self.q_weight * s_q).squeeze(0).cpu().numpy()
        total_s = np.asarray(total_s, dtype=np.float32).copy()
        total_s[0] += self.search_score_bias
        ranked = []
        for action in physical_candidates(obs, top_k=MAXT):
            base, sensor = xs_decode_action(int(action), MAXT)
            if sensor != 0 or int(base) < 0 or int(base) in selected:
                continue
            ranked.append((float(total_s[int(base)]), int(base), int(action), tok, slot, obs))
        ranked.sort(reverse=True, key=lambda item: item[0])
        return ranked[: self.s_top_k]

    def _rank_x_for_s(self, obs, tok, slot, s_base: int, selected):
        with torch.inference_mode():
            x = torch.from_numpy(tok).float().unsqueeze(0)
            s = torch.from_numpy(slot).float().unsqueeze(0)
            _s_scores, _s_q, x_scores, x_q = self.model.forward_x_conditioned_on_s(x, s, torch.tensor([int(s_base)], dtype=torch.long))
            total_x = (self.policy_weight * x_scores + self.q_weight * x_q).squeeze(0).cpu().numpy()
        total_x = np.asarray(total_x, dtype=np.float32).copy()
        total_x[0] += self.search_score_bias
        ranked = []
        for action in physical_candidates(obs, top_k=MAXT):
            base, sensor = xs_decode_action(int(action), MAXT)
            if sensor != 1 or int(base) < 0 or int(base) in selected:
                continue
            if int(base) > 0 and int(base) == int(s_base):
                continue
            ranked.append((float(total_x[int(base)]), int(base), int(action)))
        ranked.sort(reverse=True, key=lambda item: item[0])
        return ranked[: self.x_top_k]

    def _best_joint(self, obs, selected, elapsed, search_count, track_count, last):
        best = None
        for score, action in self._joint_candidates(obs, selected, elapsed, search_count, track_count, last):
            if best is None or score > best[0]:
                best = (score, action)
        return int(best[1]) if best is not None else None

    def _joint_candidates(self, obs, selected, elapsed, search_count, track_count, last):
        out = []
        for s_score, s_base, s_action, tok, slot, obs_attached in self._rank_s_actions(obs, selected, elapsed, search_count, track_count, last):
            for x_score, _x_base, x_action in self._rank_x_for_s(obs_attached, tok, slot, int(s_base), selected):
                score = float(s_score) + float(x_score)
                action = encode_joint_action_local(int(s_action), int(x_action))
                out.append((score, int(action)))
        out.sort(reverse=True, key=lambda item: item[0])
        return out

    def _append_effect(self, obs, action: int, selected: set[int], elapsed: float, search_count: int, track_count: int, last: int):
        encoded = int(action) - JOINT_ACTION_BASE_LOCAL
        atoms = [encoded // JOINT_ACTION_STRIDE_LOCAL, encoded % JOINT_ACTION_STRIDE_LOCAL] if int(action) >= JOINT_ACTION_BASE_LOCAL else [int(action)]
        dt = min(self._action_duration(obs, int(a)) for a in atoms) if len(atoms) > 1 else self._action_duration(obs, int(atoms[0]))
        for atom in atoms:
            base, _sensor = xs_decode_action(int(atom), MAXT)
            if int(base) == 0:
                search_count += 1
            elif int(base) > 0:
                selected.add(int(base))
                track_count += 1
            last = int(base)
        return selected, elapsed + max(1.0, float(dt)), search_count, track_count, last

    def plan(self, obs, budget_ms=200):
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        selected: set[int] = set()
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        plan = []
        while elapsed < float(budget_ms) and len(plan) < 64:
            action = self._best_joint(obs, selected, elapsed, search_count, track_count, last)
            if action is None:
                break
            plan.append(int(action))
            selected, elapsed, search_count, track_count, last = self._append_effect(
                obs, int(action), selected, elapsed, search_count, track_count, last
            )
        return plan if plan else [encode_joint_action_local(xs_s_search_action(MAXT), xs_x_search_action(MAXT))]


class AutoregressiveBeamProposalPlanner(AutoregressiveBeamPlanner):
    def __init__(self, *args, beams: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.beams = max(1, int(beams))

    def _tail_from(self, obs, first_action: int, budget_ms: float):
        selected: set[int] = set()
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        selected, elapsed, search_count, track_count, last = self._append_effect(
            attach_env_obs(obs, self.env_cfg, True, True),
            int(first_action),
            selected,
            elapsed,
            search_count,
            track_count,
            last,
        )
        plan = [int(first_action)]
        while elapsed < float(budget_ms) and len(plan) < 64:
            action = self._best_joint(obs, selected, elapsed, search_count, track_count, last)
            if action is None:
                break
            plan.append(int(action))
            selected, elapsed, search_count, track_count, last = self._append_effect(
                attach_env_obs(obs, self.env_cfg, True, True),
                int(action),
                selected,
                elapsed,
                search_count,
                track_count,
                last,
            )
        return plan

    def plan(self, obs, budget_ms=200):
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        firsts = self._joint_candidates(obs, set(), 0.0, 0, 0, -1)[: self.beams]
        plans = []
        seen = set()
        for _score, action in firsts:
            plan = self._tail_from(obs, int(action), float(budget_ms))
            key = tuple(plan)
            if key and key not in seen:
                seen.add(key)
                plans.append(plan)
        if plans:
            return plans
        return [[encode_joint_action_local(xs_s_search_action(MAXT), xs_x_search_action(MAXT))]]


class AlphaStarBeamPlanner(AutoregressiveBeamPlanner):
    def __init__(self, model: AlphaStarFactorizedNet, env_cfg: dict, **kwargs):
        super().__init__(model, env_cfg, **kwargs)

    def _rank_s_actions(self, obs, selected, elapsed, search_count, track_count, last):
        obs, tok, slot = self._state_tensors(obs, selected, elapsed, search_count, track_count, last)
        with torch.inference_mode():
            x = torch.from_numpy(tok).float().unsqueeze(0)
            s = torch.from_numpy(slot).float().unsqueeze(0)
            _scores, q, aux = self.model.forward_scores_conditioned(x, s, torch.zeros((1,), dtype=torch.long))
            type_logits = aux["s_type_logits"].squeeze(0).cpu().numpy()
            target_logits = aux["s_target_logits"].squeeze(0).cpu().numpy()
            q_s = q[:, :, 0].squeeze(0).cpu().numpy()
        ranked = []
        search_score = float(self.policy_weight * type_logits[0] + self.q_weight * q_s[0] + self.search_score_bias)
        ranked.append((search_score, 0, xs_s_search_action(MAXT), tok, slot, obs))
        track_type_score = float(type_logits[1])
        for action in physical_candidates(obs, top_k=MAXT):
            base, sensor = xs_decode_action(int(action), MAXT)
            if sensor != 0 or int(base) <= 0 or int(base) in selected:
                continue
            score = float(self.policy_weight * (track_type_score + target_logits[int(base)]) + self.q_weight * q_s[int(base)])
            ranked.append((score, int(base), int(action), tok, slot, obs))
        ranked.sort(reverse=True, key=lambda item: item[0])
        return ranked[: self.s_top_k]

    def _rank_x_for_s(self, obs, tok, slot, s_base: int, selected):
        with torch.inference_mode():
            x = torch.from_numpy(tok).float().unsqueeze(0)
            s = torch.from_numpy(slot).float().unsqueeze(0)
            _scores, q, aux = self.model.forward_scores_conditioned(x, s, torch.tensor([int(s_base)], dtype=torch.long))
            type_logits = aux["x_type_logits"].squeeze(0).cpu().numpy()
            target_logits = aux["x_target_logits"].squeeze(0).cpu().numpy()
            q_x = q[:, :, 1].squeeze(0).cpu().numpy()
        ranked = []
        search_score = float(self.policy_weight * type_logits[0] + self.q_weight * q_x[0] + self.search_score_bias)
        ranked.append((search_score, 0, xs_x_search_action(MAXT)))
        track_type_score = float(type_logits[1])
        for action in physical_candidates(obs, top_k=MAXT):
            base, sensor = xs_decode_action(int(action), MAXT)
            if sensor != 1 or int(base) <= 0 or int(base) in selected:
                continue
            if int(base) > 0 and int(base) == int(s_base):
                continue
            score = float(self.policy_weight * (track_type_score + target_logits[int(base)]) + self.q_weight * q_x[int(base)])
            ranked.append((score, int(base), int(action)))
        ranked.sort(reverse=True, key=lambda item: item[0])
        return ranked[: self.x_top_k]

    def _joint_candidates(self, obs, selected, elapsed, search_count, track_count, last):
        s_ranked = self._rank_s_actions(obs, selected, elapsed, search_count, track_count, last)
        if not s_ranked:
            return []
        s_rows = [int(item[1]) for item in s_ranked]
        tok = s_ranked[0][3]
        slot = s_ranked[0][4]
        obs_attached = s_ranked[0][5]
        with torch.inference_mode():
            x = torch.from_numpy(tok).float().unsqueeze(0).expand(len(s_rows), -1, -1).contiguous()
            s = torch.from_numpy(slot).float().unsqueeze(0).expand(len(s_rows), -1).contiguous()
            _scores, q, aux = self.model.forward_scores_conditioned(x, s, torch.tensor(s_rows, dtype=torch.long))
            type_logits = aux["x_type_logits"].cpu().numpy()
            target_logits = aux["x_target_logits"].cpu().numpy()
            q_x = q[:, :, 1].cpu().numpy()
        x_actions = []
        for action in physical_candidates(obs_attached, top_k=MAXT):
            base, sensor = xs_decode_action(int(action), MAXT)
            if sensor != 1 or int(base) < 0 or int(base) in selected:
                continue
            x_actions.append((int(base), int(action)))

        out = []
        for row_idx, (s_score, s_base, s_action, _tok, _slot, _obs) in enumerate(s_ranked):
            ranked_x = []
            search_score = float(self.policy_weight * type_logits[row_idx, 0] + self.q_weight * q_x[row_idx, 0] + self.search_score_bias)
            ranked_x.append((search_score, 0, xs_x_search_action(MAXT)))
            track_type_score = float(type_logits[row_idx, 1])
            for x_base, x_action in x_actions:
                if int(x_base) <= 0:
                    continue
                if int(x_base) == int(s_base):
                    continue
                score = float(self.policy_weight * (track_type_score + target_logits[row_idx, int(x_base)]) + self.q_weight * q_x[row_idx, int(x_base)])
                ranked_x.append((score, int(x_base), int(x_action)))
            ranked_x.sort(reverse=True, key=lambda item: item[0])
            for x_score, _x_base, x_action in ranked_x[: self.x_top_k]:
                out.append((float(s_score) + float(x_score), encode_joint_action_local(int(s_action), int(x_action))))
        out.sort(reverse=True, key=lambda item: item[0])
        return out


class AlphaStarBeamProposalPlanner(AlphaStarBeamPlanner):
    def __init__(self, *args, beams: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.beams = max(1, int(beams))

    def _tail_from(self, obs, first_action: int, budget_ms: float):
        selected: set[int] = set()
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        selected, elapsed, search_count, track_count, last = self._append_effect(
            attach_env_obs(obs, self.env_cfg, True, True),
            int(first_action),
            selected,
            elapsed,
            search_count,
            track_count,
            last,
        )
        plan = [int(first_action)]
        while elapsed < float(budget_ms) and len(plan) < 64:
            action = self._best_joint(obs, selected, elapsed, search_count, track_count, last)
            if action is None:
                break
            plan.append(int(action))
            selected, elapsed, search_count, track_count, last = self._append_effect(
                attach_env_obs(obs, self.env_cfg, True, True),
                int(action),
                selected,
                elapsed,
                search_count,
                track_count,
                last,
            )
        return plan

    def plan(self, obs, budget_ms=200):
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        firsts = self._joint_candidates(obs, set(), 0.0, 0, 0, -1)[: self.beams]
        plans = []
        seen = set()
        for _score, action in firsts:
            plan = self._tail_from(obs, int(action), float(budget_ms))
            key = tuple(plan)
            if key and key not in seen:
                seen.add(key)
                plans.append(plan)
        if plans:
            return plans
        return [[encode_joint_action_local(xs_s_search_action(MAXT), xs_x_search_action(MAXT))]]


class BinaryTypePlanner:
    def __init__(self, model, env_cfg: dict, margin: float = 0.0):
        self.model = model.eval()
        self.env_cfg = dict(env_cfg)
        self.margin = float(margin)
        self.adapt = adapter()

    def plan(self, obs, budget_ms=200):
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        selected = set()
        plan = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        edf = EDFPlanner(MAXT)
        while elapsed < float(budget_ms) and len(plan) < 64:
            tok = tokenize(self.adapt, obs, selected=selected, search_count=search_count)
            slot = slot_features(obs, elapsed, search_count, track_count, last, float(budget_ms))
            with torch.inference_mode():
                logit, _q, _value = self.model(torch.from_numpy(tok).float().unsqueeze(0), torch.from_numpy(slot).float().unsqueeze(0))
            choose_search = bool(float(logit[0].cpu()) >= self.margin)
            if choose_search:
                action = xs_s_search_action(MAXT)
                dt = 10.0
                search_count += 1
                base = 0
            else:
                tracks = [int(a) for a in edf.plan(obs, budget_ms=int(max(1.0, float(budget_ms) - elapsed))) if xs_decode_action(int(a), MAXT)[0] != 0]
                if not tracks:
                    action = xs_s_search_action(MAXT)
                    dt = 10.0
                    search_count += 1
                    base = 0
                else:
                    action = int(tracks[0])
                    base, _ = xs_decode_action(action, MAXT)
                    selected.add(int(base))
                    track_count += 1
                    dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
                    dt = float(dwell[int(base) - 1]) if int(base) - 1 < len(dwell) else 10.0
            plan.append(int(action))
            elapsed += max(1.0, float(dt))
            last = int(base)
        return plan if plan else [xs_s_search_action(MAXT)]


class TwoSensorTypePlanner:
    def __init__(self, model, env_cfg: dict, margin: float = 0.0, q_weight: float = 0.0):
        self.model = model.eval()
        self.env_cfg = dict(env_cfg)
        self.margin = float(margin)
        self.q_weight = float(q_weight)
        self.adapt = adapter()

    def _edf_track_for_sensor(self, obs, sensor: int, selected: set[int], remaining_ms: float) -> int | None:
        edf = EDFPlanner(MAXT)
        for action in edf.plan(obs, budget_ms=int(max(1.0, remaining_ms))):
            base, _old_sensor = xs_decode_action(int(action), MAXT)
            if int(base) <= 0 or int(base) in selected:
                continue
            if int(sensor) == 0:
                return xs_s_track_action(int(base), MAXT)
            return xs_x_track_action(int(base), MAXT)
        return None

    def plan(self, obs, budget_ms=200):
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        selected = set()
        plan = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        while elapsed < float(budget_ms) and len(plan) < 64:
            tok = tokenize(self.adapt, obs, selected=selected, search_count=search_count)
            slot = slot_features(obs, elapsed, search_count, track_count, last, float(budget_ms))
            with torch.inference_mode():
                logits, q, _value = self.model(torch.from_numpy(tok).float().unsqueeze(0), torch.from_numpy(slot).float().unsqueeze(0))
            sensor_actions = []
            for sensor in (0, 1):
                busy_key = "s_band_busy_ms" if sensor == 0 else "x_band_busy_ms"
                if sensor == 1 and not int(obs.get("enable_x_band", 0)):
                    continue
                if float(obs.get(busy_key, 0.0)) > 0.0:
                    continue
                q_advantage = float((q[0, sensor, 0] - q[0, sensor, 1]).cpu())
                type_score = float(logits[0, sensor].cpu()) + self.q_weight * q_advantage
                choose_search = bool(type_score >= self.margin)
                if choose_search:
                    sensor_actions.append(xs_s_search_action(MAXT) if sensor == 0 else xs_x_search_action(MAXT))
                else:
                    action = self._edf_track_for_sensor(obs, sensor, selected, float(budget_ms) - elapsed)
                    if action is not None:
                        sensor_actions.append(int(action))
            if not sensor_actions:
                sensor_actions = [xs_s_search_action(MAXT)]
            progressed = False
            for action in sensor_actions:
                base, _sensor = xs_decode_action(int(action), MAXT)
                if int(base) > 0 and int(base) in selected:
                    continue
                plan.append(int(action))
                if int(base) == 0:
                    search_count += 1
                    dt = 10.0
                elif int(base) > 0:
                    selected.add(int(base))
                    track_count += 1
                    dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
                    dt = float(dwell[int(base) - 1]) if int(base) - 1 < len(dwell) else 10.0
                else:
                    dt = 1.0
                elapsed += max(1.0, float(dt))
                last = int(base)
                progressed = True
                if elapsed >= float(budget_ms) or len(plan) >= 64:
                    break
            if not progressed:
                break
        return plan if plan else [xs_s_search_action(MAXT)]


class SensorModeGate(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedTwoSensorTypePlanner:
    def __init__(
        self,
        policy_model,
        qpolicy_model,
        gate: SensorModeGate,
        env_cfg: dict,
        policy_margin: float = 1.5,
        qpolicy_margin: float = 0.0,
    ):
        self.policy_model = policy_model.eval()
        self.qpolicy_model = qpolicy_model.eval()
        self.gate = gate.eval()
        self.env_cfg = dict(env_cfg)
        self.policy_margin = float(policy_margin)
        self.qpolicy_margin = float(qpolicy_margin)
        self.adapt = adapter()

    def _edf_track_for_sensor(self, obs, sensor: int, selected: set[int], remaining_ms: float) -> int | None:
        edf = EDFPlanner(MAXT)
        for action in edf.plan(obs, budget_ms=int(max(1.0, remaining_ms))):
            base, _old_sensor = xs_decode_action(int(action), MAXT)
            if int(base) <= 0 or int(base) in selected:
                continue
            if int(sensor) == 0:
                return xs_s_track_action(int(base), MAXT)
            return xs_x_track_action(int(base), MAXT)
        return None

    def plan(self, obs, budget_ms=200):
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        selected = set()
        plan = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        while elapsed < float(budget_ms) and len(plan) < 64:
            tok = tokenize(self.adapt, obs, selected=selected, search_count=search_count)
            slot = slot_features(obs, elapsed, search_count, track_count, last, float(budget_ms))
            xt = torch.from_numpy(tok).float().unsqueeze(0)
            st = torch.from_numpy(slot).float().unsqueeze(0)
            with torch.inference_mode():
                policy_logits, _policy_q, _policy_value = self.policy_model(xt, st)
                qpolicy_logits, _qpolicy_q, _qpolicy_value = self.qpolicy_model(xt, st)
                gate_in = torch.cat(
                    [
                        st[:, None, :].expand(-1, 2, -1),
                        policy_logits[:, :, None],
                        qpolicy_logits[:, :, None],
                    ],
                    dim=-1,
                )
                use_qpolicy = self.gate(gate_in).argmax(dim=-1).squeeze(0).cpu().numpy()
            sensor_actions = []
            for sensor in (0, 1):
                busy_key = "s_band_busy_ms" if sensor == 0 else "x_band_busy_ms"
                if sensor == 1 and not int(obs.get("enable_x_band", 0)):
                    continue
                if float(obs.get(busy_key, 0.0)) > 0.0:
                    continue
                if int(use_qpolicy[int(sensor)]) == 1:
                    choose_search = bool(float(qpolicy_logits[0, sensor].cpu()) >= self.qpolicy_margin)
                else:
                    choose_search = bool(float(policy_logits[0, sensor].cpu()) >= self.policy_margin)
                if choose_search:
                    sensor_actions.append(xs_s_search_action(MAXT) if sensor == 0 else xs_x_search_action(MAXT))
                else:
                    action = self._edf_track_for_sensor(obs, sensor, selected, float(budget_ms) - elapsed)
                    if action is not None:
                        sensor_actions.append(int(action))
            if not sensor_actions:
                sensor_actions = [xs_s_search_action(MAXT)]
            progressed = False
            for action in sensor_actions:
                base, _sensor = xs_decode_action(int(action), MAXT)
                if int(base) > 0 and int(base) in selected:
                    continue
                plan.append(int(action))
                if int(base) == 0:
                    search_count += 1
                    dt = 10.0
                elif int(base) > 0:
                    selected.add(int(base))
                    track_count += 1
                    dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
                    dt = float(dwell[int(base) - 1]) if int(base) - 1 < len(dwell) else 10.0
                else:
                    dt = 1.0
                elapsed += max(1.0, float(dt))
                last = int(base)
                progressed = True
                if elapsed >= float(budget_ms) or len(plan) >= 64:
                    break
            if not progressed:
                break
        return plan if plan else [xs_s_search_action(MAXT)]


class LoadGatedTwoSensorTypePlanner:
    def __init__(
        self,
        policy_model,
        qpolicy_model,
        env_cfg: dict,
        policy_margin: float = 1.5,
        qpolicy_margin: float = 0.0,
        active_cutoff: float = 30.0,
        rate_cutoff: float = 3.5,
    ):
        self.policy = TwoSensorTypePlanner(policy_model, env_cfg, margin=float(policy_margin), q_weight=0.0)
        self.qpolicy = TwoSensorTypePlanner(qpolicy_model, env_cfg, margin=float(qpolicy_margin), q_weight=0.0)
        self.env_cfg = dict(env_cfg)
        self.active_cutoff = float(active_cutoff)
        self.rate_cutoff = float(rate_cutoff)
        self.episode_initial_active: float | None = None

    def plan(self, obs, budget_ms=200):
        obs2 = attach_env_obs(obs, self.env_cfg, True, True)
        active = float(np.asarray(obs2.get("active_mask", []), dtype=bool)[:MAXT].sum())
        if self.episode_initial_active is None:
            self.episode_initial_active = active
        rate = float(obs2.get("arrival_rate", self.env_cfg.get("arrival_rate", 0.0)))
        if float(self.episode_initial_active) <= self.active_cutoff and rate <= self.rate_cutoff:
            return self.qpolicy.plan(obs, budget_ms=budget_ms)
        return self.policy.plan(obs, budget_ms=budget_ms)


class DiverseFirstActionPlanner:
    def __init__(self, base: PhysicalHeadPlanner, branches: int = 4):
        self.base = base
        self.branches = max(1, int(branches))
        self.env_cfg = dict(base.env_cfg)

    def _top_first_actions(self, obs) -> list[int]:
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        score = self.base.score_actions(obs)
        ranked = []
        for action in physical_candidates(obs, top_k=MAXT):
            base_id, sensor = xs_decode_action(int(action), MAXT)
            if int(base_id) < 0:
                continue
            sidx = 0 if sensor is None else int(sensor)
            ranked.append((float(score[int(base_id), sidx]), int(action)))
        ranked.sort(reverse=True, key=lambda x: x[0])
        out = []
        seen = set()
        for _score, action in ranked:
            if action in seen:
                continue
            out.append(int(action))
            seen.add(int(action))
            if len(out) >= self.branches:
                break
        return out

    def plan(self, obs, budget_ms=200):
        first_actions = self._top_first_actions(obs)
        if not first_actions:
            return [self.base.plan(obs, budget_ms=budget_ms)]
        plans = []
        seen = set()
        for first in first_actions:
            plan = [int(first)]
            base_id, _sensor = xs_decode_action(int(first), MAXT)
            selected = {int(base_id)} if int(base_id) > 0 else set()
            elapsed = 10.0
            search_count = 1 if int(base_id) == 0 else 0
            track_count = 0 if int(base_id) == 0 else 1
            last = int(base_id)
            while elapsed < float(budget_ms) and len(plan) < 64:
                score = self.base.score_actions(obs, selected=selected, elapsed=elapsed, search_count=search_count, track_count=track_count, last=last)
                cands = physical_candidates(attach_env_obs(obs, self.env_cfg, True, True), top_k=MAXT)
                best_action = None
                best_score = -np.inf
                for action in cands:
                    b, sensor = xs_decode_action(int(action), MAXT)
                    if int(b) < 0 or int(b) in selected:
                        continue
                    sidx = 0 if sensor is None else int(sensor)
                    val = float(score[int(b), sidx])
                    if val > best_score:
                        best_action, best_score = int(action), val
                if best_action is None:
                    break
                plan.append(best_action)
                b, _sensor = xs_decode_action(best_action, MAXT)
                if int(b) == 0:
                    search_count += 1
                    dt = 10.0
                else:
                    selected.add(int(b))
                    track_count += 1
                    dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
                    dt = float(dwell[int(b) - 1]) if int(b) - 1 < len(dwell) else 10.0
                elapsed += max(1.0, dt)
                last = int(b)
            key = tuple(plan)
            if key not in seen:
                plans.append(plan)
                seen.add(key)
        return plans if plans else [self.base.plan(obs, budget_ms=budget_ms)]


class PairedFirstActionsPlanner:
    def __init__(self, base: PhysicalHeadPlanner, pairs: int = 8, per_sensor_top: int = 6):
        self.base = base
        self.pairs = max(1, int(pairs))
        self.per_sensor_top = max(1, int(per_sensor_top))
        self.env_cfg = dict(base.env_cfg)

    def _ranked_by_sensor(self, obs) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        score = self.base.score_actions(obs)
        ranked = [[], []]
        for action in physical_candidates(obs, top_k=MAXT):
            base_id, sensor = xs_decode_action(int(action), MAXT)
            if int(base_id) < 0 or sensor is None:
                continue
            sidx = int(sensor)
            if sidx not in {0, 1}:
                continue
            ranked[sidx].append((float(score[int(base_id), sidx]), int(action)))
        for sidx in (0, 1):
            ranked[sidx].sort(reverse=True, key=lambda x: x[0])
            deduped = []
            seen = set()
            for item_score, action in ranked[sidx]:
                if action in seen:
                    continue
                deduped.append((float(item_score), int(action)))
                seen.add(int(action))
                if len(deduped) >= self.per_sensor_top:
                    break
            ranked[sidx] = deduped
        return ranked[0], ranked[1]

    def _tail(self, obs, prefix: list[int], budget_ms: float) -> list[int]:
        selected = set()
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        for action in prefix:
            base_id, _sensor = xs_decode_action(int(action), MAXT)
            if int(base_id) == 0:
                search_count += 1
                elapsed += 10.0
            elif int(base_id) > 0:
                selected.add(int(base_id))
                track_count += 1
                dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
                elapsed += float(dwell[int(base_id) - 1]) if int(base_id) - 1 < len(dwell) else 10.0
            last = int(base_id)

        plan = list(prefix)
        while elapsed < float(budget_ms) and len(plan) < 64:
            score = self.base.score_actions(obs, selected=selected, elapsed=elapsed, search_count=search_count, track_count=track_count, last=last)
            best_action = None
            best_score = -np.inf
            for action in physical_candidates(attach_env_obs(obs, self.env_cfg, True, True), top_k=MAXT):
                b, sensor = xs_decode_action(int(action), MAXT)
                if int(b) < 0 or int(b) in selected:
                    continue
                sidx = 0 if sensor is None else int(sensor)
                val = float(score[int(b), sidx])
                if val > best_score:
                    best_action, best_score = int(action), val
            if best_action is None:
                break
            plan.append(best_action)
            b, _sensor = xs_decode_action(best_action, MAXT)
            if int(b) == 0:
                search_count += 1
                dt = 10.0
            else:
                selected.add(int(b))
                track_count += 1
                dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
                dt = float(dwell[int(b) - 1]) if int(b) - 1 < len(dwell) else 10.0
            elapsed += max(1.0, dt)
            last = int(b)
        return plan

    def plan(self, obs, budget_ms=200):
        s_ranked, x_ranked = self._ranked_by_sensor(obs)
        pairs = []
        for s_score, s_action in s_ranked:
            s_base, _ = xs_decode_action(int(s_action), MAXT)
            for x_score, x_action in x_ranked:
                x_base, _ = xs_decode_action(int(x_action), MAXT)
                if int(s_base) > 0 and int(s_base) == int(x_base):
                    continue
                pairs.append((float(s_score) + float(x_score), [int(s_action), int(x_action)]))
                pairs.append((float(s_score) + float(x_score), [int(x_action), int(s_action)]))
        pairs.sort(reverse=True, key=lambda x: x[0])
        plans = []
        seen = set()
        for _score, prefix in pairs:
            key = tuple(prefix)
            if key in seen:
                continue
            plans.append(self._tail(obs, prefix, float(budget_ms)))
            seen.add(key)
            if len(plans) >= self.pairs:
                break
        if plans:
            return plans
        return [self.base.plan(obs, budget_ms=budget_ms)]


def eval_models(models: dict, args, exact_args):
    rows = []
    search_biases = parse_floats(args.search_score_biases)
    for seed in parse_ints(args.eval_seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                env_cfg = env_cfg_for(float(rate), exact_args)
                env_cfg["enable_x_band"] = 1
                planners = {"EDF": EDFPlanner(MAXT), "EST": ESTPlanner(MAXT)}
                for name, model in models.items():
                    if name == "binary_type":
                        planners[name] = BinaryTypePlanner(model, env_cfg, margin=float(args.binary_margin))
                    elif name in {"two_sensor_type", "two_sensor_type_qpolicy"}:
                        for bias in search_biases:
                            suffix = f"{float(bias):+g}".replace("+", "p").replace("-", "m").replace(".", "p")
                            planners[f"{name}_m{suffix}"] = TwoSensorTypePlanner(
                                model,
                                env_cfg,
                                margin=float(bias),
                                q_weight=float(args.q_score_weight),
                            )
                    elif name == "two_row_action_attention_autoregressive":
                        planners[name + "_beam"] = AutoregressiveBeamPlanner(
                            model,
                            env_cfg,
                            s_top_k=6,
                            x_top_k=6,
                            policy_weight=float(args.policy_score_weight),
                            q_weight=float(args.q_score_weight),
                        )
                    elif name == "alphastar_factorized":
                        planners[name + "_beam"] = AlphaStarBeamPlanner(
                            model,
                            env_cfg,
                            s_top_k=6,
                            x_top_k=6,
                            policy_weight=float(args.policy_score_weight),
                            q_weight=float(args.q_score_weight),
                        )
                    else:
                        for bias in search_biases:
                            suffix = f"{float(bias):+g}".replace("+", "p").replace("-", "m").replace(".", "p")
                            planners[f"{name}_sb{suffix}"] = PhysicalHeadPlanner(
                                model,
                                name,
                                env_cfg,
                                policy_weight=float(args.policy_score_weight),
                                q_weight=float(args.q_score_weight),
                                search_score_bias=float(bias),
                            )
                for name, planner in planners.items():
                    t0 = time.perf_counter()
                    w, _ = run_fixed(planner, name, int(initial), MAXT, int(seed), int(args.eval_windows), 200, engine_env_cfg(env_cfg))
                    s = summarize_window_df(w, "fixed")
                    rows.append({"method": name, "initial": int(initial), "rate": float(rate), "seed": int(seed), "reward": float(s.get("reward_per_200ms_eq", np.nan)), "search": float(s.get("search_fraction", np.nan)), "seconds": time.perf_counter() - t0})
                    print(rows[-1], flush=True)
    raw = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out, index=False)
    summary = raw.groupby("method").agg(reward=("reward", "mean"), search=("search", "mean"), n=("reward", "size")).reset_index().sort_values("reward", ascending=False)
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False), flush=True)
    return raw


def selected_variants(text: str) -> list[str]:
    allowed = {
        "flat",
        "two_row_factorized",
        "two_row_factorized_qnorm",
        "two_row_factorized_adaptive",
        "two_row_factored_loss",
        "two_row_auxflat_factorized",
        "two_row_calibrated_factorized",
        "two_row_learned_blend_factorized",
        "two_row_joint_factorized",
        "two_row_strong_joint_factorized",
        "two_row_coupled_factorized",
        "two_row_coupled_factored_loss",
        "two_row_coupled_qpolicy",
        "two_row_coupled_qpolicy_factored_loss",
        "two_row_action_attention",
        "two_row_action_attention_factored_loss",
        "two_row_action_attention_factor_only",
        "two_row_action_attention_qpolicy",
        "two_row_action_attention_qpolicy_factored_loss",
        "two_row_action_attention_branchfair_qpolicy",
        "two_row_action_attention_autoregressive",
        "alphastar_factorized",
        "two_row_calibrated_action_attention_qpolicy_factored_loss",
        "two_row_full_shared_action_qpolicy_factored_loss",
        "two_row_flat_residual",
        "two_row_flat_residual_factored_loss",
        "flat_action_attention",
        "binary_type",
        "two_sensor_type",
        "two_sensor_type_qpolicy",
        "separated_type_target_factorized",
    }
    out = []
    for item in str(text).split(","):
        name = item.strip()
        if not name:
            continue
        if name == "all":
            return ["flat", "two_row_factorized", "binary_type"]
        if name not in allowed and flat_search_slots(name) is None:
            raise ValueError(f"unknown variant: {name}")
        if name not in out:
            out.append(name)
    return out or ["two_row_factorized"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "data", "train_eval", "critic"], default="all")
    ap.add_argument("--targets-out", default="CreateValid1/results/two_sensor_physical_smoke_targets.pt")
    ap.add_argument("--out", default="CreateValid1/results/two_sensor_physical_head_eval.csv")
    ap.add_argument("--initials", default="40,60")
    ap.add_argument("--rates", default="2,4")
    ap.add_argument("--train-seeds", default="901,902")
    ap.add_argument("--eval-seeds", default="903")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--eval-windows", type=int, default=100)
    ap.add_argument("--max-targets", type=int, default=512)
    ap.add_argument("--max-targets-per-cell", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--tail-windows", type=int, default=1)
    ap.add_argument("--behavior-policy", choices=["est", "edf"], default="edf")
    ap.add_argument("--tail-policy", choices=["est", "edf"], default="edf")
    ap.add_argument("--policy-tau", type=float, default=5.0)
    ap.add_argument("--potential-weight", type=float, default=1.0)
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--train-steps", type=int, default=180)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--q-loss-weight", type=float, default=0.25)
    ap.add_argument("--value-loss-weight", type=float, default=0.25)
    ap.add_argument("--search-calibration-weight", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=45)
    ap.add_argument("--model-seed", type=int, default=123)
    ap.add_argument("--cell-balanced-sampling", action="store_true")
    ap.add_argument("--dagger-rounds", type=int, default=0)
    ap.add_argument("--dagger-variant", choices=["flat", "two_row_factorized", "binary_type"], default="binary_type")
    ap.add_argument("--binary-margin", type=float, default=0.0)
    ap.add_argument("--variants", default="flat,two_row_factorized,binary_type")
    ap.add_argument("--policy-score-weight", type=float, default=1.0)
    ap.add_argument("--q-score-weight", type=float, default=1.0)
    ap.add_argument("--search-score-biases", default="0")
    ap.add_argument("--bootstrap-state", default="")
    ap.add_argument("--bootstrap-variant", default="flat")
    ap.add_argument("--bootstrap-value-weight", type=float, default=0.0)
    ap.add_argument("--hard-policy-target", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(int(args.model_seed))
    np.random.seed(int(args.model_seed))
    torch.set_num_threads(1)
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = True
    exact_args.single_sensor = False
    target_path = Path(args.targets_out)
    if args.mode in {"all", "data"}:
        targets = collect_targets(args, exact_args, target_path)
    else:
        targets = usable_targets(target_path)
    if args.mode == "critic":
        device = torch.device("cpu")
        for variant in selected_variants(args.variants):
            train_value_critic(variant, targets, args, device)
        return
    if args.mode in {"all", "train_eval"}:
        device = torch.device("cpu")
        models = {}
        for variant in selected_variants(args.variants):
            models[variant] = train_head(variant, targets, args, device)
            state_out = Path(args.out).with_name(Path(args.out).stem + f"_{variant}.pt")
            torch.save(models[variant].state_dict(), state_out)
            print({"saved_state": str(state_out), "variant": variant}, flush=True)
        for round_idx in range(int(args.dagger_rounds)):
            source = models[str(args.dagger_variant)]

            def behavior_factory(env_cfg, source=source):
                if str(args.dagger_variant) == "binary_type":
                    return BinaryTypePlanner(source, env_cfg, margin=float(args.binary_margin))
                return PhysicalHeadPlanner(source, str(args.dagger_variant), env_cfg, policy_weight=float(args.policy_score_weight), q_weight=float(args.q_score_weight), search_score_bias=0.0)

            old_max = int(args.max_targets)
            args.max_targets = int(old_max)
            dagger_path = target_path.with_name(target_path.stem + f"_dagger{round_idx + 1}.pt")
            new_targets = collect_targets(args, exact_args, dagger_path, behavior_factory=behavior_factory)
            targets = [*targets, *new_targets]
            args.max_targets = old_max
            print({"dagger_round": round_idx + 1, "total_targets": len(targets), "new_targets": len(new_targets)}, flush=True)
            models = {}
            for variant in selected_variants(args.variants):
                models[variant] = train_head(variant, targets, args, device)
                state_out = Path(args.out).with_name(Path(args.out).stem + f"_dagger{round_idx + 1}_{variant}.pt")
                torch.save(models[variant].state_dict(), state_out)
                print({"saved_state": str(state_out), "variant": variant}, flush=True)
        eval_models(models, args, exact_args)


if __name__ == "__main__":
    main()
