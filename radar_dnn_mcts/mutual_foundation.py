"""MutualRadar foundation training.

This is the clean AlphaZero/Akbar-style path for the radar scheduler:

1. A factorized two-head transformer predicts a prior over atomic actions:
   Search vs Track, then target rank if Track is chosen.
2. MCTS uses that prior in PUCT to produce an improved visit-count policy.
3. The same model is trained on MCTS policy targets plus value/Q targets.

The latest high-performing one-pass policy+Q utility remains useful for fast
deployment, but this file is the cleaner foundation-model training loop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from final_radar_campaign import MAXT, build_env, run_fixed, seedall, summarize_window_df
from mutual_features import SLOT_DIM, TOKEN_DIM, slot_features, tokenize
from load_adaptive_train_eval import OUT, make_env
from realistic_reward_retrain import adapter
from repaired_campaign_tools import EDFPlanner, ESTPlanner, SEARCH_DWELL_MS, make_reference_planner, planner_delay_cfg
from strict_window_report import execute_plan_until_budget
from pufferlib.ocean.radarxs.models.mcts import MCTSPlanner, Node


RUN_OUT = Path(os.environ.get("MutualRadar_RUN_OUT", str(OUT / "MutualRadar_foundation")))
RUN_OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(int(os.environ.get("RADARXS_TORCH_THREADS", "1")))


def _copy_plan_obs(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    for k, v in obs.items():
        out[k] = v.copy() if isinstance(v, np.ndarray) else v
    return out


def _with_arrival_context(obs: Dict[str, np.ndarray], arrival_rate: float = 0.0, use_arrival_feature: bool = False):
    if not use_arrival_feature and "arrival_rate" in obs and "use_arrival_feature" in obs:
        return obs
    out = _copy_plan_obs(obs)
    out["arrival_rate"] = float(out.get("arrival_rate", out.get("poisson_rate_per_second", arrival_rate)))
    if use_arrival_feature:
        out["use_arrival_feature"] = 1.0
    return out


def _infer_refresh_timers(t_desired: np.ndarray, t_deadline: np.ndarray, priority: np.ndarray, revisit_time_scale: float = 1.0):
    denom = np.maximum(1e-3, 1.5 + 0.75 * priority.astype(np.float32))
    base_desired = (t_deadline.astype(np.float32) - t_desired.astype(np.float32)) / denom
    base_desired = np.clip(base_desired * float(revisit_time_scale), 100.0, 30000.0).astype(np.float32)
    deadline_mult = np.maximum(1.0, 2.5 - 0.75 * priority.astype(np.float32))
    base_deadline = np.clip(base_desired * deadline_mult, 100.0, 30000.0).astype(np.float32)
    return base_desired, base_deadline


def _refresh_stalest_sectors(grid: np.ndarray, n_sectors: int = 4) -> np.ndarray:
    if grid is None or len(grid) < 300:
        return np.array([], dtype=np.int32)
    grid2d = grid.reshape(10, 30)
    best_r, best_c, best_sum = 0, 0, np.inf
    for r in range(0, 9, 2):
        for c in range(0, 29, 2):
            s = grid2d[r, c] + grid2d[r, c + 1] + grid2d[r + 1, c] + grid2d[r + 1, c + 1]
            if s < best_sum:
                best_r, best_c, best_sum = r, c, float(s)
    idx = np.array([best_r * 30 + best_c, best_r * 30 + best_c + 1, (best_r + 1) * 30 + best_c, (best_r + 1) * 30 + best_c + 1], dtype=np.int32)
    grid[idx] = 3010.0
    return idx


def _apply_search_refresh_to_plan_obs(obs: Dict[str, np.ndarray], refreshed: np.ndarray, refresh_tracked: bool = False, refresh_gain: float = 1.0):
    if refreshed is None or len(refreshed) == 0:
        return
    active = np.asarray(obs["active_mask"]).astype(bool)
    tracked = active & (np.asarray(obs["t_deadline"], dtype=np.float32) >= 0.0)
    az_bin = np.asarray(obs.get("az_bin", np.zeros_like(obs["t_desired"])), dtype=np.float32)
    el_bin = np.asarray(obs.get("el_bin", np.zeros_like(obs["t_desired"])), dtype=np.float32)
    priority = np.asarray(obs.get("priority", np.zeros_like(obs["t_desired"])), dtype=np.float32)
    refresh_des, refresh_dead = _infer_refresh_timers(np.asarray(obs["t_desired"], dtype=np.float32), np.asarray(obs["t_deadline"], dtype=np.float32), priority)
    az = np.clip(np.round(az_bin * 29.0).astype(np.int32), 0, 29)
    el = np.clip(np.round(el_bin * 9.0).astype(np.int32), 0, 9)
    sectors = el * 30 + az
    hit = np.isin(sectors, refreshed)
    untracked_hit = active & (~tracked) & hit
    obs["t_desired"][untracked_hit] = refresh_des[untracked_hit]
    obs["t_deadline"][untracked_hit] = refresh_dead[untracked_hit]
    if refresh_tracked and refresh_gain > 0.0:
        tracked_hit = active & tracked & hit
        gain = float(np.clip(refresh_gain, 0.0, 1.0))
        obs["t_desired"][tracked_hit] = obs["t_desired"][tracked_hit] + gain * (refresh_des[tracked_hit] - obs["t_desired"][tracked_hit])
        obs["t_deadline"][tracked_hit] = obs["t_deadline"][tracked_hit] + gain * (refresh_dead[tracked_hit] - obs["t_deadline"][tracked_hit])


def advance_plan_obs(
    obs: Dict[str, np.ndarray],
    action: int,
    dt: float,
    *,
    search_refresh_tracked: bool = False,
    search_refresh_gain: float = 1.0,
):
    """Approximate the C environment transition inside a planned 200 ms window.

    This is not used for reward accounting; it only keeps learned direct decoders
    from evaluating later slots against stale pre-action target timers.
    """
    if action == 0:
        grid = obs.get("grid")
        if isinstance(grid, np.ndarray):
            refreshed = _refresh_stalest_sectors(grid)
            _apply_search_refresh_to_plan_obs(obs, refreshed, refresh_tracked=search_refresh_tracked, refresh_gain=search_refresh_gain)
    else:
        idx = int(action) - 1
        active = np.asarray(obs["active_mask"]).astype(bool)
        deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
        if 0 <= idx < len(deadline) and active[idx] and deadline[idx] >= 0.0:
            priority = np.asarray(obs.get("priority", np.zeros_like(obs["t_desired"])), dtype=np.float32)
            refresh_des, refresh_dead = _infer_refresh_timers(np.asarray(obs["t_desired"], dtype=np.float32), deadline, priority)
            obs["t_desired"][idx] = refresh_des[idx]
            obs["t_deadline"][idx] = refresh_dead[idx]
    active = np.asarray(obs["active_mask"]).astype(bool)
    tracked = active & (np.asarray(obs["t_deadline"], dtype=np.float32) >= 0.0)
    obs["t_desired"][tracked] = obs["t_desired"][tracked] - float(dt)
    obs["t_deadline"][tracked] = obs["t_deadline"][tracked] - float(dt)
    expired = active & (np.asarray(obs["t_deadline"], dtype=np.float32) < 0.0)
    # Keep expired targets active but untracked, matching the observation-level
    # convention where search can rediscover them later.
    if "grid" in obs and isinstance(obs["grid"], np.ndarray):
        obs["grid"] = obs["grid"] - float(dt)


class MutualRadarNet(nn.Module):
    """Factorized policy/value/Q network for one atomic scheduling decision.

    Policy factorization:
        P(search | s) from type_head
        P(track_i | s) = P(track | s) * softmax_i(track_head)

    Q factorization:
        Q_type(s, search/track) captures branch value.
        Q_track(s, i) captures target-specific action value.
    """

    def __init__(
        self,
        token_dim: int = TOKEN_DIM,
        slot_dim: int = SLOT_DIM,
        d_model: int = 96,
        nhead: int = 4,
        nlayers: int = 2,
        head_arch: str = "baseline",
    ):
        super().__init__()
        self.head_arch = str(head_arch)
        self.token_proj = nn.Linear(token_dim, d_model)
        self.slot_proj = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, d_model), nn.GELU())
        self.cls_token = nn.Parameter(torch.randn(d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.05,
            activation="gelu",
        )
        # Nested tensor conversion is expensive for our small, single-state
        # online MCTS calls.  Keep the standard padded-mask path instead.
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers, enable_nested_tensor=False, mask_check=False)
        self.type_head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.track_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.value_head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.type_q_head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.track_q_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

        # Optional specialist heads: the encoder is shared, but branch choice
        # and target ranking get separate post-encoder experts instead of one
        # shallow projection.  This is the closest lightweight analogue to a
        # soft MoE without introducing discrete routing.
        self.type_specialist = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.GELU())
        self.track_specialist = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.GELU())
        self.value_specialist = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.GELU())
        self.type_head_special = nn.Linear(d_model, 1)
        self.track_head_special = nn.Linear(d_model, 1)
        self.value_head_special = nn.Linear(d_model, 1)
        self.type_q_head_special = nn.Linear(d_model, 2)
        self.track_q_head_special = nn.Linear(d_model, 1)
        self.type_head_branch_context = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.type_q_head_branch_context = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        moe_experts = 3
        self.moe_gate = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, moe_experts))
        self.moe_value_gate = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, moe_experts))
        self.moe_type_residual = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)) for _ in range(moe_experts)]
        )
        self.moe_type_q_residual = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2)) for _ in range(moe_experts)]
        )
        self.moe_value_residual = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)) for _ in range(moe_experts)]
        )
        self.moe_track_residual = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)) for _ in range(moe_experts)]
        )
        self.moe_track_q_residual = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)) for _ in range(moe_experts)]
        )
        self.moe_residual_logit_scale = nn.Parameter(torch.tensor(-3.0))
        self.sensor_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.sensor_q_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.physical_flat_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.physical_flat_q_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.type_logit_scale = nn.Parameter(torch.zeros(()))
        self.type_logit_bias = nn.Parameter(torch.zeros(()))
        self.track_logit_scale = nn.Parameter(torch.zeros(()))
        self.type_track_coupling = nn.Parameter(torch.zeros(()))

    def forward(self, tokens: torch.Tensor, slot: torch.Tensor):
        token_active = tokens[:, :, 4] > 0.5
        token_active[:, 0] = True
        selected = tokens[:, :, 8] > 0.5

        emb = self.token_proj(tokens)
        cls = self.cls_token.unsqueeze(0).unsqueeze(0).expand(tokens.shape[0], 1, -1)
        emb = torch.cat([cls, emb], dim=1)
        cls_valid = torch.ones((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device)
        out = self.encoder(emb, src_key_padding_mask=~torch.cat([cls_valid, token_active], dim=1))
        cls_out = out[:, 0, :]
        tok_out = out[:, 1:, :]
        return self.forward_heads(cls_out, tok_out, selected, token_active, slot)

    def encode_tokens(self, tokens: torch.Tensor):
        token_active = tokens[:, :, 4] > 0.5
        token_active[:, 0] = True
        selected = tokens[:, :, 8] > 0.5

        emb = self.token_proj(tokens)
        cls = self.cls_token.unsqueeze(0).unsqueeze(0).expand(tokens.shape[0], 1, -1)
        emb = torch.cat([cls, emb], dim=1)
        cls_valid = torch.ones((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device)
        out = self.encoder(emb, src_key_padding_mask=~torch.cat([cls_valid, token_active], dim=1))
        return out[:, 0, :], out[:, 1:, :], selected, token_active

    def forward_heads(self, cls_out: torch.Tensor, tok_out: torch.Tensor, selected: torch.Tensor, token_active: torch.Tensor, slot: torch.Tensor):
        slot_emb = self.slot_proj(slot)

        type_ctx = torch.cat([cls_out, slot_emb], dim=-1)
        search_tok = tok_out[:, 0, :]
        type_ctx_branch = torch.cat([cls_out, search_tok, slot_emb], dim=-1)
        cls_rep = cls_out.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
        slot_rep = slot_emb.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
        track_ctx = torch.cat([tok_out, cls_rep, slot_rep], dim=-1)
        if self.head_arch == "branch_context":
            type_logit = self.type_head_branch_context(type_ctx_branch).squeeze(-1)
            type_q = self.type_q_head_branch_context(type_ctx_branch)
            value = self.value_head(type_ctx).squeeze(-1)
            track_logits = self.track_head(track_ctx).squeeze(-1)
            track_q = self.track_q_head(track_ctx).squeeze(-1)
        elif self.head_arch == "moe":
            base_type_logit = self.type_head_branch_context(type_ctx_branch).squeeze(-1)
            base_type_q = self.type_q_head_branch_context(type_ctx_branch)
            base_value = self.value_head(type_ctx).squeeze(-1)
            base_track_logits = self.track_head(track_ctx).squeeze(-1)
            base_track_q = self.track_q_head(track_ctx).squeeze(-1)

            branch_gate = torch.softmax(self.moe_gate(type_ctx_branch), dim=-1)
            value_gate = torch.softmax(self.moe_value_gate(type_ctx), dim=-1)
            type_res = torch.stack([head(type_ctx_branch).squeeze(-1) for head in self.moe_type_residual], dim=-1)
            type_q_res = torch.stack([head(type_ctx_branch) for head in self.moe_type_q_residual], dim=-1)
            value_res = torch.stack([head(type_ctx).squeeze(-1) for head in self.moe_value_residual], dim=-1)
            track_res = torch.stack([head(track_ctx).squeeze(-1) for head in self.moe_track_residual], dim=-1)
            track_q_res = torch.stack([head(track_ctx).squeeze(-1) for head in self.moe_track_q_residual], dim=-1)
            residual_scale = 0.2 * torch.sigmoid(self.moe_residual_logit_scale)
            type_logit = base_type_logit + residual_scale * torch.sum(type_res * branch_gate, dim=-1)
            type_q = base_type_q + residual_scale * torch.sum(type_q_res * branch_gate.unsqueeze(1), dim=-1)
            value = base_value + residual_scale * torch.sum(value_res * value_gate, dim=-1)
            track_logits = base_track_logits + residual_scale * torch.sum(track_res * branch_gate.unsqueeze(1), dim=-1)
            track_q = base_track_q + residual_scale * torch.sum(track_q_res * branch_gate.unsqueeze(1), dim=-1)
        elif self.head_arch == "specialized":
            type_feat = self.type_specialist(type_ctx)
            track_feat = self.track_specialist(track_ctx)
            value_feat = self.value_specialist(type_ctx)
            type_logit = self.type_head_special(type_feat).squeeze(-1)
            type_q = self.type_q_head_special(type_feat)
            value = self.value_head_special(value_feat).squeeze(-1)
            track_logits = self.track_head_special(track_feat).squeeze(-1)
            track_q = self.track_q_head_special(track_feat).squeeze(-1)
        else:
            type_logit = self.type_head(type_ctx).squeeze(-1)
            type_q = self.type_q_head(type_ctx)
            value = self.value_head(type_ctx).squeeze(-1)
            track_logits = self.track_head(track_ctx).squeeze(-1)
            track_q = self.track_q_head(track_ctx).squeeze(-1)

        type_scale = torch.exp(torch.clamp(self.type_logit_scale, -2.0, 2.0))
        track_scale = torch.exp(torch.clamp(self.track_logit_scale, -2.0, 2.0))
        type_logit = type_logit * type_scale + self.type_logit_bias
        track_logits = track_logits * track_scale

        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        track_logits_for_coupling = track_logits.masked_fill(~track_mask, -1e9)
        has_track = torch.any(track_mask, dim=1)
        track_energy = torch.zeros_like(type_logit)
        if bool(has_track.any()):
            track_energy = track_energy.clone()
            track_energy[has_track] = torch.logsumexp(track_logits_for_coupling[has_track], dim=1)
        coupling = torch.clamp(self.type_track_coupling, 0.0, 1.0)
        type_logit = type_logit - coupling * track_energy
        track_logits = track_logits.masked_fill(~track_mask, -1e9)
        track_q = track_q.masked_fill(~track_mask, 0.0)
        return type_logit, track_logits, value, type_q, track_q

    def forward_with_sensor(self, tokens: torch.Tensor, slot: torch.Tensor):
        token_active = tokens[:, :, 4] > 0.5
        token_active[:, 0] = True
        selected = tokens[:, :, 8] > 0.5
        cls_out, tok_out, selected, token_active = self.encode_tokens(tokens)
        type_logit, track_logits, value, type_q, track_q = self.forward_heads(
            cls_out, tok_out, selected, token_active, slot
        )
        slot_emb = self.slot_proj(slot)
        cls_rep = cls_out.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
        slot_rep = slot_emb.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
        sensor_ctx = torch.cat([tok_out, cls_rep, slot_rep], dim=-1)
        sensor_logits = self.sensor_head(sensor_ctx)
        sensor_q = self.sensor_q_head(sensor_ctx)
        action_mask = token_active & ~selected
        action_mask[:, 0] = True
        sensor_logits = sensor_logits.masked_fill(~action_mask[:, :, None], -1e9)
        sensor_q = sensor_q.masked_fill(~action_mask[:, :, None], 0.0)
        return type_logit, track_logits, value, type_q, track_q, sensor_logits, sensor_q

    def forward_physical_flat(self, tokens: torch.Tensor, slot: torch.Tensor):
        token_active = tokens[:, :, 4] > 0.5
        token_active[:, 0] = True
        selected = tokens[:, :, 8] > 0.5
        cls_out, tok_out, _, _ = self.encode_tokens(tokens)
        slot_emb = self.slot_proj(slot)
        cls_rep = cls_out.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
        slot_rep = slot_emb.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
        ctx = torch.cat([tok_out, cls_rep, slot_rep], dim=-1)
        logits = self.physical_flat_head(ctx)
        q = self.physical_flat_q_head(ctx)
        action_mask = token_active & ~selected
        action_mask[:, 0] = True
        logits = logits.masked_fill(~action_mask[:, :, None], -1e9)
        q = q.masked_fill(~action_mask[:, :, None], 0.0)
        value = self.value_head(torch.cat([cls_out, slot_emb], dim=-1)).squeeze(-1)
        return logits, q, value


def action_priors_from_logits(type_logit: torch.Tensor, track_logits: torch.Tensor, prior_mode: str = "factorized") -> np.ndarray:
    probs = np.zeros((MAXT + 1,), dtype=np.float32)
    finite = torch.isfinite(track_logits) & (track_logits > -1e8)
    if prior_mode == "flat":
        logits = torch.full((MAXT + 1,), -1e9, dtype=track_logits.dtype, device=track_logits.device)
        logits[0] = type_logit
        logits[1:][finite[1:]] = track_logits[1:][finite[1:]]
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy().astype(np.float32)
    else:
        if prior_mode == "branch_corrected":
            search_energy = type_logit
            if bool(torch.any(finite)):
                track_branch_energy = torch.logsumexp(track_logits[finite], dim=0)
                branch_probs = torch.softmax(torch.stack([track_branch_energy, search_energy]), dim=0)
                p_search = float(branch_probs[1].detach().cpu().item())
            else:
                p_search = 1.0
        else:
            p_search = float(torch.sigmoid(type_logit).detach().cpu().item())
        probs[0] = p_search
        if bool(torch.any(finite)):
            track_p = torch.softmax(track_logits[finite], dim=0).detach().cpu().numpy().astype(np.float32)
            idx = torch.where(finite)[0].detach().cpu().numpy()
            probs[idx] = (1.0 - p_search) * track_p
    s = float(np.sum(probs))
    if s > 0:
        probs /= s
    return probs


class MutualRadarDirectPlanner:
    """Fast one-pass deployment planner using policy + optional Q utility.

    For factorized policy heads, ``direct_mode="branch"`` is the correct
    decoder: first decide search-vs-track, then rank targets only inside the
    track branch.  Comparing the search atom against each individual target
    atom biases the planner toward search whenever many targets split the track
    probability mass.
    """

    def __init__(
        self,
        model: MutualRadarNet,
        alpha: float = 0.0,
        beta: float = 0.0,
        threshold: float = 0.0,
        direct_mode: str = "prob",
        q_residual_gate: str = "off",
        q_gate_margin: float = 0.0,
        allow_retrack: bool = False,
        stateless_context: bool = False,
        cache_encoder: bool = False,
        simulate_state: bool = True,
        search_refresh_tracked: bool = False,
        search_refresh_gain: float = 1.0,
        sensor_action_mode: str = "implicit",
        disable_x_search: bool = False,
    ):
        self.model = model.eval()
        self.adapt = adapter()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.direct_mode = str(direct_mode)
        self.q_residual_gate = str(q_residual_gate)
        self.q_gate_margin = float(q_gate_margin)
        self.allow_retrack = bool(allow_retrack)
        self.stateless_context = bool(stateless_context)
        self.cache_encoder = bool(cache_encoder)
        self.simulate_state = bool(simulate_state)
        self.search_refresh_tracked = bool(search_refresh_tracked)
        self.search_refresh_gain = float(search_refresh_gain)
        self.sensor_action_mode = str(sensor_action_mode)
        self.disable_x_search = bool(disable_x_search)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def warmup(self, obs, budget_ms=200):
        _ = self.plan(obs, budget_ms=budget_ms)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def plan(self, obs, budget_ms=200):
        plan_obs = _copy_plan_obs(obs) if self.simulate_state else obs
        selected = set()
        plan: List[int] = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        dwell = np.asarray(plan_obs["t_dwell"], dtype=np.float32)
        s_busy = float(plan_obs.get("s_band_busy_ms", 0.0))
        x_busy = float(plan_obs.get("x_band_busy_ms", 0.0))
        encoded = None
        encoded_search_count = None
        while elapsed < float(budget_ms) and len(plan) < 64:
            feature_selected = set() if self.allow_retrack else selected
            feature_elapsed = 0.0 if self.stateless_context else elapsed
            feature_search_count = 0 if self.stateless_context else search_count
            feature_track_count = 0 if self.stateless_context else track_count
            feature_last = -1 if self.stateless_context else last
            if isinstance(plan_obs, dict):
                plan_obs["s_band_busy_ms"] = float(max(0.0, s_busy))
                plan_obs["x_band_busy_ms"] = float(max(0.0, x_busy))
            s = slot_features(plan_obs, feature_elapsed, feature_search_count, feature_track_count, feature_last, float(budget_ms))
            sensor_logits_t = sensor_q_t = None
            with torch.inference_mode():
                slot_t = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
                can_cache = self.cache_encoder and self.allow_retrack and (not feature_selected) and (not self.simulate_state)
                if can_cache and encoded is not None and encoded_search_count == feature_search_count:
                    cls_out, tok_out, selected_t, token_active = encoded
                    tl, tr, _, tq, tq_track = self.model.forward_heads(cls_out, tok_out, selected_t, token_active, slot_t)
                elif can_cache:
                    x = tokenize(self.adapt, plan_obs, selected=feature_selected, search_count=feature_search_count)
                    tokens_t = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
                    encoded = self.model.encode_tokens(tokens_t)
                    encoded_search_count = feature_search_count
                    cls_out, tok_out, selected_t, token_active = encoded
                    tl, tr, _, tq, tq_track = self.model.forward_heads(cls_out, tok_out, selected_t, token_active, slot_t)
                else:
                    x = tokenize(self.adapt, plan_obs, selected=feature_selected, search_count=feature_search_count)
                    tokens_t = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
                    if self.sensor_action_mode == "explicit_head" and hasattr(self.model, "forward_with_sensor"):
                        tl, tr, _, tq, tq_track, sensor_logits_t, sensor_q_t = self.model.forward_with_sensor(tokens_t, slot_t)
                    else:
                        tl, tr, _, tq, tq_track = self.model(tokens_t, slot_t)
                        sensor_logits_t = sensor_q_t = None
            type_logit = tl[0]
            type_q = tq[0].cpu().numpy()
            track_logits_t = tr[0]
            if self.direct_mode == "physical_q" and sensor_q_t is not None:
                max_trackers = int(len(np.asarray(plan_obs["active_mask"])))
                s_search = max_trackers + 3
                x_search = max_trackers + 4
                s_track_base = max_trackers + 5
                x_track_base = max_trackers + 5 + max_trackers
                active = np.asarray(plan_obs["active_mask"]).astype(bool)
                deadline = np.asarray(plan_obs["t_deadline"], dtype=np.float32)
                ranges = np.asarray(plan_obs.get("target_range", np.zeros_like(deadline)), dtype=np.float32)
                sensor_q_np = sensor_q_t[0].detach().cpu().numpy()
                track_q_np = tq_track[0].detach().cpu().numpy()
                candidates: List[Tuple[float, int, int, int, float]] = []

                def add_candidate(score: float, action: int, base_action: int, sensor: int, dt_value: float) -> None:
                    if np.isfinite(score):
                        candidates.append((float(score), int(action), int(base_action), int(sensor), float(dt_value)))

                if s_busy <= 0.0:
                    add_candidate(float(type_q[1] + sensor_q_np[0, 0]), s_search, 0, 0, SEARCH_DWELL_MS)
                x_enabled = bool(int(plan_obs.get("enable_x_band", 0))) and not self.disable_x_search
                if x_enabled and x_busy <= 0.0:
                    add_candidate(float(type_q[1] + sensor_q_np[0, 1]), x_search, 0, 1, SEARCH_DWELL_MS)

                tracked = active & (deadline >= 0.0)
                for idx in np.where(tracked)[0].astype(int).tolist():
                    base_candidate = int(idx) + 1
                    if (not self.allow_retrack) and base_candidate in selected:
                        continue
                    dwell_arr = np.asarray(plan_obs["t_dwell"], dtype=np.float32)
                    dt_s = float(dwell_arr[idx]) if idx < len(dwell_arr) else SEARCH_DWELL_MS
                    r = float(ranges[idx]) if idx < len(ranges) else 0.0
                    base_score = float(type_q[0] + track_q_np[base_candidate])
                    if s_busy <= 0.0 and 10_000_000.0 < r < 184_000_000.0:
                        add_candidate(
                            base_score + float(sensor_q_np[base_candidate, 0]),
                            s_track_base + base_candidate - 1,
                            base_candidate,
                            0,
                            dt_s,
                        )
                    if x_enabled and x_busy <= 0.0 and 5_000_000.0 < r < 100_000_000.0:
                        add_candidate(
                            base_score + float(sensor_q_np[base_candidate, 1]),
                            x_track_base + base_candidate - 1,
                            base_candidate,
                            1,
                            max(1.0, 0.5 * dt_s),
                        )

                if candidates:
                    _score, a, base_a, sensor, dt = max(candidates, key=lambda item: item[0])
                    if base_a == 0:
                        search_count += 1
                    else:
                        if not self.allow_retrack:
                            selected.add(base_a)
                        track_count += 1
                    if sensor == 0:
                        s_busy = max(s_busy, float(dt))
                    else:
                        x_busy = max(x_busy, float(dt))
                    plan.append(int(a))
                    if self.simulate_state:
                        advance_plan_obs(
                            plan_obs,
                            int(base_a),
                            max(1.0, float(dt)),
                            search_refresh_tracked=self.search_refresh_tracked,
                            search_refresh_gain=self.search_refresh_gain,
                        )
                    elapsed += max(1.0, float(dt))
                    s_busy = max(0.0, s_busy - max(1.0, float(dt)))
                    x_busy = max(0.0, x_busy - max(1.0, float(dt)))
                    last = int(base_a)
                    continue
                return plan if plan else [max_trackers + 1]
            if self.direct_mode == "q":
                track_scores = type_q[0] + tq_track[0].cpu().numpy()
                search_score = float(type_q[1])
                track_branch_score = None
            elif self.direct_mode == "branch":
                p_search = float(torch.sigmoid(type_logit).detach().cpu().item())
                if self.beta != 0.0:
                    q_shift = float(type_q[1] - type_q[0])
                    p_search = float(1.0 / (1.0 + np.exp(-(float(type_logit.detach().cpu()) + self.beta * q_shift))))
                track_scores = track_logits_t.cpu().numpy()
                if self.alpha != 0.0:
                    q_track_np = tq_track[0].cpu().numpy()
                    use_q_residual = True
                    gate = self.q_residual_gate.lower()
                    if gate not in {"", "off", "none"}:
                        finite_policy = np.isfinite(track_scores) & (track_scores > -1e8)
                        if np.any(finite_policy):
                            policy_order = np.argsort(np.where(finite_policy, track_scores, -np.inf))
                            policy_best = int(policy_order[-1])
                            policy_second = int(policy_order[-2]) if int(np.sum(finite_policy)) >= 2 else policy_best
                            policy_margin = (
                                float(track_scores[policy_best] - track_scores[policy_second])
                                if policy_best != policy_second
                                else np.inf
                            )
                            q_best = int(np.argmax(np.where(finite_policy, q_track_np, -np.inf)))
                            if gate == "agree":
                                use_q_residual = q_best == policy_best
                            elif gate == "uncertain":
                                use_q_residual = policy_margin <= self.q_gate_margin
                            elif gate == "agree_or_uncertain":
                                use_q_residual = (q_best == policy_best) or (policy_margin <= self.q_gate_margin)
                            elif gate == "agree_and_uncertain":
                                use_q_residual = (q_best == policy_best) and (policy_margin <= self.q_gate_margin)
                        else:
                            use_q_residual = False
                    if use_q_residual:
                        track_scores = track_scores + self.alpha * q_track_np
                search_score = p_search
                track_branch_score = None
            elif self.direct_mode == "flat":
                track_scores = track_logits_t.cpu().numpy() + self.alpha * tq_track[0].cpu().numpy()
                search_score = float(type_logit.cpu()) + self.beta * type_q[1]
                track_branch_score = None
            else:
                log_p_search = F.logsigmoid(type_logit).cpu().item()
                log_p_track_branch = F.logsigmoid(-type_logit).cpu().item()
                log_track = F.log_softmax(track_logits_t, dim=0).cpu().numpy()
                track_scores = log_p_track_branch + log_track + self.alpha * tq_track[0].cpu().numpy()
                search_score = log_p_search + self.beta * type_q[1]
                track_branch_score = None
            best_track = int(np.argmax(track_scores))
            has_track = np.isfinite(track_scores[best_track]) and track_scores[best_track] > -1e8
            if self.beta != 0.0 and has_track:
                track_scores = track_scores.copy()
                track_scores[best_track] += self.beta * type_q[0]
                best_track = int(np.argmax(track_scores))
                has_track = np.isfinite(track_scores[best_track]) and track_scores[best_track] > -1e8
            best_track_score = float(track_scores[best_track]) if has_track else -np.inf
            if self.direct_mode == "branch":
                choose_search = (not has_track) or (search_score >= self.threshold)
            else:
                choose_search = (not has_track) or (search_score - best_track_score >= self.threshold)
            if choose_search:
                base_a, dt = 0, SEARCH_DWELL_MS
                search_count += 1
            else:
                base_a = best_track
                if not self.allow_retrack:
                    selected.add(base_a)
                dwell = np.asarray(plan_obs["t_dwell"], dtype=np.float32)
                dt = float(dwell[base_a - 1]) if 1 <= base_a <= len(dwell) else SEARCH_DWELL_MS
                track_count += 1
            a = int(base_a)
            if self.sensor_action_mode == "explicit_head" and sensor_logits_t is not None:
                max_trackers = int(len(np.asarray(plan_obs["active_mask"])))
                s_search = max_trackers + 3
                x_search = max_trackers + 4
                s_track_base = max_trackers + 5
                x_track_base = max_trackers + 5 + max_trackers
                sensor_scores = sensor_logits_t[0, int(base_a)].detach().cpu().numpy().astype(np.float32)
                if sensor_q_t is not None and self.beta != 0.0:
                    sensor_scores = sensor_scores + self.beta * sensor_q_t[0, int(base_a)].detach().cpu().numpy().astype(np.float32)
                mask = np.ones((2,), dtype=bool)
                if int(plan_obs.get("enable_x_band", 0)) == 0:
                    mask[1] = False
                if base_a == 0 and self.disable_x_search:
                    mask[1] = False
                if s_busy > 0.0:
                    mask[0] = False
                if x_busy > 0.0:
                    mask[1] = False
                if base_a > 0 and "target_range" in plan_obs:
                    ranges = np.asarray(plan_obs["target_range"], dtype=np.float32)
                    r = float(ranges[base_a - 1]) if base_a - 1 < len(ranges) else 0.0
                    mask[0] = mask[0] and (10_000_000.0 < r < 184_000_000.0)
                    mask[1] = mask[1] and (5_000_000.0 < r < 100_000_000.0)
                if np.any(mask):
                    sensor_scores = np.where(mask, sensor_scores, -1e9)
                    sensor = int(np.argmax(sensor_scores))
                    if base_a == 0:
                        a = s_search if sensor == 0 else x_search
                    else:
                        a = (s_track_base + base_a - 1) if sensor == 0 else (x_track_base + base_a - 1)
                    if base_a > 0 and sensor == 1:
                        dt = max(1.0, 0.5 * float(dt))
                    if sensor == 0:
                        s_busy = max(s_busy, float(dt))
                    else:
                        x_busy = max(x_busy, float(dt))
                else:
                    a = max_trackers + 1
                    base_a = -1
                    dt = SEARCH_DWELL_MS
            plan.append(int(a))
            if self.simulate_state:
                advance_plan_obs(
                    plan_obs,
                    int(base_a),
                    max(1.0, float(dt)),
                    search_refresh_tracked=self.search_refresh_tracked,
                    search_refresh_gain=self.search_refresh_gain,
                )
            elapsed += max(1.0, float(dt))
            s_busy = max(0.0, s_busy - max(1.0, float(dt)))
            x_busy = max(0.0, x_busy - max(1.0, float(dt)))
            last = int(base_a)
        return plan if plan else [0]


@dataclass
class SearchTarget:
    x: np.ndarray
    slot: np.ndarray
    pi: np.ndarray
    q: np.ndarray
    q_mask: np.ndarray
    search_count: int
    track_count: int
    reward: float = 0.0
    ret: float = 0.0
    sensor_pi: Optional[np.ndarray] = None
    sensor_q: Optional[np.ndarray] = None
    sensor_q_mask: Optional[np.ndarray] = None
    initial: int = -1
    rate: float = 0.0
    seed: int = -1
    window: int = -1
    action_index: int = -1


class MutualRadarMCTSPlanner:
    """AlphaZero-style MCTS guided by MutualRadarNet priors."""

    def __init__(
        self,
        model: MutualRadarNet,
        env_cfg: Dict[str, float],
        rollouts: int = 16,
        c_puct: float = 1.25,
        expand_top_k: int = 16,
        simulation_window_ms: float = 200.0,
        training: bool = False,
        root_dirichlet_eps: float = 0.15,
        root_dirichlet_alpha: float = 0.3,
        q_scale: float = 1.0,
        use_q_head: bool = False,
        q_utility_weight: float = 0.0,
        q_prior_weight: float = 0.0,
        use_value_head: bool = False,
        leaf_value_mix: float = 0.0,
        prior_mode: str = "factorized",
        belief_search_weight: float = 0.0,
        belief_search_cap: float = 8.0,
    ):
        self.model = model.eval()
        self.adapt = adapter()
        self.rollouts = int(rollouts)
        self.c_puct = float(c_puct)
        self.expand_top_k = int(expand_top_k)
        self.training = bool(training)
        self.root_dirichlet_eps = float(root_dirichlet_eps)
        self.root_dirichlet_alpha = float(root_dirichlet_alpha)
        self.q_scale = float(max(1e-6, q_scale))
        self.use_q_head = bool(use_q_head)
        self.q_utility_weight = float(q_utility_weight)
        self.q_prior_weight = float(q_prior_weight)
        self.use_value_head = bool(use_value_head)
        self.leaf_value_mix = float(np.clip(leaf_value_mix, 0.0, 1.0))
        self.prior_mode = str(prior_mode)
        self.prior_uniform_mix = float(env_cfg.get("mcts_prior_uniform_mix", 0.0))
        self.belief_search_weight = float(belief_search_weight)
        self.belief_search_cap = float(max(0.0, belief_search_cap))
        self.poisson_rate_per_second = float(env_cfg.get("poisson_rate_per_second", 0.0))
        self.track_loss_penalty = float(env_cfg["track_loss_penalty"])
        delay_cfg = planner_delay_cfg("repaired_stress")
        self.pure = MCTSPlanner(
            max_trackers=MAXT,
            num_rollouts=rollouts,
            exploration_constant=c_puct,
            **delay_cfg,
            enable_search_refresh_tracked=bool(env_cfg["enable_search_refresh_tracked"]),
            search_refresh_gain=float(env_cfg["search_refresh_gain"]),
            search_action_reward=float(env_cfg["search_action_reward"]),
            track_update_reward=float(env_cfg["track_update_reward"]),
            track_loss_penalty=float(env_cfg["track_loss_penalty"]),
            track_urgency_bonus_weight=float(env_cfg["track_urgency_bonus_weight"]),
            sector_staleness_weight=float(env_cfg["sector_staleness_weight"]),
            searched_sector_reward_weight=float(env_cfg.get("searched_sector_reward_weight", 0.0)),
            search_frame_overdue_weight=float(env_cfg.get("search_frame_overdue_weight", 0.0)),
            search_frame_desired_ms=float(env_cfg.get("search_frame_desired_ms", 3000.0)),
            search_frame_deadline_ms=float(env_cfg.get("search_frame_deadline_ms", 4500.0)),
            search_frame_drop_penalty=float(env_cfg.get("search_frame_drop_penalty", 0.0)),
            enable_track_beam_scan=bool(env_cfg["enable_track_beam_scan"]),
            revisit_time_scale=float(env_cfg["revisit_time_scale"]),
            search_delay_mode=int(env_cfg.get("planner_search_delay_mode", env_cfg["search_delay_mode"])),
            search_debt_penalty_weight=float(env_cfg.get("planner_search_debt_penalty_weight", env_cfg["search_debt_penalty_weight"])),
            search_debt_tau_ms=float(env_cfg.get("planner_search_debt_tau_ms", env_cfg["search_debt_tau_ms"])),
            search_delay_penalty_cap=float(env_cfg.get("planner_search_delay_penalty_cap", env_cfg["search_delay_penalty_cap"])),
            penalize_hidden_targets=bool(env_cfg.get("penalize_hidden_targets", False)),
            rollout_candidate_cap=max(8, expand_top_k),
            simulation_window_ms=simulation_window_ms,
        )
        self.pure.rollout_policy = str(env_cfg.get("mcts_rollout_policy", "greedy"))
        self.pure.rollout_search_period_ms = float(env_cfg.get("mcts_rollout_search_period_ms", 120.0))

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _root_from_obs(self, obs) -> Node:
        root = Node(
            t_desired=obs["t_desired"],
            t_deadline=obs["t_deadline"],
            t_dwell=obs["t_dwell"],
            priority=obs["priority"],
            active_mask=obs["active_mask"],
            grid=obs.get("grid", None),
            az_bin=obs.get("az_bin", None),
            el_bin=obs.get("el_bin", None),
            tracked_mask=(obs["active_mask"] & (obs["t_deadline"] > 0)),
            search_debt_ms=float(obs.get("search_debt_ms", 0.0)),
        )
        root.refresh_t_desired, root.refresh_t_deadline = Node._infer_refresh_timers(
            root.t_desired, root.t_deadline, root.priority, self.pure.revisit_time_scale
        )
        return root

    def _node_to_obs(self, node: Node) -> Dict:
        return {
            "t_desired": node.t_desired,
            "t_deadline": node.t_deadline,
            "t_dwell": node.t_dwell,
            "priority": node.priority,
            "active_mask": node.active_mask,
            "tracked_mask": node.tracked_mask,
            "grid": node.grid,
            "az_bin": node.az_bin,
            "el_bin": node.el_bin,
            "search_debt_ms": node.search_debt_ms,
        }

    def _path_stats(self, node: Node) -> Tuple[float, int, int, int, set]:
        actions = []
        cur = node
        while cur is not None and cur.parent is not None:
            actions.append(int(cur.action))
            cur = cur.parent
        actions.reverse()
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        selected = set()
        for a in actions:
            last = a
            if a == 0:
                elapsed += SEARCH_DWELL_MS
                search_count += 1
            else:
                idx = a - 1
                if 0 <= idx < len(node.t_dwell):
                    elapsed += max(1.0, float(node.t_dwell[idx]))
                selected.add(a)
                track_count += 1
        return elapsed, search_count, track_count, last, selected

    def _features_for_node(self, node: Node, budget_ms: float = 200.0):
        obs = self._node_to_obs(node)
        elapsed, search_count, track_count, last, selected = self._path_stats(node)
        x = tokenize(self.adapt, obs, selected=selected, search_count=search_count)
        slot = slot_features(obs, elapsed, search_count, track_count, last, budget_ms)
        return obs, x, slot

    def _priors_for_node(self, node: Node, add_root_noise: bool = False) -> np.ndarray:
        _, x, slot = self._features_for_node(node)
        with torch.inference_mode():
            tl, tr, value, type_q, track_q = self.model(
                torch.from_numpy(x).float().unsqueeze(0).to(self.device),
                torch.from_numpy(slot).float().unsqueeze(0).to(self.device),
            )
        priors = action_priors_from_logits(tl[0], tr[0], self.prior_mode)
        qvalues = np.zeros_like(priors)
        tq = type_q[0].detach().cpu().numpy()
        trq = track_q[0].detach().cpu().numpy()
        qvalues[0] = float(tq[1] * self.q_scale)
        qvalues[1:] = float(tq[0] * self.q_scale) + trq[1:] * self.q_scale
        node.nn_qvalues = qvalues
        node.nn_value = float(value[0].detach().cpu()) * self.q_scale
        valid = node.get_valid_actions()
        mask = np.zeros_like(priors)
        mask[np.asarray(valid, dtype=np.int64)] = 1.0
        priors *= mask
        if add_root_noise and len(valid) > 1:
            noise = np.random.dirichlet([self.root_dirichlet_alpha] * len(valid)).astype(np.float32)
            noisy = np.zeros_like(priors)
            noisy[np.asarray(valid, dtype=np.int64)] = noise
            priors = (1.0 - self.root_dirichlet_eps) * priors + self.root_dirichlet_eps * noisy
        s = float(np.sum(priors))
        if s <= 0:
            priors[np.asarray(valid, dtype=np.int64)] = 1.0 / max(1, len(valid))
        else:
            priors /= s
        if self.prior_uniform_mix > 0.0 and len(valid) > 0:
            valid_arr = np.asarray(valid, dtype=np.int64)
            uniform = np.zeros_like(priors)
            uniform[valid_arr] = 1.0 / float(len(valid_arr))
            mix = float(np.clip(self.prior_uniform_mix, 0.0, 1.0))
            priors = (1.0 - mix) * priors + mix * uniform
        if self.use_q_head and self.q_prior_weight != 0.0:
            # In low-rollout MCTS, many children remain unvisited, so adding Q
            # only inside the visited-child PUCT term has almost no effect at
            # the root. Shape the policy prior with normalized predicted Q so
            # action values can affect expansion/first-visit ordering without
            # depending on reward-unit calibration.
            valid_arr = np.asarray(valid, dtype=np.int64)
            if len(valid_arr) > 1:
                qv = qvalues[valid_arr].astype(np.float32)
                q_std = float(np.std(qv))
                if q_std > 1e-6:
                    q_norm = (qv - float(np.mean(qv))) / q_std
                    shaped = priors[valid_arr] * np.exp(
                        np.clip(self.q_prior_weight * q_norm, -4.0, 4.0)
                    )
                    shaped_sum = float(np.sum(shaped))
                    if shaped_sum > 0.0:
                        priors = np.zeros_like(priors)
                        priors[valid_arr] = shaped / shaped_sum
        node.nn_priors = priors
        return priors

    def _select_child(self, node: Node) -> Node:
        unvisited = [c for c in node.children if c.visits == 0]
        if unvisited:
            # With 100+ possible target actions and 8-16 rollouts, random
            # first visits are not a useful AlphaZero target.  Use the exact
            # transition reward as the first-visit ordering signal, then let
            # backed-up rollout value take over after a child has visits.
            best = max(float(c.edge_reward) + self.c_puct * float(c.prior_prob) for c in unvisited)
            top = [
                c for c in unvisited
                if float(c.edge_reward) + self.c_puct * float(c.prior_prob) == best
            ]
            return top[np.random.randint(len(top))]
        best_score = -np.inf
        best_child = node.children[0]
        for child in node.children:
            q = float(child.edge_reward + child.total_reward / max(1, child.visits))
            u = self.c_puct * float(child.prior_prob) * math.sqrt(node.visits + 1.0) / (1.0 + child.visits)
            score = q + u
            if self.use_q_head and self.q_utility_weight != 0.0:
                score += self.q_utility_weight * float(getattr(node, "nn_qvalues", np.zeros((MAXT + 1,), dtype=np.float32))[child.action])
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _after_expand(self, node: Node):
        """Add belief-state information value for search.

        The observation contains tracked targets, but the environment can also
        penalize hidden targets whose deadlines expire before rediscovery.  A
        tree built only on tracked targets under-values search.  This term is a
        compact belief reward: expected hidden target count grows with search
        debt and arrival intensity; search clears that accumulated risk.
        """
        if self.belief_search_weight <= 0.0 or not node.children:
            return None
        debt_s = max(0.0, float(getattr(node, "search_debt_ms", 0.0))) / 1000.0
        if debt_s <= 0.0:
            return None
        active = np.asarray(node.active_mask).astype(bool)
        pr = np.asarray(node.priority, dtype=np.float32)
        if np.any(active):
            priority_scale = float(np.mean(1.0 + 2.0 * pr[active]))
        else:
            priority_scale = 2.0
        expected_hidden_loss = (
            self.poisson_rate_per_second
            * debt_s
            * self.track_loss_penalty
            * priority_scale
        )
        bonus = self.belief_search_weight * expected_hidden_loss
        if self.belief_search_cap > 0.0:
            bonus = min(bonus, self.belief_search_cap)
        for child in node.children:
            if int(child.action) == 0:
                child.edge_reward += float(bonus)
        return None

    def _run_search(self, obs, budget_ms=200.0) -> Node:
        root = self._root_from_obs(obs)
        root_priors = self._priors_for_node(root, add_root_noise=self.training)
        self.pure._expand(root, priors=root_priors, top_k=self.expand_top_k)
        self._after_expand(root)
        for _ in range(self.rollouts):
            node = root
            while node.expanded and node.children and not node.is_terminal():
                node = self._select_child(node)
            if not node.is_terminal() and not node.expanded:
                priors = self._priors_for_node(node)
                self.pure._expand(node, priors=priors, top_k=self.expand_top_k)
                self._after_expand(node)
            if node.is_terminal():
                reward = 0.0
            else:
                rollout_reward = self.pure._simulate(node)
                leaf_value = float(getattr(node, "nn_value", 0.0))
                reward = (1.0 - self.leaf_value_mix) * rollout_reward + self.leaf_value_mix * leaf_value if self.use_value_head else rollout_reward
            self.pure._backprop(node, reward)
        return root

    def _target_for_node(self, node: Node, budget_ms: float = 200.0) -> SearchTarget:
        """Extract AlphaZero-style policy/Q targets for a searched node.

        The model is used autoregressively inside a 200 ms scheduling window, so
        training only on the root state is insufficient.  Each selected internal
        node is also a real decision state with different elapsed/search-count
        context and must contribute a policy target.
        """
        total_visits = max(1, sum(c.visits for c in node.children))
        pi = np.zeros((MAXT + 1,), dtype=np.float32)
        q = np.zeros((MAXT + 1,), dtype=np.float32)
        q_mask = np.zeros((MAXT + 1,), dtype=np.float32)
        for child in node.children:
            pi[child.action] = child.visits / total_visits
            if child.visits > 0:
                q[child.action] = float(child.edge_reward + child.total_reward / max(1, child.visits))
                q_mask[child.action] = 1.0
        _, x, slot = self._features_for_node(node, float(budget_ms))
        elapsed, search_count, track_count, _, _ = self._path_stats(node)
        return SearchTarget(
            x=x,
            slot=slot,
            pi=pi,
            q=q,
            q_mask=q_mask,
            search_count=search_count,
            track_count=track_count,
        )

    def plan_with_target(self, obs, budget_ms=200):
        plan, targets = self.plan_with_targets(obs, budget_ms)
        return plan, targets[0]

    def plan_with_targets(self, obs, budget_ms=200):
        root = self._run_search(obs, budget_ms)
        plan = []
        targets: List[SearchTarget] = []
        node = root
        elapsed = 0.0
        for _ in range(64):
            if elapsed >= float(budget_ms):
                break
            if not node.children and not node.is_terminal():
                priors = self._priors_for_node(node)
                self.pure._expand(node, priors=priors, top_k=self.expand_top_k)
                self._after_expand(node)
            if not node.children:
                break
            targets.append(self._target_for_node(node, float(budget_ms)))
            visited = [c for c in node.children if c.visits > 0]
            if visited:
                child = max(visited, key=lambda c: c.edge_reward + c.total_reward / max(1, c.visits))
            else:
                # Once the backed-up path ends, continue with the same
                # reward-aware selection rule used during search.  Falling back
                # to raw neural prior here creates a self-reinforcing all-search
                # tail when the policy is still random or poorly calibrated.
                child = self._select_child(node)
            plan.append(int(child.action))
            if child.action == 0:
                elapsed += SEARCH_DWELL_MS
            else:
                idx = child.action - 1
                elapsed += max(1.0, float(node.t_dwell[idx]) if 0 <= idx < len(node.t_dwell) else SEARCH_DWELL_MS)
            node = child
        if not targets:
            targets.append(self._target_for_node(root, float(budget_ms)))
        return (plan if plan else [0]), targets

    def plan(self, obs, budget_ms=200):
        plan, _ = self.plan_with_target(obs, budget_ms)
        return plan


class CalibratedMutualRadarDirectPlanner(MutualRadarDirectPlanner):
    """Direct factorized planner with a macro-state search calibration.

    The neural heads still rank search versus track and select the target.  This
    wrapper only shifts the search threshold as a smooth function of observable
    load, which is the calibration failure seen when one global threshold is
    used across easy and saturated scenarios.
    """

    def __init__(
        self,
        model: MutualRadarNet,
        base_threshold: float = -5.0,
        active_coef: float = 0.0,
        rate_coef: float = 0.0,
        active_ref: float = 60.0,
        **kwargs,
    ):
        super().__init__(model, threshold=float(base_threshold), **kwargs)
        self.base_threshold = float(base_threshold)
        self.active_coef = float(active_coef)
        self.rate_coef = float(rate_coef)
        self.active_ref = float(active_ref)

    def _calibrated_threshold(self, obs) -> float:
        active = float(np.sum(np.asarray(obs.get("active_mask", []), dtype=bool)))
        rate = float(obs.get("arrival_rate", obs.get("poisson_rate_per_second", 0.0)))
        return float(self.base_threshold + self.active_coef * ((active - self.active_ref) / 40.0) + self.rate_coef * rate)

    def plan(self, obs, budget_ms=200):
        old_threshold = self.threshold
        self.threshold = self._calibrated_threshold(obs)
        try:
            return super().plan(obs, budget_ms=budget_ms)
        finally:
            self.threshold = old_threshold


class LoadAdaptiveDirectPlanner:
    """Switch between two neural direct decoders by observable load.

    Both arms use the same learned factorized heads.  This is a small
    deployment policy over decoders, not a heuristic action fallback: low-load
    cases prefer branch-probability calibration, while saturated cases prefer
    flat logit-margin calibration.
    """

    def __init__(
        self,
        model: MutualRadarNet,
        low_load_cutoff: float = 40.0,
        low_threshold: float = 0.4,
        high_threshold: float = -5.0,
        low_flat_rate_min: float = 2.0,
        low_flat_rate_max: float = 6.0,
        low_high_rate_threshold: float = 0.0,
        arrival_rate: float = 0.0,
        use_arrival_feature: bool = False,
        **kwargs,
    ):
        self.low_load_cutoff = float(low_load_cutoff)
        self.low_flat_rate_min = float(low_flat_rate_min)
        self.low_flat_rate_max = float(low_flat_rate_max)
        self.arrival_rate = float(arrival_rate)
        self.use_arrival_feature = bool(use_arrival_feature)
        self.low_high_rate_threshold = float(low_high_rate_threshold)
        self.low = MutualRadarDirectPlanner(
            model,
            direct_mode="branch",
            threshold=float(low_threshold),
            **kwargs,
        )
        self.high = MutualRadarDirectPlanner(
            model,
            direct_mode="flat",
            threshold=float(high_threshold),
            **kwargs,
        )
        self.low_high_rate = MutualRadarDirectPlanner(
            model,
            direct_mode="flat",
            threshold=float(low_high_rate_threshold),
            **kwargs,
        )

    def _planner(self, obs):
        active = float(np.sum(np.asarray(obs.get("active_mask", []), dtype=bool)))
        rate = float(obs.get("arrival_rate", obs.get("poisson_rate_per_second", self.arrival_rate)))
        if active <= self.low_load_cutoff:
            if self.low_flat_rate_min <= rate <= self.low_flat_rate_max:
                return self.high
            if rate > self.low_flat_rate_max:
                return self.low_high_rate
            return self.low
        return self.high

    def warmup(self, obs, budget_ms=200):
        obs2 = _with_arrival_context(obs, self.arrival_rate, self.use_arrival_feature)
        self.low.warmup(obs2, budget_ms=budget_ms)
        self.high.warmup(obs2, budget_ms=budget_ms)
        self.low_high_rate.warmup(obs2, budget_ms=budget_ms)

    def plan(self, obs, budget_ms=200):
        obs2 = _with_arrival_context(obs, self.arrival_rate, self.use_arrival_feature)
        return self._planner(obs2).plan(obs2, budget_ms=budget_ms)


class NeuralDecoderPortfolioPlanner:
    """Portfolio over neural decoders with a pluggable selector.

    This is the research-clean replacement for ad hoc manual decoder selection:
    each arm is still a deployment of the same learned heads, and the selector
    can be trained by policy iteration on completed-episode outcomes.
    """

    def __init__(
        self,
        model: MutualRadarNet,
        arms: Optional[Dict[str, MutualRadarDirectPlanner]] = None,
        selector=None,
        arrival_rate: float = 0.0,
        use_arrival_feature: bool = False,
    ):
        self.model = model
        self.arrival_rate = float(arrival_rate)
        self.use_arrival_feature = bool(use_arrival_feature)
        self.arms = arms or {
            "flat_m5": MutualRadarDirectPlanner(
                model,
                direct_mode="flat",
                threshold=-5.0,
                alpha=0.0,
                beta=0.0,
                allow_retrack=False,
                cache_encoder=True,
                sensor_action_mode="explicit_head",
            ),
            "flat_0": MutualRadarDirectPlanner(
                model,
                direct_mode="flat",
                threshold=0.0,
                alpha=0.0,
                beta=0.0,
                allow_retrack=False,
                cache_encoder=True,
                sensor_action_mode="explicit_head",
            ),
            "branch_04": MutualRadarDirectPlanner(
                model,
                direct_mode="branch",
                threshold=0.4,
                alpha=0.0,
                beta=0.0,
                allow_retrack=False,
                cache_encoder=True,
                sensor_action_mode="explicit_head",
            ),
            "prob_0": MutualRadarDirectPlanner(
                model,
                direct_mode="prob",
                threshold=0.0,
                alpha=0.0,
                beta=0.0,
                allow_retrack=False,
                cache_encoder=True,
                sensor_action_mode="explicit_head",
            ),
        }
        self.selector = selector

    @staticmethod
    def macro_features_from_obs(obs, arrival_rate: float = 0.0) -> Dict[str, float]:
        active_mask = np.asarray(obs.get("active_mask", []), dtype=bool)
        active = active_mask.astype(bool)
        deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
        desired = np.asarray(obs.get("t_desired", []), dtype=np.float32)
        active_deadline = deadline[active[: len(deadline)]] if len(deadline) else np.zeros(0, dtype=np.float32)
        active_desired = desired[active[: len(desired)]] if len(desired) else np.zeros(0, dtype=np.float32)
        return {
            "active": float(np.sum(active)),
            "arrival_rate": float(obs.get("arrival_rate", obs.get("poisson_rate_per_second", arrival_rate))),
            "deadline_min": float(np.min(active_deadline)) if active_deadline.size else 0.0,
            "deadline_mean": float(np.mean(active_deadline)) if active_deadline.size else 0.0,
            "desired_min": float(np.min(active_desired)) if active_desired.size else 0.0,
            "desired_mean": float(np.mean(active_desired)) if active_desired.size else 0.0,
        }

    def choose_arm(self, obs) -> str:
        obs2 = _with_arrival_context(obs, self.arrival_rate, self.use_arrival_feature)
        f = self.macro_features_from_obs(obs2, self.arrival_rate)
        if self.selector is not None:
            name = str(self.selector(f))
            if name in self.arms:
                return name
        # Conservative default learned from the current validation/holdout
        # evidence: flat_0 is robust for low-load mid/high arrival-rate cases,
        # branch_04 is best for many zero-arrival low-load cases, and flat_m5
        # dominates saturated cases.
        if f["active"] > 40.0:
            return "flat_m5"
        if f["arrival_rate"] <= 1.0:
            return "branch_04"
        return "flat_0"

    def warmup(self, obs, budget_ms=200):
        obs = _with_arrival_context(obs, self.arrival_rate, self.use_arrival_feature)
        for arm in self.arms.values():
            arm.warmup(obs, budget_ms=budget_ms)

    def plan(self, obs, budget_ms=200):
        obs2 = _with_arrival_context(obs, self.arrival_rate, self.use_arrival_feature)
        return self.arms[self.choose_arm(obs2)].plan(obs2, budget_ms=budget_ms)


class LowLoadMCTSVerifierPlanner:
    """Factorized policy with a narrow exact-search verifier.

    The learned policy handles ordinary states.  For low-load, zero-arrival
    states where the learned root margin says search and track are badly
    calibrated, a small reference MCTS is used as a verifier.  This keeps MCTS
    in the deployed improvement path without making every decision search-heavy.
    """

    def __init__(
        self,
        model: MutualRadarNet,
        env_cfg: Dict[str, float],
        low_threshold: float = 0.5,
        high_threshold: float = -5.0,
        low_load_cutoff: float = 40.0,
        arrival_rate: float = 0.0,
        uncertainty_margin: float = -5.0,
        verifier_rollouts: int = 1,
        verifier_c_puct: float = 1.25,
        verifier_horizon_ms: float = 200.0,
        max_verifier_calls: int = 1,
        use_arrival_feature: bool = True,
    ):
        self.model = model.eval()
        self.env_cfg = dict(env_cfg)
        self.low_load_cutoff = float(low_load_cutoff)
        self.arrival_rate = float(arrival_rate)
        self.uncertainty_margin = float(uncertainty_margin)
        self.max_verifier_calls = int(max(0, max_verifier_calls))
        self.verifier_calls = 0
        self.use_arrival_feature = bool(use_arrival_feature)
        self.low = MutualRadarDirectPlanner(
            model,
            direct_mode="branch",
            threshold=float(low_threshold),
            alpha=0.0,
            beta=0.0,
            allow_retrack=False,
            cache_encoder=True,
            sensor_action_mode="explicit_head",
        )
        self.high = MutualRadarDirectPlanner(
            model,
            direct_mode="flat",
            threshold=float(high_threshold),
            alpha=0.0,
            beta=0.0,
            allow_retrack=False,
            cache_encoder=True,
            sensor_action_mode="explicit_head",
        )
        self.verifier = make_reference_planner(
            MAXT,
            int(verifier_rollouts),
            float(verifier_c_puct),
            self.env_cfg,
            "repaired_stress",
            simulation_window_ms=float(verifier_horizon_ms),
        )
        self.verifier.rollout_policy = "edf"
        self.verifier.action_selection = "q"
        self.verifier.rollout_search_period_ms = 160.0

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _root_flat_margin(self, obs) -> float:
        s = slot_features(obs, 0.0, 0, 0, -1, 200.0)
        x = tokenize(adapter(), obs, selected=set(), search_count=0)
        with torch.inference_mode():
            tl, tr, _, _, _ = self.model(
                torch.from_numpy(x).float().unsqueeze(0).to(self.device),
                torch.from_numpy(s).float().unsqueeze(0).to(self.device),
            )
        track = tr[0].detach().cpu().numpy().astype(np.float32)
        finite = np.isfinite(track) & (track > -1e8)
        finite[0] = False if finite.size else False
        best_track = float(np.max(track[finite])) if np.any(finite) else -np.inf
        return float(tl[0].detach().cpu()) - best_track

    def plan(self, obs, budget_ms=200):
        obs2 = _with_arrival_context(obs, self.arrival_rate, self.use_arrival_feature)
        active = float(np.sum(np.asarray(obs2.get("active_mask", []), dtype=bool)))
        rate = float(obs2.get("arrival_rate", obs2.get("poisson_rate_per_second", self.arrival_rate)))
        if (
            self.verifier_calls < self.max_verifier_calls
            and active <= self.low_load_cutoff
            and rate <= 1.0
            and self._root_flat_margin(obs2) <= self.uncertainty_margin
        ):
            self.verifier_calls += 1
            return self.verifier.plan(obs, budget_ms=budget_ms)
        if active <= self.low_load_cutoff:
            return self.low.plan(obs2, budget_ms=budget_ms)
        return self.high.plan(obs2, budget_ms=budget_ms)


class ConfidenceGatedPortfolioPlanner:
    """Confidence-gated portfolio of two learned factorized PV policies.

    The base policy is the broad accepted scheduler.  The repair policy is a
    separately trained low-load/rate-zero calibration head.  The gate uses only
    the base model's root search-vs-track margin and arrival context, so it is
    a learned uncertainty test rather than a heuristic action rule.
    """

    def __init__(
        self,
        base_model: MutualRadarNet,
        repair_model: MutualRadarNet,
        arrival_rate: float = 0.0,
        low_load_cutoff: float = 40.0,
        gate_margin: float = -8.0,
        low_threshold: float = 0.5,
        high_threshold: float = -5.0,
        use_arrival_feature: bool = True,
    ):
        self.base_model = base_model.eval()
        self.repair_model = repair_model.eval()
        self.arrival_rate = float(arrival_rate)
        self.low_load_cutoff = float(low_load_cutoff)
        self.gate_margin = float(gate_margin)
        self.use_arrival_feature = bool(use_arrival_feature)
        self.selected_policy: Optional[str] = None
        self.used_repair = 0
        self.base_low = MutualRadarDirectPlanner(
            self.base_model,
            direct_mode="branch",
            threshold=float(low_threshold),
            alpha=0.0,
            beta=0.0,
            allow_retrack=False,
            cache_encoder=True,
            sensor_action_mode="explicit_head",
        )
        self.base_high = MutualRadarDirectPlanner(
            self.base_model,
            direct_mode="flat",
            threshold=float(high_threshold),
            alpha=0.0,
            beta=0.0,
            allow_retrack=False,
            cache_encoder=True,
            sensor_action_mode="explicit_head",
        )
        self.repair_low = MutualRadarDirectPlanner(
            self.repair_model,
            direct_mode="branch",
            threshold=float(low_threshold),
            alpha=0.0,
            beta=0.0,
            allow_retrack=False,
            cache_encoder=True,
            sensor_action_mode="explicit_head",
        )

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    def _base_root_flat_margin(self, obs) -> float:
        s = slot_features(obs, 0.0, 0, 0, -1, 200.0)
        x = tokenize(adapter(), obs, selected=set(), search_count=0)
        with torch.inference_mode():
            tl, tr, _, _, _ = self.base_model(
                torch.from_numpy(x).float().unsqueeze(0).to(self.device),
                torch.from_numpy(s).float().unsqueeze(0).to(self.device),
            )
        track = tr[0].detach().cpu().numpy().astype(np.float32)
        finite = np.isfinite(track) & (track > -1e8)
        finite[0] = False if finite.size else False
        best_track = float(np.max(track[finite])) if np.any(finite) else -np.inf
        return float(tl[0].detach().cpu()) - best_track

    def _select_policy_once(self, obs) -> str:
        active = float(np.sum(np.asarray(obs.get("active_mask", []), dtype=bool)))
        rate = float(obs.get("arrival_rate", obs.get("poisson_rate_per_second", self.arrival_rate)))
        if active <= self.low_load_cutoff and rate <= 1.0 and self._base_root_flat_margin(obs) <= self.gate_margin:
            self.used_repair = 1
            return "repair"
        return "base"

    def warmup(self, obs, budget_ms=200):
        obs2 = _with_arrival_context(obs, self.arrival_rate, self.use_arrival_feature)
        self.base_low.warmup(obs2, budget_ms=budget_ms)
        self.base_high.warmup(obs2, budget_ms=budget_ms)
        self.repair_low.warmup(obs2, budget_ms=budget_ms)

    def plan(self, obs, budget_ms=200):
        obs2 = _with_arrival_context(obs, self.arrival_rate, self.use_arrival_feature)
        active = float(np.sum(np.asarray(obs2.get("active_mask", []), dtype=bool)))
        if self.selected_policy is None:
            self.selected_policy = self._select_policy_once(obs2)
        if active > self.low_load_cutoff:
            return self.base_high.plan(obs2, budget_ms=budget_ms)
        if self.selected_policy == "repair":
            return self.repair_low.plan(obs2, budget_ms=budget_ms)
        return self.base_low.plan(obs2, budget_ms=budget_ms)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.items: List[SearchTarget] = []

    def extend(self, rows: Iterable[SearchTarget]):
        for row in rows:
            self.items.append(row)
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity :]

    def sample(self, n: int) -> List[SearchTarget]:
        return random.sample(self.items, min(int(n), len(self.items)))

    def __len__(self):
        return len(self.items)


def make_train_eval_env(rate: float, args) -> Dict[str, float]:
    env = make_env(rate)
    if getattr(args, "track_loss_penalty", None) is not None and args.track_loss_penalty >= 0:
        env["track_loss_penalty"] = float(args.track_loss_penalty)
    if getattr(args, "track_urgency_bonus_weight", None) is not None and args.track_urgency_bonus_weight >= 0:
        env["track_urgency_bonus_weight"] = float(args.track_urgency_bonus_weight)
    if getattr(args, "search_debt_penalty_weight", None) is not None and args.search_debt_penalty_weight >= 0:
        env["search_debt_penalty_weight"] = float(args.search_debt_penalty_weight)
    return env


def collect_selfplay(model: MutualRadarNet, args, env_cfg) -> Tuple[List[SearchTarget], float]:
    rows: List[SearchTarget] = []
    rewards = []
    init_choices = [int(x) for x in str(args.train_initials).split(",") if x]
    rate_choices = [float(x) for x in str(args.train_rates).split(",") if x]
    for ep in range(args.episodes_per_iter):
        init = random.choice(init_choices)
        rate = random.choice(rate_choices)
        seed = int(args.seed + 1000 * args.iteration + ep)
        seedall(seed)
        env = make_train_eval_env(rate, args)
        planner_cfg = dict(env)
        if getattr(args, "planner_search_debt_weight", None) is not None and args.planner_search_debt_weight >= 0:
            planner_cfg["planner_search_debt_penalty_weight"] = float(args.planner_search_debt_weight)
        if getattr(args, "planner_search_action_reward", None) is not None:
            planner_cfg["search_action_reward"] = float(args.planner_search_action_reward)
        planner = MutualRadarMCTSPlanner(
            model,
            planner_cfg,
            rollouts=args.rollouts,
            c_puct=args.c_puct,
            expand_top_k=args.expand_top_k,
            training=True,
            prior_mode=args.prior_mode,
            belief_search_weight=args.belief_search_weight,
            belief_search_cap=args.belief_search_cap,
        )
        eng = build_env(planner, init, MAXT, seed, 200, env)
        eng.reset(seed=seed)
        debt = 0.0
        traj: List[SearchTarget] = []
        for w in range(args.windows_per_episode):
            if eng.term_buf[0]:
                break
            from final_radar_campaign import get_obs

            obs = get_obs(eng, debt)
            plan, targets = planner.plan_with_targets(obs, 200)
            reward, spent, debt, executed, _, _ = execute_plan_until_budget(eng, plan, 200.0, debt, "MutualRadar_selfplay", seed, w)
            per_decision_reward = float(reward) / max(1, len(targets))
            for target in targets:
                target.reward = per_decision_reward
                traj.append(target)
            rewards.append(float(reward))
            if executed <= 0 or spent <= 0:
                break
        eng.close()
        G = 0.0
        for target in reversed(traj):
            G = float(target.reward) + args.gamma * G
            target.ret = G
        rows.extend(traj)
    return rows, float(np.mean(rewards)) if rewards else 0.0


def train_step(model: MutualRadarNet, opt, replay: ReplayBuffer, batch_size: int, q_scale: float):
    if len(replay) < max(4, batch_size // 4):
        return {}
    batch = replay.sample(batch_size)
    x = torch.from_numpy(np.stack([b.x for b in batch]).astype(np.float32)).to(DEVICE)
    slot = torch.from_numpy(np.stack([b.slot for b in batch]).astype(np.float32)).to(DEVICE)
    pi = torch.from_numpy(np.stack([b.pi for b in batch]).astype(np.float32)).to(DEVICE)
    q = torch.from_numpy(np.stack([b.q for b in batch]).astype(np.float32) / q_scale).to(DEVICE)
    q_mask = torch.from_numpy(np.stack([b.q_mask for b in batch]).astype(np.float32)).to(DEVICE)
    ret = torch.tensor([b.ret / q_scale for b in batch], dtype=torch.float32, device=DEVICE)

    has_sensor_targets = any(getattr(b, "sensor_pi", None) is not None for b in batch)
    sensor_logits = sensor_q_pred = None
    if has_sensor_targets and hasattr(model, "forward_with_sensor"):
        type_logit, track_logits, value, type_q, track_q, sensor_logits, sensor_q_pred = model.forward_with_sensor(x, slot)
    else:
        type_logit, track_logits, value, type_q, track_q = model(x, slot)
    pi_search = pi[:, 0].clamp(0.0, 1.0)
    type_loss = F.binary_cross_entropy_with_logits(type_logit, pi_search)

    track_mass = pi[:, 1:].sum(dim=1)
    has_track = track_mass > 1e-6
    if bool(torch.any(has_track)):
        target_track = pi[has_track].clone()
        target_track[:, 0] = 0.0
        finite_track = torch.isfinite(track_logits[has_track]) & (track_logits[has_track] > -1e8)
        target_track = target_track * finite_track.float()
        has_finite_target = target_track.sum(dim=1) > 1e-6
        target_track = target_track[has_finite_target]
        track_logits_for_loss = track_logits[has_track][has_finite_target]
    else:
        target_track = torch.zeros((0, pi.shape[1]), device=DEVICE)
        track_logits_for_loss = torch.zeros((0, track_logits.shape[1]), device=DEVICE)
    if target_track.shape[0] > 0:
        target_track = target_track / target_track.sum(dim=1, keepdim=True).clamp_min(1e-6)
        rank_loss = F.kl_div(F.log_softmax(track_logits_for_loss, dim=1), target_track, reduction="batchmean")
    else:
        rank_loss = torch.zeros((), device=DEVICE)

    v_loss = F.smooth_l1_loss(value, ret)
    search_q_target = q[:, 0]
    # The inference score is factorized as
    #   Q(track_i) = Q_type(track) + Q_track(i).
    # Therefore the target-specific head must learn a residual, not the full
    # action value.  Training Q_track(i) directly to Q(s, track_i) double-counts
    # the branch value at inference and destabilizes search-vs-track calibration.
    track_branch_target = (q[:, 1:] * pi[:, 1:]).sum(dim=1) / pi[:, 1:].sum(dim=1).clamp_min(1e-6)
    type_q_target = torch.stack([track_branch_target, search_q_target], dim=1)
    type_q_mask = torch.stack([(q_mask[:, 1:] > 0.5).any(dim=1).float(), q_mask[:, 0]], dim=1)
    type_q_err = F.smooth_l1_loss(type_q, type_q_target, reduction="none")
    type_q_loss = (type_q_err * type_q_mask).sum() / type_q_mask.sum().clamp_min(1.0)
    q_valid = q_mask > 0.5
    q_valid[:, 0] = False
    track_q_residual_target = q - track_branch_target[:, None]
    track_q_loss = F.smooth_l1_loss(track_q[q_valid], track_q_residual_target[q_valid]) if bool(torch.any(q_valid)) else torch.zeros((), device=DEVICE)

    loss = type_loss + rank_loss + 0.5 * v_loss + 0.25 * type_q_loss + 0.5 * track_q_loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return {
        "loss": float(loss.detach().cpu()),
        "type_loss": float(type_loss.detach().cpu()),
        "rank_loss": float(rank_loss.detach().cpu()),
        "v_loss": float(v_loss.detach().cpu()),
        "type_q_loss": float(type_q_loss.detach().cpu()),
        "track_q_loss": float(track_q_loss.detach().cpu()),
    }


def train_step_branch_balanced(
    model: MutualRadarNet,
    opt,
    replay: ReplayBuffer,
    batch_size: int,
    q_scale: float,
    policy_tau: float = 0.75,
):
    """Train factorized policy from MCTS Q at the branch level.

    Raw visit-count targets are poorly calibrated for this radar action space:
    one search action competes against many track actions.  The type head should
    learn the marginal branch mass search-vs-sum(track_i), while the target head
    learns the conditional ranking inside the track branch.
    """
    if len(replay) < max(4, batch_size // 4):
        return {}
    batch = replay.sample(batch_size)
    x = torch.from_numpy(np.stack([b.x for b in batch]).astype(np.float32)).to(DEVICE)
    slot = torch.from_numpy(np.stack([b.slot for b in batch]).astype(np.float32)).to(DEVICE)
    pi = torch.from_numpy(np.stack([b.pi for b in batch]).astype(np.float32)).to(DEVICE)
    q = torch.from_numpy(np.stack([b.q for b in batch]).astype(np.float32) / q_scale).to(DEVICE)
    q_mask = torch.from_numpy(np.stack([b.q_mask for b in batch]).astype(np.float32)).to(DEVICE)
    ret = torch.tensor([b.ret / q_scale for b in batch], dtype=torch.float32, device=DEVICE)

    has_sensor_targets = any(getattr(b, "sensor_pi", None) is not None for b in batch)
    sensor_logits = sensor_q_pred = None
    if has_sensor_targets and hasattr(model, "forward_with_sensor"):
        type_logit, track_logits, value, type_q, track_q, sensor_logits, sensor_q_pred = model.forward_with_sensor(x, slot)
    else:
        type_logit, track_logits, value, type_q, track_q = model(x, slot)
    tau = max(float(policy_tau), 1e-3)

    search_valid = q_mask[:, 0] > 0.5
    track_valid = q_mask[:, 1:] > 0.5
    has_track = track_valid.any(dim=1)
    q_track_scaled = (q[:, 1:] / tau).masked_fill(~track_valid, -1e9)
    track_branch_logmass = torch.logsumexp(q_track_scaled, dim=1)
    search_branch_logmass = q[:, 0] / tau
    q_branch_logits = torch.stack([track_branch_logmass, search_branch_logmass], dim=1)
    q_branch_target = torch.softmax(q_branch_logits, dim=1)[:, 1]
    fallback_search = pi[:, 0].clamp(0.0, 1.0)
    type_target = torch.where(search_valid & has_track, q_branch_target, fallback_search)
    type_loss = F.binary_cross_entropy_with_logits(type_logit, type_target)

    finite_track = torch.isfinite(track_logits) & (track_logits > -1e8)
    track_target = torch.zeros_like(track_logits)
    q_rank_mask = track_valid & finite_track[:, 1:]
    if bool(torch.any(q_rank_mask)):
        q_rank = q[:, 1:].masked_fill(~q_rank_mask, -1e9) / tau
        q_rank_target = torch.softmax(q_rank, dim=1)
        track_target[:, 1:] = q_rank_target
    pi_track_mass = pi[:, 1:].sum(dim=1)
    fallback_rows = (~q_rank_mask.any(dim=1)) & (pi_track_mass > 1e-6)
    if bool(torch.any(fallback_rows)):
        fb = pi[fallback_rows].clone()
        fb[:, 0] = 0.0
        fb = fb * finite_track[fallback_rows].float()
        fb = fb / fb.sum(dim=1, keepdim=True).clamp_min(1e-6)
        track_target[fallback_rows] = fb
    rank_rows = track_target.sum(dim=1) > 1e-6
    rank_loss = (
        F.kl_div(F.log_softmax(track_logits[rank_rows], dim=1), track_target[rank_rows], reduction="batchmean")
        if bool(torch.any(rank_rows))
        else torch.zeros((), device=DEVICE)
    )

    v_loss = F.smooth_l1_loss(value, ret)
    search_q_target = q[:, 0]
    track_branch_target = (q[:, 1:] * q_mask[:, 1:]).sum(dim=1) / q_mask[:, 1:].sum(dim=1).clamp_min(1e-6)
    type_q_target = torch.stack([track_branch_target, search_q_target], dim=1)
    type_q_mask = torch.stack([(q_mask[:, 1:] > 0.5).any(dim=1).float(), q_mask[:, 0]], dim=1)
    type_q_err = F.smooth_l1_loss(type_q, type_q_target, reduction="none")
    type_q_loss = (type_q_err * type_q_mask).sum() / type_q_mask.sum().clamp_min(1.0)
    q_valid = q_mask > 0.5
    q_valid[:, 0] = False
    track_q_residual_target = q - track_branch_target[:, None]
    track_q_loss = F.smooth_l1_loss(track_q[q_valid], track_q_residual_target[q_valid]) if bool(torch.any(q_valid)) else torch.zeros((), device=DEVICE)

    loss = type_loss + rank_loss + 0.5 * v_loss + 0.25 * type_q_loss + 0.5 * track_q_loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return {
        "loss": float(loss.detach().cpu()),
        "type_loss": float(type_loss.detach().cpu()),
        "rank_loss": float(rank_loss.detach().cpu()),
        "v_loss": float(v_loss.detach().cpu()),
        "type_q_loss": float(type_q_loss.detach().cpu()),
        "track_q_loss": float(track_q_loss.detach().cpu()),
    }


def train_step_branch_max(
    model: MutualRadarNet,
    opt,
    replay: ReplayBuffer,
    batch_size: int,
    q_scale: float,
    policy_tau: float = 0.75,
    type_loss_weight: float = 1.0,
    rank_loss_weight: float = 1.0,
    value_loss_weight: float = 0.5,
    type_q_loss_weight: float = 0.25,
    track_q_loss_weight: float = 0.5,
):
    """Factorized branch training without track-cardinality bias.

    The type head decides search-vs-track.  That branch decision should compare
    search against the best available track action, not the log-sum of every
    target action; otherwise simply having many targets biases the type head
    toward tracking.  The target head still learns the conditional ranking
    inside the track branch.
    """
    if len(replay) < max(4, batch_size // 4):
        return {}
    batch = replay.sample(batch_size)
    device = next(model.parameters()).device
    x = torch.from_numpy(np.stack([b.x for b in batch]).astype(np.float32)).to(device)
    slot = torch.from_numpy(np.stack([b.slot for b in batch]).astype(np.float32)).to(device)
    pi = torch.from_numpy(np.stack([b.pi for b in batch]).astype(np.float32)).to(device)
    q = torch.from_numpy(np.stack([b.q for b in batch]).astype(np.float32) / q_scale).to(device)
    q_mask = torch.from_numpy(np.stack([b.q_mask for b in batch]).astype(np.float32)).to(device)
    ret = torch.tensor([b.ret / q_scale for b in batch], dtype=torch.float32, device=device)

    has_sensor_targets = any(getattr(b, "sensor_pi", None) is not None for b in batch)
    sensor_logits = sensor_q_pred = None
    if has_sensor_targets and hasattr(model, "forward_with_sensor"):
        type_logit, track_logits, value, type_q, track_q, sensor_logits, sensor_q_pred = model.forward_with_sensor(x, slot)
    else:
        type_logit, track_logits, value, type_q, track_q = model(x, slot)
    tau = max(float(policy_tau), 1e-3)

    search_valid = q_mask[:, 0] > 0.5
    track_valid = q_mask[:, 1:] > 0.5
    has_track = track_valid.any(dim=1)
    q_track_masked = q[:, 1:].masked_fill(~track_valid, -1e9)
    best_track_q = torch.max(q_track_masked, dim=1).values
    search_q = q[:, 0]
    branch_logits = torch.stack([best_track_q / tau, search_q / tau], dim=1)
    q_branch_target = torch.softmax(branch_logits, dim=1)[:, 1]
    fallback_search = pi[:, 0].clamp(0.0, 1.0)
    type_target = torch.where(search_valid & has_track, q_branch_target, fallback_search)
    type_loss = F.binary_cross_entropy_with_logits(type_logit, type_target)

    finite_track = torch.isfinite(track_logits) & (track_logits > -1e8)
    q_rank_mask = track_valid & finite_track[:, 1:]
    track_target = torch.zeros_like(track_logits)
    if bool(torch.any(q_rank_mask)):
        q_rank = q[:, 1:].masked_fill(~q_rank_mask, -1e9) / tau
        q_rank_target = torch.softmax(q_rank, dim=1)
        track_target[:, 1:] = q_rank_target
    pi_track_mass = pi[:, 1:].sum(dim=1)
    fallback_rows = (~q_rank_mask.any(dim=1)) & (pi_track_mass > 1e-6)
    if bool(torch.any(fallback_rows)):
        fb = pi[fallback_rows].clone()
        fb[:, 0] = 0.0
        fb = fb * finite_track[fallback_rows].float()
        fb = fb / fb.sum(dim=1, keepdim=True).clamp_min(1e-6)
        track_target[fallback_rows] = fb
    rank_rows = track_target.sum(dim=1) > 1e-6
    rank_loss = (
        F.kl_div(F.log_softmax(track_logits[rank_rows], dim=1), track_target[rank_rows], reduction="batchmean")
        if bool(torch.any(rank_rows))
        else torch.zeros((), device=device)
    )

    v_loss = F.smooth_l1_loss(value, ret)
    track_branch_target = torch.where(has_track, best_track_q, torch.zeros_like(best_track_q))
    type_q_target = torch.stack([track_branch_target, search_q], dim=1)
    type_q_mask = torch.stack([has_track.float(), search_valid.float()], dim=1)
    type_q_err = F.smooth_l1_loss(type_q, type_q_target, reduction="none")
    type_q_loss = (type_q_err * type_q_mask).sum() / type_q_mask.sum().clamp_min(1.0)
    q_valid = q_mask > 0.5
    q_valid[:, 0] = False
    track_q_residual_target = q - track_branch_target[:, None]
    track_q_loss = F.smooth_l1_loss(track_q[q_valid], track_q_residual_target[q_valid]) if bool(torch.any(q_valid)) else torch.zeros((), device=device)
    sensor_loss = torch.zeros((), device=device)
    sensor_q_loss = torch.zeros((), device=device)
    if has_sensor_targets and sensor_logits is not None and sensor_q_pred is not None:
        sensor_pi_np = np.stack([
            b.sensor_pi if b.sensor_pi is not None else np.zeros((MAXT + 1, 2), dtype=np.float32)
            for b in batch
        ]).astype(np.float32)
        sensor_q_np = np.stack([
            b.sensor_q if b.sensor_q is not None else np.zeros((MAXT + 1, 2), dtype=np.float32)
            for b in batch
        ]).astype(np.float32)
        sensor_mask_np = np.stack([
            b.sensor_q_mask if b.sensor_q_mask is not None else np.zeros((MAXT + 1, 2), dtype=np.float32)
            for b in batch
        ]).astype(np.float32)
        sensor_pi = torch.from_numpy(sensor_pi_np).to(device)
        sensor_q_target = torch.from_numpy(sensor_q_np / q_scale).to(device)
        sensor_mask = torch.from_numpy(sensor_mask_np).to(device) > 0.5
        row_mass = sensor_pi.sum(dim=2)
        row_mask = row_mass > 1e-6
        if bool(torch.any(row_mask)):
            sensor_target = sensor_pi / row_mass[:, :, None].clamp_min(1e-6)
            sensor_loss = -(sensor_target[row_mask] * F.log_softmax(sensor_logits[row_mask], dim=1)).sum(dim=1).mean()
        if bool(torch.any(sensor_mask)):
            sensor_q_loss = F.smooth_l1_loss(sensor_q_pred[sensor_mask], sensor_q_target[sensor_mask])

    loss = (
        float(type_loss_weight) * type_loss
        + float(rank_loss_weight) * rank_loss
        + float(value_loss_weight) * v_loss
        + float(type_q_loss_weight) * type_q_loss
        + float(track_q_loss_weight) * track_q_loss
        + 0.5 * sensor_loss
        + 0.25 * sensor_q_loss
    )
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return {
        "loss": float(loss.detach().cpu()),
        "type_loss": float(type_loss.detach().cpu()),
        "rank_loss": float(rank_loss.detach().cpu()),
        "v_loss": float(v_loss.detach().cpu()),
        "type_q_loss": float(type_q_loss.detach().cpu()),
        "track_q_loss": float(track_q_loss.detach().cpu()),
        "sensor_loss": float(sensor_loss.detach().cpu()),
        "sensor_q_loss": float(sensor_q_loss.detach().cpu()),
    }


def eval_factory(factory, name, cells, seeds, windows, args):
    rows = []
    for init, rate in cells:
        env = make_train_eval_env(rate, args)
        for seed in seeds:
            seedall(seed)
            try:
                planner = factory(init, rate)
            except TypeError:
                planner = factory()
            w, _ = run_fixed(planner, name, init, MAXT, seed, windows, 200, env)
            s = summarize_window_df(w, "fixed")
            s.update(planner=name, initial_targets=init, rate=rate, seed=seed)
            rows.append(s)
    return rows


def run_eval(model: MutualRadarNet, args, tag: str):
    from adaptive_context_factorized import base_model, ContextPlanner

    cells = [(int(i), float(r)) for i in str(args.eval_initials).split(",") for r in str(args.eval_rates).split(",")]
    seeds = [int(x) for x in str(args.eval_seeds).split(",") if x]
    def planner_env(rate: float = 0.0):
        env = make_train_eval_env(rate, args)
        if getattr(args, "planner_search_debt_weight", None) is not None and args.planner_search_debt_weight >= 0:
            env["planner_search_debt_penalty_weight"] = float(args.planner_search_debt_weight)
        if getattr(args, "planner_search_action_reward", None) is not None:
            env["planner_search_action_reward"] = float(args.planner_search_action_reward)
        return env
    planners = {
        "MutualRadar_DirectPolicy": lambda: MutualRadarDirectPlanner(model, alpha=0.0, beta=0.0, threshold=args.direct_threshold, direct_mode=args.direct_mode),
        "MutualRadar_DirectPolicyQ": lambda: MutualRadarDirectPlanner(model, alpha=1.0, beta=1.0, threshold=args.direct_threshold, direct_mode=args.direct_mode),
        "FastBase_t0.20": lambda: ContextPlanner.from_base(base_model(), 0.20),
        "FastBase_t0.30": lambda: ContextPlanner.from_base(base_model(), 0.30),
        "EDF": lambda: EDFPlanner(MAXT),
        "EST": lambda: ESTPlanner(MAXT),
    }
    if not args.skip_mcts_eval:
        planners.update({
            f"MutualRadar_MCTS_P_r{args.eval_rollouts}": lambda _init=None, rate=0.0: MutualRadarMCTSPlanner(model, planner_env(rate), rollouts=args.eval_rollouts, c_puct=args.c_puct, expand_top_k=args.expand_top_k, q_scale=args.eval_q_scale, prior_mode=args.prior_mode, belief_search_weight=args.belief_search_weight, belief_search_cap=args.belief_search_cap),
            f"MutualRadar_MCTS_PQ_r{args.eval_rollouts}": lambda _init=None, rate=0.0: MutualRadarMCTSPlanner(model, planner_env(rate), rollouts=args.eval_rollouts, c_puct=args.c_puct, expand_top_k=args.expand_top_k, q_scale=args.eval_q_scale, use_q_head=True, q_utility_weight=1.0, prior_mode=args.prior_mode, belief_search_weight=args.belief_search_weight, belief_search_cap=args.belief_search_cap),
            f"MutualRadar_MCTS_PV_r{args.eval_rollouts}": lambda _init=None, rate=0.0: MutualRadarMCTSPlanner(model, planner_env(rate), rollouts=args.eval_rollouts, c_puct=args.c_puct, expand_top_k=args.expand_top_k, q_scale=args.eval_q_scale, use_value_head=True, leaf_value_mix=0.5, prior_mode=args.prior_mode, belief_search_weight=args.belief_search_weight, belief_search_cap=args.belief_search_cap),
            f"MutualRadar_MCTS_PVQ_r{args.eval_rollouts}": lambda _init=None, rate=0.0: MutualRadarMCTSPlanner(model, planner_env(rate), rollouts=args.eval_rollouts, c_puct=args.c_puct, expand_top_k=args.expand_top_k, q_scale=args.eval_q_scale, use_q_head=True, q_utility_weight=1.0, use_value_head=True, leaf_value_mix=0.5, prior_mode=args.prior_mode, belief_search_weight=args.belief_search_weight, belief_search_cap=args.belief_search_cap),
        })
    rows = []
    for name, fac in planners.items():
        r = eval_factory(fac, name, cells, seeds, args.eval_windows, args)
        rows.extend(r)
        print("eval", tag, name, float(pd.DataFrame(r)["reward_per_200ms_eq"].mean()), flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(RUN_OUT / f"{tag}_eval_raw.csv", index=False)
    summary = raw.groupby("planner").agg(
        reward=("reward_per_200ms_eq", "mean"),
        drop=("mean_drop_pct_active", "mean"),
        delay=("mean_delay_active", "mean"),
        search=("search_fraction", "mean"),
        latency=("planning_ms_per_200ms_eq", "mean"),
    ).reset_index().sort_values("reward", ascending=False)
    summary.to_csv(RUN_OUT / f"{tag}_eval_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    return summary


def load_or_init(path: Path, args) -> MutualRadarNet:
    model = MutualRadarNet(d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers, head_arch=getattr(args, "head_arch", "baseline"))
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
    elif args.init_ckpt:
        model.load_state_dict(torch.load(args.init_ckpt, map_location="cpu"), strict=False)
    model.to(DEVICE)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "train", "eval"], default="smoke")
    ap.add_argument("--seed", type=int, default=76)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--episodes-per-iter", type=int, default=3)
    ap.add_argument("--windows-per-episode", type=int, default=8)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--eval-rollouts", type=int, default=8)
    ap.add_argument("--eval-q-scale", type=float, default=3.78)
    ap.add_argument("--skip-mcts-eval", action="store_true")
    ap.add_argument("--expand-top-k", type=int, default=12)
    ap.add_argument("--c-puct", type=float, default=1.25)
    ap.add_argument("--prior-mode", choices=["factorized", "flat", "branch_corrected"], default="factorized")
    ap.add_argument("--direct-mode", choices=["prob", "flat", "branch"], default="prob")
    ap.add_argument("--direct-threshold", type=float, default=0.5)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--head-arch", choices=["baseline", "branch_context", "specialized", "moe"], default="baseline")
    ap.add_argument("--init-ckpt", default="")
    ap.add_argument("--track-loss-penalty", type=float, default=-1.0)
    ap.add_argument("--track-urgency-bonus-weight", type=float, default=-1.0)
    ap.add_argument("--search-debt-penalty-weight", type=float, default=-1.0)
    ap.add_argument("--planner-search-debt-weight", type=float, default=-1.0)
    ap.add_argument("--planner-search-action-reward", type=float, default=0.0)
    ap.add_argument("--belief-search-weight", type=float, default=0.0)
    ap.add_argument("--belief-search-cap", type=float, default=8.0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--train-steps", type=int, default=16)
    ap.add_argument("--replay-size", type=int, default=50000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train-initials", default="5,15,25,35,50,75,100")
    ap.add_argument("--train-rates", default="0,1,2,5")
    ap.add_argument("--eval-initials", default="15,50")
    ap.add_argument("--eval-rates", default="0,2")
    ap.add_argument("--eval-seeds", default="100")
    ap.add_argument("--eval-windows", type=int, default=20)
    ap.add_argument("--ckpt", default=str(RUN_OUT / "MutualRadar_foundation.pt"))
    args = ap.parse_args()
    seedall(args.seed)
    ckpt = Path(args.ckpt)
    model = load_or_init(ckpt, args)

    if args.mode == "eval":
        run_eval(model.to(DEVICE).eval(), args, "eval")
        return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    replay = ReplayBuffer(args.replay_size)
    train_log = []
    q_scale = 100.0
    for it in range(1, args.iterations + 1):
        args.iteration = it
        model.to(DEVICE).eval()
        rows, mean_reward = collect_selfplay(model.eval(), args, make_env(0.0))
        replay.extend(rows)
        abs_targets = [abs(x.ret) for x in replay.items] + [abs(float(v)) for r in replay.items for v in r.q[r.q_mask > 0.5]]
        q_scale = float(max(1.0, np.percentile(abs_targets, 90))) if abs_targets else q_scale
        model.to(DEVICE).train()
        metrics = []
        for _ in range(args.train_steps):
            m = train_step(model, opt, replay, args.batch_size, q_scale)
            if m:
                metrics.append(m)
        row = {
            "iteration": it,
            "collected": len(rows),
            "replay": len(replay),
            "mean_selfplay_reward": mean_reward,
            "q_scale": q_scale,
        }
        if metrics:
            for k in metrics[0]:
                row[k] = float(np.mean([m[k] for m in metrics]))
        train_log.append(row)
        print("MutualRadar_train", json.dumps(row), flush=True)
        torch.save(model.cpu().state_dict(), ckpt)
        pd.DataFrame(train_log).to_csv(RUN_OUT / "MutualRadar_train_log.csv", index=False)
    tag = "smoke" if args.mode == "smoke" else "train"
    run_eval(model.to(DEVICE).eval(), args, tag)


if __name__ == "__main__":
    main()
    from adaptive_context_factorized import base_model, ContextPlanner

