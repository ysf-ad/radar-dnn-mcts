"""Clean mutual MCTS training loop for radar scheduling.

This script is intentionally separate from the older exploratory files.  The
training path is:

    model_k -> model-guided MCTS -> improved P/Q/V targets -> model_{k+1}

No fixed teacher policy is used for the core model.  A fast batch head can be
trained afterward from the current mutual MCTS targets, but the shared
foundation model is improved through its own MCTS loop.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from final_radar_campaign import MAXT, build_env, get_obs, run_fixed, seedall, summarize_window_df
from mutual_foundation import (
    DEVICE,
    MutualRadarDirectPlanner,
    MutualRadarMCTSPlanner,
    MutualRadarNet,
    ReplayBuffer,
    SearchTarget,
    _copy_plan_obs,
    action_priors_from_logits,
    advance_plan_obs,
    train_step,
)
from mutual_features import slot_features, tokenize
from realistic_reward_retrain import adapter
from repaired_campaign_tools import EDFPlanner, ESTPlanner, SEARCH_DWELL_MS
from single_sensor_qdistill_batch import single_env
from strict_window_report import execute_plan_until_budget


def configured_env(rate: float, args) -> Dict[str, float]:
    """Return the radar environment used for both self-play and evaluation.

    The legacy/current setting lets search refresh tracked targets and gives
    every search a positive reward. Under heavy load that makes all-search a
    locally valid strategy, so the operational mode separates discovery from
    tracking: search discovers hidden targets/sectors, while tracked-target
    deadlines must be maintained by track actions.
    """
    env = single_env(rate)
    mode = getattr(args, "env_mode", "current")
    if mode == "current":
        return env
    if mode == "no_refresh":
        env.update(enable_search_refresh_tracked=0, search_refresh_gain=0.0)
        return env
    if mode == "operational":
        env.update(
            enable_search_refresh_tracked=int(getattr(args, "search_refresh_tracked", 0)),
            search_refresh_gain=float(getattr(args, "search_refresh_gain", 0.0)),
            search_action_reward=0.0,
            track_update_reward=float(getattr(args, "track_update_reward", 0.30)),
            track_loss_penalty=float(getattr(args, "track_loss_penalty", 3.0)),
            penalize_hidden_targets=0,
            search_delay_mode=0,
            search_debt_penalty_weight=float(getattr(args, "search_debt_penalty_weight", 0.00025)),
            search_delay_penalty_cap=-1.0,
        )
        return env
    if mode in {"original_reward", "radarxs_original", "radarxs_original_global"}:
        # Match original_radarxs.h as closely as the current wrapped env allows:
        # explicit 0.10 search/track rewards, local track-delay cost, and a
        # linear uncapped search-delay cost with SEARCH_PENALTY=0.1/1000.
        env.update(
            enable_global_delay=1 if mode == "radarxs_original_global" else 0,
            enable_local_delay=1,
            enable_search_refresh_tracked=1,
            search_refresh_gain=1.0,
            search_action_reward=0.10,
            track_update_reward=0.10,
            track_loss_penalty=1.0,
            penalize_hidden_targets=0,
            search_delay_mode=0,
            search_debt_penalty_weight=0.0001,
            search_debt_tau_ms=10.0,
            search_delay_penalty_cap=-1.0,
        )
        return env
    if mode == "radarxs_balanced":
        # Original-style timing penalties, but remove the per-search reward
        # exploit. Search is useful by reducing linear search debt and finding
        # targets, not by receiving a constant payment every 10 ms.
        env.update(
            enable_global_delay=1,
            enable_local_delay=1,
            enable_search_refresh_tracked=0,
            search_refresh_gain=0.0,
            search_action_reward=0.0,
            track_update_reward=float(getattr(args, "track_update_reward", 0.10)),
            track_loss_penalty=float(getattr(args, "track_loss_penalty", 1.0)),
            penalize_hidden_targets=0,
            search_delay_mode=0,
            search_debt_penalty_weight=float(getattr(args, "search_debt_penalty_weight", 0.0001)),
            search_debt_tau_ms=10.0,
            search_delay_penalty_cap=-1.0,
        )
        return env
    if mode == "radarxs_mission_delta":
        # Atari-like mission-score delta: actions receive no intrinsic reward.
        # Reward is produced by changes in objective state costs: accumulated
        # target tardiness, dropped tracks, and surveillance-sector staleness.
        env.update(
            enable_global_delay=1,
            enable_local_delay=0,
            enable_search_refresh_tracked=0,
            search_refresh_gain=0.0,
            search_action_reward=0.0,
            track_update_reward=0.0,
            track_loss_penalty=float(getattr(args, "track_loss_penalty", 4.0)),
            penalize_hidden_targets=0,
            search_delay_mode=0,
            search_debt_penalty_weight=0.0,
            search_debt_tau_ms=10.0,
            search_delay_penalty_cap=-1.0,
            target_service_weight=float(getattr(args, "target_service_weight", 1.0)),
            target_service_horizon_ms=float(getattr(args, "target_service_horizon_ms", 1000.0)),
            sector_staleness_weight=float(getattr(args, "sector_staleness_weight", 0.001)),
            searched_sector_reward_weight=0.0,
            search_frame_overdue_weight=float(getattr(args, "search_frame_overdue_weight", 0.05)),
            search_frame_desired_ms=float(getattr(args, "search_frame_desired_ms", 3000.0)),
            search_frame_deadline_ms=float(getattr(args, "search_frame_deadline_ms", 4500.0)),
            search_frame_drop_penalty=float(getattr(args, "search_frame_drop_penalty", 4.0)),
        )
        return env
    if mode == "penalty_only_frame":
        # Pure cost objective for radar scheduling.  Search and track actions
        # receive no positive reward.  The return is only the negative cost of
        # target lateness, target drops, and stale surveillance frames.
        env.update(
            enable_global_delay=1,
            enable_local_delay=0,
            enable_search_refresh_tracked=0,
            search_refresh_gain=0.0,
            search_action_reward=0.0,
            track_update_reward=0.0,
            track_loss_penalty=float(getattr(args, "track_loss_penalty", 8.0)),
            penalize_hidden_targets=1,
            search_delay_mode=0,
            search_debt_penalty_weight=0.0,
            search_debt_tau_ms=10.0,
            search_delay_penalty_cap=-1.0,
            target_service_weight=0.0,
            sector_staleness_weight=0.0,
            searched_sector_reward_weight=0.0,
            search_frame_overdue_weight=float(getattr(args, "search_frame_overdue_weight", 0.20)),
            search_frame_desired_ms=float(getattr(args, "search_frame_desired_ms", 3000.0)),
            search_frame_deadline_ms=float(getattr(args, "search_frame_deadline_ms", 4500.0)),
            search_frame_drop_penalty=float(getattr(args, "search_frame_drop_penalty", 8.0)),
        )
        return env
    if mode == "repaired_stress_reward":
        # Closest to the earlier "repaired_stress" environment that produced
        # a sane decreasing search fraction in the heuristic probes: search is
        # useful but not as overpowering as the original 0.25 reward.
        env.update(
            enable_search_refresh_tracked=1,
            search_refresh_gain=0.75,
            search_action_reward=0.08,
            track_update_reward=0.30,
            track_loss_penalty=1.0,
            penalize_hidden_targets=0,
            search_delay_mode=1,
            search_debt_penalty_weight=0.058,
            search_debt_tau_ms=200.0,
            search_delay_penalty_cap=2.0,
        )
        return env
    if mode == "balanced_linear":
        # Operational compromise: no constant search bonus, no search refresh
        # shortcut for tracked targets, but meaningful linear surveillance debt.
        env.update(
            enable_search_refresh_tracked=0,
            search_refresh_gain=0.0,
            search_action_reward=0.0,
            track_update_reward=float(getattr(args, "track_update_reward", 0.30)),
            track_loss_penalty=float(getattr(args, "track_loss_penalty", 4.0)),
            penalize_hidden_targets=1,
            search_delay_mode=0,
            search_debt_penalty_weight=float(getattr(args, "search_debt_penalty_weight", 0.006)),
            search_delay_penalty_cap=-1.0,
            sector_staleness_weight=float(getattr(args, "sector_staleness_weight", 0.0)),
        )
        return env
    if mode == "staleness_potential":
        # Search earns value by reducing stale surveillance sectors rather than
        # by a constant per-search bonus. This should create cadence in light
        # load while still letting deadline pressure dominate in heavy load.
        env.update(
            enable_search_refresh_tracked=0,
            search_refresh_gain=0.0,
            search_action_reward=0.0,
            track_update_reward=float(getattr(args, "track_update_reward", 0.30)),
            track_loss_penalty=float(getattr(args, "track_loss_penalty", 4.0)),
            penalize_hidden_targets=1,
            search_delay_mode=0,
            search_debt_penalty_weight=float(getattr(args, "search_debt_penalty_weight", 0.001)),
            search_delay_penalty_cap=-1.0,
            sector_staleness_weight=float(getattr(args, "sector_staleness_weight", 0.003)),
        )
        return env
    if mode == "searched_sector_frame":
        # Completion-style search reward plus an explicit overdue-frame cost.
        # This models surveillance sectors as useful only when they are stale
        # enough to be worth refreshing, instead of rewarding every search beam.
        env.update(
            enable_search_refresh_tracked=0,
            search_refresh_gain=0.0,
            search_action_reward=0.0,
            track_update_reward=float(getattr(args, "track_update_reward", 0.30)),
            track_loss_penalty=float(getattr(args, "track_loss_penalty", 4.0)),
            penalize_hidden_targets=1,
            search_delay_mode=0,
            search_debt_penalty_weight=0.0,
            search_delay_penalty_cap=-1.0,
            sector_staleness_weight=0.0,
            searched_sector_reward_weight=float(getattr(args, "searched_sector_reward_weight", 0.25)),
            search_frame_overdue_weight=float(getattr(args, "search_frame_overdue_weight", 0.10)),
            search_frame_desired_ms=float(getattr(args, "search_frame_desired_ms", 3000.0)),
            search_frame_deadline_ms=float(getattr(args, "search_frame_deadline_ms", 4500.0)),
            search_frame_drop_penalty=float(getattr(args, "search_frame_drop_penalty", 4.0)),
        )
        return env
    if mode == "ding_moo_frame":
        # Ding/Moo-style surveillance frame objective: search sectors are
        # first-class periodic tasks with desired revisit and deadline costs.
        env.update(
            enable_search_refresh_tracked=0,
            search_refresh_gain=0.0,
            search_action_reward=0.0,
            track_update_reward=float(getattr(args, "track_update_reward", 0.30)),
            track_loss_penalty=float(getattr(args, "track_loss_penalty", 4.0)),
            penalize_hidden_targets=1,
            search_delay_mode=0,
            search_debt_penalty_weight=0.0,
            search_delay_penalty_cap=-1.0,
            sector_staleness_weight=0.0,
            searched_sector_reward_weight=0.0,
            search_frame_overdue_weight=float(getattr(args, "search_frame_overdue_weight", 0.10)),
            search_frame_desired_ms=float(getattr(args, "search_frame_desired_ms", 3000.0)),
            search_frame_deadline_ms=float(getattr(args, "search_frame_deadline_ms", 4500.0)),
            search_frame_drop_penalty=float(getattr(args, "search_frame_drop_penalty", 4.0)),
        )
        return env
    if mode == "mcts_sched_v1":
        # Clean finite-horizon scheduling objective.  Track jobs and search
        # sectors are both deadline-bearing tasks; no constant search bonus and
        # no search-refresh shortcut.  This keeps EDF/EST applicable while
        # giving MCTS dense branch-comparable rewards inside a planning window.
        env.update(
            enable_search_refresh_tracked=0,
            search_refresh_gain=0.0,
            search_action_reward=0.0,
            track_update_reward=float(getattr(args, "track_update_reward", 0.65)),
            track_loss_penalty=float(getattr(args, "track_loss_penalty", 8.0)),
            penalize_hidden_targets=1,
            search_delay_mode=0,
            search_debt_penalty_weight=0.0,
            search_delay_penalty_cap=-1.0,
            sector_staleness_weight=0.0,
            # One search refreshes a 2x2 sector block.  Keep the per-sector
            # reward below the track-service reward so search cannot dominate
            # merely by touching four cells.
            searched_sector_reward_weight=float(getattr(args, "searched_sector_reward_weight", 0.12)),
            search_frame_overdue_weight=float(getattr(args, "search_frame_overdue_weight", 0.20)),
            search_frame_desired_ms=float(getattr(args, "search_frame_desired_ms", 1800.0)),
            search_frame_deadline_ms=float(getattr(args, "search_frame_deadline_ms", 3600.0)),
            search_frame_drop_penalty=float(getattr(args, "search_frame_drop_penalty", 8.0)),
        )
        return env
    raise ValueError(f"unknown env_mode={mode!r}")


def train_step_hard_policy(model: MutualRadarNet, opt, replay: ReplayBuffer, batch_size: int, q_scale: float, type_loss_weight: float = 1.0):
    """Temperature-zero AlphaZero update for low-rollout scheduling MCTS.

    Visit-count targets are very noisy with 8-16 rollouts and a large target
    set.  For deterministic scheduling, the robust target is the MCTS-improved
    best action, while Q/V still regress to searched returns.
    """
    if len(replay) < max(4, batch_size // 4):
        return {}
    import random
    import torch.nn.functional as F

    batch = random.sample(replay.items, min(int(batch_size), len(replay.items)))
    x = torch.from_numpy(np.stack([b.x for b in batch]).astype(np.float32)).to(DEVICE)
    slot = torch.from_numpy(np.stack([b.slot for b in batch]).astype(np.float32)).to(DEVICE)
    pi = torch.from_numpy(np.stack([b.pi for b in batch]).astype(np.float32)).to(DEVICE)
    q = torch.from_numpy(np.stack([b.q for b in batch]).astype(np.float32) / q_scale).to(DEVICE)
    q_mask = torch.from_numpy(np.stack([b.q_mask for b in batch]).astype(np.float32)).to(DEVICE)
    ret = torch.tensor([b.ret / q_scale for b in batch], dtype=torch.float32, device=DEVICE)

    type_logit, track_logits, value, type_q, track_q = model(x, slot)
    best_action = torch.argmax(pi, dim=1)
    search_label = (best_action == 0).float()
    type_loss = F.binary_cross_entropy_with_logits(type_logit, search_label)

    track_rows = best_action > 0
    rank_loss = F.cross_entropy(track_logits[track_rows], best_action[track_rows]) if bool(track_rows.any()) else torch.zeros((), device=DEVICE)

    v_loss = F.smooth_l1_loss(value, ret)
    search_q_target = q[:, 0]
    track_available = (q_mask[:, 1:] > 0.5).any(dim=1)
    track_q_candidates = q[:, 1:].masked_fill(q_mask[:, 1:] <= 0.5, -1e9)
    max_track_q = track_q_candidates.amax(dim=1)
    max_track_q = torch.where(track_available, max_track_q, torch.zeros_like(max_track_q))
    # Match the deployed Q factorization:
    #   Q(track_i) = Q_type(track) + Q_track(i).
    # The target-specific head is a residual around the branch value.  Training
    # it to the full Q_i double-counts the branch at inference.
    type_q_target = torch.stack([max_track_q, search_q_target], dim=1)
    type_q_mask = torch.stack([track_available.float(), q_mask[:, 0]], dim=1)
    type_q_err = F.smooth_l1_loss(type_q, type_q_target, reduction="none")
    type_q_loss = (type_q_err * type_q_mask).sum() / type_q_mask.sum().clamp_min(1.0)
    q_valid = q_mask > 0.5
    q_valid[:, 0] = False
    track_q_residual_target = q - max_track_q[:, None]
    track_q_loss = F.smooth_l1_loss(track_q[q_valid], track_q_residual_target[q_valid]) if bool(torch.any(q_valid)) else torch.zeros((), device=DEVICE)

    loss = float(type_loss_weight) * type_loss + rank_loss + 0.5 * v_loss + 0.25 * type_q_loss + 0.5 * track_q_loss
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


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CreateValid1" / "results" / "mutual_alpha_radar_loop"
CLEAN = Path("results")
OUT.mkdir(parents=True, exist_ok=True)
CLEAN.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)


@dataclass
class WindowSeqTarget:
    x: np.ndarray
    slots: np.ndarray
    y: np.ndarray


class LearnedSlotSequenceHead(nn.Module):
    """Fast learned-slot sequence head sharing the mutual encoder.

    It fixes the static-slot failure: the head has explicit learned slot
    identities, so it can learn patterns like "mostly search, but insert a
    track at slot k" without relying on threshold tuning.
    """

    def __init__(self, d_model: int = 96, seq_len: int = 32):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pos = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        self.type_head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.track_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, base: MutualRadarNet, x: torch.Tensor):
        cls, tok, selected, active = base.encode_tokens(x)
        bsz, ntok, d_model = tok.shape
        pos = self.pos.unsqueeze(0).expand(bsz, -1, -1)
        cls_seq = cls[:, None, :].expand(-1, self.seq_len, -1)
        type_logits = self.type_head(torch.cat([cls_seq, pos], dim=-1)).squeeze(-1)
        tok_rep = tok[:, None, :, :].expand(-1, self.seq_len, -1, -1)
        cls_rep = cls[:, None, None, :].expand(-1, self.seq_len, ntok, -1)
        pos_rep = pos[:, :, None, :].expand(-1, self.seq_len, ntok, -1)
        track_logits = self.track_head(torch.cat([tok_rep, cls_rep, pos_rep], dim=-1)).squeeze(-1)
        track_mask = active.clone()
        track_mask[:, 0] = False
        track_logits = track_logits.masked_fill(~track_mask[:, None, :], -1e9)
        return type_logits, track_logits


class SlotAttentionSequenceHead(nn.Module):
    """Batch sequence head with slot-to-slot self-attention.

    The slot tokens attend to each other before emitting actions, so the head
    can represent sequence patterns rather than independent per-slot decisions.
    """

    def __init__(self, d_model: int = 96, seq_len: int = 32, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.seq_len = int(seq_len)
        self.slot_tokens = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.05,
            activation="gelu",
        )
        self.slot_encoder = nn.TransformerEncoder(layer, num_layers=nlayers, enable_nested_tensor=False, mask_check=False)
        self.cross = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.05)
        self.type_head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.track_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, base: MutualRadarNet, x: torch.Tensor):
        cls, tok, selected, active = base.encode_tokens(x)
        bsz, ntok, d_model = tok.shape
        slots = self.slot_tokens.unsqueeze(0).expand(bsz, -1, -1)
        slots = slots + cls[:, None, :]
        slots = self.slot_encoder(slots)
        attn, _ = self.cross(slots, tok, tok, key_padding_mask=~active)
        slots = slots + attn

        cls_seq = cls[:, None, :].expand(-1, self.seq_len, -1)
        type_logits = self.type_head(torch.cat([cls_seq, slots], dim=-1)).squeeze(-1)

        tok_rep = tok[:, None, :, :].expand(-1, self.seq_len, -1, -1)
        cls_rep = cls[:, None, None, :].expand(-1, self.seq_len, ntok, -1)
        slot_rep = slots[:, :, None, :].expand(-1, self.seq_len, ntok, -1)
        track_logits = self.track_head(torch.cat([tok_rep, cls_rep, slot_rep], dim=-1)).squeeze(-1)
        track_mask = active.clone()
        track_mask[:, 0] = False
        track_logits = track_logits.masked_fill(~track_mask[:, None, :], -1e9)
        return type_logits, track_logits


class MutualLearnedSlotBatchPlanner:
    def __init__(self, base: MutualRadarNet, head):
        self.base = base.eval()
        self.head = head.eval()
        self.adapt = adapter()

    @property
    def device(self):
        return next(self.base.parameters()).device

    def warmup(self, obs, budget_ms=200):
        _ = self.plan(obs, budget_ms)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def plan(self, obs, budget_ms=200):
        x = tokenize(self.adapt, obs, selected=set(), search_count=0)
        with torch.inference_mode():
            tl, tr = self.head(self.base, torch.from_numpy(x).float().unsqueeze(0).to(self.device))
            choose_search = tl[0] >= 0.0
            best = torch.argmax(tr[0], dim=-1)
            actions = torch.where(choose_search, torch.zeros_like(best), best).detach().cpu().numpy().astype(int)
            scores_np = tr[0].detach().cpu().numpy()
        used = set()
        for i, a in enumerate(actions.tolist()):
            if a <= 0:
                continue
            if a not in used:
                used.add(int(a))
                continue
            row = scores_np[i].copy()
            row[0] = -1e9
            for u in used:
                if 0 <= u < row.shape[0]:
                    row[u] = -1e9
            repl = int(np.argmax(row))
            actions[i] = repl if row[repl] > -1e8 else 0
            if actions[i] > 0:
                used.add(int(actions[i]))
        return [int(a) for a in actions]


def nominal_slots_for_obs(obs: Dict[str, np.ndarray], seq_len: int, budget_ms: float = 200.0) -> np.ndarray:
    return np.stack(
        [
            slot_features(
                obs,
                budget_ms * float(i) / max(1, seq_len),
                search_count=0,
                track_count=i,
                last_action=-1,
                budget_ms=budget_ms,
            )
            for i in range(seq_len)
        ]
    ).astype(np.float32)


def mutual_batch_forward(model: MutualRadarNet, x: torch.Tensor, slots: torch.Tensor):
    bsz, seq_len, _ = slots.shape
    cls, tok, selected, active = model.encode_tokens(x)
    cls_b = cls[:, None, :].expand(-1, seq_len, -1).reshape(bsz * seq_len, -1)
    tok_b = tok[:, None, :, :].expand(-1, seq_len, -1, -1).reshape(bsz * seq_len, tok.shape[1], tok.shape[2])
    selected_b = selected[:, None, :].expand(-1, seq_len, -1).reshape(bsz * seq_len, selected.shape[1])
    active_b = active[:, None, :].expand(-1, seq_len, -1).reshape(bsz * seq_len, active.shape[1])
    slot_b = slots.reshape(bsz * seq_len, slots.shape[-1])
    tl, tr, value, tq, tqr = model.forward_heads(cls_b, tok_b, selected_b, active_b, slot_b)
    return (
        tl.reshape(bsz, seq_len),
        tr.reshape(bsz, seq_len, -1),
        value.reshape(bsz, seq_len),
        tq.reshape(bsz, seq_len, -1),
        tqr.reshape(bsz, seq_len, -1),
    )


def train_sequence_batch_step(
    model: MutualRadarNet,
    opt,
    seq_replay: List[WindowSeqTarget],
    batch_size: int,
    seq_loss_weight: float,
    search_pos_weight: float = 1.0,
):
    if not seq_replay:
        return {}
    import random
    import torch.nn.functional as F

    batch = random.sample(seq_replay, min(int(batch_size), len(seq_replay)))
    x = torch.from_numpy(np.stack([b.x for b in batch]).astype(np.float32)).to(DEVICE)
    slots = torch.from_numpy(np.stack([b.slots for b in batch]).astype(np.float32)).to(DEVICE)
    y = torch.from_numpy(np.stack([b.y for b in batch]).astype(np.int64)).to(DEVICE)
    tl, tr, _, _, _ = mutual_batch_forward(model, x, slots)
    valid = y >= 0
    if bool(valid.any()):
        type_loss = F.binary_cross_entropy_with_logits(
            tl[valid],
            (y[valid] == 0).float(),
            pos_weight=torch.tensor(float(search_pos_weight), device=DEVICE),
        )
    else:
        type_loss = torch.zeros((), device=DEVICE)
    track_valid = valid & (y > 0)
    if bool(track_valid.any()):
        finite_track = torch.isfinite(tr) & (tr > -1e8)
        track_valid = track_valid & finite_track.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    track_loss = F.cross_entropy(tr[track_valid], y[track_valid]) if bool(track_valid.any()) else torch.zeros((), device=DEVICE)
    loss = float(seq_loss_weight) * (type_loss + track_loss)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return {
        "seq_loss": float(loss.detach().cpu()),
        "seq_type_loss": float(type_loss.detach().cpu()),
        "seq_track_loss": float(track_loss.detach().cpu()),
    }


def train_learned_slot_step(
    model: MutualRadarNet,
    head,
    opt,
    seq_replay: List[WindowSeqTarget],
    batch_size: int,
    seq_loss_weight: float,
    search_pos_weight: float = 1.0,
):
    if not seq_replay:
        return {}
    import random
    import torch.nn.functional as F

    batch = random.sample(seq_replay, min(int(batch_size), len(seq_replay)))
    x = torch.from_numpy(np.stack([b.x for b in batch]).astype(np.float32)).to(DEVICE)
    y = torch.from_numpy(np.stack([b.y for b in batch]).astype(np.int64)).to(DEVICE)
    tl, tr = head(model, x)
    valid = y >= 0
    if bool(valid.any()):
        type_loss = F.binary_cross_entropy_with_logits(
            tl[valid],
            (y[valid] == 0).float(),
            pos_weight=torch.tensor(float(search_pos_weight), device=DEVICE),
        )
    else:
        type_loss = torch.zeros((), device=DEVICE)
    track_valid = valid & (y > 0)
    if bool(track_valid.any()):
        finite_track = torch.isfinite(tr) & (tr > -1e8)
        track_valid = track_valid & finite_track.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    track_loss = F.cross_entropy(tr[track_valid], y[track_valid]) if bool(track_valid.any()) else torch.zeros((), device=DEVICE)
    loss = float(seq_loss_weight) * (type_loss + track_loss)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
    opt.step()
    return {
        "learned_slot_seq_loss": float(loss.detach().cpu()),
        "learned_slot_type_loss": float(type_loss.detach().cpu()),
        "learned_slot_track_loss": float(track_loss.detach().cpu()),
    }


def train_learned_slot_only(
    model: MutualRadarNet,
    head,
    opt,
    seq_replay: List[WindowSeqTarget],
    batch_size: int,
    steps: int,
    seq_loss_weight: float,
    search_pos_weight: float,
):
    metrics = []
    model.train()
    head.train()
    for _ in range(int(steps)):
        m = train_learned_slot_step(model, head, opt, seq_replay, batch_size, seq_loss_weight, search_pos_weight)
        if m:
            metrics.append(m)
    return metrics


class MutualArgmaxPolicyPlanner:
    """No-threshold direct policy planner.

    It chooses the argmax over the full factorized action distribution:
        P(search), (1 - P(search)) * P(target_i | track)
    Then it advances the internal planning state and repeats until the 200 ms
    budget is filled.
    """

    def __init__(self, model: MutualRadarNet, mode: str = "policy", q_scale: float = 1.0, allow_retrack: bool = False):
        self.model = model.eval()
        self.mode = str(mode)
        self.q_scale = float(q_scale)
        self.allow_retrack = bool(allow_retrack)
        self.adapt = adapter()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def warmup(self, obs, budget_ms=200):
        _ = self.plan(obs, budget_ms)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def plan(self, obs, budget_ms=200):
        plan_obs = _copy_plan_obs(obs)
        selected = set()
        plan: List[int] = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        while elapsed < float(budget_ms) and len(plan) < 64:
            x = tokenize(self.adapt, plan_obs, selected=set() if self.allow_retrack else selected, search_count=search_count)
            slot = slot_features(plan_obs, elapsed, search_count, track_count, last, float(budget_ms))
            with torch.inference_mode():
                tl, tr, _, tq, tqr = self.model(
                    torch.from_numpy(x).float().unsqueeze(0).to(self.device),
                    torch.from_numpy(slot).float().unsqueeze(0).to(self.device),
                )
            if self.mode == "q":
                scores = np.zeros((MAXT + 1,), dtype=np.float32)
                type_q = tq[0].detach().cpu().numpy()
                track_q = tqr[0].detach().cpu().numpy()
                scores[0] = float(type_q[1])
                scores[1:] = float(type_q[0]) + track_q[1:]
                visible = (
                    np.asarray(plan_obs["active_mask"], dtype=bool)
                    & (np.asarray(plan_obs["t_deadline"], dtype=np.float32) > 0.0)
                )
                for action in range(1, len(scores)):
                    idx = action - 1
                    if idx >= len(visible) or not visible[idx] or ((not self.allow_retrack) and action in selected):
                        scores[action] = -1e9
                a = int(np.argmax(scores))
            else:
                priors = action_priors_from_logits(tl[0], tr[0], "factorized")
                a = int(np.argmax(priors))
            if a == 0:
                dt = SEARCH_DWELL_MS
                search_count += 1
            else:
                if not self.allow_retrack:
                    selected.add(a)
                dwell = np.asarray(plan_obs["t_dwell"], dtype=np.float32)
                dt = float(dwell[a - 1]) if 1 <= a <= len(dwell) else SEARCH_DWELL_MS
                track_count += 1
            plan.append(a)
            advance_plan_obs(plan_obs, a, max(1.0, float(dt)), search_refresh_tracked=False, search_refresh_gain=0.0)
            elapsed += max(1.0, float(dt))
            last = a
        return plan if plan else [0]


class MutualBatchArgmaxPlanner:
    """One-encoder batched 0-rollout planner from the same mutual P/Q heads.

    This is not an extra trained decoder. It reuses the trained encoder and
    single-action heads, but evaluates all nominal 200 ms slots in one vectorized
    forward-head pass. It is therefore the right speed diagnostic for "can the
    foundation heads themselves emit a batch plan?"
    """

    def __init__(self, model: MutualRadarNet, seq_len: int = 32, mode: str = "policy", allow_retrack: bool = False):
        self.model = model.eval()
        self.seq_len = int(seq_len)
        self.mode = str(mode)
        self.allow_retrack = bool(allow_retrack)
        self.adapt = adapter()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def warmup(self, obs, budget_ms=200):
        _ = self.plan(obs, budget_ms)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def plan(self, obs, budget_ms=200):
        x = tokenize(self.adapt, obs, selected=set(), search_count=0)
        slots = np.stack(
            [
                slot_features(
                    obs,
                    float(budget_ms) * float(i) / max(1, self.seq_len),
                    search_count=0,
                    track_count=i,
                    last_action=-1,
                    budget_ms=float(budget_ms),
                )
                for i in range(self.seq_len)
            ]
        ).astype(np.float32)
        with torch.inference_mode():
            tokens = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            cls, tok, selected, active = self.model.encode_tokens(tokens)
            slot_t = torch.from_numpy(slots).float().to(self.device)
            cls_b = cls.expand(self.seq_len, -1)
            tok_b = tok.expand(self.seq_len, -1, -1)
            selected_b = selected.expand(self.seq_len, -1)
            active_b = active.expand(self.seq_len, -1)
            tl, tr, _, tq, tqr = self.model.forward_heads(cls_b, tok_b, selected_b, active_b, slot_t)
            if self.mode == "q":
                scores = tq[:, 0].unsqueeze(1) + tqr
                visible = (
                    np.asarray(obs["active_mask"], dtype=bool)
                    & (np.asarray(obs["t_deadline"], dtype=np.float32) > 0.0)
                )
                invalid = np.ones((MAXT + 1,), dtype=bool)
                upto = min(len(visible), MAXT)
                invalid[0] = False
                invalid[1 : upto + 1] = ~visible[:upto]
                scores[:, torch.from_numpy(invalid).to(self.device)] = -1e9
                search_scores = tq[:, 1]
                best = torch.argmax(scores, dim=-1)
                best_score = scores[torch.arange(scores.shape[0], device=self.device), best]
                choose_search = search_scores >= best_score
            else:
                # True factorized discrete decision: first choose branch by the
                # type head, then choose target only if the branch is Track.
                # Do not compare search against individual target atoms; that
                # incorrectly penalizes Track when probability mass is split
                # across many targets.
                choose_search = tl >= 0.0
                best = torch.argmax(tr, dim=-1)
            actions = torch.where(choose_search, torch.zeros_like(best), best).detach().cpu().numpy().astype(int)
            scores_np = (tr if self.mode != "q" else scores).detach().cpu().numpy()

        if not self.allow_retrack:
            used = set()
            for i, a in enumerate(actions.tolist()):
                if a <= 0:
                    continue
                if a not in used:
                    used.add(int(a))
                    continue
                row = scores_np[i].copy()
                row[0] = -1e9
                for u in used:
                    if 0 <= u < row.shape[0]:
                        row[u] = -1e9
                repl = int(np.argmax(row))
                actions[i] = repl if row[repl] > -1e8 else 0
                if actions[i] > 0:
                    used.add(int(actions[i]))
        return [int(a) for a in actions]


class MutualBatchQUrgencyPlanner:
    """Fast 0-rollout factorized batch planner with an urgency residual.

    The learned Q heads decide Search-vs-Track and provide target scores.  A
    simple continuous urgency residual makes the target pointer monotone in the
    operational variables that define deadline risk.  This is an architectural
    bias, not a runtime rule: the action is still selected by a single vectorized
    factorized score.
    """

    def __init__(
        self,
        model: MutualRadarNet,
        seq_len: int = 32,
        urgency_weight: float = 8.0,
        deadline_weight: float | None = None,
        overdue_weight: float | None = None,
    ):
        self.model = model.eval()
        self.seq_len = int(seq_len)
        self.urgency_weight = float(urgency_weight)
        self.deadline_weight = float(urgency_weight if deadline_weight is None else deadline_weight)
        self.overdue_weight = float(urgency_weight if overdue_weight is None else overdue_weight)
        self.adapt = adapter()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def warmup(self, obs, budget_ms=200):
        _ = self.plan(obs, budget_ms)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def plan(self, obs, budget_ms=200):
        x = tokenize(self.adapt, obs, selected=set(), search_count=0)
        slots = np.stack(
            [
                slot_features(
                    obs,
                    float(budget_ms) * float(i) / max(1, self.seq_len),
                    search_count=0,
                    track_count=i,
                    last_action=-1,
                    budget_ms=float(budget_ms),
                )
                for i in range(self.seq_len)
            ]
        ).astype(np.float32)
        with torch.inference_mode():
            tokens = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            cls, tok, selected, active = self.model.encode_tokens(tokens)
            slot_t = torch.from_numpy(slots).float().to(self.device)
            cls_b = cls.expand(self.seq_len, -1)
            tok_b = tok.expand(self.seq_len, -1, -1)
            selected_b = selected.expand(self.seq_len, -1)
            active_b = active.expand(self.seq_len, -1)
            _, _, _, type_q, track_q = self.model.forward_heads(cls_b, tok_b, selected_b, active_b, slot_t)
            track_scores = (type_q[:, 0].unsqueeze(1) + track_q).detach().cpu().numpy()
            search_scores = type_q[:, 1].detach().cpu().numpy()

        visible = np.asarray(obs["active_mask"], dtype=bool) & (np.asarray(obs["t_deadline"], dtype=np.float32) > 0.0)
        deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
        desired = np.asarray(obs["t_desired"], dtype=np.float32)
        urgency = (
            -self.deadline_weight * deadline / 1000.0
            + self.overdue_weight * np.maximum(0.0, -desired) / 500.0
        ).astype(np.float32)

        actions: List[int] = []
        used: set[int] = set()
        for i in range(self.seq_len):
            row = track_scores[i].copy()
            row[0] = -1e9
            upto = min(len(urgency), len(row) - 1)
            row[1 : upto + 1] += urgency[:upto]
            for action in range(1, len(row)):
                idx = action - 1
                if idx >= len(visible) or not visible[idx] or action in used:
                    row[action] = -1e9
            best = int(np.argmax(row))
            best_score = float(row[best])
            if float(search_scores[i]) >= best_score or best_score < -1e8:
                actions.append(0)
            else:
                actions.append(best)
                used.add(best)
        return actions


def make_mcts(model: MutualRadarNet, env: Dict[str, float], args, *, training: bool, rollouts: int, mode: str):
    use_q = "q" in mode
    use_v = "v" in mode
    planner_env = dict(env)
    planner_env["mcts_rollout_policy"] = str(getattr(args, "mcts_rollout_policy", "greedy"))
    planner_env["mcts_rollout_search_period_ms"] = float(getattr(args, "mcts_rollout_search_period_ms", 120.0))
    planner_env["mcts_prior_uniform_mix"] = float(getattr(args, "mcts_prior_uniform_mix", 0.0))
    return MutualRadarMCTSPlanner(
        model,
        planner_env,
        rollouts=rollouts,
        c_puct=args.c_puct,
        expand_top_k=args.expand_top_k,
        simulation_window_ms=float(getattr(args, "mcts_simulation_window_ms", 200.0)),
        training=training,
        prior_mode="factorized",
        q_scale=args.q_scale,
        use_q_head=use_q,
        q_utility_weight=args.q_utility_weight if use_q else 0.0,
        use_value_head=use_v,
        leaf_value_mix=args.leaf_value_mix if use_v else 0.0,
        belief_search_weight=float(getattr(args, "belief_search_weight", 0.0)),
        belief_search_cap=float(getattr(args, "belief_search_cap", 8.0)),
    )


def collect_mutual_targets(model: MutualRadarNet, args, iteration: int) -> Tuple[List[SearchTarget], List[WindowSeqTarget], List[dict]]:
    rows: List[SearchTarget] = []
    seq_rows: List[WindowSeqTarget] = []
    window_rows: List[dict] = []
    init_choices = [int(x) for x in args.train_initials.split(",") if x]
    rate_choices = [float(x) for x in args.train_rates.split(",") if x]
    model.eval()
    for ep in range(args.episodes_per_iter):
        # Sample the load/rate pair independently.  The old cyclic pairing
        # correlated specific target counts with specific arrival rates, which
        # let the policy learn load-specific search/track habits instead of a
        # load-adaptive decision boundary.
        init = int(init_choices[np.random.randint(len(init_choices))])
        rate = float(rate_choices[np.random.randint(len(rate_choices))])
        seed = int(args.seed + 1000 * iteration + ep)
        seedall(seed)
        env = configured_env(rate, args)
        planner = make_mcts(model, env, args, training=True, rollouts=args.rollouts, mode=args.train_mcts_mode)
        eng = build_env(planner, init, MAXT, seed, 200, env)
        eng.reset(seed=seed)
        debt = 0.0
        traj: List[SearchTarget] = []
        for w in range(args.windows_per_episode):
            if eng.term_buf[0]:
                break
            obs = get_obs(eng, debt)
            root_x = tokenize(adapter(), obs, selected=set(), search_count=0)
            root_slots = nominal_slots_for_obs(obs, args.seq_len, 200.0)
            t0 = time.perf_counter()
            if getattr(args, "selfplay_replan_each_action", False):
                # Generate the sequence by repeatedly querying MCTS from the
                # actually updated simulator state. This avoids contaminating
                # batch-head targets with the old "searched prefix + prior
                # fallback" tail, which was the direct cause of all-search
                # sequence targets under heavy load.
                plan: List[int] = []
                targets: List[SearchTarget] = []
                reward = 0.0
                spent = 0.0
                executed = 0
                search_count = 0
                while spent < 200.0 and not eng.term_buf[0] and len(plan) < args.seq_len:
                    step_obs = get_obs(eng, debt)
                    step_plan, step_targets = planner.plan_with_targets(step_obs, max(1.0, 200.0 - spent))
                    action = int(step_plan[0]) if step_plan else 0
                    step_reward, step_spent, debt, step_executed, step_search, _ = execute_plan_until_budget(
                        eng, [action], 200.0 - spent, debt, "MutualAlpha_selfplay", seed, w
                    )
                    if step_executed <= 0 or step_spent <= 0:
                        break
                    plan.append(action)
                    targets.extend(step_targets[:1] if step_targets else [])
                    reward += float(step_reward)
                    spent += float(step_spent)
                    executed += int(step_executed)
                    search_count += int(step_search)
                plan_ms = (time.perf_counter() - t0) * 1000.0
            else:
                plan, targets = planner.plan_with_targets(obs, 200)
                plan_ms = (time.perf_counter() - t0) * 1000.0
                reward, spent, debt, executed, search_count, _ = execute_plan_until_budget(
                    eng, plan, 200.0, debt, "MutualAlpha_selfplay", seed, w
                )
            if getattr(args, "target_selected_action", False):
                for target, action in zip(targets, plan):
                    target.pi[:] = 0.0
                    if 0 <= int(action) < len(target.pi):
                        target.pi[int(action)] = 1.0
            per_target_reward = float(reward) / max(1, len(targets))
            for target in targets:
                target.reward = per_target_reward
                traj.append(target)
            y = np.full((args.seq_len,), -100, dtype=np.int64)
            n = min(args.seq_len, len(plan))
            y[:n] = np.asarray(plan[:n], dtype=np.int64)
            seq_rows.append(WindowSeqTarget(root_x.astype(np.float32), root_slots.astype(np.float32), y))
            window_rows.append(
                dict(
                    iteration=iteration,
                    episode=ep,
                    window=w,
                    initial_targets=init,
                    rate=rate,
                    seed=seed,
                    reward=float(reward),
                    planning_ms=plan_ms,
                    targets=len(targets),
                    executed=executed,
                    search_fraction=search_count / max(1, executed),
                )
            )
            if executed <= 0 or spent <= 0:
                break
        eng.close()
        G = 0.0
        for target in reversed(traj):
            G = float(target.reward) + args.gamma * G
            target.ret = G
        rows.extend(traj)
    return rows, seq_rows, window_rows


def train_mutual_model(args):
    seedall(args.seed)
    ckpt = OUT / "mutual_alpha_model.pt"
    head_ckpt = OUT / "mutual_alpha_learned_slot_head.pt"
    attn_head_ckpt = OUT / "mutual_alpha_slot_attention_head.pt"
    model = MutualRadarNet(d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers).to(DEVICE)
    learned_head = LearnedSlotSequenceHead(d_model=args.d_model, seq_len=args.seq_len).to(DEVICE)
    attn_head = SlotAttentionSequenceHead(d_model=args.d_model, seq_len=args.seq_len, nhead=args.nhead, nlayers=args.slot_decoder_layers).to(DEVICE)
    if args.resume and ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    if args.resume and head_ckpt.exists():
        learned_head.load_state_dict(torch.load(head_ckpt, map_location=DEVICE))
    if args.resume and attn_head_ckpt.exists():
        attn_head.load_state_dict(torch.load(attn_head_ckpt, map_location=DEVICE))
    opt = torch.optim.AdamW(list(model.parameters()) + list(learned_head.parameters()) + list(attn_head.parameters()), lr=args.lr, weight_decay=1e-4)
    replay = ReplayBuffer(args.replay_size)
    seq_replay: List[WindowSeqTarget] = []
    train_rows: List[dict] = []
    selfplay_rows: List[dict] = []
    q_scale = float(args.q_scale)

    if args.eval_before:
        evaluate_suite(model, args, tag="iter0", learned_head=learned_head, attn_head=attn_head)

    for it in range(1, args.iterations + 1):
        targets, seq_targets, windows = collect_mutual_targets(model, args, it)
        replay.extend(targets)
        seq_replay.extend(seq_targets)
        if len(seq_replay) > args.replay_size:
            seq_replay = seq_replay[-args.replay_size :]
        selfplay_rows.extend(windows)
        abs_targets = [abs(x.ret) for x in replay.items]
        abs_targets.extend(abs(float(v)) for r in replay.items for v in r.q[r.q_mask > 0.5])
        if abs_targets:
            q_scale = float(max(1.0, np.percentile(abs_targets, 90)))

        model.train()
        metrics = []
        for _ in range(args.train_steps):
            if args.policy_target == "hard":
                m = train_step_hard_policy(model, opt, replay, args.batch_size, q_scale, args.type_loss_weight)
            else:
                m = train_step(model, opt, replay, args.batch_size, q_scale)
            if m:
                metrics.append(m)
            if args.seq_loss_weight > 0.0:
                sm = train_sequence_batch_step(model, opt, seq_replay, args.batch_size, args.seq_loss_weight, args.seq_search_pos_weight)
                if sm:
                    metrics.append(sm)
            if args.learned_slot_loss_weight > 0.0:
                lm = train_learned_slot_step(
                    model,
                    learned_head,
                    opt,
                    seq_replay,
                    args.batch_size,
                    args.learned_slot_loss_weight,
                    args.learned_slot_search_pos_weight,
                )
                if lm:
                    metrics.append(lm)
            if args.slot_attention_loss_weight > 0.0:
                am = train_learned_slot_step(
                    model,
                    attn_head,
                    opt,
                    seq_replay,
                    args.batch_size,
                    args.slot_attention_loss_weight,
                    args.slot_attention_search_pos_weight,
                )
                if am:
                    metrics.append({f"attn_{k}": v for k, v in am.items()})
        row = {
            "iteration": it,
            "collected_targets": len(targets),
            "collected_sequences": len(seq_targets),
            "replay_size": len(replay),
            "seq_replay_size": len(seq_replay),
            "q_scale": q_scale,
            "selfplay_reward": float(np.mean([w["reward"] for w in windows])) if windows else 0.0,
        }
        if metrics:
            keys = sorted({k for m in metrics for k in m})
            for key in keys:
                vals = [m[key] for m in metrics if key in m]
                row[key] = float(np.mean(vals))
        train_rows.append(row)
        print("mutual_alpha_train", json.dumps(row), flush=True)
        torch.save(model.cpu().state_dict(), ckpt)
        torch.save(learned_head.cpu().state_dict(), head_ckpt)
        torch.save(attn_head.cpu().state_dict(), attn_head_ckpt)
        model.to(DEVICE)
        learned_head.to(DEVICE)
        attn_head.to(DEVICE)
        pd.DataFrame(train_rows).to_csv(OUT / "mutual_alpha_train_log.csv", index=False)
        pd.DataFrame(selfplay_rows).to_csv(OUT / "mutual_alpha_selfplay_windows.csv", index=False)
        if it in set(args.eval_iters):
            evaluate_suite(model.eval(), args, tag=f"iter{it}", learned_head=learned_head.eval(), attn_head=attn_head.eval())

    shutil.copy2(OUT / "mutual_alpha_train_log.csv", CLEAN / "mutual_alpha_train_log.csv")
    return model.eval()


def evaluate_suite(model: MutualRadarNet, args, tag: str, learned_head: LearnedSlotSequenceHead | None = None, attn_head: SlotAttentionSequenceHead | None = None):
    cells = [(int(x), float(r)) for x in args.eval_initials.split(",") for r in args.eval_rates.split(",")]
    seeds = [int(x) for x in args.eval_seeds.split(",") if x]
    rows: List[dict] = []
    wins: List[pd.DataFrame] = []
    for init, rate in cells:
        env = configured_env(rate, args)
        methods = {
            "Mutual_PolicyArgmax_0r": lambda: MutualArgmaxPolicyPlanner(model, mode="policy"),
            "Mutual_QArgmax_0r": lambda: MutualArgmaxPolicyPlanner(model, mode="q"),
            "Mutual_BatchPolicyArgmax_0r": lambda: MutualBatchArgmaxPlanner(model, mode="policy"),
            "Mutual_BatchQArgmax_0r": lambda: MutualBatchArgmaxPlanner(model, mode="q"),
            f"Mutual_BatchQUrgency_lam{args.batch_q_urgency_weight:g}_0r": lambda: MutualBatchQUrgencyPlanner(
                model,
                urgency_weight=args.batch_q_urgency_weight,
                deadline_weight=args.batch_q_deadline_weight,
                overdue_weight=args.batch_q_overdue_weight,
            ),
            f"Mutual_MCTS_P_r{args.eval_rollouts}": lambda env=env: make_mcts(model, env, args, training=False, rollouts=args.eval_rollouts, mode="p"),
            f"Mutual_MCTS_PQ_r{args.eval_rollouts}": lambda env=env: make_mcts(model, env, args, training=False, rollouts=args.eval_rollouts, mode="pq"),
            f"Mutual_MCTS_PVQ_r{args.eval_rollouts}": lambda env=env: make_mcts(model, env, args, training=False, rollouts=args.eval_rollouts, mode="pvq"),
            "EDF": lambda: EDFPlanner(MAXT),
            "EST": lambda: ESTPlanner(MAXT),
        }
        if learned_head is not None:
            methods["Mutual_LearnedSlotBatch_0r"] = lambda learned_head=learned_head: MutualLearnedSlotBatchPlanner(model, learned_head)
        if attn_head is not None:
            methods["Mutual_SlotAttentionBatch_0r"] = lambda attn_head=attn_head: MutualLearnedSlotBatchPlanner(model, attn_head)
        for name, factory in methods.items():
            for seed in seeds:
                seedall(seed)
                t0 = time.perf_counter()
                w, _ = run_fixed(factory(), name, init, MAXT, seed, args.eval_windows, 200, env)
                s = summarize_window_df(w, "fixed")
                s.update(planner=name, initial_targets=init, rate=rate, seed=seed, tag=tag, wall_s=time.perf_counter() - t0)
                rows.append(s)
                ww = w.copy()
                ww["planner"] = name
                ww["initial_targets"] = init
                ww["rate"] = rate
                ww["seed"] = seed
                ww["tag"] = tag
                wins.append(ww)
                print("mutual_alpha_eval", tag, init, rate, name, round(s["reward_per_200ms_eq"], 3), round(s["planning_ms_per_200ms_eq"], 2), flush=True)

    raw = pd.DataFrame(rows)
    win = pd.concat(wins, ignore_index=True)
    raw_path = OUT / f"{tag}_eval_raw.csv"
    win_path = OUT / f"{tag}_eval_windows.csv"
    summary_path = OUT / f"{tag}_eval_summary.csv"
    raw.to_csv(raw_path, index=False)
    win.to_csv(win_path, index=False)
    summary = raw.groupby("planner", as_index=False).agg(
        reward=("reward_per_200ms_eq", "mean"),
        drop=("mean_drop_pct_active", "mean"),
        delay=("mean_delay_active", "mean"),
        search=("search_fraction", "mean"),
        latency=("planning_ms_per_200ms_eq", "mean"),
    ).sort_values("reward", ascending=False)
    summary.to_csv(summary_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    window_col = "window_idx" if "window_idx" in win.columns else "window"
    for name, g in win.groupby("planner"):
        curve = g.groupby(window_col)["window_reward"].mean().cumsum()
        ax.plot(curve.index * 0.2, curve.values, label=name, linewidth=2)
    ax.set_xlabel("Episode time (s)")
    ax.set_ylabel("Cumulative reward")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = OUT / f"{tag}_cumulative.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)
    for src in [summary_path, png]:
        shutil.copy2(src, CLEAN / f"mutual_alpha_{src.name}")
    print(summary.to_string(index=False), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=76)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--episodes-per-iter", type=int, default=2)
    ap.add_argument("--windows-per-episode", type=int, default=4)
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--train-mcts-mode", choices=["p", "pq", "pv", "pvq"], default="p")
    ap.add_argument("--eval-rollouts", type=int, default=16)
    ap.add_argument("--expand-top-k", type=int, default=10)
    ap.add_argument("--c-puct", type=float, default=1.25)
    ap.add_argument("--q-scale", type=float, default=100.0)
    ap.add_argument("--q-utility-weight", type=float, default=0.15)
    ap.add_argument("--leaf-value-mix", type=float, default=0.25)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--train-steps", type=int, default=32)
    ap.add_argument("--type-loss-weight", type=float, default=1.0)
    ap.add_argument("--policy-target", choices=["hard", "visit"], default="hard")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--seq-loss-weight", type=float, default=1.0)
    ap.add_argument("--seq-search-pos-weight", type=float, default=1.0)
    ap.add_argument("--learned-slot-loss-weight", type=float, default=1.0)
    ap.add_argument("--learned-slot-search-pos-weight", type=float, default=0.1)
    ap.add_argument("--slot-attention-loss-weight", type=float, default=1.0)
    ap.add_argument("--slot-attention-search-pos-weight", type=float, default=0.05)
    ap.add_argument("--slot-decoder-layers", type=int, default=2)
    ap.add_argument("--selfplay-replan-each-action", action="store_true")
    ap.add_argument("--target-selected-action", action="store_true")
    ap.add_argument("--replay-size", type=int, default=50000)
    ap.add_argument("--train-initials", default="15,50,75,100")
    ap.add_argument("--train-rates", default="2")
    ap.add_argument("--eval-initials", default="15,50,75,100")
    ap.add_argument("--eval-rates", default="2")
    ap.add_argument("--eval-seeds", default="82")
    ap.add_argument(
        "--env-mode",
        choices=[
            "current",
            "no_refresh",
            "operational",
            "original_reward",
            "radarxs_original",
            "radarxs_original_global",
            "radarxs_balanced",
            "radarxs_mission_delta",
            "repaired_stress_reward",
            "balanced_linear",
            "staleness_potential",
            "searched_sector_frame",
            "ding_moo_frame",
            "mcts_sched_v1",
        ],
        default="current",
    )
    ap.add_argument("--track-update-reward", type=float, default=0.30)
    ap.add_argument("--track-loss-penalty", type=float, default=3.0)
    ap.add_argument("--search-refresh-tracked", type=int, default=0)
    ap.add_argument("--search-refresh-gain", type=float, default=0.0)
    ap.add_argument("--search-debt-penalty-weight", type=float, default=0.00025)
    ap.add_argument("--sector-staleness-weight", type=float, default=0.003)
    ap.add_argument("--searched-sector-reward-weight", type=float, default=0.25)
    ap.add_argument("--search-frame-overdue-weight", type=float, default=0.10)
    ap.add_argument("--search-frame-desired-ms", type=float, default=3000.0)
    ap.add_argument("--search-frame-deadline-ms", type=float, default=4500.0)
    ap.add_argument("--search-frame-drop-penalty", type=float, default=4.0)
    ap.add_argument("--penalize-hidden-targets", type=int, default=0)
    ap.add_argument("--mcts-rollout-policy", choices=["greedy", "edf"], default="greedy")
    ap.add_argument("--mcts-rollout-search-period-ms", type=float, default=120.0)
    ap.add_argument(
        "--mcts-simulation-window-ms",
        type=float,
        default=200.0,
        help="Rollout horizon used inside MCTS. Executed window remains 200ms; larger values let self-play value surveillance/search-frame effects beyond the immediate window.",
    )
    ap.add_argument("--mcts-prior-uniform-mix", type=float, default=0.0)
    ap.add_argument("--belief-search-weight", type=float, default=0.0)
    ap.add_argument("--belief-search-cap", type=float, default=8.0)
    ap.add_argument("--batch-q-urgency-weight", type=float, default=8.0)
    ap.add_argument("--batch-q-deadline-weight", type=float, default=None)
    ap.add_argument("--batch-q-overdue-weight", type=float, default=None)
    ap.add_argument("--eval-windows", type=int, default=100)
    ap.add_argument("--eval-before", action="store_true")
    ap.add_argument("--eval-iters", type=int, nargs="*", default=[3])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    train_mutual_model(args)


if __name__ == "__main__":
    main()
