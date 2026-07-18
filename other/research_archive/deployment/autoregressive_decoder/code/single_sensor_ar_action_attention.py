from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "CreateValid1" / "experiments" / "code" / "model_code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from eval_action_attention_muzero_g import run_plan_eval, summarize
from exact_env_mutual import EDFPlanner, ESTPlanner, MAXT, engine_env_cfg, env_cfg_for, search_frame_pressure_sum, xs_decode_action, xs_s_search_action, xs_s_track_action
from foundation_mcts_fair_eval import parse_floats, parse_ints
from penalty_window_quota_learner_eval import make_exact_args
from two_sensor_physical_head_eval import PhysicalHeadPlanner, make_physical_model, physical_candidates
from clean_single_sensor_muzero_benchmark import load_muzero_model, select_factorized_accum_tensor
from literal_muzero_radar_smoke import OBS_DIM, observation_vector, ranked_rows, valid_target_rows


BASE_STATE = ROOT / "CreateValid1" / "results" / "mixed_gate_distill_180_action_attention_step40_state.pt"


class SingleSensorActionAttentionAR:
    """S-band-only autoregressive wrapper around the existing action-attention PQ scorer."""

    def __init__(
        self,
        base: PhysicalHeadPlanner,
        *,
        max_steps: int = 32,
        search_floor: int = 0,
        search_cap_frac: float = 1.0,
        search_score_bias: float = 0.0,
        env_cfg: dict | None = None,
        search_pressure_bias_weight: float = 0.0,
        search_pressure_bias_target: float = 0.0,
        search_active_bias_weight: float = 0.0,
        search_active_bias_target: float = 40.0,
        track_pressure_bias_weight: float = 0.0,
        disable_action_coupler: bool = False,
        action_coupler_top_k: int = 0,
        sparse_residuals: bool = False,
        single_sensor_action_only: bool = False,
        allow_soft_overshoot: bool = False,
        soft_overshoot_ms: float = -1.0,
    ):
        self.base = base
        self.max_steps = int(max_steps)
        self.search_floor = int(search_floor)
        self.search_cap_frac = float(search_cap_frac)
        self.search_score_bias = float(search_score_bias)
        self.env_cfg = dict(env_cfg or {})
        self.search_pressure_bias_weight = float(search_pressure_bias_weight)
        self.search_pressure_bias_target = float(search_pressure_bias_target)
        self.search_active_bias_weight = float(search_active_bias_weight)
        self.search_active_bias_target = float(search_active_bias_target)
        self.track_pressure_bias_weight = float(track_pressure_bias_weight)
        self.disable_action_coupler = bool(disable_action_coupler)
        self.action_coupler_top_k = int(action_coupler_top_k)
        self.sparse_residuals = bool(sparse_residuals)
        self.single_sensor_action_only = bool(single_sensor_action_only)
        self.allow_soft_overshoot = bool(allow_soft_overshoot)
        self.soft_overshoot_ms = float(soft_overshoot_ms)

    def _effective_remaining(self, elapsed: float, budget_ms: float) -> float:
        remaining = max(0.0, float(budget_ms) - float(elapsed))
        if not self.allow_soft_overshoot:
            return remaining
        if self.soft_overshoot_ms < 0.0:
            return float("inf")
        return remaining + max(0.0, self.soft_overshoot_ms)

    def _fits_window(self, elapsed: float, duration: float, budget_ms: float) -> bool:
        if float(elapsed) + float(duration) <= float(budget_ms):
            return True
        if not self.allow_soft_overshoot:
            return False
        if self.soft_overshoot_ms < 0.0:
            return True
        return float(elapsed) + float(duration) <= float(budget_ms) + max(0.0, self.soft_overshoot_ms)

    def warmup(self, obs: dict) -> None:
        for _ in range(3):
            self.plan(obs, budget_ms=200.0)
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    def _dynamic_search_bias(self, obs: dict) -> float:
        bias = float(self.search_score_bias)
        if self.search_pressure_bias_weight != 0.0:
            pressure = search_frame_pressure_sum(obs, self.env_cfg)
            bias += self.search_pressure_bias_weight * (float(pressure) - self.search_pressure_bias_target)
        if self.search_active_bias_weight != 0.0:
            active = float(np.asarray(obs.get("active_mask", []), dtype=bool).sum())
            bias -= self.search_active_bias_weight * max(0.0, active - self.search_active_bias_target) / max(1.0, self.search_active_bias_target)
        return bias

    def _track_pressure_bonus(self, obs: dict, rows: int) -> np.ndarray:
        bonus = np.zeros((int(rows),), dtype=np.float32)
        if self.track_pressure_bias_weight == 0.0 or rows <= 1:
            return bonus
        active = np.asarray(obs.get("active_mask", []), dtype=bool)
        desired = np.asarray(obs.get("t_desired", []), dtype=np.float32)
        deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
        dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        n = min(rows - 1, len(active), len(desired), len(deadline), len(dwell))
        if n <= 0:
            return bonus
        valid = active[:n] & (deadline[:n] >= 0.0)
        late = np.clip(-desired[:n] / 1000.0, 0.0, 4.0)
        slack = deadline[:n] - dwell[:n]
        deadline_risk = np.clip((500.0 - slack) / 500.0, 0.0, 2.0)
        pressure = np.where(valid, late + 2.0 * deadline_risk, 0.0)
        bonus[1 : n + 1] = self.track_pressure_bias_weight * pressure.astype(np.float32)
        return bonus

    def _apply_track_pressure_bonus(self, obs: dict, s_scores: np.ndarray) -> None:
        if self.track_pressure_bias_weight == 0.0:
            return
        s_scores[: len(s_scores)] += self._track_pressure_bonus(obs, len(s_scores))

    def _choose_row(self, obs: dict, selected: set[int], elapsed: float, search_count: int, track_count: int, last: int) -> int:
        scores = self.base.score_actions(
            obs,
            selected=selected,
            elapsed=float(elapsed),
            search_count=int(search_count),
            track_count=int(track_count),
            last=int(last),
        )
        s_scores = np.asarray(scores[:, 0], dtype=np.float32).copy()
        self._apply_track_pressure_bonus(obs, s_scores)
        remaining = self._effective_remaining(float(elapsed), 200.0)
        dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        if len(dwell) > 0:
            n = min(MAXT, len(dwell), s_scores.shape[0] - 1)
            too_long = np.asarray(dwell[:n], dtype=np.float32) > remaining
            bad_rows = np.nonzero(too_long)[0] + 1
            s_scores[bad_rows] = -1.0e9
        if remaining < 10.0:
            s_scores[0] = -1.0e9
        s_scores[0] += self._dynamic_search_bias(obs)
        max_search = max(self.search_floor, int(np.floor(self.search_cap_frac * max(1, self.max_steps))))
        if search_count < self.search_floor:
            return 0
        if search_count >= max_search:
            s_scores[0] = -1.0e9
        row = int(np.nanargmax(s_scores))
        return max(0, row)

    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        selected: set[int] = set()
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        plan: list[int] = []
        for _step in range(self.max_steps):
            if elapsed >= float(budget_ms):
                break
            row = self._choose_row(obs, selected, elapsed, search_count, track_count, last)
            if row <= 0:
                if elapsed + 10.0 > float(budget_ms) and plan and not self.allow_soft_overshoot:
                    break
                plan.append(xs_s_search_action(MAXT))
                search_count += 1
                elapsed += 10.0
                last = 0
            else:
                dt = float(max(1.0, dwell[row - 1] if row - 1 < len(dwell) else 5.0))
                if not self._fits_window(elapsed, dt, float(budget_ms)) and plan:
                    break
                plan.append(xs_s_track_action(row, MAXT))
                selected.add(int(row))
                track_count += 1
                elapsed += dt
                last = int(row)
        return plan if plan else [xs_s_search_action(MAXT)]


class CachedSingleSensorActionAttentionAR(SingleSensorActionAttentionAR):
    """S-band AR planner that reuses the root target encoding within a window."""

    def _scores_from_encoded(self, cls_out, tok_out, root_selected, token_active, slot_t, selected: set[int]):
        model = self.base.model
        slot_emb = model.backbone.slot_proj(slot_t)
        bsz, rows, _ = tok_out.shape

        sensor = model.sensor_embed[None, :, :].expand(bsz, -1, -1)
        cls_s = cls_out[:, None, :].expand(-1, 2, -1)
        slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
        sensor_state = model.sensor_state_proj(torch.cat([cls_s, slot_s, sensor], dim=-1))
        coupled_sensor = model.sensor_coupler(sensor_state)

        if self.single_sensor_action_only:
            cls_s0 = cls_out[:, None, :]
            slot_s0 = slot_emb[:, None, :]
            coupled_s0 = coupled_sensor[:, 0:1, :]
            type_ctx = torch.cat([cls_s0, slot_s0, coupled_s0], dim=-1)
            type_logits = model.type_head(type_ctx)[:, 0, :]
            type_q = model.type_q_head(type_ctx)[:, 0, :]

            tok_s = tok_out
            cls_t = cls_out[:, None, :].expand(-1, rows, -1)
            slot_tgt = slot_emb[:, None, :].expand(-1, rows, -1)
            sensor_tgt = coupled_sensor[:, None, 0, :].expand(-1, rows, -1)
            target_ctx = torch.cat([tok_s, cls_t, slot_tgt, sensor_tgt], dim=-1)
            target_logits = model.target_head(target_ctx).squeeze(-1)
            target_q = model.target_q_head(target_ctx).squeeze(-1)

            selected_t = root_selected.clone()
            for row in selected:
                if 0 <= int(row) < int(selected_t.shape[1]):
                    selected_t[0, int(row)] = True
            track_mask = token_active & ~selected_t
            track_mask[:, 0] = False
            row_is_search = torch.arange(rows, device=slot_t.device)[None, :] == 0
            valid = track_mask | row_is_search

            base_scores = slot_t.new_full((bsz, rows), -1e9)
            base_q = slot_t.new_zeros((bsz, rows))
            base_scores[:, 0] = type_logits[:, 0]
            base_q[:, 0] = type_q[:, 0]
            base_scores[:, 1:] = (type_logits[:, None, 1] + target_logits)[:, 1:]
            base_q[:, 1:] = (type_q[:, None, 1] + target_q)[:, 1:]

            if self.sparse_residuals and self.action_coupler_top_k > 0 and bsz == 1:
                rank_scores = base_scores[0].masked_fill(~valid[0], -1e9)
                keep = min(rows, max(1, int(self.action_coupler_top_k)) + 1)
                keep_rows = torch.topk(rank_scores, k=keep).indices.unique(sorted=True)
                action_ctx = model.action_proj(target_ctx[:, keep_rows, :])
                sparse_valid = valid[:, keep_rows]
                if not self.disable_action_coupler:
                    action_ctx = model.action_coupler(action_ctx, src_key_padding_mask=~sparse_valid)
                residual = slot_t.new_zeros((bsz, rows))
                q_residual = slot_t.new_zeros((bsz, rows))
                residual[:, keep_rows] = model.action_policy_residual(action_ctx).squeeze(-1)
                q_residual[:, keep_rows] = model.action_q_residual(action_ctx).squeeze(-1)
            else:
                action_ctx = model.action_proj(target_ctx)
                if not self.disable_action_coupler:
                    if self.action_coupler_top_k > 0 and bsz == 1:
                        rank_scores = base_scores[0].masked_fill(~valid[0], -1e9)
                        keep = min(rows, max(1, int(self.action_coupler_top_k)) + 1)
                        keep_rows = torch.topk(rank_scores, k=keep).indices.unique(sorted=True)
                        mixed = model.action_coupler(action_ctx[:, keep_rows, :], src_key_padding_mask=~valid[:, keep_rows])
                        action_ctx = action_ctx.clone()
                        action_ctx[:, keep_rows, :] = mixed
                    else:
                        action_ctx = model.action_coupler(action_ctx, src_key_padding_mask=~valid)
                residual = model.action_policy_residual(action_ctx).squeeze(-1)
                q_residual = model.action_q_residual(action_ctx).squeeze(-1)

            scores = (base_scores + residual).masked_fill(~valid, -1e9)
            q = (base_q + q_residual).masked_fill(~valid, 0.0)
            utility_s = self.base.policy_weight * scores + self.base.q_weight * q
            utility = slot_t.new_full((bsz, rows, 2), -1e9)
            utility[:, :, 0] = utility_s
            return utility

        type_ctx = torch.cat([cls_s, slot_s, coupled_sensor], dim=-1)
        type_logits = model.type_head(type_ctx)
        type_q = model.type_q_head(type_ctx)

        tok_st = tok_out[:, :, None, :].expand(-1, -1, 2, -1)
        cls_st = cls_out[:, None, None, :].expand(-1, rows, 2, -1)
        slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
        sensor_st = coupled_sensor[:, None, :, :].expand(bsz, rows, -1, -1)
        target_ctx = torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)
        target_logits = model.target_head(target_ctx).squeeze(-1)
        target_q = model.target_q_head(target_ctx).squeeze(-1)

        base_scores = slot_t.new_full((bsz, rows, 2), -1e9)
        base_q = slot_t.new_zeros((bsz, rows, 2))
        base_scores[:, 0, :] = type_logits[:, :, 0]
        base_q[:, 0, :] = type_q[:, :, 0]

        selected_t = root_selected.clone()
        for row in selected:
            if 0 <= int(row) < int(selected_t.shape[1]):
                selected_t[0, int(row)] = True
        track_mask = token_active & ~selected_t
        track_mask[:, 0] = False
        base_scores[:, 1:, :] = (type_logits[:, None, :, 1] + target_logits)[:, 1:, :]
        base_q[:, 1:, :] = (type_q[:, None, :, 1] + target_q)[:, 1:, :]

        row_is_search = torch.arange(rows, device=slot_t.device)[None, :, None] == 0
        valid = (track_mask[:, :, None] | row_is_search).expand(-1, -1, 2)
        if self.sparse_residuals and self.action_coupler_top_k > 0 and bsz == 1:
            rank_scores = base_scores[0, :, 0].masked_fill(~valid[0, :, 0], -1e9)
            keep = min(rows, max(1, int(self.action_coupler_top_k)) + 1)
            keep_rows = torch.topk(rank_scores, k=keep).indices.unique(sorted=True)
            sparse_ctx = target_ctx[:, keep_rows, :, :]
            sparse_valid = valid[:, keep_rows, :]
            action_ctx = model.action_proj(sparse_ctx).reshape(bsz, int(keep_rows.numel()) * 2, -1)
            if not self.disable_action_coupler:
                action_ctx = model.action_coupler(action_ctx, src_key_padding_mask=~sparse_valid.reshape(bsz, int(keep_rows.numel()) * 2))
            residual = slot_t.new_zeros((bsz, rows, 2))
            q_residual = slot_t.new_zeros((bsz, rows, 2))
            residual[:, keep_rows, :] = model.action_policy_residual(action_ctx).reshape(bsz, int(keep_rows.numel()), 2)
            q_residual[:, keep_rows, :] = model.action_q_residual(action_ctx).reshape(bsz, int(keep_rows.numel()), 2)
        else:
            action_ctx = model.action_proj(target_ctx).reshape(bsz, rows * 2, -1)
            if not self.disable_action_coupler:
                if self.action_coupler_top_k > 0 and bsz == 1:
                    rank_scores = base_scores[0, :, 0].masked_fill(~valid[0, :, 0], -1e9)
                    keep = min(rows, max(1, int(self.action_coupler_top_k)) + 1)
                    keep_rows = torch.topk(rank_scores, k=keep).indices.unique(sorted=True)
                    flat_idx = torch.stack([2 * keep_rows, 2 * keep_rows + 1], dim=1).reshape(-1)
                    mixed = model.action_coupler(action_ctx[:, flat_idx, :], src_key_padding_mask=~valid.reshape(bsz, rows * 2)[:, flat_idx])
                    action_ctx = action_ctx.clone()
                    action_ctx[:, flat_idx, :] = mixed
                else:
                    action_ctx = model.action_coupler(action_ctx, src_key_padding_mask=~valid.reshape(bsz, rows * 2))
            residual = model.action_policy_residual(action_ctx).reshape(bsz, rows, 2)
            q_residual = model.action_q_residual(action_ctx).reshape(bsz, rows, 2)
        scores = (base_scores + residual).masked_fill(~valid, -1e9)
        q = (base_q + q_residual).masked_fill(~valid, 0.0)
        utility = self.base.policy_weight * scores + self.base.q_weight * q
        return utility

    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        from exact_env_mutual import attach_env_obs
        from mutual_features import slot_features, tokenize

        obs = attach_env_obs(obs, self.base.env_cfg, True, True)
        root_tok = tokenize(self.base.adapt, obs, selected=set(), search_count=0).astype(np.float32)
        device = next(self.base.model.parameters()).device
        dwell_np = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        dwell_t = torch.from_numpy(dwell_np).to(device) if len(dwell_np) > 0 else None
        with torch.inference_mode():
            root_x = torch.from_numpy(root_tok).float().unsqueeze(0).to(device)
            cls_out, tok_out, root_selected, token_active = self.base.model.backbone.encode_tokens(root_x)

        selected: set[int] = set()
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        plan: list[int] = []
        while elapsed < float(budget_ms) and len(plan) < self.max_steps:
            slot = slot_features(obs, elapsed, search_count, track_count, last, float(budget_ms)).astype(np.float32)
            with torch.inference_mode():
                slot_t = torch.from_numpy(slot).float().unsqueeze(0).to(device)
                score = self._scores_from_encoded(cls_out, tok_out, root_selected, token_active, slot_t, selected)
                s_scores_raw = score[0, :, 0]
            s_scores_t = s_scores_raw.clone()
            if self.track_pressure_bias_weight != 0.0:
                bonus = torch.from_numpy(self._track_pressure_bonus(obs, int(s_scores_t.shape[0]))).to(device)
                s_scores_t = s_scores_t + bonus
            remaining = self._effective_remaining(float(elapsed), float(budget_ms))
            if dwell_t is not None and dwell_t.numel() > 0:
                n = min(MAXT, int(dwell_t.numel()), int(s_scores_t.shape[0]) - 1)
                if n > 0:
                    too_long = dwell_t[:n] > float(remaining)
                    s_scores_t[1 : n + 1] = s_scores_t[1 : n + 1].masked_fill(too_long, -1.0e9)
            if remaining < 10.0:
                s_scores_t[0] = -1.0e9
            s_scores_t[0] += float(self._dynamic_search_bias(obs))
            max_search = max(self.search_floor, int(np.floor(self.search_cap_frac * max(1, self.max_steps))))
            if search_count < self.search_floor:
                row = 0
            else:
                if search_count >= max_search:
                    s_scores_t[0] = -1.0e9
                row = int(torch.argmax(s_scores_t).item())
            if row <= 0:
                if not self._fits_window(elapsed, 10.0, float(budget_ms)) and plan:
                    break
                plan.append(xs_s_search_action(MAXT))
                search_count += 1
                elapsed += 10.0
                last = 0
            else:
                dt = float(max(1.0, dwell_np[row - 1] if row - 1 < len(dwell_np) else 5.0))
                if not self._fits_window(elapsed, dt, float(budget_ms)) and plan:
                    break
                plan.append(xs_s_track_action(row, MAXT))
                selected.add(int(row))
                track_count += 1
                elapsed += dt
                last = int(row)
        return plan if plan else [xs_s_search_action(MAXT)]


class HistoryVectorSingleSensorActionAttentionAR(CachedSingleSensorActionAttentionAR):
    """Cached AR planner that conditions the slot context on action history.

    This is a compatibility ablation for existing checkpoints. The model still
    expects the original 11 slot features, so we encode the previous in-window
    action history into the three mutable schedule fields instead of changing
    the network input size:

    slot[1] = fraction of previous actions that were search
    slot[2] = fraction of distinct targets selected in history
    slot[3] = recency-weighted search fraction over the history vector

    A true history-array model should add a learned history projection and be
    retrained; this class is a fast first test of the idea.
    """

    def __init__(self, *args, history_len: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_len = max(1, int(history_len))

    def _history_slot(self, slot: np.ndarray, history: list[int], selected: set[int]) -> np.ndarray:
        out = np.asarray(slot, dtype=np.float32).copy()
        if not history:
            out[1] = 0.0
            out[2] = 0.0
            out[3] = 0.0
            return out
        hist = np.asarray(history[-self.history_len :], dtype=np.int64)
        search = (hist <= 0).astype(np.float32)
        out[1] = float(search.mean())
        out[2] = float(len(selected)) / float(max(1, MAXT))
        weights = np.arange(1, len(search) + 1, dtype=np.float32)
        out[3] = float(np.sum(search * weights) / max(1.0, float(np.sum(weights))))
        return out

    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        from exact_env_mutual import attach_env_obs
        from mutual_features import slot_features, tokenize

        obs = attach_env_obs(obs, self.base.env_cfg, True, True)
        root_tok = tokenize(self.base.adapt, obs, selected=set(), search_count=0).astype(np.float32)
        device = next(self.base.model.parameters()).device
        dwell_np = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        dwell_t = torch.from_numpy(dwell_np).to(device) if len(dwell_np) > 0 else None
        with torch.inference_mode():
            root_x = torch.from_numpy(root_tok).float().unsqueeze(0).to(device)
            cls_out, tok_out, root_selected, token_active = self.base.model.backbone.encode_tokens(root_x)

        selected: set[int] = set()
        history: list[int] = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        plan: list[int] = []
        while elapsed < float(budget_ms) and len(plan) < self.max_steps:
            slot = slot_features(obs, elapsed, search_count, track_count, last, float(budget_ms)).astype(np.float32)
            slot = self._history_slot(slot, history, selected)
            with torch.inference_mode():
                slot_t = torch.from_numpy(slot).float().unsqueeze(0).to(device)
                score = self._scores_from_encoded(cls_out, tok_out, root_selected, token_active, slot_t, selected)
                s_scores_raw = score[0, :, 0]
            s_scores_t = s_scores_raw.clone()
            if self.track_pressure_bias_weight != 0.0:
                bonus = torch.from_numpy(self._track_pressure_bonus(obs, int(s_scores_t.shape[0]))).to(device)
                s_scores_t = s_scores_t + bonus
            remaining = self._effective_remaining(float(elapsed), float(budget_ms))
            if dwell_t is not None and dwell_t.numel() > 0:
                n = min(MAXT, int(dwell_t.numel()), int(s_scores_t.shape[0]) - 1)
                if n > 0:
                    too_long = dwell_t[:n] > float(remaining)
                    s_scores_t[1 : n + 1] = s_scores_t[1 : n + 1].masked_fill(too_long, -1.0e9)
            if remaining < 10.0:
                s_scores_t[0] = -1.0e9
            s_scores_t[0] += float(self._dynamic_search_bias(obs))
            max_search = max(self.search_floor, int(np.floor(self.search_cap_frac * max(1, self.max_steps))))
            if search_count < self.search_floor:
                row = 0
            else:
                if search_count >= max_search:
                    s_scores_t[0] = -1.0e9
                row = int(torch.argmax(s_scores_t).item())
            if row <= 0:
                if not self._fits_window(elapsed, 10.0, float(budget_ms)) and plan:
                    break
                plan.append(xs_s_search_action(MAXT))
                history.append(0)
                search_count += 1
                elapsed += 10.0
                last = 0
            else:
                dt = float(max(1.0, dwell_np[row - 1] if row - 1 < len(dwell_np) else 5.0))
                if not self._fits_window(elapsed, dt, float(budget_ms)) and plan:
                    break
                plan.append(xs_s_track_action(row, MAXT))
                history.append(int(row))
                selected.add(int(row))
                track_count += 1
                elapsed += dt
                last = int(row)
        return plan if plan else [xs_s_search_action(MAXT)]


class CudaGraphSingleSensorActionAttentionAR(CachedSingleSensorActionAttentionAR):
    """CUDA-graph S-only action-attention planner.

    This keeps the corrected S-only action-attention heads, but captures the
    fixed-shape root encode plus in-window decode loop. Startup graph capture is
    excluded through warmup(), so measured latency is steady-state deployment
    latency.
    """

    def __init__(self, *args, budget_ms: float = 200.0, **kwargs):
        kwargs["single_sensor_action_only"] = True
        super().__init__(*args, **kwargs)
        device = next(self.base.model.parameters()).device
        if device.type != "cuda":
            raise ValueError("CudaGraphSingleSensorActionAttentionAR requires CUDA")
        from mutual_features import TOKEN_DIM

        self.device = device
        self.budget_ms = float(budget_ms)
        self.root_x = torch.empty((1, MAXT + 1, TOKEN_DIM), dtype=torch.float32, device=device)
        self.dwell = torch.empty((MAXT + 1,), dtype=torch.float32, device=device)
        self.slot_const = torch.empty((11,), dtype=torch.float32, device=device)
        self.rows = torch.empty((self.max_steps,), dtype=torch.long, device=device)
        self.active_steps = torch.empty((self.max_steps,), dtype=torch.bool, device=device)
        self.selected_buf = torch.empty((MAXT + 1,), dtype=torch.bool, device=device)
        self.elapsed_buf = torch.empty((), dtype=torch.float32, device=device)
        self.search_count_buf = torch.empty((), dtype=torch.float32, device=device)
        self.track_count_buf = torch.empty((), dtype=torch.float32, device=device)
        self.last_search_buf = torch.empty((), dtype=torch.float32, device=device)
        self.slot_buf = torch.empty((11,), dtype=torch.float32, device=device)
        self.budget_buf = torch.full((), float(self.budget_ms), dtype=torch.float32, device=device)
        self.zero_scalar = torch.zeros((), dtype=torch.float32, device=device)
        self.neg_one_long = torch.full((), -1, dtype=torch.long, device=device)
        self.graph = torch.cuda.CUDAGraph()
        self._captured = False

    def _obs_to_buffers(self, obs: dict) -> None:
        from exact_env_mutual import attach_env_obs
        from mutual_features import slot_features, tokenize

        obs = attach_env_obs(obs, self.base.env_cfg, True, True)
        obs["enable_x_band"] = 0.0
        root_tok = tokenize(self.base.adapt, obs, selected=set(), search_count=0).astype(np.float32)
        dwell_np = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        dwell_full = np.ones((MAXT + 1,), dtype=np.float32) * 10.0
        n = min(MAXT, int(dwell_np.size))
        if n > 0:
            dwell_full[1 : n + 1] = np.maximum(1.0, dwell_np[:n])
        slot = slot_features(obs, 0.0, 0, 0, -1, self.budget_ms).astype(np.float32)
        self.root_x[0].copy_(torch.from_numpy(root_tok), non_blocking=True)
        self.dwell.copy_(torch.from_numpy(dwell_full), non_blocking=True)
        self.slot_const.copy_(torch.from_numpy(slot), non_blocking=True)

    def _score_graph(self, cls_out, tok_out, root_selected, token_active, slot_t, selected):
        model = self.base.model
        slot_emb = model.backbone.slot_proj(slot_t)
        bsz, rows, _ = tok_out.shape

        sensor = model.sensor_embed[None, :, :].expand(bsz, -1, -1)
        cls_s = cls_out[:, None, :].expand(-1, 2, -1)
        slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
        sensor_state = model.sensor_state_proj(torch.cat([cls_s, slot_s, sensor], dim=-1))
        coupled_sensor = model.sensor_coupler(sensor_state)

        cls_s0 = cls_out[:, None, :]
        slot_s0 = slot_emb[:, None, :]
        coupled_s0 = coupled_sensor[:, 0:1, :]
        type_ctx = torch.cat([cls_s0, slot_s0, coupled_s0], dim=-1)
        type_logits = model.type_head(type_ctx)[:, 0, :]
        type_q = model.type_q_head(type_ctx)[:, 0, :]

        cls_t = cls_out[:, None, :].expand(-1, rows, -1)
        slot_tgt = slot_emb[:, None, :].expand(-1, rows, -1)
        sensor_tgt = coupled_sensor[:, None, 0, :].expand(-1, rows, -1)
        target_ctx = torch.cat([tok_out, cls_t, slot_tgt, sensor_tgt], dim=-1)
        target_logits = model.target_head(target_ctx).squeeze(-1)
        target_q = model.target_q_head(target_ctx).squeeze(-1)

        selected_t = root_selected | selected.reshape(1, -1)
        track_mask = token_active & ~selected_t
        track_mask[:, 0] = False
        row_is_search = torch.arange(rows, device=slot_t.device)[None, :] == 0
        valid = track_mask | row_is_search

        base_scores = slot_t.new_full((bsz, rows), -1.0e9)
        base_q = slot_t.new_zeros((bsz, rows))
        base_scores[:, 0] = type_logits[:, 0]
        base_q[:, 0] = type_q[:, 0]
        base_scores[:, 1:] = (type_logits[:, None, 1] + target_logits)[:, 1:]
        base_q[:, 1:] = (type_q[:, None, 1] + target_q)[:, 1:]

        action_ctx = model.action_proj(target_ctx)
        if not self.disable_action_coupler:
            action_ctx = model.action_coupler(action_ctx, src_key_padding_mask=~valid)
        residual = model.action_policy_residual(action_ctx).squeeze(-1)
        q_residual = model.action_q_residual(action_ctx).squeeze(-1)
        scores = (base_scores + residual).masked_fill(~valid, -1.0e9)
        q = (base_q + q_residual).masked_fill(~valid, 0.0)
        return self.base.policy_weight * scores[0] + self.base.q_weight * q[0]

    def _body(self) -> None:
        cls_out, tok_out, root_selected, token_active = self.base.model.backbone.encode_tokens(self.root_x)
        selected = self.selected_buf.zero_()
        elapsed = self.elapsed_buf.zero_()
        search_count = self.search_count_buf.zero_()
        track_count = self.track_count_buf.zero_()
        last_search = self.last_search_buf.zero_()
        budget = self.budget_buf

        for step in range(self.max_steps):
            slot = self.slot_buf.copy_(self.slot_const)
            slot[0] = elapsed / budget
            slot[1] = search_count / 20.0
            slot[2] = track_count / 100.0
            slot[3] = last_search
            scores = self._score_graph(cls_out, tok_out, root_selected, token_active, slot.unsqueeze(0), selected)

            remaining = torch.clamp(budget - elapsed, min=0.0)
            remaining_safe = torch.clamp(remaining + 0.001, min=0.0)
            valid = self.dwell <= remaining_safe
            valid = valid & (~selected)
            valid[0] = valid[0] & (remaining >= 9.999)
            valid = valid & (scores > -1.0e8)
            scores = scores.masked_fill(~valid, -1.0e9)
            scores[0] = scores[0] + float(self.search_score_bias)
            any_valid = valid.any()
            row = torch.argmax(scores)
            row = torch.where(any_valid, row, torch.zeros_like(row))
            dt = self.dwell.gather(0, row.reshape(1))[0]

            self.rows[step] = torch.where(any_valid, row, self.neg_one_long)
            self.active_steps[step] = any_valid
            is_search = (row == 0) & any_valid
            is_track = (row > 0) & any_valid
            elapsed = elapsed + torch.where(any_valid, dt, self.zero_scalar)
            search_count = search_count + is_search.to(torch.float32)
            track_count = track_count + is_track.to(torch.float32)
            last_search = is_search.to(torch.float32)
            selected = selected | (torch.nn.functional.one_hot(row, num_classes=MAXT + 1).to(torch.bool) & is_track)

    def capture(self, obs: dict) -> None:
        self._obs_to_buffers(obs)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        with torch.cuda.graph(self.graph):
            self._body()
        self._captured = True

    def warmup(self, obs: dict) -> None:
        if not self._captured:
            self.capture(obs)
        for _ in range(3):
            self._obs_to_buffers(obs)
            self.graph.replay()
        torch.cuda.synchronize(self.device)

    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        if abs(float(budget_ms) - self.budget_ms) > 1e-6:
            raise ValueError("CUDA graph action-attention planner requires fixed budget_ms")
        self._obs_to_buffers(obs)
        if not self._captured:
            self.capture(obs)
        self.graph.replay()
        rows = self.rows.detach().cpu().tolist()
        plan: list[int] = []
        for row in rows:
            row = int(row)
            if row < 0:
                break
            plan.append(xs_s_search_action(MAXT) if row <= 0 else xs_s_track_action(row, MAXT))
        return plan if plan else [xs_s_search_action(MAXT)]


class _PassthroughActionCoupler(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class CudaGraphReencodeSingleSensorPQ(SingleSensorActionAttentionAR):
    """Faithful original-PQ S-only planner with graphed per-step scoring.

    Unlike the cached/one-encode planner, this rebuilds tokens with the current
    selected set and re-runs the transformer for every in-window action. The
    CUDA graph only captures the fixed-shape forward_scores call.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if bool(getattr(self, "disable_action_coupler", False)):
            self.base.model.action_coupler = _PassthroughActionCoupler()
        device = next(self.base.model.parameters()).device
        if device.type != "cuda":
            raise ValueError("CudaGraphReencodeSingleSensorPQ requires CUDA")
        from mutual_features import SLOT_DIM, TOKEN_DIM

        self.device = device
        self.root_x = torch.empty((1, MAXT + 1, TOKEN_DIM), dtype=torch.float32, device=device)
        self.slot_t = torch.empty((1, SLOT_DIM), dtype=torch.float32, device=device)
        self.utility = torch.empty((1, MAXT + 1, 2), dtype=torch.float32, device=device)
        self.graph = torch.cuda.CUDAGraph()
        self._captured = False

    def _capture(self, obs: dict) -> None:
        from exact_env_mutual import attach_env_obs
        from mutual_features import slot_features, tokenize

        obs = attach_env_obs(obs, self.base.env_cfg, True, True)
        tok = tokenize(self.base.adapt, obs, selected=set(), search_count=0).astype(np.float32)
        slot = slot_features(obs, 0.0, 0, 0, -1, 200.0).astype(np.float32)
        self.root_x.copy_(torch.from_numpy(tok).to(self.device).unsqueeze(0), non_blocking=True)
        self.slot_t.copy_(torch.from_numpy(slot).to(self.device).unsqueeze(0), non_blocking=True)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                scores, q = self.base.model.forward_scores(self.root_x, self.slot_t)
                self.utility.copy_(self.base.policy_weight * scores + self.base.q_weight * q)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        with torch.cuda.graph(self.graph):
            scores, q = self.base.model.forward_scores(self.root_x, self.slot_t)
            self.utility.copy_(self.base.policy_weight * scores + self.base.q_weight * q)
        self._captured = True

    def warmup(self, obs: dict) -> None:
        if not self._captured:
            self._capture(obs)
        for _ in range(3):
            self.graph.replay()
        torch.cuda.synchronize(self.device)

    def _score_actions(self, obs: dict, selected: set[int], elapsed: float, search_count: int, track_count: int, last: int) -> np.ndarray:
        from exact_env_mutual import attach_env_obs
        from mutual_features import slot_features, tokenize

        obs = attach_env_obs(obs, self.base.env_cfg, True, True)
        tok = tokenize(self.base.adapt, obs, selected=set(selected), search_count=int(search_count)).astype(np.float32)
        slot = slot_features(obs, float(elapsed), int(search_count), int(track_count), int(last), 200.0).astype(np.float32)
        self.root_x.copy_(torch.from_numpy(tok).to(self.device).unsqueeze(0), non_blocking=True)
        self.slot_t.copy_(torch.from_numpy(slot).to(self.device).unsqueeze(0), non_blocking=True)
        if not self._captured:
            self._capture(obs)
            self.root_x.copy_(torch.from_numpy(tok).to(self.device).unsqueeze(0), non_blocking=True)
            self.slot_t.copy_(torch.from_numpy(slot).to(self.device).unsqueeze(0), non_blocking=True)
        self.graph.replay()
        score = self.utility.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=True)
        score[0, :] += float(self.base.search_score_bias)
        return score

    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        from exact_env_mutual import attach_env_obs

        obs = attach_env_obs(obs, self.base.env_cfg, True, True)
        selected: set[int] = set()
        plan: list[int] = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        while elapsed < float(budget_ms) and len(plan) < int(self.max_steps):
            score = self._score_actions(obs, selected, elapsed, search_count, track_count, last)
            best_row = -1
            best_score = -np.inf
            for action in physical_candidates(obs, top_k=MAXT):
                base, sensor = xs_decode_action(int(action), MAXT)
                if sensor not in (None, 0) or int(base) < 0:
                    continue
                row = int(base)
                if row in selected:
                    continue
                val = float(score[row, 0])
                if val > best_score:
                    best_row = row
                    best_score = val
            if best_row < 0:
                break
            if best_row <= 0:
                dt = 10.0
                if not self._fits_window(elapsed, dt, float(budget_ms)) and plan:
                    break
                plan.append(xs_s_search_action(MAXT))
                search_count += 1
            else:
                dt = float(max(1.0, dwell[best_row - 1] if best_row - 1 < len(dwell) else 10.0))
                if not self._fits_window(elapsed, dt, float(budget_ms)) and plan:
                    break
                plan.append(xs_s_track_action(best_row, MAXT))
                selected.add(int(best_row))
                track_count += 1
            elapsed += max(1.0, float(dt))
            last = int(best_row)
        return plan if plan else [xs_s_search_action(MAXT)]


class FastSequenceSingleSensorActionAttentionAR(CachedSingleSensorActionAttentionAR):
    """Batched S-only sequence decoder.

    The corrected AR scorer remains the quality reference. This path keeps the
    same root encoding and heads, then scores all candidate schedule slots in a
    small tensor batch so decoding does not call the network once per action.
    """

    def __init__(self, *args, refine_passes: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.refine_passes = max(0, int(refine_passes))

    def _slots_from_prefix(self, obs: dict, prefix: list[int], budget_ms: float) -> np.ndarray:
        from mutual_features import slot_features

        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        slots: list[np.ndarray] = []
        step_guess_ms = max(1.0, float(budget_ms) / max(1, int(self.max_steps)))
        for step in range(int(self.max_steps)):
            slots.append(slot_features(obs, elapsed, search_count, track_count, last, float(budget_ms)).astype(np.float32))
            if step < len(prefix):
                row, _sensor = xs_decode_action(int(prefix[step]), MAXT)
                if row <= 0:
                    elapsed += 10.0
                    search_count += 1
                    last = 0
                else:
                    dt = float(max(1.0, dwell[row - 1] if row - 1 < len(dwell) else 5.0))
                    elapsed += dt
                    track_count += 1
                    last = int(row)
            else:
                elapsed = min(float(budget_ms), elapsed + step_guess_ms)
                track_count += 1
        return np.stack(slots, axis=0)

    def _score_slot_batch(self, cls_out, tok_out, root_selected, token_active, slots: np.ndarray) -> torch.Tensor:
        device = next(self.base.model.parameters()).device
        with torch.inference_mode():
            slot_t = torch.from_numpy(slots).float().to(device)
            bsz = int(slot_t.shape[0])
            score = self._scores_from_encoded(
                cls_out.expand(bsz, -1),
                tok_out.expand(bsz, -1, -1),
                root_selected.expand(bsz, -1),
                token_active.expand(bsz, -1),
                slot_t,
                selected=set(),
            )
            return score[:, :, 0].clone()

    def _decode_scores(self, obs: dict, score_seq: torch.Tensor, budget_ms: float) -> list[int]:
        device = score_seq.device
        dwell_np = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        dwell_t = torch.from_numpy(dwell_np).to(device) if len(dwell_np) > 0 else None
        active_np = np.asarray(obs.get("active_mask", []), dtype=bool)
        deadline_np = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
        active_t = torch.from_numpy(active_np).to(device) if len(active_np) > 0 else None
        deadline_t = torch.from_numpy(deadline_np).to(device) if len(deadline_np) > 0 else None

        selected: set[int] = set()
        elapsed = 0.0
        search_count = 0
        plan: list[int] = []
        max_search = max(self.search_floor, int(np.floor(self.search_cap_frac * max(1, self.max_steps))))
        for step in range(min(int(self.max_steps), int(score_seq.shape[0]))):
            if elapsed >= float(budget_ms):
                break
            scores = score_seq[step].clone()
            rows = int(scores.shape[0])
            if self.track_pressure_bias_weight != 0.0:
                scores = scores + torch.from_numpy(self._track_pressure_bonus(obs, rows)).to(device)
            remaining = self._effective_remaining(float(elapsed), float(budget_ms))
            if active_t is not None:
                n = min(MAXT, int(active_t.numel()), rows - 1)
                if n > 0:
                    invalid = ~active_t[:n]
                    scores[1 : n + 1] = scores[1 : n + 1].masked_fill(invalid, -1.0e9)
            if deadline_t is not None:
                n = min(MAXT, int(deadline_t.numel()), rows - 1)
                if n > 0:
                    invalid = deadline_t[:n] < 0.0
                    scores[1 : n + 1] = scores[1 : n + 1].masked_fill(invalid, -1.0e9)
            if selected:
                idx = torch.tensor([r for r in selected if 0 <= int(r) < rows], dtype=torch.long, device=device)
                if idx.numel() > 0:
                    scores[idx] = -1.0e9
            if dwell_t is not None and dwell_t.numel() > 0:
                n = min(MAXT, int(dwell_t.numel()), rows - 1)
                if n > 0:
                    too_long = dwell_t[:n] > float(remaining)
                    scores[1 : n + 1] = scores[1 : n + 1].masked_fill(too_long, -1.0e9)
            if remaining < 10.0:
                scores[0] = -1.0e9
            scores[0] += float(self._dynamic_search_bias(obs))
            if search_count < self.search_floor:
                row = 0
            else:
                if search_count >= max_search:
                    scores[0] = -1.0e9
                row = int(torch.argmax(scores).item())
            if row <= 0:
                if not self._fits_window(elapsed, 10.0, float(budget_ms)) and plan:
                    break
                plan.append(xs_s_search_action(MAXT))
                search_count += 1
                elapsed += 10.0
            else:
                dt = float(max(1.0, dwell_np[row - 1] if row - 1 < len(dwell_np) else 5.0))
                if not self._fits_window(elapsed, dt, float(budget_ms)) and plan:
                    break
                plan.append(xs_s_track_action(row, MAXT))
                selected.add(int(row))
                elapsed += dt
        return plan if plan else [xs_s_search_action(MAXT)]

    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        from exact_env_mutual import attach_env_obs
        from mutual_features import tokenize

        obs = attach_env_obs(obs, self.base.env_cfg, True, True)
        root_tok = tokenize(self.base.adapt, obs, selected=set(), search_count=0).astype(np.float32)
        device = next(self.base.model.parameters()).device
        with torch.inference_mode():
            root_x = torch.from_numpy(root_tok).float().unsqueeze(0).to(device)
            cls_out, tok_out, root_selected, token_active = self.base.model.backbone.encode_tokens(root_x)

        prefix: list[int] = []
        for _pass in range(self.refine_passes + 1):
            slots = self._slots_from_prefix(obs, prefix, float(budget_ms))
            score_seq = self._score_slot_batch(cls_out, tok_out, root_selected, token_active, slots)
            prefix = self._decode_scores(obs, score_seq, float(budget_ms))
        return prefix if prefix else [xs_s_search_action(MAXT)]


class SingleSensorHeuristicAdapter:
    def __init__(self, planner):
        self.planner = planner

    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        out = []
        for a in list(self.planner.plan(obs, budget_ms=budget_ms))[:64]:
            a = int(a)
            if a == 0:
                out.append(xs_s_search_action(MAXT))
            else:
                out.append(xs_s_track_action(a, MAXT))
        return out


class MuZeroLatentAccumulatorPlanAdapter:
    """Full-window S-only MuZero planner evaluated through run_plan_eval.

    The clean MuZero benchmark steps its own LiteralRadarGame one action at a
    time. For fair comparison with action-attention/heuristics, this adapter
    instead produces a complete 200 ms physical plan and lets run_plan_eval own
    execution, reward shaping, and metrics.
    """

    def __init__(
        self,
        model,
        cfg,
        *,
        env_cfg: dict,
        max_steps: int = 32,
        total_windows: int = 100,
        rank_action_space: bool | None = None,
    ):
        self.model = model.eval()
        self.cfg = cfg
        self.env_cfg = dict(env_cfg)
        self.max_steps = int(max_steps)
        self.total_windows = int(max(1, total_windows))
        self.rank_action_space = bool(getattr(cfg, "rank_action_space", True) if rank_action_space is None else rank_action_space)
        self.window_index = 0
        self.estimated_debt_ms = 0.0

    def _ranked_action_rows(self, obs: dict, action_count: int) -> list[int]:
        if self.rank_action_space:
            return ranked_rows(obs)[: max(0, action_count - 1)]
        rows = [int(r) for r in valid_target_rows(obs) if 1 <= int(r) < action_count]
        return rows[: max(0, action_count - 1)]

    def _dwell_for_actions(self, obs: dict, action_count: int, rows: list[int]) -> np.ndarray:
        dwell_np = np.full((action_count,), 10.0, dtype=np.float32)
        dwell = np.asarray(obs.get("t_dwell", np.ones(MAXT, dtype=np.float32) * 10.0), dtype=np.float32)
        for a in range(1, action_count):
            row = rows[a - 1] if a - 1 < len(rows) else -1
            if 1 <= int(row) <= int(dwell.shape[0]):
                dwell_np[a] = float(max(1.0, dwell[int(row) - 1]))
        return dwell_np

    def warmup(self, obs: dict) -> None:
        # Do not advance the logical window counter during warmup.
        old_window = self.window_index
        old_debt = self.estimated_debt_ms
        for _ in range(2):
            self.plan(obs, budget_ms=200.0)
        self.window_index = old_window
        self.estimated_debt_ms = old_debt
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        from exact_env_mutual import attach_env_obs

        obs = attach_env_obs(obs, self.env_cfg, True, True)
        device = next(self.model.parameters()).device
        action_count = int(len(self.cfg.action_space))
        rows = self._ranked_action_rows(obs, action_count)
        dwell_np = self._dwell_for_actions(obs, action_count, rows)
        dwell_t = torch.as_tensor(dwell_np, dtype=torch.float32, device=device)
        selected = torch.zeros((action_count,), dtype=torch.bool, device=device)
        remaining = torch.tensor(float(budget_ms), dtype=torch.float32, device=device)
        credit = torch.tensor(0.0, dtype=torch.float32, device=device)
        out: list[int] = []
        local_debt = float(self.estimated_debt_ms)
        with torch.inference_mode():
            root_vec = observation_vector(obs, local_debt, 0.0, int(self.window_index), int(self.total_windows))
            x = torch.as_tensor(root_vec, dtype=torch.float32, device=device).view(1, 1, 1, OBS_DIM)
            _value, _reward, _policy, hidden = self.model.initial_inference(x)
            for _slot in range(min(self.max_steps, action_count)):
                valid = (~selected) & (dwell_t <= remaining)
                valid[0] = remaining >= 10.0
                if not bool(valid.any().item()):
                    break
                decoded = select_factorized_accum_tensor(self.model, hidden, valid, credit)
                if decoded is None:
                    scores = dwell_t.new_full((action_count,), -1.0e9)
                    scores[valid] = 0.0
                    action_t = torch.argmax(scores)
                else:
                    action_t, credit = decoded
                action = int(action_t.detach().cpu().item())
                if action <= 0:
                    out.append(xs_s_search_action(MAXT))
                    local_debt = 0.0
                else:
                    row = rows[action - 1] if action - 1 < len(rows) else action
                    if not (1 <= int(row) <= MAXT):
                        break
                    out.append(xs_s_track_action(int(row), MAXT))
                    selected[action] = True
                    local_debt += float(dwell_np[action])
                remaining = remaining - dwell_t[action_t].clamp_min(1.0)
                action_in = torch.as_tensor([[action]], dtype=torch.long, device=device)
                _value, _reward, _next_policy, hidden = self.model.recurrent_inference(hidden, action_in)
                if float(remaining.detach().cpu().item()) <= 0.0:
                    break
        self.estimated_debt_ms = float(local_debt)
        self.window_index += 1
        return out if out else [xs_s_search_action(MAXT)]


def load_action_attention_model(path: Path, device: str, variant: str = "two_row_action_attention"):
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    d_model = int(state["cls_token"].shape[0]) if isinstance(state, dict) and "cls_token" in state else 48
    model_args = argparse.Namespace(d_model=d_model, nhead=4, nlayers=2)
    model = make_physical_model(str(variant), model_args).to(device).eval()
    model.load_state_dict(state, strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def maybe_quantize_model(model, enabled: bool, device: str):
    if not enabled:
        return model
    if str(device) != "cpu":
        raise ValueError("--quantize-dynamic is only supported for --device cpu")
    # Avoid quantizing TransformerEncoder internals; PyTorch's fast path expects
    # plain tensor weights there. The cached AR hot path is mostly the factorized
    # MLP heads, so quantize those modules in place and leave backbone/couplers.
    for name in [
        "sensor_state_proj",
        "type_head",
        "type_q_head",
        "target_head",
        "target_q_head",
        "action_proj",
        "action_policy_residual",
        "action_q_residual",
    ]:
        if hasattr(model, name):
            setattr(model, name, torch.quantization.quantize_dynamic(getattr(model, name), {nn.Linear}, dtype=torch.qint8))
    return model.eval()


def run_suite(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = False
    exact_args.single_sensor = True
    model_variant = str(getattr(args, "model_variant", "two_row_action_attention"))
    model = load_action_attention_model(Path(args.base_state), args.device, model_variant)
    model = maybe_quantize_model(model, bool(args.quantize_dynamic), args.device)
    all_windows = []
    all_actions = []
    summaries = []
    for initial in parse_ints(args.initials):
        for rate in parse_floats(args.rates):
            env_cfg = env_cfg_for(float(rate), exact_args)
            env_cfg["enable_x_band"] = 0
            for seed in parse_ints(args.seeds):
                base = PhysicalHeadPlanner(
                    model,
                    model_variant,
                    env_cfg,
                    policy_weight=float(args.policy_weight),
                    q_weight=float(args.q_weight),
                    search_score_bias=float(args.search_bias),
                )
                wanted = {
                    str(x).strip()
                    for x in str(getattr(args, "planners", "all")).split(",")
                    if str(x).strip()
                }
                aliases = {
                    "cached": "Cached AR ActionAttention PQ",
                    "graph_cached": "CUDA graph S-only ActionAttention PQ",
                    "graph_reencode": "CUDA graph re-encode S-only PQ",
                    "fastseq": "Fast S-only sequence decoder",
                    "cached_floor": "Cached AR ActionAttention PQ + search floor",
                    "history": "History AR ActionAttention PQ",
                    "history_ar": "History AR ActionAttention PQ",
                    "ar": "AR ActionAttention PQ",
                    "floor": "AR ActionAttention PQ + search floor",
                    "edf": "EDF",
                    "est": "EST",
                    "muzero": "MuZero latent accumulator PQ",
                    "muzero_latent": "MuZero latent accumulator PQ",
                }
                wanted_names = {aliases.get(x, x) for x in wanted}

                def want(name: str) -> bool:
                    return (not wanted) or "all" in wanted or name in wanted_names

                planners = []
                if want("CUDA graph re-encode S-only PQ"):
                    planners.append(
                        (
                            "CUDA graph re-encode S-only PQ",
                            CudaGraphReencodeSingleSensorPQ(
                            base,
                            max_steps=int(args.max_steps),
                            search_floor=0,
                            search_cap_frac=float(args.search_cap_frac),
                            search_score_bias=float(args.search_bias),
                            env_cfg=env_cfg,
                            search_pressure_bias_weight=float(args.search_pressure_bias_weight),
                            search_pressure_bias_target=float(args.search_pressure_bias_target),
                            search_active_bias_weight=float(args.search_active_bias_weight),
                            search_active_bias_target=float(args.search_active_bias_target),
                            track_pressure_bias_weight=float(args.track_pressure_bias_weight),
                            disable_action_coupler=bool(args.disable_action_coupler),
                            allow_soft_overshoot=bool(args.allow_soft_overshoot),
                            soft_overshoot_ms=float(args.soft_overshoot_ms),
                        ),
                        )
                    )
                if want("CUDA graph S-only ActionAttention PQ"):
                    planners.append(
                        (
                            "CUDA graph S-only ActionAttention PQ",
                            CudaGraphSingleSensorActionAttentionAR(
                            base,
                            max_steps=int(args.max_steps),
                            search_floor=0,
                            search_cap_frac=float(args.search_cap_frac),
                            search_score_bias=float(args.search_bias),
                            env_cfg=env_cfg,
                            search_pressure_bias_weight=float(args.search_pressure_bias_weight),
                            search_pressure_bias_target=float(args.search_pressure_bias_target),
                            search_active_bias_weight=float(args.search_active_bias_weight),
                            search_active_bias_target=float(args.search_active_bias_target),
                            track_pressure_bias_weight=float(args.track_pressure_bias_weight),
                            disable_action_coupler=bool(args.disable_action_coupler),
                            action_coupler_top_k=int(args.action_coupler_top_k),
                            sparse_residuals=bool(args.sparse_residuals),
                            allow_soft_overshoot=bool(args.allow_soft_overshoot),
                            soft_overshoot_ms=float(args.soft_overshoot_ms),
                        ),
                        )
                    )
                if want("Fast S-only sequence decoder"):
                    planners.append(
                        (
                            "Fast S-only sequence decoder",
                            FastSequenceSingleSensorActionAttentionAR(
                            base,
                            max_steps=int(args.max_steps),
                            search_floor=0,
                            search_cap_frac=float(args.search_cap_frac),
                            search_score_bias=float(args.search_bias),
                            env_cfg=env_cfg,
                            search_pressure_bias_weight=float(args.search_pressure_bias_weight),
                            search_pressure_bias_target=float(args.search_pressure_bias_target),
                            search_active_bias_weight=float(args.search_active_bias_weight),
                            search_active_bias_target=float(args.search_active_bias_target),
                            track_pressure_bias_weight=float(args.track_pressure_bias_weight),
                            disable_action_coupler=bool(args.disable_action_coupler),
                            action_coupler_top_k=int(args.action_coupler_top_k),
                            sparse_residuals=bool(args.sparse_residuals),
                            single_sensor_action_only=bool(args.single_sensor_action_only),
                            allow_soft_overshoot=bool(args.allow_soft_overshoot),
                            soft_overshoot_ms=float(args.soft_overshoot_ms),
                            refine_passes=int(args.fastseq_refine_passes),
                        ),
                        )
                    )
                if want("Cached AR ActionAttention PQ"):
                    planners.append(
                        (
                            "Cached AR ActionAttention PQ",
                            CachedSingleSensorActionAttentionAR(
                            base,
                            max_steps=int(args.max_steps),
                            search_floor=0,
                            search_cap_frac=float(args.search_cap_frac),
                            search_score_bias=float(args.search_bias),
                            env_cfg=env_cfg,
                            search_pressure_bias_weight=float(args.search_pressure_bias_weight),
                            search_pressure_bias_target=float(args.search_pressure_bias_target),
                            search_active_bias_weight=float(args.search_active_bias_weight),
                            search_active_bias_target=float(args.search_active_bias_target),
                            track_pressure_bias_weight=float(args.track_pressure_bias_weight),
                            disable_action_coupler=bool(args.disable_action_coupler),
                            action_coupler_top_k=int(args.action_coupler_top_k),
                            sparse_residuals=bool(args.sparse_residuals),
                            single_sensor_action_only=bool(args.single_sensor_action_only),
                            allow_soft_overshoot=bool(args.allow_soft_overshoot),
                            soft_overshoot_ms=float(args.soft_overshoot_ms),
                        ),
                        )
                    )
                if want("History AR ActionAttention PQ"):
                    planners.append(
                        (
                            "History AR ActionAttention PQ",
                            HistoryVectorSingleSensorActionAttentionAR(
                            base,
                            max_steps=int(args.max_steps),
                            search_floor=0,
                            search_cap_frac=float(args.search_cap_frac),
                            search_score_bias=float(args.search_bias),
                            env_cfg=env_cfg,
                            search_pressure_bias_weight=float(args.search_pressure_bias_weight),
                            search_pressure_bias_target=float(args.search_pressure_bias_target),
                            search_active_bias_weight=float(args.search_active_bias_weight),
                            search_active_bias_target=float(args.search_active_bias_target),
                            track_pressure_bias_weight=float(args.track_pressure_bias_weight),
                            disable_action_coupler=bool(args.disable_action_coupler),
                            action_coupler_top_k=int(args.action_coupler_top_k),
                            sparse_residuals=bool(args.sparse_residuals),
                            single_sensor_action_only=bool(args.single_sensor_action_only),
                            allow_soft_overshoot=bool(args.allow_soft_overshoot),
                            soft_overshoot_ms=float(args.soft_overshoot_ms),
                            history_len=int(args.history_len),
                        ),
                        )
                    )
                if want("Cached AR ActionAttention PQ + search floor"):
                    planners.append(
                        (
                            "Cached AR ActionAttention PQ + search floor",
                            CachedSingleSensorActionAttentionAR(
                            base,
                            max_steps=int(args.max_steps),
                            search_floor=int(args.search_floor),
                            search_cap_frac=float(args.search_cap_frac),
                            search_score_bias=float(args.search_bias),
                            env_cfg=env_cfg,
                            search_pressure_bias_weight=float(args.search_pressure_bias_weight),
                            search_pressure_bias_target=float(args.search_pressure_bias_target),
                            search_active_bias_weight=float(args.search_active_bias_weight),
                            search_active_bias_target=float(args.search_active_bias_target),
                            track_pressure_bias_weight=float(args.track_pressure_bias_weight),
                            disable_action_coupler=bool(args.disable_action_coupler),
                            action_coupler_top_k=int(args.action_coupler_top_k),
                            sparse_residuals=bool(args.sparse_residuals),
                            single_sensor_action_only=bool(args.single_sensor_action_only),
                            allow_soft_overshoot=bool(args.allow_soft_overshoot),
                            soft_overshoot_ms=float(args.soft_overshoot_ms),
                        ),
                        )
                    )
                if want("AR ActionAttention PQ"):
                    planners.append(
                        (
                            "AR ActionAttention PQ",
                            SingleSensorActionAttentionAR(
                            base,
                            max_steps=int(args.max_steps),
                            search_floor=0,
                            search_cap_frac=float(args.search_cap_frac),
                            search_score_bias=float(args.search_bias),
                            env_cfg=env_cfg,
                            search_pressure_bias_weight=float(args.search_pressure_bias_weight),
                            search_pressure_bias_target=float(args.search_pressure_bias_target),
                            search_active_bias_weight=float(args.search_active_bias_weight),
                            search_active_bias_target=float(args.search_active_bias_target),
                            track_pressure_bias_weight=float(args.track_pressure_bias_weight),
                            allow_soft_overshoot=bool(args.allow_soft_overshoot),
                            soft_overshoot_ms=float(args.soft_overshoot_ms),
                        ),
                        )
                    )
                if want("AR ActionAttention PQ + search floor"):
                    planners.append(
                        (
                            "AR ActionAttention PQ + search floor",
                            SingleSensorActionAttentionAR(
                            base,
                            max_steps=int(args.max_steps),
                            search_floor=int(args.search_floor),
                            search_cap_frac=float(args.search_cap_frac),
                            search_score_bias=float(args.search_bias),
                            env_cfg=env_cfg,
                            search_pressure_bias_weight=float(args.search_pressure_bias_weight),
                            search_pressure_bias_target=float(args.search_pressure_bias_target),
                            search_active_bias_weight=float(args.search_active_bias_weight),
                            search_active_bias_target=float(args.search_active_bias_target),
                            track_pressure_bias_weight=float(args.track_pressure_bias_weight),
                            allow_soft_overshoot=bool(args.allow_soft_overshoot),
                            soft_overshoot_ms=float(args.soft_overshoot_ms),
                        ),
                        )
                    )
                if want("EDF"):
                    planners.append(("EDF", SingleSensorHeuristicAdapter(EDFPlanner(MAXT))))
                if want("EST"):
                    planners.append(("EST", SingleSensorHeuristicAdapter(ESTPlanner(MAXT))))
                if want("MuZero latent accumulator PQ"):
                    if not str(getattr(args, "muzero_state", "")).strip():
                        raise ValueError("--muzero-state is required for planner 'muzero'")
                    muzero_model, muzero_cfg = load_muzero_model(
                        Path(args.muzero_state),
                        network=str(args.muzero_network),
                        simulations=0,
                        device=str(args.device),
                    )
                    planners.append(
                        (
                            "MuZero latent accumulator PQ",
                            MuZeroLatentAccumulatorPlanAdapter(
                                muzero_model,
                                muzero_cfg,
                                env_cfg=env_cfg,
                                max_steps=int(args.max_steps),
                                total_windows=int(args.windows),
                            ),
                        )
                    )
                for name, planner in planners:
                    df, actions = run_plan_eval(planner, name, int(initial), int(seed), int(args.windows), env_cfg)
                    df = df.copy()
                    df["initial"] = int(initial)
                    df["rate"] = float(rate)
                    all_windows.append(df)
                    if not actions.empty:
                        actions = actions.copy()
                        actions["initial"] = int(initial)
                        actions["rate"] = float(rate)
                        all_actions.append(actions)
                    row = {"planner": name, "initial": int(initial), "rate": float(rate), "seed": int(seed), **summarize(df)}
                    summaries.append(row)
                    print(row, flush=True)
    windows = pd.concat(all_windows, ignore_index=True)
    actions = pd.concat(all_actions, ignore_index=True) if all_actions else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(out, index=False)
    actions.to_csv(out.with_name(out.stem + "_actions.csv"), index=False)
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    return windows, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-state", default=str(BASE_STATE))
    ap.add_argument("--muzero-state", default="")
    ap.add_argument("--muzero-network", default="factorized_fullyconnected")
    ap.add_argument("--model-variant", default="two_row_action_attention")
    ap.add_argument("--out", default=str(ROOT / "CreateValid1" / "results" / "single_sensor_ar_action_attention_smoke.csv"))
    ap.add_argument("--planners", default="all", help="Comma-separated: all,fastseq,cached,history,ar,floor,edf,est")
    ap.add_argument("--initials", default="20")
    ap.add_argument("--rates", default="2")
    ap.add_argument("--seeds", default="916")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--fastseq-refine-passes", type=int, default=1)
    ap.add_argument("--history-len", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--env-mode", default="current")
    ap.add_argument("--track-update-reward", type=float, default=0.30)
    ap.add_argument("--track-loss-penalty", type=float, default=8.0)
    ap.add_argument("--sector-staleness-weight", type=float, default=0.0)
    ap.add_argument("--search-frame-overdue-weight", type=float, default=0.20)
    ap.add_argument("--search-frame-desired-ms", type=float, default=3000.0)
    ap.add_argument("--search-frame-deadline-ms", type=float, default=4500.0)
    ap.add_argument("--search-frame-drop-penalty", type=float, default=0.0)
    ap.add_argument("--search-frame-state-penalty-weight", type=float, default=0.0)
    ap.add_argument("--search-frame-delta-reward-weight", type=float, default=0.0)
    ap.add_argument("--service-pressure-delta-reward-weight", type=float, default=0.0)
    ap.add_argument("--serviced-pressure-improvement-reward-weight", type=float, default=0.0)
    ap.add_argument("--discovered-target-reward", type=float, default=0.0)
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--q-weight", type=float, default=1.0)
    ap.add_argument("--search-bias", type=float, default=0.0)
    ap.add_argument("--search-floor", type=int, default=4)
    ap.add_argument("--search-cap-frac", type=float, default=0.35)
    ap.add_argument("--search-pressure-bias-weight", type=float, default=0.0)
    ap.add_argument("--search-pressure-bias-target", type=float, default=0.0)
    ap.add_argument("--search-active-bias-weight", type=float, default=0.0)
    ap.add_argument("--search-active-bias-target", type=float, default=40.0)
    ap.add_argument("--track-pressure-bias-weight", type=float, default=0.0)
    ap.add_argument("--disable-action-coupler", action="store_true")
    ap.add_argument("--action-coupler-top-k", type=int, default=0)
    ap.add_argument("--sparse-residuals", action="store_true")
    ap.add_argument("--single-sensor-action-only", action="store_true")
    ap.add_argument("--allow-soft-overshoot", action="store_true")
    ap.add_argument("--soft-overshoot-ms", type=float, default=-1.0)
    ap.add_argument("--quantize-dynamic", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(max(1, int(args.torch_threads)))
    torch.set_num_interop_threads(1)
    _windows, summary = run_suite(args)
    cols = ["reward_per_window", "drop_pct_active", "tracked_targets", "mean_delay_active", "search_fraction", "planning_ms_per_window"]
    print(summary.groupby("planner")[cols].mean().round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
