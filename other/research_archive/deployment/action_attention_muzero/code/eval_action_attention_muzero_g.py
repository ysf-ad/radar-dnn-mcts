from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "CreateValid1" / "experiments" / "code" / "model_code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from best_model_joint_vs_seq_ablation import DirectPlanAdapter, WorkConservingAsyncCoupledPlanner, run_exact_rescore_grid_joint, train_variant
from exact_env_mutual import EDFPlanner, ESTPlanner, MAXT, _DummyPlanner, attach_env_obs, engine_env_cfg, env_cfg_for, shaped_step_reward, xs_decode_action, xs_s_search_action, xs_s_track_action, xs_x_search_action, xs_x_track_action
from final_radar_campaign import get_obs
from foundation_mcts_fair_eval import parse_floats, parse_ints
from joint_action_experiment import encode_joint_action, execute_first_valid_action_joint, is_joint_action, joint_duration, split_joint_action
from mutual_features import TOKEN_DIM, slot_features, tokenize
from penalty_window_quota_learner_eval import make_exact_args
from repaired_campaign_tools import build_env
from realistic_reward_retrain import adapter
from strict_window_report import sample_state_metrics
from train_action_attention_muzero_g import LatentG, latent_scores
from two_sensor_physical_head_eval import PhysicalHeadPlanner, make_physical_model


class _PassthroughCoupler(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class LatentGraphStep(torch.nn.Module):
    def __init__(self, model, g: LatentG, use_g_policy: bool = False, use_model_q: bool = True):
        super().__init__()
        self.model = model
        self.g = g
        self.use_g_policy = bool(use_g_policy)
        self.use_model_q = bool(use_model_q)

    def forward(self, cls, tok, slot, action_pair, selected, token_active):
        cls_next, tok_next, reward, dt = self.g(cls, tok, slot, action_pair)
        if self.use_g_policy and hasattr(self.g, "policy_scores"):
            scores = self.g.policy_scores(cls_next, tok_next, slot, selected, token_active)
            if self.use_model_q and hasattr(self, "model"):
                _model_scores, q = latent_scores(self.model, cls_next, tok_next, slot, selected, token_active)
            else:
                q = torch.zeros_like(scores)
        else:
            scores, q = latent_scores(self.model, cls_next, tok_next, slot, selected, token_active)
        return cls_next, tok_next, scores, q, reward, dt


class CudaGraphLatentStep:
    def __init__(self, model, g: LatentG, cls, tok, slot, action_pair, selected, token_active, use_g_policy: bool = False, use_model_q: bool = True):
        self.module = LatentGraphStep(model, g, use_g_policy=use_g_policy, use_model_q=use_model_q).eval()
        self.cls = cls.clone()
        self.tok = tok.clone()
        self.slot = slot.clone()
        self.action_pair = action_pair.clone()
        self.selected = selected.clone()
        self.token_active = token_active.clone()
        with torch.inference_mode():
            for _ in range(8):
                self.out = self.module(self.cls, self.tok, self.slot, self.action_pair, self.selected, self.token_active)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self.module(self.cls, self.tok, self.slot, self.action_pair, self.selected, self.token_active)
            self.graph.replay()
            torch.cuda.synchronize()

    def __call__(self, cls, tok, slot, action_pair, selected, token_active):
        with torch.inference_mode():
            self.cls.copy_(cls)
            self.tok.copy_(tok)
            self.slot.copy_(slot, non_blocking=True)
            self.action_pair.copy_(action_pair, non_blocking=True)
            self.selected.copy_(selected, non_blocking=True)
            self.token_active.copy_(token_active)
            self.graph.replay()
            return self.out


class RootEncodeScoreGraphModule(torch.nn.Module):
    def __init__(
        self,
        model,
        g: LatentG,
        use_g_policy: bool = False,
        use_model_q: bool = True,
        g_alt: LatentG | None = None,
        g_blend_alpha: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.g = g
        self.use_g_policy = bool(use_g_policy)
        self.use_model_q = bool(use_model_q)
        self.g_alt = g_alt
        self.g_blend_alpha = float(g_blend_alpha)

    def forward(self, x, slot):
        cls, tok, selected, token_active = self.model.backbone.encode_tokens(x)
        if self.use_g_policy and hasattr(self.g, "policy_scores"):
            scores = self.g.policy_scores(cls, tok, slot, selected, token_active)
            if self.g_alt is not None and hasattr(self.g_alt, "policy_scores") and self.g_blend_alpha < 1.0:
                alt_scores = self.g_alt.policy_scores(cls, tok, slot, selected, token_active)
                scores = self.g_blend_alpha * scores + (1.0 - self.g_blend_alpha) * alt_scores
            if self.use_model_q:
                _model_scores, q = latent_scores(self.model, cls, tok, slot, selected, token_active)
            else:
                q = torch.zeros_like(scores)
        else:
            scores, q = latent_scores(self.model, cls, tok, slot, selected, token_active)
        return cls, tok, selected, token_active, scores, q


class CudaGraphRootEncodeScore:
    """CUDA graph wrapper for the fixed-shape root encoder plus first action scores."""

    def __init__(
        self,
        model,
        g: LatentG,
        x,
        slot,
        *,
        use_g_policy: bool = False,
        use_model_q: bool = True,
        g_alt: LatentG | None = None,
        g_blend_alpha: float = 1.0,
    ):
        self.module = RootEncodeScoreGraphModule(
            model,
            g,
            use_g_policy=use_g_policy,
            use_model_q=use_model_q,
            g_alt=g_alt,
            g_blend_alpha=g_blend_alpha,
        ).eval()
        self.x = x.clone()
        self.slot = slot.clone()
        with torch.inference_mode():
            for _ in range(8):
                self.out = self.module(self.x, self.slot)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self.module(self.x, self.slot)
            self.graph.replay()
            torch.cuda.synchronize()

    def __call__(self, x, slot):
        with torch.inference_mode():
            self.x.copy_(x, non_blocking=True)
            self.slot.copy_(slot, non_blocking=True)
            self.graph.replay()
            return self.out


class EncodeTokensGraphModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.backbone.encode_tokens(x)


class CudaGraphEncodeTokens:
    """CUDA graph wrapper for fixed-shape state token encoding."""

    def __init__(self, model, x):
        self.module = EncodeTokensGraphModule(model).eval()
        self.x = x.clone()
        with torch.inference_mode():
            for _ in range(8):
                self.out = self.module(self.x)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self.module(self.x)
            self.graph.replay()
            torch.cuda.synchronize()

    def __call__(self, x):
        with torch.inference_mode():
            self.x.copy_(x, non_blocking=True)
            self.graph.replay()
            return self.out


class RootSeqScoresGraphModule(torch.nn.Module):
    def __init__(self, g: LatentG, use_step_context: bool = False):
        super().__init__()
        self.g = g
        self.use_step_context = bool(use_step_context)

    def forward(self, cls, tok, slot, selected, token_active, seq_slots):
        return self.g.sequence_scores(
            cls,
            tok,
            slot,
            selected,
            token_active,
            seq_slots=seq_slots,
            use_step_context=self.use_step_context,
        )


class CudaGraphRootSeqScores:
    """CUDA graph wrapper for the promoted 0-rollout root-sequence scorer."""

    def __init__(self, g: LatentG, cls, tok, slot, selected, token_active, seq_slots, use_step_context: bool = False):
        self.module = RootSeqScoresGraphModule(g, use_step_context=use_step_context).eval()
        self.cls = cls.clone()
        self.tok = tok.clone()
        self.slot = slot.clone()
        self.selected = selected.clone()
        self.token_active = token_active.clone()
        self.seq_slots = seq_slots.clone()
        with torch.inference_mode():
            for _ in range(8):
                self.out = self.module(self.cls, self.tok, self.slot, self.selected, self.token_active, self.seq_slots)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self.module(self.cls, self.tok, self.slot, self.selected, self.token_active, self.seq_slots)
            self.graph.replay()
            torch.cuda.synchronize()

    def __call__(self, cls, tok, slot, selected, token_active, seq_slots):
        with torch.inference_mode():
            self.cls.copy_(cls)
            self.tok.copy_(tok)
            self.slot.copy_(slot, non_blocking=True)
            self.selected.copy_(selected, non_blocking=True)
            self.token_active.copy_(token_active)
            self.seq_slots.copy_(seq_slots, non_blocking=True)
            self.graph.replay()
            return self.out


class ARSOnlyDecodeGraphModule(torch.nn.Module):
    """Fixed-shape S-only autoregressive decoder captured as one graph replay."""

    def __init__(self, model, g: LatentG, *, max_steps: int, search_score_bias: float):
        super().__init__()
        self.model = model
        self.g = g
        self.max_steps = int(max_steps)
        self.search_score_bias = float(search_score_bias)

    def _s_only_factor_logits(self, h, ar_tok, slot_step, pos_step, selected, token_active):
        bsz, rows, dim = ar_tok.shape
        slot_e = self.g.slot_proj(slot_step)
        sensor_e = self.g.sensor_embed[0:1].expand(bsz, -1)
        type_in = torch.cat([h, slot_e, pos_step, sensor_e], dim=-1)
        type_logits = self.g.ar_type_policy(type_in)
        h_row = h[:, None, :].expand(-1, rows, -1)
        slot_row = slot_e[:, None, :].expand(-1, rows, -1)
        pos_row = pos_step[:, None, :].expand(-1, rows, -1)
        sensor_row = sensor_e[:, None, :].expand(-1, rows, -1)
        target_in = torch.cat([ar_tok, h_row, slot_row, pos_row, sensor_row], dim=-1)
        target_logits = self.g.ar_target_policy(target_in).squeeze(-1)
        row_ids = torch.arange(rows, device=ar_tok.device)[None, :]
        track_mask = token_active & (~selected) & (row_ids != 0)
        target_logits = target_logits.masked_fill(~track_mask, -1.0e9)
        return type_logits, target_logits

    def forward(self, x, seq_slots):
        cls, tok, root_selected, token_active = self.model.backbone.encode_tokens(x)
        h, ar_tok = self.g._project_state(cls, tok)
        selected = root_selected.clone()
        prev = torch.zeros((1, 2), dtype=torch.long, device=x.device)
        prev[:, 1] = 1
        history = None
        if int(getattr(self.g, "ar_history_k", 0)) > 0:
            history = prev[:, None, :].expand(-1, int(self.g.ar_history_k), -1).clone()
        rows = torch.empty((self.max_steps,), dtype=torch.long, device=x.device)
        for step in range(self.max_steps):
            slot_step = seq_slots[:, step, :]
            pos_step = self.g.seq_pos[step][None, :].expand(1, -1)
            a0 = self.g.action_emb(prev[:, 0])
            a1 = self.g.action_emb(prev[:, 1])
            slot_e = self.g.slot_proj(slot_step)
            inp = self.g.ar_input(torch.cat([a0, a1, slot_e, pos_step], dim=-1))
            if history is not None:
                inp = inp + self.g._ar_history_embedding(history)
            h = self.g.ar_cell(inp, h)
            type_logits, target_logits = self._s_only_factor_logits(
                h,
                ar_tok,
                slot_step,
                pos_step,
                selected,
                token_active,
            )
            search_logit = type_logits[0, 0] + self.search_score_bias
            track_logit = type_logits[0, 1]
            target_col = target_logits[0].clone()
            row_ids = torch.arange(target_col.shape[0], device=target_col.device)
            target_col = target_col.masked_fill(row_ids == 0, -1.0e9)
            best_track = torch.argmax(target_col)
            best_value = target_col.gather(0, best_track.reshape(1))[0]
            has_track = torch.isfinite(best_value) & (best_value > -1.0e8)
            choose_search = (search_logit >= track_logit) | (~has_track)
            row = torch.where(choose_search, torch.zeros_like(best_track), best_track)
            rows[step] = row
            is_track = row > 0
            selected.scatter_(1, row.clamp_min(0).reshape(1, 1), is_track.reshape(1, 1))
            prev[:, 0] = row.clamp_min(0) * 2
            prev[:, 1] = 1
            if history is not None:
                history = self.g._update_ar_history(history, prev)
        return rows


class CudaGraphARSOnlyDecode:
    def __init__(self, model, g: LatentG, x, seq_slots, *, max_steps: int, search_score_bias: float):
        self.device = x.device
        self.module = ARSOnlyDecodeGraphModule(
            model,
            g,
            max_steps=int(max_steps),
            search_score_bias=float(search_score_bias),
        ).eval()
        self.x = x.clone()
        self.seq_slots = seq_slots.clone()
        with torch.inference_mode():
            for _ in range(8):
                self.rows = self.module(self.x, self.seq_slots)
            torch.cuda.synchronize(self.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.rows = self.module(self.x, self.seq_slots)
            self.graph.replay()
            torch.cuda.synchronize(self.device)

    def __call__(self, x, seq_slots):
        with torch.inference_mode():
            self.x.copy_(x, non_blocking=True)
            self.seq_slots.copy_(seq_slots, non_blocking=True)
            self.graph.replay()
            return self.rows


class ARSOnlyDynamicDecodeGraphModule(torch.nn.Module):
    """S-only AR decoder with in-graph slot updates from decoded actions."""

    def __init__(self, model, g: LatentG, *, max_steps: int, search_score_bias: float, budget_ms: float = 200.0):
        super().__init__()
        self.model = model
        self.g = g
        self.max_steps = int(max_steps)
        self.search_score_bias = float(search_score_bias)
        self.budget_ms = float(budget_ms)

    def _s_only_factor_logits(self, h, ar_tok, slot_step, pos_step, selected, token_active):
        bsz, rows, dim = ar_tok.shape
        slot_e = self.g.slot_proj(slot_step)
        sensor_e = self.g.sensor_embed[0:1].expand(bsz, -1)
        type_in = torch.cat([h, slot_e, pos_step, sensor_e], dim=-1)
        type_logits = self.g.ar_type_policy(type_in)
        h_row = h[:, None, :].expand(-1, rows, -1)
        slot_row = slot_e[:, None, :].expand(-1, rows, -1)
        pos_row = pos_step[:, None, :].expand(-1, rows, -1)
        sensor_row = sensor_e[:, None, :].expand(-1, rows, -1)
        target_in = torch.cat([ar_tok, h_row, slot_row, pos_row, sensor_row], dim=-1)
        target_logits = self.g.ar_target_policy(target_in).squeeze(-1)
        row_ids = torch.arange(rows, device=ar_tok.device)[None, :]
        track_mask = token_active & (~selected) & (row_ids != 0)
        target_logits = target_logits.masked_fill(~track_mask, -1.0e9)
        return type_logits, target_logits

    def forward(self, x, slot_const, dwell, row_map, type_noise, target_noise):
        cls, tok, root_selected, token_active = self.model.backbone.encode_tokens(x)
        s_valid = x[:, :, 10] > 0.5
        s_valid[:, 0] = True
        token_active = token_active & s_valid
        h, ar_tok = self.g._project_state(cls, tok)
        selected = root_selected.clone()
        prev = torch.zeros((1, 2), dtype=torch.long, device=x.device)
        prev[:, 1] = 1
        history = None
        if int(getattr(self.g, "ar_history_k", 0)) > 0:
            history = prev[:, None, :].expand(-1, int(self.g.ar_history_k), -1).clone()
        rows = torch.empty((self.max_steps,), dtype=torch.long, device=x.device)
        # The first four slot values carry the already executed prefix when
        # this graph is used for receding-horizon replanning.
        elapsed = slot_const[0] * self.budget_ms
        search_count = slot_const[1] * 20.0
        track_count = slot_const[2] * 100.0
        last_search = slot_const[3]
        neg_one = torch.full((), -1, dtype=torch.long, device=x.device)
        zero_long = torch.zeros((), dtype=torch.long, device=x.device)
        row_ids = torch.arange(dwell.shape[0], device=x.device)
        for step in range(self.max_steps):
            slot_step = torch.stack(
                [
                    elapsed / self.budget_ms,
                    search_count / 20.0,
                    track_count / 100.0,
                    last_search,
                    slot_const[4],
                    slot_const[5],
                    slot_const[6],
                    slot_const[7],
                    slot_const[8],
                    slot_const[9],
                    slot_const[10],
                ]
            ).reshape(1, 11)
            pos_step = self.g.seq_pos[step][None, :].expand(1, -1)
            a0 = self.g.action_emb(prev[:, 0])
            a1 = self.g.action_emb(prev[:, 1])
            slot_e = self.g.slot_proj(slot_step)
            inp = self.g.ar_input(torch.cat([a0, a1, slot_e, pos_step], dim=-1))
            if history is not None:
                inp = inp + self.g._ar_history_embedding(history)
            h = self.g.ar_cell(inp, h)
            type_logits, target_logits = self._s_only_factor_logits(
                h,
                ar_tok,
                slot_step,
                pos_step,
                selected,
                token_active,
            )
            remaining = torch.clamp(self.budget_ms - elapsed, min=0.0)
            valid_duration = dwell <= (remaining + 0.001)
            search_logit = type_logits[0, 0] + self.search_score_bias
            search_logit = torch.where(remaining >= 9.999, search_logit, torch.full_like(search_logit, -1.0e9))
            track_logit = type_logits[0, 1]
            target_col = target_logits[0].clone()
            target_col = target_col.masked_fill(row_ids == 0, -1.0e9)
            target_col = target_col.masked_fill(~valid_duration, -1.0e9)
            sampled_target_col = target_col + target_noise[step]
            best_track = torch.argmax(sampled_target_col)
            best_value = target_col.gather(0, best_track.reshape(1))[0]
            has_track = torch.isfinite(best_value) & (best_value > -1.0e8)
            has_search = search_logit > -1.0e8
            noisy_type = type_logits[0] + type_noise[step]
            choose_search = (noisy_type[0] + self.search_score_bias >= noisy_type[1]) | (~has_track)
            row = torch.where(choose_search & has_search, zero_long, best_track)
            active = has_search | has_track
            row = torch.where(active, row, zero_long)
            rows[step] = torch.where(active, row, neg_one)
            is_track = (row > 0) & active
            is_search = (row == 0) & active
            dt = dwell.gather(0, row.reshape(1))[0]
            elapsed = elapsed + torch.where(active, dt, torch.zeros_like(dt))
            search_count = search_count + is_search.to(x.dtype)
            track_count = track_count + is_track.to(x.dtype)
            last_search = is_search.to(x.dtype)
            selected.scatter_(1, row.clamp_min(0).reshape(1, 1), is_track.reshape(1, 1))
            original_row = row_map.gather(0, row.clamp_min(0).reshape(1))[0]
            prev[:, 0] = original_row.clamp_min(0) * 2
            prev[:, 1] = 1
            if history is not None:
                history = self.g._update_ar_history(history, prev)
        return rows


class CudaGraphARDynamicSOnlyDecode:
    def __init__(self, model, g: LatentG, x, slot_const, dwell, row_map, *, max_steps: int, search_score_bias: float, sample_temperature: float = 0.0, target_sample_temperature: float | None = None):
        self.device = x.device
        self.module = ARSOnlyDynamicDecodeGraphModule(
            model,
            g,
            max_steps=int(max_steps),
            search_score_bias=float(search_score_bias),
        ).eval()
        self.x = x.clone()
        self.slot_const = slot_const.clone()
        self.dwell = dwell.clone()
        self.row_map = row_map.clone()
        self.sample_temperature = max(0.0, float(sample_temperature))
        self.target_sample_temperature = self.sample_temperature if target_sample_temperature is None else max(0.0, float(target_sample_temperature))
        self.type_noise = x.new_zeros((int(max_steps), 2))
        self.target_noise = x.new_zeros((int(max_steps), int(dwell.shape[0])))
        with torch.inference_mode():
            for _ in range(8):
                self.rows = self.module(self.x, self.slot_const, self.dwell, self.row_map, self.type_noise, self.target_noise)
            torch.cuda.synchronize(self.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.rows = self.module(self.x, self.slot_const, self.dwell, self.row_map, self.type_noise, self.target_noise)
            self.graph.replay()
            torch.cuda.synchronize(self.device)

    def __call__(self, x, slot_const, dwell, row_map):
        with torch.inference_mode():
            self.x.copy_(x, non_blocking=True)
            self.slot_const.copy_(slot_const, non_blocking=True)
            self.dwell.copy_(dwell, non_blocking=True)
            self.row_map.copy_(row_map, non_blocking=True)
            if self.sample_temperature > 0.0:
                self.type_noise.copy_(-torch.log(-torch.log(torch.rand_like(self.type_noise).clamp_(1.0e-6, 1.0 - 1.0e-6))) * self.sample_temperature)
            else:
                self.type_noise.zero_()
            if self.target_sample_temperature > 0.0:
                self.target_noise.copy_(-torch.log(-torch.log(torch.rand_like(self.target_noise).clamp_(1.0e-6, 1.0 - 1.0e-6))) * self.target_sample_temperature)
            else:
                self.target_noise.zero_()
            self.graph.replay()
            return self.rows

    def copy_and_replay(self, x_cpu, slot_const_cpu, dwell_cpu, row_map_cpu):
        with torch.inference_mode():
            self.x.copy_(x_cpu, non_blocking=True)
            self.slot_const.copy_(slot_const_cpu, non_blocking=True)
            self.dwell.copy_(dwell_cpu, non_blocking=True)
            self.row_map.copy_(row_map_cpu, non_blocking=True)
            self.graph.replay()
            return self.rows


class LatentSOnlyWindowGraphModule(torch.nn.Module):
    """Fixed-shape S-only MuZero 0R latent greedy decoder.

    The ordinary tensor loop keeps the recurrent state on device, but Python
    still launches each g/head step separately. This module captures the full
    200 ms decode loop as one CUDA graph replay. It intentionally supports the
    clean deployment subset: S-only, greedy, no external caps/gates/rerank.
    """

    def __init__(
        self,
        model,
        g: LatentG,
        *,
        max_steps: int,
        policy_weight: float,
        q_weight: float,
        search_score_bias: float,
        use_g_policy: bool,
        use_model_q: bool,
        noop_x: bool,
        s_only_score: bool,
        factorized_decode: bool,
    ):
        super().__init__()
        self.model = model
        self.g = g
        self.max_steps = int(max_steps)
        self.policy_weight = float(policy_weight)
        self.q_weight = float(q_weight)
        self.search_score_bias = float(search_score_bias)
        self.use_g_policy = bool(use_g_policy)
        self.use_model_q = bool(use_model_q)
        self.noop_x = bool(noop_x)
        self.s_only_score = bool(s_only_score)
        self.factorized_decode = bool(factorized_decode)

    def _score(self, cls, tok, slot, selected, token_active):
        if not self.s_only_score:
            if self.use_g_policy and hasattr(self.g, "policy_scores"):
                scores = self.g.policy_scores(cls, tok, slot, selected, token_active)
                if self.use_model_q and abs(self.q_weight) > 0.0:
                    _model_scores, q = latent_scores(self.model, cls, tok, slot, selected, token_active)
                else:
                    q = torch.zeros_like(scores)
                return scores[:, :, 0], q[:, :, 0]
            scores, q = latent_scores(self.model, cls, tok, slot, selected, token_active)
            return scores[:, :, 0], q[:, :, 0]
        if self.use_g_policy and hasattr(self.g, "policy_scores"):
            scores_s = self._g_policy_scores_sonly(cls, tok, slot, selected, token_active)
            if self.use_model_q and abs(self.q_weight) > 0.0:
                q_s = self._model_q_sonly(cls, tok, slot, selected, token_active)
            else:
                q_s = torch.zeros_like(scores_s)
            return scores_s, q_s
        return self._model_scores_sonly(cls, tok, slot, selected, token_active)

    def _g_policy_scores_sonly(self, cls, tok, slot, selected, token_active):
        cls, tok = self.g._project_state(cls, tok)
        bsz, rows, dim = tok.shape
        slot_e = self.g.slot_proj(slot)
        sensor = self.g.sensor_embed[0:1][None, :, :].expand(bsz, -1, -1)
        cls_s = cls[:, None, :]
        slot_s = slot_e[:, None, :]
        type_logits = self.g.type_policy(torch.cat([cls_s, slot_s, sensor], dim=-1))[:, 0, :]
        type_logp = torch.nn.functional.log_softmax(type_logits, dim=-1)
        cls_t = cls[:, None, :].expand(-1, rows, -1)
        slot_t = slot_e[:, None, :].expand(-1, rows, -1)
        sensor_t = sensor.expand(-1, rows, -1)
        target_logits = self.g.target_policy(torch.cat([tok, cls_t, slot_t, sensor_t], dim=-1)).squeeze(-1)
        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        target_logp = torch.nn.functional.log_softmax(target_logits.masked_fill(~track_mask, -1e9), dim=1)
        scores = slot.new_full((bsz, rows), -1e9)
        scores[:, 0] = type_logp[:, 0]
        scores[:, 1:] = (type_logp[:, None, 1] + target_logp)[:, 1:]
        row_is_search = torch.arange(rows, device=slot.device)[None, :] == 0
        valid = track_mask | row_is_search
        action_ctx = torch.cat([tok, cls_t, slot_t, sensor_t], dim=-1)
        action_tokens = self.g.policy_action_proj(action_ctx)
        mixer = str(getattr(self.g, "policy_action_mixer", "full"))
        if mixer == "light":
            token_mask = valid[:, :, None].to(action_tokens.dtype)
            pooled = (action_tokens * token_mask).sum(dim=1, keepdim=True) / token_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            scores = scores + self.g.policy_light_residual(torch.cat([action_tokens, pooled.expand_as(action_tokens)], dim=-1)).squeeze(-1)
        elif mixer == "tiny":
            mixed = self.g.policy_tiny_coupler(action_tokens, src_key_padding_mask=~valid)
            scores = scores + self.g.policy_tiny_residual(mixed).squeeze(-1)
        elif mixer != "none":
            mixed = self.g.policy_action_coupler(action_tokens, src_key_padding_mask=~valid)
            scores = scores + self.g.policy_action_residual(mixed).squeeze(-1)
        return scores.masked_fill(~valid, -1e9)

    def _model_scores_sonly(self, cls, tok, slot, selected, token_active):
        slot_emb = self.model.backbone.slot_proj(slot)
        bsz, rows, _ = tok.shape
        sensor = self.model.sensor_embed[0:1][None, :, :].expand(bsz, -1, -1)
        cls_s = cls[:, None, :]
        slot_s = slot_emb[:, None, :]
        if hasattr(self.model, "sensor_state_proj") and hasattr(self.model, "sensor_coupler"):
            sensor_state = self.model.sensor_state_proj(torch.cat([cls_s, slot_s, sensor], dim=-1))
            coupled_sensor = self.model.sensor_coupler(sensor_state)
        else:
            coupled_sensor = sensor
        type_ctx = torch.cat([cls_s, slot_s, coupled_sensor], dim=-1)
        type_logits = self.model.type_head(type_ctx)[:, 0, :]
        type_logp = torch.nn.functional.log_softmax(type_logits, dim=-1)
        type_q = self.model.type_q_head(type_ctx)[:, 0, :]
        cls_t = cls[:, None, :].expand(-1, rows, -1)
        slot_t = slot_emb[:, None, :].expand(-1, rows, -1)
        sensor_t = coupled_sensor.expand(-1, rows, -1)
        target_ctx = torch.cat([tok, cls_t, slot_t, sensor_t], dim=-1)
        target_logits = self.model.target_head(target_ctx).squeeze(-1)
        target_q = self.model.target_q_head(target_ctx).squeeze(-1)
        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        target_logp = torch.nn.functional.log_softmax(target_logits.masked_fill(~track_mask, -1e9), dim=1)
        scores = slot.new_full((bsz, rows), -1e9)
        q = slot.new_zeros((bsz, rows))
        scores[:, 0] = type_logp[:, 0]
        q[:, 0] = type_q[:, 0]
        scores[:, 1:] = (type_logp[:, None, 1] + target_logp)[:, 1:]
        q[:, 1:] = (type_q[:, None, 1] + target_q)[:, 1:]
        row_is_search = torch.arange(rows, device=slot.device)[None, :] == 0
        valid = track_mask | row_is_search
        if (
            hasattr(self.model, "action_proj")
            and hasattr(self.model, "action_coupler")
            and hasattr(self.model, "action_policy_residual")
            and hasattr(self.model, "action_q_residual")
        ):
            action_ctx = self.model.action_proj(target_ctx)
            action_ctx = self.model.action_coupler(action_ctx, src_key_padding_mask=~valid)
            scores = scores + self.model.action_policy_residual(action_ctx).squeeze(-1)
            q = q + self.model.action_q_residual(action_ctx).squeeze(-1)
        return scores.masked_fill(~valid, -1e9), q.masked_fill(~valid, 0.0)

    def _model_q_sonly(self, cls, tok, slot, selected, token_active):
        _scores, q = self._model_scores_sonly(cls, tok, slot, selected, token_active)
        return q

    def forward(self, x, const, s_duration, service_bonus, row_map):
        cls, tok, selected_t, token_active = self.model.backbone.encode_tokens(x)
        elapsed = x.new_zeros(())
        search_count = x.new_zeros(())
        track_count = x.new_zeros(())
        last_is_search = x.new_zeros(())
        rows = torch.empty((self.max_steps,), dtype=torch.long, device=x.device)
        action_pair = torch.empty((1, 2), dtype=torch.long, device=x.device)

        for step in range(self.max_steps):
            slot = torch.stack(
                [
                    elapsed / 200.0,
                    search_count / 20.0,
                    track_count / 100.0,
                    last_is_search,
                    const[0],
                    const[1],
                    const[2],
                    const[3],
                    const[4],
                    const[5],
                    const[6],
                ]
            ).reshape(1, 11)
            scores, q = self._score(cls, tok, slot, selected_t, token_active)
            utility = self.policy_weight * scores + self.q_weight * q
            s_score = (utility[0] + service_bonus[:, 0]).clone()
            s_score[0] = s_score[0] + self.search_score_bias
            if self.factorized_decode:
                track_mass = torch.logsumexp(s_score[1:], dim=0)
                best_track = torch.argmax(s_score[1:]) + 1
                s_row = torch.where(s_score[0] >= track_mass, torch.zeros_like(best_track), best_track)
            else:
                s_row = torch.argmax(s_score)
            rows[step] = s_row

            x_action = torch.full_like(s_row, -1 if self.noop_x else 1)
            original_row = row_map.gather(0, s_row.clamp_min(0).reshape(1))[0]
            action_pair = torch.stack([original_row.clamp_min(0) * 2, x_action]).reshape(1, 2)
            selected_t.scatter_(1, s_row.clamp_min(0).reshape(1, 1), (s_row > 0).reshape(1, 1))
            search_count = search_count + (s_row <= 0).to(x.dtype)
            track_count = track_count + (s_row > 0).to(x.dtype)
            last_is_search = (s_row <= 0).to(x.dtype)
            elapsed = elapsed + s_duration.gather(0, s_row.reshape(1))[0]
            next_slot = torch.stack(
                [
                    elapsed / 200.0,
                    search_count / 20.0,
                    track_count / 100.0,
                    last_is_search,
                    const[0],
                    const[1],
                    const[2],
                    const[3],
                    const[4],
                    const[5],
                    const[6],
                ]
            ).reshape(1, 11)
            cls, tok, _r, _dt = self.g(cls, tok, next_slot, action_pair)

        return rows


class CudaGraphLatentSOnlyWindow:
    def __init__(self, planner, x, const, s_duration, service_bonus, row_map):
        self.planner = planner
        self.device = planner.device
        self.module = LatentSOnlyWindowGraphModule(
            planner.model,
            planner.g,
            max_steps=planner.max_steps,
            policy_weight=planner.policy_weight,
            q_weight=planner.q_weight,
            search_score_bias=planner.search_score_bias,
            use_g_policy=planner.use_g_policy,
            use_model_q=abs(float(planner.q_weight)) > 0.0,
            noop_x=planner.single_sensor_noop_action,
            s_only_score=planner.cuda_graph_s_only_score,
            factorized_decode=planner.tensor_loop_factorized_decode,
        ).eval()
        self.x = x.clone()
        self.const = const.clone()
        self.s_duration = s_duration.clone()
        self.service_bonus = service_bonus.clone()
        self.row_map = row_map.clone()
        with torch.inference_mode():
            for _ in range(8):
                self.rows = self.module(self.x, self.const, self.s_duration, self.service_bonus, self.row_map)
            torch.cuda.synchronize(self.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.rows = self.module(self.x, self.const, self.s_duration, self.service_bonus, self.row_map)
            self.graph.replay()
            torch.cuda.synchronize(self.device)

    def __call__(self, x, const, s_duration, service_bonus, row_map):
        with torch.inference_mode():
            self.x.copy_(x, non_blocking=True)
            self.const.copy_(const, non_blocking=True)
            self.s_duration.copy_(s_duration, non_blocking=True)
            self.service_bonus.copy_(service_bonus, non_blocking=True)
            self.row_map.copy_(row_map, non_blocking=True)
            self.graph.replay()
            return self.rows


class LatentBoundarySOnlyWindowGraphModule(LatentSOnlyWindowGraphModule):
    """Decode one S-only window from an already predicted boundary latent."""

    def forward(self, cls, tok, token_active, const, s_duration, row_map):
        selected_t = torch.zeros_like(token_active)
        elapsed = cls.new_zeros(())
        search_count = cls.new_zeros(())
        track_count = cls.new_zeros(())
        last_is_search = cls.new_zeros(())
        rows = torch.empty((self.max_steps,), dtype=torch.long, device=cls.device)
        for step in range(self.max_steps):
            slot = torch.stack(
                [
                    elapsed / 200.0,
                    search_count / 20.0,
                    track_count / 100.0,
                    last_is_search,
                    const[0],
                    const[1],
                    const[2],
                    const[3],
                    const[4],
                    const[5],
                    const[6],
                ]
            ).reshape(1, 11)
            scores, q = self._score(cls, tok, slot, selected_t, token_active)
            utility = self.policy_weight * scores + self.q_weight * q
            s_score = utility[0].clone()
            s_score[0] = s_score[0] + self.search_score_bias
            if self.factorized_decode:
                track_mass = torch.logsumexp(s_score[1:], dim=0)
                best_track = torch.argmax(s_score[1:]) + 1
                row = torch.where(s_score[0] >= track_mass, torch.zeros_like(best_track), best_track)
            else:
                row = torch.argmax(s_score)
            rows[step] = row
            original_row = row_map.gather(0, row.clamp_min(0).reshape(1))[0]
            action_pair = torch.stack(
                [original_row.clamp_min(0) * 2, torch.full_like(original_row, -1 if self.noop_x else 1)]
            ).reshape(1, 2)
            selected_t.scatter_(1, row.clamp_min(0).reshape(1, 1), (row > 0).reshape(1, 1))
            search_count = search_count + (row <= 0).to(cls.dtype)
            track_count = track_count + (row > 0).to(cls.dtype)
            last_is_search = (row <= 0).to(cls.dtype)
            elapsed = elapsed + s_duration.gather(0, row.reshape(1))[0]
            next_slot = torch.stack(
                [
                    elapsed / 200.0,
                    search_count / 20.0,
                    track_count / 100.0,
                    last_is_search,
                    const[0],
                    const[1],
                    const[2],
                    const[3],
                    const[4],
                    const[5],
                    const[6],
                ]
            ).reshape(1, 11)
            cls, tok, _reward, _duration = self.g(cls, tok, next_slot, action_pair)
        return rows


class CudaGraphLatentBoundarySOnlyWindow:
    def __init__(self, planner, cls, tok, token_active, const, s_duration, row_map):
        self.device = planner.device
        self.module = LatentBoundarySOnlyWindowGraphModule(
            planner.model,
            planner.g,
            max_steps=planner.max_steps,
            policy_weight=planner.policy_weight,
            q_weight=planner.q_weight,
            search_score_bias=planner.search_score_bias,
            use_g_policy=planner.use_g_policy,
            use_model_q=abs(float(planner.q_weight)) > 0.0,
            noop_x=planner.single_sensor_noop_action,
            s_only_score=planner.cuda_graph_s_only_score,
            factorized_decode=planner.tensor_loop_factorized_decode,
        ).eval()
        self.cls = cls.clone()
        self.tok = tok.clone()
        self.token_active = token_active.clone()
        self.const = const.clone()
        self.s_duration = s_duration.clone()
        self.row_map = row_map.clone()
        with torch.inference_mode():
            for _ in range(4):
                self.rows = self.module(
                    self.cls, self.tok, self.token_active, self.const, self.s_duration, self.row_map
                )
            torch.cuda.synchronize(self.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.rows = self.module(
                    self.cls, self.tok, self.token_active, self.const, self.s_duration, self.row_map
                )
            self.graph.replay()
            torch.cuda.synchronize(self.device)

    def __call__(self, cls, tok, token_active, const, s_duration, row_map):
        with torch.inference_mode():
            self.cls.copy_(cls)
            self.tok.copy_(tok)
            self.token_active.copy_(token_active)
            self.const.copy_(const, non_blocking=True)
            self.s_duration.copy_(s_duration, non_blocking=True)
            self.row_map.copy_(row_map, non_blocking=True)
            self.graph.replay()
            return self.rows


def load_base_policy_model(args, device: str | torch.device):
    checkpoint = torch.load(str(args.base_state), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        d_model = int(state["cls_token"].shape[0]) if "cls_token" in state else 48
        model_args = SimpleNamespace(d_model=d_model, nhead=4, nlayers=2)
        model = make_physical_model(str(args.variant), model_args).to(device).eval()
        model.load_state_dict(state, strict=True)
        return model
    if not bool(getattr(args, "lean_base_load", False)):
        model_args = SimpleNamespace(
            targets=str(ROOT / "CreateValid1" / "results" / "edf_bootstrap_r3_lmh_1024_targets.pt"),
            model_seed=123,
            q_loss_weight=0.25,
            value_loss_weight=0.25,
            use_arrival_token_feature=False,
            search_calibration_weight=0.0,
            non_strict_load=False,
        )
        return train_variant(model_args, str(args.variant), str(args.base_state)).to(device).eval()
    state = checkpoint
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    d_model = int(state["cls_token"].shape[0]) if isinstance(state, dict) and "cls_token" in state else 48
    model_args = SimpleNamespace(d_model=d_model, nhead=4, nlayers=2)
    model = make_physical_model(str(args.variant), model_args).to(device).eval()
    model.load_state_dict(state, strict=True)
    print({"lean_base_load": True, "base_state": str(args.base_state), "variant": str(args.variant)}, flush=True)
    return model


class HybridRiskPlanner:
    def __init__(
        self,
        fast_planner,
        full_planner,
        active_threshold: int = 0,
        overdue_threshold: int = 0,
        pressure_threshold: float = 0.0,
        max_full_fraction: float = 1.0,
    ):
        self.fast_planner = fast_planner
        self.full_planner = full_planner
        self.active_threshold = int(active_threshold)
        self.overdue_threshold = int(overdue_threshold)
        self.pressure_threshold = float(pressure_threshold)
        self.max_full_fraction = float(max_full_fraction)
        self.calls = 0
        self.full_calls = 0

    def warmup(self, obs: dict) -> None:
        if hasattr(self.fast_planner, "warmup"):
            self.fast_planner.warmup(obs)
        if hasattr(self.full_planner, "warmup"):
            self.full_planner.warmup(obs)

    def _risk(self, obs: dict) -> tuple[bool, dict]:
        active = np.asarray(obs.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
        desired = np.asarray(obs.get("t_desired", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)
        deadline = np.asarray(obs.get("t_deadline", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)
        n = min(MAXT, len(active), len(desired), len(deadline))
        active_count = int(active[:n].sum()) if n > 0 else 0
        valid = active[:n] & (deadline[:n] >= 0.0) if n > 0 else np.zeros(0, dtype=bool)
        overdue = int((valid & (desired[:n] < 0.0)).sum()) if n > 0 else 0
        pressure = 0.0
        if n > 0 and bool(valid.any()):
            urgency = np.clip((1500.0 - deadline[:n]) / 1500.0, 0.0, 2.0)
            late = np.clip(-desired[:n] / 1000.0, 0.0, 4.0)
            pressure = float(np.max(np.where(valid, 1.0 + urgency + late, 0.0)))
        risky = False
        if self.active_threshold > 0 and active_count >= self.active_threshold:
            risky = True
        if self.overdue_threshold > 0 and overdue >= self.overdue_threshold:
            risky = True
        if self.pressure_threshold > 0.0 and pressure >= self.pressure_threshold:
            risky = True
        return risky, {"active": active_count, "overdue": overdue, "pressure": pressure}

    def plan(self, obs: dict, budget_ms: float = 200.0):
        self.calls += 1
        risky, _meta = self._risk(obs)
        if risky and self.max_full_fraction < 1.0:
            allowed = int(np.floor(self.max_full_fraction * float(max(1, self.calls))))
            risky = self.full_calls < max(1, allowed)
        if risky:
            self.full_calls += 1
            return self.full_planner.plan(obs, budget_ms=budget_ms)
        return self.fast_planner.plan(obs, budget_ms=budget_ms)


def action_from_row(row: int, sensor: int) -> int:
    row = int(row)
    sensor = int(sensor)
    if sensor == 0:
        return xs_s_search_action() if row <= 0 else xs_s_track_action(row)
    return xs_x_search_action() if row <= 0 else xs_x_track_action(row)


def action_index_from_row(row: int, sensor: int) -> int:
    return int(max(0, row)) * 2 + int(sensor)


class LatentMuZeroPlanner:
    def __init__(
        self,
        model,
        g: LatentG,
        env_cfg: dict,
        *,
        policy_weight: float = 1.0,
        q_weight: float = 1.0,
        search_score_bias: float = -12.0,
        adaptive_search_bias: bool = False,
        adaptive_search_bias_invert: bool = False,
        search_bias_active_threshold: int = 0,
        search_bias_overdue_threshold: int = 0,
        search_bias_pressure_threshold: float = 0.0,
        search_bias_low: float = -16.0,
        search_bias_high: float = -10.0,
        service_track_weight: float = 0.0,
        service_search_weight: float = 0.0,
        service_active_goal: float = 0.0,
        decode_router_active_threshold: int = 0,
        decode_router_low_service_track: float = 0.0,
        decode_router_low_service_search: float = 0.0,
        decode_router_low_search_cap: float = 0.0,
        decode_router_high_active_threshold: int = 0,
        decode_router_high_service_track: float = 0.0,
        decode_router_high_service_search: float = 0.0,
        decode_router_high_search_cap: float = 0.0,
        decode_router_smooth_temp: float = 0.0,
        use_learned_service_gate: bool = False,
        service_target_override: bool = False,
        avoid_double_search: bool = False,
        max_steps: int = 96,
        lookahead_width: int = 0,
        lookahead_leaf_weight: float = 0.25,
        latent_candidate_topk: int = 16,
        latent_leaf_topk: int = 8,
        service_critic_weight: float = 0.0,
        service_critic_active_weight: float = 0.25,
        service_critic_tracked_weight: float = 1.0,
        service_critic_drop_weight: float = 1.5,
        service_critic_delay_weight: float = 0.3,
        direct_action_service_weight: float = 0.0,
        direct_action_value_weight: float = 0.0,
        direct_action_frame_weight: float = 0.0,
        direct_action_value_max_steps: int = 0,
        direct_action_value_margin_threshold: float = 0.0,
        direct_action_value_cache_topn: int = 0,
        direct_action_value_cache_only: bool = False,
        direct_action_value_track_only: bool = False,
        duration_penalty_weight: float = 0.0,
        planner_duration_scale: float = 1.0,
        sensor_valid_mask: bool = False,
        fill_fallback: str = "none",
        fill_steps: int = 0,
        cuda_graph_step: bool = False,
        cuda_graph_root_encode: bool = False,
        cuda_graph_root_seq: bool = False,
        cuda_graph_ar_seq: bool = False,
        cuda_graph_ar_dynamic: bool = False,
        ar_compact_max_rows: int = 0,
        cuda_graph_tensor_loop: bool = False,
        cuda_graph_s_only_score: bool = False,
        tensor_loop: bool = False,
        tensor_loop_factorized_decode: bool = False,
        single_sensor_noop_action: bool = False,
        single_sensor_s_only_choose: bool = False,
        use_g_policy: bool = False,
        use_root_seq_policy: bool = False,
        use_base_seq_policy: bool = False,
        use_ar_seq_policy: bool = False,
        ar_policy_sampling_temperature: float = 0.0,
        ar_target_sampling_temperature: float | None = None,
        use_cached_base_policy: bool = False,
        root_seq_edf_targets: bool = False,
        root_seq_base_targets: bool = False,
        root_seq_slot_passes: int = 1,
        root_seq_step_context: bool = False,
        root_seq_select_steps: str = "",
        root_seq_recurrent_rescore: bool = False,
        root_seq_rescore_weight: float = 1.0,
        root_seq_recurrent_g_only: bool = False,
        root_seq_rescore_stride: int = 1,
        root_seq_search_bias_candidates: str = "",
        root_seq_terminal_service_rerank_weight: float = 0.0,
        root_seq_terminal_proxy_weight: float = 1.0,
        root_seq_terminal_min_plan_frac: float = 0.0,
        root_seq_terminal_min_search_atoms: int = 0,
        root_seq_debug_candidates: bool = False,
        root_seq_decode_topk: int = 16,
        root_seq_cpu_decode: bool = False,
        root_seq_factorized_decode: bool = False,
        ar_dynamic_slots: bool = False,
        decoder_history_search_streak_bias: float = 0.0,
        decoder_history_track_after_search_bias: float = 0.0,
        decoder_history_track_streak_bias: float = 0.0,
        decoder_history_search_after_track_bias: float = 0.0,
        max_window_search_frac: float = 0.0,
        max_window_search_atoms: int = 0,
        min_window_search_atoms: int = 0,
        min_window_search_active_threshold: int = 0,
        repair_search_target_frac: float = 0.0,
        root_seq_search_balance_weight: float = 0.0,
        root_seq_max_search_skew: int = 0,
        root_seq_force_search_prefix: int = 0,
        pressure_repair_threshold: float = 0.0,
        pressure_repair_max_atoms: int = 0,
        frontload_track_actions: bool = False,
        service_sort_plan: bool = False,
        service_sort_search_prefix: int = 0,
        service_sort_track_burst: int = 0,
        repair_pure_search_joints: bool = False,
        root_seq_stop_threshold: float = -1.0e30,
        use_root_seq_stop_head: bool = False,
        root_seq_stop_prob: float = 0.5,
        root_seq_min_steps: int = 0,
        g_alt: LatentG | None = None,
        g_blend_alpha: float = 1.0,
        dual_plan_select: bool = False,
        dual_plan_track_weight: float = 1.0,
        dual_plan_search_weight: float = 0.25,
        dual_plan_duration_weight: float = 0.02,
        dual_plan_pressure_weight: float = 0.0,
        router_active_threshold: int = 0,
        router_alt_search_bias: float | None = None,
        router_alt_service_track_weight: float | None = None,
        router_alt_service_search_weight: float | None = None,
        router_alt_max_window_search_frac: float | None = None,
        router_alt_service_sort_search_prefix: int | None = None,
        router_alt_pressure_repair_max_atoms: int | None = None,
        device: str = "cuda",
    ):
        self.model = model.eval()
        self.g = g.eval()
        self.g_alt = g_alt.eval() if g_alt is not None else None
        self.g_blend_alpha = float(g_blend_alpha)
        self.dual_plan_select = bool(dual_plan_select)
        self.dual_plan_track_weight = float(dual_plan_track_weight)
        self.dual_plan_search_weight = float(dual_plan_search_weight)
        self.dual_plan_duration_weight = float(dual_plan_duration_weight)
        self.dual_plan_pressure_weight = float(dual_plan_pressure_weight)
        self.router_active_threshold = int(router_active_threshold)
        self.router_alt_search_bias = None if router_alt_search_bias is None else float(router_alt_search_bias)
        self.router_alt_service_track_weight = None if router_alt_service_track_weight is None else float(router_alt_service_track_weight)
        self.router_alt_service_search_weight = None if router_alt_service_search_weight is None else float(router_alt_service_search_weight)
        self.router_alt_max_window_search_frac = None if router_alt_max_window_search_frac is None else float(router_alt_max_window_search_frac)
        self.router_alt_service_sort_search_prefix = None if router_alt_service_sort_search_prefix is None else int(router_alt_service_sort_search_prefix)
        self.router_alt_pressure_repair_max_atoms = None if router_alt_pressure_repair_max_atoms is None else int(router_alt_pressure_repair_max_atoms)
        self.env_cfg = dict(env_cfg)
        self.policy_weight = float(policy_weight)
        self.q_weight = float(q_weight)
        self.search_score_bias = float(search_score_bias)
        self.adaptive_search_bias = bool(adaptive_search_bias)
        self.adaptive_search_bias_invert = bool(adaptive_search_bias_invert)
        self.search_bias_active_threshold = int(search_bias_active_threshold)
        self.search_bias_overdue_threshold = int(search_bias_overdue_threshold)
        self.search_bias_pressure_threshold = float(search_bias_pressure_threshold)
        self.search_bias_low = float(search_bias_low)
        self.search_bias_high = float(search_bias_high)
        self.service_track_weight = float(service_track_weight)
        self.service_search_weight = float(service_search_weight)
        self.base_service_track_weight = float(service_track_weight)
        self.base_service_search_weight = float(service_search_weight)
        self.service_active_goal = float(service_active_goal)
        self.decode_router_active_threshold = int(decode_router_active_threshold)
        self.decode_router_low_service_track = float(decode_router_low_service_track)
        self.decode_router_low_service_search = float(decode_router_low_service_search)
        self.decode_router_low_search_cap = float(decode_router_low_search_cap)
        self.decode_router_high_active_threshold = int(decode_router_high_active_threshold)
        self.decode_router_high_service_track = float(decode_router_high_service_track)
        self.decode_router_high_service_search = float(decode_router_high_service_search)
        self.decode_router_high_search_cap = float(decode_router_high_search_cap)
        self.decode_router_smooth_temp = float(decode_router_smooth_temp)
        self.use_learned_service_gate = bool(use_learned_service_gate)
        self.service_target_override = bool(service_target_override)
        self.avoid_double_search = bool(avoid_double_search)
        self.max_steps = int(max_steps)
        self.lookahead_width = int(lookahead_width)
        self.lookahead_leaf_weight = float(lookahead_leaf_weight)
        self.latent_candidate_topk = int(latent_candidate_topk)
        self.latent_leaf_topk = int(latent_leaf_topk)
        self.service_critic_weight = float(service_critic_weight)
        self.service_critic_active_weight = float(service_critic_active_weight)
        self.service_critic_tracked_weight = float(service_critic_tracked_weight)
        self.service_critic_drop_weight = float(service_critic_drop_weight)
        self.service_critic_delay_weight = float(service_critic_delay_weight)
        self.direct_action_service_weight = float(direct_action_service_weight)
        self.direct_action_value_weight = float(direct_action_value_weight)
        self.direct_action_frame_weight = float(direct_action_frame_weight)
        self.direct_action_value_max_steps = int(direct_action_value_max_steps)
        self.direct_action_value_margin_threshold = float(direct_action_value_margin_threshold)
        self.direct_action_value_cache_topn = int(direct_action_value_cache_topn)
        self.direct_action_value_cache_only = bool(direct_action_value_cache_only)
        self.direct_action_value_track_only = bool(direct_action_value_track_only)
        self.duration_penalty_weight = float(duration_penalty_weight)
        self.planner_duration_scale = float(planner_duration_scale)
        self.sensor_valid_mask = bool(sensor_valid_mask)
        self.fill_fallback = str(fill_fallback)
        self.fill_steps = int(fill_steps)
        self.cuda_graph_step = bool(cuda_graph_step)
        self.cuda_graph_root_encode = bool(cuda_graph_root_encode)
        self.cuda_graph_root_seq = bool(cuda_graph_root_seq)
        self.cuda_graph_ar_seq = bool(cuda_graph_ar_seq)
        self.cuda_graph_ar_dynamic = bool(cuda_graph_ar_dynamic)
        self.ar_compact_max_rows = max(0, int(ar_compact_max_rows))
        self.cuda_graph_tensor_loop = bool(cuda_graph_tensor_loop)
        self.cuda_graph_s_only_score = bool(cuda_graph_s_only_score)
        self.tensor_loop = bool(tensor_loop)
        self.tensor_loop_factorized_decode = bool(tensor_loop_factorized_decode)
        self.single_sensor_noop_action = bool(single_sensor_noop_action)
        self.single_sensor_s_only_choose = bool(single_sensor_s_only_choose)
        self.use_g_policy = bool(use_g_policy)
        self.use_root_seq_policy = bool(use_root_seq_policy)
        self.use_base_seq_policy = bool(use_base_seq_policy)
        self.use_ar_seq_policy = bool(use_ar_seq_policy)
        self.ar_policy_sampling_temperature = max(0.0, float(ar_policy_sampling_temperature))
        self.ar_target_sampling_temperature = (
            self.ar_policy_sampling_temperature
            if ar_target_sampling_temperature is None
            else max(0.0, float(ar_target_sampling_temperature))
        )
        self.use_cached_base_policy = bool(use_cached_base_policy)
        self.root_seq_edf_targets = bool(root_seq_edf_targets)
        self.root_seq_base_targets = bool(root_seq_base_targets)
        self.root_seq_slot_passes = max(0, int(root_seq_slot_passes))
        self.root_seq_step_context = bool(root_seq_step_context)
        self.root_seq_select_steps = [
            int(x)
            for x in str(root_seq_select_steps).replace(";", ",").split(",")
            if str(x).strip()
        ]
        self.root_seq_recurrent_rescore = bool(root_seq_recurrent_rescore)
        self.root_seq_rescore_weight = float(root_seq_rescore_weight)
        self.root_seq_recurrent_g_only = bool(root_seq_recurrent_g_only)
        self.root_seq_rescore_stride = max(1, int(root_seq_rescore_stride))
        self.root_seq_search_bias_candidates = [
            float(x)
            for x in str(root_seq_search_bias_candidates).replace(";", ",").split(",")
            if str(x).strip()
        ]
        self.root_seq_terminal_service_rerank_weight = float(root_seq_terminal_service_rerank_weight)
        self.root_seq_terminal_proxy_weight = float(root_seq_terminal_proxy_weight)
        self.root_seq_terminal_min_plan_frac = float(root_seq_terminal_min_plan_frac)
        self.root_seq_terminal_min_search_atoms = max(0, int(root_seq_terminal_min_search_atoms))
        self.root_seq_debug_candidates = bool(root_seq_debug_candidates)
        self.debug_candidate_rows: list[dict] = []
        self.root_seq_decode_topk = max(1, int(root_seq_decode_topk))
        self.root_seq_cpu_decode = bool(root_seq_cpu_decode)
        self.root_seq_factorized_decode = bool(root_seq_factorized_decode)
        self.ar_dynamic_slots = bool(ar_dynamic_slots)
        self.decoder_history_search_streak_bias = float(decoder_history_search_streak_bias)
        self.decoder_history_track_after_search_bias = float(decoder_history_track_after_search_bias)
        self.decoder_history_track_streak_bias = float(decoder_history_track_streak_bias)
        self.decoder_history_search_after_track_bias = float(decoder_history_search_after_track_bias)
        self.max_window_search_frac = float(max_window_search_frac)
        self.base_max_window_search_frac = float(max_window_search_frac)
        self.max_window_search_atoms = int(max_window_search_atoms)
        self.min_window_search_atoms = max(0, int(min_window_search_atoms))
        self.min_window_search_active_threshold = max(0, int(min_window_search_active_threshold))
        self._active_count_for_window = 0
        self.repair_search_target_frac = float(repair_search_target_frac)
        self.root_seq_search_balance_weight = float(root_seq_search_balance_weight)
        self.root_seq_max_search_skew = int(root_seq_max_search_skew)
        self.root_seq_force_search_prefix = max(0, int(root_seq_force_search_prefix))
        self.pressure_repair_threshold = float(pressure_repair_threshold)
        self.pressure_repair_max_atoms = int(pressure_repair_max_atoms)
        self.frontload_track_actions = bool(frontload_track_actions)
        self.service_sort_plan = bool(service_sort_plan)
        self.service_sort_search_prefix = max(0, int(service_sort_search_prefix))
        self.service_sort_track_burst = max(0, int(service_sort_track_burst))
        self.repair_pure_search_joints = bool(repair_pure_search_joints)
        self.root_seq_stop_threshold = float(root_seq_stop_threshold)
        self.use_root_seq_stop_head = bool(use_root_seq_stop_head)
        self.root_seq_stop_prob = float(root_seq_stop_prob)
        self.root_seq_min_steps = int(root_seq_min_steps)
        self.device = torch.device(device)
        self.adapt = adapter()
        self._graph_step = None
        self._graph_encode_tokens = None
        self._graph_root_encode = None
        self._graph_root_seq = None
        self._graph_ar_seq = None
        self._graph_ar_dynamic = None
        self._graph_ar_dynamic_cache = {}
        self._graph_ar_dynamic_shape = None
        self._graph_sonly_window = None
        self._graph_sonly_window_cache = {}
        self._graph_sonly_window_shape = None
        self._graph_boundary_window = None
        self._receding_selected: set[int] = set()
        self._receding_elapsed = 0.0
        self._receding_search_count = 0
        self._receding_track_count = 0
        self._receding_last = -1
        self._s_actions = np.asarray([action_from_row(row, 0) for row in range(MAXT + 1)], dtype=np.int64)
        self._x_actions = np.asarray([action_from_row(row, 1) for row in range(MAXT + 1)], dtype=np.int64)
        self._s_actions_t = torch.as_tensor(self._s_actions, dtype=torch.long, device=self.device)
        self._x_actions_t = torch.as_tensor(self._x_actions, dtype=torch.long, device=self.device)

    def set_receding_context(
        self,
        selected: set[int],
        elapsed: float,
        search_count: int,
        track_count: int,
        last: int,
    ) -> None:
        self._receding_selected = set(int(row) for row in selected if int(row) > 0)
        self._receding_elapsed = float(elapsed)
        self._receding_search_count = int(search_count)
        self._receding_track_count = int(track_count)
        self._receding_last = int(last)

    def _tokenize_fast_np(self, obs: dict, selected: set[int] | None = None, search_count: int = 0) -> np.ndarray:
        """Direct 13-feature tokenizer equivalent to mutual_features.tokenize.

        The generic path builds an intermediate [1, rows, 8] adapter tensor and
        then copies it into the final token table.  The clean S-only AR graph
        only needs the final token table once per window, so this avoids the
        intermediate allocation and extra pass while preserving feature values.
        """
        x = np.zeros((MAXT + 1, TOKEN_DIM), dtype=np.float32)
        grid = np.asarray(obs.get("grid", np.zeros((300,), dtype=np.float32)), dtype=np.float32)
        grid_min = float(np.min(grid)) if grid.size else 0.0
        search_debt_ms = float(obs.get("search_debt_ms", 0.0))
        search_debt_norm = float(np.clip(search_debt_ms / 1000.0, 0.0, 10.0))
        active = np.asarray(obs["active_mask"]).astype(bool)
        t_desired = np.asarray(obs["t_desired"], dtype=np.float32)
        t_deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
        t_dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
        tracked = np.asarray(obs.get("tracked_mask", active & (t_deadline > 0))).astype(bool)
        tracked_active = active & tracked
        tracked_n = int(np.sum(tracked_active))
        tracked_count_norm = float(tracked_n / max(1, self.adapt.max_trackers))
        if tracked_n > 0:
            tracked_delays = np.maximum(0.0, -t_desired[tracked_active])
            mean_tracked_delay_norm = float(np.clip(np.mean(tracked_delays) / 2000.0, 0.0, 10.0))
            overdue_frac = float(np.mean(t_desired[tracked_active] < 0.0))
            global_tardiness_norm = float(np.clip(np.sum(tracked_delays) / 20000.0, 0.0, 10.0))
            tracked_deadline_pressure = np.maximum(0.0, 100.0 - t_deadline[tracked_active])
            global_deadline_pressure_norm = float(np.clip(np.sum(tracked_deadline_pressure) / 2000.0, 0.0, 10.0))
        else:
            mean_tracked_delay_norm = 0.0
            overdue_frac = 0.0
            global_tardiness_norm = 0.0
            global_deadline_pressure_norm = 0.0
        search_penalty_norm = float(np.clip(self.adapt.pure_mcts._search_delay_penalty(search_debt_ms), 0.0, 10.0))
        global_penalty_norm = float(
            np.clip(
                0.001
                * (
                    self.adapt.pure_mcts.global_tardiness_weight * global_tardiness_norm
                    + self.adapt.pure_mcts.local_tardiness_weight * mean_tracked_delay_norm
                ),
                0.0,
                10.0,
            )
        )
        x[0, 0] = np.clip(tracked_count_norm / 3000.0, -2.0, 2.0)
        x[0, 1] = np.clip(grid_min / 3000.0, -2.0, 2.0)
        x[0, 2] = np.clip(global_tardiness_norm / 100.0, 0.0, 2.0)
        x[0, 3] = mean_tracked_delay_norm
        x[0, 4] = overdue_frac
        x[0, 5] = np.clip(global_deadline_pressure_norm / 3000.0, -2.0, 2.0)
        x[0, 6] = search_penalty_norm
        x[0, 7] = global_penalty_norm
        n = min(MAXT, len(t_desired), len(t_deadline), len(t_dwell), len(active))
        if n > 0:
            az_bin = np.asarray(obs.get("az_bin", np.zeros_like(t_desired, dtype=np.float32)), dtype=np.float32)[:n]
            el_bin = np.asarray(obs.get("el_bin", np.zeros_like(t_desired, dtype=np.float32)), dtype=np.float32)[:n]
            if grid.size:
                az_idx = np.clip(np.round(az_bin * 29.0).astype(np.int32), 0, 29)
                el_idx = np.clip(np.round(el_bin * 9.0).astype(np.int32), 0, 9)
                sector_idx = np.clip(el_idx * 30 + az_idx, 0, max(0, len(grid) - 1))
                sector_urgency = grid[sector_idx].astype(np.float32)
            else:
                sector_urgency = np.zeros((n,), dtype=np.float32)
            priority = np.asarray(obs.get("priority", np.zeros_like(t_desired)), dtype=np.float32)[:n]
            target_tardiness = np.maximum(0.0, -t_desired[:n]).astype(np.float32)
            local_penalty_norm = np.clip(
                0.001 * target_tardiness * (1.0 + 2.0 * priority) * self.adapt.pure_mcts.local_tardiness_weight,
                0.0,
                10.0,
            ).astype(np.float32)
            rows = slice(1, n + 1)
            x[rows, 0] = np.clip(t_desired[:n] / 3000.0, -2.0, 2.0)
            x[rows, 1] = np.clip(t_deadline[:n] / 3000.0, -2.0, 2.0)
            x[rows, 2] = np.clip(t_dwell[:n] / 100.0, 0.0, 2.0)
            x[rows, 3] = priority
            x[rows, 4] = (active[:n] & tracked[:n]).astype(np.float32)
            x[rows, 5] = np.clip(sector_urgency / 3000.0, -2.0, 2.0)
            x[rows, 6] = local_penalty_norm
            x[rows, 7] = global_penalty_norm + search_penalty_norm
            ranges = np.asarray(obs.get("target_range", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)
            rn = min(MAXT, len(ranges), n)
            if rn > 0:
                range_norm = np.clip(ranges[:rn] / 184_000_000.0, 0.0, 1.5)
                x[1 : rn + 1, 9] = range_norm
                x[1 : rn + 1, 10] = ((ranges[:rn] > 10_000_000.0) & (ranges[:rn] < 184_000_000.0)).astype(np.float32)
                x[1 : rn + 1, 11] = ((ranges[:rn] > 5_000_000.0) & (ranges[:rn] < 100_000_000.0)).astype(np.float32)
        if selected:
            for a in selected:
                if 1 <= int(a) <= MAXT:
                    x[int(a), 8] = 1.0
        x[0, 8] = float(search_count) / 20.0
        x[:, 12] = float(obs.get("sensor_id", 0.0))
        if float(obs.get("use_arrival_token_feature", 0.0)) > 0.5:
            x[0, 12] = np.clip(float(obs.get("arrival_rate", 0.0)) / 10.0, 0.0, 2.0)
        x[0, 9] = np.clip(float(obs.get("s_band_busy_ms", 0.0)) / 200.0, 0.0, 5.0)
        x[0, 10] = np.clip(float(obs.get("x_band_busy_ms", 0.0)) / 200.0, 0.0, 5.0)
        x[0, 11] = float(obs.get("enable_x_band", 0.0))
        if float(obs.get("use_grid_feature", 0.0)) > 0.5:
            if grid.size == 0:
                mean_overdue, drop_frac, max_age = 0.0, 0.0, 0.0
            else:
                age = 3000.0 - grid
                overdue = np.maximum(0.0, age - 3000.0) / 3000.0
                mean_overdue = np.clip(float(np.mean(overdue)), 0.0, 5.0)
                drop_frac = np.clip(float(np.mean(age > 4500.0)), 0.0, 1.0)
                max_age = np.clip(float(np.max(age) / 4500.0), 0.0, 5.0)
            x[0, 9] = mean_overdue
            x[0, 10] = drop_frac
            x[0, 11] = max_age
        return x

    def _compact_sonly_tokens_and_dwell(self, x_np: np.ndarray, obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        cap = int(self.ar_compact_max_rows)
        if cap <= 0 or cap >= int(x_np.shape[0]):
            return None
        active_rows = np.flatnonzero(x_np[:, 4] > 0.5).astype(np.int64)
        active_rows = active_rows[(active_rows > 0) & (active_rows <= MAXT)]
        needed = int(active_rows.size) + 1
        if needed > cap:
            return None
        row_map = np.zeros((cap,), dtype=np.int64)
        row_map[0] = 0
        if active_rows.size:
            row_map[1:needed] = active_rows[: cap - 1]
        x_comp = np.zeros((cap, x_np.shape[1]), dtype=np.float32)
        x_comp[0] = x_np[0]
        if active_rows.size:
            x_comp[1:needed] = x_np[active_rows[: cap - 1]]
        dwell_np = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        dwell_comp = np.ones((cap,), dtype=np.float32) * 10.0
        mapped = row_map[1:needed]
        valid = (mapped > 0) & (mapped - 1 < int(dwell_np.size))
        if np.any(valid):
            dwell_comp[1:needed][valid] = np.maximum(1.0, dwell_np[mapped[valid] - 1])
        return x_comp, dwell_comp, row_map

    def _candidate_k(self, n_rows: int) -> int:
        if self.latent_candidate_topk <= 0:
            return max(1, int(n_rows))
        return min(max(1, int(self.latent_candidate_topk)), int(n_rows))

    def _leaf_k(self, n_rows: int) -> int:
        if self.latent_leaf_topk <= 0:
            return max(1, int(n_rows))
        return min(max(1, int(self.latent_leaf_topk)), int(n_rows))

    def _apply_decode_router(self, obs: dict) -> None:
        active = np.asarray(obs.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
        self._active_count_for_window = int(active[:MAXT].sum())
        if self.decode_router_active_threshold <= 0 and self.decode_router_high_active_threshold <= 0:
            return
        active_count = int(self._active_count_for_window)
        if self.use_learned_service_gate and hasattr(self.g, "service_gate_logit"):
            slot_np = slot_features(attach_env_obs(obs, self.env_cfg, True, True), 0.0, 0, 0, -1, 200.0).astype(np.float32)
            slot_t = torch.from_numpy(slot_np[None]).float().to(self.device)
            with torch.inference_mode():
                gate = float(torch.sigmoid(self.g.service_gate_logit(slot_t))[0].detach().cpu())
            if self.decode_router_high_active_threshold > 0 and active_count >= int(self.decode_router_high_active_threshold):
                routed_track = self.decode_router_high_service_track
                routed_search = self.decode_router_high_service_search
                routed_cap = self.decode_router_high_search_cap
            elif self.decode_router_active_threshold > 0 and active_count < int(self.decode_router_active_threshold):
                routed_track = self.decode_router_low_service_track
                routed_search = self.decode_router_low_service_search
                routed_cap = self.decode_router_low_search_cap
            else:
                routed_track = self.base_service_track_weight
                routed_search = self.base_service_search_weight
                routed_cap = self.base_max_window_search_frac
            self.service_track_weight = (1.0 - gate) * self.base_service_track_weight + gate * routed_track
            self.service_search_weight = (1.0 - gate) * self.base_service_search_weight + gate * routed_search
            self.max_window_search_frac = (1.0 - gate) * self.base_max_window_search_frac + gate * routed_cap
        elif self.decode_router_smooth_temp > 0.0:
            temp = max(1e-3, float(self.decode_router_smooth_temp))
            low_gate = 0.0
            high_gate = 0.0
            if self.decode_router_active_threshold > 0:
                low_gate = 1.0 / (1.0 + float(np.exp((active_count - float(self.decode_router_active_threshold)) / temp)))
            if self.decode_router_high_active_threshold > 0:
                high_gate = 1.0 / (1.0 + float(np.exp((float(self.decode_router_high_active_threshold) - active_count) / temp)))
            gate_total = min(1.0, low_gate + high_gate)
            low_share = low_gate / max(1e-6, low_gate + high_gate)
            high_share = high_gate / max(1e-6, low_gate + high_gate)
            routed_track = low_share * self.decode_router_low_service_track + high_share * self.decode_router_high_service_track
            routed_search = low_share * self.decode_router_low_service_search + high_share * self.decode_router_high_service_search
            routed_cap = low_share * self.decode_router_low_search_cap + high_share * self.decode_router_high_search_cap
            self.service_track_weight = (1.0 - gate_total) * self.base_service_track_weight + gate_total * routed_track
            self.service_search_weight = (1.0 - gate_total) * self.base_service_search_weight + gate_total * routed_search
            self.max_window_search_frac = (1.0 - gate_total) * self.base_max_window_search_frac + gate_total * routed_cap
        elif self.decode_router_high_active_threshold > 0 and active_count >= int(self.decode_router_high_active_threshold):
            self.service_track_weight = float(self.decode_router_high_service_track)
            self.service_search_weight = float(self.decode_router_high_service_search)
            self.max_window_search_frac = float(self.decode_router_high_search_cap)
        elif self.decode_router_active_threshold > 0 and active_count < int(self.decode_router_active_threshold):
            self.service_track_weight = float(self.decode_router_low_service_track)
            self.service_search_weight = float(self.decode_router_low_service_search)
            self.max_window_search_frac = float(self.decode_router_low_search_cap)
        else:
            self.service_track_weight = float(self.base_service_track_weight)
            self.service_search_weight = float(self.base_service_search_weight)
            self.max_window_search_frac = float(self.base_max_window_search_frac)

    def _fallback_rows(self, obs: dict) -> tuple[int, int]:
        mode = self.fill_fallback.lower()
        active = np.asarray(obs.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
        deadline = np.asarray(obs.get("t_deadline", np.full(MAXT, -1.0, dtype=np.float32)), dtype=np.float32)
        desired = np.asarray(obs.get("t_desired", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)
        dwell = np.asarray(obs.get("t_dwell", np.ones(MAXT, dtype=np.float32)), dtype=np.float32)
        n = min(MAXT, len(active), len(deadline), len(desired), len(dwell))
        if n <= 0:
            return 0, 0
        valid = active[:n] & (deadline[:n] >= 0.0)
        rows = np.nonzero(valid)[0] + 1
        if rows.size == 0:
            return 0, 0
        if mode == "est":
            score = -desired[rows - 1]
        elif mode == "short_edf":
            score = -deadline[rows - 1] - 0.1 * dwell[rows - 1]
        else:
            score = -deadline[rows - 1]
        order = rows[np.argsort(-score)]
        s_row = int(order[0])
        x_row = int(order[1]) if len(order) > 1 else 0
        return s_row, x_row

    def _current_search_bias(self, obs: dict) -> float:
        if not self.adaptive_search_bias:
            return float(self.search_score_bias)
        triggered = False
        active = np.asarray(obs.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
        desired = np.asarray(obs.get("t_desired", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)
        deadline = np.asarray(obs.get("t_deadline", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)
        n = min(MAXT, len(active), len(desired), len(deadline))
        active_count = int(active[:n].sum()) if n > 0 else 0
        if self.search_bias_active_threshold > 0 and active_count >= self.search_bias_active_threshold:
            triggered = True
        if n > 0:
            valid = active[:n] & (deadline[:n] >= 0.0)
            overdue = int((valid & (desired[:n] < 0.0)).sum())
            if self.search_bias_overdue_threshold > 0 and overdue >= self.search_bias_overdue_threshold:
                triggered = True
            if (not triggered) and self.search_bias_pressure_threshold > 0.0 and bool(valid.any()):
                urgency = np.clip((1500.0 - deadline[:n]) / 1500.0, 0.0, 2.0)
                late = np.clip(-desired[:n] / 1000.0, 0.0, 4.0)
                pressure = float(np.max(np.where(valid, 1.0 + urgency + late, 0.0)))
                if pressure >= self.search_bias_pressure_threshold:
                    triggered = True
        if self.adaptive_search_bias_invert:
            return float(self.search_bias_low if triggered else self.search_bias_high)
        return float(self.search_bias_high if triggered else self.search_bias_low)

    def _edf_target_row(self, obs: dict, sensor: int, selected: set[int]) -> int:
        active = np.asarray(obs.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
        deadline = np.asarray(obs.get("t_deadline", np.full(MAXT, -1.0, dtype=np.float32)), dtype=np.float32)
        rng = np.asarray(obs.get("target_range", np.full(MAXT, 50_000_000.0, dtype=np.float32)), dtype=np.float32)
        n = min(MAXT, len(active), len(deadline), len(rng))
        if n <= 0:
            return 0
        alive = active[:n] & (deadline[:n] >= 0.0)
        if int(sensor) == 0:
            valid = alive & (10_000_000.0 < rng[:n]) & (rng[:n] < 184_000_000.0)
        else:
            valid = alive & (5_000_000.0 < rng[:n]) & (rng[:n] < 100_000_000.0)
        for row in selected:
            if 1 <= int(row) <= n:
                valid[int(row) - 1] = False
        rows = np.nonzero(valid)[0] + 1
        if rows.size == 0:
            return 0
        return int(rows[np.argmin(deadline[rows - 1])])

    def _base_target_rankings(self, obs: dict) -> tuple[list[int], list[int]]:
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        slot_np = slot_features(obs, 0.0, 0, 0, -1, 200.0).astype(np.float32)
        x_np = tokenize(self.adapt, obs, selected=set(), search_count=0).astype(np.float32)
        x = torch.from_numpy(x_np[None]).float().to(self.device)
        slot = torch.from_numpy(slot_np[None]).float().to(self.device)
        with torch.inference_mode():
            p_scores, q_scores = self.model.forward_scores(x, slot)
            score = self.policy_weight * p_scores[0] + self.q_weight * q_scores[0]
            if self.service_track_weight != 0.0 or self.service_search_weight != 0.0:
                score = score + torch.from_numpy(self._service_bonus_np(obs)).to(self.device)
            if self.duration_penalty_weight != 0.0:
                dwell = np.asarray(obs.get("t_dwell", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)
                duration_penalty = np.zeros((MAXT + 1, 2), dtype=np.float32)
                n = min(MAXT, len(dwell))
                if n > 0:
                    duration_penalty[1 : n + 1, 0] = np.maximum(1.0, dwell[:n])
                    duration_penalty[1 : n + 1, 1] = np.maximum(1.0, dwell[:n] * 0.5)
                score = score - self.duration_penalty_weight * torch.from_numpy(duration_penalty).to(self.device)
            valid = torch.from_numpy(self._sensor_valid_np(obs)).to(self.device)
            score = score.masked_fill(~valid, -1e9)
            score[0, :] = -1e9
            ranks = []
            for sensor in (0, 1):
                order = torch.argsort(score[:, sensor], descending=True).detach().cpu().tolist()
                ranks.append([int(r) for r in order if int(r) > 0 and float(score[int(r), sensor].detach().cpu()) > -1e8])
        return ranks[0], ranks[1]

    def _base_target_row(self, rankings: tuple[list[int], list[int]] | None, sensor: int, selected: set[int]) -> int:
        if rankings is None:
            return 0
        for row in rankings[int(sensor)]:
            if int(row) not in selected:
                return int(row)
        return 0

    def _append_fill_actions(self, obs: dict, plan: list[int]) -> list[int]:
        if self.fill_fallback.lower() == "none" or self.fill_steps <= 0:
            return plan
        if self.fill_fallback.lower() in {"search", "pure_search"}:
            fill = encode_joint_action(action_from_row(0, 0), action_from_row(0, 1))
            plan.extend([int(fill)] * int(self.fill_steps))
            return plan
        if self.fill_fallback.lower() == "rootseq_tail":
            old_fill = self.fill_fallback
            old_root = self.use_root_seq_policy
            try:
                self.fill_fallback = "none"
                self.use_root_seq_policy = True
                tail = self._plan_root_sequence(obs, 200.0)
            finally:
                self.use_root_seq_policy = old_root
                self.fill_fallback = old_fill
            if tail:
                plan.extend([int(a) for a in tail[: int(self.fill_steps)]])
            return plan
        if self.fill_fallback.lower() == "urgent_short":
            selected: set[int] = set()
            for action in plan:
                atoms = split_joint_action(int(action)) if is_joint_action(int(action)) else (int(action),)
                for atom in atoms:
                    row, _sensor = xs_decode_action(int(atom), MAXT)
                    if int(row) > 0:
                        selected.add(int(row))
            active = np.asarray(obs.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
            deadline = np.asarray(obs.get("t_deadline", np.full(MAXT, -1.0, dtype=np.float32)), dtype=np.float32)
            desired = np.asarray(obs.get("t_desired", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)
            dwell = np.asarray(obs.get("t_dwell", np.ones(MAXT, dtype=np.float32)), dtype=np.float32)
            rng = np.asarray(obs.get("target_range", np.full(MAXT, 50_000_000.0, dtype=np.float32)), dtype=np.float32)
            n = min(MAXT, len(active), len(deadline), len(desired), len(dwell), len(rng))
            def ranked(sensor: int) -> list[int]:
                if n <= 0:
                    return []
                alive = active[:n] & (deadline[:n] >= 0.0)
                if int(sensor) == 0:
                    valid = alive & (10_000_000.0 < rng[:n]) & (rng[:n] < 184_000_000.0)
                else:
                    valid = alive & (5_000_000.0 < rng[:n]) & (rng[:n] < 100_000_000.0)
                rows = np.nonzero(valid)[0] + 1
                if rows.size == 0:
                    return []
                urgency = -deadline[rows - 1] + 0.5 * np.maximum(-desired[rows - 1], 0.0) - 0.2 * dwell[rows - 1]
                return [int(r) for r in rows[np.argsort(-urgency)]]
            s_rank = ranked(0)
            x_rank = ranked(1)
            for _ in range(int(self.fill_steps)):
                s_row = next((r for r in s_rank if r not in selected), 0)
                if s_row > 0:
                    selected.add(s_row)
                x_row = next((r for r in x_rank if r not in selected), 0)
                if x_row > 0:
                    selected.add(x_row)
                if s_row <= 0 and x_row <= 0:
                    break
                plan.append(encode_joint_action(action_from_row(s_row, 0), action_from_row(x_row, 1)))
            return plan
        s_row, x_row = self._fallback_rows(obs)
        if s_row <= 0 and x_row <= 0:
            fill = encode_joint_action(action_from_row(0, 0), action_from_row(0, 1))
        else:
            fill = encode_joint_action(action_from_row(s_row, 0), action_from_row(x_row, 1))
        plan.extend([int(fill)] * int(self.fill_steps))
        return plan

    def _repair_excess_search(self, obs: dict, plan: list[int]) -> list[int]:
        if self.repair_search_target_frac <= 0.0 or not plan:
            return plan
        selected: set[int] = set()
        search_atoms = 0
        total_atoms = 0
        repaired: list[int] = []
        for action in plan:
            atoms = split_joint_action(int(action)) if is_joint_action(int(action)) else (int(action),)
            rows = []
            for sensor in (0, 1):
                atom = atoms[sensor] if sensor < len(atoms) else atoms[-1]
                row, _sensor = xs_decode_action(int(atom), MAXT)
                rows.append(int(row))
            if self.repair_pure_search_joints and rows[0] <= 0 and rows[1] <= 0:
                projected_search = search_atoms + 2
                projected_total = total_atoms + 2
                if projected_search / max(1, projected_total) > self.repair_search_target_frac:
                    s_repl = self._edf_target_row(obs, 0, selected)
                    x_selected = set(selected)
                    if s_repl > 0:
                        x_selected.add(int(s_repl))
                    x_repl = self._edf_target_row(obs, 1, x_selected)
                    if s_repl > 0 or x_repl > 0:
                        rows[0] = int(s_repl)
                        rows[1] = int(x_repl)
            for sensor in (0, 1):
                projected_search = search_atoms + (1 if rows[sensor] <= 0 else 0)
                projected_total = total_atoms + 1
                if rows[sensor] <= 0 and projected_search / max(1, projected_total) > self.repair_search_target_frac:
                    other_selected = set(selected)
                    if sensor == 1 and rows[0] > 0:
                        other_selected.add(int(rows[0]))
                    repl = self._edf_target_row(obs, sensor, other_selected)
                    if repl > 0:
                        rows[sensor] = int(repl)
                if rows[sensor] <= 0:
                    search_atoms += 1
                else:
                    selected.add(int(rows[sensor]))
                total_atoms += 1
            repaired.append(encode_joint_action(action_from_row(rows[0], 0), action_from_row(rows[1], 1)))
        return repaired

    def _repair_pressure_searches(self, obs: dict, plan: list[int]) -> list[int]:
        if self.pressure_repair_threshold <= 0.0 or self.pressure_repair_max_atoms == 0 or not plan:
            return plan
        pressure = self._service_pressure_np(obs)
        valid = self._sensor_valid_np(obs)
        threshold = float(self.pressure_repair_threshold)
        top_pressure_rows = [int(r) for r in np.argsort(-pressure)[:64]]
        ranked_by_sensor: list[list[int]] = []
        for sensor in (0, 1):
            ranked_by_sensor.append(
                [
                    int(r)
                    for r in top_pressure_rows
                    if int(r) > 0 and bool(valid[int(r), sensor]) and np.isfinite(pressure[int(r)]) and float(pressure[int(r)]) > threshold
                ]
            )
        selected: set[int] = set()
        repaired: list[int] = []
        replacements = 0
        max_replacements = int(self.pressure_repair_max_atoms)
        for action in plan:
            atoms = split_joint_action(int(action)) if is_joint_action(int(action)) else (int(action),)
            rows = []
            for sensor in (0, 1):
                atom = atoms[sensor] if sensor < len(atoms) else atoms[-1]
                row, _sensor = xs_decode_action(int(atom), MAXT)
                rows.append(int(row))
            for sensor in (0, 1):
                if rows[sensor] > 0:
                    selected.add(int(rows[sensor]))
                    continue
                if max_replacements > 0 and replacements >= max_replacements:
                    continue
                best = 0
                for cand in ranked_by_sensor[sensor]:
                    if cand not in selected:
                        best = cand
                        break
                if best > 0:
                    rows[sensor] = int(best)
                    selected.add(int(best))
                    replacements += 1
            repaired.append(encode_joint_action(action_from_row(rows[0], 0), action_from_row(rows[1], 1)))
        return repaired

    def _frontload_tracks(self, plan: list[int]) -> list[int]:
        if not self.frontload_track_actions or not plan:
            return plan
        keyed: list[tuple[int, int, int]] = []
        for idx, action in enumerate(plan):
            atoms = split_joint_action(int(action)) if is_joint_action(int(action)) else (int(action),)
            track_atoms = 0
            for atom in atoms:
                row, _sensor = xs_decode_action(int(atom), MAXT)
                if int(row) > 0:
                    track_atoms += 1
            keyed.append((-track_atoms, idx, int(action)))
        keyed.sort()
        return [action for _neg_tracks, _idx, action in keyed]

    def _service_sort_actions(self, obs: dict, plan: list[int]) -> list[int]:
        if not self.service_sort_plan or not plan:
            return plan
        deadline = np.asarray(obs.get("t_deadline", np.full(MAXT, 1e9, dtype=np.float32)), dtype=np.float32)
        desired = np.asarray(obs.get("t_desired", np.full(MAXT, 1e9, dtype=np.float32)), dtype=np.float32)
        priority = np.asarray(obs.get("priority", np.ones(MAXT, dtype=np.float32)), dtype=np.float32)
        dwell = np.asarray(obs.get("t_dwell", np.ones(MAXT, dtype=np.float32) * 10.0), dtype=np.float32)
        track_keyed: list[tuple[float, int, int]] = []
        search_keyed: list[tuple[int, int]] = []
        for idx, action in enumerate(plan):
            atoms = split_joint_action(int(action)) if is_joint_action(int(action)) else (int(action),)
            rows = []
            search_atoms = 0
            for atom in atoms:
                row, _sensor = xs_decode_action(int(atom), MAXT)
                if int(row) > 0:
                    rows.append(int(row))
                else:
                    search_atoms += 1
            if search_atoms > 0 and (self.min_window_search_atoms > 0 or not rows):
                search_keyed.append((idx, int(action)))
            elif rows:
                urgency_scores = []
                duration = 0.0
                for row in rows:
                    j = int(row) - 1
                    if 0 <= j < len(deadline):
                        dl = float(deadline[j])
                        de = float(desired[j]) if j < len(desired) else dl
                        pri = float(priority[j]) if j < len(priority) else 1.0
                        dw = float(dwell[j]) if j < len(dwell) else 10.0
                        urgency_scores.append(dl + 0.5 * de - 25.0 * pri)
                        duration += max(1.0, dw)
                key = min(urgency_scores) + 0.02 * duration - 10000.0 * len(rows)
                track_keyed.append((float(key), idx, int(action)))
            else:
                # Keep pure searches after urgent track-bearing actions, but preserve their relative order.
                search_keyed.append((idx, int(action)))
        track_keyed.sort()
        tracks = [action for _key, _idx, action in track_keyed]
        searches = [action for _idx, action in search_keyed]
        if self.service_sort_search_prefix <= 0 and self.service_sort_track_burst <= 0:
            return tracks + searches
        out: list[int] = []
        si = 0
        ti = 0
        min_prefix = int(np.ceil(float(self.min_window_search_atoms) / 2.0)) if self.min_window_search_atoms > 0 else 0
        prefix = min(max(self.service_sort_search_prefix, min_prefix), len(searches))
        if prefix > 0:
            out.extend(searches[:prefix])
            si = prefix
        burst = self.service_sort_track_burst if self.service_sort_track_burst > 0 else len(tracks)
        while ti < len(tracks) or si < len(searches):
            if ti < len(tracks):
                take = min(burst, len(tracks) - ti)
                out.extend(tracks[ti : ti + take])
                ti += take
            if si < len(searches):
                out.append(searches[si])
                si += 1
        return out

    def _apply_sensor_search_balance(self, score: torch.Tensor, s_search_count: int, x_search_count: int) -> torch.Tensor:
        if self.root_seq_search_balance_weight != 0.0:
            skew = int(s_search_count) - int(x_search_count)
            if skew > 0:
                score[0, 1] = score[0, 1] + float(self.root_seq_search_balance_weight) * float(skew)
            elif skew < 0:
                score[0, 0] = score[0, 0] + float(self.root_seq_search_balance_weight) * float(-skew)
        if self.root_seq_max_search_skew > 0:
            skew = int(s_search_count) - int(x_search_count)
            if skew >= int(self.root_seq_max_search_skew):
                score[0, 0] = -1e9
            elif -skew >= int(self.root_seq_max_search_skew):
                score[0, 1] = -1e9
        return score

    def _plan_proxy_score(self, obs: dict, plan: list[int]) -> float:
        selected: set[int] = set()
        search_atoms = 0
        duration = 0.0
        pressure = self._service_pressure_np(obs) if self.dual_plan_pressure_weight != 0.0 else None
        for action in plan:
            action = int(action)
            atoms = split_joint_action(action) if is_joint_action(action) else (action,)
            duration += float(joint_duration(obs, action))
            for atom in atoms:
                row, _sensor = xs_decode_action(int(atom), MAXT)
                row = int(row)
                if row <= 0:
                    search_atoms += 1
                else:
                    selected.add(row)
        pressure_score = 0.0
        if pressure is not None:
            for row in selected:
                if 0 <= int(row) < len(pressure):
                    val = float(pressure[int(row)])
                    if np.isfinite(val) and val > -1e8:
                        pressure_score += val
        return (
            self.dual_plan_track_weight * float(len(selected))
            + self.dual_plan_pressure_weight * float(pressure_score)
            - self.dual_plan_search_weight * float(search_atoms)
            - self.dual_plan_duration_weight * float(duration)
        )

    def _plan_root_sequence(self, obs: dict, budget_ms: float = 200.0):
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        single_sensor = not bool(int(self.env_cfg.get("enable_x_band", 1)))
        clean_ar_dynamic = (
            self.use_ar_seq_policy
            and hasattr(self.g, "ar_step_scores")
            and self.cuda_graph_ar_dynamic
            and self.device.type == "cuda"
            and single_sensor
            and self.root_seq_factorized_decode
            and self.g_alt is None
            and self.service_track_weight == 0.0
            and self.service_search_weight == 0.0
            and self.direct_action_service_weight == 0.0
            and self.direct_action_value_weight == 0.0
            and self.direct_action_frame_weight == 0.0
            and self.duration_penalty_weight == 0.0
            and not self.sensor_valid_mask
            and self.max_window_search_frac <= 0.0
            and self.max_window_search_atoms <= 0
            and self.min_window_search_atoms <= 0
            and self.decode_router_active_threshold <= 0
            and self.decode_router_high_active_threshold <= 0
            and not self.use_learned_service_gate
        )
        initial_selected = set(self._receding_selected)
        initial_elapsed = float(self._receding_elapsed)
        initial_search_count = int(self._receding_search_count)
        initial_track_count = int(self._receding_track_count)
        initial_last = int(self._receding_last)
        slot_np = slot_features(
            obs,
            initial_elapsed,
            initial_search_count,
            initial_track_count,
            initial_last,
            200.0,
        ).astype(np.float32)
        if clean_ar_dynamic:
            x_np = tokenize(
                self.adapt,
                obs,
                selected=initial_selected,
                search_count=initial_search_count,
            ).astype(np.float32)
            row_map = None
            compact = self._compact_sonly_tokens_and_dwell(x_np, obs)
            if compact is not None:
                x_np, dwell_full, row_map = compact
            else:
                dwell_np = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
                dwell_full = np.ones((MAXT + 1,), dtype=np.float32) * 10.0
                n_dwell = min(MAXT, int(dwell_np.size))
                if n_dwell > 0:
                    dwell_full[1 : n_dwell + 1] = np.maximum(1.0, dwell_np[:n_dwell])
            x = torch.from_numpy(x_np[None]).float().to(self.device)
            slot = torch.from_numpy(slot_np).float().to(self.device)
            dwell_t = torch.from_numpy(dwell_full).to(self.device)
            if row_map is None:
                row_map_np = np.arange(int(x_np.shape[0]), dtype=np.int64)
            else:
                row_map_np = row_map.astype(np.int64, copy=False)
            row_map_t = torch.from_numpy(row_map_np).to(self.device)
            graph_steps = min(int(getattr(self.g, "seq_len", self.max_steps)), int(self.max_steps))
            search_bias = self._current_search_bias(obs)
            graph_shape = (int(x.shape[1]), int(dwell_t.shape[0]))
            graph = self._graph_ar_dynamic_cache.get(graph_shape)
            if graph is None:
                graph = CudaGraphARDynamicSOnlyDecode(
                    self.model,
                    self.g,
                    x,
                    slot,
                    dwell_t,
                    row_map_t,
                    max_steps=graph_steps,
                    search_score_bias=float(search_bias),
                    sample_temperature=float(self.ar_policy_sampling_temperature),
                    target_sample_temperature=float(self.ar_target_sampling_temperature),
                )
                self._graph_ar_dynamic_cache[graph_shape] = graph
                self._graph_ar_dynamic = graph
                self._graph_ar_dynamic_shape = graph_shape
            with torch.inference_mode():
                rows_t = graph(x, slot, dwell_t, row_map_t)
                rows = rows_t.detach().cpu().tolist()
            out_graph = []
            for row in rows:
                row = int(row)
                if row < 0:
                    break
                if row_map is not None:
                    row = int(row_map[int(max(0, min(len(row_map) - 1, row)))])
                out_graph.append(int(self._s_actions[int(max(0, min(MAXT, row)))]))
            return out_graph
        x_np = tokenize(
            self.adapt,
            obs,
            selected=initial_selected,
            search_count=initial_search_count,
        ).astype(np.float32)
        x = torch.from_numpy(x_np[None]).float().to(self.device)
        slot = torch.from_numpy(slot_np[None]).float().to(self.device)
        self._apply_decode_router(obs)
        base_rankings = self._base_target_rankings(obs) if self.root_seq_base_targets else None
        duration_penalty = None
        search_bias = self._current_search_bias(obs)
        if self.duration_penalty_weight != 0.0:
            dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
            s_duration = np.full(MAXT + 1, 10.0, dtype=np.float32)
            x_duration = np.full(MAXT + 1, 10.0, dtype=np.float32)
            n_dwell = min(MAXT, len(dwell))
            if n_dwell > 0:
                s_duration[1 : n_dwell + 1] = np.maximum(1.0, dwell[:n_dwell])
                x_duration[1 : n_dwell + 1] = np.maximum(1.0, dwell[:n_dwell] * 0.5)
            duration_np = np.stack([s_duration, x_duration], axis=1).astype(np.float32)
            duration_penalty = torch.from_numpy(self.duration_penalty_weight * duration_np / 10.0).to(self.device)
        service_pressure = self._service_pressure_np(obs)
        with torch.inference_mode():
            if self.cuda_graph_root_encode and self.device.type == "cuda":
                if self._graph_encode_tokens is None:
                    self._graph_encode_tokens = CudaGraphEncodeTokens(self.model, x)
                cls, tok, selected_t, token_active = self._graph_encode_tokens(x)
            else:
                cls, tok, selected_t, token_active = self.model.backbone.encode_tokens(x)
            if single_sensor:
                s_valid = x[:, :, 10] > 0.5
                s_valid[:, 0] = True
                token_active = token_active & s_valid

            def build_seq_slots(prev_plan: list[int] | None, steps_override: int | None = None) -> torch.Tensor | None:
                steps = int(getattr(self.g, "seq_len", self.max_steps))
                if prev_plan is None:
                    return None
                if len(prev_plan) == 0:
                    slots = np.tile(slot_np[None, :], (steps, 1)).astype(np.float32, copy=True)
                    denom = float(max(1, int(self.max_steps)))
                    slots[:, 0] = np.arange(steps, dtype=np.float32) / denom
                    return torch.from_numpy(slots[None]).float().to(self.device)
                spent = 0.0
                search_count = 0
                track_count = 0
                last = -1
                slots = np.zeros((steps, slot_np.shape[0]), dtype=np.float32)
                for step in range(steps):
                    slots[step] = slot_features(obs, spent, search_count, track_count, last, 200.0).astype(np.float32)
                    if step >= len(prev_plan):
                        spent = min(200.0, spent + 200.0 / max(1, int(self.max_steps)))
                        continue
                    action = int(prev_plan[step])
                    atoms = split_joint_action(action) if is_joint_action(action) else (action,)
                    for atom in atoms:
                        row, _sensor = xs_decode_action(int(atom), MAXT)
                        row = int(row)
                        if row == 0:
                            search_count += 1
                        elif row > 0:
                            track_count += 1
                        last = row
                    spent = min(200.0, spent + float(joint_duration(obs, action)))
                return torch.from_numpy(slots[None]).float().to(self.device)

            def score_and_decode(prev_plan: list[int] | None = None, max_steps_override: int | None = None) -> list[int]:
                max_steps_local = int(max_steps_override) if max_steps_override is not None else int(self.max_steps)
                seq_slots = build_seq_slots(prev_plan, max_steps_local)
                factor_type_scores = None
                factor_target_scores = None

                def pick_single_sensor_row(score: torch.Tensor, step_i: int, selected_mask_arg: torch.Tensor | None = None) -> int:
                    if not self.root_seq_factorized_decode:
                        return int(torch.argmax(score[:, 0]).detach().cpu())
                    if factor_type_scores is not None and factor_target_scores is not None:
                        search_allowed = bool((torch.isfinite(score[0, 0]) & (score[0, 0] > -1e8)).detach().cpu())
                        track_allowed = torch.isfinite(score[1:, 0]) & (score[1:, 0] > -1e8)
                        type_row = int(torch.argmax(factor_type_scores[step_i, 0, :]).detach().cpu())
                        if type_row == 0:
                            if search_allowed:
                                return 0
                            if bool(track_allowed.any().detach().cpu()):
                                target_col = factor_target_scores[step_i, :, 0].clone()
                                target_col[0] = -1e9
                                target_col[1:].masked_fill_(~track_allowed.to(device=target_col.device), -1e9)
                                if selected_mask_arg is not None and selected_mask_arg[1:].any():
                                    mask = selected_mask_arg[1:].to(device=target_col.device)
                                    target_col[1:].masked_fill_(mask, -1e9)
                                return int(torch.argmax(target_col).detach().cpu())
                            return int(torch.argmax(score[:, 0]).detach().cpu())
                        target_col = factor_target_scores[step_i, :, 0].clone()
                        target_col[0] = -1e9
                        target_col[1:].masked_fill_(~track_allowed.to(device=target_col.device), -1e9)
                        if selected_mask_arg is not None and selected_mask_arg[1:].any():
                            mask = selected_mask_arg[1:].to(device=target_col.device)
                            target_col[1:].masked_fill_(mask, -1e9)
                        if bool((torch.isfinite(target_col) & (target_col > -1e8)).any().detach().cpu()):
                            return int(torch.argmax(target_col).detach().cpu())
                        return 0 if search_allowed else int(torch.argmax(score[:, 0]).detach().cpu())
                    search_logit = score[0, 0]
                    track_logits = score[1:, 0]
                    track_logit = torch.logsumexp(track_logits, dim=0)
                    if bool((search_logit >= track_logit).detach().cpu()):
                        return 0
                    return int(torch.argmax(track_logits).detach().cpu()) + 1

                def pick_single_sensor_row_t(score: torch.Tensor, step_i: int, selected_mask_arg: torch.Tensor | None = None) -> torch.Tensor:
                    if not self.root_seq_factorized_decode:
                        return torch.argmax(score[:, 0])
                    if factor_type_scores is not None and factor_target_scores is not None:
                        search_allowed = torch.isfinite(score[0, 0]) & (score[0, 0] > -1e8)
                        track_allowed = torch.isfinite(score[1:, 0]) & (score[1:, 0] > -1e8)
                        type_row_t = torch.argmax(factor_type_scores[step_i, 0, :])
                        target_col = factor_target_scores[step_i, :, 0].clone()
                        target_col[0] = -1e9
                        target_col[1:].masked_fill_(~track_allowed.to(device=target_col.device), -1e9)
                        if selected_mask_arg is not None and selected_mask_arg[1:].any():
                            mask = selected_mask_arg[1:].to(device=target_col.device)
                            target_col[1:].masked_fill_(mask, -1e9)
                        best_track = torch.argmax(target_col)
                        has_track = (torch.isfinite(target_col) & (target_col > -1e8)).any()
                        fallback = torch.argmax(score[:, 0])
                        search_row = torch.where(search_allowed, torch.zeros_like(best_track), torch.where(has_track, best_track, fallback))
                        track_row = torch.where(has_track, best_track, torch.where(search_allowed, torch.zeros_like(best_track), fallback))
                        return torch.where(type_row_t <= 0, search_row, track_row)
                    search_logit = score[0, 0]
                    track_logits = score[1:, 0]
                    track_logit = torch.logsumexp(track_logits, dim=0)
                    best_track = torch.argmax(track_logits) + 1
                    return torch.where(search_logit >= track_logit, torch.zeros_like(best_track), best_track)
                if self.use_ar_seq_policy and hasattr(self.g, "ar_step_scores"):
                    seq_slots = build_seq_slots(prev_plan, max_steps_local)
                    if seq_slots is None:
                        seq_slots = slot[:, None, :].expand(-1, int(getattr(self.g, "seq_len", self.max_steps)), -1)
                    steps = int(getattr(self.g, "seq_len", self.max_steps))
                    if (
                        self.cuda_graph_ar_dynamic
                        and self.device.type == "cuda"
                        and single_sensor
                        and self.root_seq_factorized_decode
                        and self.g_alt is None
                        and self.service_track_weight == 0.0
                        and self.service_search_weight == 0.0
                        and self.direct_action_service_weight == 0.0
                        and self.direct_action_value_weight == 0.0
                        and self.direct_action_frame_weight == 0.0
                        and duration_penalty is None
                        and not self.sensor_valid_mask
                        and self.max_window_search_frac <= 0.0
                        and self.max_window_search_atoms <= 0
                        and self.min_window_search_atoms <= 0
                    ):
                        dwell_np = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
                        dwell_full = np.ones((MAXT + 1,), dtype=np.float32) * 10.0
                        n_dwell = min(MAXT, int(dwell_np.size))
                        if n_dwell > 0:
                            dwell_full[1 : n_dwell + 1] = np.maximum(1.0, dwell_np[:n_dwell])
                        dwell_t = torch.from_numpy(dwell_full).to(self.device)
                        row_map_t = torch.arange(MAXT + 1, dtype=torch.long, device=self.device)
                        graph_steps = min(int(steps), int(max_steps_local))
                        graph_shape = (int(x.shape[1]), int(dwell_t.shape[0]))
                        graph = self._graph_ar_dynamic_cache.get(graph_shape)
                        if graph is None:
                            graph = CudaGraphARDynamicSOnlyDecode(
                                self.model,
                                self.g,
                                x,
                                slot[0],
                                dwell_t,
                                row_map_t,
                                max_steps=graph_steps,
                                search_score_bias=float(search_bias),
                            )
                            self._graph_ar_dynamic_cache[graph_shape] = graph
                            self._graph_ar_dynamic = graph
                            self._graph_ar_dynamic_shape = graph_shape
                        rows_t = graph(x, slot[0], dwell_t, row_map_t)
                        rows = rows_t.detach().cpu().tolist()
                        out_graph = []
                        for row in rows:
                            row = int(row)
                            if row < 0:
                                break
                            out_graph.append(int(self._s_actions[int(max(0, min(MAXT, row)))]))
                        return out_graph
                    if (
                        self.cuda_graph_ar_seq
                        and self.device.type == "cuda"
                        and single_sensor
                        and self.root_seq_factorized_decode
                        and self.g_alt is None
                        and self.service_track_weight == 0.0
                        and self.service_search_weight == 0.0
                        and self.direct_action_service_weight == 0.0
                        and self.direct_action_value_weight == 0.0
                        and self.direct_action_frame_weight == 0.0
                        and duration_penalty is None
                        and not self.sensor_valid_mask
                        and self.max_window_search_frac <= 0.0
                        and self.max_window_search_atoms <= 0
                        and self.min_window_search_atoms <= 0
                    ):
                        graph_seq_slots = seq_slots.contiguous()
                        graph_steps = min(int(steps), int(max_steps_local))
                        if self._graph_ar_seq is None:
                            self._graph_ar_seq = CudaGraphARSOnlyDecode(
                                self.model,
                                self.g,
                                x,
                                graph_seq_slots,
                                max_steps=graph_steps,
                                search_score_bias=float(search_bias),
                            )
                        rows_t = self._graph_ar_seq(x, graph_seq_slots)
                        rows = rows_t.detach().cpu().tolist()
                        return [int(self._s_actions[int(max(0, min(MAXT, int(row))))]) for row in rows]
                    selected: set[int] = set()
                    # `selected_t` is part of the encoded observation.  The AR
                    # planner needs a fresh within-window exclusion mask.
                    selected_ar = torch.zeros_like(selected_t)
                    spent_ar = 0.0
                    last_ar = -1
                    search_count = 0
                    s_search_count = 0
                    x_search_count = 0
                    track_count = 0
                    h, ar_tok = self.g._project_state(cls, tok)
                    if self.g_alt is not None and hasattr(self.g_alt, "ar_step_scores") and self.g_blend_alpha < 1.0:
                        h_alt, ar_tok_alt = self.g_alt._project_state(cls, tok)
                    else:
                        h_alt = None
                        ar_tok_alt = None
                    prev = torch.zeros((1, 2), dtype=torch.long, device=self.device)
                    prev[:, 1] = 1
                    history = None
                    if int(getattr(self.g, "ar_history_k", 0)) > 0:
                        history = prev[:, None, :].expand(-1, int(self.g.ar_history_k), -1).clone()
                    out = []
                    for step in range(min(steps, max_steps_local)):
                        if self.ar_dynamic_slots:
                            slot_np_step = slot_features(obs, spent_ar, search_count, track_count, last_ar, 200.0).astype(np.float32)
                            slot_step = torch.from_numpy(slot_np_step[None]).float().to(self.device)
                        else:
                            slot_step = seq_slots[:, step, :]
                        pos_step = self.g.seq_pos[step][None, :].expand(1, -1)
                        a0 = self.g.action_emb(prev[:, 0])
                        a1 = self.g.action_emb(prev[:, 1])
                        slot_e = self.g.slot_proj(slot_step)
                        inp = self.g.ar_input(torch.cat([a0, a1, slot_e, pos_step], dim=-1))
                        if history is not None:
                            inp = inp + self.g._ar_history_embedding(history)
                        h = self.g.ar_cell(inp, h)
                        ar_type_logits = None
                        ar_target_logits = None
                        if (
                            single_sensor
                            and self.root_seq_factorized_decode
                            and h_alt is None
                            and hasattr(self.g, "ar_step_factor_logits")
                        ):
                            ar_type_logits, ar_target_logits = self.g.ar_step_factor_logits(
                                h,
                                ar_tok,
                                slot_step,
                                pos_step,
                                selected_ar,
                                token_active,
                            )
                            score = slot_step.new_full((ar_target_logits.shape[1], 2), -1e9)
                            score[0, :] = ar_type_logits[0, :, 0]
                            score[1:, :] = (ar_type_logits[0, None, :, 1] + ar_target_logits[0])[1:, :]
                        else:
                            score = self.g.ar_step_scores(h, ar_tok, slot_step, pos_step, selected_ar, token_active)[0]
                        if h_alt is not None and ar_tok_alt is not None:
                            a0_alt = self.g_alt.action_emb(prev[:, 0])
                            a1_alt = self.g_alt.action_emb(prev[:, 1])
                            slot_e_alt = self.g_alt.slot_proj(slot_step)
                            pos_step_alt = self.g_alt.seq_pos[step][None, :].expand(1, -1)
                            inp_alt = self.g_alt.ar_input(torch.cat([a0_alt, a1_alt, slot_e_alt, pos_step_alt], dim=-1))
                            h_alt = self.g_alt.ar_cell(inp_alt, h_alt)
                            score_alt = self.g_alt.ar_step_scores(h_alt, ar_tok_alt, slot_step, pos_step_alt, selected_ar, token_active)[0]
                            score = self.g_blend_alpha * score + (1.0 - self.g_blend_alpha) * score_alt
                        if self.service_track_weight != 0.0 or self.service_search_weight != 0.0:
                            score = score + torch.from_numpy(self._service_bonus_np(obs)).to(self.device)
                        if duration_penalty is not None:
                            score = score - duration_penalty
                        score[0, :] += search_bias
                        score = self._apply_sensor_search_balance(score, s_search_count, x_search_count)
                        score = self._apply_search_cap(score, search_count, track_count)
                        if self.sensor_valid_mask:
                            valid = torch.from_numpy(self._sensor_valid_np(obs)).to(self.device)
                            score = score.masked_fill(~valid, -1e9)
                        for row in selected:
                            if 0 < int(row) < score.shape[0]:
                                score[int(row), :] = -1e9
                        if (
                            single_sensor
                            and self.root_seq_factorized_decode
                            and hasattr(self.g, "ar_step_factor_logits")
                        ):
                            if ar_type_logits is None or ar_target_logits is None:
                                ar_type_logits, ar_target_logits = self.g.ar_step_factor_logits(
                                    h,
                                    ar_tok,
                                    slot_step,
                                    pos_step,
                                    selected_ar,
                                    token_active,
                                )
                            search_logit = ar_type_logits[0, 0, 0] + float(search_bias)
                            track_logit = ar_type_logits[0, 0, 1]
                            target_col = ar_target_logits[0, :, 0].clone()
                            for row in selected:
                                if 0 < int(row) < target_col.shape[0]:
                                    target_col[int(row)] = -1e9
                            target_col[0] = -1e9
                            best_track = int(torch.argmax(target_col).detach().cpu())
                            if (not torch.isfinite(target_col[best_track])) or float(target_col[best_track].detach().cpu()) <= -1e8:
                                s_row = 0
                            else:
                                s_row = 0 if float(search_logit.detach().cpu()) >= float(track_logit.detach().cpu()) else best_track
                            x_row = -1
                            if self.root_seq_debug_candidates:
                                self.debug_candidate_rows.append(
                                    {
                                        "decode_step": int(step),
                                        "chosen_row": int(s_row),
                                        "type_search_logit": float(search_logit.detach().cpu()),
                                        "type_track_logit": float(track_logit.detach().cpu()),
                                        "best_track_row": int(best_track),
                                        "best_track_logit": float(target_col[best_track].detach().cpu()),
                                        "selected_count": int(selected_ar[0, 1:].sum().detach().cpu()),
                                        "search_count": int(search_count),
                                        "track_count": int(track_count),
                                        "ar_factorized": True,
                                    }
                                )
                        elif single_sensor:
                            s_row = int(torch.argmax(score[:, 0]).detach().cpu())
                            x_row = -1
                        else:
                            k = min(self.root_seq_decode_topk, int(score.shape[0]))
                            s_vals, s_idx = torch.topk(score[:, 0], k=k)
                            x_vals, x_idx = torch.topk(score[:, 1], k=k)
                            joint = s_vals[:, None] + x_vals[None, :]
                            conflict = (s_idx[:, None] > 0) & (s_idx[:, None] == x_idx[None, :])
                            joint = joint.masked_fill(conflict, -1e9)
                            joint = self._apply_search_floor_joint(joint, s_idx, x_idx, search_count)
                            if self.lookahead_width > 0 and self.service_critic_weight != 0.0 and hasattr(self.g, "predict_service"):
                                rk = min(max(1, int(self.lookahead_width)), k)
                                cand_joint = joint[:rk, :rk].flatten()
                                cand_s = s_idx[:rk].repeat_interleave(rk)
                                cand_x = x_idx[:rk].repeat(rk)
                                finite = torch.isfinite(cand_joint) & (cand_joint > -1e8)
                                if bool(finite.any()):
                                    cand_joint = cand_joint[finite]
                                    cand_s = cand_s[finite]
                                    cand_x = cand_x[finite]
                                    action_pair = torch.stack(
                                        [cand_s.clamp_min(0) * 2, cand_x.clamp_min(0) * 2 + 1],
                                        dim=1,
                                    ).long()
                                    h_b = h.expand(action_pair.shape[0], -1)
                                    tok_b = ar_tok.expand(action_pair.shape[0], -1, -1)
                                    slot_b = slot_step.expand(action_pair.shape[0], -1)
                                    cls_n, tok_n, r_p, _dt_p = self.g(h_b, tok_b, slot_b, action_pair)
                                    total = cand_joint + r_p + self._service_critic_value(cls_n, tok_n)
                                    best = int(torch.argmax(total).detach().cpu())
                                    s_row = int(cand_s[best].detach().cpu())
                                    x_row = int(cand_x[best].detach().cpu())
                                else:
                                    flat = int(torch.argmax(joint).detach().cpu())
                                    s_row = int(s_idx[flat // k].detach().cpu())
                                    x_row = int(x_idx[flat % k].detach().cpu())
                            else:
                                flat = int(torch.argmax(joint).detach().cpu())
                                s_row = int(s_idx[flat // k].detach().cpu())
                                x_row = int(x_idx[flat % k].detach().cpu())
                        if self.root_seq_edf_targets:
                            if s_row > 0:
                                s_row = self._edf_target_row(obs, 0, selected)
                            if x_row > 0:
                                x_row = self._edf_target_row(obs, 1, selected | ({s_row} if s_row > 0 else set()))
                        elif self.root_seq_base_targets:
                            if s_row > 0:
                                s_row = self._base_target_row(base_rankings, 0, selected)
                            if x_row > 0:
                                x_row = self._base_target_row(base_rankings, 1, selected | ({s_row} if s_row > 0 else set()))
                        s_row, x_row = self._override_targets(s_row, x_row, selected, service_pressure)
                        if single_sensor:
                            x_row = -1
                            action_out = int(self._s_actions[int(max(0, s_row))])
                            out.append(action_out)
                        else:
                            action_out = encode_joint_action(action_from_row(s_row, 0), action_from_row(x_row, 1))
                            out.append(action_out)
                        spent_ar = min(200.0, spent_ar + float(joint_duration(obs, action_out)))
                        if s_row > 0:
                            selected.add(s_row)
                            selected_ar[0, s_row] = True
                            track_count += 1
                            last_ar = int(s_row)
                        else:
                            search_count += 1
                            s_search_count += 1
                            last_ar = 0
                        if (not single_sensor) and x_row > 0:
                            selected.add(x_row)
                            selected_ar[0, x_row] = True
                            track_count += 1
                            last_ar = int(x_row)
                        elif not single_sensor:
                            search_count += 1
                            x_search_count += 1
                            last_ar = 0
                        prev[0, 0] = action_index_from_row(s_row, 0)
                        prev[0, 1] = 1 if single_sensor else action_index_from_row(x_row, 1)
                        if history is not None:
                            history = self.g._update_ar_history(history, prev)
                    return out
                if self.use_base_seq_policy:
                    steps = int(getattr(self.g, "seq_len", self.max_steps))
                    if seq_slots is None:
                        slot_batch = slot.expand(steps, -1)
                    else:
                        slot_batch = seq_slots[0]
                    x_batch = x.expand(steps, -1, -1)
                    p_scores, q_scores = self.model.forward_scores(x_batch, slot_batch)
                    scores = (self.policy_weight * p_scores + self.q_weight * q_scores)
                else:
                    if (
                        self.cuda_graph_root_seq
                        and self.device.type == "cuda"
                        and self.g_alt is None
                        and hasattr(self.g, "sequence_scores")
                    ):
                        graph_seq_slots = seq_slots
                        if graph_seq_slots is None:
                            steps = int(getattr(self.g, "seq_len", self.max_steps))
                            graph_seq_slots = slot[:, None, :].expand(-1, steps, -1).contiguous()
                        else:
                            graph_seq_slots = graph_seq_slots.contiguous()
                        if self._graph_root_seq is None:
                            self._graph_root_seq = CudaGraphRootSeqScores(
                                self.g,
                                cls,
                                tok,
                                slot,
                                selected_t,
                                token_active,
                                graph_seq_slots,
                                use_step_context=self.root_seq_step_context,
                            )
                        scores = self._graph_root_seq(cls, tok, slot, selected_t, token_active, graph_seq_slots)[0]
                    else:
                        scores = self.g.sequence_scores(
                            cls,
                            tok,
                            slot,
                            selected_t,
                            token_active,
                            seq_slots=seq_slots,
                            use_step_context=self.root_seq_step_context,
                        )[0]
                    if self.root_seq_factorized_decode and hasattr(self.g, "sequence_factor_logits"):
                        type_logits, target_logits = self.g.sequence_factor_logits(
                            cls,
                            tok,
                            slot,
                            selected_t,
                            token_active,
                            seq_slots=seq_slots,
                            use_step_context=self.root_seq_step_context,
                        )
                        factor_type_scores = type_logits[0]
                        factor_target_scores = target_logits[0]
                    # CUDA graph replay returns inference tensors; the decoder applies masks and
                    # biases in-place, so materialize a normal tensor before those updates.
                    scores = scores.clone()
                    stop_scores = None
                    if self.use_root_seq_stop_head and hasattr(self.g, "sequence_stop_scores"):
                        stop_scores = self.g.sequence_stop_scores(cls, tok, slot, seq_slots=seq_slots)[0]
                    if self.g_alt is not None and hasattr(self.g_alt, "sequence_scores") and self.g_blend_alpha < 1.0:
                        alt_scores = self.g_alt.sequence_scores(
                            cls,
                            tok,
                            slot,
                            selected_t,
                            token_active,
                            seq_slots=seq_slots,
                            use_step_context=self.root_seq_step_context,
                        )[0]
                        scores = self.g_blend_alpha * scores + (1.0 - self.g_blend_alpha) * alt_scores
                if self.use_base_seq_policy:
                    stop_scores = None
                if self.service_track_weight != 0.0 or self.service_search_weight != 0.0:
                    scores = scores + torch.from_numpy(self._service_bonus_np(obs)).to(self.device)[None, :, :]
                if duration_penalty is not None:
                    scores = scores - duration_penalty[None, :, :]
                scores[:, 0, :] += search_bias
                if self.sensor_valid_mask:
                    valid = torch.from_numpy(self._sensor_valid_np(obs)).to(self.device)
                    scores = scores.masked_fill(~valid[None, :, :], -1e9)
                fast_device_decode = (
                    not self.root_seq_recurrent_rescore
                    and not self.root_seq_edf_targets
                    and not self.root_seq_base_targets
                    and not self.service_target_override
                    and not self.use_root_seq_stop_head
                    and self.lookahead_width <= 0
                    and self.root_seq_search_balance_weight == 0.0
                    and self.root_seq_max_search_skew <= 0
                    and self.root_seq_stop_threshold <= -1.0e20
                )
                if fast_device_decode:
                    steps_local = min(int(scores.shape[0]), int(max_steps_local))
                    selected_mask = selected_t[0].clone()
                    k = min(self.root_seq_decode_topk, int(scores.shape[1]))
                    max_search_atoms = int(self.max_window_search_atoms)
                    min_search_atoms = int(self.min_window_search_atoms)
                    max_search_frac = float(self.max_window_search_frac)
                    cached_direct_value = None
                    cached_s_idx = None
                    cached_x_idx = None
                    if (
                        self.direct_action_service_weight == 0.0
                        and self.direct_action_value_weight != 0.0
                        and self.direct_action_value_cache_topn > 0
                        and hasattr(self.g, "action_value_head")
                    ):
                        cache_topn = min(int(self.direct_action_value_cache_topn), int(scores.shape[1]))
                        with torch.inference_mode():
                            cached_s_idx = torch.topk(scores[:steps_local, :, 0], k=cache_topn, dim=1).indices
                            cached_x_idx = torch.topk(scores[:steps_local, :, 1], k=cache_topn, dim=1).indices
                            pair_s = cached_s_idx[:, :, None].expand(-1, -1, cache_topn)
                            pair_x = cached_x_idx[:, None, :].expand(-1, cache_topn, -1)
                            action_pair = torch.stack(
                                [pair_s.clamp_min(0) * 2, pair_x.clamp_min(0) * 2 + 1],
                                dim=-1,
                            ).reshape(-1, 2).long()
                            if seq_slots is not None:
                                slot_steps = seq_slots[0, :steps_local, :]
                            else:
                                slot_steps = slot[0:1, :].expand(steps_local, -1)
                            slot_cache = slot_steps[:, None, None, :].expand(-1, cache_topn, cache_topn, -1).reshape(-1, slot_steps.shape[-1])
                            cached_direct_value = self._root_action_value_pairs(cls, tok, slot_cache, action_pair).reshape(
                                steps_local, cache_topn, cache_topn
                            )
                    if self.device.type == "cpu" or self.root_seq_cpu_decode:
                        if scores.device.type != "cpu":
                            scores_decode = scores.detach().cpu()
                            selected_mask = selected_t[0].detach().cpu().clone()
                        else:
                            scores_decode = scores
                            selected_mask = selected_t[0].clone()
                        fixed_type_rows: list[int] | None = None
                        if single_sensor and self.direct_action_value_track_only:
                            fixed_type_rows = []
                            type_mask = selected_mask.clone()
                            type_search_count = 0
                            type_track_count = 0
                            type_search_streak = 0
                            type_track_streak = 0
                            for type_step in range(steps_local):
                                type_score = scores_decode[type_step].clone()
                                if type_mask[1:].any():
                                    type_score[1:, :].masked_fill_(type_mask[1:, None], -1e9)
                                type_score = self._apply_decoder_history_bias(type_score, type_search_streak, type_track_streak)
                                remaining_search = min_search_atoms - type_search_count if min_search_atoms > 0 else 0
                                if remaining_search >= 2:
                                    type_score[1:, :] = -1e9
                                elif remaining_search == 1:
                                    type_score[0, :] = type_score[0, :] + 1e6
                                elif max_search_atoms > 0 and type_search_count >= max_search_atoms:
                                    type_score[0, :] = -1e9
                                if (
                                    remaining_search <= 0
                                    and max_search_frac > 0.0
                                    and type_search_count / float(max(1, type_search_count + type_track_count)) > max_search_frac
                                ):
                                    type_score[0, :] = -1e9
                                base_row = pick_single_sensor_row(type_score, type_step, type_mask)
                                fixed_type_rows.append(int(base_row))
                                type_search_count += int(base_row <= 0)
                                type_track_count += int(base_row > 0)
                                if int(base_row) <= 0:
                                    type_search_streak += 1
                                    type_track_streak = 0
                                else:
                                    type_track_streak += 1
                                    type_search_streak = 0
                                    type_mask[int(base_row)] = True
                        joint_actions: list[int] = []
                        search_count_i = 0
                        track_count_i = 0
                        search_streak_i = 0
                        track_streak_i = 0
                        for step in range(steps_local):
                            score = scores_decode[step]
                            if selected_mask[1:].any():
                                score[1:, :].masked_fill_(selected_mask[1:, None], -1e9)
                            score = self._apply_decoder_history_bias(score, search_streak_i, track_streak_i)
                            remaining_search = min_search_atoms - search_count_i if min_search_atoms > 0 else 0
                            force_search = remaining_search >= 2
                            if force_search:
                                score[1:, :] = -1e9
                            elif remaining_search == 1:
                                score[0, :] = score[0, :] + 1e6
                            elif max_search_atoms > 0 and search_count_i >= max_search_atoms:
                                score[0, :] = -1e9
                            if (
                                remaining_search <= 0
                                and max_search_frac > 0.0
                                and search_count_i / float(max(1, search_count_i + track_count_i)) > max_search_frac
                            ):
                                score[0, :] = -1e9
                            if single_sensor:
                                reranked = None
                                use_direct_value = (
                                    (self.direct_action_service_weight != 0.0 or self.direct_action_value_weight != 0.0 or self.direct_action_frame_weight != 0.0)
                                    and (self.direct_action_value_max_steps <= 0 or step < self.direct_action_value_max_steps)
                                )
                                if use_direct_value:
                                    if self.direct_action_value_track_only:
                                        base_row = int(fixed_type_rows[step]) if fixed_type_rows is not None else pick_single_sensor_row(score, step, selected_mask)
                                        if base_row > 0:
                                            track_score = score[:, 0].clone()
                                            track_score[0] = -1e9
                                            s_vals, s_idx = torch.topk(track_score, k=k)
                                            reranked = self._rerank_action_pairs(
                                                cls,
                                                tok,
                                                seq_slots[:, step, :] if seq_slots is not None else slot,
                                                s_vals,
                                                s_idx,
                                                torch.full_like(s_idx, -1),
                                            )
                                        else:
                                            reranked = (base_row, -1)
                                    else:
                                        s_vals, s_idx = torch.topk(score[:, 0], k=k)
                                        reranked = self._rerank_action_pairs(
                                            cls,
                                            tok,
                                            seq_slots[:, step, :] if seq_slots is not None else slot,
                                            s_vals,
                                            s_idx,
                                            torch.full_like(s_idx, -1),
                                        )
                                s_row = int(reranked[0]) if reranked is not None else pick_single_sensor_row(score, step, selected_mask)
                                if self.root_seq_debug_candidates and step < 24:
                                    s_col = score[:, 0]
                                    search_score = float(s_col[0].detach().cpu())
                                    finite = torch.isfinite(s_col) & (s_col > -1e8)
                                    best_row_dbg = int(torch.argmax(s_col).detach().cpu())
                                    best_score_dbg = float(s_col[best_row_dbg].detach().cpu())
                                    search_rank_dbg = int(((s_col[finite] > s_col[0]).sum() + 1).detach().cpu()) if bool(finite.any()) else -1
                                    type_search_dbg = float("nan")
                                    type_track_dbg = float("nan")
                                    if factor_type_scores is not None:
                                        type_search_dbg = float(factor_type_scores[step, 0, 0].detach().cpu())
                                        type_track_dbg = float(factor_type_scores[step, 0, 1].detach().cpu())
                                    self.debug_candidate_rows.append(
                                        {
                                            "decode_step": int(step),
                                            "chosen_row": int(s_row),
                                            "search_rank": int(search_rank_dbg),
                                            "search_score": float(search_score),
                                            "best_row": int(best_row_dbg),
                                            "best_score": float(best_score_dbg),
                                            "type_search_logit": float(type_search_dbg),
                                            "type_track_logit": float(type_track_dbg),
                                            "selected_count": int(selected_mask[1:].sum().detach().cpu()),
                                            "search_count": int(search_count_i),
                                            "track_count": int(track_count_i),
                                        }
                                    )
                                search_count_i += int(s_row <= 0)
                                track_count_i += int(s_row > 0)
                                if int(s_row) <= 0:
                                    search_streak_i += 1
                                    track_streak_i = 0
                                else:
                                    track_streak_i += 1
                                    search_streak_i = 0
                                selected_mask[s_row] = True
                                joint_actions.append(int(self._s_actions[int(max(0, s_row))]))
                                continue
                            s_vals, s_idx = torch.topk(score[:, 0], k=k)
                            x_vals, x_idx = torch.topk(score[:, 1], k=k)
                            joint = s_vals[:, None] + x_vals[None, :]
                            conflict = (s_idx[:, None] > 0) & (s_idx[:, None] == x_idx[None, :])
                            masked_joint = joint.masked_fill(conflict, -1e9)
                            masked_joint = self._apply_search_floor_joint(masked_joint, s_idx, x_idx, search_count_i)
                            reranked = None
                            use_direct_value = (
                                (self.direct_action_service_weight != 0.0 or self.direct_action_value_weight != 0.0 or self.direct_action_frame_weight != 0.0)
                                and (self.direct_action_value_max_steps <= 0 or step < self.direct_action_value_max_steps)
                            )
                            if use_direct_value:
                                cand_joint_flat = masked_joint.flatten()
                                cand_s_flat = s_idx.repeat_interleave(k)
                                cand_x_flat = x_idx.repeat(k)
                                if cached_direct_value is not None and cached_s_idx is not None and cached_x_idx is not None:
                                    if self.direct_action_value_cache_only:
                                        finite = torch.isfinite(cand_joint_flat) & (cand_joint_flat > -1e8)
                                        ps = cand_s_flat[:, None] == cached_s_idx[step][None, :]
                                        px = cand_x_flat[:, None] == cached_x_idx[step][None, :]
                                        has = (ps.sum(dim=1) > 0) & (px.sum(dim=1) > 0)
                                        si = ps.to(torch.long).argmax(dim=1)
                                        xi = px.to(torch.long).argmax(dim=1)
                                        value = cached_direct_value[step, si, xi]
                                        total = (cand_joint_flat + value).masked_fill(~finite | ~has, -1e9)
                                        best = int(torch.argmax(total).detach().cpu())
                                        reranked = (
                                            int(cand_s_flat[best].detach().cpu()),
                                            int(cand_x_flat[best].detach().cpu()),
                                        )
                                    else:
                                        reranked = self._rerank_cached_action_pairs(
                                            cand_joint_flat,
                                            cand_s_flat,
                                            cand_x_flat,
                                            cached_s_idx[step],
                                            cached_x_idx[step],
                                            cached_direct_value[step],
                                        )
                                if reranked is None:
                                    if seq_slots is not None:
                                        slot_step = seq_slots[:, step, :]
                                    else:
                                        slot_step = slot
                                    reranked = self._rerank_action_pairs(
                                        cls,
                                        tok,
                                        slot_step,
                                        cand_joint_flat,
                                        cand_s_flat,
                                        cand_x_flat,
                                    )
                            if reranked is None:
                                flat_i = int(torch.argmax(masked_joint))
                                s_row = int(s_idx[flat_i // k])
                                x_row = int(x_idx[flat_i % k])
                            else:
                                s_row, x_row = reranked
                            selected_mask[s_row] = True
                            if single_sensor:
                                search_atoms_i = int(s_row <= 0)
                                track_atoms_i = int(s_row > 0)
                                search_count_i += search_atoms_i
                                track_count_i += track_atoms_i
                                joint_actions.append(int(self._s_actions[int(max(0, s_row))]))
                            else:
                                selected_mask[x_row] = True
                                search_atoms_i = int(s_row <= 0) + int(x_row <= 0)
                                track_atoms_i = int(s_row > 0) + int(x_row > 0)
                                search_count_i += search_atoms_i
                                track_count_i += track_atoms_i
                                joint_actions.append(1_000_000 + int(self._s_actions[s_row]) * 1_000 + int(self._x_actions[x_row]))
                            if search_atoms_i > 0 and track_atoms_i <= 0:
                                search_streak_i += 1
                                track_streak_i = 0
                            elif track_atoms_i > 0 and search_atoms_i <= 0:
                                track_streak_i += 1
                                search_streak_i = 0
                            else:
                                search_streak_i = 0
                                track_streak_i = 0
                        return joint_actions
                    joint_actions_t = torch.empty((steps_local,), dtype=torch.long, device=self.device)
                    s_actions_t = self._s_actions_t
                    x_actions_t = self._x_actions_t
                    fixed_type_rows_t = None
                    if single_sensor and self.direct_action_value_track_only:
                        rows = []
                        type_mask = selected_t[0].clone()
                        type_search_count = torch.zeros((), dtype=torch.long, device=self.device)
                        type_track_count = torch.zeros((), dtype=torch.long, device=self.device)
                        type_search_streak = torch.zeros((), dtype=torch.long, device=self.device)
                        type_track_streak = torch.zeros((), dtype=torch.long, device=self.device)
                        for type_step in range(steps_local):
                            type_score = scores[type_step].clone()
                            if type_mask[1:].any():
                                type_score[1:, :].masked_fill_(type_mask[1:, None], -1e9)
                            type_score = self._apply_decoder_history_bias(type_score, type_search_streak, type_track_streak)
                            remaining_search_t = torch.zeros((), dtype=torch.long, device=self.device)
                            force_search_t = torch.zeros((), dtype=torch.bool, device=self.device)
                            if min_search_atoms > 0:
                                remaining_search_t = min_search_atoms - type_search_count
                                force_search_t = remaining_search_t >= 2
                                type_score[1:, :] = torch.where(
                                    force_search_t,
                                    type_score.new_full(type_score[1:, :].shape, -1e9),
                                    type_score[1:, :],
                                )
                                one_search_t = remaining_search_t == 1
                                type_score[0, :] = torch.where(one_search_t, type_score[0, :] + 1e6, type_score[0, :])
                            if max_search_atoms > 0:
                                cap_atoms_t = (type_search_count >= max_search_atoms) & (~force_search_t)
                                type_score[0, :] = torch.where(cap_atoms_t, type_score.new_full((2,), -1e9), type_score[0, :])
                            if max_search_frac > 0.0:
                                total_atoms_t = torch.clamp(type_search_count + type_track_count, min=1)
                                cap = (type_search_count.to(type_score.dtype) / total_atoms_t.to(type_score.dtype) > max_search_frac) & (remaining_search_t <= 0)
                                type_score[0, :] = torch.where(cap, type_score.new_full((2,), -1e9), type_score[0, :])
                            base_row_t = pick_single_sensor_row_t(type_score, type_step, type_mask)
                            rows.append(base_row_t)
                            search_atom_t = (base_row_t <= 0).long()
                            track_atom_t = (base_row_t > 0).long()
                            type_search_count = type_search_count + search_atom_t
                            type_track_count = type_track_count + track_atom_t
                            type_search_streak = torch.where(search_atom_t > 0, type_search_streak + 1, torch.zeros_like(type_search_streak))
                            type_track_streak = torch.where(track_atom_t > 0, type_track_streak + 1, torch.zeros_like(type_track_streak))
                            type_mask.scatter_(0, base_row_t.clamp_min(0).reshape(1), True)
                        fixed_type_rows_t = torch.stack(rows)
                    search_count_t = torch.zeros((), dtype=torch.long, device=self.device)
                    track_count_t = torch.zeros((), dtype=torch.long, device=self.device)
                    search_streak_t = torch.zeros((), dtype=torch.long, device=self.device)
                    track_streak_t = torch.zeros((), dtype=torch.long, device=self.device)
                    for step in range(steps_local):
                        score = scores[step]
                        if selected_mask[1:].any():
                            score[1:, :].masked_fill_(selected_mask[1:, None], -1e9)
                        score = self._apply_decoder_history_bias(score, search_streak_t, track_streak_t)
                        force_search_t = torch.zeros((), dtype=torch.bool, device=self.device)
                        remaining_search_t = torch.zeros((), dtype=torch.long, device=self.device)
                        if min_search_atoms > 0:
                            remaining_search_t = min_search_atoms - search_count_t
                            force_search_t = remaining_search_t >= 2
                            score[1:, :] = torch.where(
                                force_search_t,
                                score.new_full(score[1:, :].shape, -1e9),
                                score[1:, :],
                            )
                            one_search_t = remaining_search_t == 1
                            score[0, :] = torch.where(one_search_t, score[0, :] + 1e6, score[0, :])
                        if max_search_atoms > 0:
                            cap_atoms_t = (search_count_t >= max_search_atoms) & (~force_search_t)
                            score[0, :] = torch.where(cap_atoms_t, score.new_full((2,), -1e9), score[0, :])
                        if max_search_frac > 0.0:
                            total_atoms_t = torch.clamp(search_count_t + track_count_t, min=1)
                            cap = (search_count_t.to(score.dtype) / total_atoms_t.to(score.dtype) > max_search_frac) & (remaining_search_t <= 0)
                            score[0, :] = torch.where(cap, score.new_full((2,), -1e9), score[0, :])
                        if single_sensor:
                            reranked = None
                            use_direct_value = (
                                (self.direct_action_service_weight != 0.0 or self.direct_action_value_weight != 0.0 or self.direct_action_frame_weight != 0.0)
                                and (self.direct_action_value_max_steps <= 0 or int(step) < self.direct_action_value_max_steps)
                            )
                            if use_direct_value:
                                if self.direct_action_value_track_only:
                                    base_row_t = fixed_type_rows_t[step] if fixed_type_rows_t is not None else pick_single_sensor_row_t(score, step, selected_mask)
                                    if bool((base_row_t <= 0).detach().cpu()):
                                        reranked = (int(base_row_t.detach().cpu()), -1)
                                    else:
                                        track_score = score[:, 0].clone()
                                        track_score[0] = -1e9
                                        s_vals, s_idx = torch.topk(track_score, k=k)
                                        reranked = self._rerank_action_pairs(
                                            cls,
                                            tok,
                                            seq_slots[:, step, :] if seq_slots is not None else slot,
                                            s_vals,
                                            s_idx,
                                            torch.full_like(s_idx, -1),
                                        )
                                else:
                                    s_vals, s_idx = torch.topk(score[:, 0], k=k)
                                    reranked = self._rerank_action_pairs(
                                        cls,
                                        tok,
                                        seq_slots[:, step, :] if seq_slots is not None else slot,
                                        s_vals,
                                        s_idx,
                                        torch.full_like(s_idx, -1),
                                    )
                            s_row_t = (
                                torch.tensor(reranked[0], dtype=torch.long, device=self.device)
                                if reranked is not None
                                else pick_single_sensor_row_t(score, step, selected_mask)
                            )
                            selected_mask.scatter_(0, s_row_t.clamp_min(0).reshape(1), True)
                            search_atom_t = (s_row_t <= 0).long()
                            track_atom_t = (s_row_t > 0).long()
                            search_count_t = search_count_t + search_atom_t
                            track_count_t = track_count_t + track_atom_t
                            search_streak_t = torch.where(search_atom_t > 0, search_streak_t + 1, torch.zeros_like(search_streak_t))
                            track_streak_t = torch.where(track_atom_t > 0, track_streak_t + 1, torch.zeros_like(track_streak_t))
                            safe_s_row_t = s_row_t.clamp(min=0, max=int(s_actions_t.shape[0]) - 1)
                            joint_actions_t[step] = s_actions_t[safe_s_row_t]
                            continue
                        s_vals, s_idx = torch.topk(score[:, 0], k=k)
                        x_vals, x_idx = torch.topk(score[:, 1], k=k)
                        joint = s_vals[:, None] + x_vals[None, :]
                        conflict = (s_idx[:, None] > 0) & (s_idx[:, None] == x_idx[None, :])
                        masked_joint = joint.masked_fill(conflict, -1e9)
                        masked_joint = self._apply_search_floor_joint(masked_joint, s_idx, x_idx, search_count_t)
                        reranked = None
                        use_direct_value = (
                            (self.direct_action_service_weight != 0.0 or self.direct_action_value_weight != 0.0 or self.direct_action_frame_weight != 0.0)
                            and (self.direct_action_value_max_steps <= 0 or int(step) < self.direct_action_value_max_steps)
                        )
                        if use_direct_value:
                            if cached_direct_value is not None and cached_s_idx is not None and cached_x_idx is not None:
                                cand_joint_flat = masked_joint.flatten()
                                cand_s_flat = s_idx.repeat_interleave(k)
                                cand_x_flat = x_idx.repeat(k)
                                finite = torch.isfinite(cand_joint_flat) & (cand_joint_flat > -1e8)
                                if self.direct_action_value_cache_only:
                                    ps = cand_s_flat[:, None] == cached_s_idx[step][None, :]
                                    px = cand_x_flat[:, None] == cached_x_idx[step][None, :]
                                    has = (ps.sum(dim=1) > 0) & (px.sum(dim=1) > 0)
                                    si = ps.to(torch.long).argmax(dim=1)
                                    xi = px.to(torch.long).argmax(dim=1)
                                    value = cached_direct_value[step, si, xi]
                                    total = (cand_joint_flat + value).masked_fill(~finite | ~has, -1e9)
                                    best = torch.argmax(total)
                                    s_row_t = cand_s_flat[best]
                                    x_row_t = cand_x_flat[best]
                                    reranked = (s_row_t, x_row_t)
                                elif bool(finite.any()):
                                    cand_joint_f = cand_joint_flat[finite]
                                    cand_s_f = cand_s_flat[finite]
                                    cand_x_f = cand_x_flat[finite]
                                    ps = cand_s_f[:, None] == cached_s_idx[step][None, :]
                                    px = cand_x_f[:, None] == cached_x_idx[step][None, :]
                                    has = (ps.sum(dim=1) > 0) & (px.sum(dim=1) > 0)
                                    if bool(has.all()):
                                        si = ps.to(torch.long).argmax(dim=1)
                                        xi = px.to(torch.long).argmax(dim=1)
                                        total = cand_joint_f + cached_direct_value[step, si, xi]
                                        best = torch.argmax(total)
                                        s_row_t = cand_s_f[best]
                                        x_row_t = cand_x_f[best]
                                        reranked = (s_row_t, x_row_t)
                            if (
                                reranked is None
                                and self.direct_action_service_weight == 0.0
                                and self.direct_action_frame_weight == 0.0
                                and self.direct_action_value_weight != 0.0
                                and hasattr(self.g, "action_value_head")
                            ):
                                cand_joint_flat = masked_joint.flatten()
                                cand_s_flat = s_idx.repeat_interleave(k)
                                cand_x_flat = x_idx.repeat(k)
                                finite = torch.isfinite(cand_joint_flat) & (cand_joint_flat > -1e8)
                                action_pair = torch.stack(
                                    [cand_s_flat.clamp_min(0) * 2, cand_x_flat.clamp_min(0) * 2 + 1],
                                    dim=1,
                                ).long()
                                if seq_slots is not None:
                                    slot_step = seq_slots[:, step, :]
                                else:
                                    slot_step = slot
                                slot_pair = slot_step.expand(action_pair.shape[0], -1)
                                value = self._root_action_value_pairs(cls, tok, slot_pair, action_pair)
                                total = (cand_joint_flat + value).masked_fill(~finite, -1e9)
                                best = torch.argmax(total)
                                s_row_t = cand_s_flat[best]
                                x_row_t = cand_x_flat[best]
                                reranked = (s_row_t, x_row_t)
                            if reranked is None:
                                if seq_slots is not None:
                                    slot_step = seq_slots[:, step, :]
                                else:
                                    slot_step = slot
                                reranked = self._rerank_action_pairs(
                                    cls,
                                    tok,
                                    slot_step,
                                    masked_joint.flatten(),
                                    s_idx.repeat_interleave(k),
                                    x_idx.repeat(k),
                                )
                        if reranked is None:
                            flat = torch.argmax(masked_joint)
                            s_row_t = s_idx[flat // k]
                            x_row_t = x_idx[flat % k]
                        else:
                            s_row_t = torch.as_tensor(reranked[0], dtype=torch.long, device=self.device)
                            x_row_t = torch.as_tensor(reranked[1], dtype=torch.long, device=self.device)
                        selected_mask.scatter_(0, s_row_t.clamp_min(0).reshape(1), True)
                        if single_sensor:
                            search_atoms_t = (s_row_t <= 0).long()
                            track_atoms_t = (s_row_t > 0).long()
                            search_count_t = search_count_t + search_atoms_t
                            track_count_t = track_count_t + track_atoms_t
                            safe_s_row_t = s_row_t.clamp(min=0, max=int(s_actions_t.shape[0]) - 1)
                            joint_actions_t[step] = s_actions_t[safe_s_row_t]
                        else:
                            selected_mask.scatter_(0, x_row_t.clamp_min(0).reshape(1), True)
                            search_atoms_t = (s_row_t <= 0).long() + (x_row_t <= 0).long()
                            track_atoms_t = (s_row_t > 0).long() + (x_row_t > 0).long()
                            search_count_t = search_count_t + search_atoms_t
                            track_count_t = track_count_t + track_atoms_t
                            joint_actions_t[step] = 1_000_000 + s_actions_t[s_row_t] * 1_000 + x_actions_t[x_row_t]
                        pure_search_t = (search_atoms_t > 0) & (track_atoms_t <= 0)
                        pure_track_t = (track_atoms_t > 0) & (search_atoms_t <= 0)
                        search_streak_t = torch.where(pure_search_t, search_streak_t + 1, torch.zeros_like(search_streak_t))
                        track_streak_t = torch.where(pure_track_t, track_streak_t + 1, torch.zeros_like(track_streak_t))
                    return joint_actions_t.detach().cpu().tolist()
                selected: set[int] = set()
                search_count = 0
                s_search_count = 0
                x_search_count = 0
                track_count = 0
                search_streak = 0
                track_streak = 0
                out = []
                cur_cls = cls
                cur_tok = tok
                selected_rec = selected_t.clone()
                for step in range(min(int(scores.shape[0]), max_steps_local)):
                    if stop_scores is not None and step >= self.root_seq_min_steps:
                        stop_p = torch.sigmoid(stop_scores[step])
                        if float(stop_p.detach().cpu()) >= self.root_seq_stop_prob:
                            break
                    score = scores[step].clone()
                    if self.root_seq_recurrent_rescore and (step % self.root_seq_rescore_stride == 0):
                        if seq_slots is not None:
                            slot_step_rescore = seq_slots[:, step, :]
                        else:
                            slot_step_rescore = slot
                        if self.root_seq_recurrent_g_only and hasattr(self.g, "policy_scores"):
                            rec_utility = self.g.policy_scores(cur_cls, cur_tok, slot_step_rescore, selected_rec, token_active)[0]
                        else:
                            rec_scores, rec_q = self._score_latent(cur_cls, cur_tok, slot_step_rescore, selected_rec, token_active)
                            rec_utility = self.policy_weight * rec_scores[0] + self.q_weight * rec_q[0]
                        score = score + self.root_seq_rescore_weight * rec_utility
                    score = self._apply_decoder_history_bias(score, search_streak, track_streak)
                    for row in selected:
                        if 0 < int(row) < score.shape[0]:
                            score[int(row), :] = -1e9
                    score = self._apply_sensor_search_balance(score, s_search_count, x_search_count)
                    score = self._apply_search_cap(score, search_count, track_count)
                    k = min(self.root_seq_decode_topk, int(score.shape[0]))
                    s_vals, s_idx = torch.topk(score[:, 0], k=k)
                    x_vals, x_idx = torch.topk(score[:, 1], k=k)
                    joint = s_vals[:, None] + x_vals[None, :]
                    conflict = (s_idx[:, None] > 0) & (s_idx[:, None] == x_idx[None, :])
                    joint = joint.masked_fill(conflict, -1e9)
                    joint = self._apply_search_floor_joint(joint, s_idx, x_idx, search_count)
                    if self.lookahead_width > 0 and self.service_critic_weight != 0.0 and hasattr(self.g, "predict_service"):
                        rk = min(max(1, int(self.lookahead_width)), k)
                        cand_joint = joint[:rk, :rk].flatten()
                        cand_s = s_idx[:rk].repeat_interleave(rk)
                        cand_x = x_idx[:rk].repeat(rk)
                        finite = torch.isfinite(cand_joint) & (cand_joint > -1e8)
                        if bool(finite.any()):
                            cand_joint = cand_joint[finite]
                            cand_s = cand_s[finite]
                            cand_x = cand_x[finite]
                            action_pair = torch.stack([cand_s.clamp_min(0) * 2, cand_x.clamp_min(0) * 2 + 1], dim=1).long()
                            if seq_slots is not None:
                                slot_step = seq_slots[:, step, :]
                            else:
                                slot_step = slot
                            cls_b = cur_cls.expand(action_pair.shape[0], -1)
                            tok_b = cur_tok.expand(action_pair.shape[0], -1, -1)
                            slot_b = slot_step.expand(action_pair.shape[0], -1)
                            cls_n, tok_n, r_p, _dt_p = self.g(cls_b, tok_b, slot_b, action_pair)
                            total = cand_joint + r_p + self._service_critic_value(cls_n, tok_n)
                            best = int(torch.argmax(total).detach().cpu())
                            s_row = int(cand_s[best].detach().cpu())
                            x_row = int(cand_x[best].detach().cpu())
                            cur_cls = cls_n[best : best + 1]
                            cur_tok = tok_n[best : best + 1]
                        else:
                            flat = int(torch.argmax(joint).detach().cpu())
                            s_row = int(s_idx[flat // k].detach().cpu())
                            x_row = int(x_idx[flat % k].detach().cpu())
                    else:
                        best_joint = torch.max(joint)
                        if float(best_joint.detach().cpu()) < self.root_seq_stop_threshold:
                            break
                        reranked = None
                        use_direct_value = (
                            (self.direct_action_service_weight != 0.0 or self.direct_action_value_weight != 0.0 or self.direct_action_frame_weight != 0.0)
                            and (self.direct_action_value_max_steps <= 0 or int(step) < self.direct_action_value_max_steps)
                        )
                        if use_direct_value:
                            if seq_slots is not None:
                                slot_step = seq_slots[:, step, :]
                            else:
                                slot_step = slot
                            reranked = self._rerank_action_pairs(
                                cur_cls,
                                cur_tok,
                                slot_step,
                                joint.flatten(),
                                s_idx.repeat_interleave(k),
                                x_idx.repeat(k),
                            )
                        if reranked is None:
                            flat = int(torch.argmax(joint).detach().cpu())
                            s_row = int(s_idx[flat // k].detach().cpu())
                            x_row = int(x_idx[flat % k].detach().cpu())
                        else:
                            s_row, x_row = reranked
                    if self.root_seq_edf_targets:
                        if s_row > 0:
                            s_row = self._edf_target_row(obs, 0, selected)
                        if x_row > 0:
                            x_row = self._edf_target_row(obs, 1, selected | ({s_row} if s_row > 0 else set()))
                    elif self.root_seq_base_targets:
                        if s_row > 0:
                            s_row = self._base_target_row(base_rankings, 0, selected)
                        if x_row > 0:
                            x_row = self._base_target_row(base_rankings, 1, selected | ({s_row} if s_row > 0 else set()))
                    s_row, x_row = self._override_targets(s_row, x_row, selected, service_pressure)
                    if single_sensor:
                        x_row = -1
                        out.append(action_from_row(s_row, 0))
                    else:
                        out.append(encode_joint_action(action_from_row(s_row, 0), action_from_row(x_row, 1)))
                    if s_row > 0:
                        selected.add(s_row)
                        selected_rec[0, s_row] = True
                        track_count += 1
                        s_track_atom = 1
                        s_search_atom = 0
                    else:
                        search_count += 1
                        s_search_count += 1
                        s_search_atom = 1
                        s_track_atom = 0
                    if (not single_sensor) and x_row > 0:
                        selected.add(x_row)
                        selected_rec[0, x_row] = True
                        track_count += 1
                        x_track_atom = 1
                        x_search_atom = 0
                    elif not single_sensor:
                        search_count += 1
                        x_search_count += 1
                        x_search_atom = 1
                        x_track_atom = 0
                    else:
                        x_search_atom = 0
                        x_track_atom = 0
                    search_atoms = s_search_atom + x_search_atom
                    track_atoms = s_track_atom + x_track_atom
                    if search_atoms > 0 and track_atoms <= 0:
                        search_streak += 1
                        track_streak = 0
                    elif track_atoms > 0 and search_atoms <= 0:
                        track_streak += 1
                        search_streak = 0
                    else:
                        search_streak = 0
                        track_streak = 0
                    needs_recurrent_state = bool(self.root_seq_recurrent_rescore)
                    if needs_recurrent_state and not (
                        self.lookahead_width > 0 and self.service_critic_weight != 0.0 and hasattr(self.g, "predict_service")
                    ):
                        if seq_slots is not None:
                            slot_step_update = seq_slots[:, step, :]
                        else:
                            slot_step_update = slot
                        action_pair = torch.tensor(
                            [[action_index_from_row(s_row, 0), -1 if single_sensor else action_index_from_row(x_row, 1)]],
                            dtype=torch.long,
                            device=self.device,
                        )
                        cur_cls, cur_tok, _r_p, _dt_p = self.g(cur_cls, cur_tok, slot_step_update, action_pair)
                return out

        def decode_with_steps(step_count: int, bias_override: float | None = None) -> list[int]:
            nonlocal search_bias
            old_search_bias = search_bias
            if bias_override is not None:
                search_bias = float(bias_override)
            cand = score_and_decode(None if self.root_seq_slot_passes <= 0 else [], step_count)
            for _ in range(max(0, self.root_seq_slot_passes - 1)):
                cand = score_and_decode(cand, step_count)
            search_bias = old_search_bias
            return cand

        def latent_terminal_service_details(plan: list[int]) -> dict[str, float]:
            proxy_value = float(self._plan_proxy_score(obs, plan))
            details = {
                "proxy": proxy_value,
                "service": 0.0,
                "combined": float(self.root_seq_terminal_proxy_weight) * proxy_value,
            }
            if self.root_seq_terminal_service_rerank_weight == 0.0 or not hasattr(self.g, "predict_service"):
                return details
            if not plan:
                return details
            cur_cls = cls
            cur_tok = tok
            seq_slots = build_seq_slots(plan, len(plan))
            max_roll = min(len(plan), int(getattr(self.g, "seq_len", self.max_steps)), int(self.max_steps))
            for step, action in enumerate(plan[:max_roll]):
                atoms = split_joint_action(int(action)) if is_joint_action(int(action)) else (int(action),)
                s_row = 0
                x_row = 0
                for atom in atoms:
                    row, sensor = xs_decode_action(int(atom), MAXT)
                    if int(sensor) == 0:
                        s_row = int(max(0, row))
                    else:
                        x_row = int(max(0, row))
                if seq_slots is not None:
                    slot_step = seq_slots[:, step, :]
                else:
                    slot_step = slot
                action_pair = torch.tensor(
                    [[action_index_from_row(s_row, 0), action_index_from_row(x_row, 1)]],
                    dtype=torch.long,
                    device=self.device,
                )
                cur_cls, cur_tok, _r_p, _dt_p = self.g(cur_cls, cur_tok, slot_step, action_pair)
            svc = self.g.predict_service(cur_cls, cur_tok)
            service_value = float(self._service_metric_value(svc).detach().cpu()[0])
            details["service"] = service_value
            details["combined"] = (
                float(self.root_seq_terminal_proxy_weight) * proxy_value
                + float(self.root_seq_terminal_service_rerank_weight) * service_value
            )
            return details

        def latent_terminal_service_score(plan: list[int]) -> float:
            return float(latent_terminal_service_details(plan)["combined"])

        def plan_signature(plan: list[int]) -> tuple[int, ...]:
            return tuple(int(x) for x in plan[: min(len(plan), int(self.max_steps))])

        def plan_search_atoms(plan: list[int]) -> int:
            total = 0
            for action in plan[: min(len(plan), int(self.max_steps))]:
                atoms = split_joint_action(int(action)) if is_joint_action(int(action)) else (int(action),)
                total += sum(1 for atom in atoms if int(xs_decode_action(int(atom), MAXT)[0]) == 0)
            return int(total)

        def select_plan(candidates: list[list[int]]) -> list[int]:
            if not candidates:
                return []
            scored_candidates = list(candidates)
            if self.root_seq_terminal_min_plan_frac > 0.0 and scored_candidates:
                max_len = max(len(p) for p in scored_candidates)
                min_len = int(np.ceil(float(self.root_seq_terminal_min_plan_frac) * float(max(1, max_len))))
                filtered = [p for p in scored_candidates if len(p) >= min_len]
                if filtered:
                    scored_candidates = filtered
            if self.root_seq_terminal_min_search_atoms > 0 and scored_candidates:
                filtered = [p for p in scored_candidates if plan_search_atoms(p) >= self.root_seq_terminal_min_search_atoms]
                if filtered:
                    scored_candidates = filtered
            if self.root_seq_terminal_service_rerank_weight != 0.0 and hasattr(self.g, "predict_service"):
                selected = max(scored_candidates, key=latent_terminal_service_score)
            else:
                selected = max(scored_candidates, key=lambda p: self._plan_proxy_score(obs, p))
            if self.root_seq_debug_candidates and candidates:
                details = [latent_terminal_service_details(p) for p in candidates]
                proxies = [float(d["proxy"]) for d in details]
                services = [float(d["service"]) for d in details]
                combined = [float(d["combined"]) for d in details]
                selected_idx = next((i for i, p in enumerate(candidates) if p is selected), 0)
                self.debug_candidate_rows.append(
                    {
                        "candidate_count": int(len(candidates)),
                        "scored_candidate_count": int(len(scored_candidates)),
                        "unique_plan_count": int(len({plan_signature(p) for p in candidates})),
                        "unique_scored_plan_count": int(len({plan_signature(p) for p in scored_candidates})),
                        "selected_idx": int(selected_idx),
                        "selected_len": int(len(selected)),
                        "selected_search_atoms": int(plan_search_atoms(selected)),
                        "max_candidate_len": int(max(len(p) for p in candidates)),
                        "max_candidate_search_atoms": int(max(plan_search_atoms(p) for p in candidates)),
                        "proxy_min": float(min(proxies)),
                        "proxy_max": float(max(proxies)),
                        "proxy_range": float(max(proxies) - min(proxies)),
                        "service_min": float(min(services)),
                        "service_max": float(max(services)),
                        "service_range": float(max(services) - min(services)),
                        "combined_min": float(min(combined)),
                        "combined_max": float(max(combined)),
                        "combined_range": float(max(combined) - min(combined)),
                        "selected_proxy": float(proxies[selected_idx]),
                        "selected_service": float(services[selected_idx]),
                        "selected_combined": float(combined[selected_idx]),
                    }
                )
            return selected

        bias_candidates = list(self.root_seq_search_bias_candidates)
        if bias_candidates:
            if not any(abs(float(b) - float(search_bias)) < 1e-6 for b in bias_candidates):
                bias_candidates.insert(0, float(search_bias))
            candidates = []
            for b in bias_candidates:
                cand = decode_with_steps(int(self.max_steps), bias_override=float(b))
                candidates.append(cand)
            plan = select_plan(candidates)
        elif self.root_seq_select_steps:
            step_counts = sorted(
                {
                    int(step_count)
                    for step_count in ([int(self.max_steps)] + self.root_seq_select_steps)
                    if int(step_count) > 0
                }
            )
            full_plan = decode_with_steps(max(step_counts))
            candidates = [full_plan[:step_count] for step_count in step_counts]
            plan = select_plan(candidates)
        else:
            plan = decode_with_steps(int(self.max_steps))
            if self.dual_plan_select and self.g_alt is not None:
                main_plan = plan
                orig_g = self.g
                orig_alt = self.g_alt
                orig_alpha = self.g_blend_alpha
                try:
                    self.g = orig_alt
                    self.g_alt = None
                    self.g_blend_alpha = 1.0
                    alt_plan = score_and_decode(None if self.root_seq_slot_passes <= 0 else [])
                    for _ in range(max(0, self.root_seq_slot_passes - 1)):
                        alt_plan = score_and_decode(alt_plan)
                finally:
                    self.g = orig_g
                    self.g_alt = orig_alt
                    self.g_blend_alpha = orig_alpha
                if self._plan_proxy_score(obs, alt_plan) > self._plan_proxy_score(obs, main_plan):
                    plan = alt_plan
        if self.root_seq_force_search_prefix > 0:
            search_action = (
                action_from_row(0, 0)
                if single_sensor
                else encode_joint_action(action_from_row(0, 0), action_from_row(0, 1))
            )
            plan = [int(search_action)] * int(self.root_seq_force_search_prefix) + list(plan)
        plan = self._service_sort_actions(obs, plan)
        plan = self._frontload_tracks(plan)
        plan = self._repair_pressure_searches(obs, plan)
        plan = self._repair_excess_search(obs, plan)
        return self._append_fill_actions(obs, plan)

    def _service_bonus_np(self, obs: dict) -> np.ndarray:
        bonus = np.zeros((MAXT + 1, 2), dtype=np.float32)
        if self.service_track_weight == 0.0 and self.service_search_weight == 0.0:
            return bonus
        active = np.asarray(obs.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
        deadline = np.asarray(obs.get("t_deadline", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)
        desired = np.asarray(obs.get("t_desired", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)
        priority = np.asarray(obs.get("priority", np.ones(MAXT, dtype=np.float32)), dtype=np.float32)
        n = min(MAXT, len(active), len(deadline), len(desired))
        if n > 0 and self.service_track_weight != 0.0:
            valid = active[:n] & (deadline[:n] >= 0.0)
            urgency = np.clip((1200.0 - deadline[:n]) / 1200.0, 0.0, 2.0)
            overdue = np.clip(-desired[:n] / 1000.0, 0.0, 3.0)
            if len(priority) >= n:
                pri = np.clip(priority[:n], 0.0, 3.0)
            else:
                pri = np.ones(n, dtype=np.float32)
            pressure = valid.astype(np.float32) * (1.0 + urgency + 0.75 * overdue + 0.25 * pri)
            bonus[1 : n + 1, :] = self.service_track_weight * pressure[:, None]
        if self.service_search_weight != 0.0:
            active_n = float(np.sum(active[:n])) if n > 0 else 0.0
            tracked = active[:n] & (deadline[:n] >= 0.0) if n > 0 else np.zeros(0, dtype=bool)
            untracked_frac = max(0.0, (active_n - float(np.sum(tracked))) / max(1.0, active_n))
            active_deficit = 0.0
            if self.service_active_goal > 0.0:
                active_deficit = max(0.0, (self.service_active_goal - active_n) / max(1.0, self.service_active_goal))
            bonus[0, :] = self.service_search_weight * (1.0 + untracked_frac + active_deficit)
        return bonus

    def _sensor_valid_np(self, obs: dict) -> np.ndarray:
        valid = np.zeros((MAXT + 1, 2), dtype=bool)
        s_free = float(obs.get("s_band_busy_ms", 0.0)) <= 0.0
        x_free = bool(int(obs.get("enable_x_band", 0))) and float(obs.get("x_band_busy_ms", 0.0)) <= 0.0
        valid[0, 0] = bool(s_free)
        valid[0, 1] = bool(x_free)
        active = np.asarray(obs.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
        deadline = np.asarray(obs.get("t_deadline", np.full(MAXT, -1.0, dtype=np.float32)), dtype=np.float32)
        rng = np.asarray(obs.get("target_range", np.full(MAXT, 50_000_000.0, dtype=np.float32)), dtype=np.float32)
        n = min(MAXT, len(active), len(deadline), len(rng))
        if n > 0:
            alive = active[:n] & (deadline[:n] >= 0.0)
            valid[1 : n + 1, 0] = alive & s_free & (10_000_000.0 < rng[:n]) & (rng[:n] < 184_000_000.0)
            valid[1 : n + 1, 1] = alive & x_free & (5_000_000.0 < rng[:n]) & (rng[:n] < 100_000_000.0)
        if not np.any(valid[:, 0]):
            valid[0, 0] = True
        if not np.any(valid[:, 1]):
            valid[0, 1] = True
        return valid

    def _apply_sensor_valid(self, score: torch.Tensor, sensor_valid: torch.Tensor | None) -> torch.Tensor:
        if sensor_valid is None:
            return score
        return score.masked_fill(~sensor_valid, -1e9)

    def _apply_search_cap(self, score: torch.Tensor, search_count: int, track_count: int) -> torch.Tensor:
        min_atoms = self._effective_min_window_search_atoms()
        remaining_search = int(min_atoms) - int(search_count)
        if min_atoms > 0 and remaining_search >= 2:
            forced = score.clone()
            forced[1:, :] = -1e9
            return forced
        if min_atoms > 0 and remaining_search == 1:
            forced = score.clone()
            forced[0, :] = forced[0, :] + 1e6
            return forced
        if self.max_window_search_atoms > 0 and int(search_count) >= self.max_window_search_atoms:
            capped = score.clone()
            capped[0, :] = -1e9
            return capped
        if self.max_window_search_frac <= 0.0:
            return score
        total_atoms = max(1, int(search_count) + int(track_count))
        if float(search_count) / float(total_atoms) <= self.max_window_search_frac:
            return score
        capped = score.clone()
        capped[0, :] = -1e9
        return capped

    def _apply_decoder_history_bias(self, score: torch.Tensor, search_streak, track_streak) -> torch.Tensor:
        if (
            self.decoder_history_search_streak_bias == 0.0
            and self.decoder_history_track_after_search_bias == 0.0
            and self.decoder_history_track_streak_bias == 0.0
            and self.decoder_history_search_after_track_bias == 0.0
        ):
            return score
        out = score.clone()
        search_streak_t = torch.as_tensor(search_streak, dtype=out.dtype, device=out.device)
        track_streak_t = torch.as_tensor(track_streak, dtype=out.dtype, device=out.device)
        if self.decoder_history_search_streak_bias != 0.0:
            out[0, :] = out[0, :] + float(self.decoder_history_search_streak_bias) * search_streak_t
        if self.decoder_history_track_after_search_bias != 0.0 and out.shape[0] > 1:
            out[1:, :] = out[1:, :] + float(self.decoder_history_track_after_search_bias) * search_streak_t
        if self.decoder_history_track_streak_bias != 0.0 and out.shape[0] > 1:
            out[1:, :] = out[1:, :] + float(self.decoder_history_track_streak_bias) * track_streak_t
        if self.decoder_history_search_after_track_bias != 0.0:
            out[0, :] = out[0, :] + float(self.decoder_history_search_after_track_bias) * track_streak_t
        return out

    def _apply_search_floor_joint(
        self,
        joint: torch.Tensor,
        s_idx: torch.Tensor,
        x_idx: torch.Tensor,
        search_count: int | torch.Tensor,
    ) -> torch.Tensor:
        min_atoms = self._effective_min_window_search_atoms()
        if min_atoms <= 0:
            return joint
        if isinstance(search_count, torch.Tensor):
            remaining = int((torch.as_tensor(min_atoms, device=search_count.device) - search_count).detach().cpu())
        else:
            remaining = int(min_atoms) - int(search_count)
        if remaining <= 0:
            return joint
        if remaining >= 2:
            allowed = (s_idx[:, None] <= 0) & (x_idx[None, :] <= 0)
        else:
            allowed = ((s_idx[:, None] <= 0) & (x_idx[None, :] > 0)) | ((s_idx[:, None] > 0) & (x_idx[None, :] <= 0))
        return joint.masked_fill(~allowed, -1e9)

    def _effective_min_window_search_atoms(self) -> int:
        if self.min_window_search_atoms <= 0:
            return 0
        if self.min_window_search_active_threshold > 0 and int(self._active_count_for_window) < int(self.min_window_search_active_threshold):
            return 0
        return int(self.min_window_search_atoms)

    def _service_pressure_np(self, obs: dict) -> np.ndarray:
        pressure = np.full(MAXT + 1, -1e9, dtype=np.float32)
        active = np.asarray(obs.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
        deadline = np.asarray(obs.get("t_deadline", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)
        desired = np.asarray(obs.get("t_desired", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)
        priority = np.asarray(obs.get("priority", np.ones(MAXT, dtype=np.float32)), dtype=np.float32)
        n = min(MAXT, len(active), len(deadline), len(desired))
        if n <= 0:
            return pressure
        valid = active[:n] & (deadline[:n] >= 0.0)
        urgency = np.clip((1500.0 - deadline[:n]) / 1500.0, 0.0, 2.0)
        overdue = np.clip(-desired[:n] / 1000.0, 0.0, 4.0)
        pri = np.clip(priority[:n], 0.0, 3.0) if len(priority) >= n else np.ones(n, dtype=np.float32)
        pressure[1 : n + 1] = np.where(valid, 1.0 + urgency + overdue + 0.25 * pri, -1e9)
        return pressure

    def _override_targets(self, s_row: int, x_row: int, selected: set[int], pressure: np.ndarray) -> tuple[int, int]:
        if not self.service_target_override:
            return s_row, x_row
        used = set(int(r) for r in selected)
        out = []
        for row in (int(s_row), int(x_row)):
            if row <= 0:
                out.append(0)
                continue
            best = 0
            best_score = -1e30
            for cand in np.argsort(-pressure)[:32]:
                cand = int(cand)
                if cand <= 0 or cand in used:
                    continue
                val = float(pressure[cand])
                if val > best_score:
                    best = cand
                    best_score = val
            if best > 0:
                used.add(best)
                out.append(best)
            else:
                out.append(row)
        return int(out[0]), int(out[1])

    def warmup(self, obs: dict) -> None:
        if self.device.type == "cuda" and (
            self.cuda_graph_step
            or self.cuda_graph_root_encode
            or self.cuda_graph_tensor_loop
            or self.tensor_loop
            or self.use_root_seq_policy
            or self.use_base_seq_policy
            or self.use_ar_seq_policy
        ):
            with torch.inference_mode():
                _ = self.plan(obs, budget_ms=200.0)

    def _plan_tensor_loop(self, obs: dict, budget_ms: float = 200.0):
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        self._apply_decode_router(obs)
        const_np = self._slot_constants(obs, 200.0)
        const = torch.tensor(const_np, dtype=torch.float32, device=self.device)
        service_bonus = torch.from_numpy(self._service_bonus_np(obs)).to(self.device)
        sensor_valid = torch.from_numpy(self._sensor_valid_np(obs)).to(self.device) if self.sensor_valid_mask else None
        dwell_np = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        s_duration_np = np.full(MAXT + 1, 10.0, dtype=np.float32)
        x_duration_np = np.full(MAXT + 1, 10.0, dtype=np.float32)
        n_dwell = min(MAXT, len(dwell_np))
        if n_dwell > 0:
            s_duration_np[1 : n_dwell + 1] = np.maximum(1.0, dwell_np[:n_dwell])
            x_duration_np[1 : n_dwell + 1] = np.maximum(1.0, dwell_np[:n_dwell] * 0.5)
        row_map = None
        x_np = tokenize(self.adapt, obs, selected=set(), search_count=0).astype(np.float32)
        compact = self._compact_sonly_tokens_and_dwell(x_np, obs)
        if compact is not None:
            x_np, s_duration_np, row_map = compact
            row_map_t = torch.from_numpy(row_map.astype(np.int64, copy=False)).to(self.device)
            service_bonus = service_bonus[row_map_t]
        else:
            row_map_t = torch.arange(int(x_np.shape[0]), dtype=torch.long, device=self.device)
        s_duration = torch.from_numpy(s_duration_np).to(self.device)
        x_duration = torch.from_numpy(x_duration_np).to(self.device)
        x = torch.from_numpy(x_np[None]).float().to(self.device)
        single_sensor = not bool(int(self.env_cfg.get("enable_x_band", 1)))
        clean_sonly_graph = (
            self.cuda_graph_tensor_loop
            and self.device.type == "cuda"
            and single_sensor
            and self.lookahead_width <= 0
            and self.direct_action_service_weight == 0.0
            and self.direct_action_value_weight == 0.0
            and self.direct_action_frame_weight == 0.0
            and self._effective_min_window_search_atoms() <= 0
            and self.max_window_search_atoms <= 0
            and self.max_window_search_frac <= 0.0
            and not self.sensor_valid_mask
        )
        if clean_sonly_graph:
            graph_shape = (int(x.shape[1]), int(s_duration.shape[0]))
            graph = self._graph_sonly_window_cache.get(graph_shape)
            if graph is None:
                graph = CudaGraphLatentSOnlyWindow(self, x, const, s_duration, service_bonus, row_map_t)
                self._graph_sonly_window_cache[graph_shape] = graph
                self._graph_sonly_window = graph
                self._graph_sonly_window_shape = graph_shape
            with torch.inference_mode():
                rows = graph(x, const, s_duration, service_bonus, row_map_t)
                row_list = rows.detach().cpu().tolist()
            if row_map is not None:
                row_list = [int(row_map[int(max(0, min(len(row_map) - 1, int(row))))]) for row in row_list]
            plan = [xs_s_search_action(MAXT) if int(row) <= 0 else xs_s_track_action(int(row), MAXT) for row in row_list]
            return self._append_fill_actions(obs, plan)
        with torch.inference_mode():
            elapsed = torch.zeros((), dtype=torch.float32, device=self.device)
            search_count = torch.zeros((), dtype=torch.float32, device=self.device)
            track_count = torch.zeros((), dtype=torch.float32, device=self.device)
            last_is_search = torch.zeros((), dtype=torch.float32, device=self.device)
            slot = torch.stack(
                [
                    elapsed / 200.0,
                    search_count / 20.0,
                    track_count / 100.0,
                    last_is_search,
                    const[0],
                    const[1],
                    const[2],
                    const[3],
                    const[4],
                    const[5],
                    const[6],
                ]
            ).reshape(1, 11)
            cls, tok, selected_t, token_active = self.model.backbone.encode_tokens(x)
            scores, q = self._score_latent(cls, tok, slot, selected_t, token_active)
            action_pair = torch.empty((1, 2), dtype=torch.long, device=self.device)
            if self.cuda_graph_step and self._graph_step is None:
                action_pair.zero_()
                self._graph_step = CudaGraphLatentStep(
                    self.model,
                    self.g,
                    cls,
                    tok,
                    slot,
                    action_pair,
                    selected_t,
                    token_active,
                    use_g_policy=self.use_g_policy,
                    use_model_q=abs(float(self.q_weight)) > 0.0,
                )
            out_actions = torch.empty((self.max_steps,), dtype=torch.long, device=self.device)
            used_steps = 0
            for step in range(self.max_steps):
                utility = self.policy_weight * scores + self.q_weight * q
                score = utility[0].clone() + service_bonus
                score[0, :] += self.search_score_bias
                if (
                    self._effective_min_window_search_atoms() > 0
                    or self.max_window_search_atoms > 0
                    or self.max_window_search_frac > 0.0
                ):
                    score = self._apply_search_cap(
                        score,
                        int(search_count.detach().cpu()),
                        int(track_count.detach().cpu()),
                    )
                score = self._apply_sensor_valid(score, sensor_valid)
                if single_sensor:
                    if (
                        self.direct_action_service_weight != 0.0
                        or self.direct_action_value_weight != 0.0
                        or self.direct_action_frame_weight != 0.0
                    ):
                        reranked = None
                        if self.direct_action_value_max_steps <= 0 or int(step) < self.direct_action_value_max_steps:
                            k = self._candidate_k(score.shape[0])
                            s_vals, s_idx = torch.topk(score[:, 0], k=k)
                            reranked = self._rerank_action_pairs(
                                cls,
                                tok,
                                slot,
                                s_vals,
                                s_idx,
                                torch.full_like(s_idx, -1),
                            )
                        s_row = (
                            torch.tensor(reranked[0], dtype=torch.long, device=self.device)
                            if reranked is not None
                            else torch.argmax(score[:, 0])
                        )
                    else:
                        if self.tensor_loop_factorized_decode:
                            search_score = score[0, 0]
                            track_scores = score[1:, 0]
                            track_mass = torch.logsumexp(track_scores, dim=0)
                            best_track = torch.argmax(track_scores) + 1
                            s_row = torch.where(search_score >= track_mass, torch.zeros_like(best_track), best_track)
                        else:
                            s_row = torch.argmax(score[:, 0])
                    out_actions[step] = torch.where(
                        s_row <= 0,
                        torch.tensor(MAXT + 3, device=self.device),
                        MAXT + 5 + s_row - 1,
                    )
                    used_steps = step + 1
                    action_pair[0, 0] = s_row.clamp_min(0) * 2
                    if self.single_sensor_noop_action:
                        action_pair[0, 1] = -1
                    else:
                        x_row = torch.zeros((), dtype=torch.long, device=self.device)
                        action_pair[0, 1] = x_row * 2 + 1
                    selected_t.scatter_(
                        1,
                        s_row.clamp_min(0).reshape(1, 1),
                        (s_row > 0).reshape(1, 1),
                    )
                    search_count = search_count + (s_row <= 0).float()
                    track_count = track_count + (s_row > 0).float()
                    last_is_search = (s_row <= 0).float()
                    elapsed = elapsed + self.planner_duration_scale * s_duration[s_row]
                    if float(elapsed.detach().cpu()) >= float(budget_ms):
                        break
                    slot = torch.stack(
                        [
                            elapsed / 200.0,
                            search_count / 20.0,
                            track_count / 100.0,
                            last_is_search,
                            const[0],
                            const[1],
                            const[2],
                            const[3],
                            const[4],
                            const[5],
                            const[6],
                        ]
                    ).reshape(1, 11)
                    if self._graph_step is not None:
                        cls, tok, scores, q, _r, _dt = self._graph_step(cls, tok, slot, action_pair, selected_t, token_active)
                    else:
                        cls, tok, _r, _dt = self.g(cls, tok, slot, action_pair)
                        scores, q = self._score_latent(cls, tok, slot, selected_t, token_active)
                    continue
                k = self._candidate_k(score.shape[0])
                s_vals, s_idx = torch.topk(score[:, 0], k=k)
                x_vals, x_idx = torch.topk(score[:, 1], k=k)
                joint = s_vals[:, None] + x_vals[None, :]
                conflict = (s_idx[:, None] > 0) & (s_idx[:, None] == x_idx[None, :])
                joint = joint.masked_fill(conflict, -1e9)
                joint = self._apply_search_floor_joint(joint, s_idx, x_idx, search_count)
                reranked = None
                cls_next = None
                tok_next = None
                if self.lookahead_width > 0:
                    rk = min(max(1, int(self.lookahead_width)), k)
                    cand_joint = joint[:rk, :rk].flatten()
                    cand_s = s_idx[:rk].repeat_interleave(rk)
                    cand_x = x_idx[:rk].repeat(rk)
                    finite = torch.isfinite(cand_joint) & (cand_joint > -1e8)
                    if bool(finite.any()):
                        cand_joint_f = cand_joint[finite]
                        cand_s_f = cand_s[finite]
                        cand_x_f = cand_x[finite]
                        cand_pair = torch.stack(
                            [cand_s_f.clamp_min(0) * 2, cand_x_f.clamp_min(0) * 2 + 1],
                            dim=1,
                        ).long()
                        cls_b = cls.expand(cand_pair.shape[0], -1)
                        tok_b = tok.expand(cand_pair.shape[0], -1, -1)
                        slot_b = slot.expand(cand_pair.shape[0], -1)
                        cls_cand, tok_cand, r_p, _dt_p = self.g(cls_b, tok_b, slot_b, cand_pair)
                        selected_cand = selected_t.expand(cand_pair.shape[0], -1).clone()
                        selected_cand.scatter_(1, cand_s_f.clamp_min(0).reshape(-1, 1), True)
                        selected_cand.scatter_(1, cand_x_f.clamp_min(0).reshape(-1, 1), True)
                        scores_n, q_n = self._score_latent(
                            cls_cand,
                            tok_cand,
                            slot_b,
                            selected_cand,
                            token_active.expand(cand_pair.shape[0], -1),
                        )
                        leaf = self._best_leaf_value(self.policy_weight * scores_n + self.q_weight * q_n, selected=set())
                        total = cand_joint_f + r_p + self.lookahead_leaf_weight * leaf + self._service_critic_value(cls_cand, tok_cand)
                        best = torch.argmax(total)
                        s_row = cand_s_f[best]
                        x_row = cand_x_f[best]
                        cls_next = cls_cand[best : best + 1]
                        tok_next = tok_cand[best : best + 1]
                use_direct_value = (
                    (self.direct_action_service_weight != 0.0 or self.direct_action_value_weight != 0.0 or self.direct_action_frame_weight != 0.0)
                    and (self.direct_action_value_max_steps <= 0 or int(step) < self.direct_action_value_max_steps)
                )
                if cls_next is None and use_direct_value:
                    reranked = self._rerank_action_pairs(
                        cls,
                        tok,
                        slot,
                        joint.flatten(),
                        s_idx.repeat_interleave(k),
                        x_idx.repeat(k),
                    )
                if cls_next is not None:
                    pass
                elif reranked is None:
                    flat = torch.argmax(joint)
                    s_row = s_idx[flat // k]
                    x_row = x_idx[flat % k]
                else:
                    s_row = torch.tensor(reranked[0], dtype=torch.long, device=self.device)
                    x_row = torch.tensor(reranked[1], dtype=torch.long, device=self.device)
                s_action = torch.where(s_row <= 0, torch.tensor(MAXT + 3, device=self.device), MAXT + 5 + s_row - 1)
                x_action = torch.where(x_row <= 0, torch.tensor(MAXT + 4, device=self.device), MAXT + 5 + MAXT + x_row - 1)
                out_actions[step] = 1_000_000 + s_action * 1_000 + x_action
                used_steps = step + 1
                action_pair[0, 0] = s_row.clamp_min(0) * 2
                action_pair[0, 1] = x_row.clamp_min(0) * 2 + 1
                selected_t.scatter_(1, s_row.clamp_min(0).reshape(1, 1), True)
                selected_t.scatter_(1, x_row.clamp_min(0).reshape(1, 1), True)
                is_search = (s_row <= 0).float() + (x_row <= 0).float()
                is_track = (s_row > 0).float() + (x_row > 0).float()
                search_count = search_count + is_search
                track_count = track_count + is_track
                last_is_search = (x_row <= 0).float()
                elapsed = elapsed + self.planner_duration_scale * torch.minimum(s_duration[s_row], x_duration[x_row])
                if float(elapsed.detach().cpu()) >= float(budget_ms):
                    break
                slot = torch.stack(
                    [
                        elapsed / 200.0,
                        search_count / 20.0,
                        track_count / 100.0,
                        last_is_search,
                        const[0],
                        const[1],
                        const[2],
                        const[3],
                        const[4],
                        const[5],
                        const[6],
                    ]
                ).reshape(1, 11)
                if cls_next is not None and tok_next is not None:
                    cls, tok = cls_next, tok_next
                    scores, q = self._score_latent(cls, tok, slot, selected_t, token_active)
                elif self._graph_step is not None:
                    cls, tok, scores, q, _r, _dt = self._graph_step(cls, tok, slot, action_pair, selected_t, token_active)
                else:
                    cls, tok, _r, _dt = self.g(cls, tok, slot, action_pair)
                    scores, q = self._score_latent(cls, tok, slot, selected_t, token_active)
            plan = out_actions[:used_steps].detach().cpu().tolist()
            return self._append_fill_actions(obs, plan)

    def predict_next_window_plan(
        self,
        obs: dict,
        executed_actions: list[int],
        remaining_actions: list[int],
        elapsed_ms: float,
        budget_ms: float = 200.0,
    ) -> list[int]:
        """Predict the boundary latent with g, then decode the next S-only window."""
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        executed_rows = []
        search_count = 0
        track_count = 0
        last = -1
        for action in executed_actions:
            atoms = split_joint_action(int(action)) if is_joint_action(int(action)) else (int(action),)
            for atom in atoms:
                row, sensor = xs_decode_action(int(atom), MAXT)
                if int(sensor or 0) != 0:
                    continue
                last = int(row)
                if int(row) <= 0:
                    search_count += 1
                else:
                    track_count += 1
                    executed_rows.append(int(row))
        x_np = tokenize(self.adapt, obs, selected=set(executed_rows), search_count=search_count).astype(np.float32)
        x = torch.from_numpy(x_np[None]).float().to(self.device)
        const_values = self._slot_constants(obs, budget_ms)
        const = torch.tensor(const_values, dtype=torch.float32, device=self.device)
        dwell_np = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        dwell = np.full((MAXT + 1,), 10.0, dtype=np.float32)
        n_dwell = min(MAXT, int(dwell_np.size))
        if n_dwell > 0:
            dwell[1 : n_dwell + 1] = np.maximum(1.0, dwell_np[:n_dwell])
        dwell_t = torch.from_numpy(dwell).to(self.device)
        with torch.inference_mode():
            cls, tok, selected_t, token_active = self.model.backbone.encode_tokens(x)
            s_valid = x[:, :, 10] > 0.5
            s_valid[:, 0] = True
            token_active = token_active & s_valid
            elapsed = float(elapsed_ms)
            for action in remaining_actions:
                row, sensor = xs_decode_action(int(action), MAXT)
                if int(sensor or 0) != 0:
                    continue
                row = int(max(0, min(MAXT, int(row))))
                if row <= 0:
                    search_count += 1
                else:
                    track_count += 1
                    selected_t[0, row] = True
                last = row
                elapsed += float(dwell[row])
                slot = torch.tensor(
                    [
                        elapsed / 200.0,
                        search_count / 20.0,
                        track_count / 100.0,
                        1.0 if row <= 0 else 0.0,
                        *const_values,
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )[None, :]
                pair = torch.tensor([[row * 2, -1]], dtype=torch.long, device=self.device)
                cls, tok, _reward, _duration = self.g(cls, tok, slot, pair)
                if elapsed >= 200.0:
                    break

            if self.device.type == "cuda" and self.cuda_graph_tensor_loop:
                row_map = torch.arange(MAXT + 1, dtype=torch.long, device=self.device)
                if self._graph_boundary_window is None:
                    self._graph_boundary_window = CudaGraphLatentBoundarySOnlyWindow(
                        self, cls, tok, token_active, const, dwell_t, row_map
                    )
                rows = self._graph_boundary_window(cls, tok, token_active, const, dwell_t, row_map)
                rows_cpu = rows.detach().cpu().tolist()
                out = []
                elapsed_next = 0.0
                for row in rows_cpu:
                    row = int(max(0, min(MAXT, int(row))))
                    out.append(xs_s_search_action(MAXT) if row <= 0 else xs_s_track_action(row, MAXT))
                    elapsed_next += float(dwell[row])
                    if elapsed_next >= float(budget_ms):
                        break
                return out

            selected_next = torch.zeros_like(selected_t)
            elapsed_next = 0.0
            searches_next = 0
            tracks_next = 0
            last_next = -1
            out = []
            for _step in range(min(20, self.max_steps)):
                slot = torch.tensor(
                    [
                        elapsed_next / 200.0,
                        searches_next / 20.0,
                        tracks_next / 100.0,
                        1.0 if last_next == 0 else 0.0,
                        *const_values,
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )[None, :]
                scores, q = self._score_latent(cls, tok, slot, selected_next, token_active)
                utility = self.policy_weight * scores[0, :, 0] + self.q_weight * q[0, :, 0]
                utility[0] += self.search_score_bias
                track_mass = torch.logsumexp(utility[1:], dim=0)
                best_track = torch.argmax(utility[1:]) + 1
                row_t = torch.where(utility[0] >= track_mass, torch.zeros_like(best_track), best_track)
                row = int(row_t.detach().cpu())
                out.append(xs_s_search_action(MAXT) if row <= 0 else xs_s_track_action(row, MAXT))
                if row <= 0:
                    searches_next += 1
                else:
                    tracks_next += 1
                    selected_next[0, row] = True
                last_next = row
                elapsed_next += float(dwell[row])
                pair = torch.tensor([[row * 2, -1]], dtype=torch.long, device=self.device)
                cls, tok, _reward, _duration = self.g(cls, tok, slot, pair)
                if elapsed_next >= float(budget_ms):
                    break
        return out

    def _slot_constants(self, obs: dict, budget_ms: float) -> tuple[float, float, float, float, float, float, float]:
        active = np.asarray(obs["active_mask"]).astype(bool)
        deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
        dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
        tracked = active & (deadline >= 0.0)
        workload = float(np.sum(dwell[tracked]) / max(1.0, budget_ms))
        min_deadline = float(np.min(deadline[tracked & (deadline > 0)])) if np.any(tracked & (deadline > 0)) else 0.0
        arrival_feature = x_enabled = float(obs.get("enable_x_band", 0.0))
        if float(obs.get("use_arrival_feature", 0.0)) > 0.5:
            arrival_feature += np.clip(float(obs.get("arrival_rate", 0.0)) / 10.0, 0.0, 2.0)
        slot8 = np.clip(float(obs.get("s_band_busy_ms", 0.0)) / 200.0, 0.0, 5.0)
        slot9 = np.clip(float(obs.get("x_band_busy_ms", 0.0)) / 200.0, 0.0, 5.0)
        slot10 = arrival_feature
        if float(obs.get("use_grid_feature", 0.0)) > 0.5:
            grid = np.asarray(obs.get("grid", []), dtype=np.float32)
            if grid.size == 0:
                slot8, slot9, slot10 = 0.0, 0.0, 0.0
            else:
                age = 3000.0 - grid
                overdue = np.maximum(0.0, age - 3000.0) / 3000.0
                slot8 = np.clip(float(np.mean(overdue)), 0.0, 5.0)
                slot9 = np.clip(float(np.mean(age > 4500.0)), 0.0, 1.0)
                slot10 = np.clip(float(np.max(age) / 4500.0), 0.0, 5.0)
        return (
            float(np.sum(tracked)) / 100.0,
            float(np.sum(tracked)) / 100.0,
            min(workload / 20.0, 2.0),
            min_deadline / 3000.0,
            slot8,
            slot9,
            slot10,
        )

    def _fill_slot_tensor(self, slot: torch.Tensor, elapsed: float, search_count: int, track_count: int, last: int, const: tuple[float, float, float, float, float, float, float], budget_ms: float) -> torch.Tensor:
        slot.zero_()
        slot[0, 0] = float(elapsed) / float(budget_ms)
        slot[0, 1] = float(search_count) / 20.0
        slot[0, 2] = float(track_count) / 100.0
        slot[0, 3] = 1.0 if int(last) == 0 else 0.0
        slot[0, 4] = const[0]
        slot[0, 5] = const[1]
        slot[0, 6] = const[2]
        slot[0, 7] = const[3]
        slot[0, 8] = const[4]
        slot[0, 9] = const[5]
        slot[0, 10] = const[6]
        return slot

    def _fill_slot_array(self, slot: np.ndarray, elapsed: float, search_count: int, track_count: int, last: int, const: tuple[float, float, float, float, float, float, float], budget_ms: float) -> np.ndarray:
        slot[0, 0] = float(elapsed) / float(budget_ms)
        slot[0, 1] = float(search_count) / 20.0
        slot[0, 2] = float(track_count) / 100.0
        slot[0, 3] = 1.0 if int(last) == 0 else 0.0
        slot[0, 4] = const[0]
        slot[0, 5] = const[1]
        slot[0, 6] = const[2]
        slot[0, 7] = const[3]
        slot[0, 8] = const[4]
        slot[0, 9] = const[5]
        slot[0, 10] = const[6]
        return slot

    def _row_duration(self, obs: dict, row: int, sensor: int) -> float:
        row = int(row)
        if row <= 0:
            return 10.0
        dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        dt = float(dwell[row - 1]) if row - 1 < len(dwell) else 10.0
        if int(sensor) == 1:
            dt *= 0.5
        return max(1.0, dt)

    def _plan_cached_base(self, obs: dict, budget_ms: float = 200.0):
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        x_np = tokenize(self.adapt, obs, selected=set(), search_count=0).astype(np.float32)
        x = torch.from_numpy(x_np[None]).float().to(self.device)
        selected: set[int] = set()
        search_count = 0
        track_count = 0
        last = -1
        elapsed = 0.0
        const = self._slot_constants(obs, 200.0)
        slot_np = np.zeros((1, 11), dtype=np.float32)
        service_bonus = torch.from_numpy(self._service_bonus_np(obs)).to(self.device)
        sensor_valid = torch.from_numpy(self._sensor_valid_np(obs)).to(self.device) if self.sensor_valid_mask else None
        duration_penalty = None
        if self.duration_penalty_weight != 0.0:
            dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
            s_duration = np.full(MAXT + 1, 10.0, dtype=np.float32)
            x_duration = np.full(MAXT + 1, 10.0, dtype=np.float32)
            n_dwell = min(MAXT, len(dwell))
            if n_dwell > 0:
                s_duration[1 : n_dwell + 1] = np.maximum(1.0, dwell[:n_dwell])
                x_duration[1 : n_dwell + 1] = np.maximum(1.0, dwell[:n_dwell] * 0.5)
            duration_penalty = torch.from_numpy(self.duration_penalty_weight * np.stack([s_duration, x_duration], axis=1).astype(np.float32) / 10.0).to(self.device)
        with torch.inference_mode():
            cls, tok, selected_t, token_active = self.model.backbone.encode_tokens(x)
            plan = []
            for _step in range(int(self.max_steps)):
                self._fill_slot_array(slot_np, elapsed, search_count, track_count, last, const, 200.0)
                slot = torch.from_numpy(slot_np).float().to(self.device)
                scores, q = latent_scores(self.model, cls, tok, slot, selected_t, token_active)
                s_row, x_row = self._choose_rows(self.policy_weight * scores + self.q_weight * q, selected, service_bonus, duration_penalty, sensor_valid, search_count, track_count)
                plan.append(encode_joint_action(action_from_row(s_row, 0), action_from_row(x_row, 1)))
                if s_row > 0:
                    selected.add(s_row)
                    selected_t[0, s_row] = True
                    track_count += 1
                    last = s_row
                else:
                    search_count += 1
                    last = 0
                if x_row > 0:
                    selected.add(x_row)
                    selected_t[0, x_row] = True
                    track_count += 1
                    last = x_row
                else:
                    search_count += 1
                    last = 0
                elapsed += self.planner_duration_scale * min(self._row_duration(obs, s_row, 0), self._row_duration(obs, x_row, 1))
                if elapsed >= float(budget_ms):
                    break
        return self._append_fill_actions(obs, plan)

    def _choose_rows(self, utility: torch.Tensor, selected: set[int], service_bonus: torch.Tensor | None = None, duration_penalty: torch.Tensor | None = None, sensor_valid: torch.Tensor | None = None, search_count: int = 0, track_count: int = 0) -> tuple[int, int]:
        score = utility[0].clone()
        if service_bonus is not None:
            score = score + service_bonus
        if duration_penalty is not None:
            score = score - duration_penalty
        score[0, :] += self.search_score_bias
        score = self._apply_search_cap(score, search_count, track_count)
        score = self._apply_sensor_valid(score, sensor_valid)
        if self.single_sensor_s_only_choose and not bool(int(self.env_cfg.get("enable_x_band", 1))):
            return int(torch.argmax(score[:, 0]).detach().cpu()), 0
        k = self._candidate_k(score.shape[0])
        s_vals, s_idx = torch.topk(score[:, 0], k=k)
        x_vals, x_idx = torch.topk(score[:, 1], k=k)
        joint = s_vals[:, None] + x_vals[None, :]
        conflict = (s_idx[:, None] > 0) & (s_idx[:, None] == x_idx[None, :])
        if self.avoid_double_search:
            conflict = conflict | ((s_idx[:, None] <= 0) & (x_idx[None, :] <= 0))
        joint = joint.masked_fill(conflict, -1e9)
        flat = int(torch.argmax(joint).detach().cpu())
        si = flat // k
        xi = flat % k
        return int(s_idx[si].detach().cpu()), int(x_idx[xi].detach().cpu())

    def _candidate_pairs(self, utility: torch.Tensor, selected: set[int], service_bonus: torch.Tensor | None = None, duration_penalty: torch.Tensor | None = None, sensor_valid: torch.Tensor | None = None, search_count: int = 0, track_count: int = 0) -> list[tuple[int, int]]:
        score = utility[0].clone()
        if service_bonus is not None:
            score = score + service_bonus
        if duration_penalty is not None:
            score = score - duration_penalty
        score[0, :] += self.search_score_bias
        score = self._apply_search_cap(score, search_count, track_count)
        score = self._apply_sensor_valid(score, sensor_valid)
        k = min(max(1, self.lookahead_width), int(score.shape[0]))
        _sv, s_idx = torch.topk(score[:, 0], k=k)
        _xv, x_idx = torch.topk(score[:, 1], k=k)
        pairs = []
        for sr in s_idx.detach().cpu().tolist():
            for xr in x_idx.detach().cpu().tolist():
                if int(sr) > 0 and int(sr) == int(xr):
                    continue
                pairs.append((int(sr), int(xr)))
        return pairs[: max(1, k * k)]

    def _best_leaf_value(self, utility: torch.Tensor, selected: set[int]) -> torch.Tensor:
        score = utility.clone()
        score[:, 0, :] += self.search_score_bias
        for row in selected:
            if 0 <= int(row) < score.shape[1]:
                score[:, int(row), :] = -1e9
        k = self._leaf_k(score.shape[1])
        s_vals, s_idx = torch.topk(score[:, :, 0], k=k, dim=1)
        x_vals, x_idx = torch.topk(score[:, :, 1], k=k, dim=1)
        joint = s_vals[:, :, None] + x_vals[:, None, :]
        conflict = (s_idx[:, :, None] > 0) & (s_idx[:, :, None] == x_idx[:, None, :])
        return joint.masked_fill(conflict, -1e9).flatten(1).max(dim=1).values

    def _service_critic_value(self, cls: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        if self.service_critic_weight == 0.0 or not hasattr(self.g, "predict_service"):
            return cls.new_zeros((cls.shape[0],))
        svc = self.g.predict_service(cls, tok)
        value = self._service_metric_value(svc)
        return self.service_critic_weight * value

    def _service_metric_value(self, svc: torch.Tensor) -> torch.Tensor:
        active = svc[:, 0]
        tracked = svc[:, 1]
        drop = svc[:, 2]
        delay = svc[:, 3]
        return (
            self.service_critic_active_weight * active
            + self.service_critic_tracked_weight * tracked
            - self.service_critic_drop_weight * drop
            - self.service_critic_delay_weight * delay
        )

    def _shared_action_head_features(
        self,
        cls: torch.Tensor,
        tok: torch.Tensor,
        slot: torch.Tensor,
        action_pair: torch.Tensor,
    ) -> torch.Tensor | None:
        """Build action-head inputs without expanding the full token table.

        Candidate reranking often scores many action pairs that share one latent
        state and one slot vector.  The generic predict_* methods work, but they
        materialize a [candidate, target, dim] expanded token view internally.
        This fast path only gathers the two selected target/search tokens.
        """
        if (
            cls.shape[0] != action_pair.shape[0]
            or tok.shape[0] != action_pair.shape[0]
            or slot.shape[0] != action_pair.shape[0]
            or action_pair.shape[0] <= 1
            or cls.stride(0) != 0
            or tok.stride(0) != 0
            or slot.stride(0) != 0
            or not hasattr(self.g, "action_emb")
            or not hasattr(self.g, "slot_proj")
        ):
            return None
        cls0, tok0_all = self.g._project_state(cls[:1], tok[:1])
        pair = action_pair.clamp(min=0, max=self.g.action_emb.num_embeddings - 1)
        if hasattr(self.g, "_action_embedding"):
            a0 = self.g._action_embedding(action_pair[:, 0])
            a1 = self.g._action_embedding(action_pair[:, 1])
        else:
            a0 = self.g.action_emb(pair[:, 0])
            a1 = self.g.action_emb(pair[:, 1])
        rows = (pair // 2).clamp(min=0, max=tok0_all.shape[1] - 1)
        tok0 = tok0_all[0, rows[:, 0]]
        tok1 = tok0_all[0, rows[:, 1]]
        slot_e = self.g.slot_proj(slot[:1]).expand(action_pair.shape[0], -1)
        cls_e = cls0.expand(action_pair.shape[0], -1)
        return torch.cat([cls_e, slot_e, a0, a1, tok0, tok1], dim=-1)

    def _action_service_value(self, cls: torch.Tensor, tok: torch.Tensor, slot: torch.Tensor, action_pair: torch.Tensor) -> torch.Tensor:
        out = cls.new_zeros((action_pair.shape[0],))
        if self.direct_action_service_weight != 0.0 and hasattr(self.g, "predict_action_service"):
            svc = self.g.predict_action_service(cls, tok, slot, action_pair)
            active = svc[:, 0]
            tracked = svc[:, 1]
            drop = svc[:, 2]
            delay = svc[:, 3]
            value = (
                self.service_critic_active_weight * active
                + self.service_critic_tracked_weight * tracked
                - self.service_critic_drop_weight * drop
                - self.service_critic_delay_weight * delay
            )
            out = out + self.direct_action_service_weight * value
        if self.direct_action_value_weight != 0.0 and hasattr(self.g, "predict_action_value"):
            shared_features = None
            if hasattr(self.g, "action_value_head"):
                shared_features = self._shared_action_head_features(cls, tok, slot, action_pair)
            if shared_features is not None:
                value = self.g.action_value_head(shared_features).squeeze(-1)
            else:
                value = self.g.predict_action_value(cls, tok, slot, action_pair)
            value = value * float(getattr(self.g, "action_value_target_std", 1.0)) + float(
                getattr(self.g, "action_value_target_mean", 0.0)
            )
            out = out + self.direct_action_value_weight * value
        if self.direct_action_frame_weight != 0.0 and hasattr(self.g, "predict_action_frame"):
            frame = self.g.predict_action_frame(cls, tok, slot, action_pair)
            out = out - self.direct_action_frame_weight * frame
        return out

    def _root_action_value_pairs(self, cls: torch.Tensor, tok: torch.Tensor, slot: torch.Tensor, action_pair: torch.Tensor) -> torch.Tensor:
        """Value scores for many root-sequence action pairs sharing one latent state."""
        if self.direct_action_value_weight == 0.0 or not hasattr(self.g, "action_value_head"):
            return cls.new_zeros((action_pair.shape[0],))
        cls0, tok0_all = self.g._project_state(cls[:1], tok[:1])
        pair = action_pair.clamp(min=0, max=self.g.action_emb.num_embeddings - 1)
        if hasattr(self.g, "_action_embedding"):
            a0 = self.g._action_embedding(action_pair[:, 0])
            a1 = self.g._action_embedding(action_pair[:, 1])
        else:
            a0 = self.g.action_emb(pair[:, 0])
            a1 = self.g.action_emb(pair[:, 1])
        rows = (pair // 2).clamp(min=0, max=tok0_all.shape[1] - 1)
        tok0 = tok0_all[0, rows[:, 0]]
        tok1 = tok0_all[0, rows[:, 1]]
        slot_e = self.g.slot_proj(slot)
        cls_e = cls0.expand(action_pair.shape[0], -1)
        value = self.g.action_value_head(torch.cat([cls_e, slot_e, a0, a1, tok0, tok1], dim=-1)).squeeze(-1)
        value = value * float(getattr(self.g, "action_value_target_std", 1.0)) + float(
            getattr(self.g, "action_value_target_mean", 0.0)
        )
        return self.direct_action_value_weight * value

    def _rerank_cached_action_pairs(
        self,
        cand_joint: torch.Tensor,
        cand_s: torch.Tensor,
        cand_x: torch.Tensor,
        cache_s: torch.Tensor,
        cache_x: torch.Tensor,
        cache_value: torch.Tensor,
    ) -> tuple[int, int] | None:
        finite = torch.isfinite(cand_joint) & (cand_joint > -1e8)
        if not bool(finite.any()):
            return None
        cand_joint = cand_joint[finite]
        cand_s = cand_s[finite]
        cand_x = cand_x[finite]
        if self.direct_action_value_margin_threshold > 0.0 and cand_joint.numel() > 1:
            top2 = torch.topk(cand_joint, k=2).values
            if float((top2[0] - top2[1]).detach().cpu()) >= self.direct_action_value_margin_threshold:
                return None
        ps = (cand_s[:, None] == cache_s[None, :]).to(torch.float32)
        px = (cand_x[:, None] == cache_x[None, :]).to(torch.float32)
        has = (ps.sum(dim=1) > 0.5) & (px.sum(dim=1) > 0.5)
        if not bool(has.all()):
            return None
        si = ps.argmax(dim=1)
        xi = px.argmax(dim=1)
        total = cand_joint + cache_value[si, xi]
        best = int(torch.argmax(total).detach().cpu())
        return int(cand_s[best].detach().cpu()), int(cand_x[best].detach().cpu())

    def _rerank_action_pairs(
        self,
        cls: torch.Tensor,
        tok: torch.Tensor,
        slot_step: torch.Tensor,
        cand_joint: torch.Tensor,
        cand_s: torch.Tensor,
        cand_x: torch.Tensor,
    ) -> tuple[int, int] | None:
        finite = torch.isfinite(cand_joint) & (cand_joint > -1e8)
        if not bool(finite.any()):
            return None
        cand_joint = cand_joint[finite]
        cand_s = cand_s[finite]
        cand_x = cand_x[finite]
        if (
            self.direct_action_value_margin_threshold > 0.0
            and cand_joint.numel() > 1
        ):
            top2 = torch.topk(cand_joint, k=2).values
            if float((top2[0] - top2[1]).detach().cpu()) >= self.direct_action_value_margin_threshold:
                return None
        single_sensor = not bool(int(self.env_cfg.get("enable_x_band", 1)))
        if single_sensor:
            action_pair = torch.stack(
                [cand_s.clamp_min(0) * 2, torch.full_like(cand_s, -1)],
                dim=1,
            ).long()
        else:
            action_pair = torch.stack(
                [cand_s.clamp_min(0) * 2, cand_x.clamp_min(0) * 2 + 1],
                dim=1,
            ).long()
        cls_b = cls.expand(action_pair.shape[0], -1)
        tok_b = tok.expand(action_pair.shape[0], -1, -1)
        slot_b = slot_step.expand(action_pair.shape[0], -1)
        with torch.inference_mode():
            total = cand_joint + self._action_service_value(cls_b, tok_b, slot_b, action_pair)
            best = int(torch.argmax(total).detach().cpu())
        return int(cand_s[best].detach().cpu()), int(cand_x[best].detach().cpu())

    def _score_latent(self, cls: torch.Tensor, tok: torch.Tensor, slot: torch.Tensor, selected: torch.Tensor, token_active: torch.Tensor):
        if self.use_g_policy and hasattr(self.g, "policy_scores"):
            scores = self.g.policy_scores(cls, tok, slot, selected, token_active)
            if self.g_alt is not None and hasattr(self.g_alt, "policy_scores") and self.g_blend_alpha < 1.0:
                alt_scores = self.g_alt.policy_scores(cls, tok, slot, selected, token_active)
                scores = self.g_blend_alpha * scores + (1.0 - self.g_blend_alpha) * alt_scores
            if abs(float(self.q_weight)) > 0.0:
                _model_scores, q = latent_scores(self.model, cls, tok, slot, selected, token_active)
            else:
                q = torch.zeros_like(scores)
            return scores, q
        return latent_scores(self.model, cls, tok, slot, selected, token_active)

    def _choose_rows_lookahead(self, obs: dict, cls: torch.Tensor, tok: torch.Tensor, slot: torch.Tensor, utility: torch.Tensor, selected: set[int], elapsed: float, search_count: int, track_count: int, last: int, root_selected: torch.Tensor, token_active: torch.Tensor, service_bonus: torch.Tensor | None = None, duration_penalty: torch.Tensor | None = None, sensor_valid: torch.Tensor | None = None):
        pairs = self._candidate_pairs(utility, selected, service_bonus, duration_penalty, sensor_valid, search_count, track_count)
        if len(pairs) <= 1:
            row = pairs[0] if pairs else self._choose_rows(utility, selected, sensor_valid=sensor_valid, search_count=search_count, track_count=track_count)
            action_pair = torch.tensor([[action_index_from_row(row[0], 0), action_index_from_row(row[1], 1)]], dtype=torch.long, device=self.device)
            with torch.inference_mode():
                cls_n, tok_n, _r, _dt = self.g(cls, tok, slot, action_pair)
            return row[0], row[1], cls_n, tok_n
        action_pair = torch.tensor(
            [[action_index_from_row(sr, 0), action_index_from_row(xr, 1)] for sr, xr in pairs],
            dtype=torch.long,
            device=self.device,
        )
        cls_b = cls.expand(len(pairs), -1)
        tok_b = tok.expand(len(pairs), -1, -1)
        slot_b = slot.expand(len(pairs), -1)
        with torch.inference_mode():
            cls_n, tok_n, r_p, _dt_p = self.g(cls_b, tok_b, slot_b, action_pair)
        slot_rows = []
        selected_rows = root_selected.expand(len(pairs), -1).clone()
        for idx, (sr, xr) in enumerate(pairs):
            sel = set(selected)
            sc = int(search_count)
            tc = int(track_count)
            la = int(last)
            if int(sr) > 0:
                sel.add(int(sr))
                tc += 1
                la = int(sr)
            else:
                sc += 1
                la = 0
            if int(xr) > 0:
                sel.add(int(xr))
                tc += 1
                la = int(xr)
            else:
                sc += 1
                la = 0
            for row in sel:
                if 0 <= int(row) < selected_rows.shape[1]:
                    selected_rows[idx, int(row)] = True
            dt = min(self._row_duration(obs, sr, 0), self._row_duration(obs, xr, 1))
            slot_rows.append(slot_features(obs, elapsed + float(dt), sc, tc, la, 200.0).astype(np.float32))
        slot_next = torch.from_numpy(np.stack(slot_rows)).float().to(self.device)
        with torch.inference_mode():
            scores_n, q_n = self._score_latent(cls_n, tok_n, slot_next, selected_rows, token_active.expand(len(pairs), -1))
            leaf = self._best_leaf_value(self.policy_weight * scores_n + self.q_weight * q_n, selected=set())
            service_value = self._service_critic_value(cls_n, tok_n)
            total = r_p + self.lookahead_leaf_weight * leaf + service_value
            best_idx = int(torch.argmax(total).detach().cpu())
        return pairs[best_idx][0], pairs[best_idx][1], cls_n[best_idx : best_idx + 1], tok_n[best_idx : best_idx + 1]

    def plan(self, obs: dict, budget_ms: float = 200.0):
        if self.g_alt is not None and self.router_active_threshold > 0:
            active = np.asarray(obs.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
            if int(active[:MAXT].sum()) >= int(self.router_active_threshold):
                orig_g = self.g
                orig_search_bias = self.search_score_bias
                orig_service_track = self.service_track_weight
                orig_service_search = self.service_search_weight
                orig_base_service_track = self.base_service_track_weight
                orig_base_service_search = self.base_service_search_weight
                orig_search_frac = self.max_window_search_frac
                orig_base_search_frac = self.base_max_window_search_frac
                orig_sort_prefix = self.service_sort_search_prefix
                orig_repair_atoms = self.pressure_repair_max_atoms
                try:
                    self.g = self.g_alt
                    if self.router_alt_search_bias is not None:
                        self.search_score_bias = float(self.router_alt_search_bias)
                    if self.router_alt_service_track_weight is not None:
                        self.service_track_weight = float(self.router_alt_service_track_weight)
                        self.base_service_track_weight = float(self.router_alt_service_track_weight)
                    if self.router_alt_service_search_weight is not None:
                        self.service_search_weight = float(self.router_alt_service_search_weight)
                        self.base_service_search_weight = float(self.router_alt_service_search_weight)
                    if self.router_alt_max_window_search_frac is not None:
                        self.max_window_search_frac = float(self.router_alt_max_window_search_frac)
                        self.base_max_window_search_frac = float(self.router_alt_max_window_search_frac)
                    if self.router_alt_service_sort_search_prefix is not None:
                        self.service_sort_search_prefix = max(0, int(self.router_alt_service_sort_search_prefix))
                    if self.router_alt_pressure_repair_max_atoms is not None:
                        self.pressure_repair_max_atoms = int(self.router_alt_pressure_repair_max_atoms)
                    return self._plan_root_sequence(obs, budget_ms)
                finally:
                    self.g = orig_g
                    self.search_score_bias = orig_search_bias
                    self.service_track_weight = orig_service_track
                    self.service_search_weight = orig_service_search
                    self.base_service_track_weight = orig_base_service_track
                    self.base_service_search_weight = orig_base_service_search
                    self.max_window_search_frac = orig_search_frac
                    self.base_max_window_search_frac = orig_base_search_frac
                    self.service_sort_search_prefix = orig_sort_prefix
                    self.pressure_repair_max_atoms = orig_repair_atoms
        if (self.use_root_seq_policy and hasattr(self.g, "sequence_scores")) or self.use_base_seq_policy or self.use_ar_seq_policy:
            return self._plan_root_sequence(obs, budget_ms)
        if self.use_cached_base_policy:
            return self._plan_cached_base(obs, budget_ms)
        if self.tensor_loop:
            return self._plan_tensor_loop(obs, budget_ms)
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        selected: set[int] = set()
        search_count = 0
        track_count = 0
        last = -1
        elapsed = 0.0
        const = self._slot_constants(obs, 200.0)
        service_bonus = torch.from_numpy(self._service_bonus_np(obs)).to(self.device)
        sensor_valid = torch.from_numpy(self._sensor_valid_np(obs)).to(self.device) if self.sensor_valid_mask else None
        service_pressure = self._service_pressure_np(obs)
        dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        s_duration = np.full(MAXT + 1, 10.0, dtype=np.float32)
        x_duration = np.full(MAXT + 1, 10.0, dtype=np.float32)
        n_dwell = min(MAXT, len(dwell))
        if n_dwell > 0:
            s_duration[1 : n_dwell + 1] = np.maximum(1.0, dwell[:n_dwell])
            x_duration[1 : n_dwell + 1] = np.maximum(1.0, dwell[:n_dwell] * 0.5)
        duration_penalty = None
        if self.duration_penalty_weight != 0.0:
            duration_np = np.stack([s_duration, x_duration], axis=1).astype(np.float32)
            duration_penalty = torch.from_numpy(self.duration_penalty_weight * duration_np / 10.0).to(self.device)
        use_graph = self.cuda_graph_step and self.device.type == "cuda" and self.lookahead_width <= 0
        slot = torch.empty((1, 11), dtype=torch.float32, device=self.device)
        self._fill_slot_tensor(slot, elapsed, search_count, track_count, last, const, 200.0)
        x_np = tokenize(self.adapt, obs, selected=selected, search_count=search_count).astype(np.float32)
        x = torch.from_numpy(x_np[None]).float().to(self.device)
        with torch.inference_mode():
            if self.cuda_graph_root_encode and self.device.type == "cuda" and self.router_active_threshold <= 0:
                if self._graph_root_encode is None:
                    self._graph_root_encode = CudaGraphRootEncodeScore(
                        self.model,
                        self.g,
                        x,
                        slot,
                        use_g_policy=self.use_g_policy,
                        use_model_q=abs(float(self.q_weight)) > 0.0,
                        g_alt=self.g_alt,
                        g_blend_alpha=self.g_blend_alpha,
                    )
                cls, tok, root_selected, token_active, scores, q = self._graph_root_encode(x, slot)
            else:
                cls, tok, root_selected, token_active = self.model.backbone.encode_tokens(x)
                scores, q = self._score_latent(cls, tok, slot, root_selected, token_active)
        plan = []
        selected_t = root_selected.clone()
        action_pair = torch.empty((1, 2), dtype=torch.long, device=self.device)
        if use_graph and self._graph_step is None:
            action_pair.zero_()
            self._graph_step = CudaGraphLatentStep(self.model, self.g, cls, tok, slot, action_pair, selected_t, token_active, use_g_policy=self.use_g_policy)
        for _ in range(self.max_steps):
            if elapsed >= float(budget_ms):
                break
            if not use_graph:
                self._fill_slot_tensor(slot, elapsed, search_count, track_count, last, const, 200.0)
            utility = self.policy_weight * scores + self.q_weight * q
            if self.lookahead_width > 0:
                s_row, x_row, cls_next, tok_next = self._choose_rows_lookahead(
                    obs, cls, tok, slot, utility, selected, elapsed, search_count, track_count, last, root_selected, token_active, service_bonus, duration_penalty, sensor_valid
                )
            else:
                s_row, x_row = self._choose_rows(utility, selected, service_bonus, duration_penalty, sensor_valid, search_count, track_count)
                cls_next = tok_next = None
            s_row, x_row = self._override_targets(s_row, x_row, selected, service_pressure)
            s_action = int(self._s_actions[int(s_row)])
            x_action = int(self._x_actions[int(x_row)])
            plan.append(encode_joint_action(s_action, x_action))
            if s_row > 0:
                selected.add(int(s_row))
                if 0 <= int(s_row) < selected_t.shape[1]:
                    selected_t[:, int(s_row)] = True
                track_count += 1
                last = int(s_row)
            else:
                search_count += 1
                last = 0
            if x_row > 0:
                selected.add(int(x_row))
                if 0 <= int(x_row) < selected_t.shape[1]:
                    selected_t[:, int(x_row)] = True
                track_count += 1
                last = int(x_row)
            else:
                search_count += 1
                last = 0
            dt = self.planner_duration_scale * min(float(s_duration[int(s_row)]), float(x_duration[int(x_row)]))
            elapsed += float(dt)
            if cls_next is None or tok_next is None:
                with torch.inference_mode():
                    if self._graph_step is not None:
                        action_pair[0, 0] = action_index_from_row(s_row, 0)
                        if self.single_sensor_noop_action and not bool(int(self.env_cfg.get("enable_x_band", 1))):
                            action_pair[0, 1] = -1
                        else:
                            action_pair[0, 1] = action_index_from_row(x_row, 1)
                        self._fill_slot_tensor(slot, elapsed, search_count, track_count, last, const, 200.0)
                        cls, tok, scores, q, _r, _dt = self._graph_step(cls, tok, slot, action_pair, selected_t, token_active)
                    else:
                        action_pair[0, 0] = action_index_from_row(s_row, 0)
                        if self.single_sensor_noop_action and not bool(int(self.env_cfg.get("enable_x_band", 1))):
                            action_pair[0, 1] = -1
                        else:
                            action_pair[0, 1] = action_index_from_row(x_row, 1)
                        self._fill_slot_tensor(slot, elapsed, search_count, track_count, last, const, 200.0)
                        cls, tok, _r, _dt = self.g(cls, tok, slot, action_pair)
                        scores, q = self._score_latent(cls, tok, slot, selected_t, token_active)
            else:
                cls, tok = cls_next, tok_next
                with torch.inference_mode():
                    scores, q = self._score_latent(cls, tok, slot, selected_t, token_active)
        return self._append_fill_actions(obs, plan)


def execute_plan_until_budget_joint_shaped(eng, plan, budget_ms: float, search_debt_ms: float, planner_name: str, seed: int, window_idx: int, env_cfg: dict):
    spent_ms = 0.0
    total_reward = 0.0
    search_actions = 0
    executed = 0
    rows = []
    slot = 0
    for action in plan:
        if spent_ms >= float(budget_ms) or bool(eng.term_buf[0]):
            break
        obs_before = get_obs(eng, search_debt_ms)
        reward, dt, executed_action = execute_first_valid_action_joint(eng, [int(action)], float(budget_ms) - spent_ms)
        if executed_action is None or dt <= 0.0:
            continue
        atoms = split_joint_action(executed_action) if is_joint_action(executed_action) else (int(executed_action),)
        x_enabled = bool(int(env_cfg.get("enable_x_band", 1)))
        is_search = []
        for atom in atoms:
            base, sensor = xs_decode_action(int(atom), MAXT)
            if int(sensor or 0) == 1 and not x_enabled:
                # In S-only latent/joint evaluation the disabled X stream is carried
                # as an X-search-shaped placeholder. It is a no-op, not a real search.
                is_search.append(False)
            else:
                is_search.append(int(base) == 0)
        next_search_debt_ms = 0.0 if any(is_search) else float(search_debt_ms) + float(dt)
        obs_after = get_obs(eng, next_search_debt_ms)
        shaped_reward = shaped_step_reward(float(reward), float(dt), obs_before, obs_after, env_cfg, action=int(executed_action))
        total_reward += float(shaped_reward)
        spent_ms += float(dt)
        search_debt_ms = next_search_debt_ms
        search_actions += int(any(is_search))
        executed += 1
        if len(atoms) > 1:
            s_atom = int(atoms[0])
            x_atom = int(atoms[1])
        else:
            _base, _sensor = xs_decode_action(int(atoms[0]), MAXT)
            sensor_i = int(_sensor or 0)
            s_atom = int(atoms[0]) if sensor_i == 0 else -1
            x_atom = int(atoms[0]) if sensor_i == 1 else -1
        rows.append(
            {
                "planner": planner_name,
                "seed": int(seed),
                "bucket": int(window_idx),
                "slot": int(slot),
                "action": int(executed_action),
                "s_action": int(s_atom),
                "x_action": int(x_atom),
                "action_type": "Joint" if len(atoms) > 1 else "Atomic",
                "reward": float(shaped_reward),
                "base_reward": float(reward),
                "dt_ms": float(dt),
            }
        )
        slot += 1
    return total_reward, spent_ms, search_debt_ms, executed, search_actions, rows


def window_underuse_penalty(spent_ms: float, budget_ms: float, env_cfg: dict) -> float:
    weight = float(env_cfg.get("window_underuse_penalty_weight", 0.0))
    if weight <= 0.0:
        return 0.0
    target_frac = float(env_cfg.get("window_underuse_target_frac", 0.0))
    target_frac = float(np.clip(target_frac, 0.0, 1.0))
    target_ms = target_frac * float(budget_ms)
    shortfall_ms = max(0.0, target_ms - float(spent_ms))
    return -weight * (shortfall_ms / max(1.0, float(budget_ms)))


def window_service_penalty(metrics: dict, env_cfg: dict) -> float:
    drop_pct_weight = float(env_cfg.get("window_drop_pct_penalty_weight", 0.0))
    drop_count_weight = float(env_cfg.get("window_drop_count_penalty_weight", 0.0))
    delay_weight = float(env_cfg.get("window_delay_penalty_weight", 0.0))
    if drop_pct_weight == 0.0 and drop_count_weight == 0.0 and delay_weight == 0.0:
        return 0.0
    drop_pct = float(metrics.get("drop_pct_active", 0.0))
    active = float(metrics.get("active_targets", 0.0))
    drop_count = drop_pct * active / 100.0
    delay_ms = float(metrics.get("mean_delay_active", 0.0))
    return -(
        drop_pct_weight * drop_pct
        + drop_count_weight * drop_count
        + delay_weight * (delay_ms / 1000.0)
    )


def run_plan_eval(planner, name: str, initial: int, seed: int, windows: int, env_cfg: dict):
    eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
    eng.reset(seed=int(seed))
    debt = 0.0
    cumulative = 0.0
    rows = []
    actions = []
    try:
        if hasattr(planner, "warmup"):
            planner.warmup(get_obs(eng, debt))
        for window in range(int(windows)):
            if bool(eng.term_buf[0]):
                break
            obs = get_obs(eng, debt)
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            t0 = time.perf_counter()
            debug_start = len(getattr(planner, "debug_candidate_rows", []))
            plan = planner.plan(obs, budget_ms=200.0)
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            plan_ms = 1000.0 * (time.perf_counter() - t0)
            if hasattr(planner, "debug_candidate_rows"):
                for dbg in planner.debug_candidate_rows[debug_start:]:
                    dbg.update(planner=name, seed=int(seed), initial=int(initial), window=int(window), rate=float(env_cfg.get("arrival_rate", np.nan)))
            reward, spent, debt, executed, searches, arows = execute_plan_until_budget_joint_shaped(
                eng, plan, 200.0, debt, name, int(seed), int(window), env_cfg
            )
            underuse_penalty = window_underuse_penalty(spent, 200.0, env_cfg)
            metrics = sample_state_metrics(eng, debt)
            service_penalty = window_service_penalty(metrics, env_cfg)
            reward += float(underuse_penalty)
            reward += float(service_penalty)
            cumulative += float(reward)
            rows.append(
                {
                    "planner": name,
                    "seed": int(seed),
                    "window": int(window),
                    "elapsed_ms": float((window + 1) * 200),
                    "window_reward": float(reward),
                    "cumulative_reward": float(cumulative),
                    "search_fraction": float(searches / max(1, executed)),
                    "planning_ms_per_decision": float(plan_ms),
                    "planning_ms_per_executed_action": float(plan_ms / max(1, executed)),
                    "executed_actions": int(executed),
                    "spent_ms": float(spent),
                    "window_utilization": float(spent / 200.0),
                    "underuse_penalty": float(underuse_penalty),
                    "service_penalty": float(service_penalty),
                    **metrics,
                }
            )
            for row in arows:
                row.update(window=int(window), elapsed_ms=float((window + 1) * 200))
            actions.extend(arows)
    finally:
        eng.close()
    return pd.DataFrame(rows), pd.DataFrame(actions)


def run_receding_eval(planner, name: str, initial: int, seed: int, windows: int, env_cfg: dict):
    eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
    eng.reset(seed=int(seed))
    debt = 0.0
    cumulative = 0.0
    rows = []
    actions = []
    try:
        if hasattr(planner, "warmup"):
            planner.warmup(get_obs(eng, debt))
        for window in range(int(windows)):
            if bool(eng.term_buf[0]):
                break
            spent = 0.0
            reward = 0.0
            executed = 0
            searches = 0
            plan_ms_total = 0.0
            arows_all = []
            while spent < 200.0 and not bool(eng.term_buf[0]):
                obs = get_obs(eng, debt)
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                t0 = time.perf_counter()
                debug_start = len(getattr(planner, "debug_candidate_rows", []))
                plan = planner.plan(obs, budget_ms=max(1.0, 200.0 - spent))
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                plan_ms_total += 1000.0 * (time.perf_counter() - t0)
                if hasattr(planner, "debug_candidate_rows"):
                    for dbg in planner.debug_candidate_rows[debug_start:]:
                        dbg.update(planner=name, seed=int(seed), initial=int(initial), window=int(window), rate=float(env_cfg.get("arrival_rate", np.nan)), spent_ms=float(spent))
                if not plan:
                    break
                r, dt, debt, ex, sea, arows = execute_plan_until_budget_joint_shaped(
                    eng, [int(plan[0])], 200.0 - spent, debt, name, int(seed), int(window), env_cfg
                )
                if ex <= 0 or dt <= 0.0:
                    break
                reward += float(r)
                spent += float(dt)
                executed += int(ex)
                searches += int(sea)
                arows_all.extend(arows)
            underuse_penalty = window_underuse_penalty(spent, 200.0, env_cfg)
            metrics = sample_state_metrics(eng, debt)
            service_penalty = window_service_penalty(metrics, env_cfg)
            reward += float(underuse_penalty)
            reward += float(service_penalty)
            cumulative += float(reward)
            rows.append(
                {
                    "planner": name,
                    "seed": int(seed),
                    "window": int(window),
                    "elapsed_ms": float((window + 1) * 200),
                    "window_reward": float(reward),
                    "cumulative_reward": float(cumulative),
                    "planning_ms_per_decision": float(plan_ms_total),
                    "executed_actions": int(executed),
                    "search_fraction": float(searches / max(1, executed)),
                    "spent_ms": float(spent),
                    "window_utilization": float(spent / 200.0),
                    "underuse_penalty": float(underuse_penalty),
                    "service_penalty": float(service_penalty),
                    **metrics,
                }
            )
            for row in arows_all:
                row.update(window=int(window), elapsed_ms=float((window + 1) * 200))
            actions.extend(arows_all)
    finally:
        eng.close()
    return pd.DataFrame(rows), pd.DataFrame(actions)


def run_chunked_receding_eval(
    planner,
    name: str,
    initial: int,
    seed: int,
    windows: int,
    env_cfg: dict,
    replan_stride: int,
    plan_budget_margin_ms: float = 0.0,
):
    """Replan after each executed action chunk while keeping the environment contract fixed."""
    stride = max(1, int(replan_stride))
    eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
    eng.reset(seed=int(seed))
    debt = 0.0
    cumulative = 0.0
    rows = []
    actions = []
    try:
        if hasattr(planner, "warmup"):
            planner.warmup(get_obs(eng, debt))
        for window in range(int(windows)):
            if bool(eng.term_buf[0]):
                break
            spent = 0.0
            reward = 0.0
            executed = 0
            searches = 0
            replans = 0
            plan_ms_total = 0.0
            arows_all = []
            selected_rows: set[int] = set()
            search_count = 0
            track_count = 0
            last_row = -1
            while spent < 200.0 and not bool(eng.term_buf[0]):
                obs = get_obs(eng, debt)
                remaining_ms = 200.0 - spent
                active = np.asarray(obs.get("active", ()), dtype=bool).reshape(-1)
                dwell = np.asarray(obs.get("t_dwell", ()), dtype=np.float32).reshape(-1)
                min_feasible_ms = 10.0  # Search dwell.
                for target_idx in np.flatnonzero(active[: len(dwell)]):
                    if int(target_idx) + 1 in selected_rows:
                        continue
                    min_feasible_ms = min(min_feasible_ms, max(1.0, float(dwell[target_idx])))
                if remaining_ms + 1e-6 < min_feasible_ms:
                    break
                if hasattr(planner, "set_receding_context"):
                    planner.set_receding_context(
                        selected_rows,
                        spent,
                        search_count,
                        track_count,
                        last_row,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                debug_start = len(getattr(planner, "debug_candidate_rows", []))
                plan = planner.plan(
                    obs,
                    budget_ms=max(1.0, remaining_ms + float(plan_budget_margin_ms)),
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                plan_ms_total += 1000.0 * (time.perf_counter() - t0)
                replans += 1
                if hasattr(planner, "debug_candidate_rows"):
                    for dbg in planner.debug_candidate_rows[debug_start:]:
                        dbg.update(
                            planner=name,
                            seed=int(seed),
                            initial=int(initial),
                            window=int(window),
                            rate=float(env_cfg.get("arrival_rate", np.nan)),
                            spent_ms=float(spent),
                            replan_stride=int(stride),
                        )
                if not plan:
                    break
                chunk = [int(a) for a in plan[:stride]]
                r, dt, debt, ex, sea, arows = execute_plan_until_budget_joint_shaped(
                    eng, chunk, 200.0 - spent, debt, name, int(seed), int(window), env_cfg
                )
                if ex <= 0 or dt <= 0.0:
                    break
                reward += float(r)
                spent += float(dt)
                executed += int(ex)
                searches += int(sea)
                arows_all.extend(arows)
                for action_row in arows:
                    action = int(action_row.get("s_action", action_row.get("action", -1)))
                    if action < 0:
                        continue
                    base_row, sensor = xs_decode_action(action, MAXT)
                    if int(sensor or 0) != 0:
                        continue
                    last_row = int(base_row)
                    if int(base_row) <= 0:
                        search_count += 1
                    else:
                        track_count += 1
                        selected_rows.add(int(base_row))
            underuse_penalty = window_underuse_penalty(spent, 200.0, env_cfg)
            metrics = sample_state_metrics(eng, debt)
            service_penalty = window_service_penalty(metrics, env_cfg)
            reward += float(underuse_penalty) + float(service_penalty)
            cumulative += float(reward)
            rows.append(
                {
                    "planner": name,
                    "seed": int(seed),
                    "window": int(window),
                    "elapsed_ms": float((window + 1) * 200),
                    "window_reward": float(reward),
                    "cumulative_reward": float(cumulative),
                    "planning_ms_per_decision": float(plan_ms_total),
                    "replans_per_window": int(replans),
                    "replan_stride": int(stride),
                    "executed_actions": int(executed),
                    "search_fraction": float(searches / max(1, executed)),
                    "spent_ms": float(spent),
                    "window_utilization": float(spent / 200.0),
                    "underuse_penalty": float(underuse_penalty),
                    "service_penalty": float(service_penalty),
                    **metrics,
                }
            )
            for row in arows_all:
                row.update(
                    window=int(window),
                    elapsed_ms=float((window + 1) * 200),
                    replan_stride=int(stride),
                )
            actions.extend(arows_all)
    finally:
        eng.close()
    return pd.DataFrame(rows), pd.DataFrame(actions)


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    last = df.sort_values("window").iloc[-1]
    return {
        "reward_per_window": float(df["window_reward"].mean()),
        "final_cumulative": float(last["cumulative_reward"]),
        "drop_pct_active": float(df["drop_pct_active"].mean()),
        "tracked_targets": float(df["tracked_targets"].mean()),
        "mean_delay_active": float(df["mean_delay_active"].mean()),
        "search_fraction": float(df["search_fraction"].mean()),
        "planning_ms_per_window": float(df["planning_ms_per_decision"].mean()),
        "executed_actions": float(df["executed_actions"].mean()),
        "spent_ms": float(df["spent_ms"].mean()) if "spent_ms" in df else float("nan"),
        "window_utilization": float(df["window_utilization"].mean()) if "window_utilization" in df else float("nan"),
        "underuse_penalty": float(df["underuse_penalty"].mean()) if "underuse_penalty" in df else 0.0,
        "service_penalty": float(df["service_penalty"].mean()) if "service_penalty" in df else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-state", default=str(ROOT / "CreateValid1" / "results" / "mixed_gate_distill_180_action_attention_step40_state.pt"))
    ap.add_argument("--g-state", default=str(ROOT / "CreateValid1" / "results" / "action_attention_muzero_g_smoke.pt"))
    ap.add_argument("--g-alt-state", default="")
    ap.add_argument("--g-blend-alpha", type=float, default=1.0)
    ap.add_argument("--lean-base-load", action="store_true")
    ap.add_argument("--dual-plan-select", action="store_true")
    ap.add_argument("--dual-plan-track-weight", type=float, default=1.0)
    ap.add_argument("--dual-plan-search-weight", type=float, default=0.25)
    ap.add_argument("--dual-plan-duration-weight", type=float, default=0.02)
    ap.add_argument("--dual-plan-pressure-weight", type=float, default=0.0)
    ap.add_argument("--g-d-model", type=int, default=48)
    ap.add_argument("--compile-g", action="store_true", help="Compile the latent dynamics/sequence model with torch.compile reduce-overhead mode.")
    ap.add_argument("--variant", default="two_row_action_attention")
    ap.add_argument("--out", default=str(ROOT / "CreateValid1" / "results" / "action_attention_muzero_eval.csv"))
    ap.add_argument("--initials", default="60")
    ap.add_argument("--rates", default="4")
    ap.add_argument("--seeds", default="916")
    ap.add_argument("--windows", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--torch-interop-threads", type=int, default=1)
    ap.add_argument(
        "--search-bias",
        type=float,
        default=0.0,
        help="Optional diagnostic logit offset. Canonical learned evaluation uses zero.",
    )
    ap.add_argument("--adaptive-search-bias", action="store_true")
    ap.add_argument("--adaptive-search-bias-invert", action="store_true")
    ap.add_argument("--search-bias-active-threshold", type=int, default=0)
    ap.add_argument("--search-bias-overdue-threshold", type=int, default=0)
    ap.add_argument("--search-bias-pressure-threshold", type=float, default=0.0)
    ap.add_argument("--search-bias-low", type=float, default=-16.0)
    ap.add_argument("--search-bias-high", type=float, default=-10.0)
    ap.add_argument("--service-track-weight", type=float, default=0.0)
    ap.add_argument("--service-search-weight", type=float, default=0.0)
    ap.add_argument("--service-active-goal", type=float, default=0.0)
    ap.add_argument("--decode-router-active-threshold", type=int, default=0)
    ap.add_argument("--decode-router-low-service-track", type=float, default=0.0)
    ap.add_argument("--decode-router-low-service-search", type=float, default=0.0)
    ap.add_argument("--decode-router-low-search-cap", type=float, default=0.0)
    ap.add_argument("--decode-router-high-active-threshold", type=int, default=0)
    ap.add_argument("--decode-router-high-service-track", type=float, default=0.0)
    ap.add_argument("--decode-router-high-service-search", type=float, default=0.0)
    ap.add_argument("--decode-router-high-search-cap", type=float, default=0.0)
    ap.add_argument("--decode-router-smooth-temp", type=float, default=0.0)
    ap.add_argument("--use-learned-service-gate", action="store_true")
    ap.add_argument("--service-target-override", action="store_true")
    ap.add_argument("--avoid-double-search", action="store_true")
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--q-weight", type=float, default=1.0)
    ap.add_argument("--per-sensor-top", type=int, default=3)
    ap.add_argument("--lookahead-width", type=int, default=0)
    ap.add_argument("--lookahead-leaf-weight", type=float, default=0.25)
    ap.add_argument("--latent-candidate-topk", type=int, default=16, help="Latent planner candidate rows per sensor. Use <=0 to score all rows.")
    ap.add_argument("--latent-leaf-topk", type=int, default=8, help="Latent planner leaf rows per sensor. Use <=0 to score all rows.")
    ap.add_argument("--service-critic-weight", type=float, default=0.0)
    ap.add_argument("--service-critic-active-weight", type=float, default=0.25)
    ap.add_argument("--service-critic-tracked-weight", type=float, default=1.0)
    ap.add_argument("--service-critic-drop-weight", type=float, default=1.5)
    ap.add_argument("--service-critic-delay-weight", type=float, default=0.3)
    ap.add_argument("--direct-action-service-weight", type=float, default=0.0)
    ap.add_argument("--direct-action-value-weight", type=float, default=0.0)
    ap.add_argument("--direct-action-frame-weight", type=float, default=0.0)
    ap.add_argument("--direct-action-value-max-steps", type=int, default=0)
    ap.add_argument("--direct-action-value-margin-threshold", type=float, default=0.0)
    ap.add_argument("--direct-action-value-cache-topn", type=int, default=0)
    ap.add_argument("--direct-action-value-cache-only", action="store_true")
    ap.add_argument("--direct-action-value-track-only", action="store_true", help="For S-only factorized root-seq decode, preserve the policy search/track choice and use direct action value only to rerank track targets.")
    ap.add_argument("--duration-penalty-weight", type=float, default=0.0)
    ap.add_argument("--planner-duration-scale", type=float, default=1.0)
    ap.add_argument("--sensor-valid-mask", action="store_true")
    ap.add_argument("--fill-fallback", choices=["none", "edf", "est", "short_edf", "rootseq_tail", "urgent_short", "search", "pure_search"], default="none")
    ap.add_argument("--fill-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=96)
    ap.add_argument("--cuda-graph-step", action="store_true")
    ap.add_argument("--cuda-graph-root-encode", action="store_true")
    ap.add_argument("--cuda-graph-root-seq", action="store_true")
    ap.add_argument("--cuda-graph-ar-seq", action="store_true")
    ap.add_argument("--cuda-graph-ar-dynamic", action="store_true")
    ap.add_argument("--ar-compact-max-rows", type=int, default=0, help="For clean S-only AR dynamic graph, compact active target tokens to this fixed row count. 0 disables.")
    ap.add_argument("--cuda-graph-tensor-loop", action="store_true")
    ap.add_argument("--cuda-graph-s-only-score", action="store_true")
    ap.add_argument("--tensor-loop", action="store_true")
    ap.add_argument(
        "--tensor-loop-factorized-decode",
        action="store_true",
        help="Choose S-only search/track from aggregate type mass, then choose the track target conditionally.",
    )
    ap.add_argument("--single-sensor-noop-action", action="store_true", help="Use a disabled-sensor no-op action in latent dynamics for S-only tensor-loop evaluation.")
    ap.add_argument("--single-sensor-s-only-choose", action="store_true", help="In single-sensor latent decoding, choose only from the S stream and set X to search/no-op instead of joint S/X pairing.")
    ap.add_argument("--use-g-policy", action="store_true")
    ap.add_argument("--disable-policy-action-coupler", action="store_true")
    ap.add_argument("--policy-action-mixer", choices=["full", "light", "tiny", "none"], default="full")
    ap.add_argument("--use-root-seq-policy", action="store_true")
    ap.add_argument("--use-base-seq-policy", action="store_true")
    ap.add_argument("--use-ar-seq-policy", action="store_true")
    ap.add_argument("--ar-policy-sampling-temperature", type=float, default=0.0, help="Sample AR type/target actions from learned logits; zero uses greedy argmax.")
    ap.add_argument("--ar-target-sampling-temperature", type=float, default=-1.0, help="Target-only AR sampling temperature; -1 reuses --ar-policy-sampling-temperature and 0 keeps target selection greedy.")
    ap.add_argument("--use-cached-base-policy", action="store_true")
    ap.add_argument("--root-seq-edf-targets", action="store_true")
    ap.add_argument("--root-seq-base-targets", action="store_true")
    ap.add_argument("--root-seq-slot-passes", type=int, default=1)
    ap.add_argument("--root-seq-step-context", action="store_true")
    ap.add_argument("--root-seq-select-steps", default="")
    ap.add_argument("--root-seq-recurrent-rescore", action="store_true")
    ap.add_argument("--root-seq-rescore-weight", type=float, default=1.0)
    ap.add_argument("--root-seq-recurrent-g-only", action="store_true")
    ap.add_argument("--root-seq-rescore-stride", type=int, default=1)
    ap.add_argument("--root-seq-search-bias-candidates", default="")
    ap.add_argument("--root-seq-terminal-service-rerank-weight", type=float, default=0.0)
    ap.add_argument("--root-seq-terminal-proxy-weight", type=float, default=1.0)
    ap.add_argument("--root-seq-terminal-min-plan-frac", type=float, default=0.0)
    ap.add_argument("--root-seq-terminal-min-search-atoms", type=int, default=0)
    ap.add_argument("--root-seq-debug-candidates", action="store_true")
    ap.add_argument("--root-seq-decode-topk", type=int, default=16)
    ap.add_argument("--root-seq-cpu-decode", action="store_true", help="Decode root-sequence scores on CPU after one small transfer to avoid many tiny GPU topk/argmax launches.")
    ap.add_argument("--root-seq-factorized-decode", action="store_true", help="Decode S-only root-sequence policy by first choosing search/track type, matching the factorized training loss.")
    ap.add_argument("--ar-dynamic-slots", action="store_true", help="For AR sequence decoding, rebuild slot/context features from the decoded partial plan at each step.")
    ap.add_argument("--ar-history-k", type=int, default=0)
    ap.add_argument("--decoder-history-search-streak-bias", type=float, default=0.0, help="Online decoder-only bias added to search logits for each consecutive prior all-search decode step.")
    ap.add_argument("--decoder-history-track-after-search-bias", type=float, default=0.0, help="Online decoder-only bias added to track logits for each consecutive prior all-search decode step.")
    ap.add_argument("--decoder-history-track-streak-bias", type=float, default=0.0, help="Online decoder-only bias added to track logits for each consecutive prior all-track decode step.")
    ap.add_argument("--decoder-history-search-after-track-bias", type=float, default=0.0, help="Online decoder-only bias added to search logits for each consecutive prior all-track decode step.")
    ap.add_argument("--max-window-search-frac", type=float, default=0.0)
    ap.add_argument("--max-window-search-atoms", type=int, default=0)
    ap.add_argument("--min-window-search-atoms", type=int, default=0)
    ap.add_argument("--min-window-search-active-threshold", type=int, default=0)
    ap.add_argument("--repair-search-target-frac", type=float, default=0.0)
    ap.add_argument("--root-seq-search-balance-weight", type=float, default=0.0)
    ap.add_argument("--root-seq-max-search-skew", type=int, default=0)
    ap.add_argument("--root-seq-force-search-prefix", type=int, default=0)
    ap.add_argument("--pressure-repair-threshold", type=float, default=0.0)
    ap.add_argument("--pressure-repair-max-atoms", type=int, default=0)
    ap.add_argument("--frontload-track-actions", action="store_true")
    ap.add_argument("--service-sort-plan", action="store_true")
    ap.add_argument("--service-sort-search-prefix", type=int, default=0)
    ap.add_argument("--service-sort-track-burst", type=int, default=0)
    ap.add_argument("--repair-pure-search-joints", action="store_true")
    ap.add_argument("--root-seq-stop-threshold", type=float, default=-1.0e30)
    ap.add_argument("--use-root-seq-stop-head", action="store_true")
    ap.add_argument("--root-seq-stop-prob", type=float, default=0.5)
    ap.add_argument("--root-seq-min-steps", type=int, default=0)
    ap.add_argument("--env-mode", default="current")
    ap.add_argument("--track-update-reward", type=float, default=0.30)
    ap.add_argument("--track-loss-penalty", type=float, default=8.0)
    ap.add_argument("--sector-staleness-weight", type=float, default=0.0)
    ap.add_argument("--search-frame-overdue-weight", type=float, default=0.20)
    ap.add_argument("--search-frame-desired-ms", type=float, default=3000.0)
    ap.add_argument("--search-frame-deadline-ms", type=float, default=4500.0)
    ap.add_argument("--search-frame-drop-penalty", type=float, default=0.0)
    ap.add_argument("--zero-action-rewards", action="store_true")
    ap.add_argument("--tracked-count-delta-reward-weight", type=float, default=0.0)
    ap.add_argument("--target-service-weight", type=float, default=0.0)
    ap.add_argument("--target-service-horizon-ms", type=float, default=3000.0)
    ap.add_argument("--tracked-target-ms-reward-weight", type=float, default=0.0)
    ap.add_argument("--on-time-target-ms-reward-weight", type=float, default=0.0)
    ap.add_argument("--service-pressure-delta-reward-weight", type=float, default=0.0)
    ap.add_argument("--service-pressure-state-penalty-weight", type=float, default=0.0)
    ap.add_argument("--search-frame-state-penalty-weight", type=float, default=0.0)
    ap.add_argument("--search-frame-delta-reward-weight", type=float, default=0.0)
    ap.add_argument("--track-pressure-reward-weight", type=float, default=0.0)
    ap.add_argument("--bounded-service-reward-weight", type=float, default=0.0)
    ap.add_argument("--serviced-target-reward-weight", type=float, default=0.0)
    ap.add_argument("--serviced-target-update-reward-weight", type=float, default=0.0)
    ap.add_argument("--serviced-pressure-delta-reward-weight", type=float, default=0.0)
    ap.add_argument("--serviced-pressure-improvement-reward-weight", type=float, default=0.0)
    ap.add_argument("--discovered-target-reward", type=float, default=0.0)
    ap.add_argument("--window-underuse-penalty-weight", type=float, default=0.0)
    ap.add_argument("--window-underuse-target-frac", type=float, default=0.0)
    ap.add_argument("--window-drop-pct-penalty-weight", type=float, default=0.0)
    ap.add_argument("--window-drop-count-penalty-weight", type=float, default=0.0)
    ap.add_argument("--window-delay-penalty-weight", type=float, default=0.0)
    ap.add_argument("--router-active-threshold", type=int, default=0)
    ap.add_argument("--router-alt-search-bias", type=float, default=float("nan"))
    ap.add_argument("--router-alt-service-track-weight", type=float, default=float("nan"))
    ap.add_argument("--router-alt-service-search-weight", type=float, default=float("nan"))
    ap.add_argument("--router-alt-max-window-search-frac", type=float, default=float("nan"))
    ap.add_argument("--router-alt-service-sort-search-prefix", type=int, default=-1)
    ap.add_argument("--router-alt-pressure-repair-max-atoms", type=int, default=-1)
    ap.add_argument("--hybrid-full-gate", action="store_true")
    ap.add_argument("--hybrid-active-threshold", type=int, default=0)
    ap.add_argument("--hybrid-overdue-threshold", type=int, default=0)
    ap.add_argument("--hybrid-pressure-threshold", type=float, default=0.0)
    ap.add_argument("--hybrid-max-full-fraction", type=float, default=1.0)
    ap.add_argument("--include-baselines", action="store_true")
    ap.add_argument("--skip-primary-planner", action="store_true")
    ap.add_argument("--baseline-planners", default="ActionAttention_full,EDF,EST")
    ap.add_argument("--baseline-include-search-candidate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--single-sensor", action="store_true", help="Disable X-band and evaluate S-band-only scheduling.")
    ap.add_argument("--receding-window", action="store_true")
    ap.add_argument(
        "--replan-stride",
        type=int,
        default=1,
        help="With --receding-window, execute this many proposed actions before replanning.",
    )
    ap.add_argument(
        "--replan-plan-budget-margin-ms",
        type=float,
        default=10.0,
        help="Ask the planner for a small overflow proposal buffer while the simulator still enforces the 200 ms execution boundary.",
    )
    args = ap.parse_args()

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    if int(args.torch_interop_threads) > 0:
        try:
            torch.set_num_interop_threads(int(args.torch_interop_threads))
        except RuntimeError:
            pass
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = not bool(args.single_sensor)
    exact_args.single_sensor = bool(args.single_sensor)
    ckpt = torch.load(str(args.g_state), map_location=args.device)
    g_state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    g_seq_len = int(g_state["seq_pos"].shape[0]) if isinstance(g_state, dict) and "seq_pos" in g_state else 40
    model = load_base_policy_model(args, args.device).to(args.device).eval()
    g = LatentG(d_model=int(args.g_d_model), seq_len=g_seq_len, ar_history_k=int(args.ar_history_k)).to(args.device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    for param in g.parameters():
        param.requires_grad_(False)
    missing, unexpected = g.load_state_dict(g_state, strict=False)
    if isinstance(ckpt, dict):
        g.action_value_target_mean = float(ckpt.get("action_value_target_mean", 0.0))
        g.action_value_target_std = max(1.0e-3, float(ckpt.get("action_value_target_std", 1.0)))
    if missing or unexpected:
        print({"g_load_missing": missing, "g_load_unexpected": unexpected}, flush=True)
    g.policy_action_mixer = "none" if bool(args.disable_policy_action_coupler) else str(args.policy_action_mixer)
    if any(str(k).startswith("policy_action_") for k in missing) and hasattr(g, "policy_action_residual"):
        with torch.no_grad():
            g.policy_action_residual.weight.zero_()
            g.policy_action_residual.bias.zero_()
        print({"disabled_untrained_policy_action_residual": True}, flush=True)
    if bool(args.disable_policy_action_coupler) and hasattr(g, "policy_action_coupler") and hasattr(g, "policy_action_residual"):
        g.policy_action_coupler = _PassthroughCoupler().to(args.device).eval()
        with torch.no_grad():
            g.policy_action_residual.weight.zero_()
            g.policy_action_residual.bias.zero_()
        print({"disabled_policy_action_coupler": True}, flush=True)
    if bool(args.compile_g):
        try:
            g = torch.compile(g, mode="reduce-overhead")
            print({"compiled_g": True, "mode": "reduce-overhead"}, flush=True)
        except Exception as exc:
            print({"compiled_g": False, "error": str(exc)}, flush=True)
    g_alt = None
    if str(args.g_alt_state):
        ckpt_alt = torch.load(str(args.g_alt_state), map_location=args.device)
        g_alt_state = ckpt_alt["state_dict"] if isinstance(ckpt_alt, dict) and "state_dict" in ckpt_alt else ckpt_alt
        g_alt_seq_len = int(g_alt_state["seq_pos"].shape[0]) if isinstance(g_alt_state, dict) and "seq_pos" in g_alt_state else g_seq_len
        g_alt = LatentG(d_model=int(args.g_d_model), seq_len=g_alt_seq_len, ar_history_k=int(args.ar_history_k)).to(args.device).eval()
        for param in g_alt.parameters():
            param.requires_grad_(False)
        missing_alt, unexpected_alt = g_alt.load_state_dict(g_alt_state, strict=False)
        if missing_alt or unexpected_alt:
            print({"g_alt_load_missing": missing_alt, "g_alt_load_unexpected": unexpected_alt}, flush=True)

    all_windows = []
    all_actions = []
    all_candidate_debug = []
    summaries = []
    for initial in parse_ints(args.initials):
        for rate in parse_floats(args.rates):
            env_cfg = env_cfg_for(float(rate), exact_args)
            env_cfg["enable_x_band"] = 0 if bool(args.single_sensor) else 1
            env_cfg["window_underuse_penalty_weight"] = float(args.window_underuse_penalty_weight)
            env_cfg["window_underuse_target_frac"] = float(args.window_underuse_target_frac)
            env_cfg["window_drop_pct_penalty_weight"] = float(args.window_drop_pct_penalty_weight)
            env_cfg["window_drop_count_penalty_weight"] = float(args.window_drop_count_penalty_weight)
            env_cfg["window_delay_penalty_weight"] = float(args.window_delay_penalty_weight)
            for seed in parse_ints(args.seeds):
                # Stochastic policy decoding must be reproducible and must not
                # depend on the order of preceding cells in a sweep.
                np.random.seed(int(seed))
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
                muzero = LatentMuZeroPlanner(
                    model,
                    g,
                    env_cfg,
                    policy_weight=float(args.policy_weight),
                    q_weight=float(args.q_weight),
                    search_score_bias=float(args.search_bias),
                    adaptive_search_bias=bool(args.adaptive_search_bias),
                    adaptive_search_bias_invert=bool(args.adaptive_search_bias_invert),
                    search_bias_active_threshold=int(args.search_bias_active_threshold),
                    search_bias_overdue_threshold=int(args.search_bias_overdue_threshold),
                    search_bias_pressure_threshold=float(args.search_bias_pressure_threshold),
                    search_bias_low=float(args.search_bias_low),
                    search_bias_high=float(args.search_bias_high),
                    service_track_weight=float(args.service_track_weight),
                    service_search_weight=float(args.service_search_weight),
                    service_active_goal=float(args.service_active_goal),
                    decode_router_active_threshold=int(args.decode_router_active_threshold),
                    decode_router_low_service_track=float(args.decode_router_low_service_track),
                    decode_router_low_service_search=float(args.decode_router_low_service_search),
                    decode_router_low_search_cap=float(args.decode_router_low_search_cap),
                    decode_router_high_active_threshold=int(args.decode_router_high_active_threshold),
                    decode_router_high_service_track=float(args.decode_router_high_service_track),
                    decode_router_high_service_search=float(args.decode_router_high_service_search),
                    decode_router_high_search_cap=float(args.decode_router_high_search_cap),
                    decode_router_smooth_temp=float(args.decode_router_smooth_temp),
                    use_learned_service_gate=bool(args.use_learned_service_gate),
                    service_target_override=bool(args.service_target_override),
                    avoid_double_search=bool(args.avoid_double_search),
                    max_steps=int(args.max_steps),
                    lookahead_width=int(args.lookahead_width),
                    lookahead_leaf_weight=float(args.lookahead_leaf_weight),
                    latent_candidate_topk=int(args.latent_candidate_topk),
                    latent_leaf_topk=int(args.latent_leaf_topk),
                    service_critic_weight=float(args.service_critic_weight),
                    service_critic_active_weight=float(args.service_critic_active_weight),
                    service_critic_tracked_weight=float(args.service_critic_tracked_weight),
                    service_critic_drop_weight=float(args.service_critic_drop_weight),
                    service_critic_delay_weight=float(args.service_critic_delay_weight),
                    direct_action_service_weight=float(args.direct_action_service_weight),
                    direct_action_value_weight=float(args.direct_action_value_weight),
                    direct_action_frame_weight=float(args.direct_action_frame_weight),
                    direct_action_value_max_steps=int(args.direct_action_value_max_steps),
                    direct_action_value_margin_threshold=float(args.direct_action_value_margin_threshold),
                    direct_action_value_cache_topn=int(args.direct_action_value_cache_topn),
                    direct_action_value_cache_only=bool(args.direct_action_value_cache_only),
                    direct_action_value_track_only=bool(args.direct_action_value_track_only),
                    duration_penalty_weight=float(args.duration_penalty_weight),
                    planner_duration_scale=float(args.planner_duration_scale),
                    sensor_valid_mask=bool(args.sensor_valid_mask),
                    fill_fallback=str(args.fill_fallback),
                    fill_steps=int(args.fill_steps),
                    cuda_graph_step=bool(args.cuda_graph_step),
                    cuda_graph_root_encode=bool(args.cuda_graph_root_encode),
                    cuda_graph_root_seq=bool(args.cuda_graph_root_seq),
                    cuda_graph_ar_seq=bool(args.cuda_graph_ar_seq),
                    cuda_graph_ar_dynamic=bool(args.cuda_graph_ar_dynamic),
                    ar_compact_max_rows=int(args.ar_compact_max_rows),
                    cuda_graph_tensor_loop=bool(args.cuda_graph_tensor_loop),
                    cuda_graph_s_only_score=bool(args.cuda_graph_s_only_score),
                    tensor_loop=bool(args.tensor_loop),
                    tensor_loop_factorized_decode=bool(args.tensor_loop_factorized_decode),
                    single_sensor_noop_action=bool(args.single_sensor_noop_action),
                    single_sensor_s_only_choose=bool(args.single_sensor_s_only_choose),
                    use_g_policy=bool(args.use_g_policy),
                    use_root_seq_policy=bool(args.use_root_seq_policy),
                    use_base_seq_policy=bool(args.use_base_seq_policy),
                    use_ar_seq_policy=bool(args.use_ar_seq_policy),
                    ar_policy_sampling_temperature=float(args.ar_policy_sampling_temperature),
                    ar_target_sampling_temperature=(None if float(args.ar_target_sampling_temperature) < 0.0 else float(args.ar_target_sampling_temperature)),
                    use_cached_base_policy=bool(args.use_cached_base_policy),
                    root_seq_edf_targets=bool(args.root_seq_edf_targets),
                    root_seq_base_targets=bool(args.root_seq_base_targets),
                    root_seq_slot_passes=int(args.root_seq_slot_passes),
                    root_seq_step_context=bool(args.root_seq_step_context),
                    root_seq_select_steps=str(args.root_seq_select_steps),
                    root_seq_recurrent_rescore=bool(args.root_seq_recurrent_rescore),
                    root_seq_rescore_weight=float(args.root_seq_rescore_weight),
                    root_seq_recurrent_g_only=bool(args.root_seq_recurrent_g_only),
                    root_seq_rescore_stride=int(args.root_seq_rescore_stride),
                    root_seq_search_bias_candidates=str(args.root_seq_search_bias_candidates),
                    root_seq_terminal_service_rerank_weight=float(args.root_seq_terminal_service_rerank_weight),
                    root_seq_terminal_proxy_weight=float(args.root_seq_terminal_proxy_weight),
                    root_seq_terminal_min_plan_frac=float(args.root_seq_terminal_min_plan_frac),
                    root_seq_terminal_min_search_atoms=int(args.root_seq_terminal_min_search_atoms),
                    root_seq_debug_candidates=bool(args.root_seq_debug_candidates),
                    root_seq_decode_topk=int(args.root_seq_decode_topk),
                    root_seq_cpu_decode=bool(args.root_seq_cpu_decode),
                    root_seq_factorized_decode=bool(args.root_seq_factorized_decode),
                    ar_dynamic_slots=bool(args.ar_dynamic_slots),
                    decoder_history_search_streak_bias=float(args.decoder_history_search_streak_bias),
                    decoder_history_track_after_search_bias=float(args.decoder_history_track_after_search_bias),
                    decoder_history_track_streak_bias=float(args.decoder_history_track_streak_bias),
                    decoder_history_search_after_track_bias=float(args.decoder_history_search_after_track_bias),
                    max_window_search_frac=float(args.max_window_search_frac),
                    max_window_search_atoms=int(args.max_window_search_atoms),
                    min_window_search_atoms=int(args.min_window_search_atoms),
                    min_window_search_active_threshold=int(args.min_window_search_active_threshold),
                    repair_search_target_frac=float(args.repair_search_target_frac),
                    root_seq_search_balance_weight=float(args.root_seq_search_balance_weight),
                    root_seq_max_search_skew=int(args.root_seq_max_search_skew),
                    root_seq_force_search_prefix=int(args.root_seq_force_search_prefix),
                    pressure_repair_threshold=float(args.pressure_repair_threshold),
                    pressure_repair_max_atoms=int(args.pressure_repair_max_atoms),
                    frontload_track_actions=bool(args.frontload_track_actions),
                    service_sort_plan=bool(args.service_sort_plan),
                    service_sort_search_prefix=int(args.service_sort_search_prefix),
                    service_sort_track_burst=int(args.service_sort_track_burst),
                    repair_pure_search_joints=bool(args.repair_pure_search_joints),
                    root_seq_stop_threshold=float(args.root_seq_stop_threshold),
                    use_root_seq_stop_head=bool(args.use_root_seq_stop_head),
                    root_seq_stop_prob=float(args.root_seq_stop_prob),
                    root_seq_min_steps=int(args.root_seq_min_steps),
                    g_alt=g_alt,
                    g_blend_alpha=float(args.g_blend_alpha),
                    dual_plan_select=bool(args.dual_plan_select),
                    dual_plan_track_weight=float(args.dual_plan_track_weight),
                    dual_plan_search_weight=float(args.dual_plan_search_weight),
                    dual_plan_duration_weight=float(args.dual_plan_duration_weight),
                    dual_plan_pressure_weight=float(args.dual_plan_pressure_weight),
                    router_active_threshold=int(args.router_active_threshold),
                    router_alt_search_bias=None if np.isnan(float(args.router_alt_search_bias)) else float(args.router_alt_search_bias),
                    router_alt_service_track_weight=None if np.isnan(float(args.router_alt_service_track_weight)) else float(args.router_alt_service_track_weight),
                    router_alt_service_search_weight=None if np.isnan(float(args.router_alt_service_search_weight)) else float(args.router_alt_service_search_weight),
                    router_alt_max_window_search_frac=None if np.isnan(float(args.router_alt_max_window_search_frac)) else float(args.router_alt_max_window_search_frac),
                    router_alt_service_sort_search_prefix=None if int(args.router_alt_service_sort_search_prefix) < 0 else int(args.router_alt_service_sort_search_prefix),
                    router_alt_pressure_repair_max_atoms=None if int(args.router_alt_pressure_repair_max_atoms) < 0 else int(args.router_alt_pressure_repair_max_atoms),
                    device=str(args.device),
                )
                planner_name = "LatentMuZero_greedy"
                eval_planner = muzero
                if bool(args.hybrid_full_gate):
                    base = PhysicalHeadPlanner(
                        model,
                        str(args.variant),
                        env_cfg,
                        policy_weight=float(args.policy_weight),
                        q_weight=float(args.q_weight),
                        search_score_bias=float(args.search_bias),
                    )
                    full = WorkConservingAsyncCoupledPlanner(base, per_sensor_top=int(args.per_sensor_top), include_search_candidate=True)
                    eval_planner = HybridRiskPlanner(
                        muzero,
                        full,
                        active_threshold=int(args.hybrid_active_threshold),
                        overdue_threshold=int(args.hybrid_overdue_threshold),
                        pressure_threshold=float(args.hybrid_pressure_threshold),
                        max_full_fraction=float(args.hybrid_max_full_fraction),
                    )
                    planner_name = "HybridMuZero_fullgate"
                if bool(args.receding_window):
                    run_fn = lambda p, n, i, s, w, c: run_chunked_receding_eval(
                        p,
                        n,
                        i,
                        s,
                        w,
                        c,
                        int(args.replan_stride),
                        float(args.replan_plan_budget_margin_ms),
                    )
                else:
                    run_fn = run_plan_eval
                if not bool(args.skip_primary_planner):
                    df, _actions = run_fn(eval_planner, planner_name, int(initial), int(seed), int(args.windows), env_cfg)
                    df = df.copy()
                    df["initial"] = int(initial)
                    df["rate"] = float(rate)
                    df["seed"] = int(seed)
                    all_windows.append(df)
                    if not _actions.empty:
                        _actions = _actions.copy()
                        _actions["planner"] = planner_name
                        _actions["initial"] = int(initial)
                        _actions["rate"] = float(rate)
                        all_actions.append(_actions)
                    row = {"planner": planner_name, "initial": int(initial), "rate": float(rate), "seed": int(seed), **summarize(df)}
                    summaries.append(row)
                    if getattr(muzero, "debug_candidate_rows", None):
                        for dbg in muzero.debug_candidate_rows:
                            dbg.setdefault("initial", int(initial))
                            dbg.setdefault("rate", float(rate))
                            dbg.setdefault("seed", int(seed))
                        all_candidate_debug.extend(muzero.debug_candidate_rows)
                    print(row, flush=True)
                if bool(args.include_baselines):
                    base = PhysicalHeadPlanner(
                        model,
                        str(args.variant),
                        env_cfg,
                        policy_weight=float(args.policy_weight),
                        q_weight=float(args.q_weight),
                        search_score_bias=float(args.search_bias),
                    )
                    wc = DirectPlanAdapter(
                        WorkConservingAsyncCoupledPlanner(
                            base,
                            per_sensor_top=int(args.per_sensor_top),
                            include_search_candidate=bool(args.baseline_include_search_candidate),
                        )
                    )
                    requested_baselines = {
                        x.strip()
                        for x in str(args.baseline_planners).replace(";", ",").split(",")
                        if x.strip()
                    }
                    for name, planner in {
                        "ActionAttention_full": wc,
                        "EDF": DirectPlanAdapter(EDFPlanner(MAXT)),
                        "EST": DirectPlanAdapter(ESTPlanner(MAXT)),
                    }.items():
                        if requested_baselines and name not in requested_baselines:
                            continue
                        bdf, b_actions = run_exact_rescore_grid_joint(planner, name, int(initial), int(seed), int(args.windows), env_cfg)
                        bdf = bdf.copy()
                        bdf["initial"] = int(initial)
                        bdf["rate"] = float(rate)
                        bdf["seed"] = int(seed)
                        all_windows.append(bdf)
                        if not b_actions.empty:
                            b_actions = b_actions.copy()
                            b_actions["planner"] = name
                            b_actions["initial"] = int(initial)
                            b_actions["rate"] = float(rate)
                            all_actions.append(b_actions)
                        brow = {"planner": name, "initial": int(initial), "rate": float(rate), "seed": int(seed), **summarize(bdf)}
                        summaries.append(brow)
                        print(brow, flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out.with_name(out.stem + "_summary.csv")
    actions_path = out.with_name(out.stem + "_actions.csv")
    candidate_debug_path = out.with_name(out.stem + "_candidate_debug.csv")
    pd.concat(all_windows, ignore_index=True).to_csv(out, index=False)
    if all_actions:
        pd.concat(all_actions, ignore_index=True).to_csv(actions_path, index=False)
    if all_candidate_debug:
        pd.DataFrame(all_candidate_debug).to_csv(candidate_debug_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print({"windows": str(out), "summary": str(summary_path), "actions": str(actions_path) if all_actions else "", "candidate_debug": str(candidate_debug_path) if all_candidate_debug else ""}, flush=True)


if __name__ == "__main__":
    main()

