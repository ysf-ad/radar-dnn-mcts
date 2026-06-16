from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from exact_env_mutual import attach_env_obs, xs_decode_action, xs_s_search_action, xs_x_search_action
from exact_env_mutual import xs_s_track_action, xs_x_track_action
from mutual_features import SLOT_DIM, slot_features, slot_features_batch, tokenize, tokenize_batch
from realistic_reward_retrain import adapter
from two_sensor_physical_head_eval import MAXT, ActionAttentionFactorizedNet


@dataclass(frozen=True)
class FastPlannerStats:
    encoded_once: bool
    device: str
    use_amp: bool
    use_compile: bool


@dataclass
class BatchedScoreResult:
    scores: np.ndarray
    actions: list[np.ndarray]
    bases: list[np.ndarray]
    sensors: list[np.ndarray]


@dataclass
class BatchedRootProposals:
    actions: np.ndarray
    scores: np.ndarray
    bases: np.ndarray
    sensors: np.ndarray
    valid: np.ndarray


@dataclass
class BatchedRootActionTables:
    actions: np.ndarray
    scores: np.ndarray
    bases: np.ndarray
    sensors: np.ndarray
    valid: np.ndarray
    counts: np.ndarray


@dataclass
class BatchedPhysicalActionTable:
    actions: np.ndarray
    bases: np.ndarray
    sensors: np.ndarray
    valid: np.ndarray


@dataclass
class PreparedBatchedRootBatch:
    tokens: np.ndarray
    slots: np.ndarray
    actions: np.ndarray
    flat_indices: np.ndarray
    valid: np.ndarray
    count: int


@dataclass
class DevicePreparedBatchedRootBatch:
    tokens: torch.Tensor
    slots: torch.Tensor
    actions: torch.Tensor
    flat_indices: torch.Tensor
    valid: torch.Tensor
    count: int
    graph: object | None = None
    graph_best: torch.Tensor | None = None


def _paired_mlp_outputs(
    left: nn.Module,
    right: nn.Module,
    x: torch.Tensor,
    cache: dict[tuple, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate two same-shape LayerNorm/Linear/GELU/Linear MLP heads together."""
    if (
        isinstance(left, nn.Sequential)
        and isinstance(right, nn.Sequential)
        and len(left) == 4
        and len(right) == 4
        and isinstance(left[0], nn.LayerNorm)
        and isinstance(right[0], nn.LayerNorm)
        and isinstance(left[1], nn.Linear)
        and isinstance(right[1], nn.Linear)
        and isinstance(left[2], nn.GELU)
        and isinstance(right[2], nn.GELU)
        and isinstance(left[3], nn.Linear)
        and isinstance(right[3], nn.Linear)
        and left[1].in_features == right[1].in_features
        and left[1].out_features == right[1].out_features
        and left[3].in_features == right[3].in_features
        and left[3].out_features == right[3].out_features
    ):
        key = (
            id(left),
            id(right),
            x.device.type,
            x.device.index,
            x.dtype,
            left[1].weight._version,
            right[1].weight._version,
            left[3].weight._version,
            right[3].weight._version,
        )
        packed = cache.get(key) if cache is not None else None
        if packed is None:
            w1 = torch.block_diag(left[1].weight.detach(), right[1].weight.detach())
            b1 = torch.cat([left[1].bias.detach(), right[1].bias.detach()], dim=0) if left[1].bias is not None and right[1].bias is not None else None
            w2 = torch.block_diag(left[3].weight.detach(), right[3].weight.detach())
            b2 = torch.cat([left[3].bias.detach(), right[3].bias.detach()], dim=0) if left[3].bias is not None and right[3].bias is not None else None
            packed = (w1, b1, w2, b2)
            if cache is not None:
                cache[key] = packed
        w1, b1, w2, b2 = packed
        x_left = left[0](x)
        x_right = right[0](x)
        hidden = torch.nn.functional.linear(torch.cat([x_left, x_right], dim=-1), w1, b1)
        left_h, right_h = hidden.split(left[1].out_features, dim=-1)
        left_h = left[2](left_h)
        right_h = right[2](right_h)
        out = torch.nn.functional.linear(torch.cat([left_h, right_h], dim=-1), w2, b2)
        return out.split(left[3].out_features, dim=-1)
    return left(x), right(x)


def _maybe_direct_encoder(module: nn.Module, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(module, nn.TransformerEncoder) and len(module.layers) == 1 and module.norm is None:
        return module.layers[0](x, src_key_padding_mask=src_key_padding_mask)
    if src_key_padding_mask is None:
        return module(x)
    return module(x, src_key_padding_mask=src_key_padding_mask)


def _manual_encoder_layer(layer: nn.TransformerEncoderLayer, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
    if layer.norm_first:
        y = layer.norm1(x)
        attn = layer.self_attn(
            y,
            y,
            y,
            attn_mask=None,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
            is_causal=False,
        )[0]
        x = x + layer.dropout1(attn)
        y = layer.norm2(x)
        y = layer.linear2(layer.dropout(layer.activation(layer.linear1(y))))
        return x + layer.dropout2(y)
    attn = layer.self_attn(
        x,
        x,
        x,
        attn_mask=None,
        key_padding_mask=src_key_padding_mask,
        need_weights=False,
        is_causal=False,
    )[0]
    x = layer.norm1(x + layer.dropout1(attn))
    y = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
    return layer.norm2(x + layer.dropout2(y))


def _maybe_manual_encoder(module: nn.Module, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(module, nn.TransformerEncoder) and len(module.layers) == 1 and module.norm is None:
        return _manual_encoder_layer(module.layers[0], x, src_key_padding_mask=src_key_padding_mask)
    if src_key_padding_mask is None:
        return module(x)
    return module(x, src_key_padding_mask=src_key_padding_mask)


def physical_action_arrays(obs: dict, selected: Iterable[int] | None = None, max_trackers: int = MAXT):
    """Return candidate action ids and score-table indices as NumPy arrays.

    The model already emits a dense score table of shape [rows, 2]. This helper
    builds the valid candidate view over that table without per-action decode
    loops in the hot selection path.
    """
    selected_set = set() if selected is None else {int(x) for x in selected}
    active = np.asarray(obs["active_mask"], dtype=bool)[:max_trackers]
    deadline = np.asarray(obs["t_deadline"], dtype=np.float32)[:max_trackers]
    ranges = np.asarray(obs.get("target_range", np.zeros(max_trackers, dtype=np.float32)), dtype=np.float32)[:max_trackers]

    base_ids = [0]
    sensor_ids = [0]
    action_ids = [xs_s_search_action(max_trackers)]

    x_free = int(obs.get("enable_x_band", 0)) and float(obs.get("x_band_busy_ms", 0.0)) <= 0.0
    s_free = float(obs.get("s_band_busy_ms", 0.0)) <= 0.0
    if x_free:
        base_ids.append(0)
        sensor_ids.append(1)
        action_ids.append(xs_x_search_action(max_trackers))

    valid_target = active & np.isfinite(deadline) & (deadline >= 0.0)
    if selected_set:
        selected_idx = np.fromiter((i - 1 for i in selected_set if 1 <= i <= max_trackers), dtype=np.int64, count=len(selected_set))
        selected_idx = selected_idx[(0 <= selected_idx) & (selected_idx < max_trackers)]
        if selected_idx.size:
            valid_target[selected_idx] = False

    ranked = np.flatnonzero(valid_target)
    if ranked.size:
        order = np.lexsort((ranked, deadline[ranked]))
        ranked = ranked[order]
        if s_free:
            s_ok = ranked[(ranges[ranked] > 10_000_000.0) & (ranges[ranked] < 184_000_000.0)]
            base_ids.extend((s_ok + 1).tolist())
            sensor_ids.extend([0] * int(s_ok.size))
            action_ids.extend([xs_s_track_action(int(i) + 1, max_trackers) for i in s_ok])
        if x_free:
            x_ok = ranked[(ranges[ranked] > 5_000_000.0) & (ranges[ranked] < 100_000_000.0)]
            base_ids.extend((x_ok + 1).tolist())
            sensor_ids.extend([1] * int(x_ok.size))
            action_ids.extend([xs_x_track_action(int(i) + 1, max_trackers) for i in x_ok])

    return (
        np.asarray(action_ids, dtype=np.int64),
        np.asarray(base_ids, dtype=np.int64),
        np.asarray(sensor_ids, dtype=np.int64),
    )


def physical_action_table_batch(
    observations: list[dict],
    selected: list[Iterable[int]] | None = None,
    max_trackers: int = MAXT,
) -> BatchedPhysicalActionTable:
    """Build dense valid physical action tables for many observations.

    Candidate order matches `physical_action_arrays`: S search, optional X
    search, S tracks by earliest deadline, then X tracks by earliest deadline.
    The table is dense and validity-masked so callers can gather model scores
    without rebuilding per-root Python action lists.
    """
    n = len(observations)
    width = 2 + 2 * int(max_trackers)
    actions = np.full((n, width), -1, dtype=np.int64)
    bases = np.zeros((n, width), dtype=np.int64)
    sensors = np.zeros((n, width), dtype=np.int64)
    valid = np.zeros((n, width), dtype=bool)
    if n <= 0:
        return BatchedPhysicalActionTable(actions=actions, bases=bases, sensors=sensors, valid=valid)

    active = np.stack([np.asarray(obs["active_mask"], dtype=bool)[:max_trackers] for obs in observations], axis=0)
    deadline = np.stack([np.asarray(obs["t_deadline"], dtype=np.float32)[:max_trackers] for obs in observations], axis=0)
    ranges = np.stack(
        [
            np.asarray(obs.get("target_range", np.zeros(max_trackers, dtype=np.float32)), dtype=np.float32)[:max_trackers]
            for obs in observations
        ],
        axis=0,
    )
    selected_mask = np.zeros((n, max_trackers), dtype=bool)
    if selected is not None:
        for row, selected_row in enumerate(selected):
            for base in selected_row:
                idx = int(base) - 1
                if 0 <= idx < max_trackers:
                    selected_mask[row, idx] = True

    s_free = np.asarray([float(obs.get("s_band_busy_ms", 0.0)) <= 0.0 for obs in observations], dtype=bool)
    x_free = np.asarray(
        [
            bool(int(obs.get("enable_x_band", 0))) and float(obs.get("x_band_busy_ms", 0.0)) <= 0.0
            for obs in observations
        ],
        dtype=bool,
    )
    valid_target = active & np.isfinite(deadline) & (deadline >= 0.0) & ~selected_mask
    sort_key = np.where(valid_target, deadline, np.inf)
    target_order = np.argsort(sort_key, axis=1, kind="stable")
    row_idx = np.arange(n)[:, None]
    ordered_ranges = ranges[row_idx, target_order]
    ordered_valid = valid_target[row_idx, target_order]
    ordered_bases = target_order + 1

    s_search = xs_s_search_action(max_trackers)
    x_search = xs_x_search_action(max_trackers)
    s_track_by_target = np.asarray([xs_s_track_action(i + 1, max_trackers) for i in range(max_trackers)], dtype=np.int64)
    x_track_by_target = np.asarray([xs_x_track_action(i + 1, max_trackers) for i in range(max_trackers)], dtype=np.int64)

    actions[:, 0] = s_search
    bases[:, 0] = 0
    sensors[:, 0] = 0
    valid[:, 0] = True

    actions[:, 1] = x_search
    bases[:, 1] = 0
    sensors[:, 1] = 1
    valid[:, 1] = x_free

    s_cols = slice(2, 2 + max_trackers)
    x_cols = slice(2 + max_trackers, 2 + 2 * max_trackers)
    actions[:, s_cols] = s_track_by_target[target_order]
    bases[:, s_cols] = ordered_bases
    sensors[:, s_cols] = 0
    valid[:, s_cols] = ordered_valid & s_free[:, None] & (ordered_ranges > 10_000_000.0) & (ordered_ranges < 184_000_000.0)

    actions[:, x_cols] = x_track_by_target[target_order]
    bases[:, x_cols] = ordered_bases
    sensors[:, x_cols] = 1
    valid[:, x_cols] = ordered_valid & x_free[:, None] & (ordered_ranges > 5_000_000.0) & (ordered_ranges < 100_000_000.0)
    return BatchedPhysicalActionTable(actions=actions, bases=bases, sensors=sensors, valid=valid)


def select_best_action(score: np.ndarray, obs: dict, selected: Iterable[int] | None = None, max_trackers: int = MAXT) -> int | None:
    actions, bases, sensors = physical_action_arrays(obs, selected=selected, max_trackers=max_trackers)
    if actions.size == 0:
        return None
    vals = np.asarray(score, dtype=np.float32)[bases, sensors]
    if vals.size == 0 or not np.isfinite(vals).any():
        return None
    return int(actions[int(np.nanargmax(vals))])


def select_topk_actions(score: np.ndarray, obs: dict, selected: Iterable[int] | None = None, k: int = 8, max_trackers: int = MAXT):
    actions, bases, sensors = physical_action_arrays(obs, selected=selected, max_trackers=max_trackers)
    if actions.size == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    vals = np.asarray(score, dtype=np.float32)[bases, sensors]
    finite = np.isfinite(vals)
    if not finite.any():
        return actions[:0], vals[:0], bases[:0], sensors[:0]
    actions = actions[finite]
    bases = bases[finite]
    sensors = sensors[finite]
    vals = vals[finite]
    take = min(int(k), int(vals.size))
    if take <= 0:
        return actions[:0], vals[:0], bases[:0], sensors[:0]
    part = np.argpartition(-vals, take - 1)[:take]
    order = part[np.argsort(-vals[part])]
    return actions[order], vals[order], bases[order], sensors[order]


class FastActionAttentionPlanner:
    """Low-latency direct planner for the action-attention factorized PQ model.

    The baseline PhysicalHeadPlanner re-tokenizes and re-encodes the same root
    target set every decision inside a 200 ms scheduling window. This planner
    encodes target/context tokens once, then only updates slot features and the
    selected-target mask at each sequential decision.
    """

    def __init__(
        self,
        model: ActionAttentionFactorizedNet,
        env_cfg: dict,
        policy_weight: float = 1.0,
        q_weight: float = 1.0,
        search_score_bias: float = 0.0,
        device: str | torch.device | None = None,
        use_amp: bool = False,
        use_compile: bool = False,
        use_cuda_graph: bool = False,
        use_gpu_select: bool = False,
        use_paired_heads: bool = False,
        use_direct_couplers: bool = False,
        use_manual_couplers: bool = False,
    ):
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.eval().to(dev)
        if use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode="reduce-overhead")
        self.env_cfg = dict(env_cfg)
        self.policy_weight = float(policy_weight)
        self.q_weight = float(q_weight)
        self.search_score_bias = float(search_score_bias)
        self.device = dev
        self.use_amp = bool(use_amp and dev.type == "cuda")
        self.use_compile = bool(use_compile)
        self.use_cuda_graph = bool(use_cuda_graph and dev.type == "cuda")
        self.use_gpu_select = bool(use_gpu_select)
        self.use_paired_heads = bool(use_paired_heads)
        self.use_direct_couplers = bool(use_direct_couplers)
        self.use_manual_couplers = bool(use_manual_couplers)
        self.adapt = adapter()
        self.stats = FastPlannerStats(True, str(dev), self.use_amp, self.use_compile)
        self._row_is_search_cache: dict[tuple[int, str, int | None], torch.Tensor] = {}
        self._cuda_graph_score_cache: dict[tuple, dict[str, object]] = {}
        self._paired_head_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]] = {}
        self._action_base_cache: dict[int, np.ndarray] = {}
        self.profile_enabled = False
        self._profile_values: dict[str, list[float]] = defaultdict(list)

    def set_profile_enabled(self, enabled: bool = True) -> None:
        self.profile_enabled = bool(enabled)

    def reset_profile(self) -> None:
        self._profile_values.clear()

    def profile_summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, values in self._profile_values.items():
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                continue
            out[name] = {
                "calls": int(arr.size),
                "total_ms": float(arr.sum()),
                "mean_ms": float(arr.mean()),
                "p50_ms": float(np.percentile(arr, 50)),
                "p90_ms": float(np.percentile(arr, 90)),
                "p99_ms": float(np.percentile(arr, 99)),
            }
        return dict(sorted(out.items(), key=lambda kv: kv[1]["total_ms"], reverse=True))

    def _profile_sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _profile_start(self) -> float:
        if not self.profile_enabled:
            return 0.0
        self._profile_sync()
        return time.perf_counter()

    def _profile_end(self, name: str, start: float) -> None:
        if not self.profile_enabled:
            return
        self._profile_sync()
        self._profile_values[name].append((time.perf_counter() - start) * 1000.0)

    def _row_is_search(self, rows: int, device: torch.device) -> torch.Tensor:
        dev = torch.device(device)
        key = (int(rows), dev.type, dev.index)
        cached = self._row_is_search_cache.get(key)
        if cached is None:
            cached = (torch.arange(int(rows), device=dev)[None, :, None] == 0)
            self._row_is_search_cache[key] = cached
        return cached

    def _scores_from_encoded(self, cls_out, tok_out, selected_t, token_active, slot_t):
        model = self.model
        slot_emb = model.backbone.slot_proj(slot_t)
        bsz, rows, _ = tok_out.shape

        sensor = model.sensor_embed[None, :, :].expand(bsz, -1, -1)
        cls_s = cls_out[:, None, :].expand(-1, 2, -1)
        slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
        sensor_state = model.sensor_state_proj(torch.cat([cls_s, slot_s, sensor], dim=-1))
        if self.use_manual_couplers:
            coupled_sensor = _maybe_manual_encoder(model.sensor_coupler, sensor_state)
        elif self.use_direct_couplers:
            coupled_sensor = _maybe_direct_encoder(model.sensor_coupler, sensor_state)
        else:
            coupled_sensor = model.sensor_coupler(sensor_state)
        type_ctx = torch.cat([cls_s, slot_s, coupled_sensor], dim=-1)
        if self.use_paired_heads:
            type_logits, type_q = _paired_mlp_outputs(model.type_head, model.type_q_head, type_ctx, self._paired_head_cache)
        else:
            type_logits = model.type_head(type_ctx)
            type_q = model.type_q_head(type_ctx)

        tok_st = tok_out[:, :, None, :].expand(-1, -1, 2, -1)
        cls_st = cls_out[:, None, None, :].expand(-1, rows, 2, -1)
        slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
        sensor_st = coupled_sensor[:, None, :, :].expand(bsz, rows, -1, -1)
        target_ctx = torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)
        if self.use_paired_heads:
            target_logits_raw, target_q_raw = _paired_mlp_outputs(model.target_head, model.target_q_head, target_ctx, self._paired_head_cache)
            target_logits = target_logits_raw.squeeze(-1)
            target_q = target_q_raw.squeeze(-1)
        else:
            target_logits = model.target_head(target_ctx).squeeze(-1)
            target_q = model.target_q_head(target_ctx).squeeze(-1)

        base_scores = slot_t.new_full((bsz, rows, 2), -1e9)
        base_q = slot_t.new_zeros((bsz, rows, 2))
        base_scores[:, 0, :] = type_logits[:, :, 0]
        base_q[:, 0, :] = type_q[:, :, 0]
        track_mask = token_active & ~selected_t
        track_mask[:, 0] = False
        base_scores[:, 1:, :] = (type_logits[:, None, :, 1] + target_logits)[:, 1:, :]
        base_q[:, 1:, :] = (type_q[:, None, :, 1] + target_q)[:, 1:, :]

        row_is_search = self._row_is_search(rows, slot_t.device)
        valid = (track_mask[:, :, None] | row_is_search).expand(-1, -1, 2)
        action_ctx = model.action_proj(target_ctx).reshape(bsz, rows * 2, -1)
        action_mask = ~valid.reshape(bsz, rows * 2)
        if self.use_manual_couplers:
            action_ctx = _maybe_manual_encoder(model.action_coupler, action_ctx, src_key_padding_mask=action_mask)
        elif self.use_direct_couplers:
            action_ctx = _maybe_direct_encoder(model.action_coupler, action_ctx, src_key_padding_mask=action_mask)
        else:
            action_ctx = model.action_coupler(action_ctx, src_key_padding_mask=action_mask)
        if self.use_paired_heads:
            residual_raw, q_residual_raw = _paired_mlp_outputs(
                model.action_policy_residual,
                model.action_q_residual,
                action_ctx,
                self._paired_head_cache,
            )
            residual = residual_raw.reshape(bsz, rows, 2)
            q_residual = q_residual_raw.reshape(bsz, rows, 2)
        else:
            residual = model.action_policy_residual(action_ctx).reshape(bsz, rows, 2)
            q_residual = model.action_q_residual(action_ctx).reshape(bsz, rows, 2)
        scores = (base_scores + residual).masked_fill(~valid, -1e9)
        q = (base_q + q_residual).masked_fill(~valid, 0.0)
        return self.policy_weight * scores + self.q_weight * q

    def _combined_scores_from_encoded(self, cls_out, tok_out, selected_t, token_active, slot_t):
        """Compute policy/Q weighted scores directly for inference.

        This is algebraically equivalent to `_scores_from_encoded`, but avoids
        allocating and masking separate full policy and Q score tables before
        combining them.
        """
        model = self.model
        slot_emb = model.backbone.slot_proj(slot_t)
        bsz, rows, _ = tok_out.shape

        sensor = model.sensor_embed[None, :, :].expand(bsz, -1, -1)
        cls_s = cls_out[:, None, :].expand(-1, 2, -1)
        slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
        sensor_state = model.sensor_state_proj(torch.cat([cls_s, slot_s, sensor], dim=-1))
        if self.use_manual_couplers:
            coupled_sensor = _maybe_manual_encoder(model.sensor_coupler, sensor_state)
        elif self.use_direct_couplers:
            coupled_sensor = _maybe_direct_encoder(model.sensor_coupler, sensor_state)
        else:
            coupled_sensor = model.sensor_coupler(sensor_state)
        type_ctx = torch.cat([cls_s, slot_s, coupled_sensor], dim=-1)
        if self.use_paired_heads:
            type_logits, type_q = _paired_mlp_outputs(model.type_head, model.type_q_head, type_ctx, self._paired_head_cache)
        else:
            type_logits = model.type_head(type_ctx)
            type_q = model.type_q_head(type_ctx)

        tok_st = tok_out[:, :, None, :].expand(-1, -1, 2, -1)
        cls_st = cls_out[:, None, None, :].expand(-1, rows, 2, -1)
        slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
        sensor_st = coupled_sensor[:, None, :, :].expand(bsz, rows, -1, -1)
        target_ctx = torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)
        if self.use_paired_heads:
            target_logits_raw, target_q_raw = _paired_mlp_outputs(model.target_head, model.target_q_head, target_ctx, self._paired_head_cache)
            target_logits = target_logits_raw.squeeze(-1)
            target_q = target_q_raw.squeeze(-1)
        else:
            target_logits = model.target_head(target_ctx).squeeze(-1)
            target_q = model.target_q_head(target_ctx).squeeze(-1)

        track_mask = token_active & ~selected_t
        track_mask[:, 0] = False
        row_is_search = self._row_is_search(rows, slot_t.device)
        valid = (track_mask[:, :, None] | row_is_search).expand(-1, -1, 2)
        action_ctx = model.action_proj(target_ctx).reshape(bsz, rows * 2, -1)
        action_mask = ~valid.reshape(bsz, rows * 2)
        if self.use_manual_couplers:
            action_ctx = _maybe_manual_encoder(model.action_coupler, action_ctx, src_key_padding_mask=action_mask)
        elif self.use_direct_couplers:
            action_ctx = _maybe_direct_encoder(model.action_coupler, action_ctx, src_key_padding_mask=action_mask)
        else:
            action_ctx = model.action_coupler(action_ctx, src_key_padding_mask=action_mask)
        if self.use_paired_heads:
            residual_raw, q_residual_raw = _paired_mlp_outputs(
                model.action_policy_residual,
                model.action_q_residual,
                action_ctx,
                self._paired_head_cache,
            )
            residual = residual_raw.reshape(bsz, rows, 2)
            q_residual = q_residual_raw.reshape(bsz, rows, 2)
        else:
            residual = model.action_policy_residual(action_ctx).reshape(bsz, rows, 2)
            q_residual = model.action_q_residual(action_ctx).reshape(bsz, rows, 2)

        combined = slot_t.new_empty((bsz, rows, 2))
        combined[:, 0, :] = self.policy_weight * (type_logits[:, :, 0] + residual[:, 0, :])
        combined[:, 0, :] += self.q_weight * (type_q[:, :, 0] + q_residual[:, 0, :])
        combined[:, 1:, :] = self.policy_weight * (type_logits[:, None, :, 1] + target_logits + residual)[:, 1:, :]
        combined[:, 1:, :] += self.q_weight * (type_q[:, None, :, 1] + target_q + q_residual)[:, 1:, :]
        return combined.masked_fill(~valid, -1e9)

    def score_slots_from_encoded(self, cls_out, tok_out, selected_t, token_active, slot_t):
        """Score many slot/selected contexts against one cached target encoding.

        ``encode_tokens`` is root-state specific, while the scheduling loop
        changes the slot/context vector and selected-target mask at each
        decision. This helper batches those per-decision contexts so the action
        attention/Q-policy heads can be evaluated with a larger batch dimension.
        """
        if slot_t.ndim == 1:
            slot_t = slot_t.unsqueeze(0)
        batch = int(slot_t.shape[0])
        if selected_t.ndim == 1:
            selected_t = selected_t.unsqueeze(0)
        if selected_t.shape[0] == 1 and batch > 1:
            selected_t = selected_t.expand(batch, -1)
        if token_active.ndim == 1:
            token_active = token_active.unsqueeze(0)
        if token_active.shape[0] == 1 and batch > 1:
            token_active = token_active.expand(batch, -1)
        if cls_out.shape[0] == 1 and batch > 1:
            cls_out = cls_out.expand(batch, -1)
        if tok_out.shape[0] == 1 and batch > 1:
            tok_out = tok_out.expand(batch, -1, -1)
        return self._combined_scores_from_encoded(cls_out, tok_out, selected_t, token_active, slot_t)

    def _build_cuda_graph_score_replay(
        self,
        cls_out,
        tok_out,
        root_selected,
        token_active,
        slot_width: int,
    ):
        if not self.use_cuda_graph or self.device.type != "cuda":
            return None
        key = (
            tuple(cls_out.shape),
            tuple(tok_out.shape),
            tuple(root_selected.shape),
            tuple(token_active.shape),
            int(slot_width),
            str(cls_out.dtype),
            bool(self.use_amp),
        )
        try:
            cache = self._cuda_graph_score_cache.get(key)
            if cache is None:
                static_cls = torch.empty_like(cls_out)
                static_tok = torch.empty_like(tok_out)
                static_selected = torch.empty_like(root_selected)
                static_active = torch.empty_like(token_active)
                static_slot = torch.empty((1, int(slot_width)), device=self.device, dtype=torch.float32)
                static_cls.copy_(cls_out, non_blocking=False)
                static_tok.copy_(tok_out, non_blocking=False)
                static_selected.copy_(root_selected, non_blocking=False)
                static_active.copy_(token_active, non_blocking=False)

                with torch.inference_mode():
                    for _ in range(3):
                        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                            _ = self._combined_scores_from_encoded(
                                static_cls,
                                static_tok,
                                static_selected,
                                static_active,
                                static_slot,
                            ).squeeze(0).float()
                    torch.cuda.synchronize(self.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                            static_score = self._combined_scores_from_encoded(
                                static_cls,
                                static_tok,
                                static_selected,
                                static_active,
                                static_slot,
                            ).squeeze(0).float()
                cache = {
                    "graph": graph,
                    "static_cls": static_cls,
                    "static_tok": static_tok,
                    "static_selected": static_selected,
                    "static_active": static_active,
                    "static_slot": static_slot,
                    "static_score": static_score,
                }
                self._cuda_graph_score_cache[key] = cache
            else:
                static_cls = cache["static_cls"]
                static_tok = cache["static_tok"]
                static_selected = cache["static_selected"]
                static_active = cache["static_active"]
                static_slot = cache["static_slot"]
                static_score = cache["static_score"]
                graph = cache["graph"]
                static_cls.copy_(cls_out, non_blocking=False)
                static_tok.copy_(tok_out, non_blocking=False)
                static_selected.copy_(root_selected, non_blocking=False)
                static_active.copy_(token_active, non_blocking=False)

            def replay(slot_cpu_t: torch.Tensor) -> torch.Tensor:
                static_slot.copy_(slot_cpu_t, non_blocking=False)
                graph.replay()
                return static_score

            return replay, static_selected
        except Exception:
            return None

    def _physical_action_tensors(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions, bases, sensors = physical_action_arrays(obs, selected=None, max_trackers=MAXT)
        flat_indices = bases.astype(np.int64, copy=False) * 2 + sensors.astype(np.int64, copy=False)
        is_search = (bases == 0).astype(np.bool_, copy=False)
        return (
            torch.from_numpy(actions.astype(np.int64, copy=False)).to(self.device),
            torch.from_numpy(flat_indices).to(self.device),
            torch.from_numpy(is_search).to(self.device),
        )

    def _select_best_action_torch(
        self,
        score_t: torch.Tensor,
        actions_t: torch.Tensor,
        flat_indices_t: torch.Tensor,
        is_search_t: torch.Tensor,
    ) -> int | None:
        if actions_t.numel() <= 0:
            return None
        vals = torch.take(score_t.reshape(-1), flat_indices_t)
        if self.search_score_bias != 0.0:
            vals = vals + is_search_t.to(vals.dtype) * float(self.search_score_bias)
        action = int(actions_t[torch.max(vals, dim=0).indices].item())
        return action if action >= 0 else None

    def _slot_template(self, obs: dict, budget_ms: float) -> np.ndarray:
        return slot_features(obs, 0.0, 0, 0, -1, float(budget_ms)).astype(np.float32, copy=True)

    def _action_base_lookup(self, max_trackers: int = MAXT) -> np.ndarray:
        max_trackers = int(max_trackers)
        cached = self._action_base_cache.get(max_trackers)
        if cached is not None:
            return cached
        size = xs_x_track_action(max_trackers, max_trackers) + 1
        lookup = np.full((size,), -1, dtype=np.int32)
        lookup[xs_s_search_action(max_trackers)] = 0
        lookup[xs_x_search_action(max_trackers)] = 0
        s0 = xs_s_track_action(1, max_trackers)
        x0 = xs_x_track_action(1, max_trackers)
        bases = np.arange(1, max_trackers + 1, dtype=np.int32)
        lookup[s0 : s0 + max_trackers] = bases
        lookup[x0 : x0 + max_trackers] = bases
        self._action_base_cache[max_trackers] = lookup
        return lookup

    @staticmethod
    def _update_slot_inplace(
        slot: np.ndarray,
        template: np.ndarray,
        elapsed: float,
        search_count: int,
        track_count: int,
        last_action: int,
        budget_ms: float,
    ) -> np.ndarray:
        slot[:] = template
        slot[0] = float(elapsed) / float(budget_ms)
        slot[1] = float(search_count) / 20.0
        slot[2] = float(track_count) / 100.0
        slot[3] = 1.0 if int(last_action) == 0 else 0.0
        return slot

    def warmup(self, obs, budget_ms=200):
        """Populate CUDA kernels/graph cache before timed online planning."""
        profiling = bool(self.profile_enabled)
        self.profile_enabled = False
        try:
            plan = self.plan(obs, budget_ms=budget_ms)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            return plan
        finally:
            self.profile_enabled = profiling

    def plan(self, obs, budget_ms=200):
        t_plan = self._profile_start()
        t0 = self._profile_start()
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        self._profile_end("attach_env_obs", t0)
        t0 = self._profile_start()
        root_tok = tokenize(self.adapt, obs, selected=set(), search_count=0).astype(np.float32)
        self._profile_end("root_tokenize", t0)
        with torch.inference_mode():
            t0 = self._profile_start()
            root_x = torch.from_numpy(root_tok).to(self.device, dtype=torch.float32).unsqueeze(0)
            self._profile_end("root_h2d", t0)
            t0 = self._profile_start()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                cls_out, tok_out, root_selected, token_active = self.model.backbone.encode_tokens(root_x)
            self._profile_end("root_encode", t0)

        selected: set[int] = set()
        plan: list[int] = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        selected_t = root_selected.clone()
        action_base_lookup = self._action_base_lookup(MAXT)
        dwell_arr = np.asarray(obs["t_dwell"], dtype=np.float32)
        slot_width = int(self.model.backbone.slot_proj[0].normalized_shape[0])
        t0 = self._profile_start()
        action_tensors = None
        if self.use_gpu_select:
            action_tensors = self._physical_action_tensors(obs)
            self._profile_end("candidate_h2d", t0)
            t0 = self._profile_start()
        graph_pack = self._build_cuda_graph_score_replay(cls_out, tok_out, selected_t, token_active, slot_width)
        if graph_pack is not None:
            graph_replay, selected_t = graph_pack
        else:
            graph_replay = None
        self._profile_end("cuda_graph_prepare", t0)
        t0 = self._profile_start()
        slot_template = self._slot_template(obs, float(budget_ms))
        slot = np.empty((SLOT_DIM,), dtype=np.float32)
        slot_cpu_t = torch.from_numpy(slot).unsqueeze(0)
        self._profile_end("slot_template", t0)
        while elapsed < float(budget_ms) and len(plan) < 64:
            t0 = self._profile_start()
            slot = self._update_slot_inplace(slot, slot_template, elapsed, search_count, track_count, last, float(budget_ms))
            self._profile_end("loop_slot_features_fast", t0)
            if graph_replay is not None:
                with torch.inference_mode():
                    t0 = self._profile_start()
                    graph_out = graph_replay(slot_cpu_t)
                    self._profile_end("loop_score_graph_replay", t0)
                    score_t = graph_out
                    selected_action_t = None
                    if self.use_gpu_select and action_tensors is not None:
                        score = None
                    else:
                        t0 = self._profile_start()
                        score = score_t.cpu().numpy()
                        self._profile_end("loop_score_d2h", t0)
            else:
                with torch.inference_mode():
                    t0 = self._profile_start()
                    slot_t = torch.from_numpy(slot).to(self.device, dtype=torch.float32).unsqueeze(0)
                    self._profile_end("loop_slot_h2d", t0)
                    t0 = self._profile_start()
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                        score_t = self._combined_scores_from_encoded(cls_out, tok_out, selected_t, token_active, slot_t).squeeze(0).float()
                    self._profile_end("loop_score_forward", t0)
                    if self.use_gpu_select and action_tensors is not None:
                        score = None
                    else:
                        t0 = self._profile_start()
                        score = score_t.cpu().numpy()
                        self._profile_end("loop_score_d2h", t0)
            if self.use_gpu_select and action_tensors is not None:
                t0 = self._profile_start()
                if graph_replay is not None and selected_action_t is not None:
                    action = int(selected_action_t.item())
                    best_action = action if action >= 0 else None
                else:
                    best_action = self._select_best_action_torch(score_t, *action_tensors)
                self._profile_end("loop_select_best_action_gpu", t0)
            else:
                t0 = self._profile_start()
                score = np.asarray(score, dtype=np.float32).copy()
                score[0, :] += self.search_score_bias
                self._profile_end("loop_score_postprocess", t0)
                t0 = self._profile_start()
                best_action = select_best_action(score, obs, selected=selected, max_trackers=MAXT)
                self._profile_end("loop_select_best_action", t0)
            if best_action is None:
                break
            t0 = self._profile_start()
            plan.append(best_action)
            base = int(action_base_lookup[int(best_action)]) if 0 <= int(best_action) < int(action_base_lookup.size) else xs_decode_action(best_action, MAXT)[0]
            if int(base) == 0:
                search_count += 1
                dt = 10.0
            else:
                selected.add(int(base))
                if 0 <= int(base) < selected_t.shape[1]:
                    selected_t[0, int(base)] = True
                track_count += 1
                dt = float(dwell_arr[int(base) - 1]) if int(base) - 1 < len(dwell_arr) else 10.0
            elapsed += max(1.0, float(dt))
            last = int(base)
            self._profile_end("loop_bookkeeping", t0)
        out = plan if plan else [xs_s_search_action(MAXT)]
        self._profile_end("plan_total", t_plan)
        return out


class BatchedActionAttentionScorer:
    """Batch many radar states through the action-attention policy/Q network.

    This is the throughput-oriented API. It does not try to make one sequential
    200 ms window magically parallel; it batches many windows/root states/rollout
    branches into one model call so the GPU sees enough work.
    """

    def __init__(
        self,
        model: ActionAttentionFactorizedNet,
        env_cfg: dict,
        policy_weight: float = 1.0,
        q_weight: float = 1.0,
        search_score_bias: float = 0.0,
        device: str | torch.device | None = None,
        use_amp: bool = False,
        use_compile: bool = False,
        use_cuda_graph: bool = False,
    ):
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.eval().to(dev)
        if use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode="reduce-overhead")
        self.env_cfg = dict(env_cfg)
        self.policy_weight = float(policy_weight)
        self.q_weight = float(q_weight)
        self.search_score_bias = float(search_score_bias)
        self.device = dev
        self.use_amp = bool(use_amp and dev.type == "cuda")
        self.use_cuda_graph = bool(use_cuda_graph and dev.type == "cuda")
        self.adapt = adapter()
        self._row_is_search_cache: dict[tuple[int, str, int | None], torch.Tensor] = {}
        self._prepared_graph_cache: dict[tuple, dict[str, object]] = {}

    def _row_is_search(self, rows: int, device: torch.device) -> torch.Tensor:
        dev = torch.device(device)
        key = (int(rows), dev.type, dev.index)
        cached = self._row_is_search_cache.get(key)
        if cached is None:
            cached = (torch.arange(int(rows), device=dev)[None, :, None] == 0)
            self._row_is_search_cache[key] = cached
        return cached

    def _combined_scores_from_tokens(self, tokens: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        model = self.model
        cls_out, tok_out, selected, token_active = model.backbone.encode_tokens(tokens)
        slot_emb = model.backbone.slot_proj(slot)
        bsz, rows, _ = tok_out.shape

        sensor = model.sensor_embed[None, :, :].expand(bsz, -1, -1)
        cls_s = cls_out[:, None, :].expand(-1, 2, -1)
        slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
        sensor_state = model.sensor_state_proj(torch.cat([cls_s, slot_s, sensor], dim=-1))
        coupled_sensor = model.sensor_coupler(sensor_state)
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

        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        row_is_search = self._row_is_search(rows, tokens.device)
        valid = (track_mask[:, :, None] | row_is_search).expand(-1, -1, 2)
        action_ctx = model.action_proj(target_ctx).reshape(bsz, rows * 2, -1)
        action_ctx = model.action_coupler(action_ctx, src_key_padding_mask=~valid.reshape(bsz, rows * 2))
        residual = model.action_policy_residual(action_ctx).reshape(bsz, rows, 2)
        q_residual = model.action_q_residual(action_ctx).reshape(bsz, rows, 2)

        combined = tokens.new_empty((bsz, rows, 2))
        combined[:, 0, :] = self.policy_weight * (type_logits[:, :, 0] + residual[:, 0, :])
        combined[:, 0, :] += self.q_weight * (type_q[:, :, 0] + q_residual[:, 0, :])
        combined[:, 1:, :] = self.policy_weight * (type_logits[:, None, :, 1] + target_logits + residual)[:, 1:, :]
        combined[:, 1:, :] += self.q_weight * (type_q[:, None, :, 1] + target_q + q_residual)[:, 1:, :]
        return combined.masked_fill(~valid, -1e9)

    def _score_dense(
        self,
        observations: list[dict],
        selected: list[Iterable[int]] | None = None,
        elapsed: Iterable[float] | None = None,
        search_count: Iterable[int] | None = None,
        track_count: Iterable[int] | None = None,
        last: Iterable[int] | None = None,
        budget_ms: float = 200.0,
    ) -> tuple[np.ndarray, list[dict], list[set[int]]]:
        n = len(observations)
        selected = [set() for _ in range(n)] if selected is None else [set(x) for x in selected]
        elapsed = [0.0] * n if elapsed is None else list(elapsed)
        search_count = [0] * n if search_count is None else list(search_count)
        track_count = [0] * n if track_count is None else list(track_count)
        last = [-1] * n if last is None else list(last)

        obs2 = [attach_env_obs(obs, self.env_cfg, True, True) for obs in observations]
        tokens = tokenize_batch(self.adapt, obs2, selected=selected, search_count=search_count)
        slots = slot_features_batch(
            obs2,
            elapsed=elapsed,
            search_count=search_count,
            track_count=track_count,
            last_action=last,
            budget_ms=float(budget_ms),
        )
        with torch.inference_mode():
            x = torch.from_numpy(tokens).to(self.device, dtype=torch.float32)
            s = torch.from_numpy(slots).to(self.device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                score_t = self._combined_scores_from_tokens(x, s)
            score = score_t.float().cpu().numpy()
        score[:, 0, :] += self.search_score_bias
        return np.asarray(score, dtype=np.float32), obs2, selected

    def _score_dense_torch(
        self,
        observations: list[dict],
        selected: list[Iterable[int]] | None = None,
        elapsed: Iterable[float] | None = None,
        search_count: Iterable[int] | None = None,
        track_count: Iterable[int] | None = None,
        last: Iterable[int] | None = None,
        budget_ms: float = 200.0,
    ) -> tuple[torch.Tensor, list[dict], list[set[int]]]:
        n = len(observations)
        selected = [set() for _ in range(n)] if selected is None else [set(x) for x in selected]
        elapsed = [0.0] * n if elapsed is None else list(elapsed)
        search_count = [0] * n if search_count is None else list(search_count)
        track_count = [0] * n if track_count is None else list(track_count)
        last = [-1] * n if last is None else list(last)

        obs2 = [attach_env_obs(obs, self.env_cfg, True, True) for obs in observations]
        tokens = tokenize_batch(self.adapt, obs2, selected=selected, search_count=search_count)
        slots = slot_features_batch(
            obs2,
            elapsed=elapsed,
            search_count=search_count,
            track_count=track_count,
            last_action=last,
            budget_ms=float(budget_ms),
        )
        with torch.inference_mode():
            x = torch.from_numpy(tokens).to(self.device, dtype=torch.float32)
            s = torch.from_numpy(slots).to(self.device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                score = self._combined_scores_from_tokens(x, s).float()
            score[:, 0, :] += self.search_score_bias
        return score, obs2, selected

    def score_batch(
        self,
        observations: list[dict],
        selected: list[Iterable[int]] | None = None,
        elapsed: Iterable[float] | None = None,
        search_count: Iterable[int] | None = None,
        track_count: Iterable[int] | None = None,
        last: Iterable[int] | None = None,
        budget_ms: float = 200.0,
    ) -> BatchedScoreResult:
        score, obs2, selected = self._score_dense(
            observations,
            selected=selected,
            elapsed=elapsed,
            search_count=search_count,
            track_count=track_count,
            last=last,
            budget_ms=budget_ms,
        )
        n = len(observations)
        action_arrays = [physical_action_arrays(obs2[i], selected=selected[i], max_trackers=MAXT) for i in range(n)]
        return BatchedScoreResult(
            scores=score,
            actions=[a[0] for a in action_arrays],
            bases=[a[1] for a in action_arrays],
            sensors=[a[2] for a in action_arrays],
        )

    def best_actions(self, observations: list[dict], **kwargs) -> np.ndarray:
        result = self.score_batch(observations, **kwargs)
        out = np.full((len(observations),), -1, dtype=np.int64)
        for i in range(len(observations)):
            actions = result.actions[i]
            if actions.size == 0:
                continue
            vals = result.scores[i, result.bases[i], result.sensors[i]]
            if np.isfinite(vals).any():
                out[i] = int(actions[int(np.nanargmax(vals))])
        return out

    def best_actions_torch(self, observations: list[dict], **kwargs) -> np.ndarray:
        """Return best root actions while keeping gather/argmax on the device."""
        score_t, obs2, selected = self._score_dense_torch(observations, **kwargs)
        n = len(observations)
        if n <= 0:
            return np.empty((0,), dtype=np.int64)
        physical = physical_action_table_batch(obs2, selected=selected, max_trackers=MAXT)
        device = score_t.device
        actions_t = torch.as_tensor(physical.actions, device=device, dtype=torch.long)
        flat_t = torch.as_tensor(physical.bases * 2 + physical.sensors, device=device, dtype=torch.long)
        valid_t = torch.as_tensor(physical.valid, device=device, dtype=torch.bool)
        flat_scores = score_t.reshape(n, -1)
        candidate_scores = torch.gather(flat_scores, 1, flat_t)
        candidate_scores = candidate_scores.masked_fill(~(valid_t & torch.isfinite(candidate_scores)), -torch.inf)
        idx = torch.argmax(candidate_scores, dim=1)
        best = torch.gather(actions_t, 1, idx[:, None]).squeeze(1)
        has_valid = torch.any(torch.isfinite(candidate_scores), dim=1)
        best = torch.where(has_valid, best, torch.full_like(best, -1))
        return best.cpu().numpy().astype(np.int64, copy=False)

    def prepare_root_batch(
        self,
        observations: list[dict],
        selected: list[Iterable[int]] | None = None,
        elapsed: Iterable[float] | None = None,
        search_count: Iterable[int] | None = None,
        track_count: Iterable[int] | None = None,
        last: Iterable[int] | None = None,
        budget_ms: float = 200.0,
    ) -> PreparedBatchedRootBatch:
        """Precompute CPU-side batch inputs for repeated root scoring."""
        n = len(observations)
        selected_sets = [set() for _ in range(n)] if selected is None else [set(x) for x in selected]
        elapsed = [0.0] * n if elapsed is None else list(elapsed)
        search_count = [0] * n if search_count is None else list(search_count)
        track_count = [0] * n if track_count is None else list(track_count)
        last = [-1] * n if last is None else list(last)
        obs2 = [attach_env_obs(obs, self.env_cfg, True, True) for obs in observations]
        tokens = tokenize_batch(self.adapt, obs2, selected=selected_sets, search_count=search_count)
        slots = slot_features_batch(
            obs2,
            elapsed=elapsed,
            search_count=search_count,
            track_count=track_count,
            last_action=last,
            budget_ms=float(budget_ms),
        )
        physical = physical_action_table_batch(obs2, selected=selected_sets, max_trackers=MAXT)
        flat_indices = (physical.bases.astype(np.int64, copy=False) * 2 + physical.sensors.astype(np.int64, copy=False)).astype(
            np.int64,
            copy=False,
        )
        return PreparedBatchedRootBatch(
            tokens=tokens,
            slots=slots,
            actions=physical.actions.astype(np.int64, copy=False),
            flat_indices=flat_indices,
            valid=physical.valid.astype(bool, copy=False),
            count=int(n),
        )

    def best_actions_prepared_torch(self, prepared: PreparedBatchedRootBatch) -> np.ndarray:
        """Score a prepared root batch with GPU gather/argmax selection."""
        n = int(prepared.count)
        if n <= 0:
            return np.empty((0,), dtype=np.int64)
        with torch.inference_mode():
            x = torch.from_numpy(prepared.tokens).to(self.device, dtype=torch.float32)
            s = torch.from_numpy(prepared.slots).to(self.device, dtype=torch.float32)
            actions_t = torch.as_tensor(prepared.actions, device=self.device, dtype=torch.long)
            flat_t = torch.as_tensor(prepared.flat_indices, device=self.device, dtype=torch.long)
            valid_t = torch.as_tensor(prepared.valid, device=self.device, dtype=torch.bool)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                score_t = self._combined_scores_from_tokens(x, s).float()
            score_t[:, 0, :] += self.search_score_bias
            flat_scores = score_t.reshape(n, -1)
            candidate_scores = torch.gather(flat_scores, 1, flat_t)
            candidate_scores = candidate_scores.masked_fill(~(valid_t & torch.isfinite(candidate_scores)), -torch.inf)
            idx = torch.argmax(candidate_scores, dim=1)
            best = torch.gather(actions_t, 1, idx[:, None]).squeeze(1)
            has_valid = torch.any(torch.isfinite(candidate_scores), dim=1)
            best = torch.where(has_valid, best, torch.full_like(best, -1))
            return best.cpu().numpy().astype(np.int64, copy=False)

    def _build_prepared_graph_replay(self, prepared: PreparedBatchedRootBatch):
        if not self.use_cuda_graph or self.device.type != "cuda":
            return None
        n = int(prepared.count)
        key = (
            tuple(prepared.tokens.shape),
            tuple(prepared.slots.shape),
            tuple(prepared.actions.shape),
            tuple(prepared.flat_indices.shape),
            tuple(prepared.valid.shape),
            str(prepared.tokens.dtype),
            str(prepared.slots.dtype),
            bool(self.use_amp),
        )
        try:
            cache = self._prepared_graph_cache.get(key)
            if cache is None:
                static_tokens = torch.empty(tuple(prepared.tokens.shape), device=self.device, dtype=torch.float32)
                static_slots = torch.empty(tuple(prepared.slots.shape), device=self.device, dtype=torch.float32)
                static_actions = torch.empty(tuple(prepared.actions.shape), device=self.device, dtype=torch.long)
                static_flat = torch.empty(tuple(prepared.flat_indices.shape), device=self.device, dtype=torch.long)
                static_valid = torch.empty(tuple(prepared.valid.shape), device=self.device, dtype=torch.bool)

                def load_inputs() -> None:
                    static_tokens.copy_(torch.from_numpy(np.ascontiguousarray(prepared.tokens)), non_blocking=False)
                    static_slots.copy_(torch.from_numpy(np.ascontiguousarray(prepared.slots)), non_blocking=False)
                    static_actions.copy_(torch.from_numpy(np.ascontiguousarray(prepared.actions)), non_blocking=False)
                    static_flat.copy_(torch.from_numpy(np.ascontiguousarray(prepared.flat_indices)), non_blocking=False)
                    static_valid.copy_(torch.from_numpy(np.ascontiguousarray(prepared.valid)), non_blocking=False)

                def compute_best() -> torch.Tensor:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                        score_t = self._combined_scores_from_tokens(static_tokens, static_slots).float()
                    score_t[:, 0, :] += self.search_score_bias
                    flat_scores = score_t.reshape(n, -1)
                    candidate_scores = torch.gather(flat_scores, 1, static_flat)
                    candidate_scores = candidate_scores.masked_fill(~(static_valid & torch.isfinite(candidate_scores)), -torch.inf)
                    idx = torch.argmax(candidate_scores, dim=1)
                    best = torch.gather(static_actions, 1, idx[:, None]).squeeze(1)
                    has_valid = torch.any(torch.isfinite(candidate_scores), dim=1)
                    return torch.where(has_valid, best, torch.full_like(best, -1))

                load_inputs()
                with torch.inference_mode():
                    for _ in range(3):
                        _ = compute_best()
                    torch.cuda.synchronize(self.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        static_best = compute_best()
                cache = {
                    "graph": graph,
                    "static_tokens": static_tokens,
                    "static_slots": static_slots,
                    "static_actions": static_actions,
                    "static_flat": static_flat,
                    "static_valid": static_valid,
                    "static_best": static_best,
                }
                self._prepared_graph_cache[key] = cache
            return cache
        except Exception:
            return None

    def best_actions_prepared_graph(self, prepared: PreparedBatchedRootBatch) -> np.ndarray:
        """Score a prepared root batch through a fixed-shape CUDA Graph replay."""
        n = int(prepared.count)
        if n <= 0:
            return np.empty((0,), dtype=np.int64)
        cache = self._build_prepared_graph_replay(prepared)
        if cache is None:
            return self.best_actions_prepared_torch(prepared)
        static_tokens = cache["static_tokens"]
        static_slots = cache["static_slots"]
        static_actions = cache["static_actions"]
        static_flat = cache["static_flat"]
        static_valid = cache["static_valid"]
        static_tokens.copy_(torch.from_numpy(np.ascontiguousarray(prepared.tokens)), non_blocking=False)
        static_slots.copy_(torch.from_numpy(np.ascontiguousarray(prepared.slots)), non_blocking=False)
        static_actions.copy_(torch.from_numpy(np.ascontiguousarray(prepared.actions)), non_blocking=False)
        static_flat.copy_(torch.from_numpy(np.ascontiguousarray(prepared.flat_indices)), non_blocking=False)
        static_valid.copy_(torch.from_numpy(np.ascontiguousarray(prepared.valid)), non_blocking=False)
        cache["graph"].replay()
        return cache["static_best"].cpu().numpy().astype(np.int64, copy=False)

    def prepared_to_device(self, prepared: PreparedBatchedRootBatch) -> DevicePreparedBatchedRootBatch:
        """Move a prepared root batch to persistent device tensors."""
        return DevicePreparedBatchedRootBatch(
            tokens=torch.from_numpy(np.ascontiguousarray(prepared.tokens)).to(self.device, dtype=torch.float32),
            slots=torch.from_numpy(np.ascontiguousarray(prepared.slots)).to(self.device, dtype=torch.float32),
            actions=torch.from_numpy(np.ascontiguousarray(prepared.actions)).to(self.device, dtype=torch.long),
            flat_indices=torch.from_numpy(np.ascontiguousarray(prepared.flat_indices)).to(self.device, dtype=torch.long),
            valid=torch.from_numpy(np.ascontiguousarray(prepared.valid)).to(self.device, dtype=torch.bool),
            count=int(prepared.count),
        )

    def prepare_root_batch_device(self, observations: list[dict], **kwargs) -> DevicePreparedBatchedRootBatch:
        """Precompute root-batch inputs and keep them resident on the scoring device."""
        return self.prepared_to_device(self.prepare_root_batch(observations, **kwargs))

    def _best_actions_from_device_tensors(self, prepared: DevicePreparedBatchedRootBatch) -> torch.Tensor:
        n = int(prepared.count)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
            score_t = self._combined_scores_from_tokens(prepared.tokens, prepared.slots).float()
        score_t[:, 0, :] += self.search_score_bias
        flat_scores = score_t.reshape(n, -1)
        candidate_scores = torch.gather(flat_scores, 1, prepared.flat_indices)
        candidate_scores = candidate_scores.masked_fill(~(prepared.valid & torch.isfinite(candidate_scores)), -torch.inf)
        idx = torch.argmax(candidate_scores, dim=1)
        best = torch.gather(prepared.actions, 1, idx[:, None]).squeeze(1)
        has_valid = torch.any(torch.isfinite(candidate_scores), dim=1)
        return torch.where(has_valid, best, torch.full_like(best, -1))

    def best_actions_prepared_device(self, prepared: DevicePreparedBatchedRootBatch) -> np.ndarray:
        """Score a device-resident prepared batch without rebuilding input tensors."""
        n = int(prepared.count)
        if n <= 0:
            return np.empty((0,), dtype=np.int64)
        with torch.inference_mode():
            best = self._best_actions_from_device_tensors(prepared)
            return best.cpu().numpy().astype(np.int64, copy=False)

    def best_actions_prepared_device_graph(self, prepared: DevicePreparedBatchedRootBatch) -> np.ndarray:
        """Replay a CUDA Graph over a fixed device-resident prepared batch."""
        n = int(prepared.count)
        if n <= 0:
            return np.empty((0,), dtype=np.int64)
        if not self.use_cuda_graph or self.device.type != "cuda":
            return self.best_actions_prepared_device(prepared)
        try:
            if prepared.graph is None or prepared.graph_best is None:
                with torch.inference_mode():
                    for _ in range(3):
                        _ = self._best_actions_from_device_tensors(prepared)
                    torch.cuda.synchronize(self.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        graph_best = self._best_actions_from_device_tensors(prepared)
                prepared.graph = graph
                prepared.graph_best = graph_best
            prepared.graph.replay()
            return prepared.graph_best.cpu().numpy().astype(np.int64, copy=False)
        except Exception:
            return self.best_actions_prepared_device(prepared)

    def topk_root_proposals(self, observations: list[dict], k: int = 8, **kwargs) -> BatchedRootProposals:
        result = self.score_batch(observations, **kwargs)
        n = len(observations)
        actions = np.full((n, int(k)), -1, dtype=np.int64)
        scores = np.full((n, int(k)), -np.inf, dtype=np.float32)
        bases = np.full((n, int(k)), -1, dtype=np.int64)
        sensors = np.full((n, int(k)), -1, dtype=np.int64)
        valid = np.zeros((n, int(k)), dtype=bool)
        for i in range(n):
            vals = result.scores[i, result.bases[i], result.sensors[i]]
            finite = np.isfinite(vals)
            if not finite.any():
                continue
            row_actions = result.actions[i][finite]
            row_bases = result.bases[i][finite]
            row_sensors = result.sensors[i][finite]
            row_vals = vals[finite]
            take = min(int(k), int(row_vals.size))
            part = np.argpartition(-row_vals, take - 1)[:take]
            order = part[np.argsort(-row_vals[part])]
            actions[i, :take] = row_actions[order]
            scores[i, :take] = row_vals[order]
            bases[i, :take] = row_bases[order]
            sensors[i, :take] = row_sensors[order]
            valid[i, :take] = True
        return BatchedRootProposals(actions=actions, scores=scores, bases=bases, sensors=sensors, valid=valid)

    def all_root_action_tables(self, observations: list[dict], max_actions: int | None = None, **kwargs) -> BatchedRootActionTables:
        """Return sorted valid root action tables for many observations.

        The policy/Q model emits dense `[rows, sensors]` score tables. This
        helper gathers the physically valid action ids for each root state,
        sorts them by model score, and pads them into dense arrays suitable for
        cached root search.
        """
        result = self.score_batch(observations, **kwargs)
        n = len(observations)
        counts = np.zeros((n,), dtype=np.int32)
        sorted_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        width = 0
        for i in range(n):
            vals = result.scores[i, result.bases[i], result.sensors[i]]
            finite = np.isfinite(vals)
            row_actions = result.actions[i][finite].astype(np.int64, copy=False)
            row_bases = result.bases[i][finite].astype(np.int64, copy=False)
            row_sensors = result.sensors[i][finite].astype(np.int64, copy=False)
            row_vals = vals[finite].astype(np.float32, copy=False)
            if row_vals.size:
                order = np.argsort(-row_vals)
                row_actions = row_actions[order]
                row_bases = row_bases[order]
                row_sensors = row_sensors[order]
                row_vals = row_vals[order]
            counts[i] = int(row_vals.size)
            width = max(width, int(row_vals.size))
            sorted_rows.append((row_actions, row_vals, row_bases, row_sensors))

        if max_actions is not None:
            width = min(width, int(max_actions))
        width = max(width, 0)
        actions = np.full((n, width), -1, dtype=np.int64)
        scores = np.full((n, width), -np.inf, dtype=np.float32)
        bases = np.full((n, width), -1, dtype=np.int64)
        sensors = np.full((n, width), -1, dtype=np.int64)
        valid = np.zeros((n, width), dtype=bool)
        for i, (row_actions, row_vals, row_bases, row_sensors) in enumerate(sorted_rows):
            take = min(width, int(row_vals.size))
            if take <= 0:
                continue
            actions[i, :take] = row_actions[:take]
            scores[i, :take] = row_vals[:take]
            bases[i, :take] = row_bases[:take]
            sensors[i, :take] = row_sensors[:take]
            valid[i, :take] = True
        return BatchedRootActionTables(
            actions=actions,
            scores=scores,
            bases=bases,
            sensors=sensors,
            valid=valid,
            counts=counts,
        )

    def all_root_action_tables_vectorized(
        self,
        observations: list[dict],
        max_actions: int | None = None,
        **kwargs,
    ) -> BatchedRootActionTables:
        """Return sorted valid root action tables using batched physical masks."""
        score, obs2, selected = self._score_dense(observations, **kwargs)
        physical = physical_action_table_batch(obs2, selected=selected, max_trackers=MAXT)
        n = len(observations)
        if n <= 0:
            width = 0 if max_actions is None else int(max_actions)
            return BatchedRootActionTables(
                actions=np.full((0, width), -1, dtype=np.int64),
                scores=np.full((0, width), -np.inf, dtype=np.float32),
                bases=np.full((0, width), -1, dtype=np.int64),
                sensors=np.full((0, width), -1, dtype=np.int64),
                valid=np.zeros((0, width), dtype=bool),
                counts=np.zeros((0,), dtype=np.int32),
            )

        row_ids = np.arange(n)[:, None]
        candidate_scores = score[row_ids, physical.bases, physical.sensors]
        candidate_scores = np.where(physical.valid & np.isfinite(candidate_scores), candidate_scores, -np.inf).astype(np.float32)
        finite = np.isfinite(candidate_scores)
        counts = finite.sum(axis=1).astype(np.int32)
        width = int(counts.max(initial=0))
        if max_actions is not None:
            width = min(width, int(max_actions))

        actions = np.full((n, width), -1, dtype=np.int64)
        scores = np.full((n, width), -np.inf, dtype=np.float32)
        bases = np.full((n, width), -1, dtype=np.int64)
        sensors = np.full((n, width), -1, dtype=np.int64)
        valid = np.zeros((n, width), dtype=bool)
        for i in range(n):
            row_finite = finite[i]
            row_count = int(row_finite.sum())
            if row_count <= 0 or width <= 0:
                continue
            row_actions = physical.actions[i, row_finite]
            row_bases = physical.bases[i, row_finite]
            row_sensors = physical.sensors[i, row_finite]
            row_scores = candidate_scores[i, row_finite]
            order = np.argsort(-row_scores)
            take = min(width, row_count)
            chosen = order[:take]
            actions[i, :take] = row_actions[chosen]
            scores[i, :take] = row_scores[chosen]
            bases[i, :take] = row_bases[chosen]
            sensors[i, :take] = row_sensors[chosen]
            valid[i, :take] = True
        return BatchedRootActionTables(actions=actions, scores=scores, bases=bases, sensors=sensors, valid=valid, counts=counts)

    def all_root_action_tables_torch(
        self,
        observations: list[dict],
        max_actions: int | None = None,
        **kwargs,
    ) -> BatchedRootActionTables:
        """Return sorted root action tables with score gather/sort on torch device."""
        score_t, obs2, selected = self._score_dense_torch(observations, **kwargs)
        physical = physical_action_table_batch(obs2, selected=selected, max_trackers=MAXT)
        n = len(observations)
        if n <= 0:
            width = 0 if max_actions is None else int(max_actions)
            return BatchedRootActionTables(
                actions=np.full((0, width), -1, dtype=np.int64),
                scores=np.full((0, width), -np.inf, dtype=np.float32),
                bases=np.full((0, width), -1, dtype=np.int64),
                sensors=np.full((0, width), -1, dtype=np.int64),
                valid=np.zeros((0, width), dtype=bool),
                counts=np.zeros((0,), dtype=np.int32),
            )

        device = score_t.device
        actions_t = torch.as_tensor(physical.actions, device=device, dtype=torch.long)
        bases_t = torch.as_tensor(physical.bases, device=device, dtype=torch.long)
        sensors_t = torch.as_tensor(physical.sensors, device=device, dtype=torch.long)
        valid_t = torch.as_tensor(physical.valid, device=device, dtype=torch.bool)
        row_ids = torch.arange(n, device=device, dtype=torch.long)[:, None]
        candidate_scores = score_t[row_ids, bases_t, sensors_t]
        candidate_scores = candidate_scores.masked_fill(~(valid_t & torch.isfinite(candidate_scores)), -torch.inf)
        counts_t = torch.sum(torch.isfinite(candidate_scores), dim=1).to(torch.int32)
        width = int(counts_t.max().item()) if n else 0
        if max_actions is not None:
            width = min(width, int(max_actions))

        sorted_scores_t, order_t = torch.sort(candidate_scores, dim=1, descending=True, stable=True)
        if width > 0:
            order_t = order_t[:, :width]
            sorted_scores_t = sorted_scores_t[:, :width]
            sorted_actions_t = torch.gather(actions_t, 1, order_t)
            sorted_bases_t = torch.gather(bases_t, 1, order_t)
            sorted_sensors_t = torch.gather(sensors_t, 1, order_t)
            sorted_valid_t = torch.isfinite(sorted_scores_t)
            actions = sorted_actions_t.cpu().numpy().astype(np.int64, copy=False)
            scores = sorted_scores_t.cpu().numpy().astype(np.float32, copy=False)
            bases = sorted_bases_t.cpu().numpy().astype(np.int64, copy=False)
            sensors = sorted_sensors_t.cpu().numpy().astype(np.int64, copy=False)
            valid = sorted_valid_t.cpu().numpy().astype(bool, copy=False)
        else:
            actions = np.full((n, 0), -1, dtype=np.int64)
            scores = np.full((n, 0), -np.inf, dtype=np.float32)
            bases = np.full((n, 0), -1, dtype=np.int64)
            sensors = np.full((n, 0), -1, dtype=np.int64)
            valid = np.zeros((n, 0), dtype=bool)
        counts = counts_t.cpu().numpy().astype(np.int32, copy=False)
        return BatchedRootActionTables(actions=actions, scores=scores, bases=bases, sensors=sensors, valid=valid, counts=counts)

    def all_root_action_tables_fast(
        self,
        observations: list[dict],
        max_actions: int | None = None,
        torch_batch_threshold: int = 96,
        **kwargs,
    ) -> BatchedRootActionTables:
        """Use the fastest measured root-table path for the current batch."""
        if self.device.type == "cuda" and len(observations) >= int(torch_batch_threshold):
            return self.all_root_action_tables_torch(observations, max_actions=max_actions, **kwargs)
        return self.all_root_action_tables_vectorized(observations, max_actions=max_actions, **kwargs)
