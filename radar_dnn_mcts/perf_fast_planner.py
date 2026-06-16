from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from exact_env_mutual import attach_env_obs, xs_decode_action, xs_s_search_action, xs_x_search_action
from exact_env_mutual import xs_s_track_action, xs_x_track_action
from mutual_features import slot_features, tokenize
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
        self.adapt = adapter()
        self.stats = FastPlannerStats(True, str(dev), self.use_amp, self.use_compile)

    def _scores_from_encoded(self, cls_out, tok_out, selected_t, token_active, slot_t):
        model = self.model
        slot_emb = model.backbone.slot_proj(slot_t)
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

        base_scores = slot_t.new_full((bsz, rows, 2), -1e9)
        base_q = slot_t.new_zeros((bsz, rows, 2))
        base_scores[:, 0, :] = type_logits[:, :, 0]
        base_q[:, 0, :] = type_q[:, :, 0]
        track_mask = token_active & ~selected_t
        track_mask[:, 0] = False
        base_scores[:, 1:, :] = (type_logits[:, None, :, 1] + target_logits)[:, 1:, :]
        base_q[:, 1:, :] = (type_q[:, None, :, 1] + target_q)[:, 1:, :]

        row_is_search = torch.arange(rows, device=slot_t.device)[None, :, None] == 0
        valid = (track_mask[:, :, None] | row_is_search).expand(-1, -1, 2)
        action_ctx = model.action_proj(target_ctx).reshape(bsz, rows * 2, -1)
        action_ctx = model.action_coupler(action_ctx, src_key_padding_mask=~valid.reshape(bsz, rows * 2))
        residual = model.action_policy_residual(action_ctx).reshape(bsz, rows, 2)
        q_residual = model.action_q_residual(action_ctx).reshape(bsz, rows, 2)
        scores = (base_scores + residual).masked_fill(~valid, -1e9)
        q = (base_q + q_residual).masked_fill(~valid, 0.0)
        return self.policy_weight * scores + self.q_weight * q

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
        return self._scores_from_encoded(cls_out, tok_out, selected_t, token_active, slot_t)

    def plan(self, obs, budget_ms=200):
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        root_tok = tokenize(self.adapt, obs, selected=set(), search_count=0).astype(np.float32)
        with torch.inference_mode():
            root_x = torch.from_numpy(root_tok).to(self.device, dtype=torch.float32).unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                cls_out, tok_out, root_selected, token_active = self.model.backbone.encode_tokens(root_x)

        selected: set[int] = set()
        plan: list[int] = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        selected_t = root_selected.clone()
        while elapsed < float(budget_ms) and len(plan) < 64:
            slot = slot_features(obs, elapsed, search_count, track_count, last, float(budget_ms)).astype(np.float32)
            with torch.inference_mode():
                slot_t = torch.from_numpy(slot).to(self.device, dtype=torch.float32).unsqueeze(0)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                    score_t = self._scores_from_encoded(cls_out, tok_out, selected_t, token_active, slot_t)
                score = score_t.squeeze(0).float().cpu().numpy()
            score = np.asarray(score, dtype=np.float32).copy()
            score[0, :] += self.search_score_bias

            best_action = select_best_action(score, obs, selected=selected, max_trackers=MAXT)
            if best_action is None:
                break
            plan.append(best_action)
            base, _sensor = xs_decode_action(best_action, MAXT)
            if int(base) == 0:
                search_count += 1
                dt = 10.0
            else:
                selected.add(int(base))
                if 0 <= int(base) < selected_t.shape[1]:
                    selected_t[0, int(base)] = True
                track_count += 1
                dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
                dt = float(dwell[int(base) - 1]) if int(base) - 1 < len(dwell) else 10.0
            elapsed += max(1.0, float(dt))
            last = int(base)
        return plan if plan else [xs_s_search_action(MAXT)]


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
        self.adapt = adapter()

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
        n = len(observations)
        selected = [set() for _ in range(n)] if selected is None else [set(x) for x in selected]
        elapsed = [0.0] * n if elapsed is None else list(elapsed)
        search_count = [0] * n if search_count is None else list(search_count)
        track_count = [0] * n if track_count is None else list(track_count)
        last = [-1] * n if last is None else list(last)

        obs2 = [attach_env_obs(obs, self.env_cfg, True, True) for obs in observations]
        tokens = np.stack(
            [
                tokenize(self.adapt, obs, selected=selected[i], search_count=int(search_count[i])).astype(np.float32)
                for i, obs in enumerate(obs2)
            ],
            axis=0,
        )
        slots = np.stack(
            [
                slot_features(
                    obs,
                    float(elapsed[i]),
                    int(search_count[i]),
                    int(track_count[i]),
                    int(last[i]),
                    float(budget_ms),
                ).astype(np.float32)
                for i, obs in enumerate(obs2)
            ],
            axis=0,
        )
        with torch.inference_mode():
            x = torch.from_numpy(tokens).to(self.device, dtype=torch.float32)
            s = torch.from_numpy(slots).to(self.device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                scores, q = self.model.forward_scores(x, s)
            score = (self.policy_weight * scores + self.q_weight * q).float().cpu().numpy()
        score[:, 0, :] += self.search_score_bias
        action_arrays = [physical_action_arrays(obs2[i], selected=selected[i], max_trackers=MAXT) for i in range(n)]
        return BatchedScoreResult(
            scores=np.asarray(score, dtype=np.float32),
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
