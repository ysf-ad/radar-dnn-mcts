from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from exact_env_mutual import attach_env_obs, xs_decode_action
from mutual_features import slot_features, tokenize
from perf_fast_planner import FastActionAttentionPlanner, physical_action_arrays
from realistic_reward_retrain import adapter
from two_sensor_physical_head_eval import MAXT


@dataclass(frozen=True)
class BranchPrefix:
    """A partial within-window plan prefix.

    This is intentionally lightweight: it stores exactly the changing context
    needed by the factorized action-attention heads. The root target encoding is
    shared across all prefixes in a batch.
    """

    actions: tuple[int, ...] = ()
    selected: frozenset[int] = frozenset()
    elapsed_ms: float = 0.0
    search_count: int = 0
    track_count: int = 0
    last: int = -1
    score_sum: float = 0.0


@dataclass
class BranchExpansionResult:
    actions: np.ndarray
    scores: np.ndarray
    bases: np.ndarray
    sensors: np.ndarray
    valid: np.ndarray
    score_tables: np.ndarray


class BatchedWindowExpansionScorer:
    """Batch next-action scoring for many partial prefixes under one root obs."""

    def __init__(
        self,
        planner: FastActionAttentionPlanner,
        obs: dict,
        budget_ms: float = 200.0,
    ):
        self.planner = planner
        self.env_cfg = dict(planner.env_cfg)
        self.obs = attach_env_obs(obs, self.env_cfg, True, True)
        self.budget_ms = float(budget_ms)
        self.adapt = adapter()
        root_tok = tokenize(self.adapt, self.obs, selected=set(), search_count=0).astype(np.float32)
        with torch.inference_mode():
            root_x = torch.from_numpy(root_tok).to(planner.device, dtype=torch.float32).unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=planner.use_amp):
                self.cls_out, self.tok_out, self.root_selected, self.token_active = planner.model.backbone.encode_tokens(root_x)

    def _selected_masks(self, prefixes: list[BranchPrefix]) -> torch.Tensor:
        rows = int(self.root_selected.shape[1])
        masks = self.root_selected.expand(len(prefixes), -1).clone()
        for row, prefix in enumerate(prefixes):
            for base in prefix.selected:
                if 0 <= int(base) < rows:
                    masks[row, int(base)] = True
        return masks

    def _slots(self, prefixes: list[BranchPrefix]) -> torch.Tensor:
        slots = np.stack(
            [
                slot_features(
                    self.obs,
                    float(prefix.elapsed_ms),
                    int(prefix.search_count),
                    int(prefix.track_count),
                    int(prefix.last),
                    self.budget_ms,
                ).astype(np.float32)
                for prefix in prefixes
            ],
            axis=0,
        )
        return torch.from_numpy(slots).to(self.planner.device, dtype=torch.float32)

    def score_prefixes(self, prefixes: Iterable[BranchPrefix]) -> BranchExpansionResult:
        prefix_list = list(prefixes)
        if not prefix_list:
            empty = np.empty((0,), dtype=np.int64)
            return BranchExpansionResult(
                actions=empty,
                scores=np.empty((0,), dtype=np.float32),
                bases=empty,
                sensors=empty,
                valid=np.zeros((0,), dtype=bool),
                score_tables=np.empty((0, MAXT + 1, 2), dtype=np.float32),
            )
        with torch.inference_mode():
            selected_t = self._selected_masks(prefix_list)
            slot_t = self._slots(prefix_list)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.planner.use_amp):
                score_t = self.planner.score_slots_from_encoded(
                    self.cls_out,
                    self.tok_out,
                    selected_t,
                    self.token_active,
                    slot_t,
                )
            score_tables = score_t.float().cpu().numpy()
        score_tables[:, 0, :] += self.planner.search_score_bias

        out_actions = np.full((len(prefix_list),), -1, dtype=np.int64)
        out_scores = np.full((len(prefix_list),), -np.inf, dtype=np.float32)
        out_bases = np.full((len(prefix_list),), -1, dtype=np.int64)
        out_sensors = np.full((len(prefix_list),), -1, dtype=np.int64)
        out_valid = np.zeros((len(prefix_list),), dtype=bool)
        for row, prefix in enumerate(prefix_list):
            actions, bases, sensors = physical_action_arrays(self.obs, selected=prefix.selected, max_trackers=MAXT)
            if actions.size == 0:
                continue
            vals = score_tables[row, bases, sensors]
            finite = np.isfinite(vals)
            if not finite.any():
                continue
            pick_local = int(np.nanargmax(np.where(finite, vals, -np.inf)))
            out_actions[row] = int(actions[pick_local])
            out_scores[row] = float(vals[pick_local])
            out_bases[row] = int(bases[pick_local])
            out_sensors[row] = int(sensors[pick_local])
            out_valid[row] = True
        return BranchExpansionResult(
            actions=out_actions,
            scores=out_scores,
            bases=out_bases,
            sensors=out_sensors,
            valid=out_valid,
            score_tables=np.asarray(score_tables, dtype=np.float32),
        )

    def expand_prefixes(self, prefixes: Iterable[BranchPrefix], top_k: int = 1) -> list[BranchPrefix]:
        prefix_list = list(prefixes)
        if not prefix_list:
            return []
        scored = self.score_prefixes(prefix_list)
        out: list[BranchPrefix] = []
        for row, prefix in enumerate(prefix_list):
            actions, bases, sensors = physical_action_arrays(self.obs, selected=prefix.selected, max_trackers=MAXT)
            if actions.size == 0:
                continue
            vals = scored.score_tables[row, bases, sensors]
            finite = np.isfinite(vals)
            if not finite.any():
                continue
            row_actions = actions[finite]
            row_vals = vals[finite]
            take = min(int(top_k), int(row_vals.size))
            if take <= 0:
                continue
            part = np.argpartition(-row_vals, take - 1)[:take]
            order = part[np.argsort(-row_vals[part])]
            for idx in order:
                out.append(prefix_after_action(self.obs, prefix, int(row_actions[idx]), float(row_vals[idx])))
        return out


class BatchedBeamWindowPlanner:
    """Window planner that expands many partial prefixes per model call."""

    def __init__(
        self,
        planner: FastActionAttentionPlanner,
        beam_width: int = 8,
        branch_top_k: int = 2,
        max_depth: int = 64,
    ):
        self.planner = planner
        self.beam_width = int(beam_width)
        self.branch_top_k = int(branch_top_k)
        self.max_depth = int(max_depth)

    def plan(self, obs, budget_ms=200):
        scorer = BatchedWindowExpansionScorer(self.planner, obs, budget_ms=float(budget_ms))
        frontier = [BranchPrefix()]
        best = frontier[0]
        for _depth in range(max(1, self.max_depth)):
            live = [p for p in frontier if float(p.elapsed_ms) < float(budget_ms)]
            if not live:
                break
            children = scorer.expand_prefixes(live, top_k=max(1, self.branch_top_k))
            children = [p for p in children if p.actions]
            if not children:
                break
            children.sort(key=lambda p: (float(p.score_sum), -float(p.elapsed_ms)), reverse=True)
            frontier = children[: max(1, self.beam_width)]
            if frontier and float(frontier[0].score_sum) >= float(best.score_sum):
                best = frontier[0]
            if best.actions and float(best.elapsed_ms) >= float(budget_ms):
                break
        return list(best.actions) if best.actions else [int(MAXT) + 3]


def prefix_after_action(obs: dict, prefix: BranchPrefix, action: int, score_delta: float = 0.0) -> BranchPrefix:
    base, _sensor = xs_decode_action(int(action), MAXT)
    actions = (*prefix.actions, int(action))
    if int(base) == 0:
        return BranchPrefix(
            actions=actions,
            selected=prefix.selected,
            elapsed_ms=float(prefix.elapsed_ms) + 10.0,
            search_count=int(prefix.search_count) + 1,
            track_count=int(prefix.track_count),
            last=0,
            score_sum=float(prefix.score_sum) + float(score_delta),
        )
    selected = set(prefix.selected)
    selected.add(int(base))
    dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
    dt = float(dwell[int(base) - 1]) if int(base) - 1 < len(dwell) else 10.0
    return BranchPrefix(
        actions=actions,
        selected=frozenset(selected),
        elapsed_ms=float(prefix.elapsed_ms) + max(1.0, dt),
        search_count=int(prefix.search_count),
        track_count=int(prefix.track_count) + 1,
        last=int(base),
        score_sum=float(prefix.score_sum) + float(score_delta),
    )
