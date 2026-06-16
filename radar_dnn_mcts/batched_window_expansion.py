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


@dataclass
class DevicePreparedPrefixBatch:
    selected: torch.Tensor
    slots: torch.Tensor
    actions: torch.Tensor
    flat_indices: torch.Tensor
    bases: torch.Tensor
    sensors: torch.Tensor
    valid: torch.Tensor
    count: int
    graph: object | None = None
    graph_best_actions: torch.Tensor | None = None
    graph_best_scores: torch.Tensor | None = None
    graph_best_bases: torch.Tensor | None = None
    graph_best_sensors: torch.Tensor | None = None
    graph_has_valid: torch.Tensor | None = None


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

    def _physical_tables_for_prefixes(self, prefixes: list[BranchPrefix]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        width = 2 + 2 * int(MAXT)
        n = len(prefixes)
        actions_t = np.full((n, width), -1, dtype=np.int64)
        bases_t = np.zeros((n, width), dtype=np.int64)
        sensors_t = np.zeros((n, width), dtype=np.int64)
        valid_t = np.zeros((n, width), dtype=bool)
        for row, prefix in enumerate(prefixes):
            actions, bases, sensors = physical_action_arrays(self.obs, selected=prefix.selected, max_trackers=MAXT)
            take = min(int(actions.size), width)
            if take <= 0:
                continue
            actions_t[row, :take] = actions[:take]
            bases_t[row, :take] = bases[:take]
            sensors_t[row, :take] = sensors[:take]
            valid_t[row, :take] = True
        return actions_t, bases_t, sensors_t, valid_t

    def score_prefixes_gpu_select(self, prefixes: Iterable[BranchPrefix]) -> BranchExpansionResult:
        """Score prefixes and keep valid-action argmax on the GPU.

        This returns the same top-1 action fields as ``score_prefixes`` but does
        not copy the full dense score table back to CPU. It is the better path
        for MCTS/frontier code that only needs the selected child per prefix.
        """
        prefix_list = list(prefixes)
        if not prefix_list:
            empty = np.empty((0,), dtype=np.int64)
            return BranchExpansionResult(
                actions=empty,
                scores=np.empty((0,), dtype=np.float32),
                bases=empty,
                sensors=empty,
                valid=np.zeros((0,), dtype=bool),
                score_tables=np.empty((0, 0, 0), dtype=np.float32),
            )
        actions_np, bases_np, sensors_np, valid_np = self._physical_tables_for_prefixes(prefix_list)
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
                ).float()
            score_t[:, 0, :] += self.planner.search_score_bias
            flat_scores = score_t.reshape(len(prefix_list), -1)
            flat_idx_np = bases_np * 2 + sensors_np
            actions_dev = torch.as_tensor(actions_np, device=self.planner.device, dtype=torch.long)
            bases_dev = torch.as_tensor(bases_np, device=self.planner.device, dtype=torch.long)
            sensors_dev = torch.as_tensor(sensors_np, device=self.planner.device, dtype=torch.long)
            flat_idx = torch.as_tensor(flat_idx_np, device=self.planner.device, dtype=torch.long)
            valid_dev = torch.as_tensor(valid_np, device=self.planner.device, dtype=torch.bool)
            candidate_scores = torch.gather(flat_scores, 1, flat_idx)
            candidate_scores = candidate_scores.masked_fill(~(valid_dev & torch.isfinite(candidate_scores)), -torch.inf)
            idx = torch.argmax(candidate_scores, dim=1)
            rows = torch.arange(len(prefix_list), device=self.planner.device)
            best_scores = candidate_scores[rows, idx]
            best_actions = actions_dev[rows, idx]
            best_bases = bases_dev[rows, idx]
            best_sensors = sensors_dev[rows, idx]
            has_valid = torch.any(torch.isfinite(candidate_scores), dim=1)
            best_actions = torch.where(has_valid, best_actions, torch.full_like(best_actions, -1))
            best_scores = torch.where(has_valid, best_scores, torch.full_like(best_scores, -torch.inf))
            best_bases = torch.where(has_valid, best_bases, torch.full_like(best_bases, -1))
            best_sensors = torch.where(has_valid, best_sensors, torch.full_like(best_sensors, -1))
            return BranchExpansionResult(
                actions=best_actions.cpu().numpy().astype(np.int64, copy=False),
                scores=best_scores.cpu().numpy().astype(np.float32, copy=False),
                bases=best_bases.cpu().numpy().astype(np.int64, copy=False),
                sensors=best_sensors.cpu().numpy().astype(np.int64, copy=False),
                valid=has_valid.cpu().numpy().astype(bool, copy=False),
                score_tables=np.empty((0, 0, 0), dtype=np.float32),
            )

    def prepare_prefixes_device(self, prefixes: Iterable[BranchPrefix]) -> DevicePreparedPrefixBatch:
        """Precompute prefix masks, slots, and valid action tables on device."""
        prefix_list = list(prefixes)
        n = len(prefix_list)
        if n <= 0:
            empty_long = torch.empty((0, 0), device=self.planner.device, dtype=torch.long)
            empty_bool = torch.empty((0, 0), device=self.planner.device, dtype=torch.bool)
            return DevicePreparedPrefixBatch(
                selected=torch.empty((0, int(self.root_selected.shape[1])), device=self.planner.device, dtype=torch.bool),
                slots=torch.empty((0, 0), device=self.planner.device, dtype=torch.float32),
                actions=empty_long,
                flat_indices=empty_long,
                bases=empty_long,
                sensors=empty_long,
                valid=empty_bool,
                count=0,
            )
        actions_np, bases_np, sensors_np, valid_np = self._physical_tables_for_prefixes(prefix_list)
        return DevicePreparedPrefixBatch(
            selected=self._selected_masks(prefix_list),
            slots=self._slots(prefix_list),
            actions=torch.as_tensor(actions_np, device=self.planner.device, dtype=torch.long),
            flat_indices=torch.as_tensor(bases_np * 2 + sensors_np, device=self.planner.device, dtype=torch.long),
            bases=torch.as_tensor(bases_np, device=self.planner.device, dtype=torch.long),
            sensors=torch.as_tensor(sensors_np, device=self.planner.device, dtype=torch.long),
            valid=torch.as_tensor(valid_np, device=self.planner.device, dtype=torch.bool),
            count=int(n),
        )

    def _best_from_prepared_device(self, prepared: DevicePreparedPrefixBatch):
        n = int(prepared.count)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.planner.use_amp):
            score_t = self.planner.score_slots_from_encoded(
                self.cls_out,
                self.tok_out,
                prepared.selected,
                self.token_active,
                prepared.slots,
            ).float()
        score_t[:, 0, :] += self.planner.search_score_bias
        flat_scores = score_t.reshape(n, -1)
        candidate_scores = torch.gather(flat_scores, 1, prepared.flat_indices)
        candidate_scores = candidate_scores.masked_fill(~(prepared.valid & torch.isfinite(candidate_scores)), -torch.inf)
        idx = torch.argmax(candidate_scores, dim=1)
        rows = torch.arange(n, device=self.planner.device)
        has_valid = torch.any(torch.isfinite(candidate_scores), dim=1)
        best_actions = prepared.actions[rows, idx]
        best_scores = candidate_scores[rows, idx]
        best_bases = prepared.bases[rows, idx]
        best_sensors = prepared.sensors[rows, idx]
        best_actions = torch.where(has_valid, best_actions, torch.full_like(best_actions, -1))
        best_scores = torch.where(has_valid, best_scores, torch.full_like(best_scores, -torch.inf))
        best_bases = torch.where(has_valid, best_bases, torch.full_like(best_bases, -1))
        best_sensors = torch.where(has_valid, best_sensors, torch.full_like(best_sensors, -1))
        return best_actions, best_scores, best_bases, best_sensors, has_valid

    def score_prepared_prefixes_device(self, prepared: DevicePreparedPrefixBatch) -> BranchExpansionResult:
        n = int(prepared.count)
        if n <= 0:
            empty = np.empty((0,), dtype=np.int64)
            return BranchExpansionResult(
                actions=empty,
                scores=np.empty((0,), dtype=np.float32),
                bases=empty,
                sensors=empty,
                valid=np.zeros((0,), dtype=bool),
                score_tables=np.empty((0, 0, 0), dtype=np.float32),
            )
        with torch.inference_mode():
            best_actions, best_scores, best_bases, best_sensors, has_valid = self._best_from_prepared_device(prepared)
            return BranchExpansionResult(
                actions=best_actions.cpu().numpy().astype(np.int64, copy=False),
                scores=best_scores.cpu().numpy().astype(np.float32, copy=False),
                bases=best_bases.cpu().numpy().astype(np.int64, copy=False),
                sensors=best_sensors.cpu().numpy().astype(np.int64, copy=False),
                valid=has_valid.cpu().numpy().astype(bool, copy=False),
                score_tables=np.empty((0, 0, 0), dtype=np.float32),
            )

    def score_prepared_prefixes_device_graph(self, prepared: DevicePreparedPrefixBatch) -> BranchExpansionResult:
        n = int(prepared.count)
        if n <= 0 or self.planner.device.type != "cuda":
            return self.score_prepared_prefixes_device(prepared)
        try:
            if prepared.graph is None:
                with torch.inference_mode():
                    for _ in range(3):
                        _ = self._best_from_prepared_device(prepared)
                    torch.cuda.synchronize(self.planner.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        (
                            graph_best_actions,
                            graph_best_scores,
                            graph_best_bases,
                            graph_best_sensors,
                            graph_has_valid,
                        ) = self._best_from_prepared_device(prepared)
                prepared.graph = graph
                prepared.graph_best_actions = graph_best_actions
                prepared.graph_best_scores = graph_best_scores
                prepared.graph_best_bases = graph_best_bases
                prepared.graph_best_sensors = graph_best_sensors
                prepared.graph_has_valid = graph_has_valid
            prepared.graph.replay()
            return BranchExpansionResult(
                actions=prepared.graph_best_actions.cpu().numpy().astype(np.int64, copy=False),
                scores=prepared.graph_best_scores.cpu().numpy().astype(np.float32, copy=False),
                bases=prepared.graph_best_bases.cpu().numpy().astype(np.int64, copy=False),
                sensors=prepared.graph_best_sensors.cpu().numpy().astype(np.int64, copy=False),
                valid=prepared.graph_has_valid.cpu().numpy().astype(bool, copy=False),
                score_tables=np.empty((0, 0, 0), dtype=np.float32),
            )
        except Exception:
            return self.score_prepared_prefixes_device(prepared)

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
