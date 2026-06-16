from __future__ import annotations

import time
from collections import defaultdict
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
        profile_values: dict[str, list[float]] | None = None,
    ):
        self.planner = planner
        self._profile_values = profile_values
        self.env_cfg = dict(planner.env_cfg)
        self.obs = attach_env_obs(obs, self.env_cfg, True, True)
        self.budget_ms = float(budget_ms)
        self.adapt = adapter()
        root_tok = tokenize(self.adapt, self.obs, selected=set(), search_count=0).astype(np.float32)
        with torch.inference_mode():
            root_x = torch.from_numpy(root_tok).to(planner.device, dtype=torch.float32).unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=planner.use_amp):
                self.cls_out, self.tok_out, self.root_selected, self.token_active = planner.model.backbone.encode_tokens(root_x)
        self._root_selected_cpu = self.root_selected.squeeze(0).detach().cpu().numpy().astype(bool, copy=True)
        self._root_actions, self._root_bases, self._root_sensors = physical_action_arrays(self.obs, selected=set(), max_trackers=MAXT)
        self._root_width = max(1, int(self._root_actions.size))
        self._root_flat_indices = (self._root_bases.astype(np.int64, copy=False) * 2 + self._root_sensors.astype(np.int64, copy=False)).astype(
            np.int64,
            copy=False,
        )
        self._root_actions_dev = torch.as_tensor(self._root_actions, device=self.planner.device, dtype=torch.long)
        self._root_bases_dev = torch.as_tensor(self._root_bases, device=self.planner.device, dtype=torch.long)
        self._root_sensors_dev = torch.as_tensor(self._root_sensors, device=self.planner.device, dtype=torch.long)
        self._root_flat_indices_dev = torch.as_tensor(self._root_flat_indices, device=self.planner.device, dtype=torch.long)
        self._slot_template = slot_features(self.obs, 0.0, 0, 0, -1, self.budget_ms).astype(np.float32, copy=True)
        if not hasattr(planner, "_prefix_score_graph_cache"):
            planner._prefix_score_graph_cache = {}
        self._prefix_score_graph_cache: dict[tuple, dict[str, object]] = planner._prefix_score_graph_cache

    def _profile_start(self) -> float:
        if self._profile_values is None:
            return 0.0
        if self.planner.device.type == "cuda":
            torch.cuda.synchronize(self.planner.device)
        return time.perf_counter()

    def _profile_end(self, name: str, start: float) -> None:
        if self._profile_values is None:
            return
        if self.planner.device.type == "cuda":
            torch.cuda.synchronize(self.planner.device)
        self._profile_values[name].append((time.perf_counter() - start) * 1000.0)

    def _selected_masks(self, prefixes: list[BranchPrefix]) -> torch.Tensor:
        selected_t, _valid_t = self._prefix_selected_and_valid(prefixes)
        return selected_t

    def _prefix_selected_and_valid(self, prefixes: list[BranchPrefix]) -> tuple[torch.Tensor, np.ndarray]:
        n = len(prefixes)
        rows = int(self._root_selected_cpu.shape[0])
        masks = np.broadcast_to(self._root_selected_cpu[None, :], (n, rows)).copy()
        valid_t = np.ones((n, int(self._root_width)), dtype=bool)
        row_idx: list[int] = []
        col_idx: list[int] = []
        for row, prefix in enumerate(prefixes):
            for base in prefix.selected:
                base_i = int(base)
                if 0 <= base_i < rows:
                    row_idx.append(row)
                    col_idx.append(base_i)
        if row_idx:
            row_arr = np.asarray(row_idx, dtype=np.int64)
            col_arr = np.asarray(col_idx, dtype=np.int64)
            masks[row_arr, col_arr] = True
            track_cols = self._root_bases > 0
            if np.any(track_cols):
                selected_mask = np.zeros((n, int(MAXT) + 1), dtype=bool)
                in_range = (1 <= col_arr) & (col_arr <= int(MAXT))
                if np.any(in_range):
                    selected_mask[row_arr[in_range], col_arr[in_range]] = True
                valid_t[:, track_cols] &= ~selected_mask[:, self._root_bases[track_cols]]
        return torch.as_tensor(masks, device=self.planner.device, dtype=torch.bool), valid_t

    def _slots(self, prefixes: list[BranchPrefix]) -> torch.Tensor:
        n = len(prefixes)
        slots = np.broadcast_to(self._slot_template[None, :], (n, self._slot_template.shape[0])).copy()
        if n:
            elapsed = np.fromiter((float(prefix.elapsed_ms) for prefix in prefixes), dtype=np.float32, count=n)
            search_count = np.fromiter((int(prefix.search_count) for prefix in prefixes), dtype=np.float32, count=n)
            track_count = np.fromiter((int(prefix.track_count) for prefix in prefixes), dtype=np.float32, count=n)
            last_action = np.fromiter((int(prefix.last) for prefix in prefixes), dtype=np.int32, count=n)
            slots[:, 0] = elapsed / float(self.budget_ms)
            slots[:, 1] = search_count / 20.0
            slots[:, 2] = track_count / 100.0
            slots[:, 3] = (last_action == 0).astype(np.float32)
        return torch.from_numpy(slots).to(self.planner.device, dtype=torch.float32)

    def _score_slots(self, selected_t: torch.Tensor, slot_t: torch.Tensor) -> torch.Tensor:
        graph_pack = self._build_prefix_score_graph(selected_t, slot_t)
        if graph_pack is not None:
            static_selected = graph_pack["static_selected"]
            static_slot = graph_pack["static_slot"]
            static_score = graph_pack["static_score"]
            static_selected.copy_(selected_t, non_blocking=False)
            static_slot.copy_(slot_t, non_blocking=False)
            graph_pack["graph"].replay()
            return static_score
        return self._score_slots_raw(selected_t, slot_t)

    def _score_slots_raw(self, selected_t: torch.Tensor, slot_t: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.planner.use_amp):
            return self.planner.score_slots_from_encoded(
                self.cls_out,
                self.tok_out,
                selected_t,
                self.token_active,
                slot_t,
            ).float()

    def _build_prefix_score_graph(self, selected_t: torch.Tensor, slot_t: torch.Tensor):
        if not bool(getattr(self.planner, "use_cuda_graph", False)) or self.planner.device.type != "cuda":
            return None
        key = (
            tuple(selected_t.shape),
            tuple(slot_t.shape),
            tuple(self.cls_out.shape),
            tuple(self.tok_out.shape),
            tuple(self.token_active.shape),
            str(selected_t.dtype),
            str(slot_t.dtype),
            bool(self.planner.use_amp),
        )
        try:
            cache = self._prefix_score_graph_cache.get(key)
            if cache is None:
                static_cls = torch.empty_like(self.cls_out)
                static_tok = torch.empty_like(self.tok_out)
                static_active = torch.empty_like(self.token_active)
                static_selected = torch.empty_like(selected_t)
                static_slot = torch.empty_like(slot_t)
                static_cls.copy_(self.cls_out, non_blocking=False)
                static_tok.copy_(self.tok_out, non_blocking=False)
                static_active.copy_(self.token_active, non_blocking=False)
                static_selected.copy_(selected_t, non_blocking=False)
                static_slot.copy_(slot_t, non_blocking=False)

                def compute_score() -> torch.Tensor:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.planner.use_amp):
                        return self.planner.score_slots_from_encoded(
                            static_cls,
                            static_tok,
                            static_selected,
                            static_active,
                            static_slot,
                        ).float()

                with torch.inference_mode():
                    for _ in range(3):
                        _ = compute_score()
                    torch.cuda.synchronize(self.planner.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        static_score = compute_score()
                cache = {
                    "graph": graph,
                    "static_cls": static_cls,
                    "static_tok": static_tok,
                    "static_active": static_active,
                    "static_selected": static_selected,
                    "static_slot": static_slot,
                    "static_score": static_score,
                }
                self._prefix_score_graph_cache[key] = cache
            else:
                cache["static_cls"].copy_(self.cls_out, non_blocking=False)
                cache["static_tok"].copy_(self.tok_out, non_blocking=False)
                cache["static_active"].copy_(self.token_active, non_blocking=False)
            return cache
        except Exception:
            return None

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
            t0 = self._profile_start()
            selected_t, valid_t = self._prefix_selected_and_valid(prefix_list)
            self._profile_end("scorer_selected_valid", t0)
            t0 = self._profile_start()
            slot_t = self._slots(prefix_list)
            self._profile_end("scorer_slots", t0)
            t0 = self._profile_start()
            score_t = self._score_slots(selected_t, slot_t)
            self._profile_end("scorer_score_slots", t0)
            t0 = self._profile_start()
            score_tables = score_t.float().cpu().numpy()
            self._profile_end("scorer_score_d2h", t0)
        score_tables[:, 0, :] += self.planner.search_score_bias

        t0 = self._profile_start()
        actions_t, bases_t, sensors_t = self._physical_tables_for_prefixes(prefix_list)
        rows = np.arange(len(prefix_list), dtype=np.int64)[:, None]
        vals = score_tables[rows, bases_t, sensors_t]
        vals = np.where(valid_t & np.isfinite(vals), vals, -np.inf)
        pick = np.argmax(vals, axis=1)
        row1 = np.arange(len(prefix_list), dtype=np.int64)
        out_scores = vals[row1, pick].astype(np.float32, copy=False)
        out_valid = np.isfinite(out_scores)
        out_actions = np.where(out_valid, actions_t[row1, pick], -1).astype(np.int64, copy=False)
        out_bases = np.where(out_valid, bases_t[row1, pick], -1).astype(np.int64, copy=False)
        out_sensors = np.where(out_valid, sensors_t[row1, pick], -1).astype(np.int64, copy=False)
        self._profile_end("scorer_cpu_select", t0)
        return BranchExpansionResult(
            actions=out_actions,
            scores=out_scores,
            bases=out_bases,
            sensors=out_sensors,
            valid=out_valid,
            score_tables=np.asarray(score_tables, dtype=np.float32),
        )

    def _valid_table_for_prefixes(self, prefixes: list[BranchPrefix]) -> np.ndarray:
        _selected_t, valid_t = self._prefix_selected_and_valid(prefixes)
        return valid_t

    def _physical_tables_for_prefixes(self, prefixes: list[BranchPrefix]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(prefixes)
        width = int(self._root_width)
        actions_t = np.broadcast_to(self._root_actions[None, :], (n, width)).astype(np.int64, copy=True)
        bases_t = np.broadcast_to(self._root_bases[None, :], (n, width)).astype(np.int64, copy=True)
        sensors_t = np.broadcast_to(self._root_sensors[None, :], (n, width)).astype(np.int64, copy=True)
        return actions_t, bases_t, sensors_t

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
        with torch.inference_mode():
            selected_t, valid_np = self._prefix_selected_and_valid(prefix_list)
            slot_t = self._slots(prefix_list)
            score_t = self._score_slots(selected_t, slot_t)
            score_t[:, 0, :] += self.planner.search_score_bias
            flat_scores = score_t.reshape(len(prefix_list), -1)
            actions_dev = self._root_actions_dev.expand(len(prefix_list), -1)
            bases_dev = self._root_bases_dev.expand(len(prefix_list), -1)
            sensors_dev = self._root_sensors_dev.expand(len(prefix_list), -1)
            flat_idx = self._root_flat_indices_dev.expand(len(prefix_list), -1)
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
        t0 = self._profile_start()
        selected_t, valid_np = self._prefix_selected_and_valid(prefix_list)
        self._profile_end("prepare_selected_valid", t0)
        t0 = self._profile_start()
        slots = self._slots(prefix_list)
        valid = torch.as_tensor(valid_np, device=self.planner.device, dtype=torch.bool)
        self._profile_end("prepare_slots_valid_h2d", t0)
        return DevicePreparedPrefixBatch(
            selected=selected_t,
            slots=slots,
            actions=self._root_actions_dev.expand(n, -1),
            flat_indices=self._root_flat_indices_dev.expand(n, -1),
            bases=self._root_bases_dev.expand(n, -1),
            sensors=self._root_sensors_dev.expand(n, -1),
            valid=valid,
            count=int(n),
        )

    def _best_from_prepared_device(self, prepared: DevicePreparedPrefixBatch, use_prefix_graph: bool = True):
        n = int(prepared.count)
        if use_prefix_graph:
            score_t = self._score_slots(prepared.selected, prepared.slots)
        else:
            score_t = self._score_slots_raw(prepared.selected, prepared.slots)
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
            t0 = self._profile_start()
            best_actions, best_scores, best_bases, best_sensors, has_valid = self._best_from_prepared_device(prepared)
            self._profile_end("prepared_best_device", t0)
            t0 = self._profile_start()
            actions = best_actions.cpu().numpy().astype(np.int64, copy=False)
            scores = best_scores.cpu().numpy().astype(np.float32, copy=False)
            bases = best_bases.cpu().numpy().astype(np.int64, copy=False)
            sensors = best_sensors.cpu().numpy().astype(np.int64, copy=False)
            valid = has_valid.cpu().numpy().astype(bool, copy=False)
            self._profile_end("prepared_d2h", t0)
            return BranchExpansionResult(
                actions=actions,
                scores=scores,
                bases=bases,
                sensors=sensors,
                valid=valid,
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
                        _ = self._best_from_prepared_device(prepared, use_prefix_graph=False)
                    torch.cuda.synchronize(self.planner.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        (
                            graph_best_actions,
                            graph_best_scores,
                            graph_best_bases,
                            graph_best_sensors,
                            graph_has_valid,
                        ) = self._best_from_prepared_device(prepared, use_prefix_graph=False)
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
        if int(top_k) == 1:
            t0 = self._profile_start()
            out = [
                prefix_after_action(self.obs, prefix, int(scored.actions[row]), float(scored.scores[row]))
                for row, prefix in enumerate(prefix_list)
                if bool(scored.valid[row]) and int(scored.actions[row]) >= 0
            ]
            self._profile_end("expand_child_build_top1", t0)
            return out
        out: list[BranchPrefix] = []
        t0 = self._profile_start()
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
        self._profile_end("expand_child_build_topk", t0)
        return out

    def expand_prefixes_top1_device(self, prefixes: Iterable[BranchPrefix]) -> list[BranchPrefix]:
        """Expand each live prefix by its single best valid next action.

        This is the online-friendly top-1 path: the prefix batch is prepared on
        device and valid-action selection stays on GPU, but no CUDA Graph is
        captured because live beam frontiers usually change every depth.
        """
        prefix_list = list(prefixes)
        if not prefix_list:
            return []
        prepared = self.prepare_prefixes_device(prefix_list)
        scored = self.score_prepared_prefixes_device(prepared)
        t0 = self._profile_start()
        out: list[BranchPrefix] = []
        for row, prefix in enumerate(prefix_list):
            if not bool(scored.valid[row]):
                continue
            out.append(prefix_after_action(self.obs, prefix, int(scored.actions[row]), float(scored.scores[row])))
        self._profile_end("expand_child_build_top1", t0)
        return out


class BatchedBeamWindowPlanner:
    """Window planner that expands many partial prefixes per model call."""

    def __init__(
        self,
        planner: FastActionAttentionPlanner,
        beam_width: int = 8,
        branch_top_k: int = 2,
        max_depth: int = 64,
        use_top1_device: bool = False,
    ):
        self.planner = planner
        self.beam_width = int(beam_width)
        self.branch_top_k = int(branch_top_k)
        self.max_depth = int(max_depth)
        self.use_top1_device = bool(use_top1_device)
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

    def _profile_start(self) -> float:
        if not self.profile_enabled:
            return 0.0
        if self.planner.device.type == "cuda":
            torch.cuda.synchronize(self.planner.device)
        return time.perf_counter()

    def _profile_end(self, name: str, start: float) -> None:
        if not self.profile_enabled:
            return
        if self.planner.device.type == "cuda":
            torch.cuda.synchronize(self.planner.device)
        self._profile_values[name].append((time.perf_counter() - start) * 1000.0)

    def plan(self, obs, budget_ms=200):
        t_plan = self._profile_start()
        t0 = self._profile_start()
        profile_values = self._profile_values if self.profile_enabled else None
        scorer = BatchedWindowExpansionScorer(self.planner, obs, budget_ms=float(budget_ms), profile_values=profile_values)
        self._profile_end("beam_scorer_init", t0)
        frontier = [BranchPrefix()]
        best = frontier[0]
        for _depth in range(max(1, self.max_depth)):
            t0 = self._profile_start()
            live = [p for p in frontier if float(p.elapsed_ms) < float(budget_ms)]
            self._profile_end("beam_live_filter", t0)
            if not live:
                break
            t0 = self._profile_start()
            if self.use_top1_device and int(self.branch_top_k) == 1:
                children = scorer.expand_prefixes_top1_device(live)
            else:
                children = scorer.expand_prefixes(live, top_k=max(1, self.branch_top_k))
            self._profile_end("beam_expand_prefixes", t0)
            t0 = self._profile_start()
            children = [p for p in children if p.actions]
            if not children:
                break
            children.sort(key=lambda p: (float(p.score_sum), -float(p.elapsed_ms)), reverse=True)
            frontier = children[: max(1, self.beam_width)]
            if frontier and float(frontier[0].score_sum) >= float(best.score_sum):
                best = frontier[0]
            if best.actions and float(best.elapsed_ms) >= float(budget_ms):
                self._profile_end("beam_filter_sort_prune", t0)
                break
            self._profile_end("beam_filter_sort_prune", t0)
        out = list(best.actions) if best.actions else [int(MAXT) + 3]
        self._profile_end("beam_plan_total", t_plan)
        return out


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
