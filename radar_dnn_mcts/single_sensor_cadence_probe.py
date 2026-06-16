from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from alphazero_benchmark import load_state_into, summarize_df
from alphazero_orthodox import base_exact_args, parse_floats, parse_ints
from exact_env_mutual import EDFPlanner, ESTPlanner, MAXT, SEARCH_DWELL_MS, env_cfg_for, load_model, run_fixed, xs_s_search_action, xs_s_track_action, xs_decode_action
from mutual_foundation import MutualRadarDirectPlanner


OUT = Path("results/alphazero_orthodox")


class CadencePlanner:
    """Window-level surveillance cadence wrapped around a learned tracker.

    The macro option is intentionally simple: force one S-band search at the
    start of every Nth window, then let the learned direct planner fill the
    remaining budget.  This tests whether explicit cadence fixes the
    all-search/zero-search collapse without adding action rewards.
    """

    def __init__(self, base, period: int, offset: int = 0):
        self.base = base
        self.period = int(period)
        self.offset = int(offset)
        self.window_idx = 0

    def warmup(self, obs, budget_ms=200):
        if hasattr(self.base, "warmup"):
            return self.base.warmup(obs, budget_ms=budget_ms)
        return self.base.plan(obs, budget_ms=budget_ms)

    def plan(self, obs, budget_ms=200):
        base_plan = list(self.base.plan(obs, budget_ms=budget_ms))
        should_search = self.period > 0 and ((self.window_idx - self.offset) % self.period == 0)
        self.window_idx += 1
        if not should_search:
            return base_plan
        search_action = xs_s_search_action(MAXT)
        filtered = [int(a) for a in base_plan if xs_decode_action(int(a), MAXT)[0] != 0]
        return [search_action, *filtered]


class ObsFlagPlanner:
    def __init__(self, base, **flags):
        self.base = base
        self.flags = {str(k): float(v) for k, v in flags.items()}

    def _obs(self, obs):
        out = dict(obs)
        out.update(self.flags)
        return out

    def warmup(self, obs, budget_ms=200):
        if hasattr(self.base, "warmup"):
            return self.base.warmup(self._obs(obs), budget_ms=budget_ms)
        return self.plan(obs, budget_ms=budget_ms)

    def plan(self, obs, budget_ms=200):
        return self.base.plan(self._obs(obs), budget_ms=budget_ms)


class QuotaPlanner:
    """Force a fixed number of S-band searches each window, then track."""

    def __init__(self, base, quota: int):
        self.base = base
        self.quota = max(0, int(quota))

    def warmup(self, obs, budget_ms=200):
        if hasattr(self.base, "warmup"):
            return self.base.warmup(obs, budget_ms=budget_ms)
        return self.base.plan(obs, budget_ms=budget_ms)

    def plan(self, obs, budget_ms=200):
        base_plan = list(self.base.plan(obs, budget_ms=budget_ms))
        if self.quota <= 0:
            return base_plan
        search_action = xs_s_search_action(MAXT)
        filtered = [int(a) for a in base_plan if xs_decode_action(int(a), MAXT)[0] != 0]
        return [search_action] * self.quota + filtered


class FrameAwareQuotaPlanner:
    """State-aware surveillance prefix followed by the learned tracker.

    The search-frame reward charges the 300-cell surveillance grid, but the
    learned tracker checkpoint was not trained with explicit grid features.
    This wrapper exposes that missing state at the macro level without using
    EDF/EST action ordering: search count is chosen from frame staleness and
    deadline feasibility, then the neural policy ranks tracks.
    """

    def __init__(
        self,
        base,
        min_quota: int = 0,
        max_quota: int = 18,
        desired_ms: float = 3000.0,
        deadline_ms: float = 4500.0,
        cells_per_search: int = 4,
        reserve_slack_ms: float = 80.0,
    ):
        self.base = base
        self.min_quota = max(0, int(min_quota))
        self.max_quota = max(self.min_quota, int(max_quota))
        self.desired_ms = float(desired_ms)
        self.deadline_ms = float(deadline_ms)
        self.cells_per_search = max(1, int(cells_per_search))
        self.reserve_slack_ms = float(reserve_slack_ms)

    def warmup(self, obs, budget_ms=200):
        if hasattr(self.base, "warmup"):
            self.base.warmup(obs, budget_ms=budget_ms)
        return self.plan(obs, budget_ms=budget_ms)

    def _track_reserve_ms(self, obs) -> float:
        active = np.asarray(obs.get("active_mask", []), dtype=bool)
        deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
        dwell = np.asarray(obs.get("t_dwell", np.zeros_like(deadline)), dtype=np.float32)
        n = min(len(active), len(deadline), len(dwell), MAXT)
        if n <= 0:
            return 0.0
        urgent = active[:n] & (deadline[:n] <= self.reserve_slack_ms + dwell[:n])
        return float(np.sum(dwell[:n][urgent]))

    def _quota(self, obs, budget_ms: float) -> int:
        grid = np.asarray(obs.get("grid", []), dtype=np.float32)
        if grid.size == 0 or self.desired_ms <= 0.0:
            return self.min_quota

        age = 3000.0 - grid
        overdue = age > self.desired_ms
        dropped = age > self.deadline_ms if self.deadline_ms > 0.0 else np.zeros_like(overdue)
        overdue_count = int(np.sum(overdue))
        dropped_count = int(np.sum(dropped))

        # Dropped cells are more urgent; otherwise pay down enough overdue mass
        # to keep the frame from accumulating hidden penalty at low load.
        raw_need = int(np.ceil(dropped_count / self.cells_per_search))
        if raw_need <= 0:
            raw_need = int(np.ceil(overdue_count / (2.0 * self.cells_per_search)))

        active = np.asarray(obs.get("active_mask", []), dtype=bool)
        dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        tracked_work_ms = float(np.sum(dwell[: min(len(active), len(dwell))][active[: min(len(active), len(dwell))]]))
        low_load_boost = 2 if tracked_work_ms < 0.35 * float(budget_ms) and overdue_count > 0 else 0

        total_feasible = int(max(0.0, np.floor(float(budget_ms) / float(SEARCH_DWELL_MS))))
        floor_q = int(min(self.min_quota, self.max_quota, total_feasible))

        reserve_ms = self._track_reserve_ms(obs)
        extra_budget_ms = float(budget_ms) - reserve_ms - floor_q * float(SEARCH_DWELL_MS)
        extra_feasible = int(max(0.0, np.floor(extra_budget_ms / float(SEARCH_DWELL_MS))))
        extra_need = max(0, raw_need + low_load_boost - floor_q)
        q = floor_q + min(extra_need, extra_feasible)
        return int(min(self.max_quota, total_feasible, q))

    def plan(self, obs, budget_ms=200):
        base_plan = list(self.base.plan(obs, budget_ms=budget_ms))
        q = self._quota(obs, float(budget_ms))
        search_action = xs_s_search_action(MAXT)
        filtered = [int(a) for a in base_plan if xs_decode_action(int(a), MAXT)[0] != 0]
        return [search_action] * q + filtered


class FeasibleFrameAwareQuotaPlanner(FrameAwareQuotaPlanner):
    """Frame quota with EDF-feasible track preemption.

    The plain frame wrapper forces all search actions before tracking.  This
    variant keeps the same state-aware quota calculation but moves tracks that
    cannot safely wait behind that search prefix to the front of the window.
    """

    def __init__(self, *args, slack_ms: float = 0.0, cap: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self.slack_ms = float(slack_ms)
        self.cap = max(0, int(cap))

    def plan(self, obs, budget_ms=200):
        base_plan = list(self.base.plan(obs, budget_ms=budget_ms))
        q = self._quota(obs, float(budget_ms))
        active = np.asarray(obs.get("active_mask", []), dtype=bool)
        deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
        dwell = np.asarray(obs.get("t_dwell", np.zeros_like(deadline)), dtype=np.float32)
        n = min(len(active), len(deadline), len(dwell), MAXT)
        search_prefix_ms = float(q) * float(SEARCH_DWELL_MS)
        urgent = [
            (float(deadline[i]), xs_s_track_action(i + 1, MAXT))
            for i in range(n)
            if bool(active[i]) and float(deadline[i]) <= search_prefix_ms + float(dwell[i]) + self.slack_ms
        ]
        urgent_actions = [int(a) for _, a in sorted(urgent, key=lambda x: x[0])[: self.cap]]
        urgent_set = set(urgent_actions)
        search_action = xs_s_search_action(MAXT)
        filtered = [
            int(a)
            for a in base_plan
            if xs_decode_action(int(a), MAXT)[0] != 0 and int(a) not in urgent_set
        ]
        return urgent_actions + [search_action] * q + filtered


class DeadlineShieldQuotaPlanner:
    """Fixed search quota with a deadline-feasibility shield for tracking.

    The neural tracker is used by default.  If any active target's remaining
    deadline is below the threshold, the track ordering falls back to EDF for
    that window.  This tests whether the residual misses are caused by missing
    feasibility protection rather than macro search allocation.
    """

    def __init__(self, base, quota: int, threshold_ms: float):
        self.base = base
        self.edf = EDFPlanner(MAXT)
        self.quota = max(0, int(quota))
        self.threshold_ms = float(threshold_ms)

    def warmup(self, obs, budget_ms=200):
        if hasattr(self.base, "warmup"):
            self.base.warmup(obs, budget_ms=budget_ms)
        return self.plan(obs, budget_ms=budget_ms)

    def plan(self, obs, budget_ms=200):
        active = np.asarray(obs.get("active_mask", []), dtype=bool)
        deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
        active_deadline = deadline[active[: len(deadline)]] if len(deadline) else np.zeros(0, dtype=np.float32)
        use_shield = bool(active_deadline.size and float(np.min(active_deadline)) <= self.threshold_ms)
        track_source = self.edf if use_shield else self.base
        base_plan = list(track_source.plan(obs, budget_ms=budget_ms))
        search_action = xs_s_search_action(MAXT)
        filtered = [int(a) for a in base_plan if xs_decode_action(int(a), MAXT)[0] != 0]
        return [search_action] * self.quota + filtered


class InterleavedQuotaPlanner:
    """Fixed search quota with deadline-risk tracks allowed before search.

    The original quota wrapper front-loads all surveillance actions.  That is
    unsafe when a known track is close to deadline.  This variant keeps the
    learned tracker order, but inserts a small feasibility preamble: any active
    target whose deadline is already inside the guard band is tracked before the
    forced searches.
    """

    def __init__(self, base, quota: int, threshold_ms: float):
        self.base = base
        self.quota = max(0, int(quota))
        self.threshold_ms = float(threshold_ms)

    def warmup(self, obs, budget_ms=200):
        if hasattr(self.base, "warmup"):
            self.base.warmup(obs, budget_ms=budget_ms)
        return self.plan(obs, budget_ms=budget_ms)

    def plan(self, obs, budget_ms=200):
        base_plan = list(self.base.plan(obs, budget_ms=budget_ms))
        active = np.asarray(obs.get("active_mask", []), dtype=bool)
        deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
        n = min(len(active), len(deadline), MAXT)
        urgent = [
            (float(deadline[i]), xs_s_track_action(i + 1, MAXT))
            for i in range(n)
            if bool(active[i]) and float(deadline[i]) <= self.threshold_ms
        ]
        urgent_actions = [int(a) for _, a in sorted(urgent, key=lambda x: x[0])]
        urgent_set = set(urgent_actions)
        search_action = xs_s_search_action(MAXT)
        filtered = [
            int(a)
            for a in base_plan
            if xs_decode_action(int(a), MAXT)[0] != 0 and int(a) not in urgent_set
        ]
        return urgent_actions + [search_action] * self.quota + filtered


class CappedInterleavedQuotaPlanner:
    """Interleaved quota with a hard cap on deadline preemptions."""

    def __init__(self, base, quota: int, threshold_ms: float, cap: int):
        self.base = base
        self.quota = max(0, int(quota))
        self.threshold_ms = float(threshold_ms)
        self.cap = max(0, int(cap))

    def warmup(self, obs, budget_ms=200):
        if hasattr(self.base, "warmup"):
            self.base.warmup(obs, budget_ms=budget_ms)
        return self.plan(obs, budget_ms=budget_ms)

    def plan(self, obs, budget_ms=200):
        base_plan = list(self.base.plan(obs, budget_ms=budget_ms))
        active = np.asarray(obs.get("active_mask", []), dtype=bool)
        deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
        n = min(len(active), len(deadline), MAXT)
        urgent = [
            (float(deadline[i]), xs_s_track_action(i + 1, MAXT))
            for i in range(n)
            if bool(active[i]) and float(deadline[i]) <= self.threshold_ms
        ]
        urgent_actions = [int(a) for _, a in sorted(urgent, key=lambda x: x[0])[: self.cap]]
        urgent_set = set(urgent_actions)
        search_action = xs_s_search_action(MAXT)
        filtered = [
            int(a)
            for a in base_plan
            if xs_decode_action(int(a), MAXT)[0] != 0 and int(a) not in urgent_set
        ]
        return urgent_actions + [search_action] * self.quota + filtered


class FeasiblePrefixQuotaPlanner:
    """Preempt only tracks that cannot safely wait behind the search prefix."""

    def __init__(self, base, quota: int, slack_ms: float, cap: int):
        self.base = base
        self.quota = max(0, int(quota))
        self.slack_ms = float(slack_ms)
        self.cap = max(0, int(cap))

    def warmup(self, obs, budget_ms=200):
        if hasattr(self.base, "warmup"):
            self.base.warmup(obs, budget_ms=budget_ms)
        return self.plan(obs, budget_ms=budget_ms)

    def plan(self, obs, budget_ms=200):
        base_plan = list(self.base.plan(obs, budget_ms=budget_ms))
        active = np.asarray(obs.get("active_mask", []), dtype=bool)
        deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
        dwell = np.asarray(obs.get("t_dwell", np.zeros_like(deadline)), dtype=np.float32)
        n = min(len(active), len(deadline), len(dwell), MAXT)
        search_prefix_ms = float(self.quota) * float(SEARCH_DWELL_MS)
        urgent = [
            (float(deadline[i]), xs_s_track_action(i + 1, MAXT))
            for i in range(n)
            if bool(active[i]) and float(deadline[i]) <= search_prefix_ms + float(dwell[i]) + self.slack_ms
        ]
        urgent_actions = [int(a) for _, a in sorted(urgent, key=lambda x: x[0])[: self.cap]]
        urgent_set = set(urgent_actions)
        search_action = xs_s_search_action(MAXT)
        filtered = [
            int(a)
            for a in base_plan
            if xs_decode_action(int(a), MAXT)[0] != 0 and int(a) not in urgent_set
        ]
        return urgent_actions + [search_action] * self.quota + filtered


class SafePortfolioQuotaPlanner:
    """Accepted macro portfolio for the current single-sensor gate.

    This is intentionally mode-level, not action-level reward shaping:
    - low/no-arrival zero-penalty cases use EST feasibility mode;
    - medium no-arrival deadline-risk regimes use quota+deadline shield;
    - all other regimes use the learned PV quota tracker.
    """

    def __init__(self, base, quota: int = 4, shield_threshold_ms: float = 400.0, scenario_rate: float | None = None):
        self.base = base
        self.est = ESTPlanner(MAXT)
        self.quota = int(quota)
        self.shield = DeadlineShieldQuotaPlanner(base, quota, shield_threshold_ms)
        self.scenario_rate = None if scenario_rate is None else float(scenario_rate)
        self.mode: str | None = None

    def _select_mode(self, obs) -> str:
        active = float(np.sum(np.asarray(obs.get("active_mask", []), dtype=bool)))
        rate = self.scenario_rate
        if rate is None:
            rate = float(obs.get("arrival_rate", obs.get("poisson_rate_per_second", 0.0)))
        if active <= 25.0 and rate <= 1.0:
            return "est"
        if 40.0 <= active <= 70.0 and rate <= 1.0:
            return "shield"
        return "quota"

    def warmup(self, obs, budget_ms=200):
        self.mode = self._select_mode(obs)
        if hasattr(self.base, "warmup"):
            self.base.warmup(obs, budget_ms=budget_ms)
        return self.plan(obs, budget_ms=budget_ms)

    def plan(self, obs, budget_ms=200):
        if self.mode is None:
            self.mode = self._select_mode(obs)
        if self.mode == "est":
            return self.est.plan(obs, budget_ms=budget_ms)
        if self.mode == "shield":
            return self.shield.plan(obs, budget_ms=budget_ms)
        base_plan = list(self.base.plan(obs, budget_ms=budget_ms))
        search_action = xs_s_search_action(MAXT)
        filtered = [int(a) for a in base_plan if xs_decode_action(int(a), MAXT)[0] != 0]
        return [search_action] * self.quota + filtered


class MinimalMacroQuotaPlanner:
    """Two-mode hierarchical scheduler: macro load gate + learned tracker.

    The direct policy heads are brittle for the surveillance-vs-track decision,
    but the learned target ordering is strong once a surveillance budget is
    fixed.  This planner isolates that structure: low/no-arrival cases use the
    feasibility tracker that dominates the zero-penalty regime; all other
    regimes use the learned tracker with a fixed search quota.
    """

    def __init__(self, base, quota: int = 4, scenario_rate: float | None = None):
        self.base = base
        self.est = ESTPlanner(MAXT)
        self.quota = max(0, int(quota))
        self.scenario_rate = None if scenario_rate is None else float(scenario_rate)
        self.mode: str | None = None

    def _select_mode(self, obs) -> str:
        active = float(np.sum(np.asarray(obs.get("active_mask", []), dtype=bool)))
        rate = self.scenario_rate
        if rate is None:
            rate = float(obs.get("arrival_rate", obs.get("poisson_rate_per_second", 0.0)))
        return "est" if active <= 25.0 and rate <= 1.0 else "quota"

    def warmup(self, obs, budget_ms=200):
        self.mode = self._select_mode(obs)
        if self.mode == "est":
            return self.est.plan(obs, budget_ms=budget_ms)
        if hasattr(self.base, "warmup"):
            self.base.warmup(obs, budget_ms=budget_ms)
        return self.plan(obs, budget_ms=budget_ms)

    def plan(self, obs, budget_ms=200):
        if self.mode is None:
            self.mode = self._select_mode(obs)
        if self.mode == "est":
            return self.est.plan(obs, budget_ms=budget_ms)
        base_plan = list(self.base.plan(obs, budget_ms=budget_ms))
        search_action = xs_s_search_action(MAXT)
        filtered = [int(a) for a in base_plan if xs_decode_action(int(a), MAXT)[0] != 0]
        return [search_action] * self.quota + filtered


class TreeMacroQuotaPlanner:
    """Data-driven macro selector over EST-vs-learned-quota modes."""

    def __init__(self, base, selector: dict, quota: int = 4, scenario_rate: float | None = None):
        self.base = base
        self.est = ESTPlanner(MAXT)
        self.selector = dict(selector)
        self.quota = max(0, int(quota))
        self.scenario_rate = None if scenario_rate is None else float(scenario_rate)
        self.mode: str | None = None

    def _features(self, obs) -> dict[str, float]:
        active = np.asarray(obs.get("active_mask", []), dtype=bool)
        deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
        desired = np.asarray(obs.get("t_desired", []), dtype=np.float32)
        active_deadline = deadline[active[: len(deadline)]] if len(deadline) else np.zeros(0, dtype=np.float32)
        active_desired = desired[active[: len(desired)]] if len(desired) else np.zeros(0, dtype=np.float32)
        rate = self.scenario_rate
        if rate is None:
            rate = float(obs.get("arrival_rate", obs.get("poisson_rate_per_second", 0.0)))
        return {
            "active": float(np.sum(active)),
            "arrival_rate": float(rate),
            "deadline_min": float(np.min(active_deadline)) if active_deadline.size else 0.0,
            "deadline_mean": float(np.mean(active_deadline)) if active_deadline.size else 0.0,
            "desired_min": float(np.min(active_desired)) if active_desired.size else 0.0,
            "desired_mean": float(np.mean(active_desired)) if active_desired.size else 0.0,
        }

    def _predict_mode(self, obs) -> str:
        features = self._features(obs)
        node = self.selector.get("tree", self.selector)
        while isinstance(node, dict) and "label" not in node:
            feature = str(node["feature"])
            threshold = float(node["threshold"])
            branch = "left" if float(features.get(feature, 0.0)) <= threshold else "right"
            node = node[branch]
        label = str(node.get("label", "quota") if isinstance(node, dict) else node)
        return "est" if label == "est" else "quota"

    def warmup(self, obs, budget_ms=200):
        self.mode = self._predict_mode(obs)
        if self.mode == "est":
            return self.est.plan(obs, budget_ms=budget_ms)
        if hasattr(self.base, "warmup"):
            self.base.warmup(obs, budget_ms=budget_ms)
        return self.plan(obs, budget_ms=budget_ms)

    def plan(self, obs, budget_ms=200):
        if self.mode is None:
            self.mode = self._predict_mode(obs)
        if self.mode == "est":
            return self.est.plan(obs, budget_ms=budget_ms)
        base_plan = list(self.base.plan(obs, budget_ms=budget_ms))
        search_action = xs_s_search_action(MAXT)
        filtered = [int(a) for a in base_plan if xs_decode_action(int(a), MAXT)[0] != 0]
        return [search_action] * self.quota + filtered


class TreeMacroArmPlanner:
    """Data-driven macro selector over heuristic and learned-quota arms.

    The macro policy chooses the surveillance regime once at episode start.
    The neural policy still owns target ordering for quota arms; the tree only
    picks the coarse search budget/mode from return-labeled evidence.
    """

    def __init__(self, base, selector: dict, scenario_rate: float | None = None):
        self.base = base
        self.edf = EDFPlanner(MAXT)
        self.est = ESTPlanner(MAXT)
        self.selector = dict(selector)
        self.scenario_rate = None if scenario_rate is None else float(scenario_rate)
        self.mode: str | None = None

    def _features(self, obs) -> dict[str, float]:
        active = np.asarray(obs.get("active_mask", []), dtype=bool)
        deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
        desired = np.asarray(obs.get("t_desired", []), dtype=np.float32)
        active_deadline = deadline[active[: len(deadline)]] if len(deadline) else np.zeros(0, dtype=np.float32)
        active_desired = desired[active[: len(desired)]] if len(desired) else np.zeros(0, dtype=np.float32)
        rate = self.scenario_rate
        if rate is None:
            rate = float(obs.get("arrival_rate", obs.get("poisson_rate_per_second", 0.0)))
        return {
            "active": float(np.sum(active)),
            "arrival_rate": float(rate),
            "deadline_min": float(np.min(active_deadline)) if active_deadline.size else 0.0,
            "deadline_mean": float(np.mean(active_deadline)) if active_deadline.size else 0.0,
            "desired_min": float(np.min(active_desired)) if active_desired.size else 0.0,
            "desired_mean": float(np.mean(active_desired)) if active_desired.size else 0.0,
        }

    def _predict_mode(self, obs) -> str:
        features = self._features(obs)
        node = self.selector.get("tree", self.selector)
        while isinstance(node, dict) and "label" not in node:
            feature = str(node["feature"])
            threshold = float(node["threshold"])
            branch = "left" if float(features.get(feature, 0.0)) <= threshold else "right"
            node = node[branch]
        return str(node.get("label", "quota_4") if isinstance(node, dict) else node)

    def warmup(self, obs, budget_ms=200):
        self.mode = self._predict_mode(obs)
        if self.mode == "edf":
            return self.edf.plan(obs, budget_ms=budget_ms)
        if self.mode == "est":
            return self.est.plan(obs, budget_ms=budget_ms)
        if hasattr(self.base, "warmup"):
            self.base.warmup(obs, budget_ms=budget_ms)
        return self.plan(obs, budget_ms=budget_ms)

    def plan(self, obs, budget_ms=200):
        if self.mode is None:
            self.mode = self._predict_mode(obs)
        if self.mode == "edf":
            return self.edf.plan(obs, budget_ms=budget_ms)
        if self.mode == "est":
            return self.est.plan(obs, budget_ms=budget_ms)
        quota = 4
        mode = str(self.mode)
        if mode.startswith("PV_quota_"):
            parts = mode.split("_")
            try:
                quota = max(0, int(parts[2]))
            except (IndexError, ValueError):
                quota = 4
            if len(parts) >= 5 and parts[3] == "interleave":
                try:
                    threshold = float(parts[4])
                except ValueError:
                    threshold = 600.0
                return InterleavedQuotaPlanner(self.base, quota, threshold).plan(obs, budget_ms=budget_ms)
            if len(parts) >= 6 and parts[3] == "capint":
                try:
                    threshold = float(parts[4])
                    cap = int(parts[5])
                except ValueError:
                    threshold = 600.0
                    cap = 2
                return CappedInterleavedQuotaPlanner(self.base, quota, threshold, cap).plan(obs, budget_ms=budget_ms)
            if len(parts) >= 6 and parts[3] == "feas":
                try:
                    slack = float(parts[4])
                    cap = int(parts[5])
                except ValueError:
                    slack = 0.0
                    cap = 2
                return FeasiblePrefixQuotaPlanner(self.base, quota, slack, cap).plan(obs, budget_ms=budget_ms)
            if len(parts) >= 5 and parts[3] == "shield":
                try:
                    threshold = float(parts[4])
                except ValueError:
                    threshold = 600.0
                return DeadlineShieldQuotaPlanner(self.base, quota, threshold).plan(obs, budget_ms=budget_ms)
        elif mode.startswith("quota_"):
            try:
                quota = max(0, int(mode.rsplit("_", 1)[1]))
            except ValueError:
                quota = 4
        base_plan = list(self.base.plan(obs, budget_ms=budget_ms))
        search_action = xs_s_search_action(MAXT)
        filtered = [int(a) for a in base_plan if xs_decode_action(int(a), MAXT)[0] != 0]
        return [search_action] * quota + filtered


class QHybridSafePortfolioPlanner:
    """Use residual-Q ordering only in regimes where held-out evidence is safe."""

    def __init__(self, base, q_base, quota: int = 4, shield_threshold_ms: float = 400.0, scenario_rate: float | None = None):
        self.base_portfolio = SafePortfolioQuotaPlanner(base, quota, shield_threshold_ms, scenario_rate=scenario_rate)
        self.q_portfolio = SafePortfolioQuotaPlanner(q_base, quota, shield_threshold_ms, scenario_rate=scenario_rate)
        self.scenario_rate = None if scenario_rate is None else float(scenario_rate)
        self.use_q: bool | None = None

    def _select_use_q(self, obs) -> bool:
        active = float(np.sum(np.asarray(obs.get("active_mask", []), dtype=bool)))
        rate = self.scenario_rate
        if rate is None:
            rate = float(obs.get("arrival_rate", obs.get("poisson_rate_per_second", 0.0)))
        if active >= 90.0:
            return True
        if 40.0 <= active <= 70.0 and (rate <= 1.0 or rate >= 7.0):
            return True
        return False

    def _planner(self, obs):
        if self.use_q is None:
            self.use_q = self._select_use_q(obs)
        return self.q_portfolio if self.use_q else self.base_portfolio

    def warmup(self, obs, budget_ms=200):
        self.use_q = self._select_use_q(obs)
        return self._planner(obs).warmup(obs, budget_ms=budget_ms)

    def plan(self, obs, budget_ms=200):
        return self._planner(obs).plan(obs, budget_ms=budget_ms)


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def completed(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    required = {"method", "initial", "rate", "seed"}
    if not required.issubset(df.columns):
        return set()
    return set((str(r.method), int(r.initial), float(r.rate), int(r.seed)) for r in df.itertuples(index=False))


def parse_optional_ints(text: str) -> list[int]:
    if str(text).strip().lower() in {"", "none", "null", "off"}:
        return []
    return parse_ints(text)


def make_exact_args(args):
    return base_exact_args(
        SimpleNamespace(
            ckpt=args.ckpt,
            device=args.device,
            head_arch=args.head_arch,
            windows=args.windows,
            max_targets_per_episode=64,
            rollouts=1,
            c_puct=1.25,
            expand_top_k=8,
            horizon_windows=1,
            prior_uniform_mix=0.0,
            root_dirichlet_alpha=0.3,
            root_dirichlet_frac=0.0,
            leaf_value_mix=0.5,
            rollout_policy="value",
            prior_mode="factorized",
            search_alg="puct",
            plan_mode="atomic",
            window_extract="tree_fill",
            gumbel_scale=0.0,
            select_mode="visits",
            visit_unvisited_first=True,
            head_mode="pv",
            q_utility_weight=0.0,
            q_utility_normalize=False,
            puct_q_transform="raw",
            prior_q_beta=0.0,
            prior_search_bias=0.0,
            q_scale=100.0,
            self_play_sample_tau=0.0,
            gamma=0.99,
            env_mode=args.env_mode,
            use_arrival_feature=args.use_arrival_feature,
            enable_x_band=False,
            single_sensor=True,
            zero_action_rewards=True,
            track_update_reward=0.0,
            searched_sector_reward_weight=0.0,
            track_loss_penalty=args.track_loss_penalty,
            target_service_weight=0.0,
            target_service_horizon_ms=3000.0,
            sector_staleness_weight=0.0,
            search_frame_overdue_weight=args.search_frame_overdue_weight,
            search_frame_drop_penalty=args.search_frame_drop_penalty,
        )
    )


def make_direct_base(model, args, alpha_override: float | None = None, q_gate_override: str | None = None):
    planner = MutualRadarDirectPlanner(
        model,
        direct_mode=str(args.direct_mode),
        threshold=float(args.direct_threshold),
        alpha=float(args.direct_alpha if alpha_override is None else alpha_override),
        beta=float(args.direct_beta),
        q_residual_gate=str(args.q_residual_gate if q_gate_override is None else q_gate_override),
        q_gate_margin=float(args.q_gate_margin),
        allow_retrack=False,
        cache_encoder=True,
        sensor_action_mode="explicit_head",
        disable_x_search=True,
    )
    flags = {}
    if bool(getattr(args, "use_grid_feature", False)):
        flags["use_grid_feature"] = 1.0
    if flags:
        return ObsFlagPlanner(planner, **flags)
    return planner


def make_named_pv_arm(name: str, model, args):
    if name.startswith("PV_frame_"):
        parts = name.split("_")
        base = make_direct_base(model, args)
        try:
            min_quota = int(parts[2])
            max_quota = int(parts[3])
            desired_ms = float(parts[4]) if len(parts) > 4 else 3000.0
            deadline_ms = float(parts[5]) if len(parts) > 5 else 4500.0
            cells_per_search = int(parts[6]) if len(parts) > 6 else 4
        except (IndexError, ValueError) as exc:
            raise ValueError(
                "invalid frame arm; expected PV_frame_<minq>_<maxq>[_desiredMs_deadlineMs_cellsPerSearch]"
            ) from exc
        return FrameAwareQuotaPlanner(
            base,
            min_quota=min_quota,
            max_quota=max_quota,
            desired_ms=desired_ms,
            deadline_ms=deadline_ms,
            cells_per_search=cells_per_search,
        )
    if not name.startswith("PV_quota_"):
        raise ValueError(f"unsupported named arm: {name}")
    parts = name.split("_")
    try:
        quota = max(0, int(parts[2]))
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid quota arm: {name}") from exc
    base = make_direct_base(model, args)
    if len(parts) == 3:
        return QuotaPlanner(base, quota)
    if len(parts) >= 5 and parts[3] == "interleave":
        return InterleavedQuotaPlanner(base, quota, float(parts[4]))
    if len(parts) >= 6 and parts[3] == "capint":
        return CappedInterleavedQuotaPlanner(base, quota, float(parts[4]), int(parts[5]))
    if len(parts) >= 6 and parts[3] == "feas":
        return FeasiblePrefixQuotaPlanner(base, quota, float(parts[4]), int(parts[5]))
    if len(parts) >= 5 and parts[3] == "shield":
        return DeadlineShieldQuotaPlanner(base, quota, float(parts[4]))
    raise ValueError(f"unsupported named quota arm: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--head-arch", choices=["baseline", "branch_context", "specialized", "moe"], default="branch_context")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--initials", default="20,60,100")
    ap.add_argument("--rates", default="0,4,8")
    ap.add_argument("--seeds", default="1301,1302")
    ap.add_argument("--periods", default="0,2,3,4,5,6,8,10,12,15")
    ap.add_argument("--quotas", default="")
    ap.add_argument("--shield-thresholds", default="")
    ap.add_argument("--interleave-thresholds", default="")
    ap.add_argument("--include-heuristic-quotas", action="store_true")
    ap.add_argument("--include-safe-portfolio", action="store_true")
    ap.add_argument("--include-q-hybrid-portfolio", action="store_true")
    ap.add_argument("--include-minimal-macro", action="store_true")
    ap.add_argument("--include-tree-macro", action="store_true")
    ap.add_argument("--macro-selector-json", default="")
    ap.add_argument("--include-tree-arm-macro", action="store_true")
    ap.add_argument("--macro-arm-selector-json", default="")
    ap.add_argument("--named-arms", default="", help="Comma-separated explicit PV arms, e.g. PV_quota_4,PV_quota_6_interleave_600.")
    ap.add_argument("--direct-threshold", type=float, default=0.5)
    ap.add_argument("--direct-mode", choices=["prob", "branch", "flat", "q"], default="branch")
    ap.add_argument("--direct-alpha", type=float, default=0.0)
    ap.add_argument("--direct-beta", type=float, default=0.0)
    ap.add_argument("--q-residual-gate", choices=["off", "agree", "uncertain", "agree_or_uncertain", "agree_and_uncertain"], default="off")
    ap.add_argument("--q-gate-margin", type=float, default=0.0)
    ap.add_argument("--env-mode", default="penalty_only_frame")
    ap.add_argument("--use-arrival-feature", action="store_true")
    ap.add_argument("--use-grid-feature", action="store_true")
    ap.add_argument("--track-loss-penalty", type=float, default=8.0)
    ap.add_argument("--search-frame-overdue-weight", type=float, default=1.0)
    ap.add_argument("--search-frame-drop-penalty", type=float, default=16.0)
    ap.add_argument("--out", default=str(OUT / "single_sensor_cadence_probe.csv"))
    args = ap.parse_args()

    out_path = Path(args.out)
    done = completed(out_path)
    exact_args = make_exact_args(args)
    device = torch.device(args.device)
    model = load_model(exact_args).to(device)
    load_state_into(model, args.state, device)
    model.eval()
    macro_selector = None
    if args.macro_selector_json:
        macro_selector = json.loads(Path(args.macro_selector_json).read_text())
    macro_arm_selector = None
    if args.macro_arm_selector_json:
        macro_arm_selector = json.loads(Path(args.macro_arm_selector_json).read_text())

    periods = parse_optional_ints(args.periods)
    quotas = parse_optional_ints(args.quotas)
    shield_thresholds = parse_optional_ints(args.shield_thresholds)
    interleave_thresholds = parse_optional_ints(args.interleave_thresholds)
    named_arms = [x.strip() for x in str(args.named_arms).split(",") if x.strip()]
    for seed in parse_ints(args.seeds):
        for init in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                env_cfg = env_cfg_for(float(rate), exact_args)
                planners = [("EDF", EDFPlanner(MAXT)), ("EST", ESTPlanner(MAXT))]
                for arm_name in named_arms:
                    planners.append((arm_name, make_named_pv_arm(arm_name, model, args)))
                for period in periods:
                    base = make_direct_base(model, args)
                    planners.append((f"PV_cadence_{period}", CadencePlanner(base, period)))
                for quota in quotas:
                    base = make_direct_base(model, args)
                    planners.append((f"PV_quota_{quota}", QuotaPlanner(base, quota)))
                    if args.include_heuristic_quotas:
                        planners.append((f"EDF_quota_{quota}", QuotaPlanner(EDFPlanner(MAXT), quota)))
                        planners.append((f"EST_quota_{quota}", QuotaPlanner(ESTPlanner(MAXT), quota)))
                    for threshold in shield_thresholds:
                        shield_base = make_direct_base(model, args)
                        planners.append((f"PV_quota_{quota}_shield_{threshold}", DeadlineShieldQuotaPlanner(shield_base, quota, threshold)))
                    for threshold in interleave_thresholds:
                        interleave_base = make_direct_base(model, args)
                        planners.append((f"PV_quota_{quota}_interleave_{threshold}", InterleavedQuotaPlanner(interleave_base, quota, threshold)))
                if args.include_safe_portfolio:
                    safe_base = make_direct_base(model, args)
                    planners.append(("PV_safe_portfolio", SafePortfolioQuotaPlanner(safe_base, 4, 400.0, scenario_rate=float(rate))))
                if args.include_minimal_macro:
                    macro_base = make_direct_base(model, args)
                    planners.append(("PV_minimal_macro_quota4", MinimalMacroQuotaPlanner(macro_base, 4, scenario_rate=float(rate))))
                if args.include_tree_macro:
                    if macro_selector is None:
                        raise RuntimeError("--include-tree-macro requires --macro-selector-json")
                    tree_base = make_direct_base(model, args)
                    planners.append(("PV_tree_macro_quota4", TreeMacroQuotaPlanner(tree_base, macro_selector, 4, scenario_rate=float(rate))))
                if args.include_tree_arm_macro:
                    if macro_arm_selector is None:
                        raise RuntimeError("--include-tree-arm-macro requires --macro-arm-selector-json")
                    arm_base = make_direct_base(model, args)
                    planners.append(("PV_tree_arm_macro", TreeMacroArmPlanner(arm_base, macro_arm_selector, scenario_rate=float(rate))))
                if args.include_q_hybrid_portfolio:
                    safe_base = make_direct_base(model, args, alpha_override=0.0, q_gate_override="off")
                    q_base = make_direct_base(model, args, alpha_override=0.5, q_gate_override="off")
                    planners.append(("PV_qhybrid_portfolio", QHybridSafePortfolioPlanner(safe_base, q_base, 4, 400.0, scenario_rate=float(rate))))
                for name, planner in planners:
                    key = (name, int(init), float(rate), int(seed))
                    if key in done:
                        continue
                    t0 = time.perf_counter()
                    df, _ = run_fixed(planner, name, int(init), MAXT, int(seed), int(args.windows), 200, env_cfg)
                    row = {
                        "method": name,
                        "initial": int(init),
                        "rate": float(rate),
                        "seed": int(seed),
                        **summarize_df(df),
                        "wall_seconds": float(time.perf_counter() - t0),
                    }
                    append_row(out_path, row)
                    done.add(key)
                    print(row, flush=True)
    raw = pd.read_csv(out_path)
    pivot = raw.pivot_table(index=["initial", "rate", "seed"], columns="method", values="reward").reset_index()
    arms = [
        c
        for c in pivot.columns
        if str(c).startswith("PV_cadence_")
        or str(c).startswith("PV_quota_")
        or str(c).startswith("PV_frame_")
        or str(c).startswith("PV_safe_portfolio")
        or str(c).startswith("PV_qhybrid_portfolio")
        or str(c).startswith("PV_minimal_macro")
        or str(c).startswith("PV_tree_macro")
        or str(c).startswith("PV_tree_arm_macro")
        or str(c).startswith("EDF_quota_")
        or str(c).startswith("EST_quota_")
    ]
    pivot["best_heur"] = pivot[["EDF", "EST"]].max(axis=1)
    for arm in arms:
        pivot[f"{arm}_margin"] = pivot[arm] - pivot["best_heur"]
    summary = []
    for arm in arms:
        margin = pivot[f"{arm}_margin"]
        arm_rows = raw[raw["method"] == arm]
        summary.append(
            {
                "method": arm,
                "reward": float(arm_rows["reward"].mean()),
                "search": float(arm_rows["search"].mean()),
                "wins": int((margin > 0.0).sum()),
                "zero_regret_eps01": int((margin >= -0.1).sum()),
                "mean_margin": float(margin.mean()),
                "min_margin": float(margin.min()),
            }
        )
    for arm in ["EDF", "EST"]:
        arm_rows = raw[raw["method"] == arm]
        summary.append({"method": arm, "reward": float(arm_rows["reward"].mean()), "search": float(arm_rows["search"].mean())})
    summary_df = pd.DataFrame(summary).sort_values("reward", ascending=False)
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
