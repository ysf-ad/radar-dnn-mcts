from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "CreateValid1" / "experiments" / "code" / "model_code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from compare_action_heads_smoke import usable_targets
from exact_env_mutual import EDFPlanner, ESTPlanner, MAXT, _DummyPlanner, attach_env_obs, engine_env_cfg, env_cfg_for, xs_decode_action
from exact_env_mutual import xs_s_search_action, xs_x_search_action, xs_s_track_action, xs_x_track_action
from final_radar_campaign import get_obs, summarize_window_df
from foundation_mcts_fair_eval import run_exact_rescore_grid, run_heuristic
from foundation_mcts_fair_eval import FairExactRescore, physical_candidates
from joint_action_experiment import (
    JointPhysicalHeadPlanner,
    encode_joint_action,
    execute_first_valid_action_joint,
    execute_plan_until_budget_joint,
    is_joint_action,
    joint_duration,
    split_joint_action,
)
from learned_proposal_fair_eval import LearnedProposalFairExact
from penalty_window_quota_learner_eval import make_exact_args
from python_radar_env import PyRadarState, score_plans_vectorized
from repaired_campaign_tools import build_env
from strict_window_report import sample_state_metrics
from two_sensor_physical_head_eval import AutoregressiveBeamPlanner, PhysicalHeadPlanner, train_head, state_potential
from pufferlib.ocean.radarxs import binding


def execute_plan_until_budget_joint_compatible(eng, plan, budget_ms: float, search_debt_ms: float, planner_name: str, seed: int, window_idx: int):
    spent_ms = 0.0
    total_reward = 0.0
    search_actions = 0
    executed = 0
    rows = []
    slot = 0
    for action in plan:
        if spent_ms >= float(budget_ms) or bool(eng.term_buf[0]):
            break
        obs_before = get_obs(eng)
        reward, dt, executed_action = execute_first_valid_action_joint(eng, [int(action)], float(budget_ms) - spent_ms)
        if executed_action is None or dt <= 0.0:
            continue
        total_reward += float(reward)
        spent_ms += float(dt)
        atoms = split_joint_action(executed_action) if is_joint_action(executed_action) else (int(executed_action),)
        atom_sensors = [xs_decode_action(int(a), MAXT)[1] for a in atoms]
        atom_executed = [
            (sensor == 0 and float(obs_before.get("s_band_busy_ms", 0.0)) <= 0.0)
            or (sensor == 1 and float(obs_before.get("x_band_busy_ms", 0.0)) <= 0.0)
            or sensor not in {0, 1}
            for sensor in atom_sensors
        ]
        s_busy = float(obs_before.get("s_band_busy_ms", 0.0)) > 0.0 or 0 in atom_sensors
        x_busy = float(obs_before.get("x_band_busy_ms", 0.0)) > 0.0 or 1 in atom_sensors
        is_search = [xs_decode_action(int(a), MAXT)[0] == 0 and bool(done) for a, done in zip(atoms, atom_executed)]
        if any(is_search):
            search_debt_ms = 0.0
        else:
            search_debt_ms += float(dt)
        search_actions += int(any(is_search))
        executed += 1
        rows.append(
            {
                "planner": planner_name,
                "seed": int(seed),
                "bucket": int(window_idx),
                "slot": int(slot),
                "action": int(executed_action),
                "s_action": int(atoms[0]) if len(atoms) > 1 else -1,
                "x_action": int(atoms[1]) if len(atoms) > 1 else -1,
                "action_type": "Joint" if len(atoms) > 1 else ("Search" if is_search[0] else "Track"),
                "reward": float(reward),
                "dt_ms": float(dt),
                "s_busy_ms": float(dt) if s_busy else 0.0,
                "x_busy_ms": float(dt) if x_busy else 0.0,
            }
        )
        slot += 1

    idle_action = int(eng.max_trackers) + 1
    while spent_ms < float(budget_ms) and not bool(eng.term_buf[0]):
        obs_before = get_obs(eng)
        reward, dt, executed_action = execute_first_valid_action_joint(eng, [idle_action], float(budget_ms) - spent_ms)
        if executed_action is None or dt <= 0.0:
            break
        s_busy = float(obs_before.get("s_band_busy_ms", 0.0)) > 0.0
        x_busy = float(obs_before.get("x_band_busy_ms", 0.0)) > 0.0
        search_debt_ms += float(dt)
        total_reward += float(reward)
        spent_ms += float(dt)
        executed += 1
        rows.append(
            {
                "planner": planner_name,
                "seed": int(seed),
                "bucket": int(window_idx),
                "slot": int(slot),
                "action": int(executed_action),
                "s_action": -1,
                "x_action": -1,
                "action_type": "Wait",
                "reward": float(reward),
                "dt_ms": float(dt),
                "s_busy_ms": float(dt) if s_busy else 0.0,
                "x_busy_ms": float(dt) if x_busy else 0.0,
            }
        )
        slot += 1
    return total_reward, spent_ms, search_debt_ms, executed, search_actions, rows


class SupersetJointProposalPlanner:
    """Return sequential learned plan plus Cartesian S/X macro alternatives."""

    def __init__(self, base: PhysicalHeadPlanner, per_sensor_top: int = 4, max_joint_plans: int = 8):
        self.base = base
        self.greedy_joint = JointPhysicalHeadPlanner(base, per_sensor_top=int(per_sensor_top))
        self.env_cfg = dict(base.env_cfg)
        self.per_sensor_top = max(1, int(per_sensor_top))
        self.max_joint_plans = max(1, int(max_joint_plans))

    def _ranked_by_sensor(self, obs, selected: set[int], elapsed: float, search_count: int, track_count: int, last: int):
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        scores = self.base.score_actions(
            obs,
            selected=selected,
            elapsed=elapsed,
            search_count=search_count,
            track_count=track_count,
            last=last,
        )
        ranked = {0: [], 1: []}
        for action in physical_candidates(obs, top_k=MAXT):
            base, sensor = xs_decode_action(int(action), MAXT)
            if sensor not in {0, 1} or int(base) < 0:
                continue
            if int(base) > 0 and int(base) in selected:
                continue
            ranked[int(sensor)].append((float(scores[int(base), int(sensor)]), int(action)))
        for sensor in ranked:
            ranked[sensor].sort(reverse=True, key=lambda x: x[0])
            ranked[sensor] = ranked[sensor][: self.per_sensor_top]
        return ranked

    @staticmethod
    def _append_effect(plan_state: dict, obs, action: int) -> None:
        atoms = split_joint_action(action) if is_joint_action(action) else (int(action),)
        for atom in atoms:
            base, _sensor = xs_decode_action(int(atom), MAXT)
            if int(base) == 0:
                plan_state["search_count"] += 1
            elif int(base) > 0:
                plan_state["selected"].add(int(base))
                plan_state["track_count"] += 1
            plan_state["last"] = int(base)
        if is_joint_action(action):
            dt = joint_duration(obs, int(action))
        else:
            base, sensor = xs_decode_action(int(action), MAXT)
            if int(base) == 0:
                dt = 10.0
            elif int(base) > 0:
                dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
                dt = float(dwell[int(base) - 1]) if int(base) - 1 < len(dwell) else 10.0
                if sensor == 1:
                    dt *= 0.5
            else:
                dt = 1.0
        plan_state["elapsed"] += max(1.0, float(dt))

    def _continue_greedy_joint(self, obs, first_action: int, budget_ms: float) -> list[int]:
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        state = {"selected": set(), "elapsed": 0.0, "search_count": 0, "track_count": 0, "last": -1}
        plan = [int(first_action)]
        self._append_effect(state, obs, int(first_action))
        while state["elapsed"] < float(budget_ms) and len(plan) < 64:
            ranked = self._ranked_by_sensor(
                obs,
                state["selected"],
                state["elapsed"],
                state["search_count"],
                state["track_count"],
                state["last"],
            )
            best = None
            best_score = -np.inf
            for s_score, s_action in (ranked[0] or []):
                for x_score, x_action in (ranked[1] or []):
                    s_base, _ = xs_decode_action(int(s_action), MAXT)
                    x_base, _ = xs_decode_action(int(x_action), MAXT)
                    if int(s_base) > 0 and int(s_base) == int(x_base):
                        continue
                    score = float(s_score) + float(x_score)
                    if score > best_score:
                        best_score = score
                        best = encode_joint_action(int(s_action), int(x_action))
            if best is None:
                break
            plan.append(int(best))
            self._append_effect(state, obs, int(best))
        return plan

    @staticmethod
    def _pack_atomic_plan(plan: list[int]) -> list[int]:
        packed = []
        pending = {0: None, 1: None}
        for raw in plan:
            action = int(raw)
            if is_joint_action(action):
                if pending[0] is not None:
                    packed.append(int(pending[0]))
                    pending[0] = None
                if pending[1] is not None:
                    packed.append(int(pending[1]))
                    pending[1] = None
                packed.append(action)
                continue
            base, sensor = xs_decode_action(action, MAXT)
            if sensor not in {0, 1}:
                if pending[0] is not None:
                    packed.append(int(pending[0]))
                    pending[0] = None
                if pending[1] is not None:
                    packed.append(int(pending[1]))
                    pending[1] = None
                packed.append(action)
                continue
            other = 1 - int(sensor)
            if pending[other] is not None:
                if int(sensor) == 0:
                    packed.append(encode_joint_action(action, int(pending[other])))
                else:
                    packed.append(encode_joint_action(int(pending[other]), action))
                pending[other] = None
            elif pending[int(sensor)] is None:
                pending[int(sensor)] = action
            else:
                packed.append(int(pending[int(sensor)]))
                pending[int(sensor)] = action
        for sensor in (0, 1):
            if pending[sensor] is not None:
                packed.append(int(pending[sensor]))
        return packed

    def _first_joint_actions(self, obs) -> list[int]:
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        ranked = self._ranked_by_sensor(obs, set(), 0.0, 0, 0, -1)
        pairs = []
        for s_score, s_action in ranked[0]:
            for x_score, x_action in ranked[1]:
                s_base, _ = xs_decode_action(int(s_action), MAXT)
                x_base, _ = xs_decode_action(int(x_action), MAXT)
                if int(s_base) > 0 and int(s_base) == int(x_base):
                    continue
                pairs.append((float(s_score) + float(x_score), encode_joint_action(int(s_action), int(x_action))))
        pairs.sort(reverse=True, key=lambda x: x[0])
        out = []
        seen = set()
        for _score, action in pairs:
            if action in seen:
                continue
            seen.add(action)
            out.append(int(action))
            if len(out) >= self.max_joint_plans:
                break
        return out

    def plan(self, obs, budget_ms=200):
        plans = []
        seq = list(self.base.plan(obs, budget_ms=budget_ms))
        if seq:
            plans.append(seq)
            packed_seq = self._pack_atomic_plan(seq)
            if packed_seq:
                plans.append(packed_seq)
        greedy = list(self.greedy_joint.plan(obs, budget_ms=budget_ms))
        if greedy:
            plans.append(greedy)
        for first in self._first_joint_actions(obs):
            plans.append(self._continue_greedy_joint(obs, int(first), float(budget_ms)))
        deduped = []
        seen = set()
        for plan in plans:
            key = tuple(int(a) for a in plan)
            if key and key not in seen:
                seen.add(key)
                deduped.append([int(a) for a in key])
        return deduped or [seq]


class AsyncCoupledJointPlanner:
    """Event-driven coupled planner using joint actions as async sensor commands.

    The model scores the free sensor(s) with the other sensor's busy timer in the
    state.  When only one sensor is free, the busy sensor receives a dummy action
    inside an encoded joint command so the C environment advances to the next
    sensor-completion event instead of the free sensor's full dwell.
    """

    def __init__(self, base: PhysicalHeadPlanner, per_sensor_top: int = 1, include_search_candidate: bool = False):
        self.base = base
        self.env_cfg = dict(base.env_cfg)
        self.per_sensor_top = max(1, int(per_sensor_top))
        self.include_search_candidate = bool(include_search_candidate)

    @staticmethod
    def _duration(obs: dict, action: int) -> float:
        base, sensor = xs_decode_action(int(action), MAXT)
        if int(base) == 0:
            return 10.0
        if int(base) <= 0:
            return 1.0
        dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        dt = float(dwell[int(base) - 1]) if int(base) - 1 < len(dwell) else 10.0
        if int(sensor) == 1:
            dt *= 0.5
        return max(1.0, float(dt))

    def _ranked_for_sensor(self, obs: dict, sensor: int, selected: set[int], elapsed: float, search_count: int, track_count: int, last: int):
        local_obs = dict(obs)
        local_obs = attach_env_obs(local_obs, self.env_cfg, True, True)
        scores = self.base.score_actions(
            local_obs,
            selected=selected,
            elapsed=elapsed,
            search_count=search_count,
            track_count=track_count,
            last=last,
        )
        ranked = []
        for action in physical_candidates(local_obs, top_k=MAXT):
            base, action_sensor = xs_decode_action(int(action), MAXT)
            if int(action_sensor) != int(sensor) or int(base) < 0:
                continue
            if int(base) > 0 and int(base) in selected:
                continue
            ranked.append((float(scores[int(base), int(sensor)]), int(base), int(action)))
        ranked.sort(reverse=True, key=lambda x: x[0])
        out = ranked[: self.per_sensor_top]
        search_action = xs_s_search_action(MAXT) if int(sensor) == 0 else xs_x_search_action(MAXT)
        if self.include_search_candidate and not any(int(action) == int(search_action) for _score, _base, action in out):
            for item in ranked:
                if int(item[2]) == int(search_action):
                    out.append(item)
                    break
        return out

    def _choose_pair(self, obs: dict, selected: set[int], elapsed: float, search_count: int, track_count: int, last: int):
        pairs = self._candidate_pairs(obs, selected, elapsed, search_count, track_count, last)
        return pairs[0][1] if pairs else None

    def _candidate_pairs(self, obs: dict, selected: set[int], elapsed: float, search_count: int, track_count: int, last: int):
        s_busy = float(obs.get("s_band_busy_ms", 0.0))
        x_busy = float(obs.get("x_band_busy_ms", 0.0))
        x_enabled = bool(int(obs.get("enable_x_band", 0)))
        s_free = s_busy <= 0.0
        x_free = x_enabled and x_busy <= 0.0
        s_ranked = self._ranked_for_sensor(obs, 0, selected, elapsed, search_count, track_count, last) if s_free else []
        x_ranked = self._ranked_for_sensor(obs, 1, selected, elapsed, search_count, track_count, last) if x_free else []
        s_dummy = xs_s_search_action(MAXT)
        x_dummy = xs_x_search_action(MAXT)
        out = []
        if s_ranked and x_ranked:
            for s_score, s_base, s_action in s_ranked:
                for x_score, x_base, x_action in x_ranked:
                    if int(s_base) > 0 and int(s_base) == int(x_base):
                        continue
                    score = float(s_score) + float(x_score)
                    out.append((score, encode_joint_action(int(s_action), int(x_action))))
        if s_ranked:
            for s_score, _s_base, s_action in s_ranked:
                out.append((float(s_score), encode_joint_action(int(s_action), int(x_dummy))))
        if x_ranked:
            for x_score, _x_base, x_action in x_ranked:
                out.append((float(x_score), encode_joint_action(int(s_dummy), int(x_action))))
        if s_busy > 0.0 or (x_enabled and x_busy > 0.0):
            out.append((-1e6, encode_joint_action(int(s_dummy), int(x_dummy))))
        deduped = {}
        for score, action in out:
            deduped[int(action)] = max(float(score), deduped.get(int(action), -np.inf))
        return sorted([(score, action) for action, score in deduped.items()], reverse=True, key=lambda x: x[0])

    def _advance_synthetic(self, obs: dict, action: int, selected: set[int], elapsed: float, search_count: int, track_count: int, last: int):
        s_busy = float(obs.get("s_band_busy_ms", 0.0))
        x_busy = float(obs.get("x_band_busy_ms", 0.0))
        atoms = split_joint_action(int(action)) if is_joint_action(int(action)) else (int(action),)
        for atom in atoms:
            base, sensor = xs_decode_action(int(atom), MAXT)
            if int(sensor) == 0 and s_busy <= 0.0:
                s_busy = self._duration(obs, int(atom))
            elif int(sensor) == 1 and x_busy <= 0.0:
                x_busy = self._duration(obs, int(atom))
            else:
                continue
            if int(base) == 0:
                search_count += 1
            elif int(base) > 0:
                selected.add(int(base))
                track_count += 1
            last = int(base)
        positive = [v for v in (s_busy, x_busy) if v > 0.0]
        dt = min(positive) if positive else 1.0
        s_busy = max(0.0, s_busy - dt)
        x_busy = max(0.0, x_busy - dt)
        obs["s_band_busy_ms"] = float(s_busy)
        obs["x_band_busy_ms"] = float(x_busy)
        return selected, elapsed + float(dt), search_count, track_count, last

    def plan(self, obs, budget_ms=200):
        plan_obs = attach_env_obs(dict(obs), self.env_cfg, True, True)
        selected: set[int] = set()
        plan = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        while elapsed < float(budget_ms) and len(plan) < 128:
            action = self._choose_pair(plan_obs, selected, elapsed, search_count, track_count, last)
            if action is None:
                break
            plan.append(int(action))
            selected, elapsed, search_count, track_count, last = self._advance_synthetic(
                plan_obs, int(action), selected, elapsed, search_count, track_count, last
            )
        return plan if plan else [encode_joint_action(xs_s_search_action(MAXT), xs_x_search_action(MAXT))]


class AsyncCoupledBeamProposalPlanner(AsyncCoupledJointPlanner):
    def __init__(self, base: PhysicalHeadPlanner, per_sensor_top: int = 3, beams: int = 12, include_search_candidate: bool = False):
        super().__init__(base, per_sensor_top=per_sensor_top, include_search_candidate=include_search_candidate)
        self.beams = max(1, int(beams))

    def _tail_from(self, obs, first_action: int, budget_ms: float):
        plan_obs = attach_env_obs(dict(obs), self.env_cfg, True, True)
        selected: set[int] = set()
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        plan = [int(first_action)]
        selected, elapsed, search_count, track_count, last = self._advance_synthetic(
            plan_obs, int(first_action), selected, elapsed, search_count, track_count, last
        )
        while elapsed < float(budget_ms) and len(plan) < 128:
            action = self._choose_pair(plan_obs, selected, elapsed, search_count, track_count, last)
            if action is None:
                break
            plan.append(int(action))
            selected, elapsed, search_count, track_count, last = self._advance_synthetic(
                plan_obs, int(action), selected, elapsed, search_count, track_count, last
            )
        return plan

    def plan(self, obs, budget_ms=200):
        plan_obs = attach_env_obs(dict(obs), self.env_cfg, True, True)
        firsts = self._candidate_pairs(plan_obs, set(), 0.0, 0, 0, -1)[: self.beams]
        plans = []
        seen = set()
        for _score, action in firsts:
            plan = self._tail_from(obs, int(action), float(budget_ms))
            key = tuple(int(a) for a in plan)
            if key and key not in seen:
                seen.add(key)
                plans.append([int(a) for a in key])
        return plans or [[encode_joint_action(xs_s_search_action(MAXT), xs_x_search_action(MAXT))]]


class WorkConservingAsyncCoupledPlanner(AsyncCoupledJointPlanner):
    """Async planner that fills every currently free sensor.

    Unlike AsyncCoupledJointPlanner, this does not offer a single-sensor action
    when both sensors are free. Single-sensor commands are only used when the
    other sensor is already busy, where the second slot is just the encoded
    placeholder needed by the joint-action interface.
    """

    def _candidate_pairs(self, obs: dict, selected: set[int], elapsed: float, search_count: int, track_count: int, last: int):
        s_busy = float(obs.get("s_band_busy_ms", 0.0))
        x_busy = float(obs.get("x_band_busy_ms", 0.0))
        x_enabled = bool(int(obs.get("enable_x_band", 0)))
        s_free = s_busy <= 0.0
        x_free = x_enabled and x_busy <= 0.0
        s_ranked = self._ranked_for_sensor(obs, 0, selected, elapsed, search_count, track_count, last) if s_free else []
        x_ranked = self._ranked_for_sensor(obs, 1, selected, elapsed, search_count, track_count, last) if x_free else []
        s_dummy = xs_s_search_action(MAXT)
        x_dummy = xs_x_search_action(MAXT)
        out = []
        if s_ranked and x_ranked:
            for s_score, s_base, s_action in s_ranked:
                for x_score, x_base, x_action in x_ranked:
                    if int(s_base) > 0 and int(s_base) == int(x_base):
                        continue
                    out.append((float(s_score) + float(x_score), encode_joint_action(int(s_action), int(x_action))))
        elif s_ranked:
            for s_score, _s_base, s_action in s_ranked:
                out.append((float(s_score), encode_joint_action(int(s_action), int(x_dummy))))
        elif x_ranked:
            for x_score, _x_base, x_action in x_ranked:
                out.append((float(x_score), encode_joint_action(int(s_dummy), int(x_action))))
        elif s_busy > 0.0 or (x_enabled and x_busy > 0.0):
            out.append((-1e6, encode_joint_action(int(s_dummy), int(x_dummy))))
        deduped = {}
        for score, action in out:
            deduped[int(action)] = max(float(score), deduped.get(int(action), -np.inf))
        return sorted([(score, action) for action, score in deduped.items()], reverse=True, key=lambda x: x[0])


class WorkConservingAsyncBeamPlanner(WorkConservingAsyncCoupledPlanner):
    def __init__(self, base: PhysicalHeadPlanner, per_sensor_top: int = 3, beams: int = 12, include_search_candidate: bool = False):
        super().__init__(base, per_sensor_top=per_sensor_top, include_search_candidate=include_search_candidate)
        self.beams = max(1, int(beams))

    def _tail_from(self, obs, first_action: int, budget_ms: float):
        plan_obs = attach_env_obs(dict(obs), self.env_cfg, True, True)
        selected: set[int] = set()
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        plan = [int(first_action)]
        selected, elapsed, search_count, track_count, last = self._advance_synthetic(
            plan_obs, int(first_action), selected, elapsed, search_count, track_count, last
        )
        while elapsed < float(budget_ms) and len(plan) < 128:
            action = self._choose_pair(plan_obs, selected, elapsed, search_count, track_count, last)
            if action is None:
                break
            plan.append(int(action))
            selected, elapsed, search_count, track_count, last = self._advance_synthetic(
                plan_obs, int(action), selected, elapsed, search_count, track_count, last
            )
        return plan

    def plan(self, obs, budget_ms=200):
        plan_obs = attach_env_obs(dict(obs), self.env_cfg, True, True)
        firsts = self._candidate_pairs(plan_obs, set(), 0.0, 0, 0, -1)[: self.beams]
        plans = []
        seen = set()
        for _score, action in firsts:
            plan = self._tail_from(obs, int(action), float(budget_ms))
            key = tuple(int(a) for a in plan)
            if key and key not in seen:
                seen.add(key)
                plans.append([int(a) for a in key])
        return plans or [[encode_joint_action(xs_s_search_action(MAXT), xs_x_search_action(MAXT))]]


class AsyncAutoregressiveCoupledPlanner(WorkConservingAsyncCoupledPlanner):
    """Async direct planner with learned S -> X action conditioning.

    The first sensor head proposes S actions.  The X head is then evaluated
    conditioned on each S proposal, so the second sensor is not scored as an
    independent marginal action.
    """

    def __init__(
        self,
        model,
        env_cfg: dict,
        per_sensor_top: int = 3,
        policy_weight: float = 1.0,
        q_weight: float = 1.0,
        search_score_bias: float = 0.0,
    ):
        self.ar = AutoregressiveBeamPlanner(
            model.eval(),
            env_cfg,
            s_top_k=max(1, int(per_sensor_top)),
            x_top_k=max(1, int(per_sensor_top)),
            policy_weight=float(policy_weight),
            q_weight=float(q_weight),
            search_score_bias=float(search_score_bias),
        )
        self.env_cfg = dict(env_cfg)
        self.per_sensor_top = max(1, int(per_sensor_top))
        self.include_search_candidate = True

    def _rank_x_conditioned(self, obs: dict, selected: set[int], s_base: int, tok, slot):
        return self.ar._rank_x_for_s(obs, tok, slot, int(s_base), selected)[: self.per_sensor_top]

    def _candidate_pairs(self, obs: dict, selected: set[int], elapsed: float, search_count: int, track_count: int, last: int):
        obs_attached, tok, slot = self.ar._state_tensors(obs, selected, elapsed, search_count, track_count, last)
        s_busy = float(obs_attached.get("s_band_busy_ms", 0.0))
        x_busy = float(obs_attached.get("x_band_busy_ms", 0.0))
        x_enabled = bool(int(obs_attached.get("enable_x_band", 0)))
        s_free = s_busy <= 0.0
        x_free = x_enabled and x_busy <= 0.0
        s_dummy = xs_s_search_action(MAXT)
        x_dummy = xs_x_search_action(MAXT)
        out = []
        if s_free and x_free:
            for s_score, s_base, s_action, tok_s, slot_s, obs_s in self.ar._rank_s_actions(
                obs_attached, selected, elapsed, search_count, track_count, last
            )[: self.per_sensor_top]:
                for x_score, x_base, x_action in self._rank_x_conditioned(obs_s, selected, int(s_base), tok_s, slot_s):
                    if int(s_base) > 0 and int(x_base) == int(s_base):
                        continue
                    out.append((float(s_score) + float(x_score), encode_joint_action(int(s_action), int(x_action))))
        elif s_free:
            for s_score, _s_base, s_action, _tok_s, _slot_s, _obs_s in self.ar._rank_s_actions(
                obs_attached, selected, elapsed, search_count, track_count, last
            )[: self.per_sensor_top]:
                out.append((float(s_score), encode_joint_action(int(s_action), int(x_dummy))))
        elif x_free:
            for x_score, _x_base, x_action in self._rank_x_conditioned(obs_attached, selected, 0, tok, slot):
                out.append((float(x_score), encode_joint_action(int(s_dummy), int(x_action))))
        elif s_busy > 0.0 or (x_enabled and x_busy > 0.0):
            out.append((-1e6, encode_joint_action(int(s_dummy), int(x_dummy))))
        deduped = {}
        for score, action in out:
            deduped[int(action)] = max(float(score), deduped.get(int(action), -np.inf))
        return sorted([(score, action) for action, score in deduped.items()], reverse=True, key=lambda x: x[0])


class DirectPlanAdapter:
    def __init__(self, planner):
        self.planner = planner

    def choose(self, eng, debt_ms: float, obs) -> tuple[list[int], dict]:
        t0 = time.perf_counter()
        plan = self.planner.plan(obs, budget_ms=200)
        return list(plan), {
            "planning_ms": float((time.perf_counter() - t0) * 1000.0),
            "candidate_count": 1,
            "exact_rescored": 0,
            "forced_learned_rescored": 0,
            "exact_score": np.nan,
            "seq_candidate_idx": -1,
            "seq_rescored": False,
            "seq_exact_score": np.nan,
            "protected_exact_score": np.nan,
            "exact_score_elapsed_ms": np.nan,
            "exact_score_executed": 0,
            "best_candidate_idx": 0,
        }


class DirectFirstActionAdapter(DirectPlanAdapter):
    """Direct model adapter for event-receding execution.

    The wrapped planner may synthesize a full 200 ms plan, but the harness only
    executes the first joint action and then re-observes the environment.
    """

    def choose(self, eng, debt_ms: float, obs) -> tuple[list[int], dict]:
        plan, meta = super().choose(eng, debt_ms, obs)
        return list(plan[:1]), meta


class StatefulRecedingDirectAdapter:
    """Event-receding direct adapter that preserves intra-window planner state."""

    def __init__(self, planner):
        self.planner = planner
        self.reset()

    def reset(self):
        self.selected: set[int] = set()
        self.elapsed = 0.0
        self.search_count = 0
        self.track_count = 0
        self.last = -1

    def start_window(self, _window: int):
        self.reset()

    def choose(self, eng, debt_ms: float, obs) -> tuple[list[int], dict]:
        t0 = time.perf_counter()
        plan_obs = attach_env_obs(dict(obs), self.planner.env_cfg, True, True)
        action = self.planner._choose_pair(
            plan_obs,
            self.selected,
            self.elapsed,
            self.search_count,
            self.track_count,
            self.last,
        )
        if action is None:
            action = encode_joint_action(xs_s_search_action(MAXT), xs_x_search_action(MAXT))
        return [int(action)], {
            "planning_ms": float((time.perf_counter() - t0) * 1000.0),
            "candidate_count": 1,
            "exact_rescored": 0,
            "forced_learned_rescored": 0,
            "exact_score": np.nan,
            "seq_candidate_idx": -1,
            "seq_rescored": False,
            "seq_exact_score": np.nan,
            "protected_exact_score": np.nan,
            "exact_score_elapsed_ms": np.nan,
            "exact_score_executed": 0,
            "best_candidate_idx": 0,
        }

    def observe_executed(self, obs_before: dict, executed_action: int, dt_ms: float):
        plan_obs = attach_env_obs(dict(obs_before), self.planner.env_cfg, True, True)
        self.selected, self.elapsed, self.search_count, self.track_count, self.last = self.planner._advance_synthetic(
            plan_obs,
            int(executed_action),
            self.selected,
            self.elapsed,
            self.search_count,
            self.track_count,
            self.last,
        )
        self.elapsed = max(float(self.elapsed), float(self.elapsed - 0.0))


class AsyncOneStepExactAdapter:
    def __init__(self, planner: AsyncCoupledBeamProposalPlanner, beams: int = 8, potential_weight: float = 1.0):
        self.planner = planner
        self.beams = max(1, int(beams))
        self.potential_weight = float(potential_weight)

    def choose(self, eng, debt_ms: float, obs) -> tuple[list[int], dict]:
        t0 = time.perf_counter()
        obs = attach_env_obs(dict(obs), self.planner.env_cfg, True, True)
        candidates = self.planner._candidate_pairs(obs, set(), 0.0, 0, 0, -1)[: self.beams]
        root = binding.vec_snapshot(eng.env)
        before = state_potential(eng, float(debt_ms))
        scored = []
        for prior_score, action in candidates:
            binding.vec_restore(eng.env, root)
            reward, dt, executed = execute_first_valid_action_joint(eng, [int(action)], 200.0)
            if executed is None or dt <= 0.0:
                continue
            atoms = split_joint_action(int(executed)) if is_joint_action(int(executed)) else (int(executed),)
            is_search = any(xs_decode_action(int(a), MAXT)[0] == 0 for a in atoms)
            next_debt = 0.0 if is_search else float(debt_ms) + float(dt)
            after = state_potential(eng, float(next_debt))
            score = float(reward) + self.potential_weight * (float(after) - float(before)) + 0.001 * float(prior_score)
            scored.append((score, int(action), float(reward), float(dt)))
        binding.vec_restore(eng.env, root)
        if not scored:
            fallback = self.planner.plan(obs, budget_ms=200)
            action = int(fallback[0]) if fallback else encode_joint_action(xs_s_search_action(MAXT), xs_x_search_action(MAXT))
            best_score = np.nan
        else:
            best_score, action, _reward, _dt = max(scored, key=lambda x: x[0])
        return [int(action)], {
            "planning_ms": float((time.perf_counter() - t0) * 1000.0),
            "candidate_count": int(len(candidates)),
            "exact_rescored": int(len(scored)),
            "forced_learned_rescored": 0,
            "exact_score": float(best_score),
            "seq_candidate_idx": -1,
            "seq_rescored": False,
            "seq_exact_score": np.nan,
            "protected_exact_score": np.nan,
            "exact_score_elapsed_ms": np.nan,
            "exact_score_executed": 1,
            "best_candidate_idx": 0,
        }


class FillIdleSensorAdapter:
    def __init__(self, inner, env_cfg: dict, fill_mode: str = "edf", scorer: PhysicalHeadPlanner | None = None):
        self.inner = inner
        self.env_cfg = dict(env_cfg)
        self.fill_mode = str(fill_mode)
        self.scorer = scorer

    @staticmethod
    def _physicalize_primary(action: int, obs: dict) -> int:
        action = int(action)
        if is_joint_action(action):
            return action
        base, sensor = xs_decode_action(action, MAXT)
        if sensor in {0, 1}:
            return action
        if int(base) == 0:
            return xs_s_search_action(MAXT)
        if int(base) > 0:
            return xs_s_track_action(int(base), MAXT)
        return xs_s_search_action(MAXT)

    @staticmethod
    def _edf_fill(obs: dict, sensor: int, selected: set[int]) -> int:
        active = np.asarray(obs.get("active_mask", []), dtype=bool)
        deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
        ranges = np.asarray(obs.get("target_range", np.zeros_like(deadline)), dtype=np.float32)
        ranked = []
        for idx, ok in enumerate(active[:MAXT]):
            base = int(idx) + 1
            if not bool(ok) or base in selected or idx >= len(deadline) or float(deadline[idx]) < 0.0:
                continue
            r = float(ranges[idx]) if idx < len(ranges) else 50_000_000.0
            if int(sensor) == 0 and not (10_000_000.0 < r < 184_000_000.0):
                continue
            if int(sensor) == 1 and not (5_000_000.0 < r < 100_000_000.0):
                continue
            ranked.append((float(deadline[idx]), base))
        ranked.sort(key=lambda x: (x[0], x[1]))
        if ranked:
            base = int(ranked[0][1])
            return xs_s_track_action(base, MAXT) if int(sensor) == 0 else xs_x_track_action(base, MAXT)
        return xs_s_search_action(MAXT) if int(sensor) == 0 else xs_x_search_action(MAXT)

    def _model_fill(self, obs: dict, sensor: int, selected: set[int]) -> int:
        if self.scorer is None:
            return self._edf_fill(obs, sensor, selected)
        scores = self.scorer.score_actions(obs, selected=set(selected), elapsed=0.0, search_count=0, track_count=0, last=-1)
        best = None
        for action in physical_candidates(obs, top_k=MAXT):
            base, action_sensor = xs_decode_action(int(action), MAXT)
            if int(action_sensor) != int(sensor) or int(base) < 0:
                continue
            if int(base) > 0 and int(base) in selected:
                continue
            val = float(scores[int(base), int(sensor)])
            if best is None or val > best[0]:
                best = (val, int(action))
        if best is not None:
            return int(best[1])
        return xs_s_search_action(MAXT) if int(sensor) == 0 else xs_x_search_action(MAXT)

    def _fill_plan(self, obs: dict, plan: list[int]) -> list[int]:
        obs = attach_env_obs(dict(obs), self.env_cfg, True, True)
        out = []
        selected: set[int] = set()
        for raw in plan:
            action = self._physicalize_primary(int(raw), obs)
            if is_joint_action(action):
                atoms = split_joint_action(action)
                for atom in atoms:
                    base, _sensor = xs_decode_action(int(atom), MAXT)
                    if int(base) > 0:
                        selected.add(int(base))
                out.append(int(action))
                continue
            base, sensor = xs_decode_action(int(action), MAXT)
            if int(sensor) not in {0, 1}:
                out.append(int(action))
                continue
            other = 1 - int(sensor)
            if int(base) > 0:
                selected.add(int(base))
            if self.fill_mode == "search":
                filler = xs_s_search_action(MAXT) if int(other) == 0 else xs_x_search_action(MAXT)
            elif self.fill_mode == "model":
                filler = self._model_fill(obs, other, selected)
            else:
                filler = self._edf_fill(obs, other, selected)
            fill_base, _fill_sensor = xs_decode_action(int(filler), MAXT)
            if int(fill_base) > 0:
                selected.add(int(fill_base))
            if int(sensor) == 0:
                out.append(encode_joint_action(int(action), int(filler)))
            else:
                out.append(encode_joint_action(int(filler), int(action)))
        return out

    def choose(self, eng, debt_ms: float, obs) -> tuple[list[int], dict]:
        plan, meta = self.inner.choose(eng, debt_ms, obs)
        t0 = time.perf_counter()
        filled = self._fill_plan(obs, list(plan))
        meta = dict(meta)
        meta["planning_ms"] = float(meta.get("planning_ms", 0.0)) + float((time.perf_counter() - t0) * 1000.0)
        meta["filled_idle_sensor"] = 1
        return filled, meta


class RecedingModelFillAdapter(FillIdleSensorAdapter):
    def __init__(self, inner, env_cfg: dict, scorer: PhysicalHeadPlanner):
        super().__init__(inner, env_cfg, fill_mode="model", scorer=scorer)

    def choose(self, eng, debt_ms: float, obs) -> tuple[list[int], dict]:
        t0 = time.perf_counter()
        obs = attach_env_obs(dict(obs), self.env_cfg, True, True)
        plan, meta = self.inner.choose(eng, debt_ms, obs)
        meta = dict(meta)
        if not plan:
            action = encode_joint_action(xs_s_search_action(MAXT), xs_x_search_action(MAXT))
        else:
            primary = self._physicalize_primary(int(plan[0]), obs)
            if is_joint_action(primary):
                action = int(primary)
            else:
                base, sensor = xs_decode_action(int(primary), MAXT)
                selected = {int(base)} if int(base) > 0 else set()
                s_free = float(obs.get("s_band_busy_ms", 0.0)) <= 0.0
                x_free = bool(int(obs.get("enable_x_band", 0))) and float(obs.get("x_band_busy_ms", 0.0)) <= 0.0
                s_dummy = xs_s_search_action(MAXT)
                x_dummy = xs_x_search_action(MAXT)
                if int(sensor) == 0 and s_free:
                    x_action = self._model_fill(obs, 1, selected) if x_free else x_dummy
                    action = encode_joint_action(int(primary), int(x_action))
                elif int(sensor) == 1 and x_free:
                    s_action = self._model_fill(obs, 0, selected) if s_free else s_dummy
                    action = encode_joint_action(int(s_action), int(primary))
                elif x_free:
                    x_action = self._model_fill(obs, 1, set())
                    action = encode_joint_action(int(s_dummy), int(x_action))
                elif s_free:
                    s_action = self._model_fill(obs, 0, set())
                    action = encode_joint_action(int(s_action), int(x_dummy))
                else:
                    action = encode_joint_action(int(s_dummy), int(x_dummy))
        meta["planning_ms"] = float(meta.get("planning_ms", 0.0)) + float((time.perf_counter() - t0) * 1000.0)
        meta["receding_model_fill"] = 1
        return [int(action)], meta


class JointAwareLearnedProposalFairExact(LearnedProposalFairExact):
    joint_only = False
    strict_superset = False

    def candidates(self, obs) -> np.ndarray:
        base = FairExactRescore.candidates(self, obs)
        self._last_base_count = int(len(base))
        self._last_sequential_candidate_idx = None
        self._joint_candidate_indices = set()
        learned_rows = []
        for learned_planner in self.learned_planners:
            raw_plans = learned_planner.plan(obs, budget_ms=int(self.score_horizon_ms))
            if not raw_plans:
                continue
            if raw_plans and isinstance(raw_plans[0], (list, tuple, np.ndarray)):
                plan_list = raw_plans
            else:
                plan_list = [raw_plans]
            for plan_i, learned in enumerate(plan_list):
                learned = list(learned)
                if not learned:
                    continue
                tiled = []
                while len(tiled) < self.learned_slots:
                    tiled.extend([int(action) for action in learned])
                learned_row = np.asarray(tiled[: self.learned_slots], dtype=np.int32)[None, :]
                if learned_row.shape[1] != base.shape[1]:
                    if learned_row.shape[1] < base.shape[1]:
                        pad = np.full((1, base.shape[1] - learned_row.shape[1]), int(learned_row[0, -1]), dtype=np.int32)
                        learned_row = np.concatenate([learned_row, pad], axis=1)
                    else:
                        learned_row = learned_row[:, : base.shape[1]]
                if self._last_sequential_candidate_idx is None and plan_i == 0:
                    self._last_sequential_candidate_idx = int(len(base) + len(learned_rows))
                if any(is_joint_action(int(action)) for action in learned):
                    self._joint_candidate_indices.add(int(len(base) + len(learned_rows)))
                learned_rows.append(learned_row)
        if not learned_rows:
            self._last_candidate_count = int(len(base))
            return base
        out = np.concatenate([base, *learned_rows], axis=0)
        self._last_candidate_count = int(len(out))
        return out

    @staticmethod
    def _logical_approx_candidates(candidates: np.ndarray) -> np.ndarray:
        approx = np.asarray(candidates, dtype=np.int64).copy()
        flat = approx.reshape(-1)
        for idx, action in enumerate(flat):
            action = int(action)
            if is_joint_action(action):
                s_action, x_action = split_joint_action(action)
                s_base, _ = xs_decode_action(int(s_action), MAXT)
                x_base, _ = xs_decode_action(int(x_action), MAXT)
                flat[idx] = max(0, int(s_base) if int(s_base) > 0 else int(x_base))
            else:
                base, _sensor = xs_decode_action(action, MAXT)
                flat[idx] = max(0, int(base))
        return approx.astype(np.int32)

    def choose(self, eng, debt_ms: float, obs) -> tuple[list[int], dict]:
        candidates = self.candidates(obs)
        t0 = time.perf_counter()
        approx = score_plans_vectorized(
            PyRadarState.from_obs(obs),
            self._logical_approx_candidates(candidates),
            self.gen.cfg,
            budget_ms=200.0,
        )
        learned_start = int(getattr(self, "_last_base_count", len(candidates)))
        top = np.argsort(approx)[-self.top_k :].astype(int).tolist()
        if not bool(getattr(self, "joint_only", False)):
            top.extend([self.n_plans - 2, self.n_plans - 1])
        if bool(getattr(self, "strict_superset", False)) and not bool(getattr(self, "joint_only", False)):
            top.extend(range(0, learned_start))
        if self.force_learned_rescore:
            top.extend(range(learned_start, len(candidates)))
        elif int(self.learned_extra_top_k) > 0 and learned_start < len(candidates):
            learned_scores = approx[learned_start:]
            take = min(int(self.learned_extra_top_k), int(len(learned_scores)))
            top.extend([learned_start + int(i) for i in np.argsort(learned_scores)[-take:].astype(int).tolist()])
        if bool(getattr(self, "joint_only", False)) and learned_start < len(candidates):
            joint_indices = set(int(i) for i in getattr(self, "_joint_candidate_indices", set()))
            top = [i for i in top if int(i) in joint_indices]
        top = sorted(set(i for i in top if 0 <= i < len(candidates)))
        if not top:
            joint_indices = sorted(int(i) for i in getattr(self, "_joint_candidate_indices", set()))
            top = [joint_indices[0]] if joint_indices else ([int(learned_start)] if learned_start < len(candidates) else [int(np.argmax(approx))])

        root = binding.vec_snapshot(eng.env)
        exact_scores = []
        for idx in top:
            binding.vec_restore(eng.env, root)
            reward, spent, _debt, executed, _searches, _arows = execute_plan_until_budget_joint_compatible(
                eng,
                candidates[int(idx)].tolist(),
                float(self.score_horizon_ms),
                float(debt_ms),
                "score",
                0,
                0,
            )
            exact_scores.append((int(idx), float(reward), float(spent), int(executed)))
        best_idx, best_score, best_elapsed, best_executed = max(exact_scores, key=lambda x: x[1])
        seq_idx = getattr(self, "_last_sequential_candidate_idx", None)
        base_count = int(getattr(self, "_last_base_count", 0))
        seq_rows = [row for row in exact_scores if int(row[0]) == seq_idx]
        protected_rows = [] if bool(getattr(self, "joint_only", False)) else [row for row in exact_scores if int(row[0]) < base_count or int(row[0]) == seq_idx]
        protected_best = max(protected_rows, key=lambda x: x[1]) if protected_rows else None
        seq_score_value = float(seq_rows[0][1]) if seq_rows else np.nan
        protected_score_value = float(protected_best[1]) if protected_best is not None else np.nan
        if protected_best is not None and (
            bool(getattr(self, "strict_superset", False)) or float(getattr(self, "sequential_guard_margin", 0.0)) > 0.0
        ):
            protected_idx, protected_score, protected_elapsed, protected_executed = protected_best
            if float(best_score) < float(protected_score) + float(self.sequential_guard_margin):
                best_idx, best_score, best_elapsed, best_executed = protected_idx, protected_score, protected_elapsed, protected_executed
        binding.vec_restore(eng.env, root)
        return candidates[best_idx].tolist(), {
            "planning_ms": float((time.perf_counter() - t0) * 1000.0),
            "candidate_count": int(len(candidates)),
            "exact_rescored": int(len(top)),
            "forced_learned_rescored": int(max(0, len(candidates) - learned_start)) if self.force_learned_rescore else 0,
            "exact_score": float(best_score),
            "seq_candidate_idx": int(seq_idx) if seq_idx is not None else -1,
            "seq_rescored": bool(len(seq_rows) > 0),
            "seq_exact_score": float(seq_score_value),
            "protected_exact_score": float(protected_score_value),
            "exact_score_elapsed_ms": float(best_elapsed),
            "exact_score_executed": int(best_executed),
            "best_candidate_idx": int(best_idx),
        }


def run_exact_rescore_grid_joint(planner, name: str, initial: int, seed: int, windows: int, env_cfg: dict):
    eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
    eng.reset(seed=int(seed))
    debt = 0.0
    cumulative = 0.0
    rows = []
    action_rows = []
    try:
        for window in range(int(windows)):
            if bool(eng.term_buf[0]):
                break
            obs = get_obs(eng, debt)
            plan, meta = planner.choose(eng, debt, obs)
            reward, spent, debt, executed, searches, arows = execute_plan_until_budget_joint_compatible(
                eng, plan, 200.0, debt, name, int(seed), int(window)
            )
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
                    "planning_ms_per_decision": float(meta.get("planning_ms", np.nan)),
                    "planning_ms_per_executed_action": float(float(meta.get("planning_ms", np.nan)) / max(1, executed)),
                    "executed_actions": int(executed),
                    "spent_ms": float(spent),
                    **sample_state_metrics(eng, debt),
                    **meta,
                }
            )
            for row in arows:
                row.update(window=int(window), elapsed_ms=float((window + 1) * 200))
            action_rows.extend(arows)
    finally:
        eng.close()
    return pd.DataFrame(rows), pd.DataFrame(action_rows)


def run_receding_exact_grid_joint(planner, name: str, initial: int, seed: int, windows: int, env_cfg: dict):
    eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
    eng.reset(seed=int(seed))
    debt = 0.0
    cumulative = 0.0
    rows = []
    action_rows = []
    try:
        for window in range(int(windows)):
            if bool(eng.term_buf[0]):
                break
            if hasattr(planner, "start_window"):
                planner.start_window(int(window))
            spent = 0.0
            reward_total = 0.0
            executed_total = 0
            search_total = 0
            plan_ms = 0.0
            meta_last = {}
            while spent < 200.0 and not bool(eng.term_buf[0]):
                obs = get_obs(eng, debt)
                plan, meta = planner.choose(eng, debt, obs)
                meta_last = dict(meta)
                plan_ms += float(meta.get("planning_ms", 0.0))
                if not plan:
                    break
                obs_before = get_obs(eng, debt)
                reward, dt, executed_action = execute_first_valid_action_joint(eng, [int(plan[0])], 200.0 - spent)
                if executed_action is None or dt <= 0.0:
                    break
                if hasattr(planner, "observe_executed"):
                    planner.observe_executed(obs_before, int(executed_action), float(dt))
                atoms = split_joint_action(int(executed_action)) if is_joint_action(int(executed_action)) else (int(executed_action),)
                atom_sensors = [xs_decode_action(int(a), MAXT)[1] for a in atoms]
                atom_executed = [
                    (sensor == 0 and float(obs_before.get("s_band_busy_ms", 0.0)) <= 0.0)
                    or (sensor == 1 and float(obs_before.get("x_band_busy_ms", 0.0)) <= 0.0)
                    or sensor not in {0, 1}
                    for sensor in atom_sensors
                ]
                is_search = [xs_decode_action(int(a), MAXT)[0] == 0 and bool(done) for a, done in zip(atoms, atom_executed)]
                if any(is_search):
                    debt = 0.0
                else:
                    debt += float(dt)
                action_rows.append(
                    {
                        "planner": name,
                        "seed": int(seed),
                        "bucket": int(window),
                        "slot": int(executed_total),
                        "action": int(executed_action),
                        "s_action": int(atoms[0]) if len(atoms) > 1 else -1,
                        "x_action": int(atoms[1]) if len(atoms) > 1 else -1,
                        "action_type": "Joint" if len(atoms) > 1 else ("Search" if is_search[0] else "Track"),
                        "reward": float(reward),
                        "dt_ms": float(dt),
                        "s_busy_ms": float(dt) if (float(obs_before.get("s_band_busy_ms", 0.0)) > 0.0 or 0 in atom_sensors) else 0.0,
                        "x_busy_ms": float(dt) if (float(obs_before.get("x_band_busy_ms", 0.0)) > 0.0 or 1 in atom_sensors) else 0.0,
                        "window": int(window),
                        "elapsed_ms": float((window + 1) * 200),
                    }
                )
                reward_total += float(reward)
                spent += float(dt)
                executed_total += 1
                search_total += int(any(is_search))
            cumulative += float(reward_total)
            rows.append(
                {
                    "planner": name,
                    "seed": int(seed),
                    "window": int(window),
                    "elapsed_ms": float((window + 1) * 200),
                    "window_reward": float(reward_total),
                    "cumulative_reward": float(cumulative),
                    "search_fraction": float(search_total / max(1, executed_total)),
                    "planning_ms_per_decision": float(plan_ms),
                    "planning_ms_per_executed_action": float(plan_ms / max(1, executed_total)),
                    "executed_actions": int(executed_total),
                    "spent_ms": float(spent),
                    **sample_state_metrics(eng, debt),
                    **meta_last,
                }
            )
    finally:
        eng.close()
    return pd.DataFrame(rows), pd.DataFrame(action_rows)


def train_best(args):
    return train_variant(args, str(getattr(args, "variant", "two_row_action_attention_factored_loss")), str(getattr(args, "load_state", "")).strip())


def train_variant(args, variant: str, load_state: str = ""):
    load_state = str(load_state).strip()
    if load_state:
        model = train_head(
            variant,
            usable_targets(Path(args.targets))[:1],
            SimpleNamespace(
                d_model=48,
                nhead=4,
                nlayers=2,
                lr=3e-4,
                train_steps=0,
                batch_size=1,
                model_seed=int(args.model_seed),
                q_loss_weight=float(getattr(args, "q_loss_weight", 0.25)),
                value_loss_weight=float(getattr(args, "value_loss_weight", 0.25)),
                search_calibration_weight=0.0,
                log_every=1,
                cell_balanced_sampling=True,
            ),
            torch.device("cpu"),
        )
        state = torch.load(load_state, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        return model.eval()

    train_args = SimpleNamespace(
        d_model=48,
        nhead=4,
        nlayers=2,
        lr=3e-4,
        train_steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        model_seed=int(args.model_seed),
        q_loss_weight=float(getattr(args, "q_loss_weight", 0.25)),
        value_loss_weight=float(getattr(args, "value_loss_weight", 0.25)),
        search_calibration_weight=0.0,
        log_every=max(1, int(args.train_steps)),
        cell_balanced_sampling=True,
    )
    torch.manual_seed(int(args.model_seed))
    np.random.seed(int(args.model_seed))
    targets = usable_targets(Path(args.targets))
    model = train_head(variant, targets, train_args, torch.device("cpu"))
    if not str(args.finetune_targets).strip():
        if str(getattr(args, "save_state", "")).strip():
            Path(args.save_state).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.save_state)
        return model
    finetune_targets = usable_targets(Path(args.finetune_targets))
    finetune_args = SimpleNamespace(**vars(train_args))
    finetune_args.train_steps = int(args.finetune_steps)
    finetune_args.log_every = max(1, int(args.finetune_steps))
    finetune_args.value_loss_weight = float(args.finetune_value_loss_weight)
    model = train_head(
        variant,
        finetune_targets,
        finetune_args,
        torch.device("cpu"),
        model=model,
    )
    if str(getattr(args, "save_state", "")).strip():
        Path(args.save_state).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_state)
    return model


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=str(ROOT / "CreateValid1" / "results" / "edf_bootstrap_r3_lmh_1024_targets.pt"))
    ap.add_argument("--finetune-targets", default=str(ROOT / "CreateValid1" / "results" / "selfplay_adv_edf_owntail_factor_r3_lmh_512_targets.pt"))
    ap.add_argument("--variant", default="two_row_action_attention_factored_loss")
    ap.add_argument("--out", default=str(ROOT / "CreateValid1" / "results" / "best_model_joint_vs_seq_ablation.csv"))
    ap.add_argument("--initials", default="20,40,60")
    ap.add_argument("--rates", default="2,3,4")
    ap.add_argument("--seed", type=int, default=916)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--eval-windows", type=int, default=100)
    ap.add_argument("--train-steps", type=int, default=120)
    ap.add_argument("--finetune-steps", type=int, default=80)
    ap.add_argument("--finetune-value-loss-weight", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--model-seed", type=int, default=123)
    ap.add_argument("--save-state", default="")
    ap.add_argument("--load-state", default="")
    ap.add_argument("--autoregressive-save-state", default="")
    ap.add_argument("--autoregressive-load-state", default="")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--search-bias", type=float, default=-12.0)
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--q-weight", type=float, default=1.0)
    ap.add_argument("--q-loss-weight", type=float, default=0.25)
    ap.add_argument("--value-loss-weight", type=float, default=0.25)
    ap.add_argument("--per-sensor-top", type=int, default=3)
    ap.add_argument("--max-joint-plans", type=int, default=8)
    ap.add_argument("--async-beams", type=int, default=12)
    ap.add_argument("--one-step-potential-weight", type=float, default=1.0)
    ap.add_argument("--force-search-candidate", action="store_true")
    ap.add_argument("--fill-mode", choices=["edf", "search", "model"], default="edf")
    ap.add_argument("--methods", default="")
    ap.add_argument("--score-horizon-ms", type=float, default=800.0)
    ap.add_argument("--sequential-guard-margin", type=float, default=0.0)
    ap.add_argument("--strict-superset", action="store_true")
    ap.add_argument("--force-joint-rescore", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(1)
    method_filter = {x.strip() for x in str(args.methods).split(",") if x.strip()}
    need_autoregressive = (not method_filter) or ("Best_async_autoregressive_direct_PQ" in method_filter)
    model = train_best(args)
    autoregressive_model = None
    if need_autoregressive:
        ar_args = SimpleNamespace(**vars(args))
        ar_args.save_state = str(args.autoregressive_save_state)
        autoregressive_model = train_variant(
            ar_args,
            "two_row_action_attention_autoregressive",
            str(args.autoregressive_load_state),
        )
    if bool(args.train_only):
        if not str(args.save_state).strip():
            raise SystemExit("--train-only requires --save-state")
        print({"trained_state": str(args.save_state)}, flush=True)
        return
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = True
    exact_args.single_sensor = False

    rows = []
    windows = []
    actions = []
    for initial in parse_ints(args.initials):
        for rate in parse_floats(args.rates):
            env_cfg = env_cfg_for(float(rate), exact_args)
            env_cfg["enable_x_band"] = 1
            base = PhysicalHeadPlanner(
                model,
                str(args.variant),
                env_cfg,
                policy_weight=float(args.policy_weight),
                q_weight=float(args.q_weight),
                search_score_bias=float(args.search_bias),
            )
            joint_base = SupersetJointProposalPlanner(
                base,
                per_sensor_top=int(args.per_sensor_top),
                max_joint_plans=int(args.max_joint_plans),
            )
            async_base = AsyncCoupledJointPlanner(
                base,
                per_sensor_top=int(args.per_sensor_top),
                include_search_candidate=bool(args.force_search_candidate),
            )
            async_beam_base = AsyncCoupledBeamProposalPlanner(
                base,
                per_sensor_top=int(args.per_sensor_top),
                beams=int(args.async_beams),
                include_search_candidate=bool(args.force_search_candidate),
            )
            workconserving_base = WorkConservingAsyncCoupledPlanner(
                base,
                per_sensor_top=int(args.per_sensor_top),
                include_search_candidate=bool(args.force_search_candidate),
            )
            workconserving_beam_base = WorkConservingAsyncBeamPlanner(
                base,
                per_sensor_top=int(args.per_sensor_top),
                beams=int(args.async_beams),
                include_search_candidate=bool(args.force_search_candidate),
            )
            autoregressive_direct_base = None
            if autoregressive_model is not None:
                autoregressive_direct_base = AsyncAutoregressiveCoupledPlanner(
                    autoregressive_model,
                    env_cfg,
                    per_sensor_top=int(args.per_sensor_top),
                    policy_weight=float(args.policy_weight),
                    q_weight=float(args.q_weight),
                    search_score_bias=float(args.search_bias),
                )
            joint_only_planner = JointAwareLearnedProposalFairExact(
                env_cfg,
                [joint_base],
                top_k=8,
                score_horizon_ms=float(args.score_horizon_ms),
                slots=96,
                generator="structured",
                seed=15008,
                learned_extra_top_k=2,
                force_learned_rescore=bool(args.force_joint_rescore),
            )
            joint_only_planner.joint_only = True
            async_exact_planner = JointAwareLearnedProposalFairExact(
                env_cfg,
                [async_base],
                top_k=8,
                score_horizon_ms=float(args.score_horizon_ms),
                slots=96,
                generator="structured",
                seed=15008,
                learned_extra_top_k=2,
                force_learned_rescore=bool(args.force_joint_rescore),
            )
            planners = {
                "EDF": EDFPlanner(MAXT),
                "EST": ESTPlanner(MAXT),
                "Best_seq_exact_PQ": JointAwareLearnedProposalFairExact(
                    env_cfg,
                    [base],
                    top_k=8,
                    score_horizon_ms=float(args.score_horizon_ms),
                    slots=96,
                    generator="structured",
                    seed=15008,
                    learned_extra_top_k=2,
                ),
                "Best_joint_exact_PQ": JointAwareLearnedProposalFairExact(
                    env_cfg,
                    [base, joint_base],
                    top_k=8,
                    score_horizon_ms=float(args.score_horizon_ms),
                    slots=96,
                    generator="structured",
                    seed=15008,
                    learned_extra_top_k=2,
                    force_learned_rescore=bool(args.force_joint_rescore),
                ),
                "Best_async_coupled_PQ": async_exact_planner,
                "Best_async_exact_filled_PQ": FillIdleSensorAdapter(async_exact_planner, env_cfg, fill_mode=str(args.fill_mode), scorer=base),
                "Best_async_receding_model_fill_PQ": RecedingModelFillAdapter(async_exact_planner, env_cfg, scorer=base),
                "Best_async_direct_PQ": DirectPlanAdapter(async_base),
                "Best_workconserving_direct_PQ": DirectPlanAdapter(workconserving_base),
                "Best_async_autoregressive_direct_PQ": DirectPlanAdapter(autoregressive_direct_base) if autoregressive_direct_base is not None else None,
                "Best_async_receding_direct_PQ": StatefulRecedingDirectAdapter(async_base),
                "Best_workconserving_receding_direct_PQ": StatefulRecedingDirectAdapter(workconserving_base),
                "Best_async_autoregressive_receding_direct_PQ": StatefulRecedingDirectAdapter(autoregressive_direct_base) if autoregressive_direct_base is not None else None,
                "Best_workconserving_exact_PQ": JointAwareLearnedProposalFairExact(
                    env_cfg,
                    [workconserving_beam_base],
                    top_k=8,
                    score_horizon_ms=float(args.score_horizon_ms),
                    slots=96,
                    generator="structured",
                    seed=15008,
                    learned_extra_top_k=0,
                    force_learned_rescore=True,
                ),
                "Best_async_beam_fullutil_PQ": JointAwareLearnedProposalFairExact(
                    env_cfg,
                    [async_beam_base],
                    top_k=8,
                    score_horizon_ms=float(args.score_horizon_ms),
                    slots=96,
                    generator="structured",
                    seed=15008,
                    learned_extra_top_k=0,
                    force_learned_rescore=True,
                ),
                "Best_async_onestep_fullutil_PQ": AsyncOneStepExactAdapter(
                    async_beam_base,
                    beams=int(args.async_beams),
                    potential_weight=float(args.one_step_potential_weight),
                ),
                "Best_async_receding_fullutil_PQ": JointAwareLearnedProposalFairExact(
                    env_cfg,
                    [async_beam_base],
                    top_k=8,
                    score_horizon_ms=float(args.score_horizon_ms),
                    slots=96,
                    generator="structured",
                    seed=15008,
                    learned_extra_top_k=0,
                    force_learned_rescore=True,
                ),
                "Best_joint_only_PQ": joint_only_planner,
            }
            planners["Best_joint_exact_PQ"].sequential_guard_margin = float(args.sequential_guard_margin)
            planners["Best_joint_exact_PQ"].strict_superset = bool(args.strict_superset)
            planners["Best_workconserving_exact_PQ"].joint_only = True
            planners["Best_async_beam_fullutil_PQ"].joint_only = True
            planners["Best_async_receding_fullutil_PQ"].joint_only = True
            planners = {k: v for k, v in planners.items() if v is not None}
            if method_filter:
                planners = {k: v for k, v in planners.items() if k in method_filter}
            for name, planner in planners.items():
                print({"running": name, "initial": initial, "rate": rate, "seed": int(args.seed)}, flush=True)
                if name in {"EDF", "EST"}:
                    w, a = run_heuristic(planner, name, int(initial), int(args.seed), int(args.eval_windows), engine_env_cfg(env_cfg))
                elif name in {
                    "Best_async_receding_fullutil_PQ",
                    "Best_async_onestep_fullutil_PQ",
                    "Best_async_receding_model_fill_PQ",
                    "Best_workconserving_exact_PQ",
                    "Best_async_receding_direct_PQ",
                    "Best_workconserving_receding_direct_PQ",
                    "Best_async_autoregressive_receding_direct_PQ",
                }:
                    w, a = run_receding_exact_grid_joint(planner, name, int(initial), int(args.seed), int(args.eval_windows), env_cfg)
                else:
                    w, a = run_exact_rescore_grid_joint(planner, name, int(initial), int(args.seed), int(args.eval_windows), env_cfg)
                s = summarize_window_df(w, "fixed")
                denom_ms = float(w["spent_ms"].sum()) if "spent_ms" in w.columns and len(w) else float(max(1, len(w)) * 200.0)
                denom_ms = max(1.0, denom_ms)
                has_busy = not a.empty and {"s_busy_ms", "x_busy_ms"}.issubset(set(a.columns))
                s_util = float(a["s_busy_ms"].sum() / denom_ms) if has_busy else np.nan
                x_util = float(a["x_busy_ms"].sum() / denom_ms) if has_busy else np.nan
                both_util = float(np.minimum(a["s_busy_ms"].to_numpy(), a["x_busy_ms"].to_numpy()).sum() / denom_ms) if has_busy else np.nan
                row = {
                    "method": name,
                    "initial": int(initial),
                    "rate": float(rate),
                    "seed": int(args.seed),
                    "reward": float(s.get("reward_per_200ms_eq", np.nan)),
                    "search": float(s.get("search_fraction", np.nan)),
                    "tracked": float(s.get("mean_tracked_targets", np.nan)),
                    "drop": float(s.get("mean_drop_pct_active", np.nan)),
                    "delay": float(s.get("mean_delay_active", np.nan)),
                    "latency_ms": float(s.get("planning_ms_per_decision", np.nan)),
                    "s_util": float(s_util),
                    "x_util": float(x_util),
                    "mean_sensor_util": float(np.nanmean([s_util, x_util])) if has_busy else np.nan,
                    "both_sensor_util": float(both_util),
                    "windows_completed": int(len(w)),
                }
                rows.append(row)
                windows.append(w.assign(method=name, initial=int(initial), rate=float(rate), seed=int(args.seed)))
                if not a.empty:
                    actions.append(a.assign(method=name, initial=int(initial), rate=float(rate), seed=int(args.seed)))
                print(row, flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out, index=False)
    pd.concat(windows, ignore_index=True).to_csv(out.with_name(out.stem + "_windows.csv"), index=False)
    if actions:
        pd.concat(actions, ignore_index=True).to_csv(out.with_name(out.stem + "_actions.csv"), index=False)
    summary = raw.groupby("method", as_index=False).agg(
        reward=("reward", "mean"),
        search=("search", "mean"),
        tracked=("tracked", "mean"),
        drop=("drop", "mean"),
        delay=("delay", "mean"),
        latency_ms=("latency_ms", "mean"),
        s_util=("s_util", "mean"),
        x_util=("x_util", "mean"),
        mean_sensor_util=("mean_sensor_util", "mean"),
        both_sensor_util=("both_sensor_util", "mean"),
        n=("reward", "size"),
    ).sort_values("reward", ascending=False)
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
