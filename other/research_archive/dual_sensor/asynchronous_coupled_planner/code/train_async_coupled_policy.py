from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "CreateValid1" / "experiments" / "code" / "model_code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from alphazero_orthodox import load_targets, save_targets
from best_model_joint_vs_seq_ablation import (
    AsyncCoupledBeamProposalPlanner,
    AsyncCoupledJointPlanner,
    JointAwareLearnedProposalFairExact,
    WorkConservingAsyncBeamPlanner,
    WorkConservingAsyncCoupledPlanner,
    execute_plan_until_budget_joint_compatible,
)
from exact_env_mutual import (
    MAXT,
    _DummyPlanner,
    attach_env_obs,
    engine_env_cfg,
    env_cfg_for,
    xs_decode_action,
    xs_s_search_action,
    xs_x_search_action,
)
from final_radar_campaign import get_obs
from mutual_features import slot_features, tokenize
from mutual_foundation import SearchTarget
from penalty_window_quota_learner_eval import make_exact_args
from repaired_campaign_tools import build_env
from strict_window_report import sample_state_metrics
from two_sensor_physical_head_eval import PhysicalHeadPlanner, make_physical_model, state_potential, train_head
from realistic_reward_retrain import adapter
from foundation_mcts_fair_eval import physical_candidates
from joint_action_experiment import encode_joint_action, execute_first_valid_action_joint, is_joint_action, split_joint_action
from pufferlib.ocean.radarxs import binding


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def collection_cells(args, seed: int) -> list[tuple[int, float]]:
    if int(getattr(args, "random_cells", 0)) <= 0:
        return [(int(initial), float(rate)) for initial in parse_ints(args.initials) for rate in parse_floats(args.rates)]
    rng = np.random.default_rng(int(seed) + int(getattr(args, "random_cell_seed_offset", 10007)))
    initials = rng.integers(
        int(args.random_initial_min),
        int(args.random_initial_max) + 1,
        size=int(args.random_cells),
    )
    rates = rng.uniform(float(args.random_rate_min), float(args.random_rate_max), size=int(args.random_cells))
    return [(int(initial), float(rate)) for initial, rate in zip(initials, rates)]


class ExploratoryWorkConservingBeamPlanner(WorkConservingAsyncBeamPlanner):
    """Beam planner with policy-guided plus heuristic root exploration.

    Exact self-play should improve the policy, not be limited to whatever a
    random or early policy already proposes.  This planner keeps the learned
    work-conserving beam, then adds EDF-style legal S/X first actions before
    exact rescoring chooses the actual target.
    """

    def __init__(
        self,
        *args,
        explore_per_sensor_top: int = 8,
        max_explore_pairs: int = 32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.explore_per_sensor_top = max(1, int(explore_per_sensor_top))
        self.max_explore_pairs = max(1, int(max_explore_pairs))

    def _heuristic_ranked_for_sensor(self, obs: dict, sensor: int, selected: set[int]) -> list[tuple[int, int]]:
        local_obs = attach_env_obs(dict(obs), self.env_cfg, True, True)
        out: list[tuple[int, int]] = []
        search_action = xs_s_search_action(MAXT) if int(sensor) == 0 else xs_x_search_action(MAXT)
        out.append((0, int(search_action)))
        for action in physical_candidates(local_obs, top_k=MAXT):
            base, sid = xs_decode_action(int(action), MAXT)
            if sid is None or int(sid) != int(sensor) or int(base) < 0:
                continue
            if int(base) > 0 and int(base) in selected:
                continue
            if int(action) == int(search_action):
                continue
            out.append((int(base), int(action)))
            if len(out) >= self.explore_per_sensor_top:
                break
        return out

    def _heuristic_first_actions(self, obs: dict) -> list[int]:
        local_obs = attach_env_obs(dict(obs), self.env_cfg, True, True)
        s_free = float(local_obs.get("s_band_busy_ms", 0.0)) <= 0.0
        x_free = bool(int(local_obs.get("enable_x_band", 0))) and float(local_obs.get("x_band_busy_ms", 0.0)) <= 0.0
        s_dummy = xs_s_search_action(MAXT)
        x_dummy = xs_x_search_action(MAXT)
        s_ranked = self._heuristic_ranked_for_sensor(local_obs, 0, set()) if s_free else []
        x_ranked = self._heuristic_ranked_for_sensor(local_obs, 1, set()) if x_free else []
        out: list[int] = []
        if s_ranked and x_ranked:
            for s_base, s_action in s_ranked:
                for x_base, x_action in x_ranked:
                    if int(s_base) == 0 and int(x_base) == 0:
                        continue
                    if int(s_base) > 0 and int(s_base) == int(x_base):
                        continue
                    out.append(encode_joint_action(int(s_action), int(x_action)))
                    if len(out) >= self.max_explore_pairs:
                        return out
        elif s_ranked:
            out.extend(encode_joint_action(int(s_action), int(x_dummy)) for _base, s_action in s_ranked)
        elif x_ranked:
            out.extend(encode_joint_action(int(s_dummy), int(x_action)) for _base, x_action in x_ranked)
        return out[: self.max_explore_pairs]

    def plan(self, obs, budget_ms=200):
        plans = list(super().plan(obs, budget_ms=budget_ms))
        seen = {tuple(int(a) for a in plan) for plan in plans if plan}
        for first in self._heuristic_first_actions(obs):
            plan = self._tail_from(obs, int(first), float(budget_ms))
            key = tuple(int(a) for a in plan)
            if key and key not in seen:
                seen.add(key)
                plans.append([int(a) for a in key])
        return plans


def action_duration(obs: dict, action: int) -> float:
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


def action_value_proxy(obs: dict, action: int) -> float:
    base, sensor = xs_decode_action(int(action), MAXT)
    if int(base) == 0:
        return 0.0
    if int(base) <= 0:
        return -1.0
    deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
    desired = np.asarray(obs.get("t_desired", []), dtype=np.float32)
    idx = int(base) - 1
    if idx >= len(deadline):
        return -1.0
    urgency = max(0.0, 300.0 - float(deadline[idx])) / 300.0
    overdue = max(0.0, -float(desired[idx])) / 300.0 if idx < len(desired) else 0.0
    return 1.0 + urgency + overdue + (0.15 if int(sensor) == 1 else 0.0)


def action_service_pressure(obs: dict, action: int) -> float:
    """State-local credit for servicing a stale or deadline-pressured target."""
    base, sensor = xs_decode_action(int(action), MAXT)
    if int(base) <= 0:
        return 0.0
    active = np.asarray(obs.get("active_mask", []), dtype=bool)
    deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
    desired = np.asarray(obs.get("t_desired", []), dtype=np.float32)
    idx = int(base) - 1
    if idx < 0 or idx >= len(active) or not bool(active[idx]) or idx >= len(deadline) or float(deadline[idx]) < 0.0:
        return 0.0
    desired_i = float(desired[idx]) if idx < len(desired) else 0.0
    deadline_i = float(deadline[idx])
    slack = max(1.0, deadline_i - desired_i)
    lateness = max(0.0, -desired_i) / slack
    deadline_pressure = max(0.0, 300.0 - deadline_i) / 300.0
    return float(lateness + deadline_pressure)


def add_target(
    targets: list[SearchTarget],
    rows: list[dict],
    adapt,
    obs: dict,
    selected: set[int],
    search_count: int,
    track_count: int,
    last: int,
    spent: float,
    actions: list[int],
    meta: dict,
    q_value: float | None = None,
):
    sensor_pi = np.zeros((MAXT + 1, 2), dtype=np.float32)
    sensor_q = np.zeros((MAXT + 1, 2), dtype=np.float32)
    sensor_q_mask = np.zeros((MAXT + 1, 2), dtype=np.float32)
    pair_pi = np.zeros((MAXT + 1, MAXT + 1), dtype=np.float32)
    real_actions = []
    for action in actions:
        base, sensor = xs_decode_action(int(action), MAXT)
        if sensor is None or int(sensor) not in {0, 1} or int(base) < 0:
            continue
        busy_key = "s_band_busy_ms" if int(sensor) == 0 else "x_band_busy_ms"
        if float(obs.get(busy_key, 0.0)) > 0.0:
            continue
        real_actions.append((int(base), int(sensor), int(action)))
    if not real_actions:
        return
    # A joint S/X decision supervises both sensor heads.  Each free sensor gets
    # one unit of policy mass, matching the rest of the two-row target format.
    mass = 1.0
    for base, sensor, action in real_actions:
        sensor_pi[int(base), int(sensor)] += mass
        sensor_q[int(base), int(sensor)] = float(q_value) if q_value is not None else action_value_proxy(obs, int(action))
        sensor_q_mask[int(base), int(sensor)] = 1.0
    s_bases = [int(base) for base, sensor, _action in real_actions if int(sensor) == 0]
    x_bases = [int(base) for base, sensor, _action in real_actions if int(sensor) == 1]
    if s_bases and x_bases:
        pair_pi[int(s_bases[0]), int(x_bases[0])] = 1.0
    pi = sensor_pi.sum(axis=1) / max(1, len(real_actions))
    q = sensor_q.max(axis=1)
    q_mask = (sensor_q_mask.max(axis=1) > 0.5).astype(np.float32)
    tok = tokenize(adapt, obs, selected=selected, search_count=search_count).astype(np.float32)
    slot = slot_features(obs, spent, search_count, track_count, last, 200.0).astype(np.float32)
    targets.append(
        SearchTarget(
            tok,
            slot,
            pi.astype(np.float32),
            q.astype(np.float32),
            q_mask.astype(np.float32),
            int(search_count),
            int(track_count),
            reward=0.0,
            ret=float(sensor_q[sensor_q_mask > 0.5].max()) if bool((sensor_q_mask > 0.5).any()) else 0.0,
            sensor_pi=sensor_pi,
            sensor_q=sensor_q,
            sensor_q_mask=sensor_q_mask,
            initial=int(meta["initial"]),
            rate=float(meta["rate"]),
            seed=int(meta["seed"]),
            window=int(meta["window"]),
            action_index=len(targets),
            pair_pi=pair_pi,
        )
    )
    rows.append(
        {
            **meta,
            "target_idx": int(len(targets) - 1),
            "search_mass": float(sensor_pi[0, :].sum()),
            "x_mass": float(sensor_pi[:, 1].sum()),
            "actions": len(real_actions),
            "q_value": np.nan if q_value is None else float(q_value),
        }
    )


def add_candidate_q_target(
    targets: list[SearchTarget],
    rows: list[dict],
    adapt,
    obs: dict,
    selected: set[int],
    search_count: int,
    track_count: int,
    last: int,
    spent: float,
    scored_actions: list[tuple[float, int]],
    meta: dict,
    policy_tau: float = 1.0,
    center_q: bool = False,
):
    usable: list[tuple[float, list[tuple[int, int, int]]]] = []
    for score, action in scored_actions:
        atoms = list(split_joint_action(int(action))) if is_joint_action(int(action)) else [int(action)]
        real_actions = []
        for atom in atoms:
            base, sensor = xs_decode_action(int(atom), MAXT)
            if sensor is None or int(sensor) not in {0, 1} or int(base) < 0:
                continue
            busy_key = "s_band_busy_ms" if int(sensor) == 0 else "x_band_busy_ms"
            if float(obs.get(busy_key, 0.0)) > 0.0:
                continue
            real_actions.append((int(base), int(sensor), int(atom)))
        if real_actions:
            usable.append((float(score), real_actions))
    if not usable:
        return

    sensor_pi = np.zeros((MAXT + 1, 2), dtype=np.float32)
    sensor_q = np.full((MAXT + 1, 2), -1e9, dtype=np.float32)
    sensor_q_mask = np.zeros((MAXT + 1, 2), dtype=np.float32)
    pair_pi = np.zeros((MAXT + 1, MAXT + 1), dtype=np.float32)

    vals = np.asarray([s for s, _atoms in usable], dtype=np.float32)
    logits = (vals - float(vals.max())) / max(1e-6, float(policy_tau))
    probs = np.exp(logits)
    probs = probs / max(1e-6, float(probs.sum()))
    q_offset = float(vals.mean()) if bool(center_q) else 0.0
    for prob, (score, atoms) in zip(probs, usable):
        for base, sensor, _atom in atoms:
            sensor_pi[int(base), int(sensor)] += float(prob)
            sensor_q[int(base), int(sensor)] = max(float(sensor_q[int(base), int(sensor)]), float(score) - q_offset)
            sensor_q_mask[int(base), int(sensor)] = 1.0
        s_bases = [int(base) for base, sensor, _atom in atoms if int(sensor) == 0]
        x_bases = [int(base) for base, sensor, _atom in atoms if int(sensor) == 1]
        if s_bases and x_bases:
            pair_pi[int(s_bases[0]), int(x_bases[0])] += float(prob)
    sensor_q[sensor_q_mask < 0.5] = 0.0
    pair_total = float(pair_pi.sum())
    if pair_total > 1e-6:
        pair_pi /= pair_total

    pi = sensor_pi.sum(axis=1) / max(1.0, float(sensor_pi.sum()))
    q = sensor_q.max(axis=1)
    q_mask = (sensor_q_mask.max(axis=1) > 0.5).astype(np.float32)
    tok = tokenize(adapt, obs, selected=selected, search_count=search_count).astype(np.float32)
    slot = slot_features(obs, spent, search_count, track_count, last, 200.0).astype(np.float32)
    targets.append(
        SearchTarget(
            tok,
            slot,
            pi.astype(np.float32),
            q.astype(np.float32),
            q_mask.astype(np.float32),
            int(search_count),
            int(track_count),
            reward=0.0,
            ret=float(vals.max()),
            sensor_pi=sensor_pi,
            sensor_q=sensor_q,
            sensor_q_mask=sensor_q_mask,
            initial=int(meta["initial"]),
            rate=float(meta["rate"]),
            seed=int(meta["seed"]),
            window=int(meta["window"]),
            action_index=len(targets),
            pair_pi=pair_pi,
        )
    )
    rows.append(
        {
            **meta,
            "target_idx": int(len(targets) - 1),
            "search_mass": float(sensor_pi[0, :].sum()),
            "x_mass": float(sensor_pi[:, 1].sum()),
            "actions": int(sum(len(atoms) for _score, atoms in usable)),
            "candidates": int(len(usable)),
            "q_value": float(vals.max()),
            "q_gap": float(vals.max() - vals.min()) if len(vals) > 1 else 0.0,
            "q_centered": bool(center_q),
        }
    )


def rollout_joint_plan_value(eng, plan: list[int], debt: float, env_cfg: dict, horizon_ms: float, potential_weight: float) -> float:
    snapshot = binding.vec_snapshot(eng.env)
    try:
        reward, _spent, next_debt, _executed, _searches, _rows = execute_plan_until_budget_joint_compatible(
            eng,
            [int(a) for a in plan],
            float(horizon_ms),
            float(debt),
            "teacher_value",
            0,
            0,
        )
        value = float(reward) + float(potential_weight) * state_potential(eng, float(next_debt))
    finally:
        binding.vec_restore(eng.env, snapshot)
    return float(value)


def operational_potential(eng, debt: float, drop_weight: float = 2.0, delay_weight: float = 0.02) -> float:
    """Operational target-health potential aligned with reporting metrics."""
    m = sample_state_metrics(eng, float(debt))
    return float(m["tracked_targets"]) - float(drop_weight) * float(m["drop_pct_active"]) - float(delay_weight) * float(m["mean_delay_active"])


def candidate_potential(eng, debt: float, args) -> float:
    mode = str(getattr(args, "candidate_score_mode", "reward_potential")).strip().lower()
    if mode == "operational":
        return operational_potential(
            eng,
            float(debt),
            drop_weight=float(getattr(args, "operational_drop_weight", 2.0)),
            delay_weight=float(getattr(args, "operational_delay_weight", 0.02)),
        )
    return state_potential(eng, float(debt))


def score_rollout_candidates(
    eng,
    obs: dict,
    debt: float,
    planner,
    selected: set[int],
    elapsed: float,
    search_count: int,
    track_count: int,
    last: int,
    beams: int,
    potential_weight: float,
    rollout_ms: float,
    exhaustive_roots: bool = False,
    exhaustive_per_sensor_top: int = 8,
    score_args=None,
) -> list[tuple[float, int]]:
    if bool(exhaustive_roots):
        raw = []
        for action in physical_candidates(obs, top_k=MAXT):
            base, sensor = xs_decode_action(int(action), MAXT)
            if sensor is None or int(sensor) not in {0, 1} or int(base) < 0:
                continue
            if int(base) > 0 and int(base) in selected:
                continue
            raw.append((int(base), int(sensor), int(action)))
        by_sensor = {0: [(0, 0, xs_s_search_action(MAXT))], 1: [(0, 1, xs_x_search_action(MAXT))]}
        for base, sensor, action in raw:
            if int(base) == 0:
                continue
            if len(by_sensor[int(sensor)]) < max(1, int(exhaustive_per_sensor_top)):
                by_sensor[int(sensor)].append((int(base), int(sensor), int(action)))
        candidates = []
        for s_base, _s_sensor, s_action in by_sensor[0]:
            for x_base, _x_sensor, x_action in by_sensor[1]:
                if int(s_base) > 0 and int(s_base) == int(x_base):
                    continue
                candidates.append((0.0, encode_joint_action(int(s_action), int(x_action))))
        candidates = list(dict((int(a), (float(s), int(a))) for s, a in candidates).values())
    else:
        candidates = planner._candidate_pairs(
            obs,
            set(selected),
            float(elapsed),
            int(search_count),
            int(track_count),
            int(last),
        )[: max(1, int(beams))]
    if not candidates:
        return []
    root = binding.vec_snapshot(eng.env)
    before = candidate_potential(eng, float(debt), score_args) if score_args is not None else state_potential(eng, float(debt))
    service_weight = float(getattr(score_args, "candidate_service_weight", 0.0)) if score_args is not None else 0.0
    scored: list[tuple[float, int]] = []
    try:
        for prior_score, action in candidates:
            binding.vec_restore(eng.env, root)
            obs_before_action = attach_env_obs(get_obs(eng, float(debt)), planner.env_cfg, True, True)
            reward, dt, executed = execute_first_valid_action_joint(eng, [int(action)], 200.0)
            if executed is None or float(dt) <= 0.0:
                continue
            atoms = list(split_joint_action(int(executed))) if is_joint_action(int(executed)) else [int(executed)]
            service_score = sum(action_service_pressure(obs_before_action, int(atom)) for atom in atoms)
            any_search = any(xs_decode_action(int(atom), MAXT)[0] == 0 for atom in atoms)
            next_debt = 0.0 if any_search else float(debt) + float(dt)
            total_reward = float(reward)
            local_elapsed = float(dt)
            local_search_count = int(search_count) + int(any_search)
            local_track_count = int(track_count) + sum(1 for atom in atoms if xs_decode_action(int(atom), MAXT)[0] > 0)
            local_last = int(xs_decode_action(int(atoms[-1]), MAXT)[0]) if atoms else int(last)
            local_selected = set(selected)
            for atom in atoms:
                base, _sensor = xs_decode_action(int(atom), MAXT)
                if int(base) > 0:
                    local_selected.add(int(base))
            while local_elapsed < float(rollout_ms) and not bool(eng.term_buf[0]):
                sim_obs = attach_env_obs(get_obs(eng, next_debt), planner.env_cfg, True, True)
                next_action = planner._choose_pair(
                    sim_obs,
                    local_selected,
                    local_elapsed,
                    local_search_count,
                    local_track_count,
                    local_last,
                )
                if next_action is None:
                    break
                obs_before_action2 = attach_env_obs(get_obs(eng, float(next_debt)), planner.env_cfg, True, True)
                reward2, dt2, executed2 = execute_first_valid_action_joint(
                    eng,
                    [int(next_action)],
                    max(1.0, float(rollout_ms) - local_elapsed),
                )
                if executed2 is None or float(dt2) <= 0.0:
                    break
                atoms2 = list(split_joint_action(int(executed2))) if is_joint_action(int(executed2)) else [int(executed2)]
                service_score += sum(action_service_pressure(obs_before_action2, int(atom)) for atom in atoms2)
                any_search2 = any(xs_decode_action(int(atom), MAXT)[0] == 0 for atom in atoms2)
                total_reward += float(reward2)
                local_elapsed += float(dt2)
                next_debt = 0.0 if any_search2 else float(next_debt) + float(dt2)
                local_search_count += int(any_search2)
                for atom in atoms2:
                    base, _sensor = xs_decode_action(int(atom), MAXT)
                    if int(base) > 0:
                        local_selected.add(int(base))
                        local_track_count += 1
                    local_last = int(base)
            after = candidate_potential(eng, float(next_debt), score_args) if score_args is not None else state_potential(eng, float(next_debt))
            if score_args is not None and str(getattr(score_args, "candidate_score_mode", "reward_potential")).strip().lower() == "operational":
                score = float(potential_weight) * (float(after) - float(before)) + float(service_weight) * float(service_score) + 0.001 * float(prior_score)
            else:
                score = float(total_reward) + float(potential_weight) * (float(after) - float(before)) + float(service_weight) * float(service_score) + 0.001 * float(prior_score)
            scored.append((score, int(action)))
    finally:
        binding.vec_restore(eng.env, root)
    return sorted(scored, reverse=True, key=lambda x: x[0])


def choose_onestep_improved_action(
    eng,
    obs: dict,
    debt: float,
    planner,
    selected: set[int],
    elapsed: float,
    search_count: int,
    track_count: int,
    last: int,
    beams: int,
    potential_weight: float,
    rollout_ms: float = 0.0,
) -> tuple[int | None, float]:
    candidates = planner._candidate_pairs(
        obs,
        set(selected),
        float(elapsed),
        int(search_count),
        int(track_count),
        int(last),
    )[: max(1, int(beams))]
    if not candidates:
        return None, float("nan")
    root = binding.vec_snapshot(eng.env)
    before = state_potential(eng, float(debt))
    scored: list[tuple[float, int]] = []
    try:
        for prior_score, action in candidates:
            binding.vec_restore(eng.env, root)
            reward, dt, executed = execute_first_valid_action_joint(eng, [int(action)], 200.0)
            if executed is None or float(dt) <= 0.0:
                continue
            atoms = list(split_joint_action(int(executed))) if is_joint_action(int(executed)) else [int(executed)]
            any_search = any(xs_decode_action(int(atom), MAXT)[0] == 0 for atom in atoms)
            next_debt = 0.0 if any_search else float(debt) + float(dt)
            total_reward = float(reward)
            local_elapsed = float(dt)
            local_search_count = int(search_count) + int(any_search)
            local_track_count = int(track_count) + sum(1 for atom in atoms if xs_decode_action(int(atom), MAXT)[0] > 0)
            local_last = int(xs_decode_action(int(atoms[-1]), MAXT)[0]) if atoms else int(last)
            local_selected = set(selected)
            for atom in atoms:
                base, _sensor = xs_decode_action(int(atom), MAXT)
                if int(base) > 0:
                    local_selected.add(int(base))
            while local_elapsed < float(rollout_ms) and not bool(eng.term_buf[0]):
                sim_obs = attach_env_obs(get_obs(eng, next_debt), planner.env_cfg, True, True)
                next_action = planner._choose_pair(
                    sim_obs,
                    local_selected,
                    local_elapsed,
                    local_search_count,
                    local_track_count,
                    local_last,
                )
                if next_action is None:
                    break
                reward2, dt2, executed2 = execute_first_valid_action_joint(
                    eng,
                    [int(next_action)],
                    max(1.0, float(rollout_ms) - local_elapsed),
                )
                if executed2 is None or float(dt2) <= 0.0:
                    break
                atoms2 = list(split_joint_action(int(executed2))) if is_joint_action(int(executed2)) else [int(executed2)]
                any_search2 = any(xs_decode_action(int(atom), MAXT)[0] == 0 for atom in atoms2)
                total_reward += float(reward2)
                local_elapsed += float(dt2)
                next_debt = 0.0 if any_search2 else float(next_debt) + float(dt2)
                local_search_count += int(any_search2)
                for atom in atoms2:
                    base, _sensor = xs_decode_action(int(atom), MAXT)
                    if int(base) > 0:
                        local_selected.add(int(base))
                        local_track_count += 1
                    local_last = int(base)
            after = state_potential(eng, float(next_debt))
            score = float(total_reward) + float(potential_weight) * (float(after) - float(before)) + 0.001 * float(prior_score)
            scored.append((score, int(action)))
    finally:
        binding.vec_restore(eng.env, root)
    if not scored:
        return None, float("nan")
    score, action = max(scored, key=lambda x: x[0])
    return int(action), float(score)


def collect_async_targets(args, model, exact_args) -> list[SearchTarget]:
    adapt = adapter()
    out = Path(args.targets_out)
    targets: list[SearchTarget] = []
    rows: list[dict] = []
    if bool(getattr(args, "resume_targets", False)) and out.exists():
        targets = list(load_targets(out))
        csv_path = out.with_suffix(".csv")
        if csv_path.exists():
            rows = pd.read_csv(csv_path).to_dict("records")
        print({"resume_targets": str(out), "loaded_targets": len(targets), "loaded_rows": len(rows)}, flush=True)

    save_every = max(0, int(getattr(args, "save_every_targets", 0)))
    last_saved = len(targets)

    def checkpoint(force: bool = False):
        nonlocal last_saved
        if not force and (save_every <= 0 or len(targets) - last_saved < save_every):
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        save_targets(out, targets)
        pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
        last_saved = len(targets)
        print({"checkpoint_targets": len(targets), "path": str(out)}, flush=True)

    completed: set[tuple[int, int, float]] = set()
    existing_counts: dict[tuple[int, int, float], int] = {}
    per_cell_limit = int(getattr(args, "max_targets_per_cell", 0))
    if bool(getattr(args, "resume_targets", False)) and per_cell_limit > 0 and rows:
        df = pd.DataFrame(rows)
        if {"seed", "initial", "rate"}.issubset(df.columns):
            counts = df.groupby(["seed", "initial", "rate"]).size()
            for (seed, initial, rate), count in counts.items():
                existing_counts[(int(seed), int(initial), float(rate))] = int(count)
                if int(count) >= per_cell_limit:
                    completed.add((int(seed), int(initial), float(rate)))
    for seed in parse_ints(args.train_seeds):
        for cell_idx, (initial, rate) in enumerate(collection_cells(args, int(seed))):
                cell_key = (int(seed), int(initial), float(rate))
                if cell_key in completed:
                    print({"skip_completed_cell": True, "initial": initial, "rate": rate, "seed": seed, "cell_idx": cell_idx}, flush=True)
                    continue
                existing_cell_count = int(existing_counts.get(cell_key, 0))
                cell_remaining = (
                    max(0, per_cell_limit - existing_cell_count)
                    if per_cell_limit > 0
                    else int(args.max_targets)
                )
                if cell_remaining <= 0:
                    continue
                cell_start = len(targets)
                env_cfg = env_cfg_for(float(rate), exact_args)
                env_cfg["enable_x_band"] = 1
                if bool(getattr(args, "use_arrival_token_feature", False)):
                    env_cfg["use_arrival_token_feature"] = 1.0
                base = PhysicalHeadPlanner(
                    model,
                    str(args.train_variant),
                    env_cfg,
                    policy_weight=1.0,
                    q_weight=float(args.q_weight),
                    search_score_bias=float(args.search_bias),
                )
                include_search = bool(getattr(args, "include_search_candidate", False))
                if str(args.teacher_mode).startswith("workconserving"):
                    teacher = WorkConservingAsyncCoupledPlanner(base, per_sensor_top=int(args.per_sensor_top), include_search_candidate=include_search)
                    beam_cls = ExploratoryWorkConservingBeamPlanner if bool(getattr(args, "explore_heuristic_roots", False)) else WorkConservingAsyncBeamPlanner
                    beam_kwargs = {}
                    if beam_cls is ExploratoryWorkConservingBeamPlanner:
                        beam_kwargs = {
                            "explore_per_sensor_top": int(args.explore_per_sensor_top),
                            "max_explore_pairs": int(args.max_explore_pairs),
                        }
                    beam_teacher = beam_cls(
                        base,
                        per_sensor_top=int(args.per_sensor_top),
                        beams=int(args.async_beams),
                        include_search_candidate=include_search,
                        **beam_kwargs,
                    )
                else:
                    teacher = AsyncCoupledJointPlanner(base, per_sensor_top=int(args.per_sensor_top), include_search_candidate=include_search)
                    beam_teacher = AsyncCoupledBeamProposalPlanner(base, per_sensor_top=int(args.per_sensor_top), beams=int(args.async_beams), include_search_candidate=include_search)
                exact_teacher = None
                if str(args.teacher_mode) in {"async_exact", "async_beam_exact", "workconserving_exact", "workconserving_beam_exact"}:
                    learned_planner = beam_teacher if "beam" in str(args.teacher_mode) else teacher
                    exact_teacher = JointAwareLearnedProposalFairExact(
                        env_cfg,
                        [learned_planner],
                        top_k=int(args.exact_top_k),
                        score_horizon_ms=float(args.score_horizon_ms),
                        slots=96,
                        generator="structured",
                        seed=15008,
                        learned_extra_top_k=2,
                        force_learned_rescore=True,
                    )
                    exact_teacher.joint_only = str(args.teacher_mode).startswith("workconserving") or "beam_exact" in str(args.teacher_mode)
                eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
                eng.reset(seed=int(seed))
                debt = 0.0
                cell_seen = 0
                try:
                    for window in range(int(args.windows)):
                        window_start = len(targets)
                        spent = 0.0
                        selected: set[int] = set()
                        search_count = 0
                        track_count = 0
                        last = -1
                        pending_plan: list[int] = []
                        while (
                            spent < 200.0
                            and len(targets) < int(args.max_targets)
                            and (
                                per_cell_limit <= 0
                                or (len(targets) - cell_start) < int(cell_remaining)
                            )
                            and (
                                int(getattr(args, "max_targets_per_window", 0)) <= 0
                                or (len(targets) - window_start) < int(args.max_targets_per_window)
                            )
                            and not bool(eng.term_buf[0])
                        ):
                            obs = attach_env_obs(get_obs(eng, debt), env_cfg, True, True)
                            should_record = (
                                int(cell_seen) >= int(existing_cell_count)
                                and int(window) >= int(getattr(args, "min_record_window", 0))
                                and float(spent) >= float(getattr(args, "min_record_elapsed_ms", 0.0))
                                and int(track_count + search_count) >= int(getattr(args, "min_record_actions_in_window", 0))
                                and int(track_count) >= int(getattr(args, "min_record_tracks_in_window", 0))
                            )
                            if str(args.teacher_mode) in {"async_onestep", "workconserving_onestep", "workconserving_candidate_qpolicy"}:
                                selected_for_model = set(selected) if bool(getattr(args, "carry_selected_mask", False)) else set()
                                if not should_record and int(window) < int(getattr(args, "min_record_window", 0)):
                                    scored_candidates = []
                                    action = teacher._choose_pair(
                                        obs,
                                        selected_for_model,
                                        spent,
                                        search_count,
                                        track_count,
                                        last,
                                    )
                                    q_value = float("nan")
                                else:
                                    scored_candidates = score_rollout_candidates(
                                        eng,
                                        obs,
                                        float(debt),
                                        teacher,
                                        selected_for_model,
                                        spent,
                                        search_count,
                                        track_count,
                                        last,
                                        int(args.async_beams),
                                        float(args.q_label_potential_weight),
                                        float(args.teacher_rollout_ms),
                                        exhaustive_roots=bool(getattr(args, "exhaustive_root_candidates", False)),
                                        exhaustive_per_sensor_top=int(getattr(args, "exhaustive_per_sensor_top", 8)),
                                        score_args=args,
                                    )
                                if str(args.teacher_mode) == "workconserving_candidate_qpolicy":
                                    if should_record:
                                        add_candidate_q_target(
                                            targets,
                                            rows,
                                            adapt,
                                            obs,
                                            selected_for_model,
                                            search_count,
                                            track_count,
                                            last,
                                            spent,
                                            scored_candidates,
                                            {"initial": initial, "rate": rate, "seed": seed, "window": window, "cell_idx": cell_idx},
                                            policy_tau=float(args.candidate_policy_tau),
                                            center_q=bool(args.center_candidate_q),
                                        )
                                    if scored_candidates:
                                        action = int(scored_candidates[0][1])
                                        q_value = float(scored_candidates[0][0])
                                else:
                                    action = int(scored_candidates[0][1]) if scored_candidates else None
                                    q_value = float(scored_candidates[0][0]) if scored_candidates else float("nan")
                                plan = [] if action is None else [int(action)]
                            elif exact_teacher is not None:
                                if not pending_plan:
                                    plan, _meta = exact_teacher.choose(eng, debt, obs)
                                    pending_plan = [int(a) for a in list(plan)]
                                plan = [int(pending_plan.pop(0))] if pending_plan else []
                            elif str(args.teacher_mode) in {"async_beam_direct", "workconserving_beam_direct"}:
                                plans = beam_teacher.plan(obs, budget_ms=max(1, int(200.0 - spent)))
                                plan = list(plans[0]) if plans and isinstance(plans[0], (list, tuple, np.ndarray)) else list(plans)
                            else:
                                plan = teacher.plan(obs, budget_ms=max(1, int(200.0 - spent)))
                            if not plan:
                                break
                            action = int(plan[0])
                            atoms = list(split_joint_action(action)) if is_joint_action(action) else [action]
                            if str(args.teacher_mode) not in {"async_onestep", "workconserving_onestep", "workconserving_candidate_qpolicy"}:
                                q_value = None
                            if str(args.q_label) == "rollout":
                                q_value = rollout_joint_plan_value(
                                    eng,
                                    list(plan),
                                    float(debt),
                                    env_cfg,
                                    float(args.q_label_horizon_ms),
                                    float(args.q_label_potential_weight),
                                )
                            if should_record and str(args.teacher_mode) != "workconserving_candidate_qpolicy":
                                add_target(
                                    targets,
                                    rows,
                                    adapt,
                                    obs,
                                    set(selected) if bool(getattr(args, "carry_selected_mask", False)) else set(),
                                    search_count,
                                    track_count,
                                    last,
                                    spent,
                                    atoms,
                                    {"initial": initial, "rate": rate, "seed": seed, "window": window, "cell_idx": cell_idx},
                                    q_value=q_value,
                                )
                            if should_record:
                                checkpoint(force=False)
                            reward, dt, executed = execute_first_valid_action_joint(eng, [action], 200.0 - spent)
                            if executed is None or float(dt) <= 0.0:
                                break
                            spent += float(dt)
                            atoms_ex = list(split_joint_action(int(executed))) if is_joint_action(int(executed)) else [int(executed)]
                            any_search = False
                            for atom in atoms_ex:
                                base_id, sensor = xs_decode_action(int(atom), MAXT)
                                if sensor is None or int(sensor) not in {0, 1}:
                                    continue
                                if int(base_id) == 0:
                                    search_count += 1
                                    any_search = True
                                elif int(base_id) > 0:
                                    selected.add(int(base_id))
                                    track_count += 1
                                last = int(base_id)
                            debt = 0.0 if any_search else debt + float(dt)
                            cell_seen += 1
                        if len(targets) >= int(args.max_targets):
                            break
                        if per_cell_limit > 0 and (len(targets) - cell_start) >= int(cell_remaining):
                            break
                finally:
                    eng.close()
                print({"targets": len(targets), "initial": initial, "rate": rate, "seed": seed, "cell_idx": cell_idx}, flush=True)
                checkpoint(force=True)
                if len(targets) >= int(args.max_targets):
                    break
        if len(targets) >= int(args.max_targets):
            break
    checkpoint(force=True)
    return targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-state", default="")
    ap.add_argument("--targets-out", default=str(ROOT / "CreateValid1" / "results" / "async_coupled_teacher_targets.pt"))
    ap.add_argument("--save-state", default=str(ROOT / "CreateValid1" / "results" / "async_coupled_finetuned_state.pt"))
    ap.add_argument("--resume-targets", action="store_true")
    ap.add_argument("--save-every-targets", type=int, default=1)
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--initials", default="20,40,60")
    ap.add_argument("--rates", default="2,3,4")
    ap.add_argument("--train-seeds", default="916")
    ap.add_argument("--random-cells", type=int, default=0)
    ap.add_argument("--random-cell-seed-offset", type=int, default=10007)
    ap.add_argument("--random-initial-min", type=int, default=20)
    ap.add_argument("--random-initial-max", type=int, default=60)
    ap.add_argument("--random-rate-min", type=float, default=2.0)
    ap.add_argument("--random-rate-max", type=float, default=4.0)
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--max-targets", type=int, default=1024)
    ap.add_argument("--max-targets-per-cell", type=int, default=0)
    ap.add_argument("--max-targets-per-window", type=int, default=0)
    ap.add_argument("--train-steps", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--model-seed", type=int, default=123)
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--cell-balanced-sampling", action="store_true", default=True)
    ap.add_argument("--hard-policy-target", action="store_true")
    ap.add_argument("--value-loss-weight", type=float, default=0.0)
    ap.add_argument("--q-loss-weight", type=float, default=0.25)
    ap.add_argument("--policy-loss-weight", type=float, default=1.0)
    ap.add_argument("--l2sp-weight", type=float, default=0.0)
    ap.add_argument("--freeze-non-q", action="store_true")
    ap.add_argument("--freeze-non-busy-context", action="store_true")
    ap.add_argument("--search-bias", type=float, default=-12.0)
    ap.add_argument("--q-weight", type=float, default=1.0)
    ap.add_argument("--q-label", choices=["proxy", "rollout"], default="proxy")
    ap.add_argument("--q-label-horizon-ms", type=float, default=200.0)
    ap.add_argument("--q-label-potential-weight", type=float, default=1.0)
    ap.add_argument("--per-sensor-top", type=int, default=3)
    ap.add_argument("--include-search-candidate", action="store_true")
    ap.add_argument(
        "--teacher-mode",
        choices=[
            "async_direct",
            "async_beam_direct",
            "async_exact",
            "async_beam_exact",
            "workconserving_direct",
            "workconserving_beam_direct",
            "async_onestep",
            "workconserving_onestep",
            "workconserving_candidate_qpolicy",
            "workconserving_exact",
            "workconserving_beam_exact",
        ],
        default="async_direct",
    )
    ap.add_argument("--exact-top-k", type=int, default=8)
    ap.add_argument("--score-horizon-ms", type=float, default=800.0)
    ap.add_argument("--async-beams", type=int, default=16)
    ap.add_argument("--explore-heuristic-roots", action="store_true")
    ap.add_argument("--explore-per-sensor-top", type=int, default=8)
    ap.add_argument("--max-explore-pairs", type=int, default=32)
    ap.add_argument("--teacher-rollout-ms", type=float, default=0.0)
    ap.add_argument("--candidate-score-mode", choices=["reward_potential", "operational"], default="reward_potential")
    ap.add_argument("--operational-drop-weight", type=float, default=2.0)
    ap.add_argument("--operational-delay-weight", type=float, default=0.02)
    ap.add_argument("--candidate-service-weight", type=float, default=0.0)
    ap.add_argument("--candidate-policy-tau", type=float, default=1.0)
    ap.add_argument("--center-candidate-q", action="store_true")
    ap.add_argument("--exhaustive-root-candidates", action="store_true")
    ap.add_argument("--exhaustive-per-sensor-top", type=int, default=8)
    ap.add_argument("--train-variant", default="two_row_action_attention_factored_loss")
    ap.add_argument("--log-every", type=int, default=40)
    ap.add_argument("--use-arrival-token-feature", action="store_true")
    ap.add_argument("--carry-selected-mask", action="store_true")
    ap.add_argument("--min-record-elapsed-ms", type=float, default=0.0)
    ap.add_argument("--min-record-actions-in-window", type=int, default=0)
    ap.add_argument("--min-record-tracks-in-window", type=int, default=0)
    ap.add_argument("--min-record-window", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(1)
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = True
    exact_args.single_sensor = False
    model = make_physical_model(str(args.train_variant), args)
    if str(args.base_state).strip():
        state = torch.load(args.base_state, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            {
                "loaded_base_state": str(args.base_state),
                "missing_keys": list(missing),
                "unexpected_keys": list(unexpected),
            },
            flush=True,
        )
    else:
        print({"loaded_base_state": None, "init": "scratch"}, flush=True)
    model.eval()
    targets = collect_async_targets(args, model, exact_args)
    if bool(args.collect_only):
        print({"collect_only": True, "targets": len(targets), "targets_out": str(args.targets_out)}, flush=True)
        return
    tuned = train_head(str(args.train_variant), targets, args, torch.device("cpu"), model=model)
    Path(args.save_state).parent.mkdir(parents=True, exist_ok=True)
    torch.save(tuned.state_dict(), args.save_state)
    print({"saved_state": args.save_state, "targets": len(targets)}, flush=True)


if __name__ == "__main__":
    main()
