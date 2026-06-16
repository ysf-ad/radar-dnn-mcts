from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exact_env_mutual import EDFPlanner, ESTPlanner, MAXT, SnapshotSimulator, ExactEnvMCTS, best_window_plan, _DummyPlanner, engine_env_cfg, env_cfg_for
from exact_env_mutual import attach_env_obs, xs_decode_action, xs_s_search_action, xs_x_search_action, xs_s_track_action, xs_x_track_action
from eval_exact_rescore128 import ExactRescore128
from final_radar_campaign import get_obs, summarize_window_df
from alphazero_orthodox import save_targets
from mutual_features import slot_features, tokenize
from mutual_foundation import SearchTarget
from mutual_foundation import DEVICE, MutualRadarNet
from penalty_window_quota_learner_eval import make_exact_args
from realistic_reward_retrain import adapter
from repaired_campaign_tools import build_env, execute_first_valid_action
from strict_window_report import execute_plan_until_budget, sample_state_metrics
from pufferlib.ocean.radarxs import binding


DEFAULT_CKPT = Path("checkpoints/exact_mutual_latest.pt")


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def load_foundation(path: Path, state_path: Path | None = None):
    model = MutualRadarNet(d_model=96, nhead=4, nlayers=2).to(DEVICE)
    raw = torch.load(path, map_location=DEVICE)
    current = model.state_dict()
    compatible = {k: v for k, v in raw.items() if k in current and tuple(v.shape) == tuple(current[k].shape)}
    skipped = sorted(k for k, v in raw.items() if k in current and tuple(v.shape) != tuple(current[k].shape))
    model.load_state_dict(compatible, strict=False)
    if state_path is not None and str(state_path).strip():
        state = torch.load(state_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state, strict=False)
        print({"loaded_state": str(state_path)}, flush=True)
    if skipped:
        print({"checkpoint": str(path), "loaded_tensors": len(compatible), "skipped_shape_mismatch": skipped[:8], "skipped_count": len(skipped)}, flush=True)
    return model.eval()


class FoundationMCTSPlanner:
    def __init__(
        self,
        model,
        rollouts: int,
        expand_top_k: int,
        q_utility_weight: float,
        rollout_policy: str,
        seed_policies: tuple[str, ...],
        horizon_windows: int,
        branch_rollout_threshold: float = 0.65,
    ):
        self.model = model
        self.rollouts = int(rollouts)
        self.expand_top_k = int(expand_top_k)
        self.q_utility_weight = float(q_utility_weight)
        self.rollout_policy = str(rollout_policy)
        self.seed_policies = tuple(seed_policies)
        self.horizon_windows = int(horizon_windows)
        self.branch_rollout_threshold = float(branch_rollout_threshold)

    def plan_from_engine(self, eng, debt: float, budget_ms: float, env_cfg: dict) -> tuple[list[int], float]:
        sim = SnapshotSimulator(eng, debt, env_cfg=env_cfg, use_arrival_feature=True, use_grid_feature=True)
        t0 = time.perf_counter()
        mcts = ExactEnvMCTS(
            self.model,
            sim,
            [],
            rollouts=self.rollouts,
            c_puct=1.25,
            expand_top_k=self.expand_top_k,
            horizon_windows=self.horizon_windows,
            rollout_policy=self.rollout_policy,
            branch_rollout_threshold=self.branch_rollout_threshold,
            prior_mode="factorized",
            policy_target="q_softmax",
            head_mode="pq",
            q_utility_weight=self.q_utility_weight,
            eager_edge_depth=1,
            seed_rollout_policies=self.seed_policies,
            sensor_action_mode="explicit",
        )
        root = mcts.run()
        plan = best_window_plan(mcts, root, "q", budget_ms)
        return [int(a) for a in plan], (time.perf_counter() - t0) * 1000.0


class FairExactRescore(ExactRescore128):
    def choose(self, eng, debt_ms: float, obs) -> tuple[list[int], dict]:
        candidates = self.candidates(obs)
        t0 = time.perf_counter()
        from python_radar_env import PyRadarState, score_plans_vectorized

        approx = score_plans_vectorized(PyRadarState.from_obs(obs), candidates, self.gen.cfg, budget_ms=200.0)
        top = np.argsort(approx)[-self.top_k :].astype(int).tolist()
        top.extend([self.n_plans - 2, self.n_plans - 1])
        top = sorted(set(i for i in top if 0 <= i < len(candidates)))
        root = binding.vec_snapshot(eng.env)
        exact_scores = []
        for idx in top:
            binding.vec_restore(eng.env, root)
            reward, spent, _debt, executed, _searches, _arows = execute_plan_until_budget(
                eng, candidates[idx].tolist(), self.score_horizon_ms, float(debt_ms), "score", 0, 0
            )
            exact_scores.append((idx, float(reward), float(spent), int(executed)))
        best_idx, best_score, best_elapsed, best_executed = max(exact_scores, key=lambda x: x[1])
        first_action_scores: dict[int, float] = {}
        for idx, score, _spent, executed in exact_scores:
            if executed <= 0:
                continue
            action = int(candidates[int(idx)][0])
            first_action_scores[action] = max(float(score), first_action_scores.get(action, -1e9))
        binding.vec_restore(eng.env, root)
        plan_ms = (time.perf_counter() - t0) * 1000.0
        return candidates[best_idx].tolist(), {
            "planning_ms": float(plan_ms),
            "candidate_count": int(len(candidates)),
            "exact_rescored": int(len(top)),
            "exact_score": float(best_score),
            "exact_score_elapsed_ms": float(best_elapsed),
            "exact_score_executed": int(best_executed),
            "vector_score": float(approx[best_idx]),
            "best_candidate_idx": int(best_idx),
            "first_action_scores": first_action_scores,
        }


class HybridPQ1MCTS(FairExactRescore):
    """Exact-score PQ1 and learned-MCTS proposals, then execute the better plan."""

    def __init__(
        self,
        env_cfg: dict,
        model,
        rollouts: int = 16,
        expand_top_k: int = 24,
        branch_rollout_threshold: float = 0.65,
        top_k: int = 8,
        score_horizon_ms: float = 800.0,
        slots: int = 96,
        generator: str = "structured",
        seed: int = 16016,
    ):
        super().__init__(env_cfg, top_k=top_k, score_horizon_ms=score_horizon_ms, slots=slots, generator=generator, seed=seed)
        self.mcts = FoundationMCTSPlanner(
            model,
            rollouts=rollouts,
            expand_top_k=expand_top_k,
            q_utility_weight=0.0,
            rollout_policy="branch",
            seed_policies=tuple(),
            horizon_windows=1,
            branch_rollout_threshold=branch_rollout_threshold,
        )

    def choose(self, eng, debt_ms: float, obs) -> tuple[list[int], dict]:
        pq1_plan, pq1_meta = super().choose(eng, debt_ms, obs)
        mcts_plan, mcts_ms = self.mcts.plan_from_engine(eng, debt_ms, 200.0, self.env_cfg)
        root = binding.vec_snapshot(eng.env)
        scored = []
        for name, plan in (("pq1", pq1_plan), ("mcts", mcts_plan)):
            binding.vec_restore(eng.env, root)
            reward, spent, _debt, executed, _searches, _arows = execute_plan_until_budget(
                eng, [int(a) for a in plan], self.score_horizon_ms, float(debt_ms), "score", 0, 0
            )
            scored.append((name, [int(a) for a in plan], float(reward), float(spent), int(executed)))
        binding.vec_restore(eng.env, root)
        best_name, best_plan, best_score, best_spent, best_executed = max(scored, key=lambda x: x[2])
        meta = dict(pq1_meta)
        meta.update(
            {
                "hybrid_choice": best_name,
                "hybrid_score": float(best_score),
                "hybrid_score_elapsed_ms": float(best_spent),
                "hybrid_score_executed": int(best_executed),
                "pq1_score": float(next(s[2] for s in scored if s[0] == "pq1")),
                "mcts_score": float(next(s[2] for s in scored if s[0] == "mcts")),
                "mcts_planning_ms": float(mcts_ms),
                "planning_ms": float(pq1_meta.get("planning_ms", 0.0)) + float(mcts_ms),
            }
        )
        return best_plan, meta


def run_foundation(planner: FoundationMCTSPlanner, name: str, initial: int, seed: int, windows: int, env_cfg: dict):
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
            _obs = get_obs(eng, debt)
            plan, plan_ms = planner.plan_from_engine(eng, debt, 200.0, env_cfg)
            reward, spent, debt, executed, searches, arows = execute_plan_until_budget(eng, plan, 200.0, debt, name, int(seed), int(window))
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
                    **sample_state_metrics(eng, debt),
                }
            )
            for row in arows:
                row.update(window=int(window), elapsed_ms=float((window + 1) * 200))
            action_rows.extend(arows)
    finally:
        eng.close()
    return pd.DataFrame(rows), pd.DataFrame(action_rows)


def run_exact_rescore_grid(planner: ExactRescore128, name: str, initial: int, seed: int, windows: int, env_cfg: dict):
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
            reward, spent, debt, executed, searches, arows = execute_plan_until_budget(eng, plan, 200.0, debt, name, int(seed), int(window))
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


def run_heuristic(planner, name: str, initial: int, seed: int, windows: int, env_cfg: dict):
    from final_radar_campaign import run_fixed

    return run_fixed(planner, name, int(initial), MAXT, int(seed), int(windows), 200, env_cfg)


def physical_candidates(obs: dict, top_k: int) -> list[int]:
    cands = [xs_s_search_action(MAXT)]
    if int(obs.get("enable_x_band", 0)) and float(obs.get("x_band_busy_ms", 0.0)) <= 0.0:
        cands.append(xs_x_search_action(MAXT))
    active = np.asarray(obs["active_mask"], dtype=bool)
    deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
    ranges = np.asarray(obs.get("target_range", np.zeros_like(deadline)), dtype=np.float32)
    ranked = []
    for i, ok in enumerate(active[:MAXT]):
        if not bool(ok) or i >= len(deadline) or float(deadline[i]) < 0.0:
            continue
        ranked.append((float(deadline[i]), i + 1))
    ranked.sort(key=lambda x: (x[0], x[1]))
    for _deadline, base in ranked[: max(0, int(top_k))]:
        rng = float(ranges[base - 1]) if base - 1 < len(ranges) else 50_000_000.0
        if float(obs.get("s_band_busy_ms", 0.0)) <= 0.0 and 10_000_000.0 < rng < 184_000_000.0:
            cands.append(xs_s_track_action(base, MAXT))
        if int(obs.get("enable_x_band", 0)) and float(obs.get("x_band_busy_ms", 0.0)) <= 0.0 and 5_000_000.0 < rng < 100_000_000.0:
            cands.append(xs_x_track_action(base, MAXT))
    return list(dict.fromkeys(int(a) for a in cands))


def score_physical_action(eng, root, action: int, tail_plan: list[int], debt: float, horizon_ms: float) -> tuple[float, int]:
    binding.vec_restore(eng.env, root)
    plan = [int(action)] + [int(a) for a in tail_plan]
    reward, _spent, _debt, executed, _searches, _arows = execute_plan_until_budget(eng, plan, float(horizon_ms), float(debt), "score", 0, 0)
    binding.vec_restore(eng.env, root)
    return float(reward), int(executed)


def apply_token_action_mask(tok: np.ndarray, sensor_pi: np.ndarray, sensor_q_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(tok[:, 4] > 0.5, dtype=bool)
    selected = np.asarray(tok[:, 8] > 0.5, dtype=bool) if tok.shape[1] > 8 else np.zeros_like(valid)
    valid[0] = True
    action_valid = valid & ~selected
    action_valid[0] = True
    masked_pi = np.asarray(sensor_pi, dtype=np.float32).copy()
    masked_q_mask = np.asarray(sensor_q_mask, dtype=np.float32).copy()
    masked_pi[~action_valid, :] = 0.0
    masked_q_mask[~action_valid, :] = 0.0
    mass = float(masked_pi.sum())
    if mass > 1e-8:
        masked_pi /= mass
    return masked_pi, masked_q_mask


def collect_fair_exact_targets(args, exact_args, out_path: Path):
    adapt = adapter()
    targets: list[SearchTarget] = []
    rows = []
    tau = max(1e-6, float(args.policy_tau))
    for seed in parse_ints(args.seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                cell_start = len(targets)
                env_cfg = env_cfg_for(float(rate), exact_args)
                env_cfg["enable_x_band"] = 1
                planner = FairExactRescore(env_cfg, top_k=8, score_horizon_ms=800.0, slots=96, generator="structured", seed=15008)
                eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
                eng.reset(seed=int(seed))
                debt = 0.0
                try:
                    for window in range(int(args.windows)):
                        spent = 0.0
                        selected: set[int] = set()
                        search_count = 0
                        track_count = 0
                        last = -1
                        while (
                            spent < 200.0
                            and not bool(eng.term_buf[0])
                            and len(targets) < int(args.max_targets)
                            and (
                                int(args.max_targets_per_cell) <= 0
                                or (len(targets) - cell_start) < int(args.max_targets_per_cell)
                            )
                        ):
                            obs = attach_env_obs(get_obs(eng, debt), env_cfg, True, True)
                            plan, meta = planner.choose(eng, debt, obs)
                            action_scores = dict(meta.get("first_action_scores", {}))
                            if action_scores:
                                actions = list(action_scores.keys())
                                vals = np.asarray([action_scores[a] for a in actions], dtype=np.float64)
                                logits = vals / tau
                                logits -= float(np.max(logits))
                                probs = np.exp(np.clip(logits, -60.0, 60.0))
                                probs /= max(float(probs.sum()), 1e-12)
                                pi = np.zeros((MAXT + 1,), dtype=np.float32)
                                q = np.zeros((MAXT + 1,), dtype=np.float32)
                                q_mask = np.zeros((MAXT + 1,), dtype=np.float32)
                                sensor_pi = np.zeros((MAXT + 1, 2), dtype=np.float32)
                                sensor_q = np.zeros((MAXT + 1, 2), dtype=np.float32)
                                sensor_q_mask = np.zeros((MAXT + 1, 2), dtype=np.float32)
                                for action, val, prob in zip(actions, vals, probs):
                                    base, sensor = xs_decode_action(int(action), MAXT)
                                    if int(base) < 0:
                                        continue
                                    sidx = 0 if sensor is None else int(sensor)
                                    pi[int(base)] += float(prob)
                                    sensor_pi[int(base), sidx] += float(prob)
                                    if q_mask[int(base)] <= 0.5 or float(val) > float(q[int(base)]):
                                        q[int(base)] = float(val)
                                        q_mask[int(base)] = 1.0
                                    sensor_q[int(base), sidx] = float(val)
                                    sensor_q_mask[int(base), sidx] = 1.0
                                tok = tokenize(adapt, obs, selected=selected, search_count=search_count).astype(np.float32)
                                slot = slot_features(obs, spent, search_count, track_count, last, 200.0).astype(np.float32)
                                targets.append(
                                    SearchTarget(
                                        tok,
                                        slot,
                                        pi,
                                        q,
                                        q_mask,
                                        search_count,
                                        track_count,
                                        reward=0.0,
                                        ret=float(np.max(vals)),
                                        sensor_pi=sensor_pi,
                                        sensor_q=sensor_q,
                                        sensor_q_mask=sensor_q_mask,
                                        initial=int(initial),
                                        rate=float(rate),
                                        seed=int(seed),
                                        window=int(window),
                                        action_index=len(targets),
                                    )
                                )
                                rows.append({"initial": int(initial), "rate": float(rate), "seed": int(seed), "window": int(window), "search_mass": float(sensor_pi[0, :].sum()), "actions": int(len(actions)), "best": float(np.max(vals))})
                            reward, dt, executed = execute_first_valid_action(eng, plan, 200.0 - spent)
                            if executed is None or float(dt) <= 0.0:
                                break
                            base, _sensor = xs_decode_action(int(executed), MAXT)
                            debt = 0.0 if int(base) == 0 else debt + float(dt)
                            spent += float(dt)
                            if int(base) == 0:
                                search_count += 1
                            elif int(base) > 0:
                                selected.add(int(base))
                                track_count += 1
                            last = int(base)
                        if len(targets) >= int(args.max_targets) or (
                            int(args.max_targets_per_cell) > 0
                            and (len(targets) - cell_start) >= int(args.max_targets_per_cell)
                        ):
                            break
                finally:
                    eng.close()
                print({"targets": len(targets), "initial": initial, "rate": rate, "seed": seed}, flush=True)
                if len(targets) >= int(args.max_targets):
                    break
            if len(targets) >= int(args.max_targets):
                break
        if len(targets) >= int(args.max_targets):
            break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_targets(out_path, targets)
    pd.DataFrame(rows).to_csv(out_path.with_suffix(".csv"), index=False)
    return targets


def collect_physical_fair_exact_targets(args, exact_args, out_path: Path):
    adapt = adapter()
    targets: list[SearchTarget] = []
    rows = []
    tau = max(1e-6, float(args.policy_tau))
    for seed in parse_ints(args.seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                cell_start = len(targets)
                env_cfg = env_cfg_for(float(rate), exact_args)
                env_cfg["enable_x_band"] = 1
                planner = FairExactRescore(env_cfg, top_k=8, score_horizon_ms=800.0, slots=96, generator="structured", seed=15008)
                eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
                eng.reset(seed=int(seed))
                debt = 0.0
                try:
                    for window in range(int(args.windows)):
                        spent = 0.0
                        selected: set[int] = set()
                        search_count = 0
                        track_count = 0
                        last = -1
                        while (
                            spent < 200.0
                            and not bool(eng.term_buf[0])
                            and len(targets) < int(args.max_targets)
                            and (
                                int(args.max_targets_per_cell) <= 0
                                or (len(targets) - cell_start) < int(args.max_targets_per_cell)
                            )
                        ):
                            obs = attach_env_obs(get_obs(eng, debt), env_cfg, True, True)
                            root = binding.vec_snapshot(eng.env)
                            teacher_plan, _meta = planner.choose(eng, debt, obs)
                            cands = physical_candidates(obs, int(args.top_k))
                            vals = []
                            tail = teacher_plan[1:] if teacher_plan else []
                            for action in cands:
                                val, executed = score_physical_action(eng, root, int(action), tail, debt, float(args.score_horizon_ms))
                                if executed > 0 and np.isfinite(val):
                                    vals.append((int(action), float(val)))
                            if vals:
                                actions = [a for a, _v in vals]
                                raw_vals = np.asarray([v for _a, v in vals], dtype=np.float64)
                                if str(args.label_mode) == "argmax":
                                    probs = np.zeros_like(raw_vals, dtype=np.float64)
                                    probs[int(np.argmax(raw_vals))] = 1.0
                                elif str(args.label_mode) in {"teacher", "mix"}:
                                    probs = np.zeros_like(raw_vals, dtype=np.float64)
                                    teacher_base = -1
                                    if teacher_plan:
                                        teacher_base, _teacher_sensor = xs_decode_action(int(teacher_plan[0]), MAXT)
                                    matching = []
                                    for j, action in enumerate(actions):
                                        base, _sensor = xs_decode_action(int(action), MAXT)
                                        if int(base) == int(teacher_base):
                                            matching.append(j)
                                    if matching:
                                        best_j = max(matching, key=lambda j: float(raw_vals[int(j)]))
                                    else:
                                        best_j = int(np.argmax(raw_vals))
                                    probs[int(best_j)] = 1.0
                                    if str(args.label_mode) == "mix":
                                        logits = raw_vals / tau
                                        logits -= float(np.max(logits))
                                        soft_probs = np.exp(np.clip(logits, -60.0, 60.0))
                                        soft_probs /= max(float(soft_probs.sum()), 1e-12)
                                        alpha = float(np.clip(args.teacher_mix, 0.0, 1.0))
                                        probs = alpha * probs + (1.0 - alpha) * soft_probs
                                else:
                                    logits = raw_vals / tau
                                    logits -= float(np.max(logits))
                                    probs = np.exp(np.clip(logits, -60.0, 60.0))
                                    probs /= max(float(probs.sum()), 1e-12)
                                pi = np.zeros((MAXT + 1,), dtype=np.float32)
                                q = np.zeros((MAXT + 1,), dtype=np.float32)
                                q_mask = np.zeros((MAXT + 1,), dtype=np.float32)
                                sensor_pi = np.zeros((MAXT + 1, 2), dtype=np.float32)
                                sensor_q = np.zeros((MAXT + 1, 2), dtype=np.float32)
                                sensor_q_mask = np.zeros((MAXT + 1, 2), dtype=np.float32)
                                for action, val, prob in zip(actions, raw_vals, probs):
                                    base, sensor = xs_decode_action(int(action), MAXT)
                                    if int(base) < 0:
                                        continue
                                    sidx = 0 if sensor is None else int(sensor)
                                    pi[int(base)] += float(prob)
                                    sensor_pi[int(base), sidx] += float(prob)
                                    if q_mask[int(base)] <= 0.5 or float(val) > float(q[int(base)]):
                                        q[int(base)] = float(val)
                                        q_mask[int(base)] = 1.0
                                    sensor_q[int(base), sidx] = float(val)
                                    sensor_q_mask[int(base), sidx] = 1.0
                                tok = tokenize(adapt, obs, selected=selected, search_count=search_count).astype(np.float32)
                                slot = slot_features(obs, spent, search_count, track_count, last, 200.0).astype(np.float32)
                                sensor_pi, sensor_q_mask = apply_token_action_mask(tok, sensor_pi, sensor_q_mask)
                                if float(sensor_pi.sum()) <= 0.0:
                                    binding.vec_restore(eng.env, root)
                                    break
                                targets.append(
                                    SearchTarget(
                                        tok,
                                        slot,
                                        pi,
                                        q,
                                        q_mask,
                                        search_count,
                                        track_count,
                                        reward=0.0,
                                        ret=float(np.max(raw_vals)),
                                        sensor_pi=sensor_pi,
                                        sensor_q=sensor_q,
                                        sensor_q_mask=sensor_q_mask,
                                        initial=int(initial),
                                        rate=float(rate),
                                        seed=int(seed),
                                        window=int(window),
                                        action_index=len(targets),
                                    )
                                )
                                rows.append(
                                    {
                                        "initial": int(initial),
                                        "rate": float(rate),
                                        "seed": int(seed),
                                        "window": int(window),
                                        "search_mass": float(sensor_pi[0, :].sum()),
                                        "x_mass": float(sensor_pi[:, 1].sum()),
                                        "actions": int(len(actions)),
                                        "spread": float(np.max(raw_vals) - np.min(raw_vals)),
                                        "best": float(np.max(raw_vals)),
                                    }
                                )
                            binding.vec_restore(eng.env, root)
                            reward, dt, executed = execute_first_valid_action(eng, teacher_plan, 200.0 - spent)
                            if executed is None or float(dt) <= 0.0:
                                break
                            base, _sensor = xs_decode_action(int(executed), MAXT)
                            debt = 0.0 if int(base) == 0 else debt + float(dt)
                            spent += float(dt)
                            if int(base) == 0:
                                search_count += 1
                            elif int(base) > 0:
                                selected.add(int(base))
                                track_count += 1
                            last = int(base)
                        if len(targets) >= int(args.max_targets) or (
                            int(args.max_targets_per_cell) > 0
                            and (len(targets) - cell_start) >= int(args.max_targets_per_cell)
                        ):
                            break
                finally:
                    eng.close()
                print({"targets": len(targets), "initial": initial, "rate": rate, "seed": seed}, flush=True)
                if len(targets) >= int(args.max_targets):
                    break
            if len(targets) >= int(args.max_targets):
                break
        if len(targets) >= int(args.max_targets):
            break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_targets(out_path, targets)
    pd.DataFrame(rows).to_csv(out_path.with_suffix(".csv"), index=False)
    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "targets", "physical_targets"], default="eval")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--state", type=Path, default=Path(""))
    ap.add_argument("--out", type=Path, default=Path("CreateValid1/results/foundation_mcts_fair_eval.csv"))
    ap.add_argument("--targets-out", type=Path, default=Path("CreateValid1/results/fair_exact_targets.pt"))
    ap.add_argument("--initials", default="40,60")
    ap.add_argument("--rates", default="2,4")
    ap.add_argument("--seeds", default="903")
    ap.add_argument("--windows", type=int, default=100)
    ap.add_argument("--max-targets", type=int, default=512)
    ap.add_argument("--max-targets-per-cell", type=int, default=0)
    ap.add_argument("--policy-tau", type=float, default=5.0)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--score-horizon-ms", type=float, default=800.0)
    ap.add_argument("--label-mode", choices=["soft", "argmax", "teacher", "mix"], default="soft")
    ap.add_argument("--teacher-mix", type=float, default=0.7)
    ap.add_argument("--variants", default="ff_r0_k4,exact_k4_h400")
    args = ap.parse_args()

    torch.set_num_threads(1)
    np.random.seed(0)
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = True
    exact_args.single_sensor = False
    if args.mode == "targets":
        collect_fair_exact_targets(args, exact_args, args.targets_out)
        return
    if args.mode == "physical_targets":
        collect_physical_fair_exact_targets(args, exact_args, args.targets_out)
        return
    model = load_foundation(args.ckpt, args.state if str(args.state).strip() else None)
    variant_cfg = {
        "ff_r0_k4": FoundationMCTSPlanner(model, 0, 4, 0.2, "edge", tuple(), 1),
        "ff_r2_k4": FoundationMCTSPlanner(model, 2, 4, 0.2, "edge", tuple(), 1),
        "ff_r4_k8_edf": FoundationMCTSPlanner(model, 4, 8, 0.2, "edf", ("planner_edf", "planner_est", "edf", "est", "edge"), 1),
        "ml_pq_r8_k16": FoundationMCTSPlanner(model, 8, 16, 0.0, "pq", tuple(), 1),
        "ml_pq_r16_k24": FoundationMCTSPlanner(model, 16, 24, 0.0, "pq", tuple(), 1),
        "ml_q_r16_k24": FoundationMCTSPlanner(model, 16, 24, 0.0, "q", tuple(), 1),
    }
    selected = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    rows = []
    all_windows = []
    for seed in parse_ints(args.seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                env_cfg = env_cfg_for(float(rate), exact_args)
                env_cfg["enable_x_band"] = 1
                planners = {"EDF": EDFPlanner(MAXT), "EST": ESTPlanner(MAXT)}
                for name in selected:
                    if name == "exact_k4_h400":
                        planners[name] = ExactRescore128(env_cfg, top_k=4, score_horizon_ms=400.0, slots=64, generator="structured", seed=14004)
                    elif name == "exact_k8_h800":
                        planners[name] = ExactRescore128(env_cfg, top_k=8, score_horizon_ms=800.0, slots=96, generator="structured", seed=14008)
                    elif name == "fair_exact_k8_h800":
                        planners[name] = FairExactRescore(env_cfg, top_k=8, score_horizon_ms=800.0, slots=96, generator="structured", seed=15008)
                    elif name == "fair_exact_k16_h1200":
                        planners[name] = FairExactRescore(env_cfg, top_k=16, score_horizon_ms=1200.0, slots=96, generator="structured", seed=15116)
                    elif name == "hybrid_pq1_branch_mcts":
                        planners[name] = HybridPQ1MCTS(
                            env_cfg,
                            model,
                            rollouts=16,
                            expand_top_k=24,
                            branch_rollout_threshold=0.65,
                            top_k=8,
                            score_horizon_ms=800.0,
                            slots=96,
                            generator="structured",
                            seed=16016,
                        )
                    else:
                        planners[name] = variant_cfg[name]
                for name, planner in planners.items():
                    print({"running": name, "initial": initial, "rate": rate, "seed": seed}, flush=True)
                    if isinstance(planner, FoundationMCTSPlanner):
                        w, _a = run_foundation(planner, name, initial, seed, args.windows, env_cfg)
                    elif isinstance(planner, ExactRescore128):
                        w, _a = run_exact_rescore_grid(planner, name, initial, seed, args.windows, env_cfg)
                    else:
                        w, _a = run_heuristic(planner, name, initial, seed, args.windows, env_cfg)
                    all_windows.append(w.assign(initial=int(initial), rate=float(rate)))
                    s = summarize_window_df(w, "fixed")
                    rows.append(
                        {
                            "method": name,
                            "initial": int(initial),
                            "rate": float(rate),
                            "seed": int(seed),
                            "reward": float(s.get("reward_per_200ms_eq", np.nan)),
                            "search": float(s.get("search_fraction", np.nan)),
                            "latency_ms": float(s.get("planning_ms_per_decision", np.nan)),
                            "windows_completed": int(len(w)),
                        }
                    )
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    pd.concat(all_windows, ignore_index=True).to_csv(args.out.with_name(args.out.stem + "_windows.csv"), index=False)
    summary = (
        out.groupby("method", as_index=False)
        .agg(reward=("reward", "mean"), search=("search", "mean"), latency_ms=("latency_ms", "mean"), n=("reward", "size"))
        .sort_values("reward", ascending=False)
    )
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    try:
        print(summary.to_string(index=False), flush=True)
    except OSError:
        pass


if __name__ == "__main__":
    main()
