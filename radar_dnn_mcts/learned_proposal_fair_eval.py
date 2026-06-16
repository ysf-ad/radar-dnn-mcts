from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval_exact_rescore128 import ExactRescore128
from exact_env_mutual import EDFPlanner, ESTPlanner, MAXT, _DummyPlanner, attach_env_obs, env_cfg_for, xs_decode_action
from foundation_mcts_fair_eval import (
    FairExactRescore,
    apply_token_action_mask,
    parse_floats,
    parse_ints,
    physical_candidates,
    run_exact_rescore_grid,
    run_heuristic,
    score_physical_action,
)
from final_radar_campaign import get_obs, summarize_window_df
from alphazero_orthodox import save_targets
from mutual_features import slot_features, tokenize
from mutual_foundation import SearchTarget
from penalty_window_quota_learner_eval import make_exact_args
from two_sensor_physical_head_eval import PhysicalHeadPlanner, train_head
from compare_action_heads_smoke import usable_targets
from pufferlib.ocean.radarxs import binding
from repaired_campaign_tools import build_env, execute_first_valid_action
from realistic_reward_retrain import adapter
from strict_window_report import execute_plan_until_budget


class LearnedProposalFairExact(FairExactRescore):
    def __init__(
        self,
        env_cfg: dict,
        learned_planner: PhysicalHeadPlanner | list[PhysicalHeadPlanner],
        learned_slots: int = 96,
        force_learned_rescore: bool = False,
        learned_extra_top_k: int = 2,
        preserve_base_topk: bool = False,
        rescore_horizons_ms: list[float] | None = None,
        rescore_horizon_weights: list[float] | None = None,
        **kwargs,
    ):
        super().__init__(env_cfg, **kwargs)
        if isinstance(learned_planner, list):
            self.learned_planners = learned_planner
        else:
            self.learned_planners = [learned_planner]
        self.learned_slots = int(learned_slots)
        self.force_learned_rescore = bool(force_learned_rescore)
        self.learned_extra_top_k = int(learned_extra_top_k)
        self.preserve_base_topk = bool(preserve_base_topk)
        self.rescore_horizons_ms = [float(x) for x in (rescore_horizons_ms or []) if float(x) > 0.0]
        self.rescore_horizon_weights = [float(x) for x in (rescore_horizon_weights or [])]
        if self.rescore_horizons_ms and len(self.rescore_horizon_weights) != len(self.rescore_horizons_ms):
            self.rescore_horizon_weights = [1.0 for _ in self.rescore_horizons_ms]

    def candidates(self, obs) -> np.ndarray:
        base = super().candidates(obs)
        self._last_base_count = int(len(base))
        learned_rows = []
        for learned_planner in self.learned_planners:
            raw_plans = learned_planner.plan(obs, budget_ms=int(self.score_horizon_ms))
            if not raw_plans:
                continue
            if raw_plans and isinstance(raw_plans[0], (list, tuple, np.ndarray)):
                plan_list = raw_plans
            else:
                plan_list = [raw_plans]
            for learned in plan_list:
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
                learned_rows.append(learned_row)
        if not learned_rows:
            self._last_candidate_count = int(len(base))
            return base
        out = np.concatenate([base, *learned_rows], axis=0)
        self._last_candidate_count = int(len(out))
        return out

    @staticmethod
    def _logical_approx_candidates(candidates: np.ndarray) -> np.ndarray:
        approx_candidates = np.asarray(candidates, dtype=np.int32).copy()
        flat = approx_candidates.reshape(-1)
        for i, action in enumerate(flat):
            base, _sensor = xs_decode_action(int(action), MAXT)
            flat[i] = max(0, int(base))
        return approx_candidates

    def choose(self, eng, debt_ms: float, obs) -> tuple[list[int], dict]:
        candidates = self.candidates(obs)
        t0 = time.perf_counter()
        from python_radar_env import PyRadarState, score_plans_vectorized

        approx = score_plans_vectorized(PyRadarState.from_obs(obs), self._logical_approx_candidates(candidates), self.gen.cfg, budget_ms=200.0)
        learned_start = int(getattr(self, "_last_base_count", len(candidates)))
        if self.preserve_base_topk:
            top = np.argsort(approx[:learned_start])[-self.top_k :].astype(int).tolist()
            top.extend([self.n_plans - 2, self.n_plans - 1])
            if int(self.learned_extra_top_k) > 0 and learned_start < len(candidates):
                learned_scores = approx[learned_start:]
                learned_take = min(int(self.learned_extra_top_k), int(len(learned_scores)))
                learned_top = np.argsort(learned_scores)[-learned_take:].astype(int).tolist()
                top.extend([learned_start + int(i) for i in learned_top])
        else:
            top = np.argsort(approx)[-self.top_k :].astype(int).tolist()
            top.extend([self.n_plans - 2, self.n_plans - 1])
        if self.force_learned_rescore:
            top.extend(range(learned_start, len(candidates)))
        top = sorted(set(i for i in top if 0 <= i < len(candidates)))
        root = binding.vec_snapshot(eng.env)
        exact_scores = []
        score_horizons = self.rescore_horizons_ms or [float(self.score_horizon_ms)]
        score_weights = self.rescore_horizon_weights or [1.0]
        for idx in top:
            combined_score = 0.0
            first_spent = 0.0
            first_executed = 0
            for h_i, (horizon_ms, weight) in enumerate(zip(score_horizons, score_weights)):
                binding.vec_restore(eng.env, root)
                reward, spent, _debt, executed, _searches, _arows = execute_plan_until_budget(
                    eng, candidates[idx].tolist(), float(horizon_ms), float(debt_ms), "score", 0, 0
                )
                combined_score += float(weight) * float(reward)
                if h_i == 0:
                    first_spent = float(spent)
                    first_executed = int(executed)
            exact_scores.append((idx, float(combined_score), float(first_spent), int(first_executed)))
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
            "learned_extra_top_k": int(self.learned_extra_top_k),
            "preserve_base_topk": bool(self.preserve_base_topk),
            "rescore_horizons_ms": list(score_horizons),
            "rescore_horizon_weights": list(score_weights),
            "forced_learned_rescored": int(max(0, len(candidates) - learned_start)) if self.force_learned_rescore else 0,
            "exact_score": float(best_score),
            "exact_score_elapsed_ms": float(best_elapsed),
            "exact_score_executed": int(best_executed),
            "vector_score": float(approx[best_idx]),
            "best_candidate_idx": int(best_idx),
            "first_action_scores": first_action_scores,
        }


def make_learned_planners(args, model, env_cfg: dict) -> list[PhysicalHeadPlanner]:
    biases = parse_floats(str(args.proposal_search_biases)) if str(args.proposal_search_biases).strip() else [float(args.search_score_bias)]
    q_weights = parse_floats(str(args.proposal_q_weights)) if str(args.proposal_q_weights).strip() else [float(args.q_score_weight)]
    variant = str(getattr(args, "variant", getattr(args, "proposal_variant", "flat")))
    planners = []
    for bias in biases:
        for q_weight in q_weights:
            planners.append(
                PhysicalHeadPlanner(
                    model,
                    variant,
                    env_cfg,
                    policy_weight=1.0,
                    q_weight=float(q_weight),
                    search_score_bias=float(bias),
                )
            )
    return planners


def collect_self_improved_targets(args, exact_args, model):
    adapt = adapter()
    out_path = Path(args.targets_out)
    targets: list[SearchTarget] = []
    rows = []
    tau = max(1e-6, float(args.policy_tau))
    for seed in parse_ints(args.eval_seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                cell_start = len(targets)
                env_cfg = env_cfg_for(float(rate), exact_args)
                env_cfg["enable_x_band"] = 1
                learned_planners = make_learned_planners(args, model, env_cfg)
                planner = LearnedProposalFairExact(
                    env_cfg,
                    learned_planners,
                    top_k=int(args.top_k),
                    score_horizon_ms=float(args.score_horizon_ms),
                    slots=96,
                    generator="structured",
                    seed=15008,
                    force_learned_rescore=bool(args.force_learned_rescore),
                    learned_extra_top_k=int(args.learned_extra_top_k),
                    preserve_base_topk=bool(args.preserve_base_topk),
                )
                eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, env_cfg)
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
                            tail = teacher_plan[1:] if teacher_plan else []
                            vals = []
                            for action in cands:
                                val, executed = score_physical_action(eng, root, int(action), tail, debt, float(args.score_horizon_ms))
                                if executed > 0 and np.isfinite(val):
                                    vals.append((int(action), float(val)))
                            if vals and int(window) >= int(args.collect_start_window):
                                actions = [a for a, _v in vals]
                                raw_vals = np.asarray([v for _a, v in vals], dtype=np.float64)
                                teacher_base = -1
                                if teacher_plan:
                                    teacher_base, _sensor = xs_decode_action(int(teacher_plan[0]), MAXT)
                                probs = np.zeros_like(raw_vals, dtype=np.float64)
                                if str(getattr(args, "teacher_label_mode", "match_teacher")) == "exact_argmax":
                                    best_j = int(np.argmax(raw_vals))
                                else:
                                    matching = [j for j, action in enumerate(actions) if int(xs_decode_action(int(action), MAXT)[0]) == int(teacher_base)]
                                    best_j = max(matching, key=lambda j: float(raw_vals[int(j)])) if matching else int(np.argmax(raw_vals))
                                probs[int(best_j)] = 1.0

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
                                if float(sensor_pi.sum()) > 0.0:
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
    print({"saved": str(out_path), "targets": len(targets)}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "collect"], default="eval")
    ap.add_argument("--targets", default="CreateValid1/results/fair_exact_physical_teacher_balanced512_targets.pt")
    ap.add_argument("--out", default="CreateValid1/results/learned_proposal_fair_eval.csv")
    ap.add_argument("--targets-out", default="CreateValid1/results/learned_proposal_self_targets.pt")
    ap.add_argument("--initials", default="40,60")
    ap.add_argument("--rates", default="2,4")
    ap.add_argument("--eval-seeds", default="903")
    ap.add_argument("--train-steps", type=int, default=180)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--eval-windows", type=int, default=100)
    ap.add_argument("--search-score-bias", type=float, default=-5.0)
    ap.add_argument("--q-score-weight", type=float, default=0.5)
    ap.add_argument("--search-calibration-weight", type=float, default=0.0)
    ap.add_argument("--proposal-search-biases", default="")
    ap.add_argument("--proposal-q-weights", default="")
    ap.add_argument("--force-learned-rescore", action="store_true")
    ap.add_argument("--learned-extra-top-k", type=int, default=2)
    ap.add_argument("--preserve-base-topk", action="store_true")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--score-horizon-ms", type=float, default=800.0)
    ap.add_argument("--max-targets", type=int, default=256)
    ap.add_argument("--max-targets-per-cell", type=int, default=64)
    ap.add_argument("--policy-tau", type=float, default=5.0)
    ap.add_argument("--collect-start-window", type=int, default=0)
    ap.add_argument("--teacher-label-mode", choices=["match_teacher", "exact_argmax"], default="match_teacher")
    ap.add_argument("--variant", default="flat")
    ap.add_argument("--model-seed", type=int, default=123)
    ap.add_argument("--cell-balanced-sampling", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(int(args.model_seed))
    np.random.seed(int(args.model_seed))

    train_args = argparse.Namespace(
        d_model=48,
        nhead=4,
        nlayers=2,
        lr=3e-4,
        train_steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        model_seed=int(args.model_seed),
        q_loss_weight=0.25,
        value_loss_weight=0.25,
        search_calibration_weight=float(args.search_calibration_weight),
        log_every=max(1, int(args.train_steps)),
        cell_balanced_sampling=bool(args.cell_balanced_sampling),
    )
    targets = usable_targets(Path(args.targets))
    model = train_head(str(args.variant), targets, train_args, torch.device("cpu"))

    exact_args = make_exact_args(args)
    exact_args.enable_x_band = True
    exact_args.single_sensor = False
    if args.mode == "collect":
        collect_self_improved_targets(args, exact_args, model)
        return
    rows = []
    all_windows = []
    for seed in parse_ints(args.eval_seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                env_cfg = env_cfg_for(float(rate), exact_args)
                env_cfg["enable_x_band"] = 1
                learned_planners = make_learned_planners(args, model, env_cfg)
                planners = {
                    "EDF": EDFPlanner(MAXT),
                    "EST": ESTPlanner(MAXT),
                    "fair_exact": FairExactRescore(
                        env_cfg,
                        top_k=int(args.top_k),
                        score_horizon_ms=float(args.score_horizon_ms),
                        slots=96,
                        generator="structured",
                        seed=15008,
                    ),
                    "learned_proposal_fair_exact": LearnedProposalFairExact(
                        env_cfg,
                        learned_planners,
                        top_k=int(args.top_k),
                        score_horizon_ms=float(args.score_horizon_ms),
                        slots=96,
                        generator="structured",
                        seed=15008,
                        force_learned_rescore=bool(args.force_learned_rescore),
                        learned_extra_top_k=int(args.learned_extra_top_k),
                        preserve_base_topk=bool(args.preserve_base_topk),
                    ),
                }
                for name, planner in planners.items():
                    print({"running": name, "initial": initial, "rate": rate, "seed": seed}, flush=True)
                    if isinstance(planner, ExactRescore128):
                        w, _ = run_exact_rescore_grid(planner, name, int(initial), int(seed), int(args.eval_windows), env_cfg)
                    else:
                        w, _ = run_heuristic(planner, name, int(initial), int(seed), int(args.eval_windows), env_cfg)
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
                    print(rows[-1], flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out, index=False)
    pd.concat(all_windows, ignore_index=True).to_csv(out.with_name(out.stem + "_windows.csv"), index=False)
    summary = raw.groupby("method", as_index=False).agg(reward=("reward", "mean"), search=("search", "mean"), latency_ms=("latency_ms", "mean"), n=("reward", "size")).sort_values("reward", ascending=False)
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
