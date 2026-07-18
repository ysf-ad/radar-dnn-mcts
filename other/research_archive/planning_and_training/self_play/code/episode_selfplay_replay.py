from __future__ import annotations

import argparse
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from alphazero_orthodox import save_targets
from compare_action_heads_smoke import usable_targets
from exact_env_mutual import EDFPlanner, MAXT, _DummyPlanner, attach_env_obs, env_cfg_for, xs_decode_action
from final_radar_campaign import get_obs
from foundation_mcts_fair_eval import (
    apply_token_action_mask,
    parse_floats,
    parse_ints,
    physical_candidates,
    score_physical_action,
)
from learned_proposal_fair_eval import LearnedProposalFairExact, make_learned_planners
from mutual_features import slot_features, tokenize
from mutual_foundation import SearchTarget
from penalty_window_quota_learner_eval import make_exact_args
from pufferlib.ocean.radarxs import binding
from repaired_campaign_tools import build_env, execute_first_valid_action
from realistic_reward_retrain import adapter
from strict_window_report import sample_state_metrics
from two_sensor_physical_head_eval import train_head


@dataclass
class PendingTarget:
    tok: np.ndarray
    slot: np.ndarray
    pi: np.ndarray
    q: np.ndarray
    q_mask: np.ndarray
    sensor_pi: np.ndarray
    sensor_q: np.ndarray
    sensor_q_mask: np.ndarray
    search_count: int
    track_count: int
    initial: int
    rate: float
    seed: int
    window: int
    action_index: int
    reward_index: int
    local_value: float


def train_base_model(args):
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
        search_calibration_weight=float(getattr(args, "search_calibration_weight", 0.0)),
        log_every=max(1, int(args.train_steps)),
        cell_balanced_sampling=bool(args.cell_balanced_sampling),
    )
    torch.manual_seed(int(args.model_seed))
    np.random.seed(int(args.model_seed))
    model = train_head(str(args.variant), usable_targets(Path(args.targets)), train_args, torch.device("cpu"))
    if str(getattr(args, "finetune_targets", "")).strip():
        ft_args = argparse.Namespace(**vars(train_args))
        ft_args.train_steps = int(args.finetune_steps) if int(args.finetune_steps) > 0 else int(args.train_steps)
        ft_args.log_every = max(1, int(ft_args.train_steps))
        ft_args.value_loss_weight = float(args.finetune_value_loss_weight)
        model = train_head(str(args.variant), usable_targets(Path(args.finetune_targets)), ft_args, torch.device("cpu"), model=model)
    return model


def softmax_policy(vals: np.ndarray, tau: float) -> np.ndarray:
    if len(vals) == 0:
        return vals
    if float(tau) <= 0.0:
        out = np.zeros_like(vals, dtype=np.float64)
        out[int(np.argmax(vals))] = 1.0
        return out
    z = vals.astype(np.float64) / max(1e-6, float(tau))
    z -= float(np.max(z))
    exp = np.exp(z)
    return exp / max(1e-12, float(exp.sum()))


def dense_state_reward(eng, debt: float) -> float:
    metrics = sample_state_metrics(eng, float(debt))
    tracked = float(metrics.get("tracked_targets", 0.0))
    dropped_pct = float(metrics.get("drop_pct_active", 0.0))
    mean_delay = float(metrics.get("mean_delay_active", 0.0))
    return 0.10 * tracked - 0.25 * dropped_pct - 0.20 * mean_delay


def make_target_from_scores(
    adapt,
    obs: dict,
    selected: set[int],
    spent: float,
    search_count: int,
    track_count: int,
    last: int,
    scored: list[tuple[int, float]],
    args,
    initial: int,
    rate: float,
    seed: int,
    window: int,
    action_index: int,
    reward_index: int,
) -> PendingTarget | None:
    if not scored:
        return None
    actions = [int(a) for a, _ in scored]
    raw_vals = np.asarray([float(v) for _a, v in scored], dtype=np.float64)
    probs = softmax_policy(raw_vals, float(args.policy_tau))

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
        return None
    return PendingTarget(
        tok=tok,
        slot=slot,
        pi=pi,
        q=q,
        q_mask=q_mask,
        sensor_pi=sensor_pi,
        sensor_q=sensor_q,
        sensor_q_mask=sensor_q_mask,
        search_count=int(search_count),
        track_count=int(track_count),
        initial=int(initial),
        rate=float(rate),
        seed=int(seed),
        window=int(window),
        action_index=int(action_index),
        reward_index=int(reward_index),
        local_value=float(np.max(raw_vals)),
    )


def filter_scored_by_baseline(args, eng, root, obs: dict, scored: list[tuple[int, float]], tail: list[int], debt: float):
    if str(args.improvement_baseline) == "none" or not scored:
        return scored, None
    if str(args.improvement_baseline) != "edf":
        raise ValueError(f"unsupported improvement baseline: {args.improvement_baseline}")

    edf_plan = list(EDFPlanner(MAXT).plan(obs, budget_ms=200))
    if not edf_plan:
        return scored, None
    baseline_action = int(edf_plan[0])
    baseline_tail = edf_plan[1:] if bool(args.baseline_uses_own_tail) else list(tail)
    baseline_val, baseline_executed = score_physical_action(
        eng,
        root,
        baseline_action,
        baseline_tail,
        float(debt),
        float(args.score_horizon_ms),
    )
    if int(baseline_executed) <= 0 or not np.isfinite(float(baseline_val)):
        return scored, None

    margin = float(args.improvement_margin)
    filtered = [(int(a), float(v)) for a, v in scored if float(v) >= float(baseline_val) + margin]
    if len(filtered) < int(args.min_improved_candidates):
        filtered = []
    if bool(args.include_baseline_action):
        by_action = {int(a): float(v) for a, v in filtered}
        by_action[int(baseline_action)] = max(float(baseline_val), by_action.get(int(baseline_action), -1e18))
        filtered = sorted(by_action.items(), key=lambda x: x[0])
    return filtered, {
        "baseline_action": int(baseline_action),
        "baseline_value": float(baseline_val),
        "raw_candidate_count": int(len(scored)),
        "improved_candidate_count": int(len(filtered)),
    }


def collect_episode(args, exact_args, model, initial: int, rate: float, seed: int) -> tuple[list[SearchTarget], list[dict]]:
    adapt = adapter()
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
        rescore_horizons_ms=parse_floats(str(args.rescore_horizons_ms)) if str(args.rescore_horizons_ms).strip() else None,
        rescore_horizon_weights=parse_floats(str(args.rescore_horizon_weights)) if str(args.rescore_horizon_weights).strip() else None,
    )

    eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, env_cfg)
    eng.reset(seed=int(seed))
    debt = 0.0
    rewards: list[float] = []
    pending: list[PendingTarget] = []
    rows: list[dict] = []
    action_index = 0
    try:
        for window in range(int(args.eval_windows)):
            if bool(eng.term_buf[0]):
                break
            spent = 0.0
            selected: set[int] = set()
            search_count = 0
            track_count = 0
            last = -1
            while spent < 200.0 and not bool(eng.term_buf[0]) and len(pending) < int(args.max_targets):
                obs = attach_env_obs(get_obs(eng, debt), env_cfg, True, True)
                root = binding.vec_snapshot(eng.env)
                teacher_plan, meta = planner.choose(eng, debt, obs)
                cands = physical_candidates(obs, int(args.candidate_top_k))
                tail = teacher_plan[1:] if teacher_plan else []
                scored: list[tuple[int, float]] = []
                for action in cands:
                    val, executed = score_physical_action(eng, root, int(action), tail, debt, float(args.score_horizon_ms))
                    if executed > 0 and np.isfinite(val):
                        scored.append((int(action), float(val)))
                scored, baseline_meta = filter_scored_by_baseline(args, eng, root, obs, scored, tail, debt)
                binding.vec_restore(eng.env, root)

                target = make_target_from_scores(
                    adapt,
                    obs,
                    selected,
                    spent,
                    search_count,
                    track_count,
                    last,
                    scored,
                    args,
                    int(initial),
                    float(rate),
                    int(seed),
                    int(window),
                    action_index,
                    len(rewards),
                )
                if target is not None:
                    pending.append(target)
                    rows.append(
                        {
                            "initial": int(initial),
                            "rate": float(rate),
                            "seed": int(seed),
                            "window": int(window),
                            "action_index": int(action_index),
                            "search_mass": float(target.sensor_pi[0, :].sum()),
                            "x_mass": float(target.sensor_pi[:, 1].sum()),
                            "candidate_count": int(len(scored)),
                            "exact_rescored": int(meta.get("exact_rescored", -1)),
                            "baseline_action": None if baseline_meta is None else int(baseline_meta["baseline_action"]),
                            "baseline_value": None if baseline_meta is None else float(baseline_meta["baseline_value"]),
                            "raw_candidate_count": int(len(scored)) if baseline_meta is None else int(baseline_meta["raw_candidate_count"]),
                            "improved_candidate_count": int(len(scored)) if baseline_meta is None else int(baseline_meta["improved_candidate_count"]),
                        }
                    )
                    action_index += 1

                reward, dt, executed = execute_first_valid_action(eng, teacher_plan, 200.0 - spent)
                if executed is None or float(dt) <= 0.0:
                    break
                base, _sensor = xs_decode_action(int(executed), MAXT)
                debt = 0.0 if int(base) == 0 else debt + float(dt)
                if str(args.return_mode) == "dense_state":
                    rewards.append(dense_state_reward(eng, debt))
                else:
                    rewards.append(float(reward))
                spent += float(dt)
                if int(base) == 0:
                    search_count += 1
                elif int(base) > 0:
                    selected.add(int(base))
                    track_count += 1
                last = int(base)
            if len(pending) >= int(args.max_targets):
                break
    finally:
        eng.close()

    returns = np.zeros((len(rewards) + 1,), dtype=np.float64)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + float(args.gamma) * running
        returns[i] = running

    targets: list[SearchTarget] = []
    use_fallback = bool(args.return_mode == "local_score" or (args.return_mode == "auto" and len(rewards) > 0 and float(np.max(np.abs(returns))) <= 1e-9))
    for i, item in enumerate(pending):
        ret = float(item.local_value) if use_fallback else float(returns[min(item.reward_index, len(returns) - 1)])
        targets.append(
            SearchTarget(
                item.tok,
                item.slot,
                item.pi,
                item.q,
                item.q_mask,
                item.search_count,
                item.track_count,
                reward=0.0,
                ret=ret,
                sensor_pi=item.sensor_pi,
                sensor_q=item.sensor_q,
                sensor_q_mask=item.sensor_q_mask,
                initial=item.initial,
                rate=item.rate,
                seed=item.seed,
                window=item.window,
                action_index=i,
            )
        )
        rows[i]["ret"] = ret
    return targets, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="CreateValid1/results/gated_accepted_fullgrid_after_60r4_late_targets.pt")
    ap.add_argument("--finetune-targets", default="")
    ap.add_argument("--targets-out", default="CreateValid1/results/episode_selfplay_replay_targets.pt")
    ap.add_argument("--initials", default="40")
    ap.add_argument("--rates", default="2")
    ap.add_argument("--eval-seeds", default="907")
    ap.add_argument("--eval-windows", type=int, default=100)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=120)
    ap.add_argument("--finetune-steps", type=int, default=0)
    ap.add_argument("--finetune-value-loss-weight", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--model-seed", type=int, default=123)
    ap.add_argument("--cell-balanced-sampling", action="store_true")
    ap.add_argument(
        "--variant",
        choices=[
            "flat",
            "two_row_factorized",
            "two_row_factorized_qnorm",
            "two_row_factorized_adaptive",
            "two_row_calibrated_factorized",
            "two_row_action_attention_qpolicy_factored_loss",
            "two_row_calibrated_action_attention_qpolicy_factored_loss",
            "two_row_full_shared_action_qpolicy_factored_loss",
        ],
        default="flat",
    )
    ap.add_argument("--proposal-search-biases", default="-14,-12,-10,-8,-5")
    ap.add_argument("--proposal-q-weights", default="0,0.5")
    ap.add_argument("--force-learned-rescore", action="store_true")
    ap.add_argument("--learned-extra-top-k", type=int, default=2)
    ap.add_argument("--preserve-base-topk", action="store_true")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--candidate-top-k", type=int, default=8)
    ap.add_argument("--score-horizon-ms", type=float, default=800.0)
    ap.add_argument("--rescore-horizons-ms", default="")
    ap.add_argument("--rescore-horizon-weights", default="")
    ap.add_argument("--search-calibration-weight", type=float, default=0.0)
    ap.add_argument("--policy-tau", type=float, default=5.0)
    ap.add_argument("--improvement-baseline", choices=["none", "edf"], default="none")
    ap.add_argument("--improvement-margin", type=float, default=0.0)
    ap.add_argument("--min-improved-candidates", type=int, default=1)
    ap.add_argument("--baseline-uses-own-tail", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--include-baseline-action", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--return-mode", choices=["auto", "episode", "local_score", "dense_state"], default="auto")
    ap.add_argument("--max-targets", type=int, default=256)
    ap.add_argument("--max-targets-per-cell", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(1)
    model = train_base_model(args)
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = True
    exact_args.single_sensor = False

    all_targets: list[SearchTarget] = []
    all_rows: list[dict] = []
    per_cell_cap = int(args.max_targets_per_cell)
    for seed in parse_ints(args.eval_seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                cell_args = args
                if per_cell_cap > 0:
                    cell_args = SimpleNamespace(**vars(args))
                    cell_args.max_targets = per_cell_cap
                targets, rows = collect_episode(cell_args, exact_args, model, int(initial), float(rate), int(seed))
                all_targets.extend(targets)
                all_rows.extend(rows)
                print({"seed": seed, "initial": initial, "rate": rate, "targets": len(targets)}, flush=True)
                if len(all_targets) >= int(args.max_targets):
                    all_targets = all_targets[: int(args.max_targets)]
                    all_rows = all_rows[: int(args.max_targets)]
                    break
            if len(all_targets) >= int(args.max_targets):
                break
        if len(all_targets) >= int(args.max_targets):
            break

    out = Path(args.targets_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_targets(out, all_targets)
    pd.DataFrame(all_rows).to_csv(out.with_suffix(".csv"), index=False)
    print({"saved": str(out), "targets": len(all_targets)}, flush=True)


if __name__ == "__main__":
    main()
