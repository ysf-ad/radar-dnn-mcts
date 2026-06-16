from __future__ import annotations

import argparse
import random
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch

from alphazero_orthodox import base_exact_args, load_state_into, parse_floats, parse_ints, train_pv
from exact_env_mutual import (
    ExactEnvMCTS,
    MAXT,
    SnapshotSimulator,
    _DummyPlanner,
    attach_env_obs,
    build_env,
    choose_root_action,
    env_cfg_for,
    get_obs,
    load_model,
    run_fixed,
    run_snapshot_exact_episode,
    xs_decode_action,
)
from mutual_features import slot_features, tokenize
from mutual_foundation import SearchTarget
from realistic_reward_retrain import adapter as get_adapter


OUT = Path("results/alphazero_orthodox")


def load_targets(path_text: str) -> list[SearchTarget]:
    out: list[SearchTarget] = []
    for part in [p.strip() for p in str(path_text).split(";") if p.strip()]:
        data = torch.load(part, map_location="cpu", weights_only=False)
        if isinstance(data, dict) and "targets" in data:
            data = data["targets"]
        out.extend(list(data))
        print(f"loaded {len(data)} targets from {part}", flush=True)
    return out


def make_exact_args(args) -> SimpleNamespace:
    return base_exact_args(
        SimpleNamespace(
            ckpt=args.ckpt,
            device=args.device,
            windows=args.windows,
            max_targets_per_episode=1000000,
            rollouts=args.rollouts,
            c_puct=args.c_puct,
            expand_top_k=args.expand_top_k,
            horizon_windows=args.horizon_windows,
            rollout_policy=args.rollout_policy,
            branch_rollout_threshold=args.branch_rollout_threshold,
            prior_mode=args.prior_mode,
            prior_uniform_mix=args.prior_uniform_mix,
            prior_search_bias=float(getattr(args, "prior_search_bias", 0.0)),
            adaptive_search_bias=float(getattr(args, "adaptive_search_bias", 0.0)),
            adaptive_search_target_load=float(getattr(args, "adaptive_search_target_load", 0.75)),
            root_dirichlet_alpha=0.0,
            root_dirichlet_frac=0.0,
            leaf_value_mix=args.leaf_value_mix,
            head_mode=args.head_mode,
            q_utility_weight=0.0,
            q_utility_normalize=False,
            prior_q_beta=0.0,
            q_scale=args.q_scale,
            self_play_sample_tau=0.0,
            gamma=args.gamma,
            env_mode=args.env_mode,
            track_loss_penalty=args.track_loss_penalty,
            target_service_weight=args.target_service_weight,
            target_service_horizon_ms=args.target_service_horizon_ms,
            sector_staleness_weight=args.sector_staleness_weight,
            search_frame_overdue_weight=args.search_frame_overdue_weight,
            search_frame_drop_penalty=args.search_frame_drop_penalty,
            use_arrival_feature=bool(getattr(args, "use_arrival_feature", False)),
            use_grid_feature=bool(getattr(args, "use_grid_feature", False)),
            enable_x_band=bool(args.enable_x_band),
            single_sensor=bool(args.single_sensor),
            zero_action_rewards=bool(args.zero_action_rewards),
        )
    )


def make_mcts(model, sim: SnapshotSimulator, prefix: Sequence[int], exact_args) -> ExactEnvMCTS:
    return ExactEnvMCTS(
        model,
        sim,
        list(prefix),
        q_scale=float(exact_args.q_scale),
        rollouts=int(exact_args.rollouts),
        c_puct=float(exact_args.c_puct),
        expand_top_k=int(exact_args.expand_top_k),
        horizon_windows=int(exact_args.horizon_windows),
        rollout_policy=str(exact_args.rollout_policy),
        prior_mode=str(exact_args.prior_mode),
        epsilon=float(exact_args.epsilon),
        policy_target=str(exact_args.policy_target),
        policy_tau=float(exact_args.policy_tau),
        search_alg=str(exact_args.search_alg),
        max_num_considered_actions=int(exact_args.max_num_considered_actions),
        gumbel_scale=float(exact_args.gumbel_scale),
        mctx_value_scale=float(exact_args.mctx_value_scale),
        mctx_maxvisit_init=float(exact_args.mctx_maxvisit_init),
        eager_edge_depth=int(exact_args.eager_edge_depth),
        prior_uniform_mix=float(exact_args.prior_uniform_mix),
        root_dirichlet_alpha=0.0,
        root_dirichlet_frac=0.0,
        rollout_est_prob=float(exact_args.rollout_est_prob),
        mask_selected=not bool(exact_args.allow_retrack_in_window),
        stateless_tree_context=bool(exact_args.stateless_tree_context),
        head_mode=str(exact_args.head_mode),
        q_utility_weight=float(exact_args.q_utility_weight),
        q_utility_normalize=bool(exact_args.q_utility_normalize),
        leaf_value_mix=float(exact_args.leaf_value_mix),
        seed_rollout_policies=(),
        fast_zero_rollout=bool(exact_args.fast_zero_rollout),
        skip_default_rollout_seed=bool(exact_args.skip_default_rollout_seed),
        complete_root_q_with_value=bool(exact_args.complete_root_q_with_value),
        visit_unvisited_first=bool(exact_args.visit_unvisited_first),
        duration_normalize_q=bool(exact_args.duration_normalize_q),
        prior_q_beta=float(exact_args.prior_q_beta),
        prior_search_bias=float(exact_args.prior_search_bias),
        forbidden_actions=(),
        sensor_action_mode=str(exact_args.sensor_action_mode),
        disable_x_search=bool(exact_args.disable_x_search),
        canonical_search_only=bool(exact_args.canonical_search_only),
        branch_rollout_threshold=float(getattr(exact_args, "branch_rollout_threshold", 0.65)),
    )


def target_from_distribution(obs, pi_phys: dict[int, float], q_phys: dict[int, float], ret: float, elapsed_ms: float, budget_ms: float) -> SearchTarget:
    x = tokenize(get_adapter(), obs, selected=set(), search_count=0).astype(np.float32)
    slot = slot_features(obs, float(elapsed_ms), 0, 0, -1, float(budget_ms)).astype(np.float32)
    pi = np.zeros((MAXT + 1,), dtype=np.float32)
    sensor_pi = np.zeros((MAXT + 1, 2), dtype=np.float32)
    q = np.zeros((MAXT + 1,), dtype=np.float32)
    q_mask = np.zeros((MAXT + 1,), dtype=np.float32)
    sensor_q = np.zeros((MAXT + 1, 2), dtype=np.float32)
    sensor_q_mask = np.zeros((MAXT + 1, 2), dtype=np.float32)
    for action, p in pi_phys.items():
        base, sensor = xs_decode_action(int(action), MAXT)
        if base < 0:
            continue
        sid = 0 if sensor is None else int(np.clip(sensor, 0, 1))
        pi[int(base)] += float(p)
        sensor_pi[int(base), sid] += float(p)
    for action, val in q_phys.items():
        base, sensor = xs_decode_action(int(action), MAXT)
        if base < 0:
            continue
        sid = 0 if sensor is None else int(np.clip(sensor, 0, 1))
        q[int(base)] = max(float(q[int(base)]), float(val)) if q_mask[int(base)] > 0 else float(val)
        q_mask[int(base)] = 1.0
        sensor_q[int(base), sid] = float(val)
        sensor_q_mask[int(base), sid] = 1.0
    total = float(sensor_pi.sum())
    if total > 0.0:
        sensor_pi /= total
        pi /= total
    target = SearchTarget(
        x=x,
        slot=slot,
        pi=pi,
        q=q,
        q_mask=q_mask,
        search_count=0,
        track_count=0,
        sensor_pi=sensor_pi,
        sensor_q=sensor_q,
        sensor_q_mask=sensor_q_mask,
    )
    target.ret = float(ret)
    return target


def candidate_actions(
    model,
    sim: SnapshotSimulator,
    prefix: Sequence[int],
    exact_args,
    max_candidates: int,
    candidate_mode: str = "prior",
) -> List[int]:
    mcts = make_mcts(model, sim, prefix, exact_args)
    st = mcts.state(())
    valid = [int(a) for a in mcts.valid_actions(st.obs)]
    priors, _, _, _, _ = mcts._net(st.obs, ())
    ranked = sorted(valid, key=lambda a: float(mcts._prior_for_action(priors, int(a))), reverse=True)
    deadline = np.asarray(st.obs.get("t_deadline", []), dtype=np.float32)
    urgent = sorted(
        [
            int(a)
            for a in valid
            if int(xs_decode_action(int(a), MAXT)[0]) > 0
            and int(xs_decode_action(int(a), MAXT)[0]) - 1 < len(deadline)
        ],
        key=lambda a: (float(deadline[int(xs_decode_action(int(a), MAXT)[0]) - 1]), int(a)),
    )
    out: List[int] = []

    def add(action: int) -> None:
        if len(out) < int(max_candidates) and int(action) not in out:
            out.append(int(action))

    for a in valid:
        base, _ = xs_decode_action(int(a), MAXT)
        if int(base) == 0:
            add(int(a))
    mode = str(candidate_mode)
    if mode == "urgent":
        for a in urgent:
            add(int(a))
    elif mode == "mixed":
        # Interleave learned-prior proposals with EDF-style urgent targets.
        # This gives the Q heads counterfactual coverage for both what the
        # model already likes and what a deadline heuristic would challenge.
        for i in range(max(len(ranked), len(urgent))):
            if i < len(ranked):
                add(int(ranked[i]))
            if i < len(urgent):
                add(int(urgent[i]))
    else:
        for a in ranked:
            add(int(a))
    return out[: int(max_candidates)]


def continue_return(model, sim: SnapshotSimulator, start_prefix: Sequence[int], exact_args, horizon_ms: float) -> float:
    prefix = [int(a) for a in start_prefix]
    root_reward = float(sim.replay(()).reward)
    start_state = sim.replay(prefix)
    if start_state.terminal:
        return float(start_state.reward - root_reward)
    elapsed = float(start_state.dt_ms)
    while elapsed < float(horizon_ms):
        mcts = make_mcts(model, sim, prefix, exact_args)
        root = mcts.run()
        if not root.children:
            break
        action = int(choose_root_action(root, exact_args.select_mode))
        next_state = sim.replay([*prefix, action])
        if next_state.dt_ms <= elapsed:
            break
        prefix.append(action)
        elapsed = float(next_state.dt_ms)
        if next_state.terminal:
            break
    final_state = sim.replay(prefix)
    return float(final_state.reward - root_reward)


def advance_real_env(model, eng, debt: float, exact_args, windows: int, env_cfg) -> float:
    for _ in range(int(windows)):
        if bool(eng.term_buf[0]):
            break
        remaining = 200.0
        while remaining > 0.0 and not bool(eng.term_buf[0]):
            sim = SnapshotSimulator(
                eng,
                debt,
                env_cfg,
                bool(getattr(exact_args, "use_arrival_feature", False)),
                bool(getattr(exact_args, "use_grid_feature", False)),
            )
            mcts = make_mcts(model, sim, [], exact_args)
            root = mcts.run()
            if not root.children:
                debt += remaining
                break
            action = int(choose_root_action(root, exact_args.select_mode))
            st = sim.replay(())
            fallback = [int(a) for a in mcts.valid_actions(st.obs) if int(a) != int(action)]
            fallback.extend([MAXT + 1, MAXT + 2])
            reward, dt, debt, executed = sim.commit_first_valid([int(action), *fallback], remaining)
            if executed is None or dt <= 0.0:
                debt += remaining
                break
            remaining -= float(dt)
    return float(debt)


def build_targets(model, exact_args, args) -> tuple[list[SearchTarget], pd.DataFrame]:
    targets: list[SearchTarget] = []
    audit = []
    checkpoint_every = max(0, int(getattr(args, "checkpoint_every", 0)))
    for seed in parse_ints(args.train_seeds):
        for init in parse_ints(args.train_initials):
            for rate in parse_floats(args.train_rates):
                env_cfg = env_cfg_for(float(rate), exact_args)
                for prefix_windows in parse_ints(args.prefix_windows):
                    eng = build_env(_DummyPlanner(), int(init), MAXT, int(seed), 200, env_cfg)
                    eng.reset(seed=int(seed))
                    try:
                        try:
                            debt = advance_real_env(model, eng, 0.0, exact_args, int(prefix_windows), env_cfg)
                            sim = SnapshotSimulator(
                                eng,
                                debt,
                                env_cfg,
                                bool(getattr(exact_args, "use_arrival_feature", False)),
                                bool(getattr(exact_args, "use_grid_feature", False)),
                            )
                            obs = attach_env_obs(
                                get_obs(eng, debt),
                                env_cfg,
                                bool(getattr(exact_args, "use_arrival_feature", False)),
                                bool(getattr(exact_args, "use_grid_feature", False)),
                            )
                            cands = candidate_actions(model, sim, [], exact_args, args.max_candidates, args.candidate_mode)
                            q_vals = {}
                            for action in cands:
                                q_vals[int(action)] = continue_return(
                                    model,
                                    sim,
                                    [int(action)],
                                    exact_args,
                                    float(args.actionq_horizon_windows) * 200.0,
                                )
                            vals = np.asarray([q_vals[a] for a in cands], dtype=np.float64)
                            logits = (vals - float(np.max(vals))) / max(1e-6, float(args.policy_tau))
                            probs = np.exp(np.clip(logits, -40.0, 40.0))
                            probs /= max(float(np.sum(probs)), 1e-12)
                            pi_phys = {int(a): float(p) for a, p in zip(cands, probs)}
                            target = target_from_distribution(obs, pi_phys, q_vals, float(np.max(vals)), 0.0, 200.0)
                            targets.append(target)
                            best_action = int(cands[int(np.argmax(vals))])
                            search_actions = [a for a in cands if xs_decode_action(int(a), MAXT)[0] == 0]
                            best_search = max([q_vals[a] for a in search_actions], default=np.nan)
                            audit.append(
                                {
                                    "seed": int(seed),
                                    "initial": int(init),
                                    "rate": float(rate),
                                    "prefix_windows": int(prefix_windows),
                                    "status": "ok",
                                    "candidates": len(cands),
                                    "best_action": best_action,
                                    "best_base": int(xs_decode_action(best_action, MAXT)[0]),
                                    "best_return": float(np.max(vals)),
                                    "best_search_return": float(best_search),
                                    "pi_search": float(sum(p for a, p in pi_phys.items() if xs_decode_action(int(a), MAXT)[0] == 0)),
                                    "error": "",
                                }
                            )
                        except Exception as exc:
                            audit.append(
                                {
                                    "seed": int(seed),
                                    "initial": int(init),
                                    "rate": float(rate),
                                    "prefix_windows": int(prefix_windows),
                                    "status": "error",
                                    "candidates": 0,
                                    "best_action": -1,
                                    "best_base": -1,
                                    "best_return": np.nan,
                                    "best_search_return": np.nan,
                                    "pi_search": np.nan,
                                    "error": repr(exc),
                                }
                            )
                        print(audit[-1], flush=True)
                        if checkpoint_every > 0 and (len(targets) % checkpoint_every == 0 or audit[-1].get("status") == "error"):
                            Path(args.save_targets).parent.mkdir(parents=True, exist_ok=True)
                            torch.save(targets, args.save_targets)
                            pd.DataFrame(audit).to_csv(args.audit_out, index=False)
                    finally:
                        eng.close()
    return targets, pd.DataFrame(audit)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--train-initials", default="60,100")
    ap.add_argument("--train-rates", default="0,4,8")
    ap.add_argument("--train-seeds", default="931")
    ap.add_argument("--prefix-windows", default="0,1,2,4,8,12")
    ap.add_argument("--max-candidates", type=int, default=10)
    ap.add_argument("--candidate-mode", choices=["prior", "urgent", "mixed"], default="prior")
    ap.add_argument("--actionq-horizon-windows", type=int, default=8)
    ap.add_argument("--policy-tau", type=float, default=2.0)
    ap.add_argument("--rollouts", type=int, default=1)
    ap.add_argument("--c-puct", type=float, default=1.25)
    ap.add_argument("--horizon-windows", type=int, default=2)
    ap.add_argument("--expand-top-k", type=int, default=48)
    ap.add_argument("--rollout-policy", choices=["model", "branch", "q", "pq", "random", "value", "edge", "edf", "est", "mixed"], default="branch")
    ap.add_argument("--branch-rollout-threshold", type=float, default=0.65)
    ap.add_argument("--head-mode", choices=["p", "pv", "pq", "pvq"], default="pq")
    ap.add_argument("--prior-mode", choices=["factorized", "flat", "branch_corrected", "physical_flat", "true_physical_flat"], default="factorized")
    ap.add_argument("--prior-uniform-mix", type=float, default=0.03)
    ap.add_argument("--prior-search-bias", type=float, default=0.0)
    ap.add_argument("--adaptive-search-bias", type=float, default=0.0)
    ap.add_argument("--adaptive-search-target-load", type=float, default=0.75)
    ap.add_argument("--leaf-value-mix", type=float, default=0.0)
    ap.add_argument("--q-scale", type=float, default=100.0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--train-steps", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=8e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--type-loss-weight", type=float, default=1.0)
    ap.add_argument("--track-loss-weight", type=float, default=1.0)
    ap.add_argument("--sensor-loss-weight", type=float, default=0.5)
    ap.add_argument("--joint-policy-loss-weight", type=float, default=0.5)
    ap.add_argument("--value-loss-weight", type=float, default=0.5)
    ap.add_argument("--type-q-loss-weight", type=float, default=0.0)
    ap.add_argument("--track-q-loss-weight", type=float, default=0.0)
    ap.add_argument("--sensor-q-loss-weight", type=float, default=0.0)
    ap.add_argument("--factor-value-loss-weight", type=float, default=0.25)
    ap.add_argument("--policy-kl-weight", type=float, default=0.0)
    ap.add_argument("--policy-positive-only", action="store_true")
    ap.add_argument("--policy-positive-margin", type=float, default=0.0)
    ap.add_argument("--train-encoder", action="store_true")
    ap.add_argument("--train-calibration-only", action="store_true")
    ap.add_argument("--load-actionq-targets", default="")
    ap.add_argument("--replay-targets", default="")
    ap.add_argument("--max-replay-targets", type=int, default=0)
    ap.add_argument("--actionq-repeat", type=int, default=1)
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--env-mode", default="radarxs_mission_delta")
    ap.add_argument("--track-loss-penalty", type=float, default=4.0)
    ap.add_argument("--target-service-weight", type=float, default=10.0)
    ap.add_argument("--target-service-horizon-ms", type=float, default=3000.0)
    ap.add_argument("--sector-staleness-weight", type=float, default=0.01)
    ap.add_argument("--search-frame-overdue-weight", type=float, default=0.01)
    ap.add_argument("--search-frame-drop-penalty", type=float, default=4.0)
    ap.add_argument("--use-arrival-feature", action="store_true")
    ap.add_argument("--use-grid-feature", action="store_true")
    ap.add_argument("--enable-x-band", action="store_true")
    ap.add_argument("--single-sensor", action="store_true")
    ap.add_argument("--zero-action-rewards", action="store_true")
    ap.add_argument("--save-targets", default=str(OUT / "onpolicy_actionq_targets.pt"))
    ap.add_argument("--save-state", default=str(OUT / "onpolicy_actionq_state.pt"))
    ap.add_argument("--audit-out", default=str(OUT / "onpolicy_actionq_audit.csv"))
    args = ap.parse_args()

    exact_args = make_exact_args(args)
    device = torch.device(args.device)
    model = load_model(exact_args).to(device)
    load_state_into(model, args.state, device)
    model.eval()
    rng = random.Random(int(args.seed))
    if args.load_actionq_targets:
        targets = load_targets(args.load_actionq_targets)
        audit = pd.DataFrame()
    else:
        targets, audit = build_targets(model, exact_args, args)
    if int(args.actionq_repeat) > 1 and targets:
        targets = [t for t in targets for _ in range(int(args.actionq_repeat))]
    if args.replay_targets:
        replay = load_targets(args.replay_targets)
        if int(args.max_replay_targets) > 0 and len(replay) > int(args.max_replay_targets):
            replay = rng.sample(replay, int(args.max_replay_targets))
        targets = [*targets, *replay]
    Path(args.audit_out).parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.audit_out, index=False)
    torch.save(targets, args.save_targets)
    metrics = train_pv(model, targets, args, device) if int(args.train_steps) > 0 else {}
    torch.save(model.state_dict(), args.save_state)
    print({"targets": len(targets), "metrics": metrics, "audit": args.audit_out, "state": args.save_state}, flush=True)


if __name__ == "__main__":
    main()
