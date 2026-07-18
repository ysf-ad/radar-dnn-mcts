from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from exact_env_mutual import run_snapshot_exact_episode
from mutual_foundation import DEVICE, MutualRadarNet
from repaired_campaign_tools import EDFPlanner, ESTPlanner
from final_radar_campaign import MAXT, run_fixed, summarize_window_df, seedall
from exact_env_mutual import env_cfg_for


OUT = Path(r"C:\Users\yousi\Downloads\radar_outputs\exact_mcts_target_sweep")
OUT.mkdir(parents=True, exist_ok=True)


def base_args(**kw):
    d = dict(
        windows=2,
        rollouts=4,
        c_puct=1.25,
        expand_top_k=12,
        horizon_windows=5,
        rollout_policy="mixed",
        seed_rollout_policies="edf,est,model",
        prior_mode="factorized",
        epsilon=0.10,
        policy_target="mixed",
        policy_tau=1.0,
        search_alg="puct",
        max_num_considered_actions=16,
        gumbel_scale=0.0,
        mctx_value_scale=0.1,
        mctx_maxvisit_init=50.0,
        eager_edge_depth=1,
        prior_uniform_mix=0.0,
        rollout_est_prob=0.5,
        allow_retrack_in_window=False,
        stateless_tree_context=False,
        head_mode="p",
        q_utility_weight=0.0,
        leaf_value_mix=1.0,
        select_mode="q",
        plan_mode="first_window",
        window_extract="best",
        add_prefix_targets=False,
        max_targets_per_episode=9999,
        gamma=0.99,
        clone_mode="snapshot",
        env_mode="searched_sector_frame",
        track_update_reward=0.30,
        track_loss_penalty=4.0,
        search_refresh_tracked=0,
        search_refresh_gain=0.0,
        search_debt_penalty_weight=0.0,
        sector_staleness_weight=0.0,
        searched_sector_reward_weight=0.10,
        search_frame_overdue_weight=0.05,
        search_frame_desired_ms=3000.0,
        search_frame_deadline_ms=4500.0,
        search_frame_drop_penalty=4.0,
        penalize_hidden_targets=1,
    )
    d.update(kw)
    return SimpleNamespace(**d)


def load_model():
    ckpt = Path(r"C:\Users\yousi\Downloads\Model1 1\CreateValid1\experiments\code\model_code\CreateValid1\results\mutual_alpha_radar_loop\mutual_alpha_model.pt")
    model = MutualRadarNet(d_model=96, nhead=4, nlayers=2).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
    return model.eval()


def main():
    model = load_model()
    cells = [(15, 0.0), (15, 2.0), (50, 0.0), (50, 2.0), (100, 0.0), (100, 2.0)]
    variants = []
    for plan_mode, window_extract, select_mode, policy_target, search_w, frame_w in itertools.product(
        ["atomic", "first_window", "window"],
        ["best", "tree"],
        ["q", "visits"],
        ["mixed", "q_softmax", "mctx"],
        [0.03, 0.10],
        [0.05, 0.15],
    ):
        if plan_mode == "atomic" and window_extract != "best":
            continue
        variants.append(
            dict(
                plan_mode=plan_mode,
                window_extract=window_extract,
                select_mode=select_mode,
                policy_target=policy_target,
                searched_sector_reward_weight=search_w,
                search_frame_overdue_weight=frame_w,
            )
        )
    rows = []
    # Baselines once.
    for init, rate in cells:
        env = env_cfg_for(rate, base_args())
        for name, planner in [("EDF", EDFPlanner(MAXT)), ("EST", ESTPlanner(MAXT))]:
            seedall(401)
            w, _ = run_fixed(planner, name, init, MAXT, 401, 2, 200, env)
            s = summarize_window_df(w, "fixed")
            rows.append(dict(variant=name, initial_targets=init, rate=rate, reward=s["reward_per_200ms_eq"], search=s["search_fraction"], delay=s["mean_delay_active"]))
    for vi, cfg in enumerate(variants):
        rewards, searches, delays = [], [], []
        for init, rate in cells:
            args = base_args(**cfg)
            seedall(401)
            df, _ = run_snapshot_exact_episode(model, args, init, rate, 401, train=False)
            rewards.append(float(df["window_reward"].mean()) if not df.empty else 0.0)
            searches.append(float(df["search_fraction"].iloc[-1]) if not df.empty else 0.0)
            delays.append(float(df["mean_delay_active"].mean()) if not df.empty else 0.0)
        row = dict(variant=f"v{vi:03d}", **cfg, reward=sum(rewards) / len(rewards), search=sum(searches) / len(searches), delay=sum(delays) / len(delays))
        rows.append(row)
        print("target_sweep", row, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "target_sweep_partial.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "target_sweep.csv", index=False)
    print(df.sort_values("reward", ascending=False).head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
