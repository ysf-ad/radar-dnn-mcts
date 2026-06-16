from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "radar_dnn_mcts"))


def stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-sizes", default="1,8,32,128")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--out", type=Path, default=Path("perf_lab_batched_branch_sim.json"))
    args = parser.parse_args()

    from batched_branch_sim import BatchedRootBranchSimulator
    from final_radar_campaign import get_obs
    from perf_fast_planner import BatchedActionAttentionScorer
    from repaired_campaign_tools import env_preset_cfg
    from two_sensor_physical_head_eval import MAXT, ActionAttentionFactorizedNet

    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)

    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1

    max_branches = max(int(x) for x in str(args.branch_sizes).split(",") if x.strip())
    model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    scorer = BatchedActionAttentionScorer(model, env_cfg, device="cpu")

    proposal_sim = BatchedRootBranchSimulator(args.initial_targets, args.seed, env_cfg, batch_size=max_branches)
    try:
        root_obs = get_obs(proposal_sim.root_eng, 0.0)
        proposals = scorer.topk_root_proposals([root_obs], k=max_branches)
        all_actions = proposals.actions[0][proposals.valid[0]].astype(np.int32)
        if all_actions.size == 0:
            raise RuntimeError("No valid root actions were produced for branch simulation")
    finally:
        proposal_sim.close()

    report = {
        "initial_targets": int(args.initial_targets),
        "rate": float(args.rate),
        "seed": int(args.seed),
        "branch_sizes": [],
    }

    for branch_size in [int(x) for x in str(args.branch_sizes).split(",") if x.strip()]:
        branch_size = min(int(branch_size), int(all_actions.size))
        actions = all_actions[:branch_size]
        scalar = BatchedRootBranchSimulator(args.initial_targets, args.seed, env_cfg, batch_size=1)
        batched = BatchedRootBranchSimulator(args.initial_targets, args.seed, env_cfg, batch_size=branch_size)
        scalar_results = []
        batched_result = None
        scalar_times = []
        legacy_times = []
        fast_times = []
        try:
            # One correctness pass.
            for action in actions:
                scalar_results.append(scalar.step_actions(np.asarray([action], dtype=np.int32), include_observations=False))
            legacy_result = batched.step_actions_legacy(actions, include_observations=False)
            batched_result = batched.step_actions(actions, include_observations=False)
            scalar_exec = np.asarray([r.executed[0] for r in scalar_results], dtype=np.int32)
            scalar_dt = np.asarray([r.dt_ms[0] for r in scalar_results], dtype=np.float32)
            scalar_rewards = np.asarray([r.rewards[0] for r in scalar_results], dtype=np.float32)

            for i in range(int(args.warmup) + int(args.iters)):
                t0 = time.perf_counter()
                for action in actions:
                    _ = scalar.step_actions(np.asarray([action], dtype=np.int32), include_observations=False)
                scalar_ms = (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                _ = batched.step_actions_legacy(actions, include_observations=False)
                legacy_ms = (time.perf_counter() - t1) * 1000.0

                t2 = time.perf_counter()
                _ = batched.step_actions(actions, include_observations=False)
                fast_ms = (time.perf_counter() - t2) * 1000.0
                if i >= int(args.warmup):
                    scalar_times.append(scalar_ms)
                    legacy_times.append(legacy_ms)
                    fast_times.append(fast_ms)
        finally:
            scalar.close()
            batched.close()

        scalar_stat = stats(scalar_times)
        legacy_stat = stats(legacy_times)
        fast_stat = stats(fast_times)
        report["branch_sizes"].append(
            {
                "branches": int(branch_size),
                "executed_match": bool(np.array_equal(scalar_exec, batched_result.executed)),
                "legacy_executed_match": bool(np.array_equal(scalar_exec, legacy_result.executed)),
                "dt_match": bool(np.allclose(scalar_dt, batched_result.dt_ms)),
                "legacy_dt_match": bool(np.allclose(scalar_dt, legacy_result.dt_ms)),
                "reward_match": bool(np.allclose(scalar_rewards, batched_result.rewards)),
                "legacy_reward_match": bool(np.allclose(scalar_rewards, legacy_result.rewards)),
                "scalar_loop": scalar_stat,
                "batched_legacy_step": legacy_stat,
                "batched_fast_step": fast_stat,
                "speedup_mean": float(scalar_stat["mean_ms"] / max(fast_stat["mean_ms"], 1e-12)),
                "fast_vs_legacy_speedup_mean": float(legacy_stat["mean_ms"] / max(fast_stat["mean_ms"], 1e-12)),
                "branches_per_second_scalar": float(branch_size / max(scalar_stat["mean_ms"], 1e-12) * 1000.0),
                "branches_per_second_batched_legacy": float(branch_size / max(legacy_stat["mean_ms"], 1e-12) * 1000.0),
                "branches_per_second_batched_fast": float(branch_size / max(fast_stat["mean_ms"], 1e-12) * 1000.0),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
