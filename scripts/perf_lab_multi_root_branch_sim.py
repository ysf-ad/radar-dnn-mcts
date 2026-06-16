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
    parser.add_argument("--root-counts", default="1,4,8,16,32")
    parser.add_argument("--branches-per-root", type=int, default=8)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--out", type=Path, default=Path("perf_lab_multi_root_branch_sim.json"))
    args = parser.parse_args()

    from batched_branch_sim import BatchedMultiRootBranchSimulator, BatchedRootBranchSimulator
    from final_radar_campaign import get_obs
    from perf_fast_planner import BatchedActionAttentionScorer
    from repaired_campaign_tools import env_preset_cfg
    from two_sensor_physical_head_eval import ActionAttentionFactorizedNet

    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)

    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1

    max_roots = max(int(x) for x in str(args.root_counts).split(",") if x.strip())
    seeds = [int(args.seed) + i for i in range(max_roots)]
    max_branches = max_roots * int(args.branches_per_root)

    multi = BatchedMultiRootBranchSimulator(
        initial_targets=args.initial_targets,
        seeds=seeds,
        env_cfg=env_cfg,
        batch_size=max_branches,
    )
    per_root = [
        BatchedRootBranchSimulator(
            initial_targets=args.initial_targets,
            seed=seed,
            env_cfg=env_cfg,
            batch_size=int(args.branches_per_root),
        )
        for seed in seeds
    ]

    try:
        observations = [get_obs(eng, 0.0) for eng in multi.root_engs]
        model = ActionAttentionFactorizedNet(48, 4, 2).eval()
        scorer = BatchedActionAttentionScorer(model, env_cfg, device="cpu")
        tables = scorer.all_root_action_tables(observations, max_actions=int(args.branches_per_root))

        report = {
            "initial_targets": int(args.initial_targets),
            "rate": float(args.rate),
            "seed": int(args.seed),
            "branches_per_root": int(args.branches_per_root),
            "root_counts": [],
        }

        for root_count in [int(x) for x in str(args.root_counts).split(",") if x.strip()]:
            root_count = int(root_count)
            flat_root_indices = []
            flat_actions = []
            root_actions = []
            for root_idx in range(root_count):
                count = min(int(args.branches_per_root), int(tables.counts[root_idx]))
                actions = np.asarray(tables.actions[root_idx, :count], dtype=np.int32)
                root_actions.append(actions)
                flat_actions.extend(actions.tolist())
                flat_root_indices.extend([root_idx] * int(actions.size))
            flat_actions = np.asarray(flat_actions, dtype=np.int32)
            flat_root_indices = np.asarray(flat_root_indices, dtype=np.int32)

            loop_results = [per_root[root_idx].step_actions(actions, include_observations=False) for root_idx, actions in enumerate(root_actions)]
            multi_result = multi.step_root_actions(flat_root_indices, flat_actions, include_observations=False)
            loop_executed = np.concatenate([r.executed for r in loop_results]) if loop_results else np.empty((0,), dtype=np.int32)
            loop_dt = np.concatenate([r.dt_ms for r in loop_results]) if loop_results else np.empty((0,), dtype=np.float32)
            loop_rewards = np.concatenate([r.rewards for r in loop_results]) if loop_results else np.empty((0,), dtype=np.float32)

            loop_times = []
            multi_times = []
            for i in range(int(args.warmup) + int(args.iters)):
                t0 = time.perf_counter()
                for root_idx, actions in enumerate(root_actions):
                    _ = per_root[root_idx].step_actions(actions, include_observations=False)
                loop_ms = (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                _ = multi.step_root_actions(flat_root_indices, flat_actions, include_observations=False)
                multi_ms = (time.perf_counter() - t1) * 1000.0
                if i >= int(args.warmup):
                    loop_times.append(loop_ms)
                    multi_times.append(multi_ms)

            loop_stat = stats(loop_times)
            multi_stat = stats(multi_times)
            branch_count = int(flat_actions.size)
            report["root_counts"].append(
                {
                    "roots": int(root_count),
                    "branches": branch_count,
                    "executed_match": bool(np.array_equal(loop_executed, multi_result.executed)),
                    "dt_match": bool(np.allclose(loop_dt, multi_result.dt_ms)),
                    "reward_match": bool(np.allclose(loop_rewards, multi_result.rewards)),
                    "per_root_loop": loop_stat,
                    "multi_root_batched": multi_stat,
                    "speedup_mean": float(loop_stat["mean_ms"] / max(multi_stat["mean_ms"], 1e-12)),
                    "branches_per_second_loop": float(branch_count / max(loop_stat["mean_ms"], 1e-12) * 1000.0),
                    "branches_per_second_multi": float(branch_count / max(multi_stat["mean_ms"], 1e-12) * 1000.0),
                }
            )
    finally:
        multi.close()
        for sim in per_root:
            sim.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
