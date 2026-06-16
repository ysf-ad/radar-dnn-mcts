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


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--wave-sizes", default="1,4,8,16,32")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--out", type=Path, default=Path("perf_lab_neural_exact_wave.json"))
    args = parser.parse_args()

    from batched_branch_sim import BatchedRootBranchSimulator
    from batched_window_expansion import BatchedWindowExpansionScorer, BranchPrefix
    from final_radar_campaign import get_obs
    from perf_fast_planner import FastActionAttentionPlanner
    from repaired_campaign_tools import env_preset_cfg
    from two_sensor_physical_head_eval import ActionAttentionFactorizedNet

    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)
    device = torch.device(args.device)

    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1

    max_wave = max(int(x) for x in str(args.wave_sizes).split(",") if x.strip())
    sim = BatchedRootBranchSimulator(args.initial_targets, args.seed, env_cfg, batch_size=max_wave)
    root_snapshot = sim.snapshot_root()
    root_obs = get_obs(sim.root_eng, 0.0)
    model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    planner = FastActionAttentionPlanner(model, env_cfg, device=device, use_amp=bool(args.amp), use_compile=bool(args.compile))

    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "amp": bool(args.amp),
        "compile": bool(args.compile),
        "initial_targets": int(args.initial_targets),
        "rate": float(args.rate),
        "seed": int(args.seed),
        "waves": [],
    }

    try:
        for wave_size in [int(x) for x in str(args.wave_sizes).split(",") if x.strip()]:
            wave_size = min(int(wave_size), int(max_wave))
            setup_times = []
            neural_times = []
            sim_times = []
            combined_times = []
            action_counts = []
            reward_sums = []
            executed_counts = []
            last_actions = []

            for i in range(int(args.warmup) + int(args.iters)):
                sync(device)
                t0 = time.perf_counter()
                scorer = BatchedWindowExpansionScorer(planner, root_obs, budget_ms=200.0)
                sync(device)
                setup_ms = (time.perf_counter() - t0) * 1000.0

                sync(device)
                t1 = time.perf_counter()
                prefixes = scorer.expand_prefixes([BranchPrefix()], top_k=wave_size)
                actions = np.asarray([p.actions[-1] for p in prefixes], dtype=np.int32)
                sync(device)
                neural_ms = (time.perf_counter() - t1) * 1000.0

                t2 = time.perf_counter()
                result = sim.step_actions(actions, snapshot=root_snapshot)
                sim_ms = (time.perf_counter() - t2) * 1000.0
                combined_ms = setup_ms + neural_ms + sim_ms
                if i >= int(args.warmup):
                    setup_times.append(setup_ms)
                    neural_times.append(neural_ms)
                    sim_times.append(sim_ms)
                    combined_times.append(combined_ms)
                    action_counts.append(int(actions.size))
                    reward_sums.append(float(np.sum(result.rewards)))
                    executed_counts.append(int(np.sum(result.executed >= 0)))
                    last_actions = [int(x) for x in actions.tolist()]

            setup_stat = stats(setup_times)
            neural_stat = stats(neural_times)
            sim_stat = stats(sim_times)
            combined_stat = stats(combined_times)
            report["waves"].append(
                {
                    "requested_wave_size": int(wave_size),
                    "mean_action_count": float(np.mean(action_counts)) if action_counts else 0.0,
                    "mean_executed_count": float(np.mean(executed_counts)) if executed_counts else 0.0,
                    "mean_reward_sum": float(np.mean(reward_sums)) if reward_sums else 0.0,
                    "last_actions": last_actions,
                    "root_encode_setup": setup_stat,
                    "neural_expand": neural_stat,
                    "exact_branch_sim": sim_stat,
                    "combined_wave": combined_stat,
                    "sim_fraction_of_combined": float(sim_stat["mean_ms"] / max(combined_stat["mean_ms"], 1e-12)),
                    "neural_fraction_of_combined": float((setup_stat["mean_ms"] + neural_stat["mean_ms"]) / max(combined_stat["mean_ms"], 1e-12)),
                }
            )
    finally:
        sim.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
