from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "radar_dnn_mcts"))


def stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def time_call(buckets: dict[str, list[float]], name: str, fn):
    t0 = time.perf_counter()
    out = fn()
    buckets[name].append((time.perf_counter() - t0) * 1000.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,8,16,32,64,128,256")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--include-observations", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/profile_branch_sim_steps.json"))
    args = parser.parse_args()

    from batched_branch_sim import BatchedRootBranchSimulator
    from perf_fast_planner import physical_action_arrays
    from repaired_campaign_tools import env_preset_cfg

    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1
    batch_sizes = [int(x) for x in str(args.batch_sizes).split(",") if x.strip()]

    rows = []
    for batch_size in batch_sizes:
        sim = BatchedRootBranchSimulator(
            int(args.initial_targets),
            int(args.seed),
            env_cfg,
            batch_size=int(batch_size),
        )
        try:
            snapshot = sim.snapshot_root()
            actions, _bases, _sensors = physical_action_arrays(sim.root_eng.driver.get_obs() if hasattr(sim.root_eng, "driver") else {}, max_trackers=0)
        except Exception:
            actions = np.empty((0,), dtype=np.int32)
        finally:
            pass
        # Use simple valid-looking physical actions from the current root scorer.
        from final_radar_campaign import get_obs
        from perf_fast_planner import physical_action_arrays as root_actions

        root_obs = get_obs(sim.root_eng, 0.0)
        all_actions, _bases, _sensors = root_actions(root_obs)
        if all_actions.size == 0:
            action_batch = np.zeros((batch_size,), dtype=np.int32)
        else:
            reps = int(np.ceil(batch_size / all_actions.size))
            action_batch = np.tile(all_actions.astype(np.int32), reps)[:batch_size]

        buckets: dict[str, list[float]] = defaultdict(list)
        for idx in range(int(args.warmup) + int(args.iters)):
            record = idx >= int(args.warmup)

            def maybe(name: str, fn):
                if record:
                    return time_call(buckets, name, fn)
                return fn()

            maybe("restore_root", lambda: sim.restore_root(snapshot, count=batch_size))
            maybe("copy_actions", lambda: sim.act_buf.__setitem__(slice(0, batch_size), action_batch))

            def step_validated():
                from pufferlib.ocean.radarxs import binding

                return binding.vec_step_validated_into(sim.env, sim.dt_buf, sim.executed_buf, batch_size)

            maybe("validated_step_into", step_validated)
            rewards = maybe("copy_rewards", lambda: np.asarray(sim.rew_buf[:batch_size], dtype=np.float32).copy())
            _dt = maybe("copy_dt", lambda: sim.dt_buf[:batch_size].copy())
            _executed = maybe("copy_executed", lambda: sim.executed_buf[:batch_size].copy())
            if bool(args.include_observations):
                from pufferlib.ocean.radarxs.engine import get_obs_from_buf
                from two_sensor_physical_head_eval import MAXT

                maybe("decode_observations", lambda: [get_obs_from_buf(sim.obs_buf[i], max_trackers=MAXT) for i in range(batch_size)])

        summary = {name: stats(vals) for name, vals in sorted(buckets.items(), key=lambda kv: sum(kv[1]), reverse=True)}
        total = sum(item["mean_ms"] for item in summary.values())
        for item in summary.values():
            item["mean_percent"] = float(100.0 * item["mean_ms"] / max(total, 1e-12))
        rows.append(
            {
                "batch_size": int(batch_size),
                "include_observations": bool(args.include_observations),
                "mean_reward_sum": float(np.sum(rewards)) if "rewards" in locals() else 0.0,
                "stage_profile": summary,
            }
        )
        sim.close()

    report = {
        "initial_targets": int(args.initial_targets),
        "rate": float(args.rate),
        "seed": int(args.seed),
        "iters": int(args.iters),
        "warmup": int(args.warmup),
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
