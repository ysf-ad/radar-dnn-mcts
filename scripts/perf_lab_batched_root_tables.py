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
    parser.add_argument("--batch-sizes", default="1,8,32,128")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--out", type=Path, default=Path("perf_lab_batched_root_tables.json"))
    args = parser.parse_args()

    from final_radar_campaign import get_obs
    from perf_fast_planner import BatchedActionAttentionScorer
    from repaired_campaign_tools import EDFPlanner, build_env, env_preset_cfg
    from two_sensor_physical_head_eval import MAXT, ActionAttentionFactorizedNet

    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)
    device = torch.device(args.device)

    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1

    max_batch = max(int(x) for x in str(args.batch_sizes).split(",") if x.strip())
    observations = []
    for idx in range(max_batch):
        seed = int(args.seed) + idx
        eng = build_env(EDFPlanner(MAXT), args.initial_targets, MAXT, seed, 200, env_cfg)
        eng.reset(seed=seed)
        observations.append(get_obs(eng, 0.0))
        eng.close()

    model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    scorer = BatchedActionAttentionScorer(
        model,
        env_cfg,
        device=device,
        use_amp=bool(args.amp),
        use_compile=bool(args.compile),
    )

    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "amp": bool(args.amp),
        "compile": bool(args.compile),
        "initial_targets": int(args.initial_targets),
        "rate": float(args.rate),
        "seed": int(args.seed),
        "batch_sizes": [],
    }

    for batch_size in [int(x) for x in str(args.batch_sizes).split(",") if x.strip()]:
        obs_batch = observations[: int(batch_size)]
        ref_tables = [scorer.all_root_action_tables([obs]) for obs in obs_batch]
        batch_ref = scorer.all_root_action_tables(obs_batch)
        batch_vec = scorer.all_root_action_tables_vectorized(obs_batch)
        batch_torch = scorer.all_root_action_tables_torch(obs_batch)
        batch_fast = scorer.all_root_action_tables_fast(obs_batch)
        actions_match = True
        scores_match = True
        vectorized_actions_match = True
        vectorized_scores_match = True
        vectorized_counts_match = np.array_equal(batch_ref.counts, batch_vec.counts)
        torch_actions_match = True
        torch_scores_match = True
        torch_counts_match = np.array_equal(batch_ref.counts, batch_torch.counts)
        fast_actions_match = True
        fast_scores_match = True
        fast_counts_match = np.array_equal(batch_ref.counts, batch_fast.counts)
        counts = []
        for row, table in enumerate(ref_tables):
            count = int(table.counts[0])
            counts.append(count)
            actions_match = actions_match and np.array_equal(table.actions[0, :count], batch_ref.actions[row, :count])
            scores_match = scores_match and np.allclose(table.scores[0, :count], batch_ref.scores[row, :count], atol=1e-5)
            vectorized_actions_match = vectorized_actions_match and np.array_equal(
                batch_ref.actions[row, :count], batch_vec.actions[row, :count]
            )
            vectorized_scores_match = vectorized_scores_match and np.allclose(
                batch_ref.scores[row, :count], batch_vec.scores[row, :count], atol=1e-5
            )
            torch_actions_match = torch_actions_match and np.array_equal(
                batch_ref.actions[row, :count], batch_torch.actions[row, :count]
            )
            torch_scores_match = torch_scores_match and np.allclose(
                batch_ref.scores[row, :count], batch_torch.scores[row, :count], atol=1e-5
            )
            fast_actions_match = fast_actions_match and np.array_equal(
                batch_ref.actions[row, :count], batch_fast.actions[row, :count]
            )
            fast_scores_match = fast_scores_match and np.allclose(
                batch_ref.scores[row, :count], batch_fast.scores[row, :count], atol=1e-5
            )

        loop_times = []
        batch_times = []
        vectorized_times = []
        torch_times = []
        fast_times = []
        for i in range(int(args.warmup) + int(args.iters)):
            sync(device)
            t0 = time.perf_counter()
            for obs in obs_batch:
                _ = scorer.all_root_action_tables([obs])
            sync(device)
            loop_ms = (time.perf_counter() - t0) * 1000.0

            sync(device)
            t1 = time.perf_counter()
            _ = scorer.all_root_action_tables(obs_batch)
            sync(device)
            batch_ms = (time.perf_counter() - t1) * 1000.0

            sync(device)
            t2 = time.perf_counter()
            _ = scorer.all_root_action_tables_vectorized(obs_batch)
            sync(device)
            vectorized_ms = (time.perf_counter() - t2) * 1000.0

            sync(device)
            t3 = time.perf_counter()
            _ = scorer.all_root_action_tables_torch(obs_batch)
            sync(device)
            torch_ms = (time.perf_counter() - t3) * 1000.0

            sync(device)
            t4 = time.perf_counter()
            _ = scorer.all_root_action_tables_fast(obs_batch)
            sync(device)
            fast_ms = (time.perf_counter() - t4) * 1000.0

            if i >= int(args.warmup):
                loop_times.append(loop_ms)
                batch_times.append(batch_ms)
                vectorized_times.append(vectorized_ms)
                torch_times.append(torch_ms)
                fast_times.append(fast_ms)

        loop_stat = stats(loop_times)
        batch_stat = stats(batch_times)
        vectorized_stat = stats(vectorized_times)
        torch_stat = stats(torch_times)
        fast_stat = stats(fast_times)
        report["batch_sizes"].append(
            {
                "batch": int(batch_size),
                "actions_match": bool(actions_match),
                "scores_match": bool(scores_match),
                "vectorized_actions_match": bool(vectorized_actions_match),
                "vectorized_scores_match": bool(vectorized_scores_match),
                "vectorized_counts_match": bool(vectorized_counts_match),
                "torch_actions_match": bool(torch_actions_match),
                "torch_scores_match": bool(torch_scores_match),
                "torch_counts_match": bool(torch_counts_match),
                "fast_actions_match": bool(fast_actions_match),
                "fast_scores_match": bool(fast_scores_match),
                "fast_counts_match": bool(fast_counts_match),
                "mean_action_count": float(np.mean(counts)) if counts else 0.0,
                "sequential_root_tables": loop_stat,
                "batched_root_tables": batch_stat,
                "vectorized_batched_root_tables": vectorized_stat,
                "torch_batched_root_tables": torch_stat,
                "fast_batched_root_tables": fast_stat,
                "speedup_mean": float(loop_stat["mean_ms"] / max(batch_stat["mean_ms"], 1e-12)),
                "vectorized_speedup_vs_sequential_mean": float(loop_stat["mean_ms"] / max(vectorized_stat["mean_ms"], 1e-12)),
                "vectorized_speedup_vs_batched_mean": float(batch_stat["mean_ms"] / max(vectorized_stat["mean_ms"], 1e-12)),
                "torch_speedup_vs_sequential_mean": float(loop_stat["mean_ms"] / max(torch_stat["mean_ms"], 1e-12)),
                "torch_speedup_vs_vectorized_mean": float(vectorized_stat["mean_ms"] / max(torch_stat["mean_ms"], 1e-12)),
                "fast_speedup_vs_sequential_mean": float(loop_stat["mean_ms"] / max(fast_stat["mean_ms"], 1e-12)),
                "fast_speedup_vs_vectorized_mean": float(vectorized_stat["mean_ms"] / max(fast_stat["mean_ms"], 1e-12)),
                "root_tables_per_second_sequential": float(batch_size / max(loop_stat["mean_ms"], 1e-12) * 1000.0),
                "root_tables_per_second_batched": float(batch_size / max(batch_stat["mean_ms"], 1e-12) * 1000.0),
                "root_tables_per_second_vectorized": float(batch_size / max(vectorized_stat["mean_ms"], 1e-12) * 1000.0),
                "root_tables_per_second_torch": float(batch_size / max(torch_stat["mean_ms"], 1e-12) * 1000.0),
                "root_tables_per_second_fast": float(batch_size / max(fast_stat["mean_ms"], 1e-12) * 1000.0),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
