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
    parser.add_argument("--batch-sizes", default="1,8,32,128")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/perf_lab_batched_roots.json"))
    args = parser.parse_args()

    from final_radar_campaign import get_obs
    from perf_fast_planner import BatchedActionAttentionScorer, select_best_action
    from repaired_campaign_tools import EDFPlanner, build_env, env_preset_cfg
    from two_sensor_physical_head_eval import MAXT, ActionAttentionFactorizedNet, PhysicalHeadPlanner

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

    baseline_model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    batch_model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    batch_model.load_state_dict(baseline_model.state_dict())
    baseline = PhysicalHeadPlanner(baseline_model, "two_row_action_attention_qpolicy_factored_loss", env_cfg)
    batcher = BatchedActionAttentionScorer(
        batch_model,
        env_cfg,
        device=device,
        use_amp=bool(args.amp),
        use_cuda_graph=bool(args.cuda_graph),
    )

    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "amp": bool(args.amp),
        "cuda_graph": bool(args.cuda_graph),
        "batch_sizes": [],
    }
    for batch_size in [int(x) for x in str(args.batch_sizes).split(",") if x.strip()]:
        obs_batch = observations[:batch_size]
        # Compare first-action equivalence. The full baseline planner chooses a
        # whole plan; score_actions gives the directly comparable root decision.
        base_actions = []
        for obs in obs_batch:
            score = baseline.score_actions(obs)
            best = select_best_action(score, obs)
            base_actions.append(-1 if best is None else int(best))
        fast_actions = batcher.best_actions(obs_batch).astype(np.int64).tolist()
        fast_gpu_actions = batcher.best_actions_torch(obs_batch).astype(np.int64).tolist()
        prepared = batcher.prepare_root_batch(obs_batch)
        device_prepared = batcher.prepared_to_device(prepared)
        fast_prepared_actions = batcher.best_actions_prepared_torch(prepared).astype(np.int64).tolist()
        fast_prepared_graph_actions = batcher.best_actions_prepared_graph(prepared).astype(np.int64).tolist()
        fast_device_prepared_actions = batcher.best_actions_prepared_device_graph(device_prepared).astype(np.int64).tolist()

        seq_times = []
        batch_times = []
        batch_gpu_times = []
        prepared_times = []
        prepared_graph_times = []
        device_prepared_graph_times = []
        for i in range(int(args.warmup) + int(args.iters)):
            sync(device)
            t0 = time.perf_counter()
            for obs in obs_batch:
                score = baseline.score_actions(obs)
                _ = select_best_action(score, obs)
            sync(device)
            seq_ms = (time.perf_counter() - t0) * 1000.0

            sync(device)
            t1 = time.perf_counter()
            _ = batcher.best_actions(obs_batch)
            sync(device)
            batch_ms = (time.perf_counter() - t1) * 1000.0

            sync(device)
            t2 = time.perf_counter()
            _ = batcher.best_actions_torch(obs_batch)
            sync(device)
            batch_gpu_ms = (time.perf_counter() - t2) * 1000.0

            sync(device)
            t3 = time.perf_counter()
            _ = batcher.best_actions_prepared_torch(prepared)
            sync(device)
            prepared_ms = (time.perf_counter() - t3) * 1000.0

            sync(device)
            t4 = time.perf_counter()
            _ = batcher.best_actions_prepared_graph(prepared)
            sync(device)
            prepared_graph_ms = (time.perf_counter() - t4) * 1000.0

            sync(device)
            t5 = time.perf_counter()
            _ = batcher.best_actions_prepared_device_graph(device_prepared)
            sync(device)
            device_prepared_graph_ms = (time.perf_counter() - t5) * 1000.0
            if i >= int(args.warmup):
                seq_times.append(seq_ms)
                batch_times.append(batch_ms)
                batch_gpu_times.append(batch_gpu_ms)
                prepared_times.append(prepared_ms)
                prepared_graph_times.append(prepared_graph_ms)
                device_prepared_graph_times.append(device_prepared_graph_ms)

        seq = stats(seq_times)
        bat = stats(batch_times)
        bat_gpu = stats(batch_gpu_times)
        prep = stats(prepared_times)
        prep_graph = stats(prepared_graph_times)
        device_prep_graph = stats(device_prepared_graph_times)
        report["batch_sizes"].append(
            {
                "batch": int(batch_size),
                "actions_match": base_actions == fast_actions,
                "gpu_actions_match": base_actions == fast_gpu_actions,
                "prepared_actions_match": base_actions == fast_prepared_actions,
                "prepared_graph_actions_match": base_actions == fast_prepared_graph_actions,
                "device_prepared_graph_actions_match": base_actions == fast_device_prepared_actions,
                "sequential_loop": seq,
                "batched_score": bat,
                "batched_score_gpu_select": bat_gpu,
                "prepared_gpu_select": prep,
                "prepared_cuda_graph": prep_graph,
                "device_prepared_cuda_graph": device_prep_graph,
                "speedup_mean": float(seq["mean_ms"] / max(bat["mean_ms"], 1e-12)),
                "gpu_select_speedup_mean": float(seq["mean_ms"] / max(bat_gpu["mean_ms"], 1e-12)),
                "prepared_speedup_mean": float(seq["mean_ms"] / max(prep["mean_ms"], 1e-12)),
                "prepared_graph_speedup_mean": float(seq["mean_ms"] / max(prep_graph["mean_ms"], 1e-12)),
                "device_prepared_graph_speedup_mean": float(seq["mean_ms"] / max(device_prep_graph["mean_ms"], 1e-12)),
                "states_per_second_sequential": float(batch_size / max(seq["mean_ms"], 1e-12) * 1000.0),
                "states_per_second_batched": float(batch_size / max(bat["mean_ms"], 1e-12) * 1000.0),
                "states_per_second_batched_gpu_select": float(batch_size / max(bat_gpu["mean_ms"], 1e-12) * 1000.0),
                "states_per_second_prepared_gpu_select": float(batch_size / max(prep["mean_ms"], 1e-12) * 1000.0),
                "states_per_second_prepared_cuda_graph": float(batch_size / max(prep_graph["mean_ms"], 1e-12) * 1000.0),
                "states_per_second_device_prepared_cuda_graph": float(batch_size / max(device_prep_graph["mean_ms"], 1e-12) * 1000.0),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
