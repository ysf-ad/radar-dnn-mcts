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


def make_prefixes(obs: dict, count: int):
    from batched_window_expansion import BranchPrefix, prefix_after_action
    from perf_fast_planner import physical_action_arrays
    from two_sensor_physical_head_eval import MAXT

    root = BranchPrefix()
    actions, _bases, _sensors = physical_action_arrays(obs, selected=set(), max_trackers=MAXT)
    prefixes = [root]
    # Build a deterministic spread of prefix depths/actions so the benchmark
    # exercises different selected masks and slot contexts.
    for idx, action in enumerate(actions):
        if len(prefixes) >= int(count):
            break
        p1 = prefix_after_action(obs, root, int(action))
        prefixes.append(p1)
        if len(prefixes) >= int(count):
            break
        for action2 in actions[idx + 1 : idx + 3]:
            if len(prefixes) >= int(count):
                break
            prefixes.append(prefix_after_action(obs, p1, int(action2)))
    while len(prefixes) < int(count):
        prefixes.append(prefixes[len(prefixes) % max(1, len(prefixes))])
    return prefixes[: int(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--prefix-batches", default="1,4,8,16,32,64")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--out", type=Path, default=Path("perf_lab_batched_window_expansion.json"))
    args = parser.parse_args()

    from batched_window_expansion import BatchedWindowExpansionScorer
    from final_radar_campaign import get_obs
    from perf_fast_planner import FastActionAttentionPlanner
    from repaired_campaign_tools import EDFPlanner, build_env, env_preset_cfg
    from two_sensor_physical_head_eval import MAXT, ActionAttentionFactorizedNet

    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1

    eng = build_env(EDFPlanner(MAXT), args.initial_targets, MAXT, args.seed, 200, env_cfg)
    eng.reset(seed=args.seed)
    obs = get_obs(eng, 0.0)
    model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    planner = FastActionAttentionPlanner(
        model,
        env_cfg,
        device=device,
        use_amp=bool(args.amp),
        use_compile=bool(args.compile),
        use_cuda_graph=bool(args.cuda_graph),
    )
    scorer = BatchedWindowExpansionScorer(planner, obs, budget_ms=200.0)

    max_prefixes = max(int(x) for x in str(args.prefix_batches).split(",") if x.strip())
    prefixes = make_prefixes(scorer.obs, max_prefixes)

    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "amp": bool(args.amp),
        "compile": bool(args.compile),
        "cuda_graph": bool(args.cuda_graph),
        "prefix_batches": [],
    }

    for batch in [int(x) for x in str(args.prefix_batches).split(",") if x.strip()]:
        batch_prefixes = prefixes[: min(int(batch), len(prefixes))]

        with torch.inference_mode():
            seq_parts = [scorer.score_prefixes([prefix]) for prefix in batch_prefixes]
            batched_ref = scorer.score_prefixes(batch_prefixes)
            gpu_ref = scorer.score_prefixes_gpu_select(batch_prefixes)
            prepared = scorer.prepare_prefixes_device(batch_prefixes)
            prepared_ref = scorer.score_prepared_prefixes_device(prepared)
            prepared_graph_ref = scorer.score_prepared_prefixes_device_graph(prepared)
        seq_actions = np.asarray([part.actions[0] for part in seq_parts], dtype=np.int64)
        seq_scores = np.asarray([part.scores[0] for part in seq_parts], dtype=np.float32)
        actions_match = bool(np.array_equal(seq_actions, batched_ref.actions))
        gpu_actions_match = bool(np.array_equal(seq_actions, gpu_ref.actions))
        prepared_actions_match = bool(np.array_equal(seq_actions, prepared_ref.actions))
        prepared_graph_actions_match = bool(np.array_equal(seq_actions, prepared_graph_ref.actions))
        max_abs_score_diff = float(np.max(np.abs(seq_scores - batched_ref.scores))) if len(seq_scores) else 0.0
        gpu_max_abs_score_diff = float(np.max(np.abs(seq_scores - gpu_ref.scores))) if len(seq_scores) else 0.0
        prepared_max_abs_score_diff = float(np.max(np.abs(seq_scores - prepared_ref.scores))) if len(seq_scores) else 0.0
        prepared_graph_max_abs_score_diff = float(np.max(np.abs(seq_scores - prepared_graph_ref.scores))) if len(seq_scores) else 0.0

        seq_times = []
        batch_times = []
        gpu_select_times = []
        prepared_times = []
        prepared_graph_times = []
        with torch.inference_mode():
            for i in range(int(args.warmup) + int(args.iters)):
                sync(device)
                t0 = time.perf_counter()
                for prefix in batch_prefixes:
                    _ = scorer.score_prefixes([prefix])
                sync(device)
                seq_ms = (time.perf_counter() - t0) * 1000.0

                sync(device)
                t1 = time.perf_counter()
                _ = scorer.score_prefixes(batch_prefixes)
                sync(device)
                batch_ms = (time.perf_counter() - t1) * 1000.0

                sync(device)
                t2 = time.perf_counter()
                _ = scorer.score_prefixes_gpu_select(batch_prefixes)
                sync(device)
                gpu_select_ms = (time.perf_counter() - t2) * 1000.0

                sync(device)
                t3 = time.perf_counter()
                _ = scorer.score_prepared_prefixes_device(prepared)
                sync(device)
                prepared_ms = (time.perf_counter() - t3) * 1000.0

                sync(device)
                t4 = time.perf_counter()
                _ = scorer.score_prepared_prefixes_device_graph(prepared)
                sync(device)
                prepared_graph_ms = (time.perf_counter() - t4) * 1000.0
                if i >= int(args.warmup):
                    seq_times.append(seq_ms)
                    batch_times.append(batch_ms)
                    gpu_select_times.append(gpu_select_ms)
                    prepared_times.append(prepared_ms)
                    prepared_graph_times.append(prepared_graph_ms)

        seq_stat = stats(seq_times)
        batch_stat = stats(batch_times)
        gpu_select_stat = stats(gpu_select_times)
        prepared_stat = stats(prepared_times)
        prepared_graph_stat = stats(prepared_graph_times)
        report["prefix_batches"].append(
            {
                "prefixes": int(len(batch_prefixes)),
                "actions_match": actions_match,
                "gpu_select_actions_match": gpu_actions_match,
                "prepared_actions_match": prepared_actions_match,
                "prepared_graph_actions_match": prepared_graph_actions_match,
                "max_abs_score_diff": max_abs_score_diff,
                "gpu_select_max_abs_score_diff": gpu_max_abs_score_diff,
                "prepared_max_abs_score_diff": prepared_max_abs_score_diff,
                "prepared_graph_max_abs_score_diff": prepared_graph_max_abs_score_diff,
                "sequential_prefixes": seq_stat,
                "batched_prefixes": batch_stat,
                "batched_prefixes_gpu_select": gpu_select_stat,
                "prepared_prefixes_device": prepared_stat,
                "prepared_prefixes_device_graph": prepared_graph_stat,
                "speedup_mean": float(seq_stat["mean_ms"] / max(batch_stat["mean_ms"], 1e-12)),
                "gpu_select_speedup_mean": float(seq_stat["mean_ms"] / max(gpu_select_stat["mean_ms"], 1e-12)),
                "prepared_speedup_mean": float(seq_stat["mean_ms"] / max(prepared_stat["mean_ms"], 1e-12)),
                "prepared_graph_speedup_mean": float(seq_stat["mean_ms"] / max(prepared_graph_stat["mean_ms"], 1e-12)),
                "prefixes_per_second_sequential": float(len(batch_prefixes) / max(seq_stat["mean_ms"], 1e-12) * 1000.0),
                "prefixes_per_second_batched": float(len(batch_prefixes) / max(batch_stat["mean_ms"], 1e-12) * 1000.0),
                "prefixes_per_second_batched_gpu_select": float(
                    len(batch_prefixes) / max(gpu_select_stat["mean_ms"], 1e-12) * 1000.0
                ),
                "prefixes_per_second_prepared_device": float(
                    len(batch_prefixes) / max(prepared_stat["mean_ms"], 1e-12) * 1000.0
                ),
                "prefixes_per_second_prepared_device_graph": float(
                    len(batch_prefixes) / max(prepared_graph_stat["mean_ms"], 1e-12) * 1000.0
                ),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    eng.close()


if __name__ == "__main__":
    main()
