from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
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
        "calls": int(arr.size),
        "total_ms": float(arr.sum()),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def timed(device, buckets, name, fn):
    sync(device)
    t0 = time.perf_counter()
    out = fn()
    sync(device)
    buckets[name].append((time.perf_counter() - t0) * 1000.0)
    return out


def sort_legacy_scores(score, action_arrays, max_actions=None):
    n = int(score.shape[0])
    counts = np.zeros((n,), dtype=np.int32)
    rows = []
    width = 0
    for i, (actions, bases, sensors) in enumerate(action_arrays):
        vals = score[i, bases, sensors]
        finite = np.isfinite(vals)
        row_actions = actions[finite].astype(np.int64, copy=False)
        row_bases = bases[finite].astype(np.int64, copy=False)
        row_sensors = sensors[finite].astype(np.int64, copy=False)
        row_vals = vals[finite].astype(np.float32, copy=False)
        if row_vals.size:
            order = np.argsort(-row_vals)
            row_actions = row_actions[order]
            row_bases = row_bases[order]
            row_sensors = row_sensors[order]
            row_vals = row_vals[order]
        counts[i] = int(row_vals.size)
        width = max(width, int(row_vals.size))
        rows.append((row_actions, row_vals, row_bases, row_sensors))
    if max_actions is not None:
        width = min(width, int(max_actions))
    return counts, rows, width


def sort_vectorized_scores(score, physical, max_actions=None):
    n = int(score.shape[0])
    row_ids = np.arange(n)[:, None]
    candidate_scores = score[row_ids, physical.bases, physical.sensors]
    candidate_scores = np.where(physical.valid & np.isfinite(candidate_scores), candidate_scores, -np.inf).astype(np.float32)
    finite = np.isfinite(candidate_scores)
    counts = finite.sum(axis=1).astype(np.int32)
    width = int(counts.max(initial=0))
    if max_actions is not None:
        width = min(width, int(max_actions))
    rows = []
    for i in range(n):
        row_finite = finite[i]
        row_scores = candidate_scores[i, row_finite]
        if row_scores.size:
            order = np.argsort(-row_scores)
            rows.append(
                (
                    physical.actions[i, row_finite][order],
                    row_scores[order],
                    physical.bases[i, row_finite][order],
                    physical.sensors[i, row_finite][order],
                )
            )
        else:
            rows.append(
                (
                    np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=np.int64),
                )
            )
    return counts, rows, width


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--out", type=Path, default=Path("profile_root_table_steps.json"))
    args = parser.parse_args()

    from exact_env_mutual import attach_env_obs
    from final_radar_campaign import get_obs
    from mutual_features import slot_features, slot_features_batch, tokenize, tokenize_batch
    from perf_fast_planner import physical_action_arrays, physical_action_table_batch
    from realistic_reward_retrain import adapter
    from repaired_campaign_tools import EDFPlanner, build_env, env_preset_cfg
    from two_sensor_physical_head_eval import MAXT, ActionAttentionFactorizedNet

    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)
    device = torch.device(args.device)

    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1
    observations = []
    for idx in range(int(args.batch_size)):
        seed = int(args.seed) + idx
        eng = build_env(EDFPlanner(MAXT), int(args.initial_targets), MAXT, seed, 200, env_cfg)
        eng.reset(seed=seed)
        observations.append(get_obs(eng, 0.0))
        eng.close()

    adapt = adapter()
    model = ActionAttentionFactorizedNet(48, 4, 2).eval().to(device)
    buckets: dict[str, list[float]] = defaultdict(list)
    selected = [set() for _ in observations]

    for idx in range(int(args.warmup) + int(args.iters)):
        record = idx >= int(args.warmup)

        def maybe_time(name, fn):
            if record:
                return timed(device, buckets, name, fn)
            return fn()

        obs2 = maybe_time("attach_env_obs", lambda: [attach_env_obs(obs, env_cfg, True, True) for obs in observations])
        legacy_tokens = maybe_time(
            "legacy_tokenize_stack",
            lambda: np.stack(
                [tokenize(adapt, obs, selected=selected[i], search_count=0).astype(np.float32) for i, obs in enumerate(obs2)],
                axis=0,
            ),
        )
        tokens = maybe_time("batched_tokenize", lambda: tokenize_batch(adapt, obs2, selected=selected, search_count=[0] * len(obs2)))
        legacy_slots = maybe_time(
            "legacy_slot_features_stack",
            lambda: np.stack(
                [slot_features(obs, 0.0, 0, 0, -1, 200.0).astype(np.float32) for obs in obs2],
                axis=0,
            ),
        )
        slots = maybe_time(
            "batched_slot_features",
            lambda: slot_features_batch(
                obs2,
                elapsed=[0.0] * len(obs2),
                search_count=[0] * len(obs2),
                track_count=[0] * len(obs2),
                last_action=[-1] * len(obs2),
                budget_ms=200.0,
            ),
        )
        x, s = maybe_time(
            "host_to_device",
            lambda: (
                torch.from_numpy(tokens).to(device, dtype=torch.float32),
                torch.from_numpy(slots).to(device, dtype=torch.float32),
            ),
        )

        def forward():
            with torch.inference_mode():
                scores, q = model.forward_scores(x, s)
                return (scores + q).float().cpu().numpy()

        score = maybe_time("model_forward_and_cpu_transfer", forward)
        action_arrays = maybe_time("legacy_physical_action_arrays", lambda: [physical_action_arrays(obs, selected=selected[i], max_trackers=MAXT) for i, obs in enumerate(obs2)])
        physical = maybe_time("vectorized_physical_action_table", lambda: physical_action_table_batch(obs2, selected=selected, max_trackers=MAXT))
        legacy_counts, legacy_rows, legacy_width = maybe_time("legacy_gather_sort", lambda: sort_legacy_scores(score, action_arrays))
        vector_counts, vector_rows, vector_width = maybe_time("vectorized_gather_sort", lambda: sort_vectorized_scores(score, physical))

    equal_features = bool(np.allclose(legacy_tokens, tokens, atol=1e-6) and np.allclose(legacy_slots, slots, atol=1e-6))
    equal_counts = np.array_equal(legacy_counts, vector_counts)
    equal_width = int(legacy_width) == int(vector_width)
    equal_rows = True
    for legacy_row, vector_row, count in zip(legacy_rows, vector_rows, legacy_counts):
        count = int(count)
        equal_rows = equal_rows and np.array_equal(legacy_row[0][:count], vector_row[0][:count])
        equal_rows = equal_rows and np.allclose(legacy_row[1][:count], vector_row[1][:count], atol=1e-5)

    summary = {name: stats(vals) for name, vals in sorted(buckets.items(), key=lambda kv: sum(kv[1]), reverse=True)}
    total_mean_ms = float(sum(item["mean_ms"] for item in summary.values()))
    for item in summary.values():
        item["mean_percent_of_profiled_steps"] = float(100.0 * item["mean_ms"] / max(total_mean_ms, 1e-12))

    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "batch_size": int(args.batch_size),
        "initial_targets": int(args.initial_targets),
        "rate": float(args.rate),
        "seed": int(args.seed),
        "vectorized_matches_legacy_counts": bool(equal_counts),
        "vectorized_matches_legacy_width": bool(equal_width),
        "vectorized_matches_legacy_rows": bool(equal_rows),
        "batched_features_match_legacy": bool(equal_features),
        "stage_profile": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
