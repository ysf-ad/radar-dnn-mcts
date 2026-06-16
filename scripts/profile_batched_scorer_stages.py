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


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "p99_ms": 0.0}
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def timed(device: torch.device, buckets: dict[str, list[float]], name: str, fn):
    sync(device)
    t0 = time.perf_counter()
    out = fn()
    sync(device)
    buckets[name].append((time.perf_counter() - t0) * 1000.0)
    return out


def profile_batch(batcher, observations: list[dict], iters: int, warmup: int, budget_ms: float) -> dict:
    from exact_env_mutual import attach_env_obs
    from mutual_features import slot_features_batch, tokenize_batch
    from perf_fast_planner import physical_action_table_batch
    from two_sensor_physical_head_eval import MAXT

    device = batcher.device
    n = len(observations)
    buckets: dict[str, list[float]] = {
        "attach_env_obs": [],
        "tokenize_batch": [],
        "slot_features_batch": [],
        "tokens_h2d": [],
        "slots_h2d": [],
        "model_forward": [],
        "physical_action_table_batch": [],
        "action_tables_h2d": [],
        "gpu_gather_argmax": [],
        "best_actions_d2h": [],
        "total": [],
    }
    last_actions = None

    selected = [set() for _ in range(n)]
    elapsed = [0.0] * n
    search_count = [0] * n
    track_count = [0] * n
    last = [-1] * n

    for i in range(int(warmup) + int(iters)):
        record = i >= int(warmup)
        run_buckets = buckets if record else {key: [] for key in buckets}
        sync(device)
        t_total = time.perf_counter()

        obs2 = timed(device, run_buckets, "attach_env_obs", lambda: [attach_env_obs(obs, batcher.env_cfg, True, True) for obs in observations])
        tokens = timed(device, run_buckets, "tokenize_batch", lambda: tokenize_batch(batcher.adapt, obs2, selected=selected, search_count=search_count))
        slots = timed(
            device,
            run_buckets,
            "slot_features_batch",
            lambda: slot_features_batch(
                obs2,
                elapsed=elapsed,
                search_count=search_count,
                track_count=track_count,
                last_action=last,
                budget_ms=float(budget_ms),
            ),
        )

        with torch.inference_mode():
            x = timed(device, run_buckets, "tokens_h2d", lambda: torch.from_numpy(tokens).to(device, dtype=torch.float32))
            s = timed(device, run_buckets, "slots_h2d", lambda: torch.from_numpy(slots).to(device, dtype=torch.float32))

            def forward():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=batcher.use_amp):
                    score = batcher._combined_scores_from_tokens(x, s).float()
                score[:, 0, :] += batcher.search_score_bias
                return score

            score_t = timed(device, run_buckets, "model_forward", forward)

        physical = timed(device, run_buckets, "physical_action_table_batch", lambda: physical_action_table_batch(obs2, selected=selected, max_trackers=MAXT))

        def tables_h2d():
            actions_t = torch.as_tensor(physical.actions, device=device, dtype=torch.long)
            flat_t = torch.as_tensor(physical.bases * 2 + physical.sensors, device=device, dtype=torch.long)
            valid_t = torch.as_tensor(physical.valid, device=device, dtype=torch.bool)
            return actions_t, flat_t, valid_t

        actions_t, flat_t, valid_t = timed(device, run_buckets, "action_tables_h2d", tables_h2d)

        def select_gpu():
            flat_scores = score_t.reshape(n, -1)
            candidate_scores = torch.gather(flat_scores, 1, flat_t)
            candidate_scores = candidate_scores.masked_fill(~(valid_t & torch.isfinite(candidate_scores)), -torch.inf)
            idx = torch.argmax(candidate_scores, dim=1)
            best = torch.gather(actions_t, 1, idx[:, None]).squeeze(1)
            has_valid = torch.any(torch.isfinite(candidate_scores), dim=1)
            return torch.where(has_valid, best, torch.full_like(best, -1))

        best_t = timed(device, run_buckets, "gpu_gather_argmax", select_gpu)
        last_actions = timed(device, run_buckets, "best_actions_d2h", lambda: best_t.cpu().numpy().astype(np.int64, copy=False))

        sync(device)
        if record:
            buckets["total"].append((time.perf_counter() - t_total) * 1000.0)

    return {
        "batch": int(n),
        "selected_actions_head": [int(x) for x in np.asarray(last_actions[: min(8, len(last_actions))], dtype=np.int64)],
        "stage_timing": {key: stats(values) for key, values in buckets.items()},
    }


def profile_prepared_batch(batcher, observations: list[dict], iters: int, warmup: int, budget_ms: float) -> dict:
    device = batcher.device
    buckets: dict[str, list[float]] = {"prepare_root_batch": [], "score_prepared": [], "total": []}
    prepared = timed(device, buckets, "prepare_root_batch", lambda: batcher.prepare_root_batch(observations, budget_ms=float(budget_ms)))
    last_actions = None
    for i in range(int(warmup) + int(iters)):
        record = i >= int(warmup)
        run_buckets = buckets if record else {key: [] for key in buckets}
        sync(device)
        t_total = time.perf_counter()
        last_actions = timed(device, run_buckets, "score_prepared", lambda: batcher.best_actions_prepared_torch(prepared))
        sync(device)
        if record:
            buckets["total"].append((time.perf_counter() - t_total) * 1000.0)
    return {
        "prepare_once": stats(buckets["prepare_root_batch"]),
        "score_prepared": stats(buckets["score_prepared"]),
        "total_prepared_score_call": stats(buckets["total"]),
        "selected_actions_head": [int(x) for x in np.asarray(last_actions[: min(8, len(last_actions))], dtype=np.int64)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-sizes", default="1,8,32,128")
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/profile_batched_scorer_stages.json"))
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
        eng = build_env(EDFPlanner(MAXT), int(args.initial_targets), MAXT, seed, 200, env_cfg)
        eng.reset(seed=seed)
        observations.append(get_obs(eng, 0.0))
        eng.close()

    model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    batcher = BatchedActionAttentionScorer(model, env_cfg, device=device, use_amp=bool(args.amp))
    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "amp": bool(args.amp),
        "initial_targets": int(args.initial_targets),
        "rate": float(args.rate),
        "seed": int(args.seed),
        "batches": [],
    }
    for batch_size in [int(x) for x in str(args.batch_sizes).split(",") if x.strip()]:
        obs_batch = observations[:batch_size]
        result = profile_batch(batcher, obs_batch, int(args.iters), int(args.warmup), 200.0)
        result["prepared_path"] = profile_prepared_batch(batcher, obs_batch, int(args.iters), int(args.warmup), 200.0)
        report["batches"].append(result)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
