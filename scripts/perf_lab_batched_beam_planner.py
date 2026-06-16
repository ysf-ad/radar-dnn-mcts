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


def bench_plan(planner, obs, device, iters: int, warmup: int, profile: bool = False):
    times = []
    last_plan = None
    if profile and hasattr(planner, "reset_profile"):
        planner.reset_profile()
    if hasattr(planner, "set_profile_enabled"):
        planner.set_profile_enabled(False)
    for i in range(int(warmup) + int(iters)):
        if hasattr(planner, "set_profile_enabled"):
            planner.set_profile_enabled(bool(profile and i >= int(warmup)))
        sync(device)
        t0 = time.perf_counter()
        last_plan = planner.plan(obs, budget_ms=200)
        sync(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if i >= int(warmup):
            times.append(dt_ms)
    if hasattr(planner, "set_profile_enabled"):
        planner.set_profile_enabled(False)
    profile_summary = planner.profile_summary() if profile and hasattr(planner, "profile_summary") else {}
    return last_plan, stats(times), profile_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--beam-widths", default="1,4,8,16")
    parser.add_argument("--branch-top-k", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--out", type=Path, default=Path("perf_lab_batched_beam_planner.json"))
    args = parser.parse_args()

    from batched_window_expansion import BatchedBeamWindowPlanner
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
    fast = FastActionAttentionPlanner(
        model,
        env_cfg,
        device=device,
        use_amp=bool(args.amp),
        use_compile=bool(args.compile),
        use_cuda_graph=bool(args.cuda_graph),
    )

    fast_plan, fast_stat, fast_profile = bench_plan(fast, obs, device, args.iters, args.warmup, profile=bool(args.profile))
    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "amp": bool(args.amp),
        "compile": bool(args.compile),
        "cuda_graph": bool(args.cuda_graph),
        "branch_top_k": int(args.branch_top_k),
        "max_depth": int(args.max_depth),
        "fast_plan": [int(x) for x in fast_plan],
        "fast_cached_planner": fast_stat,
        "fast_profile": fast_profile,
        "beam_planners": [],
    }

    for beam_width in [int(x) for x in str(args.beam_widths).split(",") if x.strip()]:
        branch_top_k = 1 if int(beam_width) == 1 else int(args.branch_top_k)
        for use_top1_device in ([False, True] if int(branch_top_k) == 1 else [False]):
            beam = BatchedBeamWindowPlanner(
                fast,
                beam_width=int(beam_width),
                branch_top_k=branch_top_k,
                max_depth=int(args.max_depth),
                use_top1_device=bool(use_top1_device),
            )
            beam_plan, beam_stat, beam_profile = bench_plan(
                beam,
                obs,
                device,
                args.iters,
                args.warmup,
                profile=bool(args.profile),
            )
            report["beam_planners"].append(
                {
                    "beam_width": int(beam_width),
                    "branch_top_k": int(branch_top_k),
                    "use_top1_device": bool(use_top1_device),
                    "plan": [int(x) for x in beam_plan],
                    "plan_len": int(len(beam_plan)),
                    "matches_fast_plan": [int(x) for x in beam_plan] == [int(x) for x in fast_plan],
                    "matches_fast_prefix": [int(x) for x in beam_plan] == [int(x) for x in fast_plan[: len(beam_plan)]],
                    "timing": beam_stat,
                    "profile": beam_profile,
                    "relative_to_fast_mean": float(beam_stat["mean_ms"] / max(fast_stat["mean_ms"], 1e-12)),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    eng.close()


if __name__ == "__main__":
    main()
