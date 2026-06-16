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
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def bench(planner, obs: dict, device: torch.device, iters: int, warmup: int) -> tuple[list[int], dict[str, float]]:
    times: list[float] = []
    plan: list[int] = []
    for i in range(int(warmup) + int(iters)):
        sync(device)
        t0 = time.perf_counter()
        plan = list(planner.plan(obs, budget_ms=200))
        sync(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if i >= int(warmup):
            times.append(float(dt_ms))
    return [int(x) for x in plan], stats(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--out", type=Path, default=Path("results/perf_lab_cuda_graph_planner.json"))
    args = parser.parse_args()

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

    eng = build_env(EDFPlanner(MAXT), int(args.initial_targets), MAXT, int(args.seed), 200, env_cfg)
    eng.reset(seed=int(args.seed))
    obs = get_obs(eng, 0.0)

    model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    gpu_select_model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    graph_model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    graph_gpu_select_model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    gpu_select_model.load_state_dict(model.state_dict())
    graph_model.load_state_dict(model.state_dict())
    graph_gpu_select_model.load_state_dict(model.state_dict())
    base = FastActionAttentionPlanner(model, env_cfg, device=device, use_amp=bool(args.amp), use_cuda_graph=False)
    gpu_select = FastActionAttentionPlanner(
        gpu_select_model,
        env_cfg,
        device=device,
        use_amp=bool(args.amp),
        use_cuda_graph=False,
        use_gpu_select=True,
    )
    graph = FastActionAttentionPlanner(graph_model, env_cfg, device=device, use_amp=bool(args.amp), use_cuda_graph=True)
    graph_gpu_select = FastActionAttentionPlanner(
        graph_gpu_select_model,
        env_cfg,
        device=device,
        use_amp=bool(args.amp),
        use_cuda_graph=True,
        use_gpu_select=True,
    )

    base_plan, base_timing = bench(base, obs, device, int(args.iters), int(args.warmup))
    gpu_select_plan, gpu_select_timing = bench(gpu_select, obs, device, int(args.iters), int(args.warmup))
    graph_plan, graph_timing = bench(graph, obs, device, int(args.iters), int(args.warmup))
    graph_gpu_select_plan, graph_gpu_select_timing = bench(graph_gpu_select, obs, device, int(args.iters), int(args.warmup))
    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "amp": bool(args.amp),
        "initial_targets": int(args.initial_targets),
        "rate": float(args.rate),
        "seed": int(args.seed),
        "iters": int(args.iters),
        "warmup": int(args.warmup),
        "base_plan": base_plan,
        "gpu_select_plan": gpu_select_plan,
        "cuda_graph_plan": graph_plan,
        "cuda_graph_gpu_select_plan": graph_gpu_select_plan,
        "plans_match": {
            "gpu_select": base_plan == gpu_select_plan,
            "cuda_graph": base_plan == graph_plan,
            "cuda_graph_gpu_select": base_plan == graph_gpu_select_plan,
        },
        "base_fast_planner": base_timing,
        "gpu_select_fast_planner": gpu_select_timing,
        "cuda_graph_fast_planner": graph_timing,
        "cuda_graph_gpu_select_fast_planner": graph_gpu_select_timing,
        "gpu_select_speedup_mean": float(base_timing["mean_ms"] / max(gpu_select_timing["mean_ms"], 1e-12)),
        "speedup_mean": float(base_timing["mean_ms"] / max(graph_timing["mean_ms"], 1e-12)),
        "cuda_graph_gpu_select_speedup_mean": float(base_timing["mean_ms"] / max(graph_gpu_select_timing["mean_ms"], 1e-12)),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    eng.close()


if __name__ == "__main__":
    main()
