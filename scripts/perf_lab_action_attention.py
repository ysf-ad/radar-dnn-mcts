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


def quantiles(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def bench_forward_scores(model, device, rows, token_dim, slot_dim, use_amp, batch_sizes, iters, warmup):
    out = {}
    model = model.eval().to(device)
    for batch in batch_sizes:
        torch.manual_seed(1000 + int(batch))
        tokens = torch.randn(int(batch), rows, token_dim, device=device)
        slots = torch.randn(int(batch), slot_dim, device=device)
        tokens[:, :, 4] = 1.0
        tokens[:, :, 8] = 0.0
        times = []
        with torch.inference_mode():
            for i in range(int(warmup) + int(iters)):
                sync(device)
                t0 = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(use_amp and device.type == "cuda")):
                    scores, q = model.forward_scores(tokens, slots)
                _ = scores.shape, q.shape
                sync(device)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if i >= int(warmup):
                    times.append(dt_ms)
        stats = quantiles(times)
        actions = int(batch) * rows * 2
        stats["actions_per_second"] = float(actions / max(stats["mean_ms"], 1e-12) * 1000.0)
        stats["batch"] = int(batch)
        out[str(batch)] = stats
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--out", type=Path, default=Path("perf_lab_action_attention.json"))
    parser.add_argument("--forward-batches", default="1,8,32,128")
    args = parser.parse_args()

    from final_radar_campaign import get_obs
    from mutual_features import SLOT_DIM, TOKEN_DIM
    from perf_fast_planner import FastActionAttentionPlanner
    from repaired_campaign_tools import EDFPlanner, build_env, env_preset_cfg
    from two_sensor_physical_head_eval import MAXT, ActionAttentionFactorizedNet, PhysicalHeadPlanner

    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    model = ActionAttentionFactorizedNet(args.d_model, args.nhead, args.nlayers).eval()
    fast_model = ActionAttentionFactorizedNet(args.d_model, args.nhead, args.nlayers).eval()
    fast_model.load_state_dict(model.state_dict())

    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1
    eng = build_env(EDFPlanner(MAXT), args.initial_targets, MAXT, args.seed, 200, env_cfg)
    eng.reset(seed=args.seed)
    obs = get_obs(eng, 0.0)

    baseline = PhysicalHeadPlanner(
        model,
        "two_row_action_attention_qpolicy_factored_loss",
        env_cfg,
        policy_weight=1.0,
        q_weight=1.0,
    )
    fast = FastActionAttentionPlanner(
        fast_model,
        env_cfg,
        policy_weight=1.0,
        q_weight=1.0,
        device=device,
        use_amp=bool(args.amp),
        use_compile=bool(args.compile),
    )

    # Verify that the optimized planner is behavior-compatible for the same model/obs.
    base_plan = baseline.plan(obs, budget_ms=200)
    fast_plan = fast.plan(obs, budget_ms=200)

    def bench_plan(planner):
        times = []
        for i in range(int(args.warmup) + int(args.iters)):
            sync(device)
            t0 = time.perf_counter()
            _ = planner.plan(obs, budget_ms=200)
            sync(device)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if i >= int(args.warmup):
                times.append(dt_ms)
        return times

    baseline_times = bench_plan(baseline)
    fast_times = bench_plan(fast)

    speedup = float(np.mean(baseline_times) / max(np.mean(fast_times), 1e-12))
    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "amp": bool(args.amp),
        "compile": bool(args.compile),
        "iters": int(args.iters),
        "warmup": int(args.warmup),
        "base_plan": [int(x) for x in base_plan],
        "fast_plan": [int(x) for x in fast_plan],
        "plans_match": [int(x) for x in base_plan] == [int(x) for x in fast_plan],
        "baseline_physical_head_planner": quantiles(baseline_times),
        "fast_cached_action_attention_planner": quantiles(fast_times),
        "speedup_mean": speedup,
        "fast_stats": fast.stats.__dict__,
    }
    throughput_model = ActionAttentionFactorizedNet(args.d_model, args.nhead, args.nlayers).eval()
    throughput_model.load_state_dict(model.state_dict())
    report["forward_scores_throughput"] = bench_forward_scores(
        throughput_model,
        device,
        rows=MAXT + 1,
        token_dim=TOKEN_DIM,
        slot_dim=SLOT_DIM,
        use_amp=bool(args.amp),
        batch_sizes=[int(x) for x in str(args.forward_batches).split(",") if x.strip()],
        iters=max(10, int(args.iters) // 2),
        warmup=max(3, int(args.warmup) // 2),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    eng.close()


if __name__ == "__main__":
    main()
