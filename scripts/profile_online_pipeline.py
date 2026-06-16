from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "radar_dnn_mcts"))


class StageTimer:
    def __init__(self):
        self.values = defaultdict(list)

    def time(self, name: str):
        outer = self

        class Ctx:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, exc_type, exc, tb):
                outer.values[name].append((time.perf_counter() - self.t0) * 1000.0)

        return Ctx()

    def add(self, name: str, value_ms: float) -> None:
        self.values[name].append(float(value_ms))

    def summary(self) -> dict:
        out = {}
        for name, values in self.values.items():
            arr = np.asarray(values, dtype=np.float64)
            out[name] = {
                "calls": int(arr.size),
                "total_ms": float(arr.sum()),
                "mean_ms": float(arr.mean()),
                "p50_ms": float(np.percentile(arr, 50)),
                "p90_ms": float(np.percentile(arr, 90)),
                "p99_ms": float(np.percentile(arr, 99)),
            }
        return dict(sorted(out.items(), key=lambda kv: kv[1]["total_ms"], reverse=True))


class ProfiledPlanner:
    def __init__(self, planner, timer: StageTimer, name: str):
        self.planner = planner
        self.timer = timer
        self.name = str(name)

    def warmup(self, obs, budget_ms=200):
        if hasattr(self.planner, "warmup"):
            with self.timer.time(f"{self.name}.warmup"):
                return self.planner.warmup(obs, budget_ms=budget_ms)
        return None

    def plan(self, obs, budget_ms=200):
        with self.timer.time(f"{self.name}.planner_plan"):
            return self.planner.plan(obs, budget_ms=budget_ms)


def _pstats_top(profile: cProfile.Profile, limit: int) -> list[dict]:
    stats = pstats.Stats(profile)
    rows = []
    for func, stat in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[: int(limit)]:
        cc, nc, tt, ct, _callers = stat
        filename, line, name = func
        rows.append(
            {
                "function": f"{Path(filename).name}:{line}:{name}",
                "primitive_calls": int(cc),
                "total_calls": int(nc),
                "self_seconds": float(tt),
                "cumulative_seconds": float(ct),
            }
        )
    return rows


def run_profiled_episode(planner, planner_name: str, args, env_cfg: dict) -> dict:
    from final_radar_campaign import get_obs
    from repaired_campaign_tools import build_env
    from strict_window_report import execute_plan_until_budget, sample_state_metrics
    from two_sensor_physical_head_eval import MAXT

    timer = StageTimer()
    profiled = ProfiledPlanner(planner, timer, planner_name)
    with timer.time("env_build_reset"):
        eng = build_env(profiled, int(args.initial_targets), MAXT, int(args.seed), int(args.window_ms), env_cfg)
        eng.reset(seed=int(args.seed))

    cprof = cProfile.Profile()
    cumulative_reward = 0.0
    search_debt_ms = 0.0
    windows = 0
    executed_total = 0
    spent_total = 0.0

    try:
        if hasattr(profiled, "warmup"):
            with timer.time("get_obs_warmup"):
                warm_obs = get_obs(eng, search_debt_ms)
            profiled.warmup(warm_obs, budget_ms=int(args.window_ms))

        cprof.enable()
        for window_idx in range(int(args.windows)):
            if eng.term_buf[0]:
                break
            with timer.time("get_obs_before_plan"):
                obs = get_obs(eng, search_debt_ms)
            plan = profiled.plan(obs, budget_ms=int(args.window_ms))
            with timer.time("execute_plan_until_budget"):
                reward, spent_ms, search_debt_ms, executed, _search_actions, _arows = execute_plan_until_budget(
                    eng,
                    plan,
                    float(args.window_ms),
                    search_debt_ms,
                    planner_name,
                    int(args.seed),
                    int(window_idx),
                )
            with timer.time("sample_state_metrics"):
                _ = sample_state_metrics(eng, search_debt_ms)
            cumulative_reward += float(reward)
            spent_total += float(spent_ms)
            executed_total += int(executed)
            windows += 1
        cprof.disable()
    finally:
        with timer.time("env_close"):
            eng.close()

    stage = timer.summary()
    return {
        "planner": planner_name,
        "windows": int(windows),
        "executed_actions": int(executed_total),
        "mean_spent_ms_per_window": float(spent_total / max(1, windows)),
        "total_reward": float(cumulative_reward),
        "stage_timing": stage,
        "cprofile_top_cumulative": _pstats_top(cprof, int(args.profile_top)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--window-ms", type=int, default=200)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--planners", default="edf,physical,fast")
    parser.add_argument("--profile-top", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("profile_online_pipeline.json"))
    args = parser.parse_args()

    from perf_fast_planner import FastActionAttentionPlanner
    from repaired_campaign_tools import EDFPlanner, ESTPlanner, env_preset_cfg
    from two_sensor_physical_head_eval import MAXT, ActionAttentionFactorizedNet, PhysicalHeadPlanner

    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)
    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1

    device = torch.device(args.device)
    base_model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    fast_model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    fast_model.load_state_dict(base_model.state_dict())

    planner_factories = {
        "edf": lambda: EDFPlanner(MAXT),
        "est": lambda: ESTPlanner(MAXT),
        "physical": lambda: PhysicalHeadPlanner(
            base_model,
            "two_row_action_attention_qpolicy_factored_loss",
            env_cfg,
        ),
        "fast": lambda: FastActionAttentionPlanner(fast_model, env_cfg, device=device),
    }
    requested = [name.strip().lower() for name in str(args.planners).split(",") if name.strip()]
    unknown = sorted(set(requested) - set(planner_factories))
    if unknown:
        raise ValueError(f"Unknown planners: {', '.join(unknown)}")

    report = {
        "device": str(device),
        "initial_targets": int(args.initial_targets),
        "rate": float(args.rate),
        "seed": int(args.seed),
        "windows": int(args.windows),
        "planners": [],
    }
    for name in requested:
        planner = planner_factories[name]()
        report["planners"].append(run_profiled_episode(planner, name, args, env_cfg))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
