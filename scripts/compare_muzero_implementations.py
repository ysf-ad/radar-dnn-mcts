"""Compare production and reference MuZero searches on the same radar cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from radar_dnn_mcts.env.config import BenchmarkConfig
from radar_dnn_mcts.env.features import FeatureBuilder
from radar_dnn_mcts.evaluation.benchmark import BenchmarkRunner
from radar_dnn_mcts.models.checkpoint import load_checkpoint
from radar_dnn_mcts.models.dynamics import LatentDynamics
from radar_dnn_mcts.models.scheduler import RadarSchedulerModel
from radar_dnn_mcts.schedulers import MuZeroPUCTScheduler
from reference_implementations.muzero_general_stepwise import (
    MuZeroGeneralStepwiseScheduler,
)
from reference_implementations.muzero_general_windowed import (
    MuZeroGeneralWindowedScheduler,
)


def load_config(path: Path) -> BenchmarkConfig:
    data = yaml.safe_load(path.read_text())
    return BenchmarkConfig(
        initial_targets=tuple(data["initial_targets"]),
        arrival_rates=tuple(data["arrival_rates"]),
        seeds=tuple(data["seeds"]),
        windows=int(data["windows"]),
        window_ms=float(data["window_ms"]),
        max_targets=int(data["max_targets"]),
        sensors=tuple(data["sensors"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B production window PUCT, MuZero-General stepwise, and its windowed port."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/eval_9cell.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--methods", default="ours,stepwise,windowed")
    parser.add_argument("--ours-rollouts", type=int, default=8)
    parser.add_argument("--stepwise-simulations", type=int, default=8)
    parser.add_argument("--windowed-rollouts", type=int, default=8)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--pb-c-base", type=float, default=19652.0)
    parser.add_argument("--pb-c-init", type=float, default=1.25)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-windows", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/muzero_comparison"))
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    features = FeatureBuilder(
        max_targets=config.max_targets,
        window_ms=config.window_ms,
    )
    model = RadarSchedulerModel().to(device)
    dynamics = LatentDynamics(max_rows=config.max_targets + 1).to(device)
    if args.checkpoint:
        load_checkpoint(
            args.checkpoint,
            {"core": model, "dynamics": dynamics},
            device,
        )

    planners = {
        "ours": MuZeroPUCTScheduler(
            model,
            dynamics,
            features,
            simulations=args.ours_rollouts,
            c_puct=args.c_puct,
            discount=args.discount,
            max_steps=args.max_steps,
            random_seed=args.seed,
        ),
        "stepwise": MuZeroGeneralStepwiseScheduler(
            model,
            dynamics,
            features,
            simulations=args.stepwise_simulations,
            discount=args.discount,
            pb_c_base=args.pb_c_base,
            pb_c_init=args.pb_c_init,
            max_steps=args.max_steps,
            random_seed=args.seed,
        ),
        "windowed": MuZeroGeneralWindowedScheduler(
            model,
            dynamics,
            features,
            simulations=args.windowed_rollouts,
            discount=args.discount,
            pb_c_base=args.pb_c_base,
            pb_c_init=args.pb_c_init,
            max_steps=args.max_steps,
            random_seed=args.seed,
        ),
    }
    requested = [name.strip() for name in args.methods.split(",") if name.strip()]
    unknown = set(requested) - set(planners)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")

    windows_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    for name in requested:
        planner = planners[name]
        windows, summary = BenchmarkRunner(config).run(
            {name: planner},
            warmup_windows=args.warmup_windows,
        )
        calls = np.asarray(planner.g_call_history, dtype=np.float64)
        summary["g_calls_mean"] = float(calls.mean()) if calls.size else 0.0
        summary["g_calls_p90"] = (
            float(np.quantile(calls, 0.9)) if calls.size else 0.0
        )
        summary["observations_per_window"] = planner.last_observations
        windows_parts.append(windows)
        summary_parts.append(summary)

    all_windows = pd.concat(windows_parts, ignore_index=True)
    all_summary = pd.concat(summary_parts, ignore_index=True)
    aggregate = all_summary.groupby("method", as_index=False).agg(
        reward_per_window=("reward_per_window", "mean"),
        dropped_targets=("dropped_targets", "mean"),
        drop_pct_active=("drop_pct_active", "mean"),
        latency_ms_mean=("latency_ms_mean", "mean"),
        g_calls_mean=("g_calls_mean", "mean"),
        g_calls_p90=("g_calls_p90", "mean"),
        observations_per_window=("observations_per_window", "mean"),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    all_windows.to_csv(args.out / "windows.csv", index=False)
    all_summary.to_csv(args.out / "summary_by_cell.csv", index=False)
    aggregate.to_csv(args.out / "summary.csv", index=False)
    manifest = {
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "config": str(args.config),
        "methods": requested,
        "ours_rollouts": args.ours_rollouts,
        "stepwise_simulations_per_action": args.stepwise_simulations,
        "windowed_rollouts": args.windowed_rollouts,
        "c_puct": args.c_puct,
        "pb_c_base": args.pb_c_base,
        "pb_c_init": args.pb_c_init,
        "discount": args.discount,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "comparison_note": "Search counts have different semantics; use measured g_calls for compute matching.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.set_option("display.max_columns", None)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
