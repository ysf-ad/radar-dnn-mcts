from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from pufferlib.ocean.radarxs.models.edf import EDFPlanner
from pufferlib.ocean.radarxs.models.est import ESTPlanner
from radar_dnn_mcts.env.config import BenchmarkConfig
from radar_dnn_mcts.evaluation.benchmark import BenchmarkRunner
from radar_dnn_mcts.evaluation.metrics import aggregate_methods
from radar_dnn_mcts.evaluation.plots import plot_suite
from radar_dnn_mcts.models.checkpoint import load_checkpoint
from radar_dnn_mcts.models.decoders import AutoregressiveDecoder, BatchDecoder
from radar_dnn_mcts.models.dynamics import LatentDynamics
from radar_dnn_mcts.models.scheduler import RadarSchedulerModel
from radar_dnn_mcts.schedulers import (
    AutoregressiveScheduler,
    BatchScheduler,
    FullReencodeScheduler,
    FullWindowPUCTScheduler,
    MuZeroScheduler,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/eval_9cell.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--methods", default="edf,est,reencode,muzero,ar,batch")
    parser.add_argument("--out", type=Path, default=Path("results/canonical"))
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    core = RadarSchedulerModel().to(device)
    dynamics = LatentDynamics(max_rows=config.max_targets + 1).to(device)
    ar = AutoregressiveDecoder(core, max_rows=config.max_targets + 1).to(device)
    batch = BatchDecoder().to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, {"core": core, "dynamics": dynamics, "ar": ar, "batch": batch}, device)

    available = {
        "edf": EDFPlanner(config.max_targets),
        "est": ESTPlanner(config.max_targets),
        "reencode": FullReencodeScheduler(core),
        "puct": FullWindowPUCTScheduler(core),
        "muzero": MuZeroScheduler(core, dynamics),
        "ar": AutoregressiveScheduler(ar),
        "batch": BatchScheduler(batch),
    }
    requested = [name.strip() for name in args.methods.split(",") if name.strip()]
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    args.out.mkdir(parents=True, exist_ok=True)
    windows, summary = BenchmarkRunner(config).run({name: available[name] for name in requested})
    aggregate = aggregate_methods(windows)
    windows.to_csv(args.out / "windows.csv", index=False)
    summary.to_csv(args.out / "summary_by_cell.csv", index=False)
    aggregate.to_csv(args.out / "summary.csv", index=False)
    plot_suite(windows, args.out / "plot_suite.png")
    pd.set_option("display.max_columns", None)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
