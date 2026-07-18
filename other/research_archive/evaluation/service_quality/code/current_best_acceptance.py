from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_TARGETS = "CreateValid1/results/edf_bootstrap_r3_lmh_1024_targets.pt"
DEFAULT_FINETUNE = "CreateValid1/results/selfplay_adv_edf_owntail_factor_r3_lmh_512_targets.pt"
LEARNED_METHOD = "two_row_factorized_conservative"
BASELINES = ("EDF", "EST", "fair_exact")


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def run_seed(args, seed: int) -> Path:
    out = Path(args.out_dir) / f"{args.prefix}_seed{seed}_h800_100w.csv"
    cmd = [
        sys.executable,
        "CreateValid1/experiments/code/model_code/portfolio_sweep_eval.py",
        "--targets",
        str(args.targets),
        "--finetune-targets",
        str(args.finetune_targets),
        "--out",
        str(out),
        "--initials",
        str(args.initials),
        "--rates",
        str(args.rates),
        "--eval-seeds",
        str(seed),
        "--eval-windows",
        str(args.eval_windows),
        "--windows",
        str(args.windows),
        "--train-steps",
        str(args.train_steps),
        "--finetune-steps",
        str(args.finetune_steps),
        "--batch-size",
        str(args.batch_size),
        "--model-seed",
        str(args.model_seed),
        "--cell-balanced-sampling",
        "--variants",
        "two_row_factorized",
        "--configs",
        "conservative|-14,-12,-10,-8,-5|0,0.5",
        "--top-k",
        str(args.top_k),
        "--score-horizon-ms",
        str(args.score_horizon_ms),
        "--force-learned-rescore",
        "--preserve-base-topk",
        "--learned-extra-top-k",
        str(args.learned_extra_top_k),
        "--resume",
    ]
    print({"running_seed": int(seed), "out": str(out)}, flush=True)
    subprocess.run(cmd, check=True)
    return out


def aggregate(paths: list[Path], out_prefix: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.concat([pd.read_csv(path).assign(source=path.name) for path in paths], ignore_index=True)
    raw["cum_reward_100w"] = raw["reward"] * raw["windows_completed"]
    wide = raw.pivot_table(index=["seed", "initial", "rate"], columns="method", values="reward", aggfunc="first").reset_index()
    missing = [name for name in (LEARNED_METHOD, *BASELINES) if name not in wide.columns]
    if missing:
        raise RuntimeError(f"missing required methods in aggregate: {missing}")
    for baseline in BASELINES:
        wide[f"beats_{baseline}"] = wide[LEARNED_METHOD] > wide[baseline]
    wide["beats_all"] = wide[[f"beats_{baseline}" for baseline in BASELINES]].all(axis=1)
    summary = raw.groupby("method", as_index=False).agg(
        mean_reward=("reward", "mean"),
        mean_cum=("cum_reward_100w", "mean"),
        mean_search=("search", "mean"),
        mean_latency_ms=("latency_ms", "mean"),
        n=("reward", "size"),
    ).sort_values("mean_reward", ascending=False)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_prefix.with_suffix(".csv"), index=False)
    summary.to_csv(out_prefix.with_name(out_prefix.name + "_summary.csv"), index=False)
    by_load = wide.groupby(["initial", "rate"], as_index=False).agg(
        n=("seed", "size"),
        learned=(LEARNED_METHOD, "mean"),
        fair_exact=("fair_exact", "mean"),
        EDF=("EDF", "mean"),
        EST=("EST", "mean"),
        beats_all=("beats_all", "sum"),
    )
    by_load.to_csv(out_prefix.with_name(out_prefix.name + "_by_load.csv"), index=False)
    return wide, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--finetune-targets", default=DEFAULT_FINETUNE)
    ap.add_argument("--out-dir", default="CreateValid1/results/current_best_acceptance")
    ap.add_argument("--prefix", default="factorized_physical_acceptance")
    ap.add_argument("--aggregate-out", default="CreateValid1/results/current_best_acceptance/factorized_physical_acceptance_aggregate")
    ap.add_argument("--initials", default="20,40,60")
    ap.add_argument("--rates", default="2,3,4")
    ap.add_argument("--eval-seeds", default="916,918,920")
    ap.add_argument("--eval-windows", type=int, default=100)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=120)
    ap.add_argument("--finetune-steps", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--model-seed", type=int, default=123)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--score-horizon-ms", type=float, default=800.0)
    ap.add_argument("--learned-extra-top-k", type=int, default=4)
    ap.add_argument("--skip-run", action="store_true", help="Only aggregate existing per-seed CSV files.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_ints(args.eval_seeds)
    paths = [out_dir / f"{args.prefix}_seed{seed}_h800_100w.csv" for seed in seeds]
    if not bool(args.skip_run):
        paths = [run_seed(args, seed) for seed in seeds]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing acceptance result files: {missing}")

    wide, summary = aggregate(paths, Path(args.aggregate_out))
    wins = {f"beats_{baseline}": int(wide[f"beats_{baseline}"].sum()) for baseline in BASELINES}
    wins["beats_all"] = int(wide["beats_all"].sum())
    n_cells = int(len(wide))
    print(summary.to_string(index=False), flush=True)
    print({"n_cells": n_cells, **wins}, flush=True)
    if wins["beats_all"] != n_cells:
        failures = wide.loc[~wide["beats_all"], ["seed", "initial", "rate", LEARNED_METHOD, *BASELINES]]
        print(failures.to_string(index=False), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
