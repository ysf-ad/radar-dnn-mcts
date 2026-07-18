from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = Path(r"C:\Users\yousi\Downloads\radar_outputs\exact_train_qw02_seeded_more\exact_mutual_latest.pt")
DEFAULT_OUT = Path(r"C:\Users\yousi\Downloads\Model1 1\CreateValid1\results\pq1_alphazero")


def run(cmd: list[str], dry_run: bool = False) -> None:
    print(" ".join(str(x) for x in cmd), flush=True)
    if not dry_run:
        subprocess.run([str(x) for x in cmd], cwd=str(ROOT), check=True)


def csv_mean(path: Path, method_col: str, reward_col: str = "reward") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or method_col not in df.columns:
        return pd.DataFrame()
    group_cols = [method_col]
    out = (
        df.groupby(group_cols, as_index=False)
        .agg(
            reward=(reward_col, "mean"),
            search=("search", "mean") if "search" in df.columns else (reward_col, "size"),
            latency_ms=("latency_ms", "mean") if "latency_ms" in df.columns else (reward_col, "size"),
            cells=(reward_col, "size"),
        )
        .rename(columns={method_col: "method"})
    )
    return out


def write_summary(out_dir: Path, exact_eval: Path, direct_eval: Path, mcts_eval: Path) -> None:
    rows = []
    exact = csv_mean(exact_eval, "method")
    if not exact.empty:
        exact["source"] = "foundation_eval"
        rows.append(exact)
    if direct_eval.exists():
        d = pd.read_csv(direct_eval)
        if not d.empty and "planner" in d.columns:
            direct = (
                d.groupby("planner", as_index=False)
                .agg(
                    reward=("reward", "mean"),
                    total_reward=("total_reward", "mean"),
                    search=("search", "mean"),
                    latency_ms=("latency", "mean"),
                    cells=("reward", "size"),
                )
                .rename(columns={"planner": "method"})
            )
            direct["source"] = "direct_eval"
            rows.append(direct)
    if mcts_eval.exists():
        m = csv_mean(mcts_eval, "method")
        if not m.empty:
            m["source"] = "benchmark_mcts"
            rows.append(m)
    if not rows:
        return
    summary = pd.concat(rows, ignore_index=True, sort=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    exact_df = pd.read_csv(exact_eval) if exact_eval.exists() else pd.DataFrame()
    if not exact_df.empty and {"method", "initial", "rate", "seed", "reward"}.issubset(exact_df.columns):
        keys = ["initial", "rate", "seed"]
        best_heur = (
            exact_df[exact_df["method"].isin(["EDF", "EST"])]
            .groupby(keys, as_index=False)["reward"]
            .max()
            .rename(columns={"reward": "best_heuristic"})
        )
        cell_rows = []
        for method in sorted(set(exact_df["method"]) - {"EDF", "EST"}):
            joined = exact_df[exact_df["method"] == method][[*keys, "reward"]].merge(best_heur, on=keys, how="inner")
            if joined.empty:
                continue
            margin = joined["reward"] - joined["best_heuristic"]
            cell_rows.append(
                {
                    "method": method,
                    "cells": int(len(joined)),
                    "wins_vs_best_heuristic": int((margin > 0.0).sum()),
                    "win_rate_vs_best_heuristic": float((margin > 0.0).mean()),
                    "mean_margin_vs_best_heuristic": float(margin.mean()),
                    "min_margin_vs_best_heuristic": float(margin.min()),
                }
            )
        pd.DataFrame(cell_rows).to_csv(out_dir / "cell_wins.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--initials", default="60")
    ap.add_argument("--rates", default="1,3,6")
    ap.add_argument("--seeds", default="916")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--teacher-targets", type=int, default=384)
    ap.add_argument("--teacher-targets-per-cell", type=int, default=128)
    ap.add_argument("--teacher-horizon-ms", type=float, default=1200.0)
    ap.add_argument("--teacher-top-k", type=int, default=12)
    ap.add_argument("--teacher-train-steps", type=int, default=300)
    ap.add_argument("--selfplay-episodes", type=int, default=3)
    ap.add_argument("--selfplay-rollouts", type=int, default=16)
    ap.add_argument("--selfplay-train-steps", type=int, default=200)
    ap.add_argument("--eval-variants", default="fair_exact_k8_h800,ml_pq_r8_k16,ml_pq_r16_k24")
    ap.add_argument("--direct-mode", choices=["prob", "branch", "flat", "q"], default="branch")
    ap.add_argument("--direct-threshold", type=float, default=0.65)
    ap.add_argument("--direct-alpha", type=float, default=0.0)
    ap.add_argument("--direct-beta", type=float, default=0.0)
    ap.add_argument("--gate-mcts-targets", action="store_true")
    ap.add_argument("--mcts-gate-baseline", choices=["max_heuristic", "edf", "est", "accepted", "max_all"], default="max_heuristic")
    ap.add_argument("--mcts-gate-margin", type=float, default=0.0)
    ap.add_argument("--skip-teacher-targets", action="store_true")
    ap.add_argument("--skip-teacher-train", action="store_true")
    ap.add_argument("--skip-selfplay", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_targets = out_dir / "pq1_physical_targets.pt"
    teacher_state = out_dir / "pq1_teacher_pq_state.pt"
    selfplay_targets = out_dir / "ml_mcts_selfplay_targets.pt"
    final_state = out_dir / "pq1_plus_mcts_pq_state.pt"
    direct_eval_prefix = out_dir / "direct_eval"
    exact_eval = out_dir / "foundation_eval.csv"
    mcts_eval = out_dir / "benchmark_mcts_eval.csv"

    env_args = [
        "--device",
        args.device,
        "--enable-x-band",
        "--use-arrival-feature",
        "--use-grid-feature",
        "--env-mode",
        "radarxs_mission_delta",
    ]

    if not args.skip_teacher_targets:
        run(
            [
                sys.executable,
                "foundation_mcts_fair_eval.py",
                "--mode",
                "physical_targets",
                "--ckpt",
                args.ckpt,
                "--targets-out",
                teacher_targets,
                "--initials",
                args.initials,
                "--rates",
                args.rates,
                "--seeds",
                args.seeds,
                "--windows",
                str(args.windows),
                "--max-targets",
                str(args.teacher_targets),
                "--max-targets-per-cell",
                str(args.teacher_targets_per_cell),
                "--top-k",
                str(args.teacher_top_k),
                "--score-horizon-ms",
                str(args.teacher_horizon_ms),
                "--label-mode",
                "mix",
                "--teacher-mix",
                "0.65",
            ],
            args.dry_run,
        )

    if not args.skip_teacher_train:
        run(
            [
                sys.executable,
                "alphazero_orthodox.py",
                "--ckpt",
                args.ckpt,
                *env_args,
                "--load-targets",
                teacher_targets,
                "--head-mode",
                "pq",
                "--prior-mode",
                "factorized",
                "--train-steps",
                str(args.teacher_train_steps),
                "--batch-size",
                "64",
                "--lr",
                "1e-4",
                "--type-loss-weight",
                "1.0",
                "--track-loss-weight",
                "1.0",
                "--sensor-loss-weight",
                "0.5",
                "--joint-policy-loss-weight",
                "0.5",
                "--value-loss-weight",
                "0.1",
                "--type-q-loss-weight",
                "0.5",
                "--track-q-loss-weight",
                "0.5",
                "--sensor-q-loss-weight",
                "0.5",
                "--skip-before-eval",
                "--skip-after-eval",
                "--save-state",
                teacher_state,
                "--out-prefix",
                out_dir / "teacher_train",
            ],
            args.dry_run,
        )

    if not args.skip_selfplay:
        selfplay_cmd = [
            sys.executable,
            "alphazero_orthodox.py",
            "--ckpt",
            args.ckpt,
            "--load-state",
            teacher_state,
            *env_args,
            "--target-source",
            "mcts",
            "--episodes",
            str(args.selfplay_episodes),
            "--seed",
            args.seeds.split(",")[0].strip(),
            "--windows",
            str(args.windows),
            "--max-targets-per-episode",
            "256",
            "--train-initials",
            args.initials,
            "--train-rates",
            args.rates,
            "--rollouts",
            str(args.selfplay_rollouts),
            "--horizon-windows",
            "1",
            "--expand-top-k",
            "24",
            "--max-num-considered-actions",
            "32",
            "--c-puct",
            "1.25",
            "--rollout-policy",
            "pq",
            "--skip-default-rollout-seed",
            "--head-mode",
            "pq",
            "--prior-mode",
            "factorized",
            "--plan-mode",
            "window",
            "--window-extract",
            "best",
            "--select-mode",
            "q",
            "--add-prefix-targets",
            "--policy-target",
            "mctx",
            "--puct-q-transform",
            "mctx",
            "--mctx-value-scale",
            "0.1",
            "--append-targets",
            teacher_targets,
            "--save-targets",
            selfplay_targets,
            "--train-steps",
            str(args.selfplay_train_steps),
            "--batch-size",
            "64",
            "--lr",
            "5e-5",
            "--type-loss-weight",
            "1.0",
            "--track-loss-weight",
            "1.0",
            "--sensor-loss-weight",
            "0.5",
            "--joint-policy-loss-weight",
            "0.5",
            "--value-loss-weight",
            "0.1",
            "--type-q-loss-weight",
            "0.5",
            "--track-q-loss-weight",
            "0.5",
            "--sensor-q-loss-weight",
            "0.5",
            "--skip-before-eval",
            "--skip-after-eval",
            "--save-state",
            final_state,
            "--out-prefix",
            out_dir / "selfplay_train",
        ]
        if bool(args.gate_mcts_targets):
            selfplay_cmd.extend(
                [
                    "--reject-selfplay-below-baseline",
                    "--reject-baseline-mode",
                    args.mcts_gate_baseline,
                    "--reject-margin",
                    str(args.mcts_gate_margin),
                ]
            )
            if args.mcts_gate_baseline in {"accepted", "max_all"}:
                selfplay_cmd.extend(["--accepted-baseline-state", teacher_state])
        run(selfplay_cmd, args.dry_run)

    if not args.skip_eval:
        eval_state = final_state if final_state.exists() or not args.dry_run else teacher_state
        run(
            [
                sys.executable,
                "foundation_mcts_fair_eval.py",
                "--mode",
                "eval",
                "--ckpt",
                args.ckpt,
                "--state",
                eval_state,
                "--out",
                exact_eval,
                "--initials",
                args.initials,
                "--rates",
                args.rates,
                "--seeds",
                args.seeds,
                "--windows",
                str(args.windows),
                "--variants",
                args.eval_variants,
            ],
            args.dry_run,
        )
        run(
            [
                sys.executable,
                "alphazero_orthodox.py",
                "--ckpt",
                args.ckpt,
                "--load-state",
                eval_state,
                *env_args,
                "--head-mode",
                "pq",
                "--prior-mode",
                "factorized",
                "--eval-mode",
                "direct",
                "--direct-mode",
                args.direct_mode,
                "--direct-alpha",
                str(args.direct_alpha),
                "--direct-beta",
                str(args.direct_beta),
                "--direct-threshold",
                str(args.direct_threshold),
                "--windows",
                str(args.windows),
                "--eval-initials",
                args.initials,
                "--eval-rates",
                args.rates,
                "--eval-seeds",
                args.seeds,
                "--load-targets",
                teacher_targets,
                "--train-steps",
                "0",
                "--skip-after-eval",
                "--out-prefix",
                direct_eval_prefix,
            ],
            args.dry_run,
        )
        run(
            [
                sys.executable,
                "alphazero_benchmark.py",
                "--ckpt",
                args.ckpt,
                "--state",
                eval_state,
                "--method-name",
                "trained_ml_mcts",
                *env_args,
                "--windows",
                str(args.windows),
                "--initials",
                args.initials,
                "--rates",
                args.rates,
                "--seeds",
                args.seeds,
                "--include-heuristics",
                "--include-model",
                "--rollouts",
                "16",
                "--rollout-policy",
                "pq",
                "--skip-default-rollout-seed",
                "--plan-mode",
                "window",
                "--window-extract",
                "best",
                "--select-mode",
                "q",
                "--head-mode",
                "pq",
                "--prior-mode",
                "factorized",
                "--sensor-action-mode",
                "explicit_head",
                "--horizon-windows",
                "1",
                "--expand-top-k",
                "24",
                "--max-num-considered-actions",
                "32",
                "--puct-q-transform",
                "mctx",
                "--out",
                mcts_eval,
            ],
            args.dry_run,
        )
        if not args.dry_run:
            write_summary(out_dir, exact_eval, direct_eval_prefix.with_name(direct_eval_prefix.name + "_eval.csv"), mcts_eval)
            summary = out_dir / "summary.csv"
            if summary.exists():
                print(pd.read_csv(summary).sort_values("reward", ascending=False).to_string(index=False), flush=True)
            wins = out_dir / "cell_wins.csv"
            if wins.exists():
                print(pd.read_csv(wins).sort_values("mean_margin_vs_best_heuristic", ascending=False).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
