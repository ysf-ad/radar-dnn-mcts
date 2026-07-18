from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = Path(r"C:\Users\yousi\Downloads\radar_outputs\alphazero_orthodox")


ENV_ARGS = [
    "--device", "cpu",
    "--env-mode", "radarxs_mission_delta",
    "--enable-x-band",
    "--track-loss-penalty", "4.0",
    "--target-service-weight", "10.0",
    "--target-service-horizon-ms", "3000",
    "--sector-staleness-weight", "0.01",
    "--search-frame-overdue-weight", "0.01",
    "--search-frame-drop-penalty", "4.0",
]


BOARDGAME_ARGS = [
    "--terminal-baseline-target", "sign",
    "--terminal-score-metric", "mean_health",
    "--terminal-baseline-mode", "max_heuristic",
    "--terminal-tracked-weight", "1.0",
    "--terminal-drop-weight", "0.5",
    "--terminal-delay-weight", "0.01",
    "--terminal-baseline-margin", "0.5",
]


def run_cmd(args: list[str]) -> None:
    print(" ".join(args), flush=True)
    subprocess.run(args, cwd=str(ROOT), check=True)


def eval_score(csv_path: Path) -> float:
    df = pd.read_csv(csv_path)
    if df.empty:
        return -1e18
    if "planner" in df.columns:
        df = df[df["planner"].fillna("") == "model_direct"]
    return float(df["reward"].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--start-state", required=True)
    ap.add_argument("--accepted-state", default=str(OUT / "selfplay_accepted.pt"))
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--stability-window", type=int, default=2)
    ap.add_argument("--min-improvement", type=float, default=0.02)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--train-steps", type=int, default=180)
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--train-initials", default="20,60,100")
    ap.add_argument("--train-rates", default="0,4,8")
    ap.add_argument("--eval-seeds", default="914,915")
    ap.add_argument("--eval-initials", default="100")
    ap.add_argument("--eval-rates", default="4")
    ap.add_argument("--direct-threshold", type=float, default=0.95)
    args = ap.parse_args()

    accepted = Path(args.accepted_state)
    accepted.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.start_state, accepted)
    best_score = -1e18
    stable = 0
    rows = []

    for gen in range(1, int(args.generations) + 1):
        prefix = f"selfplay_gen{gen:02d}"
        target_path = OUT / f"{prefix}_targets.pt"
        state_path = OUT / f"{prefix}_state.pt"
        train_prefix = f"{prefix}_train"
        eval_prefix = f"{prefix}_direct_gate"

        train_cmd = [
            sys.executable, "alphazero_orthodox.py",
            "--ckpt", args.ckpt,
            "--load-state", str(accepted),
            *ENV_ARGS,
            "--target-source", "model_direct",
            "--episodes", str(args.episodes),
            "--windows", str(args.windows),
            "--train-initials", args.train_initials,
            "--train-rates", args.train_rates,
            "--train-steps", str(args.train_steps),
            "--batch-size", "128",
            "--lr", "3e-5",
            *BOARDGAME_ARGS,
            "--type-loss-weight", "1.0",
            "--track-loss-weight", "1.0",
            "--sensor-loss-weight", "0.5",
            "--value-loss-weight", "0.5",
            "--policy-positive-only",
            "--policy-positive-margin", "0.0",
            "--policy-kl-weight", "0.2",
            "--direct-mode", "branch",
            "--direct-threshold", str(args.direct_threshold),
            "--skip-before-eval",
            "--skip-after-eval",
            "--save-targets", str(target_path),
            "--save-state", str(state_path),
            "--out-prefix", train_prefix,
        ]
        run_cmd(train_cmd)

        eval_cmd = [
            sys.executable, "alphazero_orthodox.py",
            "--ckpt", args.ckpt,
            "--load-state", str(state_path),
            "--load-targets", str(target_path),
            *ENV_ARGS,
            "--windows", str(args.windows),
            "--eval-mode", "direct",
            "--eval-seeds", args.eval_seeds,
            "--eval-initials", args.eval_initials,
            "--eval-rates", args.eval_rates,
            "--train-steps", "0",
            "--batch-size", "128",
            "--direct-mode", "branch",
            "--direct-threshold", str(args.direct_threshold),
            "--skip-before-eval",
            "--out-prefix", eval_prefix,
        ]
        run_cmd(eval_cmd)

        score = eval_score(OUT / f"{eval_prefix}_eval.csv")
        improved = score > best_score + float(args.min_improvement)
        if improved:
            shutil.copyfile(state_path, accepted)
            best_score = score
            stable = 0
        else:
            stable += 1
        row = {"generation": gen, "score": score, "best_score": best_score, "accepted": bool(improved), "stable_count": stable}
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "selfplay_loop_summary.csv", index=False)
        print(row, flush=True)
        if stable >= int(args.stability_window):
            print("stabilized", flush=True)
            break


if __name__ == "__main__":
    main()
