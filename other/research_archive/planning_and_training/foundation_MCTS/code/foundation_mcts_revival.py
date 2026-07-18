from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


OUT = Path(r"C:\Users\yousi\Downloads\radar_outputs\foundation_mcts_revival")
OUT.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="foundation_mcts_gen01")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default=r"C:\Users\yousi\Downloads\radar_outputs\alphazero_orthodox\paper_factorized_pv_current_best.pt")
    ap.add_argument("--env-mode", default="mcts_sched_v1")
    ap.add_argument("--episodes", type=int, default=9)
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--rollouts", type=int, default=24)
    ap.add_argument("--horizon-windows", type=int, default=3)
    ap.add_argument("--expand-top-k", type=int, default=16)
    ap.add_argument("--train-steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-targets-per-episode", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-initials", default="20,60,100")
    ap.add_argument("--train-rates", default="0,4,8")
    ap.add_argument("--eval-initials", default="20,60,100")
    ap.add_argument("--eval-rates", default="0,4,8")
    ap.add_argument("--eval-seeds", default="931,932")
    ap.add_argument("--seed", type=int, default=970)
    ap.add_argument("--rollout-policy", choices=["model", "value", "edf", "est", "mixed"], default="edf")
    ap.add_argument("--eval-mode", choices=["mcts", "direct"], default="mcts")
    ap.add_argument("--target-temperature", type=float, default=0.75)
    ap.add_argument("--policy-kl-weight", type=float, default=0.5)
    ap.add_argument("--mode", choices=["all", "cache", "train"], default="all")
    args = ap.parse_args()

    script = Path(__file__).with_name("alphazero_orthodox.py")
    state_path = OUT / f"{args.tag}_state.pt"
    target_path = OUT / f"{args.tag}_targets.pt"
    prefix = f"..\\foundation_mcts_revival\\{args.tag}"

    base = [
        sys.executable,
        str(script),
        "--ckpt",
        str(args.ckpt),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--episodes",
        str(args.episodes),
        "--windows",
        str(args.windows),
        "--max-targets-per-episode",
        str(args.max_targets_per_episode),
        "--rollouts",
        str(args.rollouts),
        "--horizon-windows",
        str(args.horizon_windows),
        "--expand-top-k",
        str(args.expand_top_k),
        "--c-puct",
        "1.25",
        "--head-mode",
        "pv",
        "--prior-mode",
        "factorized",
        "--search-alg",
        "puct",
        "--select-mode",
        "visits",
        "--disable-visit-unvisited-first",
        "--rollout-policy",
        str(args.rollout_policy),
        "--leaf-value-mix",
        "0.5",
        "--gamma",
        "0.997",
        "--target-source",
        "mcts",
        "--terminal-baseline-target",
        "none",
        "--eval-mode",
        str(args.eval_mode),
        "--train-steps",
        str(args.train_steps),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--train-initials",
        str(args.train_initials),
        "--train-rates",
        str(args.train_rates),
        "--eval-initials",
        str(args.eval_initials),
        "--eval-rates",
        str(args.eval_rates),
        "--eval-seeds",
        str(args.eval_seeds),
        "--env-mode",
        str(args.env_mode),
        "--enable-x-band",
        "--target-temperature",
        str(args.target_temperature),
        "--policy-kl-weight",
        str(args.policy_kl_weight),
        "--joint-policy-loss-weight",
        "1.0",
        "--value-loss-weight",
        "1.0",
        "--sensor-loss-weight",
        "0.5",
        "--track-loss-penalty",
        "8.0",
        "--search-frame-overdue-weight",
        "0.20",
        "--search-frame-drop-penalty",
        "8.0",
        "--out-prefix",
        prefix,
    ]

    if args.mode == "cache":
        run(base + ["--cache-only", "--save-targets", str(target_path), "--skip-before-eval"])
        return
    if args.mode == "train":
        run(base + ["--load-targets", str(target_path), "--save-state", str(state_path)])
        return
    run(base + ["--save-targets", str(target_path), "--save-state", str(state_path)])


if __name__ == "__main__":
    main()
