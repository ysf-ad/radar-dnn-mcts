from __future__ import annotations

import argparse
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np


GROUPED_KEYS = (
    "tokens",
    "context",
    "policy",
    "actions",
    "rewards",
    "returns",
    "durations_ms",
    "action_mask",
)

SCALAR_METADATA_KEYS = (
    "puct_rollouts",
    "puct_c",
    "puct_discount",
    "return_scale",
    "dirichlet_alpha",
    "dirichlet_fraction",
    "teacher",
    "search_model",
)


def csv_values(value: str, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def run(command: list[str], root: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    subprocess.run(command, cwd=root, env=environment, check=True)


def merge_trajectories(paths: list[Path], output: Path) -> None:
    sources = [np.load(path) for path in paths]
    for path, source in zip(paths, sources):
        missing = set(GROUPED_KEYS) - set(source.files)
        if missing:
            raise KeyError(f"{path} is missing grouped arrays: {sorted(missing)}")
    max_steps = max(source["action_mask"].shape[1] for source in sources)
    merged: dict[str, np.ndarray] = {}
    for key in GROUPED_KEYS:
        chunks = []
        for source in sources:
            array = source[key]
            steps = source["action_mask"].shape[1]
            if array.ndim >= 2 and array.shape[1] == steps and steps < max_steps:
                padding_shape = list(array.shape)
                padding_shape[1] = max_steps - steps
                padding = np.zeros(padding_shape, dtype=array.dtype)
                array = np.concatenate((array, padding), axis=1)
            chunks.append(array)
        merged[key] = np.concatenate(chunks, axis=0)
    for key in SCALAR_METADATA_KEYS:
        if all(key in source.files for source in sources):
            values = [source[key] for source in sources]
            if all(np.array_equal(values[0], value) for value in values[1:]):
                merged[key] = values[0]
    merged["source_files"] = np.asarray([str(path) for path in paths])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    for source in sources:
        source.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alternate full-window PUCT collection and policy/value training."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--rollouts",
        type=int,
        default=32,
        help="Complete scheduling-window PUCT rollouts.",
    )
    parser.add_argument("--windows", type=int, default=50)
    parser.add_argument("--initial-targets", default="20,40,60")
    parser.add_argument("--arrival-rates", default="2,3,4")
    parser.add_argument("--seeds", default="916")
    parser.add_argument(
        "--search-model",
        choices=("core", "ar"),
        default="ar",
        help="Policy/value evaluator used inside PUCT collection.",
    )
    parser.add_argument(
        "--learner",
        choices=("core", "ar", "muzero", "batch"),
        help="Model updated after collection; defaults to the selected search model.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--unroll-steps", type=int, default=10)
    parser.add_argument("--core-lr-scale", type=float, default=1.0)
    parser.add_argument("--reward-loss-weight", type=float, default=32.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.iterations < 1 or args.rollouts < 1 or args.windows < 1:
        parser.error("iterations, rollouts, and windows must be positive")

    root = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint.resolve()
    initial_targets = csv_values(args.initial_targets, int)
    arrival_rates = csv_values(args.arrival_rates, float)
    seeds = csv_values(args.seeds, int)
    learner = args.learner or args.search_model

    for iteration in range(1, args.iterations + 1):
        iteration_dir = args.out_dir.resolve() / f"iteration_{iteration:03d}"
        trajectory_paths = []
        for initial, arrival, seed in product(
            initial_targets, arrival_rates, seeds
        ):
            trajectory = (
                iteration_dir
                / "trajectories"
                / f"i{initial}_r{arrival:g}_s{seed}.npz"
            )
            trajectory_paths.append(trajectory)
            run(
                [
                    sys.executable,
                    str(root / "scripts" / "collect_puct_targets.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--teacher",
                    "puct",
                    "--search-model",
                    args.search_model,
                    "--rollouts",
                    str(args.rollouts),
                    "--windows",
                    str(args.windows),
                    "--initial-targets",
                    str(initial),
                    "--arrival-rate",
                    str(arrival),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                    "--out",
                    str(trajectory),
                ],
                root,
            )

        dataset = iteration_dir / "trajectories.npz"
        merge_trajectories(trajectory_paths, dataset)
        next_checkpoint = iteration_dir / "checkpoint.pt"
        if learner in ("ar", "batch"):
            trainer = root / "scripts" / "train_sequence.py"
            command = [
                sys.executable,
                str(trainer),
                "--decoder",
                learner,
                "--value-loss-weight",
                "1",
            ]
        elif learner == "muzero":
            trainer = root / "scripts" / "train_dynamics.py"
            command = [
                sys.executable,
                str(trainer),
                "--unroll-steps",
                str(args.unroll_steps),
                "--core-lr-scale",
                str(args.core_lr_scale),
                "--reward-loss-weight",
                str(args.reward_loss_weight),
            ]
        else:
            trainer = root / "scripts" / "train.py"
            command = [sys.executable, str(trainer)]
        command.extend(
            [
                "--data",
                str(dataset),
                "--checkpoint",
                str(checkpoint),
                "--out",
                str(next_checkpoint),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--device",
                args.device,
            ]
        )
        run(command, root)
        checkpoint = next_checkpoint
        print(f"completed iteration {iteration}: {checkpoint}")


if __name__ == "__main__":
    main()
