"""Alternate simulator collection, replay training, and held-out promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


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
    "return_horizon_windows",
    "dirichlet_alpha",
    "dirichlet_fraction",
    "teacher",
    "search_model",
)


def csv_values(value: str, cast):
    """Parse a comma-separated command-line value."""
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def run(command: list[str], root: Path) -> None:
    """Run one repository command with the local package importable."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    subprocess.run(command, cwd=root, env=environment, check=True)


def merge_trajectories(paths: list[Path], output: Path) -> None:
    """Merge cell datasets while padding only the schedule-step axis."""
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


def trim_replay(path: Path, capacity_windows: int) -> None:
    """Keep the newest grouped windows in a bounded replay dataset."""
    if capacity_windows <= 0:
        return
    source = np.load(path)
    window_count = source["action_mask"].shape[0]
    if window_count <= capacity_windows:
        source.close()
        return
    start = window_count - capacity_windows
    arrays = {}
    for key in source.files:
        value = source[key]
        arrays[key] = (
            value[start:]
            if value.ndim and value.shape[0] == window_count
            else value
        )
    source.close()
    np.savez_compressed(path, **arrays)


def evaluate_checkpoint(
    checkpoint: Path,
    config: Path,
    method: str,
    output: Path,
    root: Path,
    device: str,
) -> tuple[float, float]:
    """Return learned-method and EDF rewards on one fixed validation grid."""
    run(
        [
            sys.executable,
            str(root / "scripts" / "evaluate.py"),
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config),
            "--methods",
            f"{method},edf",
            "--device",
            device,
            "--out",
            str(output),
        ],
        root,
    )
    summary = pd.read_csv(output / "summary.csv").set_index("method")
    return (
        float(summary.loc[method, "reward_per_window"]),
        float(summary.loc["edf", "reward_per_window"]),
    )


def checkpoint_hash(path: Path) -> str:
    """Compute the checkpoint identity recorded in the run manifest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Alternate PUCT collection and a selected learner."""
    parser = argparse.ArgumentParser(
        description="Alternate full-window PUCT collection and policy/value training."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--validation-config", type=Path)
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
        choices=("core", "ar", "muzero"),
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
    parser.add_argument("--return-horizon-windows", type=int, default=5)
    parser.add_argument("--replay-capacity-windows", type=int, default=5000)
    parser.add_argument("--greedy-training-actions", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.iterations < 1 or args.rollouts < 1 or args.windows < 1:
        parser.error("iterations, rollouts, and windows must be positive")

    root = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint.resolve()
    # A YAML grid replaces only workload axes; CLI flags still control the
    # optimization and search budgets.
    if args.train_config:
        train_config = yaml.safe_load(args.train_config.read_text())
        initial_targets = tuple(int(x) for x in train_config["initial_targets"])
        arrival_rates = tuple(float(x) for x in train_config["arrival_rates"])
        seeds = tuple(int(x) for x in train_config["seeds"])
    else:
        initial_targets = csv_values(args.initial_targets, int)
        arrival_rates = csv_values(args.arrival_rates, float)
        seeds = csv_values(args.seeds, int)
    learner = args.learner or args.search_model
    method = {
        "core": "reencode",
        "ar": "ar",
        "muzero": "muzero",
        "batch": "batch",
    }[learner]
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    incumbent = output / "incumbent.pt"
    shutil.copy2(checkpoint, incumbent)
    checkpoint = incumbent
    history: list[dict] = []
    incumbent_reward = None
    edf_reward = None
    if args.validation_config:
        # Establish the acceptance thresholds before the first update.
        incumbent_reward, edf_reward = evaluate_checkpoint(
            incumbent,
            args.validation_config.resolve(),
            method,
            output / "validation_000",
            root,
            args.device,
        )
        history.append(
            {
                "iteration": 0,
                "candidate_reward": incumbent_reward,
                "incumbent_reward": incumbent_reward,
                "edf_reward": edf_reward,
                "accepted": True,
            }
        )

    for iteration in range(1, args.iterations + 1):
        # Collect fresh on-policy windows for every configured load cell.
        iteration_dir = output / f"iteration_{iteration:03d}"
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
                    "--return-horizon-windows",
                    str(args.return_horizon_windows),
                    *(
                        ["--greedy-training-actions"]
                        if args.greedy_training_actions
                        else []
                    ),
                    "--out",
                    str(trajectory),
                ],
                root,
            )

        current = iteration_dir / "trajectories.npz"
        merge_trajectories(trajectory_paths, current)

        # Append fresh windows, retain the newest capacity, and atomically
        # replace the replay file used by the selected trainer.
        replay = output / "replay_buffer.npz"
        replay_sources = [current]
        if replay.exists():
            replay_sources.insert(0, replay)
        temporary_replay = output / "replay_buffer.next.npz"
        merge_trajectories(replay_sources, temporary_replay)
        trim_replay(temporary_replay, args.replay_capacity_windows)
        temporary_replay.replace(replay)
        next_checkpoint = iteration_dir / "candidate.pt"

        # All learners consume the same replay contract; only their model
        # unrolling and loss implementation differ.
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
                str(replay),
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
        accepted = True
        candidate_reward = None
        if args.validation_config:
            # Validation is held fixed across iterations. A candidate must
            # improve both the current model and the EDF reference.
            candidate_reward, candidate_edf = evaluate_checkpoint(
                next_checkpoint,
                args.validation_config.resolve(),
                method,
                iteration_dir / "validation",
                root,
                args.device,
            )
            edf_reward = candidate_edf
            accepted = (
                candidate_reward > float(incumbent_reward)
                and candidate_reward > float(edf_reward)
            )
        if accepted:
            shutil.copy2(next_checkpoint, incumbent)
            checkpoint = incumbent
            if candidate_reward is not None:
                incumbent_reward = candidate_reward
        history.append(
            {
                "iteration": iteration,
                "candidate_reward": candidate_reward,
                "incumbent_reward": incumbent_reward,
                "edf_reward": edf_reward,
                "accepted": accepted,
            }
        )
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(
            f"completed iteration {iteration}: "
            f"{'accepted' if accepted else 'rejected'}"
        )

    # Record the exact accepted checkpoint and essential run settings without
    # copying generated datasets into version control.
    manifest = {
        "checkpoint": str(incumbent),
        "sha256": checkpoint_hash(incumbent),
        "learner": learner,
        "search_model": args.search_model,
        "rollouts": args.rollouts,
        "windows": args.windows,
        "iterations": args.iterations,
        "return_horizon_windows": args.return_horizon_windows,
        "seeds": list(seeds),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
