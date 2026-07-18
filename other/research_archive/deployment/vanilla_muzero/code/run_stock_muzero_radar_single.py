from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "CreateValid1" / "experiments" / "code" / "model_code"
MUZERO = CODE / "third_party" / "muzero-general"
for p in (CODE, MUZERO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ray  # noqa: E402
from muzero import MuZero  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-steps", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=4)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--action-ranks", type=int, default=500)
    parser.add_argument("--max-actions-per-window", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--service-reward", type=float, default=0.0)
    parser.add_argument("--discovery-reward", type=float, default=0.0)
    parser.add_argument("--search-refresh-reward", type=float, default=0.0)
    parser.add_argument("--network", choices=["fullyconnected", "factorized_fullyconnected"], default="fullyconnected")
    parser.add_argument("--factorized-search-logit-offset", type=float, default=0.0)
    args = parser.parse_args()

    os.environ["RADAR_MUZERO_WINDOWS"] = str(args.windows)
    os.environ["RADAR_MUZERO_ACTION_RANKS"] = str(args.action_ranks)
    os.environ["RADAR_MUZERO_MAX_ACTIONS_PER_WINDOW"] = str(args.max_actions_per_window)
    os.environ["RADAR_MUZERO_WORKERS"] = str(args.workers)
    os.environ["RADAR_MUZERO_SIMULATIONS"] = str(args.simulations)
    os.environ["RADAR_MUZERO_BATCH_SIZE"] = str(args.batch_size)
    os.environ["RADAR_MUZERO_TRAINING_STEPS"] = str(args.training_steps)
    os.environ["RADAR_MUZERO_SERVICE_REWARD"] = str(args.service_reward)
    os.environ["RADAR_MUZERO_DISCOVERY_REWARD"] = str(args.discovery_reward)
    os.environ["RADAR_MUZERO_SEARCH_REFRESH_REWARD"] = str(args.search_refresh_reward)
    os.environ["RADAR_MUZERO_NETWORK"] = str(args.network)
    os.environ["RADAR_MUZERO_FACTORIZED_SEARCH_LOGIT_OFFSET"] = str(args.factorized_search_logit_offset)
    os.environ["PYTHONPATH"] = str(MUZERO) + os.pathsep + str(CODE) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.chdir(MUZERO)

    config = {
        "training_steps": args.training_steps,
        "num_workers": args.workers,
        "num_simulations": args.simulations,
        "max_moves": args.windows * args.max_actions_per_window,
        "batch_size": args.batch_size,
        "checkpoint_interval": max(1, min(5, args.training_steps)),
        "replay_buffer_size": max(16, args.workers * 8),
        "ratio": None,
        "save_model": True,
    }

    muzero = MuZero("radarxs_single", config)
    t0 = time.perf_counter()
    muzero.train(log_in_tensorboard=False)
    try:
        while True:
            step = ray.get(muzero.shared_storage_worker.get_info.remote("training_step"))
            games = ray.get(muzero.shared_storage_worker.get_info.remote("num_played_games"))
            print({"training_step": step, "num_played_games": games})
            if step >= args.training_steps:
                break
            time.sleep(2.0)
    finally:
        muzero.terminate_workers()
        ray.shutdown()
    print({"elapsed_s": time.perf_counter() - t0, "results_path": str(muzero.config.results_path)})


if __name__ == "__main__":
    main()
