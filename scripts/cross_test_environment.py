from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


TRACE_CODE = r'''
import json
import numpy as np
from pufferlib.ocean.radarxs.engine import RadarEngine
from pufferlib.ocean.radarxs.engine import get_obs_from_buf
from pufferlib.ocean.radarxs.models.edf import EDFPlanner
from pufferlib.ocean.radarxs.models.est import ESTPlanner

rows = {}
for name, planner in {"EDF": EDFPlanner(100), "EST": ESTPlanner(100)}.items():
    engine = RadarEngine(
        planner, initial_targets=20, max_trackers=100, seed=916, window_ms=200,
        enable_global_delay=True, enable_local_delay=False,
        enable_x_band=False, enable_search_refresh_tracked=False,
        search_refresh_gain=0.0, enable_priority=False,
        enable_poisson_arrivals=True, activate_all_targets_without_poisson=True,
        poisson_rate_per_second=2.0, search_action_reward=0.0,
        track_update_reward=0.0, track_loss_penalty=8.0,
        target_service_weight=0.0, sector_staleness_weight=0.0,
        searched_sector_reward_weight=0.0, search_frame_overdue_weight=0.5,
        search_frame_desired_ms=3000.0, search_frame_deadline_ms=4500.0,
        search_frame_drop_penalty=8.0, search_task_cost_mode=1,
        revisit_time_scale=0.75, penalize_hidden_targets=False,
        episode_time_limit_ms=2_000_000_000, search_delay_mode=0,
        search_debt_penalty_weight=0.0, search_debt_tau_ms=200.0,
        search_delay_penalty_cap=-1.0,
    )
    rewards = [float(engine.step_window()) for _ in range(20)]
    obs = get_obs_from_buf(engine.obs_buf, 100)
    active = np.asarray(obs["active_mask"], dtype=bool)
    tracked = active & (np.asarray(obs["t_deadline"]) >= 0.0)
    rows[name] = {
        "rewards": rewards,
        "active": int(active.sum()),
        "tracked": int(tracked.sum()),
        "desired": np.asarray(obs["t_desired"])[active].round(5).tolist(),
        "deadline": np.asarray(obs["t_deadline"])[active].round(5).tolist(),
        "total": float(engine.total_reward),
    }
    engine.close()
print(json.dumps(rows))
'''


def trace(root: Path) -> dict:
    output = subprocess.check_output([sys.executable, "-c", TRACE_CODE], cwd=root, text=True)
    return json.loads(output)


STATE_KEYS = ("active", "tracked", "desired", "deadline")
REWARD_KEYS = ("rewards", "total")


def maximum_difference(left: dict, right: dict, keys: tuple[str, ...]) -> float:
    maximum = 0.0
    for method in sorted(left):
        for key in keys:
            left_values = np.asarray(left[method][key], dtype=np.float64)
            right_values = np.asarray(right[method][key], dtype=np.float64)
            if left_values.shape != right_values.shape:
                raise SystemExit(f"trace shapes differ for {method}.{key}")
            if left_values.size:
                maximum = max(maximum, float(np.max(np.abs(left_values - right_values))))
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--other-root", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument(
        "--compare-reward",
        action="store_true",
        help="also require reward parity; use only when both checkouts share a reward contract",
    )
    args = parser.parse_args()
    current = Path(__file__).resolve().parents[1]
    left = trace(current)
    right = trace(args.other_root.resolve())
    if left.keys() != right.keys():
        raise SystemExit("method sets differ")
    state_difference = maximum_difference(left, right, STATE_KEYS)
    reward_difference = maximum_difference(left, right, REWARD_KEYS)
    if state_difference > args.atol:
        raise SystemExit(f"state trajectory mismatch: max_abs_diff={state_difference:.6g}")
    if args.compare_reward and reward_difference > args.atol:
        raise SystemExit(f"reward mismatch: max_abs_diff={reward_difference:.6g}")
    print(
        json.dumps(
            {
                "state_match": True,
                "reward_compared": args.compare_reward,
                "windows": 20,
                "methods": ["EDF", "EST"],
                "state_max_abs_diff": state_difference,
                "reward_max_abs_diff": reward_difference,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
