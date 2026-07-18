"""Collect causal boundary transitions from the deployed S-only AR planner."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from boundary_dataset import BoundaryRecord, save_records
from boundary_realtime_eval import add_reward_args, load_ar, plan_from_tokens
from eval_action_attention_muzero_g import build_env, execute_plan_until_budget_joint_shaped, get_obs
from exact_env_mutual import MAXT, _DummyPlanner, engine_env_cfg, env_cfg_for, xs_decode_action
from foundation_mcts_fair_eval import parse_floats, parse_ints
from mutual_features import tokenize
from penalty_window_quota_learner_eval import make_exact_args
from realistic_reward_retrain import adapter


def canonical_action(simulator_action: int) -> int:
    row, sensor = xs_decode_action(int(simulator_action), MAXT)
    if sensor != 0:
        raise ValueError(f"non-S action in S-only plan: {simulator_action}")
    return 0 if row <= 0 else 2 * int(row)


def collect_cell(model, initial: int, rate: float, seed: int, windows: int, args) -> list[BoundaryRecord]:
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = False
    exact_args.single_sensor = True
    env_cfg = env_cfg_for(float(rate), exact_args)
    env_cfg["enable_x_band"] = 0
    eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
    eng.reset(seed=int(seed))
    adapt = adapter()
    debt = 0.0
    records: list[BoundaryRecord] = []
    current = plan_from_tokens(model, tokenize(adapt, get_obs(eng, debt), set(), 0), args.max_steps)
    try:
        for window in range(int(windows)):
            spent = 0.0
            midpoint_tokens = None
            midpoint_spent = 0.0
            executed_suffix: list[int] = []
            for action in current:
                result = execute_plan_until_budget_joint_shaped(
                    eng,
                    [int(action)],
                    200.0 - spent,
                    debt,
                    "Boundary collection",
                    seed,
                    window,
                    env_cfg,
                )
                _, dt, debt, count, _, _ = result
                if count <= 0 or dt <= 0:
                    continue
                spent += float(dt)
                if midpoint_tokens is None and spent >= float(args.plan_start_ms):
                    midpoint_tokens = tokenize(adapt, get_obs(eng, debt), set(), 0).astype(np.float32)
                    midpoint_spent = float(spent)
                elif midpoint_tokens is not None and len(executed_suffix) < int(args.max_suffix):
                    executed_suffix.append(canonical_action(int(action)))

            boundary_tokens = tokenize(adapt, get_obs(eng, debt), set(), 0).astype(np.float32)
            if midpoint_tokens is not None:
                suffix = np.zeros((int(args.max_suffix),), dtype=np.int64)
                suffix[: len(executed_suffix)] = np.asarray(executed_suffix, dtype=np.int64)
                records.append(
                    BoundaryRecord(
                        midpoint_tokens=midpoint_tokens,
                        suffix_actions=suffix,
                        suffix_length=len(executed_suffix),
                        remaining_time_ms=max(0.0, 200.0 - midpoint_spent),
                        boundary_tokens=boundary_tokens,
                        seed=int(seed),
                        episode=int(initial * 100000 + round(rate * 1000) * 100 + window),
                        midpoint_step=0,
                    )
                )
            current = plan_from_tokens(model, boundary_tokens, args.max_steps)
    finally:
        eng.close()
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--initials", default="20,40,60")
    parser.add_argument("--rates", default="2,3,4")
    parser.add_argument("--seeds", default="928,932,944")
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--plan-start-ms", type=float, default=100.0)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--max-suffix", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_reward_args(parser)
    args = parser.parse_args()
    model = load_ar(args.ar_checkpoint, torch.device(args.device))
    records: list[BoundaryRecord] = []
    for initial in parse_ints(args.initials):
        for rate in parse_floats(args.rates):
            for seed in parse_ints(args.seeds):
                cell = collect_cell(model, initial, rate, seed, args.windows, args)
                records.extend(cell)
                print({"initial": initial, "rate": rate, "seed": seed, "records": len(cell)}, flush=True)
    save_records(
        args.out,
        records,
        metadata={
            "source": str(args.ar_checkpoint),
            "plan_start_ms": float(args.plan_start_ms),
            "semantics": "deployed AR midpoint plus actually executed suffix to true boundary",
        },
    )
    print({"saved": str(args.out), "records": len(records)})


if __name__ == "__main__":
    main()
