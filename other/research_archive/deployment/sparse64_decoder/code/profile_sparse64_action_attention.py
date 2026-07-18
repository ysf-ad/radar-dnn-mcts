from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from eval_action_attention_muzero_g import run_plan_eval
from exact_env_mutual import MAXT, env_cfg_for
from foundation_mcts_fair_eval import parse_floats, parse_ints
from penalty_window_quota_learner_eval import make_exact_args
from single_sensor_ar_action_attention import (
    CachedSingleSensorActionAttentionAR,
    load_action_attention_model,
)
from two_sensor_physical_head_eval import PhysicalHeadPlanner


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_STATE = ROOT / "CreateValid1" / "results" / "single_sensor_fair_exact_action_attention_train_two_row_action_attention_qpolicy_factored_loss.pt"


class ProfiledCachedPlanner(CachedSingleSensorActionAttentionAR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_seconds = 0.0
        self.score_calls = 0
        self.plan_seconds = 0.0
        self.plan_calls = 0

    def _scores_from_encoded(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = super()._scores_from_encoded(*args, **kwargs)
        self.score_seconds += time.perf_counter() - t0
        self.score_calls += 1
        return out

    def plan(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = super().plan(*args, **kwargs)
        self.plan_seconds += time.perf_counter() - t0
        self.plan_calls += 1
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-state", default=str(DEFAULT_STATE))
    ap.add_argument("--initials", default="40")
    ap.add_argument("--rates", default="3")
    ap.add_argument("--seeds", default="916")
    ap.add_argument("--windows", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--q-weight", type=float, default=0.5)
    ap.add_argument("--single-sensor-action-only", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--out", default="CreateValid1/results/profile_sparse64_action_attention.csv")
    args = ap.parse_args()

    import torch

    torch.set_num_threads(int(args.torch_threads))
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = False
    exact_args.single_sensor = True
    model = load_action_attention_model(Path(args.base_state), args.device, "two_row_action_attention")

    rows = []
    for initial in parse_ints(args.initials):
        for rate in parse_floats(args.rates):
            env_cfg = env_cfg_for(float(rate), exact_args)
            env_cfg["enable_x_band"] = 0
            for seed in parse_ints(args.seeds):
                base = PhysicalHeadPlanner(
                    model,
                    "two_row_action_attention",
                    env_cfg,
                    policy_weight=1.0,
                    q_weight=float(args.q_weight),
                    search_score_bias=0.0,
                )
                planner = ProfiledCachedPlanner(
                    base,
                    max_steps=int(args.max_steps),
                    search_floor=0,
                    search_cap_frac=1.0,
                    env_cfg=env_cfg,
                    action_coupler_top_k=int(args.top_k),
                    sparse_residuals=True,
                    single_sensor_action_only=bool(args.single_sensor_action_only),
                )
                df, actions = run_plan_eval(planner, "profiled_sparse", int(initial), int(seed), int(args.windows), env_cfg)
                n_windows = max(1, int(args.windows))
                score_ms = 1000.0 * planner.score_seconds / n_windows
                plan_ms = 1000.0 * planner.plan_seconds / n_windows
                row = {
                    "initial": int(initial),
                    "rate": float(rate),
                    "seed": int(seed),
                    "windows": n_windows,
                    "reward_per_window": float(df["window_reward"].mean()),
                    "plans_per_window": float(planner.plan_calls / n_windows),
                    "score_calls_per_window": float(planner.score_calls / n_windows),
                    "score_ms_per_window": score_ms,
                    "non_score_ms_per_window": plan_ms - score_ms,
                    "plan_ms_per_window": plan_ms,
                    "actions_per_window": float(actions.groupby("window").size().mean()) if not actions.empty else 0.0,
                }
                rows.append(row)
                print(row, flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(str(out.resolve()), flush=True)


if __name__ == "__main__":
    main()
