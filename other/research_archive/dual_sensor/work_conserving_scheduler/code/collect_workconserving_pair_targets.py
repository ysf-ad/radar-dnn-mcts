from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "CreateValid1" / "experiments" / "code" / "model_code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from best_model_joint_vs_seq_ablation import PairHeadWorkConservingPlanner, WorkConservingAsyncCoupledPlanner
from penalty_window_quota_learner_eval import make_exact_args
from two_sensor_physical_head_eval import PhysicalHeadPlanner, collect_targets, make_physical_model
from alphazero_orthodox import save_targets


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior-state", required=True)
    ap.add_argument("--behavior-variant", default="two_row_action_attention_factored_loss")
    ap.add_argument("--targets-out", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--initials", default="20,40,60")
    ap.add_argument("--rates", default="2,3,4")
    ap.add_argument("--train-seeds", default="916")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--max-targets", type=int, default=540)
    ap.add_argument("--max-targets-per-cell", type=int, default=60)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--tail-windows", type=int, default=10)
    ap.add_argument("--tail-policy", choices=["est", "edf"], default="edf")
    ap.add_argument("--policy-tau", type=float, default=0.1)
    ap.add_argument("--potential-weight", type=float, default=0.0)
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--q-weight", type=float, default=0.0)
    ap.add_argument("--search-bias", type=float, default=0.0)
    ap.add_argument("--per-sensor-top", type=int, default=3)
    ap.add_argument("--force-search-candidate", action="store_true")
    ap.add_argument("--preserve-busy-features", action="store_true")
    ap.add_argument("--planner-scored-candidates", action="store_true")
    ap.add_argument("--hybrid-scored-candidates", action="store_true")
    ap.add_argument("--pair-type-count-corrected", action="store_true")
    ap.add_argument("--model-seed", type=int, default=123)
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--cell-balanced-sampling", action="store_true")
    return ap.parse_args()


def load_behavior_model(args: argparse.Namespace):
    model_args = SimpleNamespace(d_model=int(args.d_model), nhead=int(args.nhead), nlayers=int(args.nlayers))
    model = make_physical_model(str(args.behavior_variant), model_args)
    state = torch.load(args.behavior_state, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        {
            "loaded_behavior_state": str(args.behavior_state),
            "missing": list(missing),
            "unexpected": list(unexpected),
        },
        flush=True,
    )
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.model_seed))
    np.random.seed(int(args.model_seed))
    torch.set_num_threads(1)

    behavior_model = load_behavior_model(args)

    def behavior_factory(env_cfg: dict):
        base = PhysicalHeadPlanner(
            behavior_model,
            str(args.behavior_variant),
            env_cfg,
            policy_weight=float(args.policy_weight),
            q_weight=float(args.q_weight),
            search_score_bias=float(args.search_bias),
        )
        if str(args.behavior_variant) == "two_row_pair_action_attention":
            return PairHeadWorkConservingPlanner(
                behavior_model,
                env_cfg,
                variant=str(args.behavior_variant),
                per_sensor_top=int(args.per_sensor_top),
                policy_weight=float(args.policy_weight),
                q_weight=float(args.q_weight),
                search_score_bias=float(args.search_bias),
                include_search_candidate=bool(args.force_search_candidate),
            )
        return WorkConservingAsyncCoupledPlanner(
            base,
            per_sensor_top=int(args.per_sensor_top),
            include_search_candidate=bool(args.force_search_candidate),
        )

    collect_args = SimpleNamespace(**vars(args))
    collect_args.behavior_policy = "edf"
    collect_args.joint_candidate_targets = True
    collect_args.checkpoint_each_cell = True
    collect_args.bootstrap_state = ""
    collect_args.bootstrap_variant = str(args.behavior_variant)
    collect_args.bootstrap_value_weight = 0.0
    collect_args.preserve_busy_features = bool(args.preserve_busy_features)
    collect_args.planner_scored_candidates = bool(args.planner_scored_candidates)
    collect_args.hybrid_scored_candidates = bool(args.hybrid_scored_candidates)
    collect_args.pair_type_count_corrected = bool(args.pair_type_count_corrected)
    exact_args = make_exact_args(collect_args)
    exact_args.enable_x_band = True
    exact_args.single_sensor = False
    targets = collect_targets(collect_args, exact_args, Path(args.targets_out), behavior_factory=behavior_factory)
    # collect_targets normally saves only at the end. Keep this explicit save so
    # callers of this wrapper can rely on the target path being materialized.
    save_targets(Path(args.targets_out), targets)
    print({"saved_targets": str(args.targets_out), "targets": len(targets)}, flush=True)


if __name__ == "__main__":
    main()
