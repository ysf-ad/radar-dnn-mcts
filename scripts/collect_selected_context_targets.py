from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "radar_dnn_mcts"))


def load_state(model, path: Path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state, strict=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/selected_context_teacher_targets.pt"))
    parser.add_argument("--behavior-state", type=Path, default=Path("../CreateValid1/results/critic_bootstrap_medium_eval_two_row_action_attention_qpolicy_factored_loss.pt"))
    parser.add_argument("--behavior-variant", default="two_row_action_attention_qpolicy_factored_loss")
    parser.add_argument("--initials", default="20,40,60")
    parser.add_argument("--rates", default="2,3,4")
    parser.add_argument("--train-seeds", default="916,917")
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--max-targets", type=int, default=512)
    parser.add_argument("--max-targets-per-cell", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--tail-windows", type=int, default=1)
    parser.add_argument("--tail-policy", choices=["edf", "est"], default="edf")
    parser.add_argument("--policy-tau", type=float, default=5.0)
    parser.add_argument("--potential-weight", type=float, default=1.0)
    parser.add_argument("--policy-score-weight", type=float, default=1.0)
    parser.add_argument("--q-score-weight", type=float, default=1.0)
    args = parser.parse_args()

    from penalty_window_quota_learner_eval import make_exact_args
    from two_sensor_physical_head_eval import (
        PhysicalHeadPlanner,
        collect_targets,
        make_physical_model,
    )

    torch.set_num_threads(1)
    np.random.seed(123)
    torch.manual_seed(123)
    collect_args = SimpleNamespace(
        targets_out=str(args.out),
        initials=str(args.initials),
        rates=str(args.rates),
        train_seeds=str(args.train_seeds),
        windows=int(args.windows),
        max_targets=int(args.max_targets),
        max_targets_per_cell=int(args.max_targets_per_cell),
        top_k=int(args.top_k),
        tail_windows=int(args.tail_windows),
        behavior_policy="edf",
        tail_policy=str(args.tail_policy),
        policy_tau=float(args.policy_tau),
        potential_weight=float(args.potential_weight),
        d_model=48,
        nhead=4,
        nlayers=2,
        bootstrap_state="",
        bootstrap_variant="flat",
        bootstrap_value_weight=0.0,
    )
    exact_args = make_exact_args(collect_args)
    exact_args.enable_x_band = True
    exact_args.single_sensor = False
    behavior_model = load_state(make_physical_model(str(args.behavior_variant), collect_args).eval(), args.behavior_state)

    def behavior_factory(env_cfg):
        return PhysicalHeadPlanner(
            behavior_model,
            str(args.behavior_variant),
            env_cfg,
            policy_weight=float(args.policy_score_weight),
            q_weight=float(args.q_score_weight),
        )

    targets = collect_targets(collect_args, exact_args, args.out, behavior_factory=behavior_factory)
    report = {
        "out": str(args.out),
        "targets": len(targets),
        "behavior_state": str(args.behavior_state),
        "behavior_variant": str(args.behavior_variant),
        "initials": str(args.initials),
        "rates": str(args.rates),
        "train_seeds": str(args.train_seeds),
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
