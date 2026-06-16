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


def load_state_if_present(model, path: Path | None):
    if path is None or not str(path).strip():
        return model
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    return model, [str(x) for x in missing], [str(x) for x in unexpected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, default=Path("../CreateValid1/results/clean9_replace20r3_edfbehavior_targets.pt"))
    parser.add_argument("--init-state", type=Path, default=Path("../CreateValid1/results/critic_bootstrap_medium_eval_two_row_action_attention_qpolicy_factored_loss.pt"))
    parser.add_argument("--out", type=Path, default=Path("results/cached_root_action_attention_finetuned.pt"))
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--model-seed", type=int, default=123)
    parser.add_argument("--cell-balanced-sampling", action="store_true")
    parser.add_argument("--search-calibration-weight", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    from compare_action_heads_smoke import usable_targets
    from two_sensor_physical_head_eval import CachedRootActionAttentionFactorizedNet, train_head

    torch.manual_seed(int(args.model_seed))
    np.random.seed(int(args.model_seed))
    torch.set_num_threads(1)
    targets = usable_targets(args.targets)
    model = CachedRootActionAttentionFactorizedNet(48, 4, 2).eval()
    missing = []
    unexpected = []
    if args.init_state:
        loaded = load_state_if_present(model, args.init_state)
        if isinstance(loaded, tuple):
            model, missing, unexpected = loaded
    train_args = SimpleNamespace(
        d_model=48,
        nhead=4,
        nlayers=2,
        train_steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        q_loss_weight=0.25,
        value_loss_weight=0.25,
        search_calibration_weight=float(args.search_calibration_weight),
        log_every=int(args.log_every),
        model_seed=int(args.model_seed),
        cell_balanced_sampling=bool(args.cell_balanced_sampling),
        hard_policy_target=False,
    )
    trained = train_head(
        "cached_root_action_attention_qpolicy_factored_loss",
        targets,
        train_args,
        torch.device("cpu"),
        model=model,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained.state_dict(), args.out)
    report = {
        "targets": str(args.targets),
        "init_state": str(args.init_state) if args.init_state else "",
        "out": str(args.out),
        "train_steps": int(args.train_steps),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "missing_init_keys": missing,
        "unexpected_init_keys": unexpected,
    }
    report_path = args.out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
