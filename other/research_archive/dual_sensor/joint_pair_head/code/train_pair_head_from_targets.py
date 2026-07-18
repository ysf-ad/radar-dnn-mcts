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

from alphazero_orthodox import load_targets
from two_sensor_physical_head_eval import make_physical_model, train_head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--init-state", required=True)
    ap.add_argument("--save-state", required=True)
    ap.add_argument("--train-steps", type=int, default=260)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--q-loss-weight", type=float, default=0.15)
    ap.add_argument("--value-loss-weight", type=float, default=0.0)
    ap.add_argument("--pair-loss-weight", type=float, default=0.65)
    ap.add_argument("--pair-loss-mode", choices=["sampled", "factorized"], default="sampled")
    ap.add_argument("--freeze-base", action="store_true")
    ap.add_argument("--model-seed", type=int, default=123)
    ap.add_argument("--log-every", type=int, default=40)
    ap.add_argument("--use-arrival-token-feature", action="store_true")
    ap.add_argument("--cell-balanced-sampling", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(int(args.model_seed))
    np.random.seed(int(args.model_seed))
    torch.set_num_threads(1)

    train_args = SimpleNamespace(
        d_model=48,
        nhead=4,
        nlayers=2,
        lr=float(args.lr),
        train_steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        model_seed=int(args.model_seed),
        q_loss_weight=float(args.q_loss_weight),
        value_loss_weight=float(args.value_loss_weight),
        pair_loss_weight=float(args.pair_loss_weight),
        pair_loss_mode=str(args.pair_loss_mode),
        use_arrival_token_feature=bool(args.use_arrival_token_feature),
        search_calibration_weight=0.0,
        log_every=max(1, int(args.log_every)),
        cell_balanced_sampling=bool(args.cell_balanced_sampling),
    )

    targets = load_targets(Path(args.targets))
    model = make_physical_model("two_row_pair_action_attention", train_args)
    state = torch.load(args.init_state, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print({"loaded_init": args.init_state, "missing": list(missing), "unexpected": list(unexpected), "targets": len(targets)}, flush=True)
    if bool(args.freeze_base):
        trainable = []
        for name, param in model.named_parameters():
            keep = name.startswith("pair_policy_head.") or name.startswith("pair_q_head.")
            param.requires_grad = bool(keep)
            if keep:
                trainable.append(name)
        print({"freeze_base": True, "trainable": trainable}, flush=True)
    model = train_head("two_row_pair_action_attention", targets, train_args, torch.device("cpu"), model=model)
    out = Path(args.save_state)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print({"saved_state": str(out), "targets": len(targets)}, flush=True)


if __name__ == "__main__":
    main()
