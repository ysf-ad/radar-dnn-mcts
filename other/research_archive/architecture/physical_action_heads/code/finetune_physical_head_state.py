from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "CreateValid1" / "experiments" / "code" / "model_code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from compare_action_heads_smoke import usable_targets
from two_sensor_physical_head_eval import make_physical_model, train_head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-state", default="")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--save-state", required=True)
    ap.add_argument("--variant", default="two_row_action_attention_factored_loss")
    ap.add_argument("--train-steps", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--model-seed", type=int, default=123)
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--policy-loss-weight", type=float, default=1.0)
    ap.add_argument("--q-loss-weight", type=float, default=0.0)
    ap.add_argument("--value-loss-weight", type=float, default=0.0)
    ap.add_argument("--freeze-non-q", action="store_true")
    ap.add_argument("--non-strict-load", action="store_true")
    ap.add_argument("--search-calibration-weight", type=float, default=0.0)
    ap.add_argument("--type-aux-loss-weight", type=float, default=0.25)
    ap.add_argument("--pair-loss-weight", type=float, default=0.65)
    ap.add_argument("--pair-loss-mode", choices=["sampled", "factorized"], default="sampled")
    ap.add_argument("--use-arrival-token-feature", action="store_true")
    ap.add_argument("--cell-balanced-sampling", action="store_true")
    ap.add_argument("--hard-policy-target", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    torch.set_num_threads(1)
    model = make_physical_model(str(args.variant), args)
    if str(args.base_state).strip():
        state = torch.load(args.base_state, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=not bool(args.non_strict_load))
        if bool(args.non_strict_load):
            print({"non_strict_load": True, "missing": list(missing), "unexpected": list(unexpected)}, flush=True)
        print({"loaded_base_state": str(args.base_state)}, flush=True)
    else:
        print({"loaded_base_state": None, "init": "scratch"}, flush=True)
    targets = usable_targets(Path(args.targets))
    tuned = train_head(str(args.variant), targets, args, torch.device("cpu"), model=model)
    out = Path(args.save_state)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tuned.state_dict(), out)
    print({"saved_state": str(out), "targets": len(targets)}, flush=True)


if __name__ == "__main__":
    main()
