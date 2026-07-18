from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from alphazero_orthodox import base_exact_args, joint_sensor_log_probs, load_target_paths
from exact_env_mutual import load_model


def load_state_into(model, path: str, device) -> None:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state, strict=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    exact_args = base_exact_args(
        SimpleNamespace(
            ckpt=args.ckpt,
            device=args.device,
            windows=20,
            max_targets_per_episode=64,
            rollouts=1,
            c_puct=1.25,
            expand_top_k=4,
            horizon_windows=2,
            prior_uniform_mix=0.03,
            root_dirichlet_alpha=0.3,
            root_dirichlet_frac=0.0,
            leaf_value_mix=0.5,
            head_mode="pv",
            q_utility_weight=0.0,
            q_utility_normalize=False,
            prior_q_beta=0.0,
            q_scale=100.0,
            self_play_sample_tau=0.0,
            gamma=0.99,
            env_mode="radarxs_mission_delta",
            track_loss_penalty=4.0,
            target_service_weight=10.0,
            target_service_horizon_ms=3000.0,
            sector_staleness_weight=0.01,
            search_frame_overdue_weight=0.01,
            search_frame_drop_penalty=4.0,
            enable_x_band=True,
        )
    )
    device = torch.device(args.device)
    model = load_model(exact_args).to(device)
    load_state_into(model, args.state, device)
    model.eval()
    targets = [t for t in load_target_paths(args.targets) if getattr(t, "sensor_pi", None) is not None and np.sum(t.sensor_pi) > 0]
    rows = []
    with torch.inference_mode():
        for i in range(0, len(targets), 128):
            batch = targets[i : i + 128]
            x = torch.from_numpy(np.stack([t.x for t in batch]).astype(np.float32)).to(device)
            slot = torch.from_numpy(np.stack([t.slot for t in batch]).astype(np.float32)).to(device)
            target = torch.from_numpy(np.stack([t.sensor_pi for t in batch]).astype(np.float32)).to(device)
            target = target / target.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
            tl, tr, value, _, _, sl, _ = model.forward_with_sensor(x, slot)
            logp = joint_sensor_log_probs(tl, tr, sl)
            p = logp.exp()
            kl = (target * (target.clamp_min(1e-12).log() - logp)).sum(dim=(1, 2))
            rows.append(
                pd.DataFrame(
                    {
                        "target_search": target[:, 0, :].sum(dim=1).cpu().numpy(),
                        "pred_search": p[:, 0, :].sum(dim=1).cpu().numpy(),
                        "joint_kl": kl.cpu().numpy(),
                        "value": value.cpu().numpy(),
                    }
                )
            )
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary = df.agg(["mean", "std", "min", "max"]) if not df.empty else pd.DataFrame()
    print(summary.to_string(), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        summary.to_csv(str(Path(args.out).with_suffix("")) + "_summary.csv")


if __name__ == "__main__":
    main()
