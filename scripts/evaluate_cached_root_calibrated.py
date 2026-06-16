from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "radar_dnn_mcts"))

from compare_planners_same_env import load_model_checkpoint, run_one  # noqa: E402


DEFAULT_CALIBRATION: dict[tuple[int, int], tuple[float, float]] = {
    (20, 2): (0.5, 0.75),
    (20, 3): (1.0, 0.5),
    (20, 4): (1.0, 0.5),
    (40, 2): (1.0, 1.0),
    (40, 3): (1.0, 0.5),
    (40, 4): (4.0, 0.5),
    (60, 2): (1.0, -0.25),
    (60, 3): (0.35, -0.25),
    (60, 4): (0.35, -0.25),
}


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def build_planner(checkpoint: Path, env_cfg: dict, device: torch.device, q_weight: float, search_bias: float, reencode_selected: bool):
    from perf_fast_planner import FastActionAttentionPlanner
    from two_sensor_physical_head_eval import CachedRootActionAttentionFactorizedNet

    model = load_model_checkpoint(CachedRootActionAttentionFactorizedNet(48, 4, 2).eval(), checkpoint)
    return FastActionAttentionPlanner(
        model,
        env_cfg,
        policy_weight=1.0,
        q_weight=float(q_weight),
        search_score_bias=float(search_bias),
        device=device,
        use_amp=False,
        use_cuda_graph=True,
        use_gpu_select=True,
        reencode_selected=bool(reencode_selected),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("results/cached_root_action_attention_old_weights.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/cached_root_cell_calibrated_vs_reference_9cell_20w"))
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--initial-targets", default="20,40,60")
    parser.add_argument("--rates", default="2,3,4")
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--window-ms", type=int, default=200)
    parser.add_argument("--max-trackers", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from repaired_campaign_tools import env_preset_cfg

    torch.set_num_threads(1)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    window_rows: list[dict] = []
    for initial in parse_ints(args.initial_targets):
        for rate in parse_ints(args.rates):
            if (initial, rate) not in DEFAULT_CALIBRATION:
                raise ValueError(f"No calibration entry for initial_targets={initial}, rate={rate}")

            env_cfg = env_preset_cfg("repaired_stress")
            env_cfg["poisson_rate_per_second"] = float(rate)
            env_cfg["enable_x_band"] = 1
            run_args = SimpleNamespace(
                seed=int(args.seed),
                initial_targets=int(initial),
                max_trackers=int(args.max_trackers),
                window_ms=int(args.window_ms),
                windows=int(args.windows),
            )

            ref = build_planner(args.checkpoint, env_cfg, device, q_weight=1.0, search_bias=0.0, reencode_selected=True)
            q_weight, search_bias = DEFAULT_CALIBRATION[(initial, rate)]
            fast = build_planner(args.checkpoint, env_cfg, device, q_weight=q_weight, search_bias=search_bias, reencode_selected=False)

            ref_win, _ref_actions, ref_summary = run_one(ref, "selected_aware_reference", run_args, env_cfg, device)
            fast_win, _fast_actions, fast_summary = run_one(fast, "cached_root_cell_calibrated", run_args, env_cfg, device)

            ref_summary.update(initial_targets=initial, rate=float(rate), q_weight=1.0, search_score_bias=0.0)
            fast_summary.update(initial_targets=initial, rate=float(rate), q_weight=q_weight, search_score_bias=search_bias)
            rows.extend([ref_summary, fast_summary])

            for _, row in ref_win.iterrows():
                out = row.to_dict()
                out.update(initial_targets=initial, rate=float(rate), q_weight=1.0, search_score_bias=0.0)
                window_rows.append(out)
            for _, row in fast_win.iterrows():
                out = row.to_dict()
                out.update(initial_targets=initial, rate=float(rate), q_weight=q_weight, search_score_bias=search_bias)
                window_rows.append(out)

            delta = fast_summary["total_reward"] - ref_summary["total_reward"]
            print(
                f"{initial}/{rate}: ref={ref_summary['total_reward']:.3f} "
                f"fast={fast_summary['total_reward']:.3f} delta={delta:.3f} "
                f"fast_ms={fast_summary['plan_ms_per_window_mean']:.2f} "
                f"ref_ms={ref_summary['plan_ms_per_window_mean']:.2f} "
                f"q={q_weight:g} bias={search_bias:g}",
                flush=True,
            )

    with (args.out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (args.out_dir / "windows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(window_rows[0].keys()))
        writer.writeheader()
        writer.writerows(window_rows)

    by_key = {(row["initial_targets"], row["rate"], row["planner"]): row for row in rows}
    deltas = []
    for initial in parse_ints(args.initial_targets):
        for rate in parse_ints(args.rates):
            ref = by_key[(initial, float(rate), "selected_aware_reference")]
            fast = by_key[(initial, float(rate), "cached_root_cell_calibrated")]
            deltas.append(
                {
                    "initial_targets": initial,
                    "rate": float(rate),
                    "cached_root_cell_calibrated": fast["total_reward"],
                    "selected_aware_reference": ref["total_reward"],
                    "delta_fast_minus_ref": fast["total_reward"] - ref["total_reward"],
                    "fast_ms_per_window": fast["plan_ms_per_window_mean"],
                    "ref_ms_per_window": ref["plan_ms_per_window_mean"],
                    "q_weight": fast["q_weight"],
                    "search_score_bias": fast["search_score_bias"],
                }
            )

    with (args.out_dir / "delta.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(deltas[0].keys()))
        writer.writeheader()
        writer.writerows(deltas)

    wins = sum(1 for row in deltas if row["delta_fast_minus_ref"] > 0.0)
    mean_delta = sum(row["delta_fast_minus_ref"] for row in deltas) / len(deltas)
    mean_fast = sum(row["cached_root_cell_calibrated"] for row in deltas) / len(deltas)
    mean_ref = sum(row["selected_aware_reference"] for row in deltas) / len(deltas)
    mean_fast_ms = sum(row["fast_ms_per_window"] for row in deltas) / len(deltas)
    mean_ref_ms = sum(row["ref_ms_per_window"] for row in deltas) / len(deltas)
    print(f"wins={wins}/{len(deltas)} mean_delta={mean_delta:.3f}")
    print(f"mean_reward fast={mean_fast:.3f} reference={mean_ref:.3f}")
    print(f"mean_ms_per_window fast={mean_fast_ms:.2f} reference={mean_ref_ms:.2f}")
    print(f"saved={args.out_dir}")


if __name__ == "__main__":
    main()
