from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "radar_dnn_mcts"))

from compare_planners_same_env import load_model_checkpoint, run_one  # noqa: E402


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("results/cached_root_action_attention_old_weights.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/cached_root_score_sweep"))
    parser.add_argument("--policy-weights", default="1")
    parser.add_argument("--q-weights", default="0,0.25,0.5,1,2,4")
    parser.add_argument("--search-biases", default="-2,-1,-0.5,0,0.5")
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--initial-targets", type=int, default=60)
    parser.add_argument("--rate", type=float, default=4.0)
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--window-ms", type=int, default=200)
    parser.add_argument("--max-trackers", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--gpu-select", action="store_true")
    parser.add_argument("--manual-action-coupler", action="store_true")
    args = parser.parse_args()

    from perf_fast_planner import FastActionAttentionPlanner
    from repaired_campaign_tools import env_preset_cfg
    from two_sensor_physical_head_eval import CachedRootActionAttentionFactorizedNet

    torch.set_num_threads(1)
    device = torch.device(args.device)
    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for policy_weight in parse_floats(args.policy_weights):
        for q_weight in parse_floats(args.q_weights):
            for search_bias in parse_floats(args.search_biases):
                model = load_model_checkpoint(CachedRootActionAttentionFactorizedNet(48, 4, 2).eval(), args.checkpoint)
                planner = FastActionAttentionPlanner(
                    model,
                    env_cfg,
                    policy_weight=float(policy_weight),
                    q_weight=float(q_weight),
                    search_score_bias=float(search_bias),
                    device=device,
                    use_amp=bool(args.amp),
                    use_cuda_graph=bool(args.graph),
                    use_gpu_select=bool(args.gpu_select),
                    use_manual_action_coupler=bool(args.manual_action_coupler),
                )
                run_args = argparse.Namespace(
                    seed=int(args.seed),
                    initial_targets=int(args.initial_targets),
                    max_trackers=int(args.max_trackers),
                    window_ms=int(args.window_ms),
                    windows=int(args.windows),
                )
                _win, _act, summary = run_one(
                    planner,
                    f"cached_pw{policy_weight:g}_qw{q_weight:g}_sb{search_bias:g}",
                    run_args,
                    env_cfg,
                    device,
                )
                row = dict(summary)
                row.update(
                    policy_weight=float(policy_weight),
                    q_weight=float(q_weight),
                    search_score_bias=float(search_bias),
                    checkpoint=str(args.checkpoint),
                )
                rows.append(row)
                print(json.dumps(row), flush=True)

    df = pd.DataFrame(rows)
    summary_path = args.out_dir / "score_sweep_summary.csv"
    df.to_csv(summary_path, index=False)
    best = df.sort_values(["total_reward", "final_drop_pct_active"], ascending=[False, True]).head(10)
    best_path = args.out_dir / "score_sweep_top10.csv"
    best.to_csv(best_path, index=False)
    report = {
        "summary": str(summary_path),
        "top10": str(best_path),
        "best": best.iloc[0].to_dict() if not best.empty else {},
    }
    (args.out_dir / "score_sweep_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
