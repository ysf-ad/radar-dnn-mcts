from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def add_code_root(code_root: Path) -> None:
    sys.path.insert(0, str(code_root))
    package_dir = code_root / "radar_dnn_mcts"
    if package_dir.exists():
        sys.path.insert(0, str(package_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--initials", default="20,40,60")
    parser.add_argument("--rates", default="2,3,4")
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--window-ms", type=int, default=200)
    parser.add_argument("--preset", default="repaired_stress")
    args = parser.parse_args()

    add_code_root(args.code_root.resolve())

    from final_radar_campaign import MAXT, run_fixed, summarize_window_df
    from repaired_campaign_tools import EDFPlanner, ESTPlanner, env_preset_cfg

    env_cfg = env_preset_cfg(args.preset)
    rows = []
    for initial in [int(x) for x in args.initials.split(",") if x.strip()]:
        for rate in [float(x) for x in args.rates.split(",") if x.strip()]:
            cfg = dict(env_cfg)
            cfg["poisson_rate_per_second"] = float(rate)
            for name, planner in {
                "EDF": EDFPlanner(MAXT),
                "EST": ESTPlanner(MAXT),
            }.items():
                windows, _actions = run_fixed(
                    planner,
                    name,
                    initial,
                    MAXT,
                    int(args.seed),
                    int(args.windows),
                    int(args.window_ms),
                    cfg,
                )
                summary = summarize_window_df(windows, "fixed")
                rows.append(
                    {
                        "method": name,
                        "initial": initial,
                        "rate": rate,
                        "seed": int(args.seed),
                        "windows": int(args.windows),
                        "reward_per_window": float(summary.get("reward_per_200ms_eq", float("nan"))),
                        "final_cumulative_reward": float(summary.get("final_cumulative_reward", float("nan"))),
                        "drop_rate": float(summary.get("drop_rate", float("nan"))),
                        "tracked_targets": float(summary.get("final_tracked_targets", float("nan"))),
                        "active_targets": float(summary.get("final_active_targets", float("nan"))),
                        "search_fraction": float(summary.get("search_fraction", float("nan"))),
                    }
                )
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["method", "initial", "rate"]).reset_index(drop=True)
    df.to_csv(out, index=False)
    print(json.dumps({"out": str(out), "rows": len(df)}, indent=2))


if __name__ == "__main__":
    main()
