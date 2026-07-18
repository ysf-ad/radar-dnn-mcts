from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
EVAL = ROOT / "CreateValid1" / "experiments" / "code" / "model_code" / "eval_action_attention_muzero_g.py"
DEFAULT_G = ROOT / "CreateValid1" / "results" / "action_attention_muzero_g_direct_service005_balanced.pt"


BASE_ARGS = [
    "--base-state",
    str(ROOT / "CreateValid1" / "results" / "mixed_gate_distill_180_action_attention_step40_state.pt"),
    "--variant",
    "two_row_action_attention",
    "--lean-base-load",
    "--env-mode",
    "pufferlib_service",
    "--search-frame-overdue-weight",
    "0.5",
    "--search-frame-drop-penalty",
    "8",
    "--initials",
    "{initials}",
    "--rates",
    "{rates}",
    "--seeds",
    "{seeds}",
    "--windows",
    "{windows}",
    "--torch-threads",
    "8",
    "--torch-interop-threads",
    "1",
    "--use-root-seq-policy",
    "--root-seq-step-context",
    "--root-seq-decode-topk",
    "2",
    "--root-seq-min-steps",
    "8",
    "--max-steps",
    "32",
    "--max-window-search-frac",
    "{max_window_search_frac}",
    "--service-sort-plan",
    "--service-sort-search-prefix",
    "1",
    "--search-bias",
    "-10",
    "--policy-weight",
    "1.0",
    "--q-weight",
    "1.0",
    "--per-sensor-top",
    "3",
    "--decode-router-active-threshold",
    "35",
    "--decode-router-low-service-track",
    "0.5",
    "--decode-router-low-service-search",
    "0",
    "--decode-router-low-search-cap",
    "0.45",
    "--decode-router-high-active-threshold",
    "55",
    "--decode-router-high-service-track",
    "0.75",
    "--decode-router-high-service-search",
    "0",
    "--decode-router-high-search-cap",
    "0.45",
]


CASES = [
    ("base_corrected", []),
    (
        "grid_penalty",
        ["--search-frame-state-penalty-weight", "2.0"],
    ),
    (
        "grid_delta",
        ["--search-frame-state-penalty-weight", "2.0", "--search-frame-delta-reward-weight", "5.0"],
    ),
    (
        "service_delta",
        [
            "--search-frame-state-penalty-weight",
            "2.0",
            "--search-frame-delta-reward-weight",
            "5.0",
            "--service-pressure-delta-reward-weight",
            "0.30",
            "--serviced-pressure-delta-reward-weight",
            "0.15",
        ],
    ),
    (
        "update_reward",
        [
            "--search-frame-state-penalty-weight",
            "2.0",
            "--search-frame-delta-reward-weight",
            "5.0",
            "--serviced-target-update-reward-weight",
            "0.05",
        ],
    ),
    (
        "bounded_service",
        [
            "--search-frame-state-penalty-weight",
            "2.0",
            "--search-frame-delta-reward-weight",
            "5.0",
            "--service-pressure-delta-reward-weight",
            "0.30",
            "--bounded-service-reward-weight",
            "0.25",
            "--serviced-target-update-reward-weight",
            "0.05",
            "--serviced-pressure-delta-reward-weight",
            "0.15",
            "--discovered-target-reward",
            "0.08",
        ],
    ),
    (
        "raw_serviced_count",
        [
            "--search-frame-state-penalty-weight",
            "2.0",
            "--search-frame-delta-reward-weight",
            "5.0",
            "--service-pressure-delta-reward-weight",
            "0.30",
            "--serviced-target-reward-weight",
            "0.05",
            "--serviced-pressure-delta-reward-weight",
            "0.15",
            "--discovered-target-reward",
            "0.08",
        ],
    ),
    (
        "serviced_pressure_improvement",
        [
            "--search-frame-state-penalty-weight",
            "2.0",
            "--search-frame-delta-reward-weight",
            "5.0",
            "--service-pressure-delta-reward-weight",
            "0.30",
            "--serviced-pressure-improvement-reward-weight",
            "0.15",
            "--discovered-target-reward",
            "0.08",
        ],
    ),
    (
        "tracked_ms",
        [
            "--search-frame-state-penalty-weight",
            "2.0",
            "--search-frame-delta-reward-weight",
            "5.0",
            "--tracked-target-ms-reward-weight",
            "0.01",
        ],
    ),
    (
        "on_time_ms",
        [
            "--search-frame-state-penalty-weight",
            "2.0",
            "--search-frame-delta-reward-weight",
            "5.0",
            "--on-time-target-ms-reward-weight",
            "0.01",
        ],
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g-state", default=str(DEFAULT_G))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--initials", default="60")
    ap.add_argument("--rates", default="4")
    ap.add_argument("--seeds", default="916")
    ap.add_argument("--windows", default="30")
    ap.add_argument("--max-window-search-frac", default="0.25")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, extra in CASES:
        out = out_dir / f"{name}.csv"
        replacements = {
            "{initials}": str(args.initials),
            "{rates}": str(args.rates),
            "{seeds}": str(args.seeds),
            "{windows}": str(args.windows),
            "{max_window_search_frac}": str(args.max_window_search_frac),
        }
        base_args = [replacements.get(str(x), str(x)) for x in BASE_ARGS]
        cmd = [
            args.python,
            str(EVAL),
            "--g-state",
            str(args.g_state),
            *base_args,
            *extra,
            "--device",
            str(args.device),
            "--out",
            str(out),
        ]
        subprocess.run(cmd, check=True)
        summary = pd.read_csv(out.with_name(out.stem + "_summary.csv"))
        row = summary.iloc[0].to_dict()
        row["case"] = name
        rows.append(row)

    keys = [
        "case",
        "reward_per_window",
        "final_cumulative",
        "drop_pct_active",
        "tracked_targets",
        "mean_delay_active",
        "search_fraction",
        "planning_ms_per_window",
        "executed_actions",
    ]
    with (out_dir / "reward_term_sweep_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})
    print(out_dir / "reward_term_sweep_summary.csv")


if __name__ == "__main__":
    main()
