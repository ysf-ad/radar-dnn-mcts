from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize warmed per-window latency from benchmark window CSVs.")
    ap.add_argument("--windows-csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--warmup-windows", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.windows_csv)
    if "method" not in df.columns and "planner" in df.columns:
        df["method"] = df["planner"].astype(str)
    if "planning_ms_per_decision" not in df.columns:
        if "planning_ms_per_window" in df.columns:
            df["planning_ms_per_decision"] = df["planning_ms_per_window"]
        elif "planning_ms" in df.columns:
            df["planning_ms_per_decision"] = df["planning_ms"]
        else:
            raise ValueError("No latency column found: expected planning_ms_per_decision, planning_ms_per_window, or planning_ms")
    if "window" not in df.columns:
        raise ValueError("No window column found")

    group_cols = [c for c in ["method", "initial", "rate", "seed"] if c in df.columns]
    warmed = df[df["window"].astype(int) >= int(args.warmup_windows)].copy()
    if warmed.empty:
        warmed = df.copy()
    rows = []
    for keys, sub in warmed.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        lat = sub["planning_ms_per_decision"].astype(float)
        row.update(
            windows=int(len(sub)),
            mean_ms=float(lat.mean()),
            median_ms=float(lat.median()),
            p90_ms=float(lat.quantile(0.90)),
            p95_ms=float(lat.quantile(0.95)),
            max_ms=float(lat.max()),
        )
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(group_cols if group_cols else ["mean_ms"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.round(4).to_string(index=False))
    print(args.out)


if __name__ == "__main__":
    main()
