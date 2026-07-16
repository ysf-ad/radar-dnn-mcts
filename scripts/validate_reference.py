from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def aggregate(windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in windows.groupby("method"):
        rows.append(
            {
                "method": method,
                "reward_per_window": group["window_reward"].mean(),
                "drop_pct_active": group["drop_pct_active"].mean(),
                "tracked_targets": group["tracked_targets"].mean(),
                "mean_delay_active": group["mean_delay_active"].mean(),
                "search_fraction": group["search_fraction"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-9)
    args = parser.parse_args()
    actual = aggregate(pd.read_csv(args.windows))
    expected = pd.read_csv(args.expected).rename(columns={"mean_delay_active_ms": "mean_delay_active"})
    columns = [column for column in actual.columns if column in expected.columns]
    expected = expected[columns].sort_values("method").reset_index(drop=True)
    actual = actual[columns]
    if actual["method"].tolist() != expected["method"].tolist():
        raise SystemExit("method sets differ")
    numeric = [column for column in columns if column != "method"]
    differences = (actual[numeric] - expected[numeric]).abs()
    maximum = float(np.nanmax(differences.to_numpy()))
    if maximum > args.atol:
        raise SystemExit(f"reference mismatch: max absolute difference={maximum}")
    print(f"reference validation passed: methods={len(actual)}, max_abs_diff={maximum:.3g}")


if __name__ == "__main__":
    main()
