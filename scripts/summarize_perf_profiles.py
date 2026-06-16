from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _planner_rows(profile: dict) -> list[tuple[str, float, float, float]]:
    rows = []
    for planner in profile.get("planners", []):
        timing = planner.get("stage_timing", {})
        name = str(planner.get("planner"))
        plan_stage = timing.get(f"{name}.planner_plan", {})
        exec_stage = timing.get("execute_plan_until_budget", {})
        rows.append(
            (
                name,
                float(plan_stage.get("mean_ms", 0.0)),
                float(exec_stage.get("mean_ms", 0.0)),
                float(planner.get("total_reward", 0.0)),
            )
        )
    return rows


def _top_stage_rows(profile: dict, key: str, limit: int) -> list[tuple[str, float, float]]:
    rows = []
    for name, item in profile.get(key, {}).items():
        rows.append(
            (
                str(name),
                float(item.get("mean_ms", 0.0)),
                float(item.get("mean_percent_of_profiled_steps", 0.0)),
            )
        )
    rows.sort(key=lambda row: row[1], reverse=True)
    return rows[:limit]


def _cached_rows(profile: dict) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for item in profile.get("prefix_batches", []):
        stages = item.get("stages", {})
        rows.append(
            (
                int(item.get("prefixes", 0)),
                float(stages.get("full_cached_score", {}).get("mean_ms", 0.0)),
                float(stages.get("sensor_coupling", {}).get("mean_ms", 0.0)),
                float(stages.get("action_self_attention", {}).get("mean_ms", 0.0)),
                float(stages.get("target_heads", {}).get("mean_ms", 0.0))
                + float(stages.get("type_heads", {}).get("mean_ms", 0.0))
                + float(stages.get("residual_heads", {}).get("mean_ms", 0.0)),
            )
        )
    return rows


def _write_table(lines: list[str], headers: tuple[str, ...], rows: list[tuple]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        formatted = []
        for value in row:
            if isinstance(value, float):
                formatted.append(f"{value:.3f}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", type=Path, required=True)
    parser.add_argument("--root-table", type=Path, required=True)
    parser.add_argument("--cached", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/perf_profile_summary.md"))
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    online = _load(args.online)
    root_table = _load(args.root_table)
    cached = _load(args.cached)

    lines = [
        "# Performance Profile Summary",
        "",
        "## Online Episode",
        "",
    ]
    _write_table(lines, ("planner", "plan ms/window", "execute ms/window", "total reward"), _planner_rows(online))

    lines.extend(["## Batched Root Table Stages", ""])
    _write_table(lines, ("stage", "mean ms", "profile %"), _top_stage_rows(root_table, "stage_profile", int(args.top)))

    lines.extend(["## Cached Action-Attention Internals", ""])
    _write_table(
        lines,
        ("prefix batch", "score ms", "sensor coupling ms", "action attention ms", "heads ms"),
        _cached_rows(cached),
    )

    lines.extend(
        [
            "## Main Opportunities",
            "",
            "1. Batch MCTS/root expansion so model scoring sees prefix batches instead of one scalar decision at a time.",
            "2. Keep using batched feature construction; legacy per-state tokenization dominates large root batches.",
            "3. Treat custom kernels as a later step. Current bottlenecks are transformer/head dispatch and Python search structure, not one obvious scalar loop.",
            "",
        ]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
