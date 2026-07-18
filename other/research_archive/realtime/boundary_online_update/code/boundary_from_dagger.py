"""Convert committed PUCT trajectories into causal boundary-prediction rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from boundary_dataset import BoundaryRecord, save_records
from exact_env_mutual import MAXT


def load_windows(path: Path):
    import __main__

    from train_sonly_puct_dagger_ar import DaggerWindow

    __main__.DaggerWindow = DaggerWindow
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["windows"] if isinstance(payload, dict) else payload


def action_id(pair: np.ndarray) -> int:
    """Return the S-only action id used by the simulator interface."""
    encoded = int(pair[0])
    if encoded < 0:
        return 0
    row = max(0, min(MAXT, encoded // 2))
    return int(row * 2)


def convert(path: Path, midpoint_ms: float, max_suffix: int) -> list[BoundaryRecord]:
    records: list[BoundaryRecord] = []
    for episode, window in enumerate(load_windows(path)):
        state = getattr(window, "state_tokens", None)
        next_state = getattr(window, "next_state_tokens", None)
        if state is None or next_state is None:
            continue
        mask = np.asarray(window.mask, dtype=np.bool_)
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            continue
        elapsed = np.asarray(window.slots, dtype=np.float32)[:, 0] * 200.0
        candidates = valid[elapsed[valid] >= float(midpoint_ms)]
        midpoint = int(candidates[0] if candidates.size else valid[max(0, valid.size // 2)])
        suffix_steps = valid[valid >= midpoint]
        if suffix_steps.size == 0:
            continue
        suffix_steps = suffix_steps[:max_suffix]
        suffix = np.zeros((max_suffix,), dtype=np.int64)
        suffix[: suffix_steps.size] = [action_id(window.target_pairs[index]) for index in suffix_steps]
        meta = dict(getattr(window, "meta", {}) or {})
        records.append(
            BoundaryRecord(
                midpoint_tokens=np.asarray(state[midpoint], dtype=np.float32).copy(),
                suffix_actions=suffix,
                suffix_length=int(suffix_steps.size),
                remaining_time_ms=max(0.0, 200.0 - float(elapsed[midpoint])),
                boundary_tokens=np.asarray(next_state[int(suffix_steps[-1])], dtype=np.float32).copy(),
                seed=int(meta.get("seed", 0)),
                episode=int(episode),
                midpoint_step=int(midpoint),
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--midpoint-ms", type=float, default=100.0)
    parser.add_argument("--max-suffix", type=int, default=20)
    args = parser.parse_args()
    records = convert(args.dataset, args.midpoint_ms, args.max_suffix)
    if not records:
        raise RuntimeError("no boundary rows could be extracted; dataset needs state_tokens and next_state_tokens")
    save_records(
        args.out,
        records,
        metadata={
            "source": str(args.dataset),
            "midpoint_ms": float(args.midpoint_ms),
            "semantics": "actual committed suffix; future boundary state is label only",
        },
    )
    seeds = sorted({record.seed for record in records})
    print({"saved": str(args.out), "records": len(records), "seeds": seeds})


if __name__ == "__main__":
    main()
