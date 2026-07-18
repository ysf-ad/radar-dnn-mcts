from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_sonly_puct_dagger_ar as train_mod  # noqa: E402


def window_search_stats(window) -> tuple[int, int, float]:
    mask = np.asarray(window.mask).astype(bool)
    type_probs = np.asarray(window.type_probs)
    if type_probs.ndim != 2 or type_probs.shape[1] < 2:
        raise ValueError("expected type_probs with shape [steps, 2]")
    types = np.argmax(type_probs, axis=1)
    valid = mask[: types.shape[0]]
    n = int(valid.sum())
    s = int(((types == 0) & valid).sum())
    return s, n, float(s / max(1, n))


def greedy_select(windows, target_frac: float, min_windows: int, max_windows: int) -> list[int]:
    stats = [window_search_stats(w) for w in windows]
    order = sorted(range(len(windows)), key=lambda i: abs(stats[i][2] - target_frac))
    selected: list[int] = []
    search = 0
    total = 0
    for i in order:
        if len(selected) >= max_windows:
            break
        s, n, _frac = stats[i]
        if n <= 0:
            continue
        if len(selected) < min_windows:
            selected.append(i)
            search += s
            total += n
            continue
        cur = search / max(1, total)
        nxt = (search + s) / max(1, total + n)
        if abs(nxt - target_frac) <= abs(cur - target_frac):
            selected.append(i)
            search += s
            total += n
    if len(selected) < min_windows:
        for i in order:
            if i in selected:
                continue
            s, n, _frac = stats[i]
            if n <= 0:
                continue
            selected.append(i)
            search += s
            total += n
            if len(selected) >= min_windows:
                break
    return selected


def exhaustive_select_small(windows, target_frac: float, choose: int) -> list[int] | None:
    if len(windows) > 30 or choose <= 0 or choose > len(windows):
        return None
    stats = [window_search_stats(w) for w in windows]
    best = None
    best_err = float("inf")
    for combo in itertools.combinations(range(len(windows)), choose):
        s = sum(stats[i][0] for i in combo)
        n = sum(stats[i][1] for i in combo)
        if n <= 0:
            continue
        err = abs((s / n) - target_frac)
        if err < best_err:
            best_err = err
            best = list(combo)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-search-frac", type=float, default=0.37)
    ap.add_argument("--min-windows", type=int, default=12)
    ap.add_argument("--max-windows", type=int, default=45)
    ap.add_argument("--exact-windows", type=int, default=0)
    args = ap.parse_args()

    # Datasets created by train_sonly_puct_dagger_ar pickle DaggerWindow as __main__.
    import __main__

    __main__.DaggerWindow = train_mod.DaggerWindow
    ckpt = torch.load(str(args.inp), map_location="cpu", weights_only=False)
    windows = ckpt["windows"] if isinstance(ckpt, dict) and "windows" in ckpt else ckpt
    if not isinstance(windows, list) or not windows:
        raise RuntimeError(f"no windows found in {args.inp}")

    selected = exhaustive_select_small(windows, float(args.target_search_frac), int(args.exact_windows))
    if selected is None:
        selected = greedy_select(
            windows,
            target_frac=float(args.target_search_frac),
            min_windows=int(args.min_windows),
            max_windows=int(args.max_windows),
        )
    out_windows = [windows[i] for i in selected]
    s = 0
    n = 0
    per = []
    for w in out_windows:
        sw, nw, fw = window_search_stats(w)
        s += sw
        n += nw
        per.append(fw)
    out = {
        "windows": out_windows,
        "meta": {
            **(ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}),
            "source_dataset": str(args.inp),
            "selected_indices": selected,
            "target_search_frac": float(args.target_search_frac),
            "actual_search_frac": float(s / max(1, n)),
            "mean_window_search_frac": float(np.mean(per) if per else 0.0),
            "selected_windows": len(out_windows),
            "valid_steps": int(n),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(args.out))
    print(out["meta"], flush=True)


if __name__ == "__main__":
    main()
