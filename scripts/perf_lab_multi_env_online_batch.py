from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "radar_dnn_mcts"))
_FAST_STEP_BINDING = None
_FAST_STEP_SEARCH_DWELL_MS = None
_CUDA_EVENT_STAGE_BUCKETS: dict[str, list[float]] | None = None


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _get_sdp_state() -> dict[str, bool]:
    cuda = torch.backends.cuda
    return {
        "flash": bool(cuda.flash_sdp_enabled()),
        "mem_efficient": bool(cuda.mem_efficient_sdp_enabled()),
        "math": bool(cuda.math_sdp_enabled()),
        "cudnn": bool(cuda.cudnn_sdp_enabled()) if hasattr(cuda, "cudnn_sdp_enabled") else False,
    }


def _set_sdp_state(state: dict[str, bool]) -> None:
    cuda = torch.backends.cuda
    cuda.enable_flash_sdp(bool(state.get("flash", False)))
    cuda.enable_mem_efficient_sdp(bool(state.get("mem_efficient", False)))
    cuda.enable_math_sdp(bool(state.get("math", False)))
    if hasattr(cuda, "enable_cudnn_sdp"):
        cuda.enable_cudnn_sdp(bool(state.get("cudnn", False)))


@contextmanager
def sdp_backend(name: str):
    old = _get_sdp_state()
    variants = {
        "default": old,
        "math_only": {"flash": False, "mem_efficient": False, "math": True, "cudnn": False},
        "flash_only": {"flash": True, "mem_efficient": False, "math": False, "cudnn": False},
        "mem_efficient_only": {"flash": False, "mem_efficient": True, "math": False, "cudnn": False},
        "cudnn_only": {"flash": False, "mem_efficient": False, "math": False, "cudnn": True},
        "flash_math": {"flash": True, "mem_efficient": False, "math": True, "cudnn": False},
        "all_no_cudnn": {"flash": True, "mem_efficient": True, "math": True, "cudnn": False},
    }
    if name not in variants:
        raise ValueError(f"unknown SDP backend variant: {name}")
    if name != "default":
        _set_sdp_state(variants[name])
    try:
        yield _get_sdp_state()
    finally:
        _set_sdp_state(old)


def stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "calls": int(arr.size),
        "total_ms": float(arr.sum()) if arr.size else 0.0,
        "mean_ms": float(arr.mean()) if arr.size else 0.0,
        "p50_ms": float(np.percentile(arr, 50)) if arr.size else 0.0,
        "p90_ms": float(np.percentile(arr, 90)) if arr.size else 0.0,
        "p99_ms": float(np.percentile(arr, 99)) if arr.size else 0.0,
    }


def add_profile_stage(buckets: dict[str, list[float]], name: str, value_ms: float) -> None:
    buckets.setdefault(name, []).append(float(value_ms))


def profile_summary(buckets: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    total_profiled_ms = sum(float(np.asarray(values, dtype=np.float64).sum()) for values in buckets.values())
    rows = []
    for name, values in buckets.items():
        row = stats(values)
        row["pct_profiled_ms"] = float(100.0 * row["total_ms"] / max(total_profiled_ms, 1e-12))
        rows.append((name, row))
    return dict(sorted(rows, key=lambda item: item[1]["total_ms"], reverse=True))


def int_distribution(values: list[int], full_size: int | None = None) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.int64)
    if arr.size == 0:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "full_count": 0,
            "partial_count": 0,
            "full_fraction": 0.0,
            "histogram": {},
        }
    full = int(full_size) if full_size is not None else int(arr.max())
    unique, counts = np.unique(arr, return_counts=True)
    full_count = int(np.sum(arr == full))
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "full_count": full_count,
        "partial_count": int(arr.size - full_count),
        "full_fraction": float(full_count / max(1, int(arr.size))),
        "histogram": {str(int(k)): int(v) for k, v in zip(unique, counts)},
    }


def cprofile_top(profile: cProfile.Profile, limit: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (filename, line, func), stat in pstats.Stats(profile).stats.items():
        ccalls, ncalls, total_time, cumulative_time, _callers = stat
        rows.append(
            {
                "function": f"{Path(filename).name}:{line}:{func}",
                "ncalls": int(ncalls),
                "primitive_calls": int(ccalls),
                "total_ms": float(total_time * 1000.0),
                "cumulative_ms": float(cumulative_time * 1000.0),
            }
        )
    rows.sort(key=lambda item: float(item["cumulative_ms"]), reverse=True)
    return rows[: max(0, int(limit))]


def run_maybe_profiled(name: str, fn, limit: int) -> tuple[dict, list[dict[str, object]]]:
    if int(limit) <= 0:
        return fn(), []
    profile = cProfile.Profile()
    try:
        result = profile.runcall(fn)
    finally:
        profile.disable()
    return result, cprofile_top(profile, int(limit))


def load_model_checkpoint(model, checkpoint: str | Path | None):
    if checkpoint is None or str(checkpoint).strip() == "":
        return model
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state, strict=True)
    return model


def time_stage(device: torch.device, enabled: bool, buckets: dict[str, list[float]], name: str, fn):
    global _CUDA_EVENT_STAGE_BUCKETS
    if not enabled:
        return fn()
    sync(device)
    start_event = None
    end_event = None
    if _CUDA_EVENT_STAGE_BUCKETS is not None and device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    t0 = time.perf_counter()
    out = fn()
    if start_event is not None and end_event is not None:
        end_event.record()
        end_event.synchronize()
        add_profile_stage(_CUDA_EVENT_STAGE_BUCKETS, name, float(start_event.elapsed_time(end_event)))
    else:
        sync(device)
    add_profile_stage(buckets, name, (time.perf_counter() - t0) * 1000.0)
    return out


def make_live_slots(slot_template: np.ndarray, live_pos: list[int], elapsed, search_count, track_count, last, budget_ms: float) -> np.ndarray:
    slots = np.asarray(slot_template, dtype=np.float32)[np.asarray(live_pos, dtype=np.int64)].copy()
    if not live_pos:
        return slots
    elapsed_arr = np.asarray([elapsed[p] for p in live_pos], dtype=np.float32)
    search_arr = np.asarray([search_count[p] for p in live_pos], dtype=np.float32)
    track_arr = np.asarray([track_count[p] for p in live_pos], dtype=np.float32)
    last_arr = np.asarray([last[p] for p in live_pos], dtype=np.int32)
    slots[:, 0] = elapsed_arr / float(budget_ms)
    slots[:, 1] = search_arr / 20.0
    slots[:, 2] = track_arr / 100.0
    slots[:, 3] = (last_arr == 0).astype(np.float32)
    return slots


@dataclass
class PackedRootObs:
    observations: list[dict]
    t_desired: np.ndarray
    deadline: np.ndarray
    dwell: np.ndarray
    active: np.ndarray
    tracked: np.ndarray
    priority: np.ndarray
    ranges: np.ndarray
    grids: np.ndarray
    az_bin: np.ndarray
    el_bin: np.ndarray
    search_debt_ms: np.ndarray
    s_busy_ms: np.ndarray
    x_busy_ms: np.ndarray
    enable_x_band: np.ndarray
    sensor_id: np.ndarray
    use_grid_feature: np.ndarray
    use_arrival_feature: np.ndarray
    arrival_rate: np.ndarray


@dataclass
class PhysicalActionTemplate:
    actions: np.ndarray
    bases: np.ndarray
    sensors: np.ndarray
    valid: np.ndarray


def pack_root_observations(observations: list[dict], max_trackers: int) -> PackedRootObs:
    n = len(observations)
    zeros_t = np.zeros(max_trackers, dtype=np.float32)
    zeros_g = np.zeros(300, dtype=np.float32)
    t_desired = np.stack([np.asarray(obs["t_desired"], dtype=np.float32)[:max_trackers] for obs in observations], axis=0)
    deadline = np.stack([np.asarray(obs["t_deadline"], dtype=np.float32)[:max_trackers] for obs in observations], axis=0)
    dwell = np.stack([np.asarray(obs["t_dwell"], dtype=np.float32)[:max_trackers] for obs in observations], axis=0)
    active = np.stack([np.asarray(obs["active_mask"], dtype=bool)[:max_trackers] for obs in observations], axis=0)
    tracked = np.stack(
        [
            np.asarray(obs.get("tracked_mask", np.asarray(obs["active_mask"], dtype=bool) & (np.asarray(obs["t_deadline"]) > 0)), dtype=bool)[
                :max_trackers
            ]
            for obs in observations
        ],
        axis=0,
    )
    return PackedRootObs(
        observations=observations,
        t_desired=t_desired,
        deadline=deadline,
        dwell=dwell,
        active=active,
        tracked=tracked,
        priority=np.stack([np.asarray(obs.get("priority", zeros_t), dtype=np.float32)[:max_trackers] for obs in observations], axis=0),
        ranges=np.stack([np.asarray(obs.get("target_range", zeros_t), dtype=np.float32)[:max_trackers] for obs in observations], axis=0),
        grids=np.stack([np.asarray(obs.get("grid", zeros_g), dtype=np.float32)[:300] for obs in observations], axis=0)
        if n
        else np.zeros((0, 300), dtype=np.float32),
        az_bin=np.stack([np.asarray(obs.get("az_bin", zeros_t), dtype=np.float32)[:max_trackers] for obs in observations], axis=0),
        el_bin=np.stack([np.asarray(obs.get("el_bin", zeros_t), dtype=np.float32)[:max_trackers] for obs in observations], axis=0),
        search_debt_ms=np.asarray([float(obs.get("search_debt_ms", 0.0)) for obs in observations], dtype=np.float32),
        s_busy_ms=np.asarray([float(obs.get("s_band_busy_ms", 0.0)) for obs in observations], dtype=np.float32),
        x_busy_ms=np.asarray([float(obs.get("x_band_busy_ms", 0.0)) for obs in observations], dtype=np.float32),
        enable_x_band=np.asarray([float(obs.get("enable_x_band", 0.0)) for obs in observations], dtype=np.float32),
        sensor_id=np.asarray([float(obs.get("sensor_id", 0.0)) for obs in observations], dtype=np.float32),
        use_grid_feature=np.asarray([float(obs.get("use_grid_feature", 0.0)) for obs in observations], dtype=np.float32),
        use_arrival_feature=np.asarray([float(obs.get("use_arrival_feature", 0.0)) for obs in observations], dtype=np.float32),
        arrival_rate=np.asarray([float(obs.get("arrival_rate", 0.0)) for obs in observations], dtype=np.float32),
    )


def pack_root_envs_direct(envs, root_env_ids: list[int], search_debt: list[float], env_cfg: dict, max_trackers: int, aux_vec=None) -> PackedRootObs:
    from pufferlib.ocean.radarxs import binding
    from pufferlib.ocean.radarxs.engine import FEATURES_PER_TRACKER, GRID_SIZE, NO_TARGET

    n = len(root_env_ids)
    if n <= 0:
        return PackedRootObs(
            observations=[],
            t_desired=np.zeros((0, max_trackers), dtype=np.float32),
            deadline=np.zeros((0, max_trackers), dtype=np.float32),
            dwell=np.zeros((0, max_trackers), dtype=np.float32),
            active=np.zeros((0, max_trackers), dtype=bool),
            tracked=np.zeros((0, max_trackers), dtype=bool),
            priority=np.zeros((0, max_trackers), dtype=np.float32),
            ranges=np.zeros((0, max_trackers), dtype=np.float32),
            grids=np.zeros((0, GRID_SIZE), dtype=np.float32),
            az_bin=np.zeros((0, max_trackers), dtype=np.float32),
            el_bin=np.zeros((0, max_trackers), dtype=np.float32),
            search_debt_ms=np.zeros((0,), dtype=np.float32),
            s_busy_ms=np.zeros((0,), dtype=np.float32),
            x_busy_ms=np.zeros((0,), dtype=np.float32),
            enable_x_band=np.zeros((0,), dtype=np.float32),
            sensor_id=np.zeros((0,), dtype=np.float32),
            use_grid_feature=np.zeros((0,), dtype=np.float32),
            use_arrival_feature=np.zeros((0,), dtype=np.float32),
            arrival_rate=np.zeros((0,), dtype=np.float32),
        )

    obs_mat = np.stack([np.asarray(envs[i].obs_buf[0], dtype=np.float32) for i in root_env_ids], axis=0)
    grids = obs_mat[:, :GRID_SIZE]
    inferred = int((obs_mat.shape[1] - GRID_SIZE - 1) / max_trackers)
    features_per_tracker = inferred if inferred in (4, 6) else FEATURES_PER_TRACKER
    end_idx = GRID_SIZE + int(max_trackers) * int(features_per_tracker)
    trackers = obs_mat[:, GRID_SIZE:end_idx].reshape(n, int(max_trackers), int(features_per_tracker))
    t_desired = np.where(np.isfinite(trackers[:, :, 0]), trackers[:, :, 0], float(NO_TARGET)).astype(np.float32, copy=False)
    deadline = np.where(np.isfinite(trackers[:, :, 1]), trackers[:, :, 1], float(NO_TARGET)).astype(np.float32, copy=False)
    dwell = np.where(np.isfinite(trackers[:, :, 2]), trackers[:, :, 2], 10.0).astype(np.float32, copy=False)
    dwell = np.clip(dwell, 1.0, 2000.0)
    priority = np.where(np.isfinite(trackers[:, :, 3]), trackers[:, :, 3], 0.0).astype(np.float32, copy=False)
    if features_per_tracker >= 6:
        az_bin = np.where(np.isfinite(trackers[:, :, 4]), trackers[:, :, 4], 0.0).astype(np.float32, copy=False)
        el_bin = np.where(np.isfinite(trackers[:, :, 5]), trackers[:, :, 5], 0.0).astype(np.float32, copy=False)
        az_bin = np.clip(az_bin, 0.0, 1.0)
        el_bin = np.clip(el_bin, 0.0, 1.0)
    else:
        az_bin = np.zeros((n, max_trackers), dtype=np.float32)
        el_bin = np.zeros((n, max_trackers), dtype=np.float32)
    active = np.isfinite(t_desired) & (t_desired != float(NO_TARGET))
    tracked = active & (deadline > 0.0)
    sensor_id = np.zeros((n,), dtype=np.float32)
    if end_idx < obs_mat.shape[1]:
        raw_sensor = obs_mat[:, end_idx]
        sensor_id = np.where(np.isfinite(raw_sensor), raw_sensor, 0.0).astype(np.float32, copy=False)

    ranges = np.zeros((n, max_trackers), dtype=np.float32)
    s_busy_ms = np.zeros((n,), dtype=np.float32)
    x_busy_ms = np.zeros((n,), dtype=np.float32)
    enable_x_band = np.zeros((n,), dtype=np.float32)
    if hasattr(binding, "vec_aux_arrays") and hasattr(binding, "vec_view_firsts"):
        view = None
        try:
            use_existing_view = aux_vec is not None and len(root_env_ids) == len(envs) and all(int(v) == idx for idx, v in enumerate(root_env_ids))
            if use_existing_view:
                view = aux_vec
            else:
                view = binding.vec_view_firsts(*[envs[i].env for i in root_env_ids])
            aux = binding.vec_aux_arrays(view)
            s_busy_ms = np.asarray(aux["s_band_busy_ms"], dtype=np.float32)
            x_busy_ms = np.asarray(aux["x_band_busy_ms"], dtype=np.float32)
            enable_x_band = np.asarray(aux["enable_x_band"], dtype=np.float32)
            ranges = np.asarray(aux["target_range"], dtype=np.float32)
        finally:
            if view is not None and view is not aux_vec and hasattr(binding, "vec_release"):
                binding.vec_release(view)
    elif hasattr(binding, "vec_aux"):
        for row, env_idx in enumerate(root_env_ids):
            try:
                aux = binding.vec_aux(envs[env_idx].env)
            except Exception:
                continue
            s_busy_ms[row] = float(aux.get("s_band_busy_ms", 0.0))
            x_busy_ms[row] = float(aux.get("x_band_busy_ms", 0.0))
            enable_x_band[row] = float(aux.get("enable_x_band", 0.0))
            aux_ranges = np.asarray(aux.get("target_range", []), dtype=np.float32)
            if aux_ranges.size:
                ranges[row, : min(max_trackers, aux_ranges.size)] = aux_ranges[:max_trackers]

    observations = [None] * n
    return PackedRootObs(
        observations=observations,
        t_desired=t_desired,
        deadline=deadline,
        dwell=dwell,
        active=active,
        tracked=tracked,
        priority=priority,
        ranges=ranges,
        grids=grids,
        az_bin=az_bin,
        el_bin=el_bin,
        search_debt_ms=np.asarray([float(search_debt[i]) for i in root_env_ids], dtype=np.float32),
        s_busy_ms=s_busy_ms,
        x_busy_ms=x_busy_ms,
        enable_x_band=enable_x_band,
        sensor_id=sensor_id,
        use_grid_feature=np.ones((n,), dtype=np.float32),
        use_arrival_feature=np.ones((n,), dtype=np.float32),
        arrival_rate=np.full((n,), float(env_cfg.get("arrival_rate", env_cfg.get("poisson_rate_per_second", 0.0))), dtype=np.float32),
    )


def slot_template_from_packed(packed: PackedRootObs, budget_ms: float) -> np.ndarray:
    tracked = packed.active & (packed.deadline >= 0.0)
    workload = np.sum(np.where(tracked, packed.dwell, 0.0), axis=1) / max(1.0, float(budget_ms))
    positive_deadline = tracked & (packed.deadline > 0.0)
    min_deadline_arr = np.where(positive_deadline, packed.deadline, np.inf).min(axis=1)
    min_deadline_arr = np.where(np.isfinite(min_deadline_arr), min_deadline_arr, 0.0)
    arrival_feature = packed.enable_x_band.copy()
    use_arrival = packed.use_arrival_feature > 0.5
    arrival_feature = np.where(use_arrival, arrival_feature + np.clip(packed.arrival_rate / 10.0, 0.0, 2.0), arrival_feature)
    feat = np.empty((len(packed.observations), 11), dtype=np.float32)
    feat[:, 0] = 0.0
    feat[:, 1] = 0.0
    feat[:, 2] = 0.0
    feat[:, 3] = 0.0
    feat[:, 4] = np.sum(packed.active, axis=1).astype(np.float32) / 100.0
    feat[:, 5] = np.sum(tracked, axis=1).astype(np.float32) / 100.0
    feat[:, 6] = np.minimum(workload / 20.0, 2.0)
    feat[:, 7] = min_deadline_arr / 3000.0
    feat[:, 8] = np.clip(packed.s_busy_ms / 200.0, 0.0, 5.0)
    feat[:, 9] = np.clip(packed.x_busy_ms / 200.0, 0.0, 5.0)
    feat[:, 10] = arrival_feature
    if np.any(packed.use_grid_feature > 0.5):
        age = 3000.0 - packed.grids
        overdue = np.maximum(0.0, age - 3000.0) / 3000.0
        feat[:, 8] = np.where(packed.use_grid_feature > 0.5, np.clip(np.mean(overdue, axis=1), 0.0, 5.0), feat[:, 8])
        feat[:, 9] = np.where(packed.use_grid_feature > 0.5, np.clip(np.mean(age > 4500.0, axis=1), 0.0, 1.0), feat[:, 9])
        feat[:, 10] = np.where(packed.use_grid_feature > 0.5, np.clip(np.max(age, axis=1) / 4500.0, 0.0, 5.0), feat[:, 10])
    return feat


def tokenize_root_batch_fast(adapt, observations, max_trackers: int, token_dim: int) -> np.ndarray:
    """Root-only tokenizer for cached multi-env windows.

    Equivalent to ``tokenize_batch(adapt, observations)`` for the root call
    where no targets are selected and search_count is zero, but avoids selected
    set construction and a few optional per-row branches in the hot benchmark.
    """
    n = len(observations)
    x = np.zeros((n, int(max_trackers) + 1, int(token_dim)), dtype=np.float32)
    if n <= 0:
        return x

    t_desired = np.stack([np.asarray(obs["t_desired"], dtype=np.float32)[:max_trackers] for obs in observations], axis=0)
    deadline = np.stack([np.asarray(obs["t_deadline"], dtype=np.float32)[:max_trackers] for obs in observations], axis=0)
    dwell = np.stack([np.asarray(obs["t_dwell"], dtype=np.float32)[:max_trackers] for obs in observations], axis=0)
    active = np.stack([np.asarray(obs["active_mask"], dtype=bool)[:max_trackers] for obs in observations], axis=0)
    tracked = np.stack(
        [
            np.asarray(obs.get("tracked_mask", np.asarray(obs["active_mask"], dtype=bool) & (np.asarray(obs["t_deadline"]) > 0)), dtype=bool)[
                :max_trackers
            ]
            for obs in observations
        ],
        axis=0,
    )
    priority = np.stack(
        [np.asarray(obs.get("priority", np.zeros(max_trackers, dtype=np.float32)), dtype=np.float32)[:max_trackers] for obs in observations],
        axis=0,
    )
    ranges = np.stack(
        [np.asarray(obs.get("target_range", np.zeros(max_trackers, dtype=np.float32)), dtype=np.float32)[:max_trackers] for obs in observations],
        axis=0,
    )
    grids = [np.asarray(obs.get("grid", np.zeros((300,), dtype=np.float32)), dtype=np.float32) for obs in observations]
    grid_min = np.asarray([float(np.min(grid)) for grid in grids], dtype=np.float32)
    search_debt_ms = np.asarray([float(obs.get("search_debt_ms", 0.0)) for obs in observations], dtype=np.float32)
    pure = adapt.pure_mcts
    if float(pure.search_debt_penalty_weight) <= 0.0:
        search_penalty_norm = np.zeros((n,), dtype=np.float32)
    elif int(pure.search_delay_mode) == 0:
        search_penalty_norm = (float(pure.search_debt_penalty_weight) * search_debt_ms).astype(np.float32)
    else:
        arg = np.minimum(search_debt_ms / max(1e-3, float(pure.search_debt_tau_ms)), 20.0)
        search_penalty_norm = (float(pure.search_debt_penalty_weight) * (np.exp(arg) - 1.0)).astype(np.float32)
    if float(pure.search_delay_penalty_cap) >= 0.0:
        search_penalty_norm = np.minimum(search_penalty_norm, float(pure.search_delay_penalty_cap))
    search_penalty_norm = np.clip(np.where(search_debt_ms > 0.0, search_penalty_norm, 0.0), 0.0, 10.0).astype(np.float32)

    tracked_active = active & tracked
    tracked_n = np.sum(tracked_active, axis=1).astype(np.float32)
    tracked_delays = np.maximum(0.0, -t_desired) * tracked_active.astype(np.float32)
    tracked_delay_sum = np.sum(tracked_delays, axis=1)
    mean_tracked_delay_norm = np.divide(
        tracked_delay_sum,
        np.maximum(tracked_n, 1.0),
        out=np.zeros_like(tracked_delay_sum),
        where=tracked_n > 0,
    )
    mean_tracked_delay_norm = np.clip(mean_tracked_delay_norm / 2000.0, 0.0, 10.0)
    overdue_count = np.sum((t_desired < 0.0) & tracked_active, axis=1).astype(np.float32)
    overdue_frac = np.divide(overdue_count, np.maximum(tracked_n, 1.0), out=np.zeros_like(overdue_count), where=tracked_n > 0)
    global_tardiness_norm = np.clip(tracked_delay_sum / 20000.0, 0.0, 10.0)
    deadline_pressure = np.maximum(0.0, 100.0 - deadline) * tracked_active.astype(np.float32)
    global_deadline_pressure_norm = np.clip(np.sum(deadline_pressure, axis=1) / 2000.0, 0.0, 10.0)
    global_penalty_norm = np.clip(
        0.001
        * (
            float(pure.global_tardiness_weight) * global_tardiness_norm
            + float(pure.local_tardiness_weight) * mean_tracked_delay_norm
        ),
        0.0,
        10.0,
    )

    x[:, 0, :8] = np.stack(
        [
            tracked_n / max(1, int(adapt.max_trackers)),
            grid_min,
            global_tardiness_norm,
            mean_tracked_delay_norm,
            overdue_frac,
            global_deadline_pressure_norm,
            search_penalty_norm,
            global_penalty_norm,
        ],
        axis=1,
    )
    x[:, 0, 0] = np.clip(x[:, 0, 0] / 3000.0, -2.0, 2.0)
    x[:, 0, 1] = np.clip(x[:, 0, 1] / 3000.0, -2.0, 2.0)
    x[:, 0, 2] = np.clip(x[:, 0, 2] / 100.0, 0.0, 2.0)
    x[:, 0, 5] = np.clip(x[:, 0, 5] / 3000.0, -2.0, 2.0)

    az_bin = np.stack([np.asarray(obs.get("az_bin", np.zeros(max_trackers, dtype=np.float32)), dtype=np.float32)[:max_trackers] for obs in observations], axis=0)
    el_bin = np.stack([np.asarray(obs.get("el_bin", np.zeros(max_trackers, dtype=np.float32)), dtype=np.float32)[:max_trackers] for obs in observations], axis=0)
    sector_idx = np.clip(np.round(el_bin * 9.0).astype(np.int32) * 30 + np.round(az_bin * 29.0).astype(np.int32), 0, 299)
    sector_urgency = np.zeros((n, max_trackers), dtype=np.float32)
    for i, grid in enumerate(grids):
        if len(grid) > 0:
            sector_urgency[i] = grid[np.clip(sector_idx[i], 0, len(grid) - 1)].astype(np.float32)

    target_tardiness = np.maximum(0.0, -t_desired).astype(np.float32)
    local_penalty_norm = np.clip(
        0.001 * target_tardiness * (1.0 + 2.0 * priority) * float(pure.local_tardiness_weight),
        0.0,
        10.0,
    ).astype(np.float32)
    x[:, 1 : max_trackers + 1, 0] = np.clip(t_desired / 3000.0, -2.0, 2.0)
    x[:, 1 : max_trackers + 1, 1] = np.clip(deadline / 3000.0, -2.0, 2.0)
    x[:, 1 : max_trackers + 1, 2] = np.clip(dwell / 100.0, 0.0, 2.0)
    x[:, 1 : max_trackers + 1, 3] = priority
    x[:, 1 : max_trackers + 1, 4] = (active & tracked).astype(np.float32)
    x[:, 1 : max_trackers + 1, 5] = np.clip(sector_urgency / 3000.0, -2.0, 2.0)
    x[:, 1 : max_trackers + 1, 6] = local_penalty_norm
    x[:, 1 : max_trackers + 1, 7] = (global_penalty_norm + search_penalty_norm)[:, None]
    range_norm = np.clip(ranges / 184_000_000.0, 0.0, 1.5)
    x[:, 1 : max_trackers + 1, 9] = range_norm
    x[:, 1 : max_trackers + 1, 10] = ((ranges > 10_000_000.0) & (ranges < 184_000_000.0)).astype(np.float32)
    x[:, 1 : max_trackers + 1, 11] = ((ranges > 5_000_000.0) & (ranges < 100_000_000.0)).astype(np.float32)
    x[:, :, 12] = np.asarray([float(obs.get("sensor_id", 0.0)) for obs in observations], dtype=np.float32)[:, None]
    x[:, 0, 9] = np.clip(np.asarray([float(obs.get("s_band_busy_ms", 0.0)) for obs in observations], dtype=np.float32) / 200.0, 0.0, 5.0)
    x[:, 0, 10] = np.clip(np.asarray([float(obs.get("x_band_busy_ms", 0.0)) for obs in observations], dtype=np.float32) / 200.0, 0.0, 5.0)
    x[:, 0, 11] = np.asarray([float(obs.get("enable_x_band", 0.0)) for obs in observations], dtype=np.float32)
    for i, obs in enumerate(observations):
        if float(obs.get("use_grid_feature", 0.0)) > 0.5:
            grid = grids[i]
            if grid.size == 0:
                mean_overdue, drop_frac, max_age = 0.0, 0.0, 0.0
            else:
                age = 3000.0 - grid
                overdue = np.maximum(0.0, age - 3000.0) / 3000.0
                mean_overdue = float(np.clip(float(np.mean(overdue)), 0.0, 5.0))
                drop_frac = float(np.clip(float(np.mean(age > 4500.0)), 0.0, 1.0))
                max_age = float(np.clip(float(np.max(age) / 4500.0), 0.0, 5.0))
            x[i, 0, 9] = mean_overdue
            x[i, 0, 10] = drop_frac
            x[i, 0, 11] = max_age
    return x


def tokenize_packed_root_fast(adapt, packed: PackedRootObs, max_trackers: int, token_dim: int) -> np.ndarray:
    n = len(packed.observations)
    x = np.zeros((n, int(max_trackers) + 1, int(token_dim)), dtype=np.float32)
    if n <= 0:
        return x

    t_desired = packed.t_desired
    deadline = packed.deadline
    dwell = packed.dwell
    active = packed.active
    tracked = packed.tracked
    priority = packed.priority
    ranges = packed.ranges
    grids = packed.grids
    pure = adapt.pure_mcts
    if float(pure.search_debt_penalty_weight) <= 0.0:
        search_penalty_norm = np.zeros((n,), dtype=np.float32)
    elif int(pure.search_delay_mode) == 0:
        search_penalty_norm = (float(pure.search_debt_penalty_weight) * packed.search_debt_ms).astype(np.float32)
    else:
        arg = np.minimum(packed.search_debt_ms / max(1e-3, float(pure.search_debt_tau_ms)), 20.0)
        search_penalty_norm = (float(pure.search_debt_penalty_weight) * (np.exp(arg) - 1.0)).astype(np.float32)
    if float(pure.search_delay_penalty_cap) >= 0.0:
        search_penalty_norm = np.minimum(search_penalty_norm, float(pure.search_delay_penalty_cap))
    search_penalty_norm = np.clip(np.where(packed.search_debt_ms > 0.0, search_penalty_norm, 0.0), 0.0, 10.0).astype(np.float32)

    tracked_active = active & tracked
    tracked_n = np.sum(tracked_active, axis=1).astype(np.float32)
    tracked_delays = np.maximum(0.0, -t_desired) * tracked_active.astype(np.float32)
    tracked_delay_sum = np.sum(tracked_delays, axis=1)
    mean_tracked_delay_norm = np.divide(
        tracked_delay_sum,
        np.maximum(tracked_n, 1.0),
        out=np.zeros_like(tracked_delay_sum),
        where=tracked_n > 0,
    )
    mean_tracked_delay_norm = np.clip(mean_tracked_delay_norm / 2000.0, 0.0, 10.0)
    overdue_count = np.sum((t_desired < 0.0) & tracked_active, axis=1).astype(np.float32)
    overdue_frac = np.divide(overdue_count, np.maximum(tracked_n, 1.0), out=np.zeros_like(overdue_count), where=tracked_n > 0)
    global_tardiness_norm = np.clip(tracked_delay_sum / 20000.0, 0.0, 10.0)
    deadline_pressure = np.maximum(0.0, 100.0 - deadline) * tracked_active.astype(np.float32)
    global_deadline_pressure_norm = np.clip(np.sum(deadline_pressure, axis=1) / 2000.0, 0.0, 10.0)
    global_penalty_norm = np.clip(
        0.001
        * (
            float(pure.global_tardiness_weight) * global_tardiness_norm
            + float(pure.local_tardiness_weight) * mean_tracked_delay_norm
        ),
        0.0,
        10.0,
    )

    x[:, 0, :8] = np.stack(
        [
            tracked_n / max(1, int(adapt.max_trackers)),
            np.min(grids, axis=1),
            global_tardiness_norm,
            mean_tracked_delay_norm,
            overdue_frac,
            global_deadline_pressure_norm,
            search_penalty_norm,
            global_penalty_norm,
        ],
        axis=1,
    )
    x[:, 0, 0] = np.clip(x[:, 0, 0] / 3000.0, -2.0, 2.0)
    x[:, 0, 1] = np.clip(x[:, 0, 1] / 3000.0, -2.0, 2.0)
    x[:, 0, 2] = np.clip(x[:, 0, 2] / 100.0, 0.0, 2.0)
    x[:, 0, 5] = np.clip(x[:, 0, 5] / 3000.0, -2.0, 2.0)

    sector_idx = np.clip(np.round(packed.el_bin * 9.0).astype(np.int32) * 30 + np.round(packed.az_bin * 29.0).astype(np.int32), 0, 299)
    sector_urgency = np.take_along_axis(grids, sector_idx, axis=1).astype(np.float32)
    target_tardiness = np.maximum(0.0, -t_desired).astype(np.float32)
    local_penalty_norm = np.clip(
        0.001 * target_tardiness * (1.0 + 2.0 * priority) * float(pure.local_tardiness_weight),
        0.0,
        10.0,
    ).astype(np.float32)
    x[:, 1 : max_trackers + 1, 0] = np.clip(t_desired / 3000.0, -2.0, 2.0)
    x[:, 1 : max_trackers + 1, 1] = np.clip(deadline / 3000.0, -2.0, 2.0)
    x[:, 1 : max_trackers + 1, 2] = np.clip(dwell / 100.0, 0.0, 2.0)
    x[:, 1 : max_trackers + 1, 3] = priority
    x[:, 1 : max_trackers + 1, 4] = (active & tracked).astype(np.float32)
    x[:, 1 : max_trackers + 1, 5] = np.clip(sector_urgency / 3000.0, -2.0, 2.0)
    x[:, 1 : max_trackers + 1, 6] = local_penalty_norm
    x[:, 1 : max_trackers + 1, 7] = (global_penalty_norm + search_penalty_norm)[:, None]
    x[:, 1 : max_trackers + 1, 9] = np.clip(ranges / 184_000_000.0, 0.0, 1.5)
    x[:, 1 : max_trackers + 1, 10] = ((ranges > 10_000_000.0) & (ranges < 184_000_000.0)).astype(np.float32)
    x[:, 1 : max_trackers + 1, 11] = ((ranges > 5_000_000.0) & (ranges < 100_000_000.0)).astype(np.float32)
    x[:, :, 12] = packed.sensor_id[:, None]
    x[:, 0, 9] = np.clip(packed.s_busy_ms / 200.0, 0.0, 5.0)
    x[:, 0, 10] = np.clip(packed.x_busy_ms / 200.0, 0.0, 5.0)
    x[:, 0, 11] = packed.enable_x_band
    if np.any(packed.use_grid_feature > 0.5):
        age = 3000.0 - grids
        mean_overdue = np.clip(np.mean(np.maximum(0.0, age - 3000.0) / 3000.0, axis=1), 0.0, 5.0)
        drop_frac = np.clip(np.mean(age > 4500.0, axis=1), 0.0, 1.0)
        max_age = np.clip(np.max(age, axis=1) / 4500.0, 0.0, 5.0)
        mask = packed.use_grid_feature > 0.5
        x[:, 0, 9] = np.where(mask, mean_overdue, x[:, 0, 9])
        x[:, 0, 10] = np.where(mask, drop_frac, x[:, 0, 10])
        x[:, 0, 11] = np.where(mask, max_age, x[:, 0, 11])
    return x


def physical_action_table_from_packed(packed: PackedRootObs, live_pos: list[int], selected: list[set[int]], max_trackers: int):
    from exact_env_mutual import xs_s_search_action, xs_s_track_action, xs_x_search_action, xs_x_track_action
    from perf_fast_planner import BatchedPhysicalActionTable

    n = len(live_pos)
    width = 2 + 2 * int(max_trackers)
    actions = np.full((n, width), -1, dtype=np.int64)
    bases = np.zeros((n, width), dtype=np.int64)
    sensors = np.zeros((n, width), dtype=np.int64)
    valid = np.zeros((n, width), dtype=bool)
    if n <= 0:
        return BatchedPhysicalActionTable(actions=actions, bases=bases, sensors=sensors, valid=valid)

    idx = np.asarray(live_pos, dtype=np.int64)
    active = packed.active[idx]
    deadline = packed.deadline[idx]
    ranges = packed.ranges[idx]
    selected_mask = np.zeros((n, max_trackers), dtype=bool)
    for row, pos in enumerate(live_pos):
        for base in selected[pos]:
            target_idx = int(base) - 1
            if 0 <= target_idx < max_trackers:
                selected_mask[row, target_idx] = True

    s_free = packed.s_busy_ms[idx] <= 0.0
    x_free = (packed.enable_x_band[idx].astype(np.int32) != 0) & (packed.x_busy_ms[idx] <= 0.0)
    valid_target = active & np.isfinite(deadline) & (deadline >= 0.0) & ~selected_mask
    sort_key = np.where(valid_target, deadline, np.inf)
    target_order = np.argsort(sort_key, axis=1, kind="stable")
    row_idx = np.arange(n)[:, None]
    ordered_ranges = ranges[row_idx, target_order]
    ordered_valid = valid_target[row_idx, target_order]
    ordered_bases = target_order + 1

    s_track_by_target = np.asarray([xs_s_track_action(i + 1, max_trackers) for i in range(max_trackers)], dtype=np.int64)
    x_track_by_target = np.asarray([xs_x_track_action(i + 1, max_trackers) for i in range(max_trackers)], dtype=np.int64)
    actions[:, 0] = xs_s_search_action(max_trackers)
    bases[:, 0] = 0
    sensors[:, 0] = 0
    valid[:, 0] = True
    actions[:, 1] = xs_x_search_action(max_trackers)
    bases[:, 1] = 0
    sensors[:, 1] = 1
    valid[:, 1] = x_free
    s_cols = slice(2, 2 + max_trackers)
    x_cols = slice(2 + max_trackers, 2 + 2 * max_trackers)
    actions[:, s_cols] = s_track_by_target[target_order]
    bases[:, s_cols] = ordered_bases
    sensors[:, s_cols] = 0
    valid[:, s_cols] = ordered_valid & s_free[:, None] & (ordered_ranges > 10_000_000.0) & (ordered_ranges < 184_000_000.0)
    actions[:, x_cols] = x_track_by_target[target_order]
    bases[:, x_cols] = ordered_bases
    sensors[:, x_cols] = 1
    valid[:, x_cols] = ordered_valid & x_free[:, None] & (ordered_ranges > 5_000_000.0) & (ordered_ranges < 100_000_000.0)
    return BatchedPhysicalActionTable(actions=actions, bases=bases, sensors=sensors, valid=valid)


def physical_action_template_from_packed(packed: PackedRootObs, max_trackers: int) -> PhysicalActionTemplate:
    from exact_env_mutual import xs_s_search_action, xs_s_track_action, xs_x_search_action, xs_x_track_action

    n = int(packed.active.shape[0])
    width = 2 + 2 * int(max_trackers)
    actions = np.full((n, width), -1, dtype=np.int64)
    bases = np.zeros((n, width), dtype=np.int64)
    sensors = np.zeros((n, width), dtype=np.int64)
    valid = np.zeros((n, width), dtype=bool)
    if n <= 0:
        return PhysicalActionTemplate(actions=actions, bases=bases, sensors=sensors, valid=valid)

    active = packed.active
    deadline = packed.deadline
    ranges = packed.ranges
    s_free = packed.s_busy_ms <= 0.0
    x_free = (packed.enable_x_band.astype(np.int32) != 0) & (packed.x_busy_ms <= 0.0)
    valid_target = active & np.isfinite(deadline) & (deadline >= 0.0)
    sort_key = np.where(valid_target, deadline, np.inf)
    target_order = np.argsort(sort_key, axis=1, kind="stable")
    row_idx = np.arange(n)[:, None]
    ordered_ranges = ranges[row_idx, target_order]
    ordered_valid = valid_target[row_idx, target_order]
    ordered_bases = target_order + 1

    s_track_by_target = np.asarray([xs_s_track_action(i + 1, max_trackers) for i in range(max_trackers)], dtype=np.int64)
    x_track_by_target = np.asarray([xs_x_track_action(i + 1, max_trackers) for i in range(max_trackers)], dtype=np.int64)
    actions[:, 0] = xs_s_search_action(max_trackers)
    bases[:, 0] = 0
    sensors[:, 0] = 0
    valid[:, 0] = True
    actions[:, 1] = xs_x_search_action(max_trackers)
    bases[:, 1] = 0
    sensors[:, 1] = 1
    valid[:, 1] = x_free
    s_cols = slice(2, 2 + max_trackers)
    x_cols = slice(2 + max_trackers, 2 + 2 * max_trackers)
    actions[:, s_cols] = s_track_by_target[target_order]
    bases[:, s_cols] = ordered_bases
    sensors[:, s_cols] = 0
    valid[:, s_cols] = ordered_valid & s_free[:, None] & (ordered_ranges > 10_000_000.0) & (ordered_ranges < 184_000_000.0)
    actions[:, x_cols] = x_track_by_target[target_order]
    bases[:, x_cols] = ordered_bases
    sensors[:, x_cols] = 1
    valid[:, x_cols] = ordered_valid & x_free[:, None] & (ordered_ranges > 5_000_000.0) & (ordered_ranges < 100_000_000.0)
    return PhysicalActionTemplate(actions=actions, bases=bases, sensors=sensors, valid=valid)


def physical_action_table_from_template(template: PhysicalActionTemplate, live_pos: list[int], selected: list[set[int]]):
    from perf_fast_planner import BatchedPhysicalActionTable

    idx = np.asarray(live_pos, dtype=np.int64)
    actions = template.actions[idx]
    bases = template.bases[idx]
    sensors = template.sensors[idx]
    valid = template.valid[idx].copy()
    for row, pos in enumerate(live_pos):
        for base in selected[pos]:
            valid[row, bases[row] == int(base)] = False
    return BatchedPhysicalActionTable(actions=actions, bases=bases, sensors=sensors, valid=valid)


def execute_known_valid_action_fast(eng, action: int, base: int, dwell_ms: float, remaining_ms: float):
    """Execute one already-validated physical action without observation rereads.

    The benchmark's candidate table has already checked active/deadline/range
    and sensor availability. The generic executor repeats that validation and
    reads observations before/after each action only to infer elapsed time. This
    fast path keeps the same C environment transition/reward and derives elapsed
    time from the selected action type.
    """
    if eng.term_buf[0] or remaining_ms <= 0.0:
        return 0.0, 0.0, None
    global _FAST_STEP_BINDING, _FAST_STEP_SEARCH_DWELL_MS
    if _FAST_STEP_BINDING is None:
        from pufferlib.ocean.radarxs import binding as radar_binding
        from repaired_campaign_tools import SEARCH_DWELL_MS

        _FAST_STEP_BINDING = radar_binding
        _FAST_STEP_SEARCH_DWELL_MS = float(SEARCH_DWELL_MS)

    eng.act_buf[0] = int(action)
    _FAST_STEP_BINDING.vec_step(eng.env)
    if int(base) == 0:
        dt = float(_FAST_STEP_SEARCH_DWELL_MS)
    else:
        dt = float(dwell_ms)
    if not np.isfinite(dt) or dt <= 0.0:
        return float(eng.rew_buf[0]), 0.0, None
    return float(eng.rew_buf[0]), dt, int(action)


def run_serial(planner, envs, args, device: torch.device) -> dict:
    from final_radar_campaign import get_obs
    from strict_window_report import execute_plan_until_budget

    search_debt = [0.0 for _ in envs]
    plan_times: list[float] = []
    rewards = [0.0 for _ in envs]
    executed = [0 for _ in envs]
    windows_done = 0
    if envs and hasattr(planner, "warmup"):
        warm_obs = get_obs(envs[0], search_debt[0])
        planner.warmup(warm_obs, budget_ms=int(args.window_ms))
    sync(device)
    wall0 = time.perf_counter()
    for window_idx in range(int(args.windows)):
        active_ids = [i for i, eng in enumerate(envs) if not eng.term_buf[0]]
        if not active_ids:
            break
        for env_idx in active_ids:
            eng = envs[env_idx]
            obs = get_obs(eng, search_debt[env_idx])
            sync(device)
            t0 = time.perf_counter()
            plan = planner.plan(obs, budget_ms=int(args.window_ms))
            sync(device)
            plan_times.append((time.perf_counter() - t0) * 1000.0)
            reward, _spent, new_debt, n_exec, _search_actions, _rows = execute_plan_until_budget(
                eng,
                plan,
                float(args.window_ms),
                search_debt[env_idx],
                "serial_fast_graph_gpu_select",
                int(args.seed) + env_idx,
                int(window_idx),
            )
            search_debt[env_idx] = float(new_debt)
            rewards[env_idx] += float(reward)
            executed[env_idx] += int(n_exec)
        windows_done += 1
    sync(device)
    wall_ms = (time.perf_counter() - wall0) * 1000.0
    total_windows = int(windows_done * len(envs))
    return {
        "wall_ms": float(wall_ms),
        "windows_requested": int(args.windows * len(envs)),
        "window_rounds": int(windows_done),
        "envs": int(len(envs)),
        "planned_env_windows": int(total_windows),
        "window_throughput_per_s": float(1000.0 * total_windows / max(wall_ms, 1e-12)),
        "planning_ms_per_env_window": float(np.mean(plan_times)) if plan_times else 0.0,
        "planning_stats": stats(plan_times),
        "total_reward": float(sum(rewards)),
        "executed_actions": int(sum(executed)),
    }


def run_batched(scorer, envs, args, device: torch.device) -> dict:
    from exact_env_mutual import xs_decode_action
    from final_radar_campaign import get_obs
    from repaired_campaign_tools import decode_sensor_action, execute_first_valid_action
    from two_sensor_physical_head_eval import MAXT

    search_debt = [0.0 for _ in envs]
    rewards = [0.0 for _ in envs]
    executed = [0 for _ in envs]
    plan_round_times: list[float] = []
    batch_sizes: list[int] = []
    depth_counts: list[int] = []
    if envs:
        warm_obs = [get_obs(eng, 0.0) for eng in envs if not eng.term_buf[0]]
        if warm_obs:
            _ = scorer.best_actions_torch(warm_obs, budget_ms=float(args.window_ms))
    sync(device)
    wall0 = time.perf_counter()
    windows_done = 0
    for window_idx in range(int(args.windows)):
        selected = [set() for _ in envs]
        elapsed = [0.0 for _ in envs]
        search_count = [0 for _ in envs]
        track_count = [0 for _ in envs]
        last = [-1 for _ in envs]
        active_ids = [i for i, eng in enumerate(envs) if not eng.term_buf[0]]
        if not active_ids:
            break
        depth = 0
        while active_ids and depth < int(args.max_depth):
            obs_batch = [get_obs(envs[i], search_debt[i]) for i in active_ids]
            selected_batch = [selected[i] for i in active_ids]
            elapsed_batch = [elapsed[i] for i in active_ids]
            search_count_batch = [search_count[i] for i in active_ids]
            track_count_batch = [track_count[i] for i in active_ids]
            last_batch = [last[i] for i in active_ids]
            sync(device)
            t0 = time.perf_counter()
            actions = scorer.best_actions_torch(
                obs_batch,
                selected=selected_batch,
                elapsed=elapsed_batch,
                search_count=search_count_batch,
                track_count=track_count_batch,
                last=last_batch,
                budget_ms=float(args.window_ms),
            )
            sync(device)
            plan_round_times.append((time.perf_counter() - t0) * 1000.0)
            batch_sizes.append(len(active_ids))
            next_active: list[int] = []
            for local_idx, env_idx in enumerate(active_ids):
                eng = envs[env_idx]
                if eng.term_buf[0] or elapsed[env_idx] >= float(args.window_ms):
                    continue
                remaining = max(0.0, float(args.window_ms) - float(elapsed[env_idx]))
                reward, dt, executed_action = execute_first_valid_action(eng, [int(actions[local_idx])], remaining)
                if executed_action is None or dt <= 0.0:
                    continue
                logical_action, _sensor = decode_sensor_action(int(executed_action), eng.max_trackers)
                base, _ = xs_decode_action(int(executed_action), MAXT)
                if int(logical_action) == 0:
                    search_debt[env_idx] = 0.0
                    search_count[env_idx] += 1
                else:
                    search_debt[env_idx] += max(float(dt), 0.0)
                    if int(base) > 0:
                        selected[env_idx].add(int(base))
                    track_count[env_idx] += 1
                rewards[env_idx] += float(reward)
                elapsed[env_idx] += float(dt)
                executed[env_idx] += 1
                last[env_idx] = int(base)
                if not eng.term_buf[0] and elapsed[env_idx] < float(args.window_ms):
                    next_active.append(env_idx)
            active_ids = next_active
            depth += 1
        depth_counts.append(depth)
        windows_done += 1
    sync(device)
    wall_ms = (time.perf_counter() - wall0) * 1000.0
    total_env_windows = int(windows_done * len(envs))
    return {
        "wall_ms": float(wall_ms),
        "windows_requested": int(args.windows * len(envs)),
        "window_rounds": int(windows_done),
        "envs": int(len(envs)),
        "planned_env_windows": int(total_env_windows),
        "window_throughput_per_s": float(1000.0 * total_env_windows / max(wall_ms, 1e-12)),
        "neural_rounds": int(len(plan_round_times)),
        "mean_batch_size": float(np.mean(batch_sizes)) if batch_sizes else 0.0,
        "batch_size_distribution": int_distribution(batch_sizes, full_size=len(envs)),
        "mean_depth": float(np.mean(depth_counts)) if depth_counts else 0.0,
        "depth_distribution": int_distribution(depth_counts),
        "planning_round_stats": stats(plan_round_times),
        "planning_ms_per_env_action": float(sum(plan_round_times) / max(1, sum(batch_sizes))),
        "total_reward": float(sum(rewards)),
        "executed_actions": int(sum(executed)),
    }


def run_batched_cached(planner, envs, args, device: torch.device) -> dict:
    from exact_env_mutual import attach_env_obs, xs_decode_action
    from final_radar_campaign import get_obs
    from mutual_features import TOKEN_DIM
    from realistic_reward_retrain import adapter
    from repaired_campaign_tools import decode_sensor_action, execute_first_valid_action
    from two_sensor_physical_head_eval import MAXT

    adapt = adapter()
    search_debt = [0.0 for _ in envs]
    rewards = [0.0 for _ in envs]
    executed = [0 for _ in envs]
    plan_round_times: list[float] = []
    encode_times: list[float] = []
    batch_sizes: list[int] = []
    depth_counts: list[int] = []
    profile_enabled = bool(getattr(args, "profile_stages", False))
    stage_buckets: dict[str, list[float]] = {}
    if envs and hasattr(planner, "warmup"):
        planner.warmup(get_obs(envs[0], 0.0), budget_ms=int(args.window_ms))
    sync(device)
    wall0 = time.perf_counter()
    windows_done = 0
    for window_idx in range(int(args.windows)):
        root_env_ids = [i for i, eng in enumerate(envs) if not eng.term_buf[0]]
        if not root_env_ids:
            break
        if bool(getattr(args, "direct_root_pack", False)):
            packed = time_stage(
                device,
                profile_enabled,
                stage_buckets,
                "root_pack_direct",
                lambda: pack_root_envs_direct(envs, root_env_ids, search_debt, planner.env_cfg, MAXT),
            )
        else:
            obs2 = time_stage(
                device,
                profile_enabled,
                stage_buckets,
                "root_obs_attach",
                lambda: [attach_env_obs(get_obs(envs[i], search_debt[i]), planner.env_cfg, True, True) for i in root_env_ids],
            )
            packed = time_stage(
                device,
                profile_enabled,
                stage_buckets,
                "root_pack_observations",
                lambda: pack_root_observations(obs2, MAXT),
            )
        selected = [set() for _ in root_env_ids]
        elapsed = [0.0 for _ in root_env_ids]
        search_count = [0 for _ in root_env_ids]
        track_count = [0 for _ in root_env_ids]
        last = [-1 for _ in root_env_ids]
        slot_template = time_stage(
            device,
            profile_enabled,
            stage_buckets,
            "root_slot_template",
            lambda: slot_template_from_packed(packed, float(args.window_ms)),
        )
        root_tokens = time_stage(
            device,
            profile_enabled,
            stage_buckets,
            "root_tokenize_batch",
            lambda: tokenize_packed_root_fast(adapt, packed, MAXT, TOKEN_DIM),
        )
        sync(device)
        t0 = time.perf_counter()
        def encode_root():
            with torch.inference_mode():
                root_x = torch.from_numpy(root_tokens).to(device, dtype=torch.float32)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=planner.use_amp):
                    return planner.model.backbone.encode_tokens(root_x)

        cls_out, tok_out, selected_t_all, token_active = time_stage(
            device,
            profile_enabled,
            stage_buckets,
            "root_h2d_encode",
            encode_root,
        )
        selected_t_all = selected_t_all.clone()
        sync(device)
        encode_times.append((time.perf_counter() - t0) * 1000.0)
        live_pos = list(range(len(root_env_ids)))
        depth = 0
        while live_pos and depth < int(args.max_depth):
            slots = time_stage(
                device,
                profile_enabled,
                stage_buckets,
                "slot_context_update",
                lambda: make_live_slots(slot_template, live_pos, elapsed, search_count, track_count, last, float(args.window_ms)),
            )
            physical = time_stage(
                device,
                profile_enabled,
                stage_buckets,
                "physical_action_table_batch",
                lambda: physical_action_table_from_packed(packed, live_pos, selected, MAXT),
            )
            sync(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                def prep_tensors():
                    pos_t = torch.as_tensor(live_pos, device=device, dtype=torch.long)
                    slot_t = torch.from_numpy(slots).to(device, dtype=torch.float32)
                    selected_t = selected_t_all.index_select(0, pos_t)
                    cls_live = cls_out.index_select(0, pos_t)
                    tok_live = tok_out.index_select(0, pos_t)
                    active_live = token_active.index_select(0, pos_t)
                    actions_t = torch.as_tensor(physical.actions, device=device, dtype=torch.long)
                    flat_t = torch.as_tensor(physical.bases * 2 + physical.sensors, device=device, dtype=torch.long)
                    valid_t = torch.as_tensor(physical.valid, device=device, dtype=torch.bool)
                    return slot_t, selected_t, cls_live, tok_live, active_live, actions_t, flat_t, valid_t

                slot_t, selected_t, cls_live, tok_live, active_live, actions_t, flat_t, valid_t = time_stage(
                    device,
                    profile_enabled,
                    stage_buckets,
                    "decision_tensor_prep_h2d",
                    prep_tensors,
                )

                def score_forward():
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=planner.use_amp):
                        score = planner.score_slots_from_encoded(cls_live, tok_live, selected_t, active_live, slot_t).float()
                    if planner.search_score_bias != 0.0:
                        score[:, 0, :] += planner.search_score_bias
                    return score

                score_t = time_stage(device, profile_enabled, stage_buckets, "decision_score_forward", score_forward)

                def select_actions():
                    candidate_scores = torch.gather(score_t.reshape(len(live_pos), -1), 1, flat_t)
                    candidate_scores.masked_fill_(~valid_t, -torch.inf)
                    idx = torch.argmax(candidate_scores, dim=1)
                    return torch.gather(actions_t, 1, idx[:, None]).squeeze(1)

                best = time_stage(device, profile_enabled, stage_buckets, "decision_select_device", select_actions)
                actions = time_stage(
                    device,
                    profile_enabled,
                    stage_buckets,
                    "decision_action_d2h",
                    lambda: best.cpu().numpy().astype(np.int64, copy=False),
                )
            sync(device)
            plan_round_times.append((time.perf_counter() - t0) * 1000.0)
            batch_sizes.append(len(live_pos))
            next_live: list[int] = []
            def step_envs():
                next_ids: list[int] = []
                for local_idx, pos in enumerate(live_pos):
                    env_idx = root_env_ids[pos]
                    eng = envs[env_idx]
                    if eng.term_buf[0] or elapsed[pos] >= float(args.window_ms):
                        continue
                    remaining = max(0.0, float(args.window_ms) - float(elapsed[pos]))
                    action = int(actions[local_idx])
                    base, _ = xs_decode_action(action, MAXT)
                    if bool(getattr(args, "fast_env_step", False)):
                        dwell = float(packed.dwell[pos, int(base) - 1]) if int(base) > 0 else 0.0
                        reward, dt, executed_action = execute_known_valid_action_fast(eng, action, int(base), dwell, remaining)
                    else:
                        reward, dt, executed_action = execute_first_valid_action(eng, [action], remaining)
                    if executed_action is None or dt <= 0.0:
                        continue
                    logical_action, _sensor = decode_sensor_action(int(executed_action), eng.max_trackers)
                    base, _ = xs_decode_action(int(executed_action), MAXT)
                    if int(logical_action) == 0:
                        search_debt[env_idx] = 0.0
                        search_count[pos] += 1
                    else:
                        search_debt[env_idx] += max(float(dt), 0.0)
                        if int(base) > 0:
                            selected[pos].add(int(base))
                            if 0 <= int(base) < selected_t_all.shape[1]:
                                selected_t_all[pos, int(base)] = True
                        track_count[pos] += 1
                    rewards[env_idx] += float(reward)
                    elapsed[pos] += float(dt)
                    executed[env_idx] += 1
                    last[pos] = int(base)
                    if not eng.term_buf[0] and elapsed[pos] < float(args.window_ms):
                        next_ids.append(pos)
                return next_ids

            next_live = time_stage(device, profile_enabled, stage_buckets, "env_step_batch", step_envs)
            live_pos = next_live
            depth += 1
        depth_counts.append(depth)
        windows_done += 1
    sync(device)
    wall_ms = (time.perf_counter() - wall0) * 1000.0
    total_env_windows = int(windows_done * len(envs))
    return {
        "wall_ms": float(wall_ms),
        "windows_requested": int(args.windows * len(envs)),
        "window_rounds": int(windows_done),
        "envs": int(len(envs)),
        "planned_env_windows": int(total_env_windows),
        "window_throughput_per_s": float(1000.0 * total_env_windows / max(wall_ms, 1e-12)),
        "encode_stats": stats(encode_times),
        "neural_rounds": int(len(plan_round_times)),
        "mean_batch_size": float(np.mean(batch_sizes)) if batch_sizes else 0.0,
        "batch_size_distribution": int_distribution(batch_sizes, full_size=len(envs)),
        "mean_depth": float(np.mean(depth_counts)) if depth_counts else 0.0,
        "depth_distribution": int_distribution(depth_counts),
        "planning_round_stats": stats(plan_round_times),
        "planning_ms_per_env_action": float(sum(plan_round_times) / max(1, sum(batch_sizes))),
        "total_reward": float(sum(rewards)),
        "executed_actions": int(sum(executed)),
        "stage_profile": profile_summary(stage_buckets) if profile_enabled else {},
    }


def _build_score_graph(planner, graph_cache: dict, cls_out, tok_out, selected_t, token_active, slot_t):
    if planner.device.type != "cuda":
        return None
    key = (
        tuple(cls_out.shape),
        tuple(tok_out.shape),
        tuple(selected_t.shape),
        tuple(token_active.shape),
        tuple(slot_t.shape),
        str(cls_out.dtype),
        bool(planner.use_amp),
    )
    try:
        cache = graph_cache.get(key)
        if cache is None:
            static_cls = torch.empty_like(cls_out)
            static_tok = torch.empty_like(tok_out)
            static_selected = torch.empty_like(selected_t)
            static_active = torch.empty_like(token_active)
            static_slot = torch.empty_like(slot_t)
            static_cls.copy_(cls_out, non_blocking=False)
            static_tok.copy_(tok_out, non_blocking=False)
            static_selected.copy_(selected_t, non_blocking=False)
            static_active.copy_(token_active, non_blocking=False)
            static_slot.copy_(slot_t, non_blocking=False)

            def compute_score():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=planner.use_amp):
                    return planner.score_slots_from_encoded(
                        static_cls,
                        static_tok,
                        static_selected,
                        static_active,
                        static_slot,
                    ).float()

            with torch.inference_mode():
                for _ in range(3):
                    _ = compute_score()
                torch.cuda.synchronize(planner.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_score = compute_score()
            cache = {
                "graph": graph,
                "static_cls": static_cls,
                "static_tok": static_tok,
                "static_selected": static_selected,
                "static_active": static_active,
                "static_slot": static_slot,
                "static_score": static_score,
            }
            graph_cache[key] = cache
        else:
            graph = cache["graph"]
            static_cls = cache["static_cls"]
            static_tok = cache["static_tok"]
            static_selected = cache["static_selected"]
            static_active = cache["static_active"]
            static_slot = cache["static_slot"]
            static_score = cache["static_score"]
            static_cls.copy_(cls_out, non_blocking=False)
            static_tok.copy_(tok_out, non_blocking=False)
            static_active.copy_(token_active, non_blocking=False)

        def replay(next_selected_t, next_slot_t):
            static_selected.copy_(next_selected_t, non_blocking=False)
            static_slot.copy_(next_slot_t, non_blocking=False)
            graph.replay()
            return static_score

        return replay
    except Exception:
        return None


def _build_score_select_graph(
    planner,
    graph_cache: dict,
    cls_out,
    tok_out,
    selected_t,
    token_active,
    slot_t,
    actions_t,
    flat_t,
    gather_t,
    search_action_t,
    template_valid_t,
):
    if planner.device.type != "cuda":
        return None
    key = (
        tuple(cls_out.shape),
        tuple(tok_out.shape),
        tuple(selected_t.shape),
        tuple(token_active.shape),
        tuple(slot_t.shape),
        tuple(actions_t.shape),
        str(cls_out.dtype),
        str(actions_t.dtype),
        bool(planner.use_amp),
        float(planner.search_score_bias),
    )
    try:
        cache = graph_cache.get(key)
        if cache is None:
            static_cls = torch.empty_like(cls_out)
            static_tok = torch.empty_like(tok_out)
            static_selected = torch.empty_like(selected_t)
            static_active = torch.empty_like(token_active)
            static_slot = torch.empty_like(slot_t)
            static_actions = torch.empty_like(actions_t)
            static_flat = torch.empty_like(flat_t)
            static_gather = torch.empty_like(gather_t)
            static_search = torch.empty_like(search_action_t)
            static_template_valid = torch.empty_like(template_valid_t)
            static_cls.copy_(cls_out, non_blocking=False)
            static_tok.copy_(tok_out, non_blocking=False)
            static_selected.copy_(selected_t, non_blocking=False)
            static_active.copy_(token_active, non_blocking=False)
            static_slot.copy_(slot_t, non_blocking=False)
            static_actions.copy_(actions_t, non_blocking=False)
            static_flat.copy_(flat_t, non_blocking=False)
            static_gather.copy_(gather_t, non_blocking=False)
            static_search.copy_(search_action_t, non_blocking=False)
            static_template_valid.copy_(template_valid_t, non_blocking=False)

            def compute_best():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=planner.use_amp):
                    score_t = planner.score_slots_from_encoded(
                        static_cls,
                        static_tok,
                        static_selected,
                        static_active,
                        static_slot,
                    ).float()
                if planner.search_score_bias != 0.0:
                    score_t[:, 0, :] += planner.search_score_bias
                selected_by_action = torch.gather(static_selected, 1, static_gather)
                valid_t = static_template_valid & (static_search | ~selected_by_action)
                candidate_scores = torch.gather(score_t.reshape(static_actions.shape[0], -1), 1, static_flat)
                candidate_scores.masked_fill_(~valid_t, -torch.inf)
                idx = torch.argmax(candidate_scores, dim=1)
                return torch.gather(static_actions, 1, idx[:, None]).squeeze(1)

            with torch.inference_mode():
                for _ in range(3):
                    _ = compute_best()
                torch.cuda.synchronize(planner.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_best = compute_best()
            cache = {
                "graph": graph,
                "static_cls": static_cls,
                "static_tok": static_tok,
                "static_selected": static_selected,
                "static_active": static_active,
                "static_slot": static_slot,
                "static_actions": static_actions,
                "static_flat": static_flat,
                "static_gather": static_gather,
                "static_search": static_search,
                "static_template_valid": static_template_valid,
                "static_best": static_best,
            }
            graph_cache[key] = cache
        else:
            graph = cache["graph"]
            static_cls = cache["static_cls"]
            static_tok = cache["static_tok"]
            static_selected = cache["static_selected"]
            static_active = cache["static_active"]
            static_slot = cache["static_slot"]
            static_actions = cache["static_actions"]
            static_flat = cache["static_flat"]
            static_gather = cache["static_gather"]
            static_search = cache["static_search"]
            static_template_valid = cache["static_template_valid"]
            static_best = cache["static_best"]
            static_cls.copy_(cls_out, non_blocking=False)
            static_tok.copy_(tok_out, non_blocking=False)
            static_active.copy_(token_active, non_blocking=False)
            static_actions.copy_(actions_t, non_blocking=False)
            static_flat.copy_(flat_t, non_blocking=False)
            static_gather.copy_(gather_t, non_blocking=False)
            static_search.copy_(search_action_t, non_blocking=False)
            static_template_valid.copy_(template_valid_t, non_blocking=False)

        def replay(next_selected_t, next_slot_t):
            static_selected.copy_(next_selected_t, non_blocking=False)
            static_slot.copy_(next_slot_t, non_blocking=False)
            graph.replay()
            return static_best

        return replay
    except Exception:
        return None


def _build_root_encode_graph(planner, graph_cache: dict, root_x: torch.Tensor):
    if planner.device.type != "cuda":
        return None
    key = (tuple(root_x.shape), str(root_x.dtype), bool(planner.use_amp))
    try:
        cache = graph_cache.get(key)
        if cache is None:
            static_root = torch.empty_like(root_x)
            static_root.copy_(root_x, non_blocking=False)

            def encode_root():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=planner.use_amp):
                    return planner.model.backbone.encode_tokens(static_root)

            with torch.inference_mode():
                for _ in range(3):
                    _ = encode_root()
                torch.cuda.synchronize(planner.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_cls, static_tok, static_selected, static_active = encode_root()
            cache = {
                "graph": graph,
                "static_root": static_root,
                "static_cls": static_cls,
                "static_tok": static_tok,
                "static_selected": static_selected,
                "static_active": static_active,
            }
            graph_cache[key] = cache
        else:
            graph = cache["graph"]
            static_root = cache["static_root"]
            static_cls = cache["static_cls"]
            static_tok = cache["static_tok"]
            static_selected = cache["static_selected"]
            static_active = cache["static_active"]
            static_root.copy_(root_x, non_blocking=False)

        def replay(next_root_x: torch.Tensor):
            static_root.copy_(next_root_x, non_blocking=False)
            graph.replay()
            return static_cls, static_tok, static_selected, static_active

        return replay
    except Exception:
        return None


def run_batched_cached_graph(planner, envs, args, device: torch.device) -> dict:
    from exact_env_mutual import attach_env_obs, xs_decode_action
    from final_radar_campaign import get_obs
    from mutual_features import TOKEN_DIM
    from pufferlib.ocean.radarxs import binding as radar_binding
    from realistic_reward_retrain import adapter
    from repaired_campaign_tools import decode_sensor_action, execute_first_valid_action
    from two_sensor_physical_head_eval import MAXT

    adapt = adapter()
    search_debt = [0.0 for _ in envs]
    rewards = [0.0 for _ in envs]
    executed = [0 for _ in envs]
    plan_round_times: list[float] = []
    encode_times: list[float] = []
    graph_build_times: list[float] = []
    batch_sizes: list[int] = []
    depth_counts: list[int] = []
    root_graph_cache: dict = {}
    score_graph_cache: dict = {}
    score_select_graph_cache: dict = {}
    stage_buckets: dict[str, list[float]] = {}
    profile_enabled = bool(getattr(args, "profile_stages", False))
    root_graph_replays = 0
    root_raw_encodes = 0
    graph_replay_rounds = 0
    graph_select_replay_rounds = 0
    raw_rounds = 0
    padded_graph_rounds = 0
    batch_env_step_fn = getattr(radar_binding, "vec_step_selected_known_valid_into", None)
    if batch_env_step_fn is None:
        batch_env_step_fn = getattr(radar_binding, "vec_step_selected_validated_into", None)
    batch_env_step_available = bool(getattr(args, "batch_env_step", False)) and bool(getattr(args, "fast_env_step", False)) and batch_env_step_fn is not None
    env_vec = radar_binding.vec_view_firsts(*[eng.env for eng in envs]) if batch_env_step_available and envs else None
    env_index_buf = np.empty((len(envs),), dtype=np.int32)
    env_action_buf = np.empty((len(envs),), dtype=np.int32)
    env_dt_buf = np.empty((len(envs),), dtype=np.float32)
    env_executed_buf = np.empty((len(envs),), dtype=np.int32)
    batch_env_step_calls = 0
    scalar_env_step_calls = 0
    pinned_action_cpu = None
    if bool(getattr(args, "pinned_action_d2h", False)) and device.type == "cuda" and len(envs) > 0:
        try:
            pinned_action_cpu = torch.empty((len(envs),), dtype=torch.long, pin_memory=True)
        except Exception:
            pinned_action_cpu = None
    if envs and hasattr(planner, "warmup"):
        planner.warmup(get_obs(envs[0], 0.0), budget_ms=int(args.window_ms))
    sync(device)
    wall0 = time.perf_counter()
    windows_done = 0
    for window_idx in range(int(args.windows)):
        root_env_ids = [i for i, eng in enumerate(envs) if not eng.term_buf[0]]
        if not root_env_ids:
            break
        if bool(getattr(args, "direct_root_pack", False)):
            packed = time_stage(
                device,
                profile_enabled,
                stage_buckets,
                "graph_root_pack_direct",
                lambda: pack_root_envs_direct(envs, root_env_ids, search_debt, planner.env_cfg, MAXT, aux_vec=env_vec),
            )
        else:
            obs2 = time_stage(
                device,
                profile_enabled,
                stage_buckets,
                "graph_root_obs_attach",
                lambda: [attach_env_obs(get_obs(envs[i], search_debt[i]), planner.env_cfg, True, True) for i in root_env_ids],
            )
            packed = time_stage(
                device,
                profile_enabled,
                stage_buckets,
                "graph_root_pack_observations",
                lambda: pack_root_observations(obs2, MAXT),
            )
        selected = [set() for _ in root_env_ids]
        elapsed = [0.0 for _ in root_env_ids]
        search_count = [0 for _ in root_env_ids]
        track_count = [0 for _ in root_env_ids]
        last = [-1 for _ in root_env_ids]
        slot_template = time_stage(
            device,
            profile_enabled,
            stage_buckets,
            "graph_root_slot_template",
            lambda: slot_template_from_packed(packed, float(args.window_ms)),
        )
        root_tokens = time_stage(
            device,
            profile_enabled,
            stage_buckets,
            "graph_root_tokenize_batch",
            lambda: tokenize_packed_root_fast(adapt, packed, MAXT, TOKEN_DIM),
        )
        physical_template = (
            time_stage(
                device,
                profile_enabled,
                stage_buckets,
                "graph_physical_action_template",
                lambda: physical_action_template_from_packed(packed, MAXT),
            )
            if bool(getattr(args, "cached_action_table", False))
            else None
        )
        gpu_action_template = None
        if physical_template is not None and bool(getattr(args, "gpu_action_template", False)):
            def upload_action_template():
                actions_t = torch.from_numpy(physical_template.actions).to(device, dtype=torch.long)
                flat_t = torch.from_numpy(physical_template.bases * 2 + physical_template.sensors).to(device, dtype=torch.long)
                bases_t = torch.from_numpy(physical_template.bases).to(device, dtype=torch.long)
                gather_t = bases_t.clamp_min(0).clamp_max(MAXT)
                search_action_t = bases_t == 0
                template_valid_t = torch.from_numpy(physical_template.valid).to(device, dtype=torch.bool)
                valid_t = torch.empty_like(template_valid_t)
                return actions_t, flat_t, gather_t, search_action_t, template_valid_t, valid_t

            gpu_action_template = time_stage(
                device,
                profile_enabled,
                stage_buckets,
                "graph_action_template_h2d",
                upload_action_template,
            )
        sync(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            root_x = torch.from_numpy(root_tokens).to(device, dtype=torch.float32)
            root_encode_replay = _build_root_encode_graph(planner, root_graph_cache, root_x)
            if root_encode_replay is not None:
                cls_out, tok_out, selected_t_all, token_active = root_encode_replay(root_x)
                root_graph_replays += 1
            else:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=planner.use_amp):
                    cls_out, tok_out, selected_t_all, token_active = planner.model.backbone.encode_tokens(root_x)
                root_raw_encodes += 1
        selected_t_all = selected_t_all.clone()
        sync(device)
        encode_times.append((time.perf_counter() - t0) * 1000.0)

        current_slots = slot_template.copy()
        current_slots_cpu_t = torch.from_numpy(current_slots)
        full_slot_t = current_slots_cpu_t.to(device, dtype=torch.float32)
        sync(device)
        t0 = time.perf_counter()
        graph_replay = _build_score_graph(planner, score_graph_cache, cls_out, tok_out, selected_t_all, token_active, full_slot_t)
        graph_select_replay = None
        if (
            bool(getattr(args, "full_select_graph", False))
            and gpu_action_template is not None
            and bool(getattr(args, "gpu_valid_mask", False))
        ):
            actions_t0, flat_t0, gather_t0, search_action_t0, template_valid_t0, _valid_t0 = gpu_action_template
            graph_select_replay = _build_score_select_graph(
                planner,
                score_select_graph_cache,
                cls_out,
                tok_out,
                selected_t_all,
                token_active,
                full_slot_t,
                actions_t0,
                flat_t0,
                gather_t0,
                search_action_t0,
                template_valid_t0,
            )
        sync(device)
        graph_build_times.append((time.perf_counter() - t0) * 1000.0)
        live_pos = list(range(len(root_env_ids)))
        root_env_ids_arr = np.asarray(root_env_ids, dtype=np.int32)
        table_width = 2 + 2 * int(MAXT)
        prealloc_action_t = torch.empty((len(root_env_ids), table_width), device=device, dtype=torch.long)
        prealloc_flat_t = torch.empty((len(root_env_ids), table_width), device=device, dtype=torch.long)
        prealloc_valid_t = torch.empty((len(root_env_ids), table_width), device=device, dtype=torch.bool)
        prealloc_full_slot_t = torch.empty((len(root_env_ids), slot_template.shape[1]), device=device, dtype=torch.float32)
        live_pos_tensor_cache: dict[tuple[int, ...], torch.Tensor] = {}
        live_pos_np_cache: dict[tuple[int, ...], np.ndarray] = {}
        depth = 0
        while live_pos and depth < int(args.max_depth):
            slots = time_stage(
                device,
                profile_enabled,
                stage_buckets,
                "graph_slot_context_update",
                lambda: current_slots if len(live_pos) == len(root_env_ids) else current_slots[np.asarray(live_pos, dtype=np.int64)].copy(),
            )
            physical = time_stage(
                device,
                profile_enabled,
                stage_buckets,
                "graph_physical_action_table",
                lambda: None
                if gpu_action_template is not None and bool(getattr(args, "gpu_valid_mask", False))
                else physical_action_table_from_template(physical_template, live_pos, selected)
                if physical_template is not None
                else physical_action_table_from_packed(packed, live_pos, selected, MAXT),
            )
            sync(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                best = None
                if graph_select_replay is not None and len(live_pos) == len(root_env_ids):
                    slot_t = time_stage(
                        device,
                        profile_enabled,
                        stage_buckets,
                        "graph_select_slot_h2d",
                        lambda: current_slots_cpu_t.to(device, dtype=torch.float32),
                    )
                    best = time_stage(
                        device,
                        profile_enabled,
                        stage_buckets,
                        "graph_score_select_replay",
                        lambda: graph_select_replay(selected_t_all, slot_t),
                    )
                    score_t = None
                    graph_select_replay_rounds += 1
                elif graph_replay is not None and len(live_pos) == len(root_env_ids):
                    slot_t = time_stage(
                        device,
                        profile_enabled,
                        stage_buckets,
                        "graph_slot_h2d",
                        lambda: current_slots_cpu_t.to(device, dtype=torch.float32),
                    )
                    score_t = time_stage(
                        device,
                        profile_enabled,
                        stage_buckets,
                        "graph_score_replay",
                        lambda: graph_replay(selected_t_all, slot_t),
                    )
                    graph_replay_rounds += 1
                elif graph_replay is not None and bool(getattr(args, "padded_live_graph", False)):
                    def padded_graph_score_path():
                        key = tuple(live_pos)
                        pos_t = live_pos_tensor_cache.get(key)
                        if pos_t is None:
                            pos_t = torch.as_tensor(live_pos, device=device, dtype=torch.long)
                            live_pos_tensor_cache[key] = pos_t
                        prealloc_full_slot_t.copy_(current_slots_cpu_t, non_blocking=False)
                        full_score_t = graph_replay(selected_t_all, prealloc_full_slot_t)
                        return full_score_t.index_select(0, pos_t)

                    score_t = time_stage(device, profile_enabled, stage_buckets, "graph_padded_score_replay", padded_graph_score_path)
                    padded_graph_rounds += 1
                else:
                    def raw_score_path():
                        pos_t = torch.as_tensor(live_pos, device=device, dtype=torch.long)
                        slot_t = torch.from_numpy(slots).to(device, dtype=torch.float32)
                        selected_t = selected_t_all.index_select(0, pos_t)
                        cls_live = cls_out.index_select(0, pos_t)
                        tok_live = tok_out.index_select(0, pos_t)
                        active_live = token_active.index_select(0, pos_t)
                        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=planner.use_amp):
                            return planner.score_slots_from_encoded(cls_live, tok_live, selected_t, active_live, slot_t).float()

                    score_t = time_stage(device, profile_enabled, stage_buckets, "graph_raw_score_forward", raw_score_path)
                    raw_rounds += 1
                if score_t is not None and planner.search_score_bias != 0.0:
                    score_t[:, 0, :] += planner.search_score_bias
                if best is None:
                    def action_tensor_prep():
                        if gpu_action_template is not None and bool(getattr(args, "gpu_valid_mask", False)) and len(live_pos) == len(root_env_ids):
                            actions_t, flat_t, gather_t, search_action_t, template_valid_t, valid_t = gpu_action_template
                            selected_by_action = torch.gather(selected_t_all, 1, gather_t)
                            valid_t.copy_(template_valid_t & (search_action_t | ~selected_by_action), non_blocking=False)
                            return actions_t, flat_t, valid_t
                        if gpu_action_template is not None and bool(getattr(args, "gpu_valid_mask", False)):
                            actions_t, flat_t, gather_t, search_action_t, template_valid_t, _valid_t = gpu_action_template
                            key = tuple(live_pos)
                            pos_t = live_pos_tensor_cache.get(key)
                            if pos_t is None:
                                pos_t = torch.as_tensor(live_pos, device=device, dtype=torch.long)
                                live_pos_tensor_cache[key] = pos_t
                            actions_live = actions_t.index_select(0, pos_t)
                            flat_live = flat_t.index_select(0, pos_t)
                            gather_live = gather_t.index_select(0, pos_t)
                            search_live = search_action_t.index_select(0, pos_t)
                            template_valid_live = template_valid_t.index_select(0, pos_t)
                            selected_live = selected_t_all.index_select(0, pos_t)
                            selected_by_action = torch.gather(selected_live, 1, gather_live)
                            valid_live = template_valid_live & (search_live | ~selected_by_action)
                            return actions_live, flat_live, valid_live
                        if gpu_action_template is not None and len(live_pos) == len(root_env_ids):
                            actions_t, flat_t, _gather_t, _search_action_t, _template_valid_t, valid_t = gpu_action_template
                            valid_t.copy_(torch.from_numpy(physical.valid), non_blocking=False)
                            return actions_t, flat_t, valid_t
                        if len(live_pos) == len(root_env_ids):
                            prealloc_action_t.copy_(torch.from_numpy(physical.actions), non_blocking=False)
                            prealloc_flat_t.copy_(torch.from_numpy(physical.bases * 2 + physical.sensors), non_blocking=False)
                            prealloc_valid_t.copy_(torch.from_numpy(physical.valid), non_blocking=False)
                            return prealloc_action_t, prealloc_flat_t, prealloc_valid_t
                        return (
                            torch.as_tensor(physical.actions, device=device, dtype=torch.long),
                            torch.as_tensor(physical.bases * 2 + physical.sensors, device=device, dtype=torch.long),
                            torch.as_tensor(physical.valid, device=device, dtype=torch.bool),
                        )

                    actions_t, flat_t, valid_t = time_stage(device, profile_enabled, stage_buckets, "graph_action_tensor_prep_h2d", action_tensor_prep)

                    def select_actions():
                        candidate_scores = torch.gather(score_t.reshape(len(live_pos), -1), 1, flat_t)
                        candidate_scores.masked_fill_(~valid_t, -torch.inf)
                        idx = torch.argmax(candidate_scores, dim=1)
                        return torch.gather(actions_t, 1, idx[:, None]).squeeze(1)

                    best = time_stage(device, profile_enabled, stage_buckets, "graph_decision_select_device", select_actions)

                def action_d2h():
                    if pinned_action_cpu is not None:
                        view = pinned_action_cpu[: len(live_pos)]
                        view.copy_(best, non_blocking=True)
                        return view.numpy().astype(np.int64, copy=False)
                    return best.cpu().numpy().astype(np.int64, copy=False)

                actions = time_stage(
                    device,
                    profile_enabled,
                    stage_buckets,
                    "graph_decision_action_d2h",
                    action_d2h,
                )
            sync(device)
            plan_round_times.append((time.perf_counter() - t0) * 1000.0)
            batch_sizes.append(len(live_pos))
            next_live: list[int] = []
            def step_envs():
                nonlocal batch_env_step_calls, scalar_env_step_calls
                next_ids: list[int] = []
                if env_vec is not None and len(live_pos) > 0:
                    count = int(len(live_pos))
                    key = tuple(live_pos)
                    live_np = live_pos_np_cache.get(key)
                    if live_np is None:
                        live_np = np.asarray(live_pos, dtype=np.int64)
                        live_pos_np_cache[key] = live_np
                    env_index_buf[:count] = root_env_ids_arr[live_np]
                    env_action_buf[:count] = actions[:count]
                    batch_env_step_fn(
                        env_vec,
                        env_index_buf,
                        env_action_buf,
                        env_dt_buf,
                        env_executed_buf,
                        count,
                    )
                    batch_env_step_calls += 1
                else:
                    scalar_env_step_calls += int(len(live_pos))
                for local_idx, pos in enumerate(live_pos):
                    env_idx = root_env_ids[pos]
                    eng = envs[env_idx]
                    if eng.term_buf[0] or elapsed[pos] >= float(args.window_ms):
                        continue
                    remaining = max(0.0, float(args.window_ms) - float(elapsed[pos]))
                    action = int(actions[local_idx])
                    base, _ = xs_decode_action(action, MAXT)
                    if env_vec is not None:
                        executed_action = int(env_executed_buf[local_idx])
                        if executed_action < 0:
                            continue
                        dwell = float(packed.dwell[pos, int(base) - 1]) if int(base) > 0 else 0.0
                        dt = float(10.0 if int(base) == 0 else dwell)
                        reward = float(eng.rew_buf[0])
                    elif bool(getattr(args, "fast_env_step", False)):
                        dwell = float(packed.dwell[pos, int(base) - 1]) if int(base) > 0 else 0.0
                        reward, dt, executed_action = execute_known_valid_action_fast(eng, action, int(base), dwell, remaining)
                    else:
                        reward, dt, executed_action = execute_first_valid_action(eng, [action], remaining)
                    if executed_action is None or dt <= 0.0:
                        continue
                    logical_action, _sensor = decode_sensor_action(int(executed_action), eng.max_trackers)
                    base, _ = xs_decode_action(int(executed_action), MAXT)
                    if int(logical_action) == 0:
                        search_debt[env_idx] = 0.0
                        search_count[pos] += 1
                    else:
                        search_debt[env_idx] += max(float(dt), 0.0)
                        if int(base) > 0:
                            selected[pos].add(int(base))
                            if 0 <= int(base) < selected_t_all.shape[1]:
                                selected_t_all[pos, int(base)] = True
                        track_count[pos] += 1
                    rewards[env_idx] += float(reward)
                    elapsed[pos] += float(dt)
                    executed[env_idx] += 1
                    last[pos] = int(base)
                    current_slots[pos, 0] = float(elapsed[pos]) / float(args.window_ms)
                    current_slots[pos, 1] = float(search_count[pos]) / 20.0
                    current_slots[pos, 2] = float(track_count[pos]) / 100.0
                    current_slots[pos, 3] = 1.0 if int(last[pos]) == 0 else 0.0
                    if not eng.term_buf[0] and elapsed[pos] < float(args.window_ms):
                        next_ids.append(pos)
                return next_ids

            next_live = time_stage(device, profile_enabled, stage_buckets, "graph_env_step_batch", step_envs)
            live_pos = next_live
            depth += 1
        depth_counts.append(depth)
        windows_done += 1
    sync(device)
    wall_ms = (time.perf_counter() - wall0) * 1000.0
    if env_vec is not None and hasattr(radar_binding, "vec_release"):
        radar_binding.vec_release(env_vec)
        env_vec = None
    total_env_windows = int(windows_done * len(envs))
    return {
        "wall_ms": float(wall_ms),
        "windows_requested": int(args.windows * len(envs)),
        "window_rounds": int(windows_done),
        "envs": int(len(envs)),
        "planned_env_windows": int(total_env_windows),
        "window_throughput_per_s": float(1000.0 * total_env_windows / max(wall_ms, 1e-12)),
        "encode_stats": stats(encode_times),
        "graph_build_stats": stats(graph_build_times),
        "neural_rounds": int(len(plan_round_times)),
        "root_graph_replays": int(root_graph_replays),
        "root_raw_encodes": int(root_raw_encodes),
        "graph_replay_rounds": int(graph_replay_rounds),
        "graph_select_replay_rounds": int(graph_select_replay_rounds),
        "padded_graph_rounds": int(padded_graph_rounds),
        "raw_rounds": int(raw_rounds),
        "batch_env_step_calls": int(batch_env_step_calls),
        "scalar_env_step_calls": int(scalar_env_step_calls),
        "pinned_action_d2h": bool(pinned_action_cpu is not None),
        "mean_batch_size": float(np.mean(batch_sizes)) if batch_sizes else 0.0,
        "batch_size_distribution": int_distribution(batch_sizes, full_size=len(envs)),
        "mean_depth": float(np.mean(depth_counts)) if depth_counts else 0.0,
        "depth_distribution": int_distribution(depth_counts),
        "planning_round_stats": stats(plan_round_times),
        "planning_ms_per_env_action": float(sum(plan_round_times) / max(1, sum(batch_sizes))),
        "total_reward": float(sum(rewards)),
        "executed_actions": int(sum(executed)),
        "stage_profile": profile_summary(stage_buckets) if profile_enabled else {},
    }


def build_envs(args, env_cfg):
    from repaired_campaign_tools import EDFPlanner, build_env
    from two_sensor_physical_head_eval import MAXT

    envs = []
    for idx in range(int(args.envs)):
        seed = int(args.seed) + idx
        eng = build_env(EDFPlanner(MAXT), int(args.initial_targets), MAXT, seed, int(args.window_ms), env_cfg)
        eng.reset(seed=seed)
        envs.append(eng)
    return envs


def main() -> None:
    global _CUDA_EVENT_STAGE_BUCKETS
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--window-ms", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--initial-targets", type=int, default=60)
    parser.add_argument("--rate", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--paired-heads", action="store_true", help="Use inference-only paired policy/Q MLP head execution.")
    parser.add_argument("--direct-couplers", action="store_true", help="Call one-layer TransformerEncoder couplers through their layer directly.")
    parser.add_argument("--manual-couplers", action="store_true", help="Use a manual exact single-layer TransformerEncoder path for couplers.")
    parser.add_argument("--cached-action-table", action="store_true", help="Cache per-window physical action ordering/layout in the graph path.")
    parser.add_argument("--gpu-action-template", action="store_true", help="Keep cached action IDs and score indices resident on GPU.")
    parser.add_argument("--gpu-valid-mask", action="store_true", help="Derive per-decision action validity from cached GPU bases and selected masks.")
    parser.add_argument("--padded-live-graph", action="store_true", help="Replay the full-batch score graph for partial live batches and gather live rows.")
    parser.add_argument("--full-select-graph", action="store_true", help="Use a CUDA graph that captures score replay plus action selection for full-live graph rounds.")
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--profile-cuda-events", action="store_true", help="Also record CUDA-event elapsed time for profiled stages.")
    parser.add_argument("--profile-cpu-top", type=int, default=0, help="Record top cumulative cProfile functions for each benchmark path.")
    parser.add_argument("--fast-env-step", action="store_true", help="Skip redundant per-action observation validation in cached-root env stepping.")
    parser.add_argument("--batch-env-step", action="store_true", help="Use the selected-index C batch step in the cached-root graph path.")
    parser.add_argument("--direct-root-pack", action="store_true", help="Pack cached-root observations directly from C engine buffers.")
    parser.add_argument("--pinned-action-d2h", action="store_true", help="Use a preallocated pinned CPU buffer for graph-path selected action transfers.")
    parser.add_argument(
        "--sdp-backend",
        default="default",
        choices=["default", "math_only", "flash_only", "mem_efficient_only", "cudnn_only", "flash_math", "all_no_cudnn"],
        help="Torch scaled-dot-product attention backend selection for score-body experiments.",
    )
    parser.add_argument(
        "--matmul-precision",
        default="",
        choices=["", "highest", "high", "medium"],
        help="Optional torch.set_float32_matmul_precision value for TF32/FP32 matmul experiments.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional ActionAttentionFactorizedNet state dict to benchmark.")
    parser.add_argument(
        "--paths",
        default="all",
        help="Comma-separated benchmark paths to run: all,serial,reencode,cached,graph.",
    )
    parser.add_argument("--out", type=Path, default=Path("results/perf_lab_multi_env_online_batch.json"))
    args = parser.parse_args()
    if bool(args.profile_cuda_events):
        args.profile_stages = True

    from perf_fast_planner import BatchedActionAttentionScorer, FastActionAttentionPlanner
    from repaired_campaign_tools import env_preset_cfg
    from two_sensor_physical_head_eval import ActionAttentionFactorizedNet

    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)
    if str(args.matmul_precision):
        torch.set_float32_matmul_precision(str(args.matmul_precision))
    device = torch.device(args.device)
    _CUDA_EVENT_STAGE_BUCKETS = {} if bool(args.profile_cuda_events) and device.type == "cuda" else None
    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1

    serial_model = load_model_checkpoint(ActionAttentionFactorizedNet(48, 4, 2).eval(), args.checkpoint)
    batch_model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    batch_model.load_state_dict(serial_model.state_dict())
    serial = FastActionAttentionPlanner(
        serial_model,
        env_cfg,
        device=device,
        use_amp=bool(args.amp),
        use_cuda_graph=True,
        use_gpu_select=True,
        use_paired_heads=bool(args.paired_heads),
        use_direct_couplers=bool(args.direct_couplers),
        use_manual_couplers=bool(args.manual_couplers),
    )
    batched = BatchedActionAttentionScorer(
        batch_model,
        env_cfg,
        device=device,
        use_amp=bool(args.amp),
    )

    requested_paths = {p.strip().lower() for p in str(args.paths).split(",") if p.strip()}
    valid_paths = {"all", "serial", "reencode", "cached", "graph"}
    unknown_paths = requested_paths - valid_paths
    if unknown_paths:
        raise ValueError(f"unknown benchmark path(s): {sorted(unknown_paths)}")
    if not requested_paths or "all" in requested_paths:
        requested_paths = {"serial", "reencode", "cached", "graph"}
    if bool(args.skip_graph):
        requested_paths.discard("graph")

    serial_envs = build_envs(args, env_cfg) if "serial" in requested_paths else []
    batch_envs = build_envs(args, env_cfg) if "reencode" in requested_paths else []
    cached_envs = build_envs(args, env_cfg) if "cached" in requested_paths else []
    graph_envs = build_envs(args, env_cfg) if "graph" in requested_paths else []
    serial_report: dict = {}
    batched_report: dict = {}
    cached_report: dict = {}
    graph_report: dict = {}
    cpu_profiles: dict[str, list[dict[str, object]]] = {}
    active_sdp_state: dict[str, bool] = {}
    try:
        with sdp_backend(str(args.sdp_backend)) as active_sdp_state:
            if "serial" in requested_paths:
                serial_report, cpu_profiles["serial_fast_graph_gpu_select"] = run_maybe_profiled(
                    "serial_fast_graph_gpu_select",
                    lambda: run_serial(serial, serial_envs, args, device),
                    int(args.profile_cpu_top),
                )
            else:
                cpu_profiles["serial_fast_graph_gpu_select"] = []
            if "reencode" in requested_paths:
                batched_report, cpu_profiles["batched_multi_env_reencode"] = run_maybe_profiled(
                    "batched_multi_env_reencode",
                    lambda: run_batched(batched, batch_envs, args, device),
                    int(args.profile_cpu_top),
                )
            else:
                cpu_profiles["batched_multi_env_reencode"] = []
            if "cached" in requested_paths:
                cached_report, cpu_profiles["batched_multi_env_cached_root"] = run_maybe_profiled(
                    "batched_multi_env_cached_root",
                    lambda: run_batched_cached(serial, cached_envs, args, device),
                    int(args.profile_cpu_top),
                )
            else:
                cpu_profiles["batched_multi_env_cached_root"] = []
            if "graph" in requested_paths:
                graph_report, cpu_profiles["batched_multi_env_cached_root_graph"] = run_maybe_profiled(
                    "batched_multi_env_cached_root_graph",
                    lambda: run_batched_cached_graph(serial, graph_envs, args, device),
                    int(args.profile_cpu_top),
                )
            else:
                graph_report = {}
                cpu_profiles["batched_multi_env_cached_root_graph"] = []
    finally:
        for eng in [*serial_envs, *batch_envs, *cached_envs, *graph_envs]:
            eng.close()

    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "amp": bool(args.amp),
        "sdp_backend": str(args.sdp_backend),
        "active_sdp_state": active_sdp_state,
        "matmul_precision": str(args.matmul_precision) if str(args.matmul_precision) else None,
        "pinned_action_d2h_requested": bool(args.pinned_action_d2h),
        "paths_requested": sorted(requested_paths),
        "envs": int(args.envs),
        "windows": int(args.windows),
        "window_ms": int(args.window_ms),
        "max_depth": int(args.max_depth),
        "initial_targets": int(args.initial_targets),
        "rate": float(args.rate),
        "seed": int(args.seed),
        "serial_fast_graph_gpu_select": serial_report,
        "batched_multi_env_reencode": batched_report,
        "batched_multi_env_cached_root": cached_report,
        "batched_multi_env_cached_root_graph": graph_report,
        "throughput_speedup": float(
            cached_report["window_throughput_per_s"] / max(serial_report["window_throughput_per_s"], 1e-12)
        )
        if cached_report and serial_report
        else None,
        "graph_throughput_speedup": float(
            graph_report.get("window_throughput_per_s", 0.0) / max(serial_report["window_throughput_per_s"], 1e-12)
        )
        if graph_report and serial_report
        else None,
        "reward_delta_cached_minus_serial": float(cached_report["total_reward"] - serial_report["total_reward"])
        if cached_report and serial_report
        else None,
        "reward_delta_graph_minus_serial": float(graph_report.get("total_reward", serial_report["total_reward"]) - serial_report["total_reward"])
        if graph_report and serial_report
        else None,
        "reward_delta_reencode_minus_serial": float(batched_report["total_reward"] - serial_report["total_reward"])
        if batched_report and serial_report
        else None,
        "cpu_profile_top": cpu_profiles if int(args.profile_cpu_top) > 0 else {},
        "cuda_stage_profile": profile_summary(_CUDA_EVENT_STAGE_BUCKETS)
        if _CUDA_EVENT_STAGE_BUCKETS is not None
        else {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
