from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from alphazero_orthodox import base_exact_args, episode_total_reward, parse_floats, parse_ints
from exact_env_mutual import EDFPlanner, ESTPlanner, MAXT, env_cfg_for, run_fixed, xs_decode_action, xs_s_search_action
from final_radar_campaign import get_obs, summarize_window_df
from pufferlib.ocean.radarxs import binding
from repaired_campaign_tools import build_env
try:
    from single_sensor_cadence_probe import FrameAwareQuotaPlanner
except ModuleNotFoundError:
    FrameAwareQuotaPlanner = None
from strict_window_report import execute_plan_until_budget


def make_exact_args(args0) -> SimpleNamespace:
    return base_exact_args(
        SimpleNamespace(
            ckpt="",
            device="cpu",
            windows=args0.windows,
            max_targets_per_episode=128,
            rollouts=1,
            c_puct=1.25,
            expand_top_k=4,
            horizon_windows=1,
            prior_uniform_mix=0.03,
            root_dirichlet_alpha=0.3,
            root_dirichlet_frac=0.0,
            leaf_value_mix=0.5,
            rollout_policy="model",
            prior_mode="factorized",
            policy_target="visits",
            policy_tau=1.0,
            search_alg="puct",
            plan_mode="atomic",
            window_extract="tree_fill",
            select_mode="visits",
            visit_unvisited_first=True,
            head_mode="pv",
            q_utility_weight=0.0,
            q_utility_normalize=False,
            puct_q_transform="raw",
            q_scale=100.0,
            prior_q_beta=0.0,
            prior_search_bias=0.0,
            gumbel_scale=0.0,
            self_play_sample_tau=0.0,
            gamma=0.99,
            env_mode="penalty_only_frame",
            use_arrival_feature=False,
            use_grid_feature=False,
            enable_x_band=False,
            single_sensor=True,
            zero_action_rewards=True,
            track_loss_penalty=8.0,
            target_service_weight=10.0,
            target_service_horizon_ms=3000.0,
            sector_staleness_weight=0.01,
            search_frame_overdue_weight=1.0,
            search_frame_drop_penalty=16.0,
        )
    )


def parse_quota_grid(text: str) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(max(0, int(part)))
    return sorted(set(out))


def parse_behavior_names(text: str) -> list[str]:
    names: list[str] = []
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            continue
        if item == "edf_frame":
            names.extend(["edf", "frame_edf"])
        elif item == "all":
            names.extend(["edf", "frame_edf", "learned_rf", "learned_extra"])
        else:
            names.append(item)
    out: list[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def finite_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            f"{prefix}_n": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
        }
    return {
        f"{prefix}_n": float(x.size),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_p10": float(np.percentile(x, 10)),
        f"{prefix}_p50": float(np.percentile(x, 50)),
        f"{prefix}_p90": float(np.percentile(x, 90)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_mean": float(np.mean(x)),
    }


def quota_features(obs: dict, debt_ms: float, initial: int, rate: float, window: int, windows: int) -> dict[str, float]:
    active = np.asarray(obs.get("active_mask", []), dtype=bool)[:MAXT]
    deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)[: len(active)]
    desired = np.asarray(obs.get("t_desired", []), dtype=np.float32)[: len(active)]
    dwell = np.asarray(obs.get("t_dwell", np.zeros_like(deadline)), dtype=np.float32)[: len(active)]
    tracked = active & (deadline >= 0.0)
    dropped = active & (deadline < 0.0)
    urgent_50 = tracked & (deadline <= dwell + 50.0)
    urgent_100 = tracked & (deadline <= dwell + 100.0)
    grid = np.asarray(obs.get("grid", []), dtype=np.float32)
    age = 3000.0 - grid if grid.size else np.zeros(0, dtype=np.float32)
    feat = {
        "initial": float(initial),
        "rate": float(rate),
        "window_frac": float(window) / max(1.0, float(windows)),
        "debt_ms": float(debt_ms),
        "active_n": float(np.sum(active)),
        "tracked_n": float(np.sum(tracked)),
        "dropped_n": float(np.sum(dropped)),
        "urgent_50_n": float(np.sum(urgent_50)),
        "urgent_100_n": float(np.sum(urgent_100)),
        "tracked_work_ms": float(np.sum(dwell[tracked])) if dwell.size else 0.0,
        "all_work_ms": float(np.sum(dwell[active])) if dwell.size else 0.0,
        "overdue_cells": float(np.sum(age > 3000.0)),
        "dropped_cells": float(np.sum(age > 4500.0)),
        "nearly_overdue_cells": float(np.sum(age > 2500.0)),
        "search_budget_floor": float(np.floor(200.0 / 10.0)),
    }
    feat.update(finite_stats(deadline[tracked], "deadline"))
    feat.update(finite_stats(desired[tracked], "desired"))
    feat.update(finite_stats(dwell[tracked], "dwell"))
    feat.update(finite_stats(age, "grid_age"))
    return feat


def edf_tracks(obs: dict) -> list[int]:
    plan = list(EDFPlanner(MAXT).plan(obs, budget_ms=200))
    return [int(a) for a in plan if xs_decode_action(int(a), MAXT)[0] != 0]


def quota_plan(obs: dict, quota: int, schedule: str = "prefix") -> list[int]:
    tracks = edf_tracks(obs)
    q = max(0, int(quota))
    if q <= 0:
        return tracks
    search = xs_s_search_action(MAXT)
    if str(schedule) != "interleave" or not tracks:
        return [search] * q + tracks
    out: list[int] = []
    stride = max(1, int(np.ceil(len(tracks) / float(q))))
    pos = 0
    for _ in range(q):
        out.append(search)
        out.extend(tracks[pos : pos + stride])
        pos += stride
    out.extend(tracks[pos:])
    return out


def tail_plan(name: str, frame_planner: FrameAwareQuotaPlanner | None, obs: dict, window_ms: float) -> list[int]:
    if str(name) == "frame_edf":
        if frame_planner is None:
            raise ValueError("frame_edf tail requested without frame planner")
        return list(frame_planner.plan(obs, budget_ms=int(window_ms)))
    if str(name) == "edf":
        return list(EDFPlanner(MAXT).plan(obs, budget_ms=int(window_ms)))
    raise ValueError(f"unsupported tail policy: {name}")


def eval_quota_from_snapshot(
    eng,
    snapshot,
    obs: dict,
    debt_ms: float,
    quota: int,
    window_ms: float,
    horizon_windows: int,
    tail_policy: str,
    frame_planner: FrameAwareQuotaPlanner | None,
) -> float:
    binding.vec_restore(eng.env, snapshot)
    reward, _, debt, *_ = execute_plan_until_budget(
        eng,
        quota_plan(obs, int(quota)),
        float(window_ms),
        float(debt_ms),
        "quota_cf",
        0,
        0,
    )
    total = float(reward)
    for h in range(1, max(1, int(horizon_windows))):
        if bool(eng.term_buf[0]):
            break
        obs_h = get_obs(eng, float(debt))
        plan_h = tail_plan(str(tail_policy), frame_planner, obs_h, float(window_ms))
        reward_h, _, debt, *_ = execute_plan_until_budget(
            eng,
            plan_h,
            float(window_ms),
            float(debt),
            "quota_cf_tail",
            0,
            int(h),
        )
        total += float(reward_h)
    return float(total)


def collect_episode(
    initial: int,
    rate: float,
    seed: int,
    args,
    env_cfg: dict,
    behavior_name: str,
    frame_planner: FrameAwareQuotaPlanner | None,
    behavior_bundle: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    if behavior_name == "edf":
        behavior = EDFPlanner(MAXT)
    elif behavior_name == "frame_edf":
        behavior = frame_planner
    elif str(behavior_name).startswith("learned_"):
        if behavior_bundle is None:
            raise ValueError(f"{behavior_name} behavior requires --behavior-model")
        model_name = str(behavior_name).removeprefix("learned_")
        behavior = LearnedQuotaEDFPlanner(
            behavior_bundle,
            model_name,
            int(initial),
            float(rate),
            int(args.windows),
            q_margin=float(args.q_margin),
            uncertainty_penalty=float(args.learned_uncertainty_penalty),
            schedule=str(args.learned_schedule),
        )
    else:
        raise ValueError(f"unknown behavior policy: {behavior_name}")
    if behavior is None:
        raise ValueError("frame behavior requires a frame planner")
    eng = build_env(behavior, int(initial), MAXT, int(seed), 200, env_cfg)
    eng.reset(seed=int(seed))
    rows: list[dict] = []
    window_rows: list[dict] = []
    debt = 0.0
    cumulative = 0.0
    try:
        for window in range(int(args.windows)):
            if bool(eng.term_buf[0]):
                break
            obs = get_obs(eng, debt)
            snapshot = binding.vec_snapshot(eng.env)
            scores = {
                int(q): eval_quota_from_snapshot(
                    eng,
                    snapshot,
                    obs,
                    debt,
                    int(q),
                    float(args.window_ms),
                    int(args.cf_horizon_windows),
                    str(args.cf_tail_policy),
                    frame_planner,
                )
                for q in args.quota_grid
            }
            binding.vec_restore(eng.env, snapshot)
            vals = np.asarray([scores[q] for q in args.quota_grid], dtype=np.float64)
            best_value = float(np.max(vals))
            tied = [int(q) for q in args.quota_grid if scores[int(q)] >= best_value - float(args.tie_tolerance)]
            best_q = min(tied)
            baseline_q = 0 if 0 in scores else min(scores)
            row = {
                "initial": int(initial),
                "rate": float(rate),
                "seed": int(seed),
                "window": int(window),
                "behavior": behavior_name,
                "best_q": int(best_q),
                "best_reward": best_value,
                "edf_reward": float(scores[baseline_q]),
                "baseline_q": int(baseline_q),
                "frame_q": int(frame_planner._quota(obs, float(args.window_ms))) if frame_planner is not None else -1,
                **quota_features(obs, debt, int(initial), float(rate), int(window), int(args.windows)),
            }
            for q, score in scores.items():
                row[f"q{q}_reward"] = float(score)
            rows.append(row)

            binding.vec_restore(eng.env, snapshot)
            plan = behavior.plan(obs, budget_ms=int(args.window_ms))
            reward, spent_ms, debt, executed, searches, _ = execute_plan_until_budget(
                eng,
                plan,
                float(args.window_ms),
                float(debt),
                behavior_name,
                int(seed),
                int(window),
            )
            cumulative += float(reward)
            window_rows.append(
                {
                    "initial": int(initial),
                    "rate": float(rate),
                    "seed": int(seed),
                    "window": int(window),
                    "behavior": behavior_name,
                    "window_reward": float(reward),
                    "cumulative_reward": float(cumulative),
                    "search_fraction": float(searches / max(1, executed)),
                    "executed_actions": int(executed),
                    "spent_ms": float(spent_ms),
                }
            )
    finally:
        eng.close()
    return rows, window_rows


def collect_dataset(args, exact_args, out_path: Path, behavior_bundle: dict | None = None) -> pd.DataFrame:
    if out_path.exists() and not bool(args.rebuild_data):
        return pd.read_csv(out_path)
    all_rows: list[dict] = []
    all_behavior: list[dict] = []
    for seed in parse_ints(args.train_seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                env_cfg = env_cfg_for(float(rate), exact_args)
                frame = FrameAwareQuotaPlanner(
                    EDFPlanner(MAXT),
                    min_quota=int(args.frame_min_quota),
                    max_quota=int(args.frame_max_quota),
                    desired_ms=float(args.frame_desired_ms),
                    deadline_ms=float(args.frame_deadline_ms),
                    cells_per_search=int(args.frame_cells_per_search),
                )
                behaviors = parse_behavior_names(str(args.behaviors))
                for behavior in behaviors:
                    rows, brow = collect_episode(
                        int(initial),
                        float(rate),
                        int(seed),
                        args,
                        env_cfg,
                        behavior,
                        frame,
                        behavior_bundle=behavior_bundle,
                    )
                    all_rows.extend(rows)
                    all_behavior.extend(brow)
                    print(
                        {"collect": behavior, "initial": initial, "rate": rate, "seed": seed, "rows": len(rows)},
                        flush=True,
                    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)
    pd.DataFrame(all_behavior).to_csv(out_path.with_name(out_path.stem + "_behavior_windows.csv"), index=False)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    blocked_prefixes = ("q",)
    blocked = {"best_q", "best_reward", "edf_reward", "behavior"}
    cols = []
    for col in df.columns:
        if col in blocked or col in {"seed", "window"}:
            continue
        if any(str(col).startswith(prefix) and str(col).endswith("_reward") for prefix in blocked_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(str(col))
    return cols


def expand_quota_training(df: pd.DataFrame, quotas: list[int], feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y = []
    for row in df.itertuples(index=False):
        base = np.asarray([float(getattr(row, c)) for c in feat_cols], dtype=np.float32)
        edf_reward = float(getattr(row, "edf_reward"))
        for q in quotas:
            score = float(getattr(row, f"q{q}_reward"))
            x_rows.append(np.concatenate([base, np.asarray([float(q), float(q) / 20.0], dtype=np.float32)]))
            y.append(score - edf_reward)
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y, dtype=np.float32)


def train_models(df: pd.DataFrame, quotas: list[int], args, model_path: Path) -> dict:
    feat_cols = feature_columns(df)
    x, y = expand_quota_training(df, quotas, feat_cols)
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "rf": RandomForestRegressor(
            n_estimators=int(args.trees),
            max_depth=int(args.tree_depth),
            min_samples_leaf=int(args.min_leaf),
            random_state=int(args.model_seed),
            n_jobs=-1,
        ),
        "extra": ExtraTreesRegressor(
            n_estimators=int(args.trees),
            max_depth=int(args.tree_depth),
            min_samples_leaf=int(args.min_leaf),
            random_state=int(args.model_seed) + 11,
            n_jobs=-1,
        ),
    }
    trained = {}
    for name, model in models.items():
        model.fit(x, y)
        trained[name] = model
        pred = np.asarray(model.predict(x), dtype=np.float32)
        print({"fit": name, "mae": float(np.mean(np.abs(pred - y))), "rows": int(x.shape[0])}, flush=True)
    bundle = {
        "models": trained,
        "feature_columns": feat_cols,
        "quotas": list(quotas),
        "metadata": {
            "data_path": str(args.data_out),
            "data_rows": int(len(df)),
            "quota_grid": list(quotas),
            "models": [str(m) for m in trained.keys()],
            "trees": int(args.trees),
            "tree_depth": int(args.tree_depth),
            "min_leaf": int(args.min_leaf),
            "model_seed": int(args.model_seed),
            "cf_horizon_windows": int(args.cf_horizon_windows),
            "cf_tail_policy": str(args.cf_tail_policy),
            "behaviors": str(args.behaviors),
            "behavior_model": str(args.behavior_model),
        },
    }
    if hasattr(args, "behavior_model_sha256") and str(args.behavior_model_sha256):
        bundle["metadata"]["behavior_model_sha256"] = str(args.behavior_model_sha256)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def load_model_bundle(model_path: Path) -> dict:
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "models" not in bundle or "feature_columns" not in bundle or "quotas" not in bundle:
        raise ValueError(f"invalid quota model bundle: {model_path}")
    metadata = dict(bundle.get("metadata", {}))
    metadata["loaded_model_path"] = str(model_path)
    try:
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        metadata["loaded_model_sha256"] = digest
    except OSError:
        pass
    bundle["metadata"] = metadata
    print(
        {
            "loaded_model": str(model_path),
            "models": sorted([str(k) for k in bundle["models"].keys()]),
            "quotas": [int(q) for q in bundle["quotas"]],
            "feature_columns": int(len(bundle["feature_columns"])),
            "sha256": metadata.get("loaded_model_sha256", "")[:16],
        },
        flush=True,
    )
    return bundle


def selector_feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = {"selected", "seed", "window"}
    cols = []
    for col in df.columns:
        if col in blocked or str(col).startswith("score_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(str(col))
    return cols


def train_selector(teacher_df: pd.DataFrame, args, out_path: Path) -> dict:
    feat_cols = selector_feature_columns(teacher_df)
    x = teacher_df[feat_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y = teacher_df["selected"].astype(str).to_numpy()
    all_models = {
        "rf": RandomForestClassifier(
            n_estimators=int(args.selector_trees),
            max_depth=int(args.selector_depth),
            min_samples_leaf=int(args.selector_min_leaf),
            class_weight="balanced",
            random_state=int(args.model_seed) + 101,
            n_jobs=-1,
        ),
        "extra": ExtraTreesClassifier(
            n_estimators=int(args.selector_trees),
            max_depth=int(args.selector_depth),
            min_samples_leaf=int(args.selector_min_leaf),
            class_weight="balanced",
            random_state=int(args.model_seed) + 202,
            n_jobs=-1,
        ),
    }
    trained = {}
    requested = [m.strip() for m in str(args.selector_models).split(",") if m.strip()]
    for name in requested:
        if name not in all_models:
            raise ValueError(f"unknown selector model: {name}")
        model = all_models[name]
        model.fit(x, y)
        pred = model.predict(x)
        print({"selector_fit": name, "acc": float(np.mean(pred == y)), "rows": int(len(y))}, flush=True)
        trained[name] = model
    bundle = {"models": trained, "feature_columns": feat_cols}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    return bundle


def train_value_selector(teacher_df: pd.DataFrame, args, out_path: Path) -> dict:
    feat_cols = selector_feature_columns(teacher_df)
    score_cols = [str(c) for c in teacher_df.columns if str(c).startswith("score_")]
    if not score_cols:
        raise ValueError("teacher data must include score_* columns for value selector training")
    candidates = [c.removeprefix("score_") for c in score_cols]
    x = teacher_df[feat_cols].fillna(0.0).to_numpy(dtype=np.float32)
    score_mat = teacher_df[score_cols].fillna(-1e9).to_numpy(dtype=np.float32)
    base = np.max(score_mat[:, [i for i, c in enumerate(candidates) if c in {"EDF", "EST"}]], axis=1)
    y = score_mat - base[:, None]
    all_models = {
        "rf": RandomForestRegressor(
            n_estimators=int(args.value_selector_trees),
            max_depth=int(args.value_selector_depth),
            min_samples_leaf=int(args.value_selector_min_leaf),
            random_state=int(args.model_seed) + 303,
            n_jobs=-1,
        ),
        "extra": ExtraTreesRegressor(
            n_estimators=int(args.value_selector_trees),
            max_depth=int(args.value_selector_depth),
            min_samples_leaf=int(args.value_selector_min_leaf),
            random_state=int(args.model_seed) + 404,
            n_jobs=-1,
        ),
    }
    trained = {}
    requested = [m.strip() for m in str(args.value_selector_models).split(",") if m.strip()]
    for name in requested:
        if name not in all_models:
            raise ValueError(f"unknown value selector model: {name}")
        model = all_models[name]
        model.fit(x, y)
        pred = np.asarray(model.predict(x), dtype=np.float32)
        mae = np.mean(np.abs(pred - y), axis=0)
        pick_acc = float(np.mean(np.argmax(pred, axis=1) == np.argmax(y, axis=1)))
        print(
            {
                "value_selector_fit": name,
                "pick_acc": pick_acc,
                "mae_mean": float(np.mean(mae)),
                "rows": int(x.shape[0]),
            },
            flush=True,
        )
        trained[name] = model
    bundle = {"models": trained, "feature_columns": feat_cols, "candidates": candidates}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    return bundle


def load_teacher_frames(primary_path: str, extra_paths: str = "") -> pd.DataFrame:
    paths = [p.strip() for p in str(primary_path).split(",") if p.strip()]
    paths.extend([p.strip() for p in str(extra_paths).split(",") if p.strip()])
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        score_cols = [str(c) for c in df.columns if str(c).startswith("score_")]
        if not score_cols:
            raise ValueError(f"teacher data lacks score_* columns: {path}")
        clean = df.dropna(subset=score_cols).copy()
        if "selected" not in clean.columns:
            raise ValueError(f"teacher data lacks selected column: {path}")
        clean["teacher_source_file"] = str(path)
        frames.append(clean)
    if not frames:
        raise ValueError("no teacher data paths provided")
    out = pd.concat(frames, ignore_index=True, sort=False)
    print(
        {
            "teacher_rows": int(len(out)),
            "teacher_sources": int(len(frames)),
            "teacher_selected_counts": out["selected"].astype(str).value_counts().to_dict(),
        },
        flush=True,
    )
    return out


class LearnedQuotaEDFPlanner:
    def __init__(
        self,
        bundle: dict,
        model_name: str,
        initial: int,
        rate: float,
        windows: int,
        q_margin: float = 0.0,
        uncertainty_penalty: float = 0.0,
        schedule: str = "prefix",
    ):
        self.model = bundle["models"][str(model_name)]
        self.feat_cols = list(bundle["feature_columns"])
        self.quotas = [int(q) for q in bundle["quotas"]]
        self.initial = int(initial)
        self.rate = float(rate)
        self.windows = int(windows)
        self.q_margin = float(q_margin)
        self.uncertainty_penalty = float(uncertainty_penalty)
        self.schedule = str(schedule)
        self.window = 0

    def warmup(self, obs, budget_ms=200):
        q = self._predict(obs, budget_ms=budget_ms)
        return quota_plan(obs, int(q), str(self.schedule))

    def _predict(self, obs, budget_ms=200) -> int:
        _, pred = self._predict_scores(obs, budget_ms=budget_ms)
        score = np.asarray(pred, dtype=np.float64)
        if self.uncertainty_penalty > 0.0:
            _, pred_std = self._predict_tree_std(obs, budget_ms=budget_ms)
            score = score - float(self.uncertainty_penalty) * np.asarray(pred_std, dtype=np.float64)
        best = float(np.max(score))
        tied = [q for q, p in zip(self.quotas, score) if float(p) >= best - self.q_margin]
        return int(min(tied))

    def _predict_scores(self, obs, budget_ms=200) -> tuple[dict[str, float], np.ndarray]:
        feat = quota_features(obs, float(obs.get("search_debt_ms", 0.0)), self.initial, self.rate, self.window, self.windows)
        base = np.asarray([float(feat.get(c, 0.0)) for c in self.feat_cols], dtype=np.float32)
        x = np.asarray(
            [np.concatenate([base, np.asarray([float(q), float(q) / 20.0], dtype=np.float32)]) for q in self.quotas],
            dtype=np.float32,
        )
        pred = np.asarray(self.model.predict(x), dtype=np.float64)
        return feat, pred

    def _predict_tree_std(self, obs, budget_ms=200) -> tuple[dict[str, float], np.ndarray]:
        feat, _ = self._predict_scores(obs, budget_ms=budget_ms)
        if not hasattr(self.model, "estimators_"):
            return feat, np.zeros(len(self.quotas), dtype=np.float64)
        base = np.asarray([float(feat.get(c, 0.0)) for c in self.feat_cols], dtype=np.float32)
        x = np.asarray(
            [np.concatenate([base, np.asarray([float(q), float(q) / 20.0], dtype=np.float32)]) for q in self.quotas],
            dtype=np.float32,
        )
        tree_pred = np.asarray([est.predict(x) for est in self.model.estimators_], dtype=np.float64)
        return feat, np.std(tree_pred, axis=0)

    def plan(self, obs, budget_ms=200):
        q = self._predict(obs, budget_ms=budget_ms)
        self.window += 1
        return quota_plan(obs, int(q), str(self.schedule))


class ShieldedLearnedQuotaPlanner:
    def __init__(
        self,
        learned: LearnedQuotaEDFPlanner,
        fallback: FrameAwareQuotaPlanner,
        args,
    ):
        self.learned = learned
        self.fallback = fallback
        self.args = args

    def _unsafe(self, obs) -> bool:
        feat = quota_features(
            obs,
            float(obs.get("search_debt_ms", 0.0)),
            self.learned.initial,
            self.learned.rate,
            self.learned.window,
            self.learned.windows,
        )
        checks = [
            ("shield_min_dropped_targets", "dropped_n", lambda x, t: x >= t),
            ("shield_min_urgent_100_targets", "urgent_100_n", lambda x, t: x >= t),
            ("shield_max_deadline_min", "deadline_min", lambda x, t: x <= t),
            ("shield_min_grid_age_p90", "grid_age_p90", lambda x, t: x >= t),
            ("shield_min_grid_age_max", "grid_age_max", lambda x, t: x >= t),
            ("shield_min_debt_ms", "debt_ms", lambda x, t: x >= t),
        ]
        for arg_name, feat_name, pred in checks:
            threshold = float(getattr(self.args, arg_name))
            if np.isfinite(threshold) and pred(float(feat.get(feat_name, 0.0)), threshold):
                return True
        max_frame_gap = float(getattr(self.args, "shield_max_frame_pred_gap"))
        if np.isfinite(max_frame_gap) and hasattr(self.fallback, "_quota"):
            frame_q = int(self.fallback._quota(obs, float(getattr(self.args, "window_ms", 200))))
            _, pred_values = self.learned._predict_scores(obs)
            best_value = float(np.max(pred_values))
            q_to_value = {int(q): float(v) for q, v in zip(self.learned.quotas, pred_values)}
            frame_value = float(q_to_value.get(frame_q, q_to_value[min(q_to_value, key=lambda q: abs(q - frame_q))]))
            if best_value - frame_value <= max_frame_gap:
                return True
        return False

    def warmup(self, obs, budget_ms=200):
        if self._unsafe(obs):
            return list(self.fallback.plan(obs, budget_ms=int(budget_ms)))
        return self.learned.warmup(obs, budget_ms=budget_ms)

    def plan(self, obs, budget_ms=200):
        if self._unsafe(obs):
            self.learned.window += 1
            return list(self.fallback.plan(obs, budget_ms=int(budget_ms)))
        return self.learned.plan(obs, budget_ms=budget_ms)


class DistilledSupervisorPlanner:
    def __init__(
        self,
        selector_bundle: dict,
        selector_name: str,
        quota_bundle: dict,
        initial: int,
        rate: float,
        windows: int,
        args,
    ):
        self.selector = selector_bundle["models"][str(selector_name)]
        self.feat_cols = list(selector_bundle["feature_columns"])
        self.quota_bundle = quota_bundle
        self.initial = int(initial)
        self.rate = float(rate)
        self.windows = int(windows)
        self.args = args
        self.window = 0

    def _select(self, obs) -> str:
        feat = quota_features(obs, float(obs.get("search_debt_ms", 0.0)), self.initial, self.rate, self.window, self.windows)
        x = np.asarray([[float(feat.get(c, 0.0)) for c in self.feat_cols]], dtype=np.float32)
        return str(self.selector.predict(x)[0])

    def warmup(self, obs, budget_ms=200):
        selected = self._select(obs)
        return candidate_plan(selected, obs, self.quota_bundle, self.initial, self.rate, self.window, self.windows, self.args)

    def plan(self, obs, budget_ms=200):
        selected = self._select(obs)
        plan = candidate_plan(selected, obs, self.quota_bundle, self.initial, self.rate, self.window, self.windows, self.args)
        self.window += 1
        return plan


class ValueDistilledSupervisorPlanner:
    def __init__(
        self,
        value_bundle: dict,
        selector_name: str,
        quota_bundle: dict,
        initial: int,
        rate: float,
        windows: int,
        args,
    ):
        self.selector = value_bundle["models"][str(selector_name)]
        self.feat_cols = list(value_bundle["feature_columns"])
        self.candidates = list(value_bundle["candidates"])
        self.quota_bundle = quota_bundle
        self.initial = int(initial)
        self.rate = float(rate)
        self.windows = int(windows)
        self.args = args
        self.window = 0

    def _select(self, obs) -> str:
        feat = quota_features(obs, float(obs.get("search_debt_ms", 0.0)), self.initial, self.rate, self.window, self.windows)
        x = np.asarray([[float(feat.get(c, 0.0)) for c in self.feat_cols]], dtype=np.float32)
        values = np.asarray(self.selector.predict(x)[0], dtype=np.float64)
        best = int(np.argmax(values))
        return str(self.candidates[best])

    def warmup(self, obs, budget_ms=200):
        selected = self._select(obs)
        return candidate_plan(selected, obs, self.quota_bundle, self.initial, self.rate, self.window, self.windows, self.args)

    def plan(self, obs, budget_ms=200):
        selected = self._select(obs)
        plan = candidate_plan(selected, obs, self.quota_bundle, self.initial, self.rate, self.window, self.windows, self.args)
        self.window += 1
        return plan


def summarize_eval_row(method: str, initial: int, rate: float, seed: int, windows_df: pd.DataFrame, seconds: float) -> dict:
    s = summarize_window_df(windows_df, "fixed")
    return {
        "method": method,
        "initial": int(initial),
        "rate": float(rate),
        "seed": int(seed),
        "reward": float(s.get("reward_per_200ms_eq", 0.0)),
        "total_reward": episode_total_reward(windows_df),
        "search": float(s.get("search_fraction", 0.0)),
        "delay": float(s.get("mean_delay_active", 0.0)),
        "drop": float(s.get("mean_drop_pct_active", 0.0)),
        "latency": float(s.get("planning_ms_per_200ms_eq", 0.0)),
        "windows": int(len(windows_df)),
        "seconds": float(seconds),
    }


def candidate_plan(
    name: str,
    obs: dict,
    bundle: dict,
    initial: int,
    rate: float,
    window: int,
    windows: int,
    args,
) -> list[int]:
    if name == "EDF":
        return list(EDFPlanner(MAXT).plan(obs, budget_ms=int(args.window_ms)))
    if name == "EST":
        return list(ESTPlanner(MAXT).plan(obs, budget_ms=int(args.window_ms)))
    if name == "frame_edf":
        return list(
            FrameAwareQuotaPlanner(
                EDFPlanner(MAXT),
                min_quota=int(args.frame_min_quota),
                max_quota=int(args.frame_max_quota),
                desired_ms=float(args.frame_desired_ms),
                deadline_ms=float(args.frame_deadline_ms),
                cells_per_search=int(args.frame_cells_per_search),
            ).plan(obs, budget_ms=int(args.window_ms))
        )
    if name.startswith("learned_"):
        model_name = name.removeprefix("learned_")
        planner = LearnedQuotaEDFPlanner(
            bundle,
            model_name,
            int(initial),
            float(rate),
            int(windows),
            q_margin=float(args.q_margin),
            uncertainty_penalty=float(args.learned_uncertainty_penalty),
            schedule=str(args.learned_schedule),
        )
        planner.window = int(window)
        return list(planner.warmup(obs, budget_ms=int(args.window_ms)))
    raise ValueError(f"unknown supervisor candidate: {name}")


def eval_candidate_rollout(
    eng,
    snapshot,
    root_obs: dict,
    root_debt: float,
    candidate: str,
    bundle: dict,
    initial: int,
    rate: float,
    window: int,
    windows: int,
    args,
    horizon_windows: int | None = None,
) -> float:
    binding.vec_restore(eng.env, snapshot)
    total = 0.0
    debt = float(root_debt)
    horizon = int(args.supervisor_horizon_windows) if horizon_windows is None else int(horizon_windows)
    for h in range(max(1, horizon)):
        if bool(eng.term_buf[0]):
            break
        obs = root_obs if h == 0 else get_obs(eng, debt)
        plan_name = candidate if str(args.supervisor_tail_policy) == "same" or h == 0 else str(args.supervisor_tail_policy)
        plan = candidate_plan(plan_name, obs, bundle, int(initial), float(rate), int(window) + h, int(windows), args)
        reward, _, debt, *_ = execute_plan_until_budget(
            eng,
            plan,
            float(args.window_ms),
            float(debt),
            f"supervisor_cf_{plan_name}",
            0,
            int(window) + h,
        )
        total += float(reward)
    return float(total)


def parse_supervisor_horizons(text: str, fallback: int) -> list[int]:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        values = [int(fallback)]
    return sorted(set(max(1, int(v)) for v in values))


def parse_rate_model_map(text: str) -> dict[float, str]:
    mapping: dict[float, str] = {}
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid rate model mapping entry: {item!r}")
        rate_s, model_name = item.split(":", 1)
        mapping[float(rate_s.strip())] = model_name.strip()
    return mapping


def guarded_learned_model_for_rate(rate: float, args) -> str:
    mapping = parse_rate_model_map(str(args.guarded_supervisor_learned_rate_models))
    if not mapping:
        return str(args.guarded_supervisor_learned_model)
    for key, model_name in mapping.items():
        if abs(float(rate) - float(key)) < 1e-9:
            return str(model_name)
    return str(args.guarded_supervisor_learned_model)


def aggregate_robust_scores(scores_by_horizon: dict[int, dict[str, float]], mode: str, risk_lambda: float) -> dict[str, float]:
    candidates = list(next(iter(scores_by_horizon.values())).keys())
    aggregated: dict[str, float] = {}
    for name in candidates:
        values = np.asarray([float(scores[name]) for scores in scores_by_horizon.values()], dtype=np.float64)
        if mode == "mean":
            score = float(np.mean(values))
        elif mode == "mean_std":
            score = float(np.mean(values) - float(risk_lambda) * np.std(values))
        elif mode == "min":
            score = float(np.min(values))
        else:
            raise ValueError(f"unknown robust supervisor mode: {mode}")
        aggregated[name] = score
    return aggregated


def predict_value_selector(value_bundle: dict, selector_name: str, obs: dict, initial: int, rate: float, window: int, windows: int) -> tuple[str, float, float, dict[str, float]]:
    model = value_bundle["models"][str(selector_name)]
    feat_cols = list(value_bundle["feature_columns"])
    candidates = list(value_bundle["candidates"])
    feat = quota_features(obs, float(obs.get("search_debt_ms", 0.0)), int(initial), float(rate), int(window), int(windows))
    x = np.asarray([[float(feat.get(c, 0.0)) for c in feat_cols]], dtype=np.float32)
    pred = np.asarray(model.predict(x)[0], dtype=np.float64)
    order = np.argsort(pred)
    best_idx = int(order[-1])
    second_idx = int(order[-2]) if len(order) > 1 else best_idx
    gap = float(pred[best_idx] - pred[second_idx])
    top_std = 0.0
    if hasattr(model, "estimators_"):
        tree_pred = np.asarray([est.predict(x)[0] for est in model.estimators_], dtype=np.float64)
        if tree_pred.ndim == 2 and tree_pred.shape[1] == len(candidates):
            top_std = float(np.std(tree_pred[:, best_idx]))
    values = {f"pred_{name}": float(value) for name, value in zip(candidates, pred)}
    return str(candidates[best_idx]), gap, top_std, values


def run_supervisor(initial: int, rate: float, seed: int, env_cfg: dict, bundle: dict, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    eng = build_env(EDFPlanner(MAXT), int(initial), MAXT, int(seed), int(args.window_ms), env_cfg)
    eng.reset(seed=int(seed))
    debt = 0.0
    cumulative = 0.0
    rows: list[dict] = []
    choices: list[dict] = []
    candidates = [m.strip() for m in str(args.supervisor_methods).split(",") if m.strip()]
    tie_order = {name: idx for idx, name in enumerate(candidates)}
    try:
        for window in range(int(args.windows)):
            if bool(eng.term_buf[0]):
                break
            obs = get_obs(eng, debt)
            root_debt = float(debt)
            snapshot = binding.vec_snapshot(eng.env)
            t0 = time.perf_counter()
            robust_horizons = parse_supervisor_horizons(
                str(args.supervisor_robust_horizons),
                int(args.supervisor_horizon_windows),
            )
            scores_by_horizon = {
                horizon: {
                    name: eval_candidate_rollout(
                        eng,
                        snapshot,
                        obs,
                        debt,
                        name,
                        bundle,
                        int(initial),
                        float(rate),
                        int(window),
                        int(args.windows),
                        args,
                        horizon_windows=int(horizon),
                    )
                    for name in candidates
                }
                for horizon in robust_horizons
            }
            robust_scores_by_horizon = (
                {
                    horizon: {name: float(score) / max(1, int(horizon)) for name, score in horizon_scores.items()}
                    for horizon, horizon_scores in scores_by_horizon.items()
                }
                if bool(args.supervisor_robust_normalize)
                else scores_by_horizon
            )
            scores = (
                next(iter(robust_scores_by_horizon.values()))
                if len(robust_horizons) == 1
                else aggregate_robust_scores(
                    robust_scores_by_horizon,
                    str(args.supervisor_robust_mode),
                    float(args.supervisor_robust_lambda),
                )
            )
            binding.vec_restore(eng.env, snapshot)
            best_score = max(scores.values())
            raw_selected = sorted([name for name, score in scores.items() if score >= best_score - float(args.supervisor_tie_tolerance)], key=lambda n: tie_order[n])[0]
            heuristic_scores = {name: score for name, score in scores.items() if name in {"EDF", "EST"}}
            heuristic_selected = max(heuristic_scores, key=heuristic_scores.get) if heuristic_scores else raw_selected
            heuristic_best_score = float(heuristic_scores[heuristic_selected]) if heuristic_scores else float(best_score)
            if raw_selected not in {"EDF", "EST"} and float(scores[raw_selected]) < heuristic_best_score + float(args.supervisor_heuristic_margin):
                selected = heuristic_selected
                selection_source = "heuristic_margin"
            else:
                selected = raw_selected
                selection_source = "rollout"
            plan = candidate_plan(selected, obs, bundle, int(initial), float(rate), int(window), int(args.windows), args)
            plan_ms = (time.perf_counter() - t0) * 1000.0
            reward, spent_ms, debt, executed, searches, _ = execute_plan_until_budget(
                eng,
                plan,
                float(args.window_ms),
                float(debt),
                "supervisor_mpc",
                int(seed),
                int(window),
            )
            cumulative += float(reward)
            state = {
                "active_targets": np.nan,
                "tracked_targets": np.nan,
                "drop_pct_active": np.nan,
                "mean_delay_active": np.nan,
            }
            try:
                from strict_window_report import sample_state_metrics

                state = sample_state_metrics(eng, float(debt))
            except Exception:
                pass
            rows.append(
                {
                    "planner": "supervisor_mpc",
                    "seed": int(seed),
                    "window": int(window),
                    "elapsed_ms": float((window + 1) * int(args.window_ms)),
                    "window_reward": float(reward),
                    "cumulative_reward": float(cumulative),
                    "search_fraction": float(searches / max(1, executed)),
                    "planning_ms_per_decision": float(plan_ms),
                    "planning_ms_per_executed_action": float(plan_ms / max(1, executed)),
                    "executed_actions": int(executed),
                    "spent_ms": float(spent_ms),
                    **state,
                }
            )
            choices.append(
                {
                    "initial": int(initial),
                    "rate": float(rate),
                    "seed": int(seed),
                    "window": int(window),
                    "selected": selected,
                    "raw_selected": raw_selected,
                    "selection_source": selection_source,
                    "heuristic_selected": heuristic_selected,
                    "heuristic_best_score": heuristic_best_score,
                    "selected_score": float(scores[selected]),
                    "robust_horizons": ",".join(str(h) for h in robust_horizons),
                    "robust_mode": str(args.supervisor_robust_mode) if len(robust_horizons) > 1 else "single",
                    "robust_normalize": bool(args.supervisor_robust_normalize),
                    **quota_features(obs, root_debt, int(initial), float(rate), int(window), int(args.windows)),
                    **{f"score_{name}": float(score) for name, score in scores.items()},
                    **{
                        f"score_h{horizon}_{name}": float(score)
                        for horizon, horizon_scores in scores_by_horizon.items()
                        for name, score in horizon_scores.items()
                    },
                }
            )
    finally:
        eng.close()
    return pd.DataFrame(rows), pd.DataFrame(choices)


def run_selective_value_supervisor(
    initial: int,
    rate: float,
    seed: int,
    env_cfg: dict,
    bundle: dict,
    value_bundle: dict,
    selector_name: str,
    args,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eng = build_env(EDFPlanner(MAXT), int(initial), MAXT, int(seed), int(args.window_ms), env_cfg)
    eng.reset(seed=int(seed))
    debt = 0.0
    cumulative = 0.0
    rows: list[dict] = []
    choices: list[dict] = []
    candidates = list(value_bundle["candidates"])
    tie_order = {name: idx for idx, name in enumerate(candidates)}
    try:
        for window in range(int(args.windows)):
            if bool(eng.term_buf[0]):
                break
            obs = get_obs(eng, debt)
            root_debt = float(debt)
            t0 = time.perf_counter()
            predicted, pred_gap, pred_top_std, pred_values = predict_value_selector(
                value_bundle,
                str(selector_name),
                obs,
                int(initial),
                float(rate),
                int(window),
                int(args.windows),
            )
            defer = (
                pred_gap < float(args.selective_value_min_gap)
                or pred_top_std > float(args.selective_value_max_top_std)
                or int(initial) <= int(args.selective_value_force_mpc_initial_cutoff)
            )
            scores = {}
            if defer:
                snapshot = binding.vec_snapshot(eng.env)
                scores = {
                    name: eval_candidate_rollout(
                        eng,
                        snapshot,
                        obs,
                        debt,
                        name,
                        bundle,
                        int(initial),
                        float(rate),
                        int(window),
                        int(args.windows),
                        args,
                    )
                    for name in candidates
                }
                binding.vec_restore(eng.env, snapshot)
                best_score = max(scores.values())
                selected = sorted(
                    [name for name, score in scores.items() if score >= best_score - float(args.supervisor_tie_tolerance)],
                    key=lambda n: tie_order[n],
                )[0]
                source = "mpc"
            else:
                selected = predicted
                source = "student"
            plan = candidate_plan(selected, obs, bundle, int(initial), float(rate), int(window), int(args.windows), args)
            plan_ms = (time.perf_counter() - t0) * 1000.0
            method = f"selective_value_{selector_name}"
            reward, spent_ms, debt, executed, searches, _ = execute_plan_until_budget(
                eng,
                plan,
                float(args.window_ms),
                float(debt),
                method,
                int(seed),
                int(window),
            )
            cumulative += float(reward)
            state = {
                "active_targets": np.nan,
                "tracked_targets": np.nan,
                "drop_pct_active": np.nan,
                "mean_delay_active": np.nan,
            }
            try:
                from strict_window_report import sample_state_metrics

                state = sample_state_metrics(eng, float(debt))
            except Exception:
                pass
            rows.append(
                {
                    "planner": method,
                    "seed": int(seed),
                    "window": int(window),
                    "elapsed_ms": float((window + 1) * int(args.window_ms)),
                    "window_reward": float(reward),
                    "cumulative_reward": float(cumulative),
                    "search_fraction": float(searches / max(1, executed)),
                    "planning_ms_per_decision": float(plan_ms),
                    "planning_ms_per_executed_action": float(plan_ms / max(1, executed)),
                    "executed_actions": int(executed),
                    "spent_ms": float(spent_ms),
                    **state,
                }
            )
            choices.append(
                {
                    "initial": int(initial),
                    "rate": float(rate),
                    "seed": int(seed),
                    "window": int(window),
                    "selected": selected,
                    "predicted": predicted,
                    "source": source,
                    "pred_gap": float(pred_gap),
                    "pred_top_std": float(pred_top_std),
                    **quota_features(obs, root_debt, int(initial), float(rate), int(window), int(args.windows)),
                    **pred_values,
                    **{f"score_{name}": float(score) for name, score in scores.items()},
                }
            )
    finally:
        eng.close()
    return pd.DataFrame(rows), pd.DataFrame(choices)


def eval_methods(
    args,
    exact_args,
    bundle: dict,
    selector_bundle: dict | None = None,
    value_selector_bundle: dict | None = None,
) -> pd.DataFrame:
    rows = []
    frames = []
    model_names = [m.strip() for m in str(args.models).split(",") if m.strip()]
    for seed in parse_ints(args.test_seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                env_cfg = env_cfg_for(float(rate), exact_args)
                planners = {
                    "EDF": EDFPlanner(MAXT),
                    "EST": ESTPlanner(MAXT),
                    "frame_edf": FrameAwareQuotaPlanner(
                        EDFPlanner(MAXT),
                        min_quota=int(args.frame_min_quota),
                        max_quota=int(args.frame_max_quota),
                        desired_ms=float(args.frame_desired_ms),
                        deadline_ms=float(args.frame_deadline_ms),
                        cells_per_search=int(args.frame_cells_per_search),
                    ),
                }
                for model_name in model_names:
                    planners[f"learned_{model_name}"] = LearnedQuotaEDFPlanner(
                        bundle,
                        model_name,
                        int(initial),
                        float(rate),
                        int(args.windows),
                        q_margin=float(args.q_margin),
                        uncertainty_penalty=float(args.learned_uncertainty_penalty),
                        schedule=str(args.learned_schedule),
                    )
                    if bool(args.shield_learned_tail):
                        planners[f"shielded_learned_{model_name}"] = ShieldedLearnedQuotaPlanner(
                            LearnedQuotaEDFPlanner(
                                bundle,
                                model_name,
                                int(initial),
                                float(rate),
                                int(args.windows),
                                q_margin=float(args.q_margin),
                                uncertainty_penalty=float(args.learned_uncertainty_penalty),
                                schedule=str(args.learned_schedule),
                            ),
                            FrameAwareQuotaPlanner(
                                EDFPlanner(MAXT),
                                min_quota=int(args.frame_min_quota),
                                max_quota=int(args.frame_max_quota),
                                desired_ms=float(args.frame_desired_ms),
                                deadline_ms=float(args.frame_deadline_ms),
                                cells_per_search=int(args.frame_cells_per_search),
                            ),
                            args,
                        )
                if selector_bundle is not None:
                    for selector_name in [m.strip() for m in str(args.selector_models).split(",") if m.strip()]:
                        planners[f"distilled_{selector_name}"] = DistilledSupervisorPlanner(
                            selector_bundle,
                            selector_name,
                            bundle,
                            int(initial),
                            float(rate),
                            int(args.windows),
                            args,
                        )
                if value_selector_bundle is not None:
                    for selector_name in [m.strip() for m in str(args.value_selector_models).split(",") if m.strip()]:
                        planners[f"value_distilled_{selector_name}"] = ValueDistilledSupervisorPlanner(
                            value_selector_bundle,
                            selector_name,
                            bundle,
                            int(initial),
                            float(rate),
                            int(args.windows),
                            args,
                        )
                for name, planner in planners.items():
                    t0 = time.perf_counter()
                    w, _ = run_fixed(planner, name, int(initial), MAXT, int(seed), int(args.windows), int(args.window_ms), env_cfg)
                    row = summarize_eval_row(name, int(initial), float(rate), int(seed), w, time.perf_counter() - t0)
                    rows.append(row)
                    ww = w.copy()
                    ww["method"] = name
                    ww["initial"] = int(initial)
                    ww["rate"] = float(rate)
                    frames.append(ww)
                    print(row, flush=True)
                if bool(args.eval_supervisor):
                    t0 = time.perf_counter()
                    w, choices = run_supervisor(int(initial), float(rate), int(seed), env_cfg, bundle, args)
                    row = summarize_eval_row("supervisor_mpc", int(initial), float(rate), int(seed), w, time.perf_counter() - t0)
                    rows.append(row)
                    ww = w.copy()
                    ww["method"] = "supervisor_mpc"
                    ww["initial"] = int(initial)
                    ww["rate"] = float(rate)
                    frames.append(ww)
                    choices.to_csv(Path(args.out).with_name(Path(args.out).stem + "_supervisor_choices.csv"), mode="a", header=not Path(args.out).with_name(Path(args.out).stem + "_supervisor_choices.csv").exists(), index=False)
                    print(row, flush=True)
                if bool(args.eval_guarded_supervisor):
                    method = f"guarded_mpc_le_{int(args.guarded_supervisor_initial_cutoff)}"
                    t0 = time.perf_counter()
                    if int(initial) <= int(args.guarded_supervisor_initial_cutoff):
                        w, choices = run_supervisor(int(initial), float(rate), int(seed), env_cfg, bundle, args)
                        choices.to_csv(
                            Path(args.out).with_name(Path(args.out).stem + "_guarded_supervisor_choices.csv"),
                            mode="a",
                            header=not Path(args.out).with_name(Path(args.out).stem + "_guarded_supervisor_choices.csv").exists(),
                            index=False,
                        )
                    else:
                        guarded_model_name = guarded_learned_model_for_rate(float(rate), args)
                        planner = LearnedQuotaEDFPlanner(
                            bundle,
                            guarded_model_name,
                            int(initial),
                            float(rate),
                            int(args.windows),
                            q_margin=float(args.q_margin),
                            uncertainty_penalty=float(args.learned_uncertainty_penalty),
                            schedule=str(args.learned_schedule),
                        )
                        if bool(args.shield_learned_tail):
                            planner = ShieldedLearnedQuotaPlanner(
                                planner,
                                FrameAwareQuotaPlanner(
                                    EDFPlanner(MAXT),
                                    min_quota=int(args.frame_min_quota),
                                    max_quota=int(args.frame_max_quota),
                                    desired_ms=float(args.frame_desired_ms),
                                    deadline_ms=float(args.frame_deadline_ms),
                                    cells_per_search=int(args.frame_cells_per_search),
                                ),
                                args,
                            )
                        w, _ = run_fixed(planner, method, int(initial), MAXT, int(seed), int(args.windows), int(args.window_ms), env_cfg)
                    row = summarize_eval_row(method, int(initial), float(rate), int(seed), w, time.perf_counter() - t0)
                    rows.append(row)
                    ww = w.copy()
                    ww["method"] = method
                    ww["initial"] = int(initial)
                    ww["rate"] = float(rate)
                    frames.append(ww)
                    print(row, flush=True)
                if bool(args.eval_selective_value_supervisor):
                    if value_selector_bundle is None:
                        raise ValueError("--eval-selective-value-supervisor requires --teacher-data and --value-selector-models")
                    for selector_name in [m.strip() for m in str(args.value_selector_models).split(",") if m.strip()]:
                        t0 = time.perf_counter()
                        w, choices = run_selective_value_supervisor(
                            int(initial),
                            float(rate),
                            int(seed),
                            env_cfg,
                            bundle,
                            value_selector_bundle,
                            selector_name,
                            args,
                        )
                        method = f"selective_value_{selector_name}"
                        row = summarize_eval_row(method, int(initial), float(rate), int(seed), w, time.perf_counter() - t0)
                        row["defer_rate"] = float(np.mean(choices["source"].astype(str).eq("mpc"))) if len(choices) else 0.0
                        rows.append(row)
                        ww = w.copy()
                        ww["method"] = method
                        ww["initial"] = int(initial)
                        ww["rate"] = float(rate)
                        frames.append(ww)
                        choices.to_csv(
                            Path(args.out).with_name(Path(args.out).stem + f"_selective_value_{selector_name}_choices.csv"),
                            mode="a",
                            header=not Path(args.out).with_name(Path(args.out).stem + f"_selective_value_{selector_name}_choices.csv").exists(),
                            index=False,
                        )
                        print(row, flush=True)
    raw = pd.DataFrame(rows)
    bundle_metadata = dict(bundle.get("metadata", {}))
    for key, value in bundle_metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            raw[f"bundle_{key}"] = value
    raw["best_heuristic_reward"] = raw.groupby(["initial", "rate", "seed"])["reward"].transform(
        lambda s: max(float(s[raw.loc[s.index, "method"].eq("EDF")].iloc[0]), float(s[raw.loc[s.index, "method"].eq("EST")].iloc[0]))
    )
    raw["margin_vs_best_heuristic"] = raw["reward"] - raw["best_heuristic_reward"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out, index=False)
    pd.concat(frames, ignore_index=True).to_csv(out.with_name(out.stem + "_windows.csv"), index=False)
    summary = (
        raw.groupby("method")
        .agg(
            reward=("reward", "mean"),
            search=("search", "mean"),
            delay=("delay", "mean"),
            drop=("drop", "mean"),
            latency=("latency", "mean"),
            defer_rate=("defer_rate", "mean") if "defer_rate" in raw.columns else ("reward", lambda x: np.nan),
            mean_margin=("margin_vs_best_heuristic", "mean"),
            min_margin=("margin_vs_best_heuristic", "min"),
            nonnegative=("margin_vs_best_heuristic", lambda x: float(np.mean(np.asarray(x) >= -1e-9))),
            n=("reward", "size"),
        )
        .reset_index()
        .sort_values("reward", ascending=False)
    )
    by_cell = (
        raw.groupby(["initial", "rate", "method"])
        .agg(
            reward=("reward", "mean"),
            mean_margin=("margin_vs_best_heuristic", "mean"),
            min_margin=("margin_vs_best_heuristic", "min"),
            nonnegative=("margin_vs_best_heuristic", lambda x: float(np.mean(np.asarray(x) >= -1e-9))),
            n=("reward", "size"),
        )
        .reset_index()
        .sort_values(["initial", "rate", "method"])
    )
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    by_cell.to_csv(out.with_name(out.stem + "_by_cell.csv"), index=False)
    if bundle_metadata:
        pd.DataFrame([bundle_metadata]).to_csv(out.with_name(out.stem + "_model_metadata.csv"), index=False)
    print("\nSUMMARY")
    print(summary.to_string(index=False), flush=True)
    print("\nBY_CELL")
    print(by_cell.to_string(index=False), flush=True)
    return raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=60)
    ap.add_argument("--window-ms", type=int, default=200)
    ap.add_argument("--initials", default="20,40,60")
    ap.add_argument("--rates", default="2,4")
    ap.add_argument("--train-seeds", default="901,902,903,904")
    ap.add_argument("--test-seeds", default="913,914,915,916")
    ap.add_argument("--quota-grid", default="0,2,4,6,8,10,12,14,16,18")
    ap.add_argument("--behaviors", default="edf_frame")
    ap.add_argument("--tie-tolerance", type=float, default=0.02)
    ap.add_argument("--cf-horizon-windows", type=int, default=3)
    ap.add_argument("--cf-tail-policy", choices=["edf", "frame_edf"], default="frame_edf")
    ap.add_argument("--q-margin", type=float, default=0.0)
    ap.add_argument("--learned-uncertainty-penalty", type=float, default=0.0)
    ap.add_argument("--learned-schedule", choices=["prefix", "interleave"], default="prefix")
    ap.add_argument("--shield-learned-tail", action="store_true")
    ap.add_argument("--shield-min-dropped-targets", type=float, default=float("inf"))
    ap.add_argument("--shield-min-urgent-100-targets", type=float, default=float("inf"))
    ap.add_argument("--shield-max-deadline-min", type=float, default=float("-inf"))
    ap.add_argument("--shield-min-grid-age-p90", type=float, default=float("inf"))
    ap.add_argument("--shield-min-grid-age-max", type=float, default=float("inf"))
    ap.add_argument("--shield-min-debt-ms", type=float, default=float("inf"))
    ap.add_argument("--shield-max-frame-pred-gap", type=float, default=float("-inf"))
    ap.add_argument("--trees", type=int, default=160)
    ap.add_argument("--tree-depth", type=int, default=10)
    ap.add_argument("--min-leaf", type=int, default=10)
    ap.add_argument("--model-seed", type=int, default=123)
    ap.add_argument("--models", default="ridge,rf,extra")
    ap.add_argument("--eval-supervisor", action="store_true")
    ap.add_argument("--eval-guarded-supervisor", action="store_true")
    ap.add_argument("--eval-selective-value-supervisor", action="store_true")
    ap.add_argument("--supervisor-methods", default="EDF,EST,frame_edf,learned_rf")
    ap.add_argument("--supervisor-horizon-windows", type=int, default=3)
    ap.add_argument("--supervisor-robust-horizons", default="")
    ap.add_argument("--supervisor-robust-mode", choices=["min", "mean", "mean_std"], default="min")
    ap.add_argument("--supervisor-robust-lambda", type=float, default=1.0)
    ap.add_argument("--supervisor-robust-normalize", action="store_true")
    ap.add_argument("--supervisor-tail-policy", choices=["same", "EDF", "EST", "frame_edf", "learned_rf"], default="same")
    ap.add_argument("--supervisor-tie-tolerance", type=float, default=0.02)
    ap.add_argument("--supervisor-heuristic-margin", type=float, default=-1000000000.0)
    ap.add_argument("--guarded-supervisor-initial-cutoff", type=int, default=40)
    ap.add_argument("--guarded-supervisor-learned-model", default="rf")
    ap.add_argument("--guarded-supervisor-learned-rate-models", default="")
    ap.add_argument("--teacher-data", default="")
    ap.add_argument("--teacher-extra-data", default="")
    ap.add_argument("--selector-out", default="CreateValid1/results/window_quota_supervisor_selector.joblib")
    ap.add_argument("--selector-models", default="rf,extra")
    ap.add_argument("--selector-trees", type=int, default=96)
    ap.add_argument("--selector-depth", type=int, default=10)
    ap.add_argument("--selector-min-leaf", type=int, default=8)
    ap.add_argument("--value-selector-out", default="CreateValid1/results/window_quota_supervisor_value_selector.joblib")
    ap.add_argument("--value-selector-models", default="")
    ap.add_argument("--value-selector-trees", type=int, default=128)
    ap.add_argument("--value-selector-depth", type=int, default=12)
    ap.add_argument("--value-selector-min-leaf", type=int, default=8)
    ap.add_argument("--selective-value-min-gap", type=float, default=0.5)
    ap.add_argument("--selective-value-max-top-std", type=float, default=1e9)
    ap.add_argument("--selective-value-force-mpc-initial-cutoff", type=int, default=0)
    ap.add_argument("--frame-min-quota", type=int, default=4)
    ap.add_argument("--frame-max-quota", type=int, default=18)
    ap.add_argument("--frame-desired-ms", type=float, default=3000.0)
    ap.add_argument("--frame-deadline-ms", type=float, default=4500.0)
    ap.add_argument("--frame-cells-per-search", type=int, default=5)
    ap.add_argument("--data-out", default="CreateValid1/results/window_quota_counterfactual_train.csv")
    ap.add_argument("--model-out", default="CreateValid1/results/window_quota_counterfactual_model.joblib")
    ap.add_argument("--load-model", default="")
    ap.add_argument("--behavior-model", default="")
    ap.add_argument("--out", default="CreateValid1/results/window_quota_learner_eval.csv")
    ap.add_argument("--rebuild-data", action="store_true")
    args = ap.parse_args()
    args.quota_grid = parse_quota_grid(args.quota_grid)

    exact_args = make_exact_args(args)
    if str(args.load_model).strip():
        bundle = load_model_bundle(Path(args.load_model))
    else:
        behavior_bundle = None
        if str(args.behavior_model).strip():
            behavior_bundle = load_model_bundle(Path(args.behavior_model))
            args.behavior_model_sha256 = str(behavior_bundle.get("metadata", {}).get("loaded_model_sha256", ""))
        data = collect_dataset(args, exact_args, Path(args.data_out), behavior_bundle=behavior_bundle)
        bundle = train_models(data, args.quota_grid, args, Path(args.model_out))
    selector_bundle = None
    value_selector_bundle = None
    if str(args.teacher_data).strip():
        teacher = load_teacher_frames(args.teacher_data, args.teacher_extra_data)
        selector_bundle = train_selector(teacher, args, Path(args.selector_out))
        if [m.strip() for m in str(args.value_selector_models).split(",") if m.strip()]:
            value_selector_bundle = train_value_selector(teacher, args, Path(args.value_selector_out))
    eval_methods(args, exact_args, bundle, selector_bundle, value_selector_bundle)


if __name__ == "__main__":
    main()
