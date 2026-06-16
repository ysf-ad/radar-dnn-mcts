from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from exact_env_mutual import (
    EDFPlanner,
    ESTPlanner,
    MAXT,
    attach_env_obs,
    build_env,
    engine_env_cfg,
    env_cfg_for,
    execute_first_valid_action,
    get_obs,
    load_model,
    run_fixed,
    run_snapshot_exact_episode,
    summarize_window_df,
    xs_decode_action,
    xs_s_search_action,
    xs_s_track_action,
    xs_x_search_action,
    xs_x_track_action,
)
from mutual_features import slot_features, tokenize
from mutual_foundation import SearchTarget
from repaired_campaign_tools import make_reference_planner
from realistic_reward_retrain import adapter
from mutual_foundation import MutualRadarDirectPlanner
from pufferlib.ocean.radarxs import binding


OUT = Path("results/alphazero_orthodox")
OUT.mkdir(parents=True, exist_ok=True)
_ADAPTER = None


def get_adapter():
    global _ADAPTER
    if _ADAPTER is None:
        _ADAPTER = adapter()
    return _ADAPTER


def parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def output_path_for_prefix(prefix: str, suffix: str) -> Path:
    path = Path(str(prefix))
    if path.parent != Path(".") or path.is_absolute():
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.with_name(path.name + suffix)
    return OUT / f"{prefix}{suffix}"


def save_targets(path: Path, targets: List[SearchTarget]) -> None:
    payload = [
        {
            "x": t.x,
            "slot": t.slot,
            "pi": t.pi,
            "q": t.q,
            "q_mask": t.q_mask,
            "search_count": int(t.search_count),
            "track_count": int(t.track_count),
            "reward": float(t.reward),
            "ret": float(t.ret),
            "sensor_pi": t.sensor_pi,
            "sensor_q": t.sensor_q,
            "sensor_q_mask": t.sensor_q_mask,
            "initial": int(getattr(t, "initial", -1)),
            "rate": float(getattr(t, "rate", 0.0)),
            "seed": int(getattr(t, "seed", -1)),
            "window": int(getattr(t, "window", -1)),
            "action_index": int(getattr(t, "action_index", -1)),
        }
        for t in targets
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path, pickle_protocol=4)


def load_targets(path: Path) -> List[SearchTarget]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "targets" in payload:
        payload = payload["targets"]

    targets: List[SearchTarget] = []
    for item in payload:
        if isinstance(item, SearchTarget):
            targets.append(item)
            continue
        if not isinstance(item, dict):
            raise TypeError(f"unsupported target item type in {path}: {type(item)!r}")
        targets.append(
            SearchTarget(
                x=np.asarray(item["x"], dtype=np.float32),
                slot=np.asarray(item["slot"], dtype=np.float32),
                pi=np.asarray(item["pi"], dtype=np.float32),
                q=np.asarray(item["q"], dtype=np.float32),
                q_mask=np.asarray(item["q_mask"], dtype=np.float32),
                search_count=int(item.get("search_count", 0)),
                track_count=int(item.get("track_count", 0)),
                reward=float(item.get("reward", 0.0)),
                ret=float(item.get("ret", 0.0)),
                sensor_pi=None if item.get("sensor_pi", None) is None else np.asarray(item["sensor_pi"], dtype=np.float32),
                sensor_q=None if item.get("sensor_q", None) is None else np.asarray(item["sensor_q"], dtype=np.float32),
                sensor_q_mask=None if item.get("sensor_q_mask", None) is None else np.asarray(item["sensor_q_mask"], dtype=np.float32),
                initial=int(item.get("initial", -1)),
                rate=float(item.get("rate", 0.0)),
                seed=int(item.get("seed", -1)),
                window=int(item.get("window", -1)),
                action_index=int(item.get("action_index", -1)),
            )
        )
    return targets


def load_target_paths(text: str) -> List[SearchTarget]:
    targets: List[SearchTarget] = []
    for part in str(text).split(";"):
        part = part.strip()
        if not part:
            continue
        loaded = load_targets(Path(part))
        targets.extend(loaded)
        print(f"loaded {len(loaded)} targets from {part}", flush=True)
    return targets


def base_exact_args(args0) -> SimpleNamespace:
    zero_action_rewards = bool(getattr(args0, "zero_action_rewards", False)) or str(getattr(args0, "env_mode", "")) == "penalty_only_frame"
    single_sensor = bool(getattr(args0, "single_sensor", False))
    max_considered = int(getattr(args0, "max_num_considered_actions", 0) or 0)
    if max_considered <= 0:
        max_considered = max(16, int(args0.expand_top_k))
    return SimpleNamespace(
        ckpt=args0.ckpt,
        device=args0.device,
        d_model=96,
        nhead=4,
        nlayers=2,
        head_arch=getattr(args0, "head_arch", "branch_context"),
        windows=args0.windows,
        max_targets_per_episode=args0.max_targets_per_episode,
        rollouts=args0.rollouts,
        c_puct=args0.c_puct,
        expand_top_k=args0.expand_top_k,
        horizon_windows=args0.horizon_windows,
        rollout_policy=getattr(args0, "rollout_policy", "model"),
        branch_rollout_threshold=float(getattr(args0, "branch_rollout_threshold", 0.65)),
        prior_mode=getattr(args0, "prior_mode", "factorized"),
        epsilon=0.0,
        policy_target=getattr(args0, "policy_target", "visits"),
        policy_tau=getattr(args0, "policy_tau", 1.0),
        search_alg=getattr(args0, "search_alg", "puct"),
        max_num_considered_actions=max_considered,
        gumbel_scale=getattr(args0, "gumbel_scale", 0.0),
        mctx_value_scale=float(getattr(args0, "mctx_value_scale", 0.1)),
        mctx_maxvisit_init=float(getattr(args0, "mctx_maxvisit_init", 50.0)),
        eager_edge_depth=1,
        prior_uniform_mix=args0.prior_uniform_mix,
        root_dirichlet_alpha=args0.root_dirichlet_alpha,
        root_dirichlet_frac=args0.root_dirichlet_frac,
        rollout_est_prob=0.5,
        allow_retrack_in_window=False,
        stateless_tree_context=False,
        head_mode=args0.head_mode,
        q_utility_weight=args0.q_utility_weight,
        q_utility_normalize=bool(args0.q_utility_normalize),
        puct_q_transform=getattr(args0, "puct_q_transform", "raw"),
        q_scale=args0.q_scale,
        leaf_value_mix=args0.leaf_value_mix,
        seed_rollout_policies=getattr(args0, "seed_rollout_policies", ""),
        fast_zero_rollout=False,
        skip_default_rollout_seed=bool(getattr(args0, "skip_default_rollout_seed", True)),
        complete_root_q_with_value=False,
        visit_unvisited_first=bool(getattr(args0, "visit_unvisited_first", True)),
        duration_normalize_q=False,
        prior_q_beta=args0.prior_q_beta,
        prior_search_bias=getattr(args0, "prior_search_bias", 0.0),
        adaptive_search_bias=getattr(args0, "adaptive_search_bias", 0.0),
        adaptive_search_target_load=getattr(args0, "adaptive_search_target_load", 0.75),
        forbid_retrack_within_window=True,
        sensor_action_mode=getattr(args0, "sensor_action_mode", "explicit_head"),
        disable_x_search=bool(getattr(args0, "disable_x_search", False)) or single_sensor,
        canonical_search_only=bool(getattr(args0, "canonical_search_only", False)) or single_sensor,
        use_arrival_feature=bool(getattr(args0, "use_arrival_feature", False)),
        use_grid_feature=bool(getattr(args0, "use_grid_feature", False)),
        counterfactual_branch_q=bool(getattr(args0, "counterfactual_branch_q", False)),
        counterfactual_top_k=int(getattr(args0, "counterfactual_top_k", getattr(args0, "cf_top_k", 8))),
        counterfactual_mode=str(getattr(args0, "counterfactual_mode", "q_softmax")),
        counterfactual_subrollouts=int(getattr(args0, "counterfactual_subrollouts", 0)),
        counterfactual_candidate_mode=str(getattr(args0, "counterfactual_candidate_mode", getattr(args0, "cf_candidate_mode", "prior"))),
        target_start_window=int(getattr(args0, "target_start_window", 1)),
        target_stride=int(getattr(args0, "target_stride", 1)),
        plan_mode=getattr(args0, "plan_mode", "atomic"),
        window_extract=getattr(args0, "window_extract", "tree_fill"),
        select_mode=getattr(args0, "select_mode", "visits"),
        load_gated_prior_threshold=int(getattr(args0, "load_gated_prior_threshold", 80)),
        self_play_sample_tau=args0.self_play_sample_tau,
        target_selected_action=bool(getattr(args0, "target_selected_action", False)),
        add_prefix_targets=bool(getattr(args0, "add_prefix_targets", False)),
        gamma=args0.gamma,
        env_mode=args0.env_mode,
        track_update_reward=0.0 if zero_action_rewards else float(getattr(args0, "track_update_reward", 0.30)),
        track_loss_penalty=args0.track_loss_penalty,
        track_urgency_bonus_weight=-1.0,
        target_service_weight=args0.target_service_weight,
        target_service_horizon_ms=args0.target_service_horizon_ms,
        tracked_target_ms_reward_weight=float(getattr(args0, "tracked_target_ms_reward_weight", 0.0)),
        discovered_target_reward=float(getattr(args0, "discovered_target_reward", 0.0)),
        search_refresh_tracked=0,
        search_refresh_gain=0.0,
        search_debt_penalty_weight=0.0,
        sector_staleness_weight=args0.sector_staleness_weight,
        searched_sector_reward_weight=0.0 if zero_action_rewards else float(getattr(args0, "searched_sector_reward_weight", 0.25)),
        search_frame_overdue_weight=args0.search_frame_overdue_weight,
        search_frame_desired_ms=3000.0,
        search_frame_deadline_ms=4500.0,
        search_frame_drop_penalty=args0.search_frame_drop_penalty,
        penalize_hidden_targets=1,
        enable_x_band=False if single_sensor else bool(args0.enable_x_band),
        single_sensor=single_sensor,
        zero_action_rewards=zero_action_rewards,
    )


def target_stats(targets: Iterable[SearchTarget]) -> dict:
    targets = list(targets)
    pi_rows = [t.sensor_pi for t in targets if getattr(t, "sensor_pi", None) is not None]
    logical_search_mass = [float(np.asarray(t.pi)[0]) for t in targets if getattr(t, "pi", None) is not None]
    search_mass = [float(np.asarray(p)[0].sum()) for p in pi_rows]
    entropy = []
    max_prob = []
    for p in pi_rows:
        flat = np.asarray(p, dtype=np.float64).reshape(-1)
        s = float(flat.sum())
        if s <= 0.0:
            continue
        flat = flat / s
        nz = flat[flat > 0]
        entropy.append(float(-(nz * np.log(np.clip(nz, 1e-12, 1.0))).sum()))
        max_prob.append(float(flat.max()))
    return {
        "targets": len(targets),
        "pi_rows": len(pi_rows),
        "logical_pi_search": float(np.mean(logical_search_mass)) if logical_search_mass else 0.0,
        "pi_search": float(np.mean(search_mass)) if search_mass else 0.0,
        "pi_search_gap": float(abs(np.mean(search_mass) - np.mean(logical_search_mass))) if search_mass and logical_search_mass else 0.0,
        "pi_entropy": float(np.mean(entropy)) if entropy else 0.0,
        "pi_max": float(np.mean(max_prob)) if max_prob else 0.0,
        "ret_mean": float(np.mean([float(t.ret) for t in targets])) if targets else 0.0,
        "ret_min": float(np.min([float(t.ret) for t in targets])) if targets else 0.0,
        "ret_max": float(np.max([float(t.ret) for t in targets])) if targets else 0.0,
        "ret_abs_p90": float(np.percentile([abs(float(t.ret)) for t in targets], 90)) if targets else 1.0,
    }


def temper_target_probs(targets: List[SearchTarget], tau: float) -> List[SearchTarget]:
    if tau <= 0.0 or abs(tau - 1.0) < 1e-9:
        return targets
    out: List[SearchTarget] = []
    power = 1.0 / float(tau)
    for t in targets:
        nt = copy.copy(t)
        if getattr(t, "sensor_pi", None) is not None:
            p = np.asarray(t.sensor_pi, dtype=np.float64)
            s = float(p.sum())
            if s > 0.0:
                p = np.power(np.clip(p / s, 0.0, 1.0), power)
                p /= max(float(p.sum()), 1e-12)
                nt.sensor_pi = p.astype(np.float32)
        p1 = np.asarray(t.pi, dtype=np.float64)
        s1 = float(p1.sum())
        if s1 > 0.0:
            p1 = np.power(np.clip(p1 / s1, 0.0, 1.0), power)
            p1 /= max(float(p1.sum()), 1e-12)
            nt.pi = p1.astype(np.float32)
        out.append(nt)
    return out


def joint_sensor_log_probs(type_logit, track_logits, sensor_logits) -> torch.Tensor:
    bsz, rows, sensors = sensor_logits.shape
    log_type_search = F.logsigmoid(type_logit).view(bsz, 1, 1)
    log_type_track = F.logsigmoid(-type_logit).view(bsz, 1, 1)
    finite = torch.isfinite(track_logits) & (track_logits > -1e8)
    track_masked = track_logits.masked_fill(~finite, -1e9)
    track_log = F.log_softmax(track_masked[:, 1:], dim=1)
    sensor_log = F.log_softmax(sensor_logits, dim=2)
    out = sensor_logits.new_full(sensor_logits.shape, -1e9)
    out[:, 0:1, :] = log_type_search + sensor_log[:, 0:1, :]
    out[:, 1:, :] = log_type_track + track_log[:, :, None] + sensor_log[:, 1:, :]
    return out


def model_sensor_probs(model, x, slot) -> torch.Tensor:
    type_logit, track_logits, _, _, _, sensor_logits, _ = model.forward_with_sensor(x, slot)
    return joint_sensor_log_probs(type_logit, track_logits, sensor_logits).exp()


def policy_value_loss(
    model,
    batch: List[SearchTarget],
    device,
    value_scale: float,
    args0,
    ref_model=None,
) -> tuple[torch.Tensor, dict]:
    x = torch.from_numpy(np.stack([t.x for t in batch]).astype(np.float32)).to(device)
    slot = torch.from_numpy(np.stack([t.slot for t in batch]).astype(np.float32)).to(device)
    sensor_pi = torch.from_numpy(np.stack([t.sensor_pi for t in batch]).astype(np.float32)).to(device)
    sensor_q_target = torch.from_numpy(
        np.stack(
            [
                t.sensor_q if getattr(t, "sensor_q", None) is not None else np.zeros_like(t.sensor_pi, dtype=np.float32)
                for t in batch
            ]
        ).astype(np.float32)
    ).to(device) / float(value_scale)
    sensor_q_mask = torch.from_numpy(
        np.stack(
            [
                t.sensor_q_mask if getattr(t, "sensor_q_mask", None) is not None else np.zeros_like(t.sensor_pi, dtype=np.float32)
                for t in batch
            ]
        ).astype(np.float32)
    ).to(device)
    q_target = torch.from_numpy(np.stack([t.q for t in batch]).astype(np.float32)).to(device) / float(value_scale)
    q_mask = torch.from_numpy(np.stack([t.q_mask for t in batch]).astype(np.float32)).to(device)
    ret = torch.tensor([float(t.ret) / value_scale for t in batch], dtype=torch.float32, device=device)
    policy_weight = torch.ones_like(ret)
    if bool(getattr(args0, "policy_positive_only", False)):
        policy_weight = (ret >= float(args0.policy_positive_margin)).float()
    action_valid = x[:, :, 4] > 0.5
    action_valid[:, 0] = True
    selected = x[:, :, 8] > 0.5
    action_valid = action_valid & ~selected
    action_valid[:, 0] = True
    sensor_pi = sensor_pi * action_valid[:, :, None].float()
    sensor_q_mask = sensor_q_mask * action_valid[:, :, None].float()
    q_mask = q_mask * action_valid.float()
    type_logit, track_logits, value, type_q, track_q, sensor_logits, sensor_q = model.forward_with_sensor(x, slot)

    mass = sensor_pi.sum(dim=(1, 2)).clamp_min(1e-6)
    sensor_pi = sensor_pi / mass[:, None, None]
    search_target = sensor_pi[:, 0, :].sum(dim=1).clamp(0.0, 1.0)
    pos_weight = torch.full_like(search_target, max(1e-6, float(getattr(args0, "type_search_pos_weight", 1.0))))
    type_loss_row = F.binary_cross_entropy_with_logits(
        type_logit,
        search_target,
        pos_weight=pos_weight,
        reduction="none",
    )
    type_loss = (type_loss_row * policy_weight).sum() / policy_weight.sum().clamp_min(1.0)

    target_mass = sensor_pi.sum(dim=2)
    track_mass = target_mass[:, 1:].sum(dim=1)
    has_track = (track_mass > 1e-6) & (policy_weight > 0.0)
    if bool(has_track.any()):
        target_cond = target_mass[has_track].clone()
        target_cond[:, 0] = 0.0
        finite = torch.isfinite(track_logits[has_track]) & (track_logits[has_track] > -1e8)
        target_cond = target_cond * finite.float()
        good = target_cond.sum(dim=1) > 1e-6
        target_cond = target_cond[good] / target_cond[good].sum(dim=1, keepdim=True).clamp_min(1e-6)
        track_loss = -(target_cond * F.log_softmax(track_logits[has_track][good], dim=1)).sum(dim=1).mean()
    else:
        track_loss = torch.zeros((), device=device)

    row_mass = sensor_pi.sum(dim=2)
    row_mask = (row_mass > 1e-6) & (policy_weight[:, None] > 0.0)
    sensor_target = sensor_pi / row_mass[:, :, None].clamp_min(1e-6)
    if bool(row_mask.any()):
        sensor_loss = -(sensor_target[row_mask] * F.log_softmax(sensor_logits[row_mask], dim=1)).sum(dim=1).mean()
    else:
        sensor_loss = torch.zeros((), device=device)
    joint_log = joint_sensor_log_probs(type_logit, track_logits, sensor_logits)
    joint_policy_loss = -(sensor_pi * joint_log).sum(dim=(1, 2))
    joint_policy_loss = (joint_policy_loss * policy_weight).sum() / policy_weight.sum().clamp_min(1.0)
    value_loss = F.smooth_l1_loss(value, ret)
    type_q_loss = torch.zeros((), device=device)
    track_q_loss = torch.zeros((), device=device)
    sensor_q_loss = torch.zeros((), device=device)
    if float(getattr(args0, "type_q_loss_weight", 0.0)) > 0.0 or float(getattr(args0, "track_q_loss_weight", 0.0)) > 0.0:
        search_valid = q_mask[:, 0] > 0.5
        track_valid = q_mask[:, 1:] > 0.5
        has_track_q = track_valid.any(dim=1)
        masked_track_q = q_target[:, 1:].masked_fill(~track_valid, -1e9)
        best_track_q = masked_track_q.max(dim=1).values
        best_track_q = torch.where(has_track_q, best_track_q, torch.zeros_like(best_track_q))
        type_q_target = torch.stack([best_track_q, q_target[:, 0]], dim=1)
        type_q_mask = torch.stack([has_track_q.float(), search_valid.float()], dim=1)
        type_q_err = F.smooth_l1_loss(type_q, type_q_target, reduction="none")
        type_q_loss = (type_q_err * type_q_mask).sum() / type_q_mask.sum().clamp_min(1.0)
        full_track_valid = q_mask > 0.5
        full_track_valid[:, 0] = False
        residual_target = q_target - best_track_q[:, None]
        if bool(full_track_valid.any()):
            track_q_loss = F.smooth_l1_loss(track_q[full_track_valid], residual_target[full_track_valid])
    if float(getattr(args0, "sensor_q_loss_weight", 0.0)) > 0.0:
        valid_sensor_q = sensor_q_mask > 0.5
        if bool(valid_sensor_q.any()):
            sensor_q_loss = F.smooth_l1_loss(sensor_q[valid_sensor_q], sensor_q_target[valid_sensor_q])
    factor_value_loss = torch.zeros((), device=device)
    if float(getattr(args0, "factor_value_loss_weight", 0.0)) > 0.0:
        flat_idx = sensor_pi.reshape(sensor_pi.shape[0], -1).argmax(dim=1)
        base_idx = flat_idx // sensor_pi.shape[2]
        sensor_idx = flat_idx % sensor_pi.shape[2]
        row_idx = torch.arange(sensor_pi.shape[0], device=device)
        is_search = base_idx == 0
        branch_value = torch.where(is_search, type_q[:, 1], type_q[:, 0])
        target_value = torch.zeros_like(ret)
        if bool((~is_search).any()):
            target_value = target_value.clone()
            target_value[~is_search] = track_q[row_idx[~is_search], base_idx[~is_search]]
        sensor_value = sensor_q[row_idx, base_idx, sensor_idx]
        selected_factor_value = branch_value + target_value + sensor_value
        factor_value_loss = F.smooth_l1_loss(selected_factor_value, ret)
    kl_loss = torch.zeros((), device=device)
    if ref_model is not None and float(args0.policy_kl_weight) > 0.0:
        with torch.no_grad():
            ref_probs = model_sensor_probs(ref_model, x, slot).clamp_min(1e-12)
            ref_probs = ref_probs / ref_probs.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        cur_log = joint_sensor_log_probs(type_logit, track_logits, sensor_logits)
        kl_loss = (ref_probs * (ref_probs.log() - cur_log)).sum(dim=(1, 2)).mean()
    loss = (
        float(args0.type_loss_weight) * type_loss
        + float(args0.track_loss_weight) * track_loss
        + float(args0.sensor_loss_weight) * sensor_loss
        + float(getattr(args0, "joint-policy-loss-weight", getattr(args0, "joint_policy_loss_weight", 0.0))) * joint_policy_loss
        + float(args0.value_loss_weight) * value_loss
        + float(getattr(args0, "type_q_loss_weight", 0.0)) * type_q_loss
        + float(getattr(args0, "track_q_loss_weight", 0.0)) * track_q_loss
        + float(getattr(args0, "sensor_q_loss_weight", 0.0)) * sensor_q_loss
        + float(getattr(args0, "factor_value_loss_weight", 0.0)) * factor_value_loss
        + float(args0.policy_kl_weight) * kl_loss
    )
    parts = {
        "type": float(type_loss.detach().cpu()),
        "track": float(track_loss.detach().cpu()),
        "sensor": float(sensor_loss.detach().cpu()),
        "joint_policy": float(joint_policy_loss.detach().cpu()),
        "value": float(value_loss.detach().cpu()),
        "type_q": float(type_q_loss.detach().cpu()),
        "track_q": float(track_q_loss.detach().cpu()),
        "sensor_q": float(sensor_q_loss.detach().cpu()),
        "factor_value": float(factor_value_loss.detach().cpu()),
        "kl": float(kl_loss.detach().cpu()),
    }
    return loss, parts


def train_pv(model, targets: List[SearchTarget], args0, device) -> dict:
    usable = [t for t in targets if getattr(t, "sensor_pi", None) is not None and float(np.sum(t.sensor_pi)) > 0.0]
    if not usable:
        raise RuntimeError("no usable AlphaZero targets with sensor_pi")
    value_scale = max(1.0, float(np.percentile([abs(float(t.ret)) for t in usable], 90)))
    train_value_only = bool(getattr(args0, "train_value_only", False))
    for p in model.parameters():
        p.requires_grad_(False if (bool(getattr(args0, "train_calibration_only", False)) or train_value_only) else bool(args0.train_encoder))
    for name, p in model.named_parameters():
        is_calibration = any(
            key in name
            for key in (
                "type_logit_scale",
                "type_logit_bias",
                "track_logit_scale",
                "type_track_coupling",
            )
        )
        if bool(getattr(args0, "train_calibration_only", False)):
            p.requires_grad_(is_calibration)
            continue
        if train_value_only:
            p.requires_grad_(
                any(
                    key in name
                    for key in (
                        "value_head",
                        "type_q_head",
                        "track_q_head",
                        "sensor_q_head",
                        "value_specialist",
                        "value_head_special",
                        "type_q_head_special",
                        "track_q_head_special",
                        "type_q_head_branch_context",
                        "moe_value",
                        "moe_type_q",
                        "moe_track_q",
                        "moe_residual_logit_scale",
                    )
                )
            )
            continue
        if any(
            key in name
            for key in (
                "type_head",
                "track_head",
                "value_head",
                "sensor_head",
                "type_q_head",
                "track_q_head",
                "sensor_q_head",
                "slot_proj",
                "type_logit_scale",
                "type_logit_bias",
                "track_logit_scale",
                "type_track_coupling",
                "moe_",
            )
        ):
            p.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args0.lr, weight_decay=1e-4)
    losses = []
    part_rows = []
    ref_model = None
    if float(args0.policy_kl_weight) > 0.0:
        ref_model = copy.deepcopy(model).to(device).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
    model.train()
    for step in range(1, args0.train_steps + 1):
        idx = np.random.randint(0, len(usable), size=args0.batch_size)
        batch = [usable[i] for i in idx]
        loss, parts = policy_value_loss(model, batch, device, value_scale, args0, ref_model)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        part_rows.append(parts)
        if step % max(1, args0.train_steps // 5) == 0:
            print(f"az step {step}/{args0.train_steps} loss={losses[-1]:.4f}", flush=True)
    model.eval()
    tail = part_rows[-20:] if part_rows else []
    metrics = {"value_scale": value_scale, "loss": float(np.mean(losses[-20:])) if losses else 0.0}
    for key in ("type", "track", "sensor", "joint_policy", "value", "type_q", "track_q", "sensor_q", "factor_value", "kl"):
        metrics[f"loss_{key}"] = float(np.mean([r[key] for r in tail])) if tail else 0.0
    return metrics


def episode_total_reward(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    if "cumulative_reward" in df.columns:
        return float(df["cumulative_reward"].iloc[-1])
    if "window_reward" in df.columns:
        return float(df["window_reward"].sum())
    if "reward" in df.columns:
        return float(df["reward"].sum())
    return 0.0


def episode_objective_score(df: pd.DataFrame, args0) -> float:
    if df.empty:
        return 0.0
    metric = str(args0.terminal_score_metric)
    if metric == "reward":
        return episode_total_reward(df)
    tracked_col = "tracked_targets" if "tracked_targets" in df.columns else ""
    drop_col = "drop_pct_active" if "drop_pct_active" in df.columns else ""
    delay_col = "mean_delay_active" if "mean_delay_active" in df.columns else ""
    if not tracked_col and not drop_col and not delay_col:
        return episode_total_reward(df)
    if metric == "final_health":
        tracked = float(df[tracked_col].iloc[-1]) if tracked_col else 0.0
        drops = float(df[drop_col].iloc[-1]) if drop_col else 0.0
        delay = float(df[delay_col].iloc[-1]) if delay_col else 0.0
    else:
        tracked = float(df[tracked_col].mean()) if tracked_col else 0.0
        drops = float(df[drop_col].mean()) if drop_col else 0.0
        delay = float(df[delay_col].mean()) if delay_col else 0.0
    return (
        float(args0.terminal_tracked_weight) * tracked
        - float(args0.terminal_drop_weight) * drops
        - float(args0.terminal_delay_weight) * delay
    )


def episode_health_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"mean_tracked": 0.0, "final_tracked": 0.0, "mean_delay": 0.0, "mean_drop_pct": 0.0}
    return {
        "mean_tracked": float(df["tracked_targets"].mean()) if "tracked_targets" in df.columns else 0.0,
        "final_tracked": float(df["tracked_targets"].iloc[-1]) if "tracked_targets" in df.columns else 0.0,
        "mean_delay": float(df["mean_delay_active"].mean()) if "mean_delay_active" in df.columns else 0.0,
        "mean_drop_pct": float(df["drop_pct_active"].mean()) if "drop_pct_active" in df.columns else 0.0,
    }


def load_state_into(model, state_path: str, device) -> None:
    state = torch.load(state_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state, strict=False)
    if missing:
        print(f"load_state_into missing keys initialized from model defaults: {', '.join(missing)}", flush=True)
    if unexpected:
        print(f"load_state_into ignored unexpected keys: {', '.join(unexpected)}", flush=True)


def baseline_episode_scores(init: int, rate: float, seed: int, exact_args, args0, accepted_model=None) -> dict:
    env_cfg = env_cfg_for(float(rate), exact_args)
    out = {}
    for name, planner in [("EDF", EDFPlanner(MAXT)), ("EST", ESTPlanner(MAXT))]:
        w, _ = run_fixed(planner, name, int(init), MAXT, int(seed), int(args0.windows), 200, engine_env_cfg(env_cfg))
        out[name] = episode_objective_score(w, args0)
    if accepted_model is not None:
        accepted_args = copy.copy(exact_args)
        if int(getattr(args0, "accepted_baseline_rollouts", 0)) > 0:
            accepted_args.rollouts = int(args0.accepted_baseline_rollouts)
        df, _ = run_snapshot_exact_episode(accepted_model.eval(), accepted_args, int(init), float(rate), int(seed), train=False)
        out["accepted"] = episode_objective_score(df, args0)
    return out


def terminal_value_target(agent_score: float, baseline_score: float, args0) -> float:
    diff = float(agent_score) - float(baseline_score)
    mode = str(args0.terminal_baseline_target)
    if mode == "raw":
        return float(agent_score)
    if mode == "raw_diff":
        return float(diff)
    if mode == "sign":
        margin = float(args0.terminal_baseline_margin)
        if diff > margin:
            return 1.0
        if diff < -margin:
            return -1.0
        return 0.0
    scale = max(float(args0.terminal_baseline_scale), 1e-6)
    return float(np.tanh(diff / scale))


def apply_terminal_episode_target(targets: List[SearchTarget], value: float) -> None:
    for target in targets:
        target.reward = 0.0
        target.ret = float(value)


def target_from_action(obs: dict, action: int, elapsed_ms: float, search_count: int, track_count: int, last_action: int, budget_ms: float) -> SearchTarget:
    base_action, sensor = xs_decode_action(int(action), MAXT)
    if base_action < 0:
        base_action = 0
    base_action = int(np.clip(base_action, 0, MAXT))
    sensor_id = 0 if sensor is None else int(np.clip(sensor, 0, 1))
    x = tokenize(get_adapter(), obs, selected=set(), search_count=int(search_count)).astype(np.float32)
    slot = slot_features(
        obs,
        float(elapsed_ms),
        int(search_count),
        int(track_count),
        int(last_action),
        float(budget_ms),
    ).astype(np.float32)
    pi = np.zeros((MAXT + 1,), dtype=np.float32)
    pi[base_action] = 1.0
    sensor_pi = np.zeros((MAXT + 1, 2), dtype=np.float32)
    sensor_pi[base_action, sensor_id] = 1.0
    q = np.zeros((MAXT + 1,), dtype=np.float32)
    q_mask = np.zeros((MAXT + 1,), dtype=np.float32)
    sensor_q = np.zeros((MAXT + 1, 2), dtype=np.float32)
    sensor_q_mask = np.zeros((MAXT + 1, 2), dtype=np.float32)
    return SearchTarget(
        x=x,
        slot=slot,
        pi=pi,
        q=q,
        q_mask=q_mask,
        search_count=int(search_count),
        track_count=int(track_count),
        sensor_pi=sensor_pi,
        sensor_q=sensor_q,
        sensor_q_mask=sensor_q_mask,
    )


def _candidate_actions(obs: dict, top_k: int, mode: str = "urgent") -> list[int]:
    active = np.asarray(obs.get("active_mask", []), dtype=bool)
    deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
    desired = np.asarray(obs.get("t_desired", []), dtype=np.float32)
    ranges = np.asarray(obs.get("target_range", np.zeros_like(deadline)), dtype=np.float32)
    explicit = "s_band_busy_ms" in obs or "x_band_busy_ms" in obs or bool(int(obs.get("enable_x_band", 0)))
    s_free = float(obs.get("s_band_busy_ms", 0.0)) <= 0.0
    x_free = bool(int(obs.get("enable_x_band", 0))) and float(obs.get("x_band_busy_ms", 0.0)) <= 0.0
    tracks = []
    for idx, ok in enumerate(active[:MAXT]):
        if not bool(ok):
            continue
        if idx < len(deadline) and float(deadline[idx]) < 0.0:
            continue
        action = int(idx + 1)
        if str(mode) == "delay":
            score = float(desired[idx]) if idx < len(desired) else 0.0
        else:
            score = float(deadline[idx]) if idx < len(deadline) else 1e9
        tracks.append((score, action))
    tracks.sort(key=lambda x: (x[0], x[1]))
    if not explicit:
        return [0] + [a for _, a in tracks[: max(0, int(top_k))]]
    out: list[int] = []
    if s_free:
        out.append(xs_s_search_action(MAXT))
    if x_free:
        out.append(xs_x_search_action(MAXT))
    for _, base_action in tracks[: max(0, int(top_k))]:
        idx = int(base_action) - 1
        rng = float(ranges[idx]) if 0 <= idx < len(ranges) else 0.0
        if s_free and 10_000_000.0 < rng < 184_000_000.0:
            out.append(xs_s_track_action(int(base_action), MAXT))
        if x_free and 5_000_000.0 < rng < 100_000_000.0:
            out.append(xs_x_track_action(int(base_action), MAXT))
    return out


def _discounted_reference_rollout(
    eng,
    debt_ms: float,
    env_cfg: dict,
    use_arrival_feature: bool,
    use_grid_feature: bool,
    planner,
    windows_left: int,
    first_window_budget_ms: float,
    gamma: float,
) -> float:
    total = 0.0
    discount = 1.0
    debt = float(debt_ms)
    for local_window in range(max(0, int(windows_left))):
        remaining_ms = float(first_window_budget_ms) if local_window == 0 else 200.0
        while remaining_ms > 0.0 and not bool(eng.term_buf[0]):
            obs = attach_env_obs(get_obs(eng, debt), env_cfg, bool(use_arrival_feature), bool(use_grid_feature))
            plan = planner.plan(obs, budget_ms=int(max(1.0, remaining_ms)))
            if not plan:
                break
            reward, dt, executed = execute_first_valid_action(eng, plan, remaining_ms)
            if executed is None or dt <= 0.0:
                break
            total += discount * float(reward)
            discount *= float(gamma)
            base_action, _ = xs_decode_action(int(executed), MAXT)
            debt = 0.0 if int(base_action) == 0 else debt + float(dt)
            remaining_ms -= float(dt)
    return float(total)


def target_from_reference_counterfactuals(
    eng,
    root_snapshot,
    obs: dict,
    env_cfg: dict,
    label_planner,
    elapsed_ms: float,
    remaining_ms: float,
    debt_ms: float,
    search_count: int,
    track_count: int,
    last_action: int,
    window_idx: int,
    args0,
) -> SearchTarget:
    x = tokenize(get_adapter(), obs, selected=set(), search_count=int(search_count)).astype(np.float32)
    slot = slot_features(obs, float(elapsed_ms), int(search_count), int(track_count), int(last_action), float(remaining_ms)).astype(np.float32)
    q = np.zeros((MAXT + 1,), dtype=np.float32)
    q_mask = np.zeros((MAXT + 1,), dtype=np.float32)
    sensor_q = np.zeros((MAXT + 1, 2), dtype=np.float32)
    sensor_q_mask = np.zeros((MAXT + 1, 2), dtype=np.float32)
    gamma = float(getattr(args0, "gamma", 0.99))
    candidates = _candidate_actions(obs, int(getattr(args0, "cf_top_k", 8)), str(getattr(args0, "cf_candidate_mode", "urgent")))
    for action in candidates:
        binding.vec_restore(eng.env, root_snapshot)
        reward, dt, executed = execute_first_valid_action(eng, [int(action)], float(remaining_ms))
        if executed is None or dt <= 0.0:
            continue
        base_action, sensor_id = xs_decode_action(int(executed), MAXT)
        if int(base_action) < 0:
            continue
        next_debt = 0.0 if int(base_action) == 0 else float(debt_ms) + float(dt)
        cf_windows = int(getattr(args0, "cf_rollout_windows", 0))
        future = 0.0
        if cf_windows >= 0:
            windows_left = max(1, int(args0.windows) - int(window_idx))
            if cf_windows > 0:
                windows_left = min(windows_left, cf_windows)
            future = _discounted_reference_rollout(
                eng,
                next_debt,
                env_cfg,
                bool(getattr(args0, "use_arrival_feature", False)),
                bool(getattr(args0, "use_grid_feature", False)),
                label_planner,
                windows_left,
                max(0.0, float(remaining_ms) - float(dt)),
                gamma,
            )
        value = float(reward) + gamma * float(future)
        base_idx = int(np.clip(int(base_action), 0, MAXT))
        sensor_idx = 0 if sensor_id is None else int(np.clip(int(sensor_id), 0, 1))
        q[base_idx] = max(float(q[base_idx]), value) if q_mask[base_idx] > 0.5 else value
        q_mask[base_idx] = 1.0
        sensor_q[base_idx, sensor_idx] = value
        sensor_q_mask[base_idx, sensor_idx] = 1.0
    binding.vec_restore(eng.env, root_snapshot)
    pi = np.zeros((MAXT + 1,), dtype=np.float32)
    sensor_pi = np.zeros((MAXT + 1, 2), dtype=np.float32)
    valid = np.where(q_mask > 0.5)[0]
    if len(valid) > 0:
        tau = max(1e-3, float(getattr(args0, "cf_policy_tau", 10.0)))
        logits = q[valid].astype(np.float64) / tau
        logits -= float(np.max(logits))
        probs = np.exp(np.clip(logits, -60.0, 60.0))
        probs /= max(float(np.sum(probs)), 1e-12)
        pi[valid] = probs.astype(np.float32)
        for base_idx in valid:
            smask = sensor_q_mask[int(base_idx)] > 0.5
            if not np.any(smask):
                continue
            slogits = sensor_q[int(base_idx), smask].astype(np.float64) / tau
            slogits -= float(np.max(slogits))
            sp = np.exp(np.clip(slogits, -60.0, 60.0))
            sp /= max(float(np.sum(sp)), 1e-12)
            sensor_pi[int(base_idx), np.where(smask)[0]] = sp.astype(np.float32) * float(pi[int(base_idx)])
    return SearchTarget(
        x=x,
        slot=slot,
        pi=pi,
        q=q,
        q_mask=q_mask,
        search_count=int(search_count),
        track_count=int(track_count),
        sensor_pi=sensor_pi,
        sensor_q=sensor_q,
        sensor_q_mask=sensor_q_mask,
    )


def rollout_planner(source: str, model, exact_args, args0, env_cfg):
    source = str(source).removesuffix("_window")
    if source == "est":
        return ESTPlanner(MAXT)
    if source == "edf":
        return EDFPlanner(MAXT)
    if source == "reference_mcts":
        planner = make_reference_planner(
            MAXT,
            int(getattr(args0, "reference_mcts_rollouts", 8)),
            float(getattr(args0, "reference_mcts_c_puct", 1.25)),
            env_cfg,
            "repaired_stress",
            simulation_window_ms=float(getattr(args0, "reference_mcts_horizon_ms", 200.0)),
        )
        planner.rollout_policy = str(getattr(args0, "reference_mcts_rollout_policy", "greedy"))
        planner.action_selection = str(getattr(args0, "reference_mcts_select", "q"))
        planner.rollout_search_period_ms = float(getattr(args0, "reference_mcts_search_period_ms", 160.0))
        return planner
    if source == "model_direct":
        return MutualRadarDirectPlanner(
            model,
            direct_mode=str(args0.direct_mode),
            alpha=float(args0.direct_alpha),
            beta=float(args0.direct_beta),
            threshold=float(args0.direct_threshold),
            allow_retrack=False,
            cache_encoder=True,
            sensor_action_mode="explicit_head",
            disable_x_search=bool(getattr(args0, "single_sensor", False)),
        )
    raise ValueError(f"unknown rollout target source: {source}")


def episode_target_source(source: str, ep: int) -> str:
    if source == "cycle_est_edf_model":
        return ["est", "edf", "model_direct"][int(ep) % 3]
    if source == "cycle_edf_est":
        return ["edf", "est"][int(ep) % 2]
    return source


def best_heuristic_source(exact_args, args0, init: int, rate: float, seed: int) -> str:
    env_cfg = env_cfg_for(float(rate), exact_args)
    scores = {}
    for name, planner in [("edf", EDFPlanner(MAXT)), ("est", ESTPlanner(MAXT))]:
        df, _ = run_fixed(planner, name.upper(), int(init), MAXT, int(seed), int(args0.windows), 200, engine_env_cfg(env_cfg))
        scores[name] = episode_objective_score(df, args0)
    return max(scores, key=scores.get)


def collect_rollout_episode_targets(model, exact_args, args0, init: int, rate: float, seed: int, source: str) -> tuple[pd.DataFrame, List[SearchTarget]]:
    env_cfg = env_cfg_for(float(rate), exact_args)
    window_plan_once = str(source).endswith("_window")
    if source == "model_cf":
        planner = rollout_planner("model_direct", model.eval(), exact_args, args0, env_cfg)
        label_planner = planner
    elif source in {"reference_on_model_direct", "reference_cf_on_model_direct"}:
        planner = rollout_planner("model_direct", model.eval(), exact_args, args0, env_cfg)
        label_planner = rollout_planner("reference_mcts", model.eval(), exact_args, args0, env_cfg)
    elif source == "reference_cf":
        planner = rollout_planner("reference_mcts", model.eval(), exact_args, args0, env_cfg)
        label_planner = planner
    else:
        planner = rollout_planner(source, model.eval(), exact_args, args0, env_cfg)
        label_planner = planner
    use_cf_targets = source in {"model_cf", "reference_cf", "reference_cf_on_model_direct"}
    eng = build_env(planner, int(init), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
    eng.reset(seed=int(seed))
    search_debt_ms = 0.0
    cumulative_reward = 0.0
    targets: List[SearchTarget] = []
    rows = []
    try:
        action_seq = 0
        for window_idx in range(int(args0.windows)):
            if eng.term_buf[0]:
                break
            remaining_ms = 200.0
            window_reward = 0.0
            window_actions = []
            elapsed_ms = 0.0
            search_count = 0
            track_count = 0
            last_action = -1
            fixed_window_plan = None
            fixed_window_pos = 0
            if window_plan_once:
                obs0 = attach_env_obs(get_obs(eng, search_debt_ms), env_cfg, bool(getattr(args0, "use_arrival_feature", False)), bool(getattr(args0, "use_grid_feature", False)))
                fixed_window_plan = planner.plan(obs0, budget_ms=200)
            while remaining_ms > 0.0 and not eng.term_buf[0]:
                obs = attach_env_obs(get_obs(eng, search_debt_ms), env_cfg, bool(getattr(args0, "use_arrival_feature", False)), bool(getattr(args0, "use_grid_feature", False)))
                collect_this = (
                    int(window_idx) >= int(getattr(args0, "target_start_window", 0))
                    and (action_seq % max(1, int(getattr(args0, "target_stride", 1)))) == 0
                )
                root_snapshot = binding.vec_snapshot(eng.env) if use_cf_targets and collect_this else None
                if fixed_window_plan is not None:
                    plan = list(fixed_window_plan[fixed_window_pos:])
                else:
                    plan = planner.plan(obs, budget_ms=int(max(1.0, remaining_ms)))
                if not plan:
                    break
                if fixed_window_plan is not None and label_planner is planner:
                    label_plan = plan
                else:
                    label_plan = label_planner.plan(obs, budget_ms=int(max(1.0, remaining_ms)))
                if not label_plan:
                    break
                target = None
                if use_cf_targets and collect_this:
                    target = target_from_reference_counterfactuals(
                        eng,
                        root_snapshot,
                        obs,
                        env_cfg,
                        label_planner,
                        elapsed_ms,
                        remaining_ms,
                        search_debt_ms,
                        search_count,
                        track_count,
                        last_action,
                        window_idx,
                        args0,
                    )
                    binding.vec_restore(eng.env, root_snapshot)
                elif collect_this:
                    commanded = int(label_plan[0])
                    target = target_from_action(obs, commanded, elapsed_ms, search_count, track_count, last_action, remaining_ms)
                reward, dt, executed = execute_first_valid_action(eng, plan, remaining_ms)
                if executed is None or dt <= 0.0:
                    break
                if target is not None:
                    target.initial = int(init)
                    target.rate = float(rate)
                    target.seed = int(seed)
                    target.window = int(window_idx + 1)
                    target.action_index = int(len(targets))
                    target.reward = float(reward)
                    targets.append(target)
                if fixed_window_plan is not None:
                    try:
                        rel = list(plan).index(int(executed))
                        fixed_window_pos += rel + 1
                    except ValueError:
                        fixed_window_pos += 1
                action_seq += 1
                if len(targets) >= int(args0.max_targets_per_episode):
                    break
                base_action, _ = xs_decode_action(int(executed), MAXT)
                if int(base_action) == 0:
                    search_debt_ms = 0.0
                    search_count += 1
                else:
                    search_debt_ms += float(dt)
                    track_count += 1
                last_action = int(base_action)
                elapsed_ms += float(dt)
                remaining_ms -= float(dt)
                window_reward += float(reward)
                window_actions.append(int(executed))
            if len(targets) >= int(args0.max_targets_per_episode):
                break
            cumulative_reward += float(window_reward)
            obs_now = attach_env_obs(get_obs(eng, search_debt_ms), env_cfg, bool(getattr(args0, "use_arrival_feature", False)), bool(getattr(args0, "use_grid_feature", False)))
            active = np.asarray(obs_now["active_mask"]).astype(bool)
            tracked = np.asarray(obs_now.get("tracked", active)).astype(bool)
            deadlines = np.asarray(obs_now["t_deadline"], dtype=np.float32)
            desired = np.asarray(obs_now["t_desired"], dtype=np.float32)
            active_n = int(np.sum(active))
            tracked_n = int(np.sum(active & tracked))
            dropped_n = int(np.sum(active & (deadlines < 0.0)))
            active_delays = np.maximum(0.0, -desired[active]) if active_n > 0 else np.zeros(0, dtype=np.float32)
            rows.append(
                {
                    "window": int(window_idx),
                    "window_reward": float(window_reward),
                    "cumulative_reward": float(cumulative_reward),
                    "search_fraction": float(np.mean([xs_decode_action(a, MAXT)[0] == 0 for a in window_actions])) if window_actions else 0.0,
                    "executed_actions": int(len(window_actions)),
                    "active_targets": float(active_n),
                    "tracked_targets": float(tracked_n),
                    "drop_pct_active": float(100.0 * dropped_n / active_n) if active_n > 0 else 0.0,
                    "mean_delay_active": float(np.mean(active_delays)) if active_n > 0 else 0.0,
                }
            )
    finally:
        eng.close()
    G = 0.0
    gamma = float(getattr(args0, "gamma", 0.99))
    for target in reversed(targets):
        G = float(target.reward) + gamma * G
        target.ret = float(G)
    return pd.DataFrame(rows), targets


def collect_quota_episode_targets(model, exact_args, args0, init: int, rate: float, seed: int, label_source: str) -> tuple[pd.DataFrame, List[SearchTarget]]:
    """Collect factorized track-ordering targets under a fixed search quota.

    The macro quota is executed by construction.  Targets are only added for
    post-quota track choices, which isolates the currently observed failure:
    target ordering under a good surveillance budget.
    """

    env_cfg = env_cfg_for(float(rate), exact_args)
    label_planner = rollout_planner(label_source, model.eval(), exact_args, args0, env_cfg)
    eng = build_env(label_planner, int(init), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
    eng.reset(seed=int(seed))
    search_debt_ms = 0.0
    cumulative_reward = 0.0
    quota = max(0, int(getattr(args0, "macro_search_quota", 0)))
    targets: List[SearchTarget] = []
    rows = []
    try:
        for window_idx in range(int(args0.windows)):
            if eng.term_buf[0]:
                break
            remaining_ms = 200.0
            elapsed_ms = 0.0
            window_reward = 0.0
            window_actions = []
            search_count = 0
            track_count = 0
            last_action = -1

            for _ in range(quota):
                if remaining_ms <= 0.0 or eng.term_buf[0]:
                    break
                reward, dt, executed = execute_first_valid_action(eng, [xs_s_search_action(MAXT)], remaining_ms)
                if executed is None or dt <= 0.0:
                    break
                base_action, _ = xs_decode_action(int(executed), MAXT)
                window_reward += float(reward)
                window_actions.append(int(executed))
                elapsed_ms += float(dt)
                remaining_ms -= float(dt)
                if int(base_action) == 0:
                    search_debt_ms = 0.0
                    search_count += 1
                else:
                    search_debt_ms += float(dt)
                    track_count += 1
                last_action = int(base_action)

            while remaining_ms > 0.0 and not eng.term_buf[0]:
                obs = attach_env_obs(get_obs(eng, search_debt_ms), env_cfg, bool(getattr(args0, "use_arrival_feature", False)), bool(getattr(args0, "use_grid_feature", False)))
                label_plan = list(label_planner.plan(obs, budget_ms=int(max(1.0, remaining_ms))))
                track_plan = [int(a) for a in label_plan if xs_decode_action(int(a), MAXT)[0] != 0]
                if not track_plan:
                    break
                commanded = int(track_plan[0])
                target = target_from_action(obs, commanded, elapsed_ms, search_count, track_count, last_action, remaining_ms)
                target.initial = int(init)
                target.rate = float(rate)
                target.seed = int(seed)
                target.window = int(window_idx + 1)
                target.action_index = int(len(targets))
                reward, dt, executed = execute_first_valid_action(eng, track_plan, remaining_ms)
                if executed is None or dt <= 0.0:
                    break
                target.reward = float(reward)
                targets.append(target)
                if len(targets) >= int(args0.max_targets_per_episode):
                    break
                base_action, _ = xs_decode_action(int(executed), MAXT)
                if int(base_action) == 0:
                    search_debt_ms = 0.0
                    search_count += 1
                else:
                    search_debt_ms += float(dt)
                    track_count += 1
                last_action = int(base_action)
                elapsed_ms += float(dt)
                remaining_ms -= float(dt)
                window_reward += float(reward)
                window_actions.append(int(executed))

            cumulative_reward += float(window_reward)
            obs_now = attach_env_obs(get_obs(eng, search_debt_ms), env_cfg, bool(getattr(args0, "use_arrival_feature", False)), bool(getattr(args0, "use_grid_feature", False)))
            active = np.asarray(obs_now["active_mask"]).astype(bool)
            tracked = np.asarray(obs_now.get("tracked", active)).astype(bool)
            deadlines = np.asarray(obs_now["t_deadline"], dtype=np.float32)
            desired = np.asarray(obs_now["t_desired"], dtype=np.float32)
            active_n = int(np.sum(active))
            tracked_n = int(np.sum(active & tracked))
            dropped_n = int(np.sum(active & (deadlines < 0.0)))
            active_delays = np.maximum(0.0, -desired[active]) if active_n > 0 else np.zeros(0, dtype=np.float32)
            rows.append(
                {
                    "window": int(window_idx),
                    "window_reward": float(window_reward),
                    "cumulative_reward": float(cumulative_reward),
                    "search_fraction": float(np.mean([xs_decode_action(a, MAXT)[0] == 0 for a in window_actions])) if window_actions else 0.0,
                    "executed_actions": int(len(window_actions)),
                    "active_targets": float(active_n),
                    "tracked_targets": float(tracked_n),
                    "drop_pct_active": float(100.0 * dropped_n / active_n) if active_n > 0 else 0.0,
                    "mean_delay_active": float(np.mean(active_delays)) if active_n > 0 else 0.0,
                }
            )
            if len(targets) >= int(args0.max_targets_per_episode):
                break
    finally:
        eng.close()
    G = 0.0
    gamma = float(getattr(args0, "gamma", 0.99))
    for target in reversed(targets):
        G = float(target.reward) + gamma * G
        target.ret = float(G)
    return pd.DataFrame(rows), targets


def collect_targets(model, exact_args, args0, accepted_model=None) -> tuple[list[SearchTarget], pd.DataFrame]:
    all_targets: List[SearchTarget] = []
    rows = []
    initials = parse_ints(args0.train_initials)
    rates = parse_floats(args0.train_rates)
    for ep in range(args0.episodes):
        init = initials[ep % len(initials)]
        rate = rates[(ep // len(initials)) % len(rates)]
        seed = int(args0.seed + ep)
        source = episode_target_source(args0.target_source, ep)
        if source == "best_heuristic":
            source = best_heuristic_source(exact_args, args0, init, rate, seed)
        if source == "mcts":
            df, targets = run_snapshot_exact_episode(model.eval(), exact_args, init, rate, seed, train=True)
        elif source.startswith("quota_"):
            df, targets = collect_quota_episode_targets(model, exact_args, args0, init, rate, seed, source.removeprefix("quota_"))
        else:
            df, targets = collect_rollout_episode_targets(model, exact_args, args0, init, rate, seed, source)
        agent_score = episode_objective_score(df, args0)
        reward_score = episode_total_reward(df)
        health = episode_health_stats(df)
        baseline_scores = {}
        baseline_score = np.nan
        value_target = np.nan
        if args0.terminal_baseline_target == "raw":
            value_target = terminal_value_target(agent_score, 0.0, args0)
            apply_terminal_episode_target(targets, value_target)
        elif args0.terminal_baseline_target != "none":
            baseline_scores = baseline_episode_scores(init, rate, seed, exact_args, args0, accepted_model)
            if args0.terminal_baseline_mode == "edf":
                baseline_score = float(baseline_scores["EDF"])
            elif args0.terminal_baseline_mode == "est":
                baseline_score = float(baseline_scores["EST"])
            elif args0.terminal_baseline_mode == "accepted":
                baseline_score = float(baseline_scores["accepted"])
            elif args0.terminal_baseline_mode == "max_all":
                baseline_score = float(max(baseline_scores.values()))
            else:
                heuristic_values = [baseline_scores[k] for k in ("EDF", "EST") if k in baseline_scores]
                baseline_score = float(max(heuristic_values))
            value_target = terminal_value_target(agent_score, baseline_score, args0)
            apply_terminal_episode_target(targets, value_target)
        elif bool(getattr(args0, "reject_selfplay_below_baseline", False)):
            baseline_scores = baseline_episode_scores(init, rate, seed, exact_args, args0, accepted_model)
            if str(getattr(args0, "reject_baseline_mode", "max_heuristic")) == "edf":
                baseline_score = float(baseline_scores["EDF"])
            elif str(getattr(args0, "reject_baseline_mode", "max_heuristic")) == "est":
                baseline_score = float(baseline_scores["EST"])
            elif str(getattr(args0, "reject_baseline_mode", "max_heuristic")) == "accepted":
                baseline_score = float(baseline_scores.get("accepted", -1e18))
            elif str(getattr(args0, "reject_baseline_mode", "max_heuristic")) == "max_all":
                baseline_score = float(max(baseline_scores.values()))
            else:
                heuristic_values = [baseline_scores[k] for k in ("EDF", "EST") if k in baseline_scores]
                baseline_score = float(max(heuristic_values))
            margin = float(agent_score - baseline_score)
            if margin < float(getattr(args0, "reject_margin", 0.0)):
                print(
                    f"rejecting selfplay ep {ep+1}: score={agent_score:.3f} baseline={baseline_score:.3f} margin={margin:.3f}",
                    flush=True,
                )
                targets = []
        all_targets.extend(targets)
        if not df.empty:
            rows.append(
                {
                    "episode": ep,
                    "target_source": source,
                    "initial": init,
                    "rate": rate,
                    "seed": seed,
                    "reward": float(df["window_reward"].mean()),
                    "windows_completed": int(len(df)),
                    "agent_score": float(agent_score),
                    "agent_reward_score": float(reward_score),
                    **health,
                    "edf_score": float(baseline_scores.get("EDF", np.nan)),
                    "est_score": float(baseline_scores.get("EST", np.nan)),
                    "accepted_score": float(baseline_scores.get("accepted", np.nan)),
                    "baseline_score": float(baseline_score),
                    "terminal_z": float(value_target),
                    "targets": len(targets),
                }
            )
        print(
            f"selfplay ep {ep+1}/{args0.episodes}: source={source} init={init} rate={rate} targets={len(targets)} "
            f"score={agent_score:.3f} z={value_target if np.isfinite(value_target) else float('nan'):.3f}",
            flush=True,
        )
    return all_targets, pd.DataFrame(rows)


def eval_mcts(model, exact_args, args0, tag: str) -> pd.DataFrame:
    rows = []
    for seed in parse_ints(args0.eval_seeds):
        for init in parse_ints(args0.eval_initials):
            for rate in parse_floats(args0.eval_rates):
                df, _ = run_snapshot_exact_episode(model.eval(), exact_args, init, rate, seed, train=False)
                rows.append(
                    {
                        "tag": tag,
                        "initial": init,
                        "rate": rate,
                        "seed": seed,
                        "reward": float(df["window_reward"].mean()) if not df.empty else 0.0,
                        "search": float(df["search_fraction"].iloc[-1]) if not df.empty else 0.0,
                    }
                )
    return pd.DataFrame(rows)


class ArrivalAwarePlanner:
    def __init__(self, planner, env_cfg, enabled: bool):
        self.planner = planner
        self.env_cfg = env_cfg
        self.enabled = bool(enabled)

    def warmup(self, obs, budget_ms=200):
        obs2 = attach_env_obs(obs, self.env_cfg, self.enabled, bool(getattr(self, "use_grid_feature", False)))
        if hasattr(self.planner, "warmup"):
            return self.planner.warmup(obs2, budget_ms=budget_ms)
        return self.planner.plan(obs2, budget_ms=budget_ms)

    def plan(self, obs, budget_ms=200):
        return self.planner.plan(attach_env_obs(obs, self.env_cfg, self.enabled, bool(getattr(self, "use_grid_feature", False))), budget_ms=budget_ms)


def eval_direct(model, exact_args, args0, tag: str) -> pd.DataFrame:
    rows = []
    for seed in parse_ints(args0.eval_seeds):
        for init in parse_ints(args0.eval_initials):
            for rate in parse_floats(args0.eval_rates):
                env_cfg = env_cfg_for(float(rate), exact_args)
                planners = [
                    (
                        "model_direct",
                        ArrivalAwarePlanner(
                            MutualRadarDirectPlanner(
                                model.eval(),
                                direct_mode=str(args0.direct_mode),
                                alpha=float(args0.direct_alpha),
                                beta=float(args0.direct_beta),
                                threshold=float(args0.direct_threshold),
                                allow_retrack=False,
                                cache_encoder=True,
                                sensor_action_mode="explicit_head",
                                disable_x_search=bool(getattr(args0, "single_sensor", False)),
                            ),
                            env_cfg,
                            bool(getattr(args0, "use_arrival_feature", False)),
                        ),
                    ),
                    ("EDF", EDFPlanner(MAXT)),
                    ("EST", ESTPlanner(MAXT)),
                ]
                for name, planner in planners:
                    df, _ = run_fixed(planner, name, init, MAXT, seed, int(args0.windows), 200, engine_env_cfg(env_cfg))
                    rows.append(
                        {
                            "tag": tag,
                            "planner": name,
                            "initial": init,
                            "rate": rate,
                            "seed": seed,
                            "reward": float(df["window_reward"].mean()) if not df.empty else 0.0,
                            "total_reward": episode_total_reward(df),
                            "search": float(df["search_fraction"].mean()) if not df.empty else 0.0,
                            "windows_completed": int(len(df)),
                            "latency": float(df["planning_ms_per_decision"].mean()) if "planning_ms_per_decision" in df.columns and not df.empty else 0.0,
                        }
                    )
    return pd.DataFrame(rows)


def eval_model(model, exact_args, args0, tag: str) -> pd.DataFrame:
    if args0.eval_mode == "none":
        return pd.DataFrame()
    if args0.eval_mode == "direct":
        return eval_direct(model, exact_args, args0, tag)
    return eval_mcts(model, exact_args, args0, tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--max-targets-per-episode", type=int, default=64)
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--horizon-windows", type=int, default=4)
    ap.add_argument("--expand-top-k", type=int, default=64)
    ap.add_argument("--c-puct", type=float, default=1.25)
    ap.add_argument("--rollout-policy", choices=["model", "branch", "branch_margin", "q", "pq", "random", "value", "edge", "edf", "est", "mixed"], default="model")
    ap.add_argument("--branch-rollout-threshold", type=float, default=0.65)
    ap.add_argument("--prior-uniform-mix", type=float, default=0.03)
    ap.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    ap.add_argument("--root-dirichlet-frac", type=float, default=0.0)
    ap.add_argument("--leaf-value-mix", type=float, default=0.5)
    ap.add_argument("--head-arch", choices=["baseline", "branch_context", "specialized", "moe"], default="branch_context")
    ap.add_argument("--head-mode", choices=["p", "pv", "pq", "pvq"], default="pv")
    ap.add_argument("--prior-mode", choices=["factorized", "flat", "branch_corrected", "physical_flat", "true_physical_flat"], default="factorized")
    ap.add_argument("--search-alg", choices=["puct", "gumbel", "hierarchical"], default="puct")
    ap.add_argument("--plan-mode", choices=["atomic", "window", "first_window"], default="atomic")
    ap.add_argument("--window-extract", choices=["tree", "tree_fill", "best", "greedy_expand", "batched_value", "model_q", "edge_q"], default="tree_fill")
    ap.add_argument("--add-prefix-targets", action="store_true")
    ap.add_argument("--target-selected-action", action="store_true")
    ap.add_argument("--gumbel-scale", type=float, default=0.0)
    ap.add_argument("--select-mode", choices=["visits", "q", "prior", "branch_visits", "branch_q", "load_gated_prior"], default="visits")
    ap.add_argument("--load-gated-prior-threshold", type=int, default=80)
    ap.add_argument("--disable-visit-unvisited-first", action="store_true")
    ap.add_argument("--q-utility-weight", type=float, default=0.0)
    ap.add_argument("--q-utility-normalize", action="store_true")
    ap.add_argument("--puct-q-transform", choices=["raw", "scale", "minmax", "completed", "completed_mix", "mctx"], default="raw")
    ap.add_argument("--mctx-value-scale", type=float, default=0.1)
    ap.add_argument("--mctx-maxvisit-init", type=float, default=50.0)
    ap.add_argument("--seed-rollout-policies", default="")
    ap.add_argument("--skip-default-rollout-seed", action="store_true")
    ap.add_argument("--max-num-considered-actions", type=int, default=0)
    ap.add_argument("--prior-q-beta", type=float, default=0.0)
    ap.add_argument("--prior-search-bias", type=float, default=0.0)
    ap.add_argument("--q-scale", type=float, default=100.0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-encoder", action="store_true")
    ap.add_argument("--train-calibration-only", action="store_true")
    ap.add_argument("--train-value-only", action="store_true")
    ap.add_argument("--target-temperature", type=float, default=1.0)
    ap.add_argument("--self-play-sample-tau", type=float, default=0.0)
    ap.add_argument("--policy-target", choices=["visits", "q_softmax", "branch_q_softmax", "branch_future_softmax", "mixed", "mctx"], default="visits")
    ap.add_argument("--policy-tau", type=float, default=1.0)
    ap.add_argument(
        "--target-source",
        choices=[
            "mcts",
            "model_direct",
            "est",
            "edf",
            "est_window",
            "edf_window",
            "quota_edf",
            "quota_est",
            "quota_model_direct",
            "quota_reference_mcts",
            "reference_mcts",
            "reference_on_model_direct",
            "reference_cf",
            "reference_cf_on_model_direct",
            "model_cf",
            "best_heuristic",
            "cycle_edf_est",
            "cycle_est_edf_model",
        ],
        default="mcts",
    )
    ap.add_argument("--terminal-baseline-target", choices=["none", "continuous", "sign", "raw", "raw_diff"], default="none")
    ap.add_argument("--terminal-baseline-mode", choices=["max_heuristic", "edf", "est", "accepted", "max_all"], default="max_heuristic")
    ap.add_argument("--reject-selfplay-below-baseline", action="store_true")
    ap.add_argument("--reject-baseline-mode", choices=["max_heuristic", "edf", "est", "accepted", "max_all"], default="max_heuristic")
    ap.add_argument("--reject-margin", type=float, default=0.0)
    ap.add_argument("--accepted-baseline-state", default="")
    ap.add_argument("--accepted-baseline-rollouts", type=int, default=0)
    ap.add_argument("--terminal-score-metric", choices=["reward", "mean_health", "final_health"], default="reward")
    ap.add_argument("--terminal-tracked-weight", type=float, default=1.0)
    ap.add_argument("--terminal-drop-weight", type=float, default=0.05)
    ap.add_argument("--terminal-delay-weight", type=float, default=0.001)
    ap.add_argument("--terminal-baseline-scale", type=float, default=1.0)
    ap.add_argument("--terminal-baseline-margin", type=float, default=0.0)
    ap.add_argument("--direct-mode", choices=["prob", "branch", "flat", "q"], default="prob")
    ap.add_argument("--direct-alpha", type=float, default=0.0)
    ap.add_argument("--direct-beta", type=float, default=0.0)
    ap.add_argument("--direct-threshold", type=float, default=0.0)
    ap.add_argument("--reference-mcts-rollouts", type=int, default=8)
    ap.add_argument("--reference-mcts-horizon-ms", type=float, default=200.0)
    ap.add_argument("--reference-mcts-c-puct", type=float, default=1.25)
    ap.add_argument("--reference-mcts-rollout-policy", choices=["greedy", "edf"], default="greedy")
    ap.add_argument("--reference-mcts-select", choices=["visits", "q"], default="q")
    ap.add_argument("--reference-mcts-search-period-ms", type=float, default=160.0)
    ap.add_argument("--cf-top-k", type=int, default=8)
    ap.add_argument("--cf-candidate-mode", choices=["urgent", "delay"], default="urgent")
    ap.add_argument("--cf-policy-tau", type=float, default=10.0)
    ap.add_argument("--cf-rollout-windows", type=int, default=0)
    ap.add_argument("--counterfactual-branch-q", action="store_true")
    ap.add_argument("--counterfactual-top-k", type=int, default=8)
    ap.add_argument(
        "--counterfactual-mode",
        choices=["value", "edge_density", "edge_greedy_rollout", "edge_greedy_potential", "rollout", "model_rollout", "subtree"],
        default="value",
    )
    ap.add_argument("--counterfactual-subrollouts", type=int, default=0)
    ap.add_argument("--counterfactual-candidate-mode", choices=["prior", "urgent"], default="urgent")
    ap.add_argument("--target-start-window", type=int, default=0)
    ap.add_argument("--target-stride", type=int, default=1)
    ap.add_argument("--macro-search-quota", type=int, default=4)
    ap.add_argument("--eval-mode", choices=["mcts", "direct", "none"], default="mcts")
    ap.add_argument("--type-loss-weight", type=float, default=1.0)
    ap.add_argument("--type-search-pos-weight", type=float, default=1.0)
    ap.add_argument("--track-loss-weight", type=float, default=1.0)
    ap.add_argument("--sensor-loss-weight", type=float, default=0.5)
    ap.add_argument("--joint-policy-loss-weight", type=float, default=0.0)
    ap.add_argument("--value-loss-weight", type=float, default=0.5)
    ap.add_argument("--type-q-loss-weight", type=float, default=0.0)
    ap.add_argument("--track-q-loss-weight", type=float, default=0.0)
    ap.add_argument("--sensor-q-loss-weight", type=float, default=0.0)
    ap.add_argument("--factor-value-loss-weight", type=float, default=0.0)
    ap.add_argument("--policy-positive-only", action="store_true")
    ap.add_argument("--policy-positive-margin", type=float, default=0.0)
    ap.add_argument("--policy-kl-weight", type=float, default=0.0)
    ap.add_argument("--train-initials", default="20,60,100")
    ap.add_argument("--train-rates", default="0,4,8")
    ap.add_argument("--eval-seeds", default="901")
    ap.add_argument("--eval-initials", default="20,60,100")
    ap.add_argument("--eval-rates", default="0,4,8")
    ap.add_argument("--env-mode", default="radarxs_mission_delta")
    ap.add_argument("--use-arrival-feature", action="store_true")
    ap.add_argument("--use-grid-feature", action="store_true")
    ap.add_argument("--enable-x-band", action="store_true")
    ap.add_argument("--single-sensor", action="store_true")
    ap.add_argument("--sensor-action-mode", choices=["explicit_head", "implicit"], default="explicit_head")
    ap.add_argument("--disable-x-search", action="store_true")
    ap.add_argument("--canonical-search-only", action="store_true")
    ap.add_argument("--zero-action-rewards", action="store_true")
    ap.add_argument("--track-update-reward", type=float, default=0.30)
    ap.add_argument("--searched-sector-reward-weight", type=float, default=0.25)
    ap.add_argument("--track-loss-penalty", type=float, default=4.0)
    ap.add_argument("--target-service-weight", type=float, default=10.0)
    ap.add_argument("--target-service-horizon-ms", type=float, default=3000.0)
    ap.add_argument("--tracked-target-ms-reward-weight", type=float, default=0.0)
    ap.add_argument("--discovered-target-reward", type=float, default=0.0)
    ap.add_argument("--sector-staleness-weight", type=float, default=0.01)
    ap.add_argument("--search-frame-overdue-weight", type=float, default=0.01)
    ap.add_argument("--search-frame-drop-penalty", type=float, default=4.0)
    ap.add_argument("--save-targets", default="")
    ap.add_argument("--load-targets", default="")
    ap.add_argument("--append-targets", default="")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--skip-before-eval", action="store_true")
    ap.add_argument("--skip-after-eval", action="store_true")
    ap.add_argument("--save-state", default="")
    ap.add_argument("--load-state", default="")
    ap.add_argument("--out-prefix", default="orthodox_smoke")
    args0 = ap.parse_args()
    args0.visit_unvisited_first = not bool(args0.disable_visit_unvisited_first)

    np.random.seed(args0.seed)
    torch.manual_seed(args0.seed)
    exact_args = base_exact_args(args0)
    device = torch.device(args0.device)
    model = load_model(exact_args).to(device)
    if args0.load_state:
        load_state_into(model, args0.load_state, device)
        print(f"loaded model state from {args0.load_state}", flush=True)
    accepted_model = None
    if args0.accepted_baseline_state:
        accepted_model = load_model(exact_args).to(device)
        load_state_into(accepted_model, args0.accepted_baseline_state, device)
        accepted_model.eval()
        print(f"loaded accepted baseline state from {args0.accepted_baseline_state}", flush=True)

    t0 = time.perf_counter()
    before = pd.DataFrame() if args0.skip_before_eval else eval_model(model, exact_args, args0, "before")
    if args0.load_targets:
        targets = load_target_paths(args0.load_targets)
        selfplay = pd.DataFrame()
    else:
        targets, selfplay = collect_targets(model, exact_args, args0, accepted_model)
    if args0.append_targets:
        extra_targets = load_target_paths(args0.append_targets)
        targets.extend(extra_targets)
        print(f"appended targets; total={len(targets)}", flush=True)
    if args0.save_targets:
        save_targets(Path(args0.save_targets), targets)
        print(f"saved {len(targets)} targets to {args0.save_targets}", flush=True)
    targets = temper_target_probs(targets, args0.target_temperature)
    stats = target_stats(targets)
    result_path = output_path_for_prefix(args0.out_prefix, "_eval.csv")
    target_path = output_path_for_prefix(args0.out_prefix, "_selfplay.csv")
    if args0.cache_only:
        pd.DataFrame().to_csv(result_path, index=False)
        selfplay.to_csv(target_path, index=False)
        print("target_stats", stats, flush=True)
        print(f"wrote {target_path}", flush=True)
        print(f"seconds={time.perf_counter() - t0:.1f}", flush=True)
        return
    train_metrics = train_pv(model, targets, args0, device)
    after = pd.DataFrame() if args0.skip_after_eval else eval_model(model, exact_args, args0, "after")
    result = pd.concat([before, after], ignore_index=True)
    result.to_csv(result_path, index=False)
    selfplay.to_csv(target_path, index=False)
    if args0.save_state:
        Path(args0.save_state).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "target_stats": stats, "train_metrics": train_metrics, "args": vars(args0)}, args0.save_state)
    print("target_stats", stats, flush=True)
    print("train_metrics", train_metrics, flush=True)
    if not result.empty and "tag" in result.columns:
        group_cols = ["tag", "planner"] if "planner" in result.columns else ["tag"]
        print(result.groupby(group_cols).agg(reward=("reward", "mean"), search=("search", "mean")).reset_index().to_string(index=False), flush=True)
    else:
        print("eval skipped", flush=True)
    print(f"wrote {result_path}", flush=True)
    print(f"seconds={time.perf_counter() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
