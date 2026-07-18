from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "CreateValid1" / "experiments" / "code" / "model_code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from best_model_joint_vs_seq_ablation import JointAwareLearnedProposalFairExact, WorkConservingAsyncBeamPlanner, WorkConservingAsyncCoupledPlanner, train_variant
from exact_env_mutual import EDFPlanner, ESTPlanner, MAXT, attach_env_obs, engine_env_cfg, env_cfg_for, search_frame_pressure_sum, shaped_step_reward, xs_decode_action, xs_s_search_action, xs_s_track_action, xs_x_search_action, xs_x_track_action
from final_radar_campaign import get_obs
from foundation_mcts_fair_eval import parse_floats, parse_ints
from joint_action_experiment import encode_joint_action, execute_first_valid_action_joint, is_joint_action, split_joint_action
from mutual_features import SLOT_DIM, slot_features, tokenize
from penalty_window_quota_learner_eval import make_exact_args
from repaired_campaign_tools import build_env
from realistic_reward_retrain import adapter
from strict_window_report import sample_state_metrics
from two_sensor_physical_head_eval import PhysicalHeadPlanner, make_physical_model
from pufferlib.ocean.radarxs import binding


@dataclass
class Transition:
    x: np.ndarray
    slot: np.ndarray
    action_pair: np.ndarray
    reward: float
    dt: float
    service_next: np.ndarray
    x_next: np.ndarray
    slot_next: np.ndarray
    episode_id: int = 0


@dataclass
class SequenceTransition:
    x: np.ndarray
    slot: np.ndarray
    slots_seq: np.ndarray
    action_pairs: np.ndarray
    mask: np.ndarray
    service_final: np.ndarray
    reward: float = 0.0
    policy_targets: np.ndarray | None = None


@dataclass
class ActionValueTransition:
    x: np.ndarray
    slot: np.ndarray
    action_pair: np.ndarray
    value: float
    service_next: np.ndarray
    frame_next: float = 0.0


@dataclass
class ActionValueGroup:
    x: np.ndarray
    slot: np.ndarray
    action_pairs: np.ndarray
    values: np.ndarray
    mask: np.ndarray


def action_index(action: int) -> int:
    row, sensor = xs_decode_action(int(action), MAXT)
    return int(max(0, int(row))) * 2 + int(sensor or 0)


def pair_indices(action: int) -> np.ndarray:
    if is_joint_action(int(action)):
        atoms = split_joint_action(int(action))
    else:
        atoms = (int(action), int(action))
    out = [action_index(int(a)) for a in atoms[:2]]
    while len(out) < 2:
        out.append(out[-1] if out else 0)
    return np.asarray(out, dtype=np.int64)


def training_action_pair(action: int, single_sensor: bool = False) -> np.ndarray:
    """Encode executed actions for training.

    In single-sensor mode the second stream is absent.  Use -1 as a no-op
    sentinel so policy/value losses can ignore that branch instead of turning it
    into a fake X-search target.
    """
    if bool(single_sensor):
        atoms = split_joint_action(int(action)) if is_joint_action(int(action)) else (int(action),)
        s_idx = 0
        for atom in atoms:
            row, sensor = xs_decode_action(int(atom), MAXT)
            if int(sensor or 0) == 0:
                s_idx = int(max(0, int(row))) * 2
                break
        return np.asarray([s_idx, -1], dtype=np.int64)
    return pair_indices(int(action))


def action_from_pair_index(idx: int) -> int:
    row = int(idx) // 2
    sensor = int(idx) % 2
    if sensor == 0:
        return xs_s_search_action(MAXT) if row <= 0 else xs_s_track_action(row, MAXT)
    return xs_x_search_action(MAXT) if row <= 0 else xs_x_track_action(row, MAXT)


def joint_action_from_pair_indices(action_pair: np.ndarray) -> int:
    if int(action_pair[1]) < 0:
        return action_from_pair_index(int(action_pair[0]))
    return encode_joint_action(action_from_pair_index(int(action_pair[0])), action_from_pair_index(int(action_pair[1])))


def pair_rows(action_pair: torch.Tensor) -> torch.Tensor:
    return torch.div(action_pair, 2, rounding_mode="floor").long()


def infer_latent_d_model(state: dict, fallback: int) -> int:
    for key in ("sensor_embed", "seq_pos", "action_emb.weight"):
        val = state.get(key) if isinstance(state, dict) else None
        if hasattr(val, "shape") and len(val.shape) >= 2:
            return int(val.shape[-1])
    return int(fallback)


def masked_pair_ce(scores: torch.Tensor, action_pair: torch.Tensor) -> tuple[torch.Tensor, int]:
    valid_action = action_pair >= 0
    rows = pair_rows(action_pair).clamp(min=0, max=MAXT)
    loss = scores.new_tensor(0.0)
    terms = 0
    for sensor in (0, 1):
        logits = scores[:, :, sensor]
        tgt = rows[:, sensor]
        tgt_logit = logits.gather(1, tgt[:, None]).squeeze(1)
        valid_tgt = valid_action[:, sensor] & torch.isfinite(tgt_logit) & (tgt_logit > -1e8)
        if bool(valid_tgt.any()):
            loss = loss + F.cross_entropy(logits[valid_tgt], tgt[valid_tgt])
            terms += 1
    return loss, terms


def masked_pair_factored_ce(
    scores: torch.Tensor,
    action_pair: torch.Tensor,
    type_weight: float = 1.0,
    target_weight: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """Root/transition analogue of the sequence factorized policy loss."""
    valid_action = action_pair >= 0
    rows = pair_rows(action_pair).clamp(min=0, max=MAXT)
    loss = scores.new_tensor(0.0)
    terms = 0
    for sensor in (0, 1):
        logits = scores[:, :, sensor]
        tgt = rows[:, sensor]
        valid = valid_action[:, sensor]
        if not bool(valid.any()):
            continue

        flat_logits = logits[valid]
        flat_tgt = tgt[valid]
        search_logit = flat_logits[:, 0]
        track_logit = torch.logsumexp(flat_logits[:, 1:], dim=1)
        type_logits = torch.stack([search_logit, track_logit], dim=1)
        type_tgt = (flat_tgt > 0).long()
        loss = loss + float(type_weight) * F.cross_entropy(type_logits, type_tgt)
        terms += 1

        track = flat_tgt > 0
        if bool(track.any()):
            target_logits = flat_logits[track, 1:]
            target_tgt = flat_tgt[track] - 1
            tgt_logit = target_logits.gather(1, target_tgt[:, None]).squeeze(1)
            valid_target = torch.isfinite(tgt_logit) & (tgt_logit > -1e8)
            if bool(valid_target.any()):
                loss = loss + float(target_weight) * F.cross_entropy(target_logits[valid_target], target_tgt[valid_target])
                terms += 1
    return loss, terms


def masked_sequence_pair_ce(
    scores: torch.Tensor,
    action_pairs: torch.Tensor,
    mask: torch.Tensor,
    search_weight: float = 1.0,
    seq_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    valid_action = action_pairs >= 0
    rows = pair_rows(action_pairs).clamp(min=0, max=MAXT)
    loss = scores.new_tensor(0.0)
    terms = 0
    for sensor in (0, 1):
        logits = scores[:, :, :, sensor]
        tgt = rows[:, :, sensor]
        valid = mask.bool() & valid_action[:, :, sensor]
        if bool(valid.any()):
            flat_logits = logits[valid]
            flat_tgt = tgt[valid]
            tgt_logit = flat_logits.gather(1, flat_tgt[:, None]).squeeze(1)
            valid_tgt = torch.isfinite(tgt_logit) & (tgt_logit > -1e8)
            if bool(valid_tgt.any()):
                per = F.cross_entropy(flat_logits[valid_tgt], flat_tgt[valid_tgt], reduction="none")
                weights = torch.where(flat_tgt[valid_tgt] == 0, per.new_full(per.shape, float(search_weight)), per.new_ones(per.shape))
                if seq_weight is not None:
                    flat_seq_weight = seq_weight[:, None].expand_as(mask)[valid][valid_tgt]
                    weights = weights * flat_seq_weight.to(weights.dtype)
                loss = loss + (per * weights).sum() / weights.sum().clamp_min(1.0)
                terms += 1
    return loss, terms


def masked_sequence_factored_ce(
    scores: torch.Tensor,
    action_pairs: torch.Tensor,
    mask: torch.Tensor,
    type_weight: float = 1.0,
    target_weight: float = 1.0,
    seq_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Train search/track type separately from track target selection.

    The sequence head is architecturally factorized, but a flat CE over all rows
    makes the single search row much easier than the many track rows. This loss
    applies the same factorization used by the policy interpretation.
    """
    valid_action = action_pairs >= 0
    rows = pair_rows(action_pairs).clamp(min=0, max=MAXT)
    loss = scores.new_tensor(0.0)
    terms = 0
    for sensor in (0, 1):
        logits = scores[:, :, :, sensor]
        tgt = rows[:, :, sensor]
        valid = mask.bool() & valid_action[:, :, sensor]
        if not bool(valid.any()):
            continue

        flat_logits = logits[valid]
        flat_tgt = tgt[valid]
        flat_weight = None
        if seq_weight is not None:
            flat_weight = seq_weight[:, None].expand_as(mask)[valid].to(scores.dtype)

        search_logit = flat_logits[:, 0]
        track_logit = torch.logsumexp(flat_logits[:, 1:], dim=1)
        type_logits = torch.stack([search_logit, track_logit], dim=1)
        type_tgt = (flat_tgt > 0).long()
        type_per = F.cross_entropy(type_logits, type_tgt, reduction="none")
        if flat_weight is not None:
            type_per = type_per * flat_weight
            loss = loss + float(type_weight) * type_per.sum() / flat_weight.sum().clamp_min(1.0)
        else:
            loss = loss + float(type_weight) * type_per.mean()
        terms += 1

        track = flat_tgt > 0
        if bool(track.any()):
            target_logits = flat_logits[track, 1:]
            target_tgt = flat_tgt[track] - 1
            valid_target = torch.isfinite(target_logits.gather(1, target_tgt[:, None]).squeeze(1))
            valid_target = valid_target & (target_logits.gather(1, target_tgt[:, None]).squeeze(1) > -1e8)
            if bool(valid_target.any()):
                target_per = F.cross_entropy(target_logits[valid_target], target_tgt[valid_target], reduction="none")
                if flat_weight is not None:
                    tw = flat_weight[track][valid_target]
                    loss = loss + float(target_weight) * target_per.mul(tw).sum() / tw.sum().clamp_min(1.0)
                else:
                    loss = loss + float(target_weight) * target_per.mean()
                terms += 1
    return loss, terms


def masked_sequence_explicit_factor_ce(
    type_logits: torch.Tensor,
    target_logits: torch.Tensor,
    action_pairs: torch.Tensor,
    mask: torch.Tensor,
    type_weight: float = 1.0,
    target_weight: float = 1.0,
    search_weight: float = 1.0,
    seq_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Train the explicit sequence type and target heads directly."""
    valid_action = action_pairs >= 0
    rows = pair_rows(action_pairs).clamp(min=0, max=MAXT)
    loss = type_logits.new_tensor(0.0)
    terms = 0
    for sensor in (0, 1):
        valid = mask.bool() & valid_action[:, :, sensor]
        if not bool(valid.any()):
            continue
        tgt = rows[:, :, sensor]
        type_tgt = (tgt > 0).long()
        flat_type = type_logits[:, :, sensor, :][valid]
        flat_type_tgt = type_tgt[valid]
        type_per = F.cross_entropy(flat_type, flat_type_tgt, reduction="none")
        weights = torch.where(
            flat_type_tgt == 0,
            type_per.new_full(type_per.shape, float(search_weight)),
            type_per.new_ones(type_per.shape),
        )
        if seq_weight is not None:
            flat_seq_weight = seq_weight[:, None].expand_as(mask)[valid].to(weights.dtype)
            weights = weights * flat_seq_weight
        loss = loss + float(type_weight) * (type_per * weights).sum() / weights.sum().clamp_min(1.0)
        terms += 1

        track = valid & (tgt > 0)
        if bool(track.any()):
            flat_target = target_logits[:, :, 1:, sensor][track]
            target_tgt = tgt[track] - 1
            target_logit = flat_target.gather(1, target_tgt[:, None]).squeeze(1)
            valid_target = torch.isfinite(target_logit) & (target_logit > -1e8)
            if not bool(valid_target.any()):
                continue
            flat_target = flat_target[valid_target]
            target_tgt = target_tgt[valid_target]
            target_per = F.cross_entropy(flat_target, target_tgt, reduction="none")
            if seq_weight is not None:
                tw = seq_weight[:, None].expand_as(mask)[track][valid_target].to(target_per.dtype)
                loss = loss + float(target_weight) * target_per.mul(tw).sum() / tw.sum().clamp_min(1.0)
            else:
                loss = loss + float(target_weight) * target_per.mean()
            terms += 1
    return loss, terms


def masked_sequence_sonly_soft_puct_loss(
    type_logits: torch.Tensor,
    target_logits: torch.Tensor,
    policy_targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy from PUCT search mass into the explicit S-only heads."""
    valid = mask.bool() & (policy_targets.sum(dim=-1) > 0.0)
    if not bool(valid.any()):
        return type_logits.new_tensor(0.0)
    target_mass = policy_targets / policy_targets.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    search_mass = target_mass[:, :, 0]
    track_mass = target_mass[:, :, 1:].sum(dim=-1)
    type_target = torch.stack([search_mass, track_mass], dim=-1)
    type_logp = F.log_softmax(type_logits[:, :, 0, :], dim=-1)
    type_loss = -(type_target * type_logp).sum(dim=-1)[valid].mean()

    track_valid = valid & (track_mass > 1.0e-8)
    if not bool(track_valid.any()):
        return type_loss
    conditional_target = target_mass[:, :, 1:] / track_mass[:, :, None].clamp_min(1.0e-8)
    target_logp = F.log_softmax(target_logits[:, :, 1:, 0], dim=-1)
    finite = torch.isfinite(target_logp) & (target_logp > -1.0e8)
    conditional_target = conditional_target * finite.to(conditional_target.dtype)
    conditional_target = conditional_target / conditional_target.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    target_loss = -(conditional_target * target_logp.masked_fill(~finite, 0.0)).sum(dim=-1)[track_valid].mean()
    return type_loss + target_loss


def masked_sequence_logits_kl(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    mask: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Keep the root sequence decoder close to a frozen teacher distribution."""
    valid_step = mask.bool()
    if not bool(valid_step.any()):
        return student_scores.new_tensor(0.0)
    tau = max(1e-4, float(tau))
    terms = []
    for sensor in (0, 1):
        s_logits = student_scores[:, :, :, sensor]
        t_logits = teacher_scores[:, :, :, sensor]
        finite = (
            torch.isfinite(s_logits)
            & torch.isfinite(t_logits)
            & (s_logits > -1e8)
            & (t_logits > -1e8)
        )
        step_valid = valid_step & finite.any(dim=-1)
        if not bool(step_valid.any()):
            continue
        s_safe = s_logits.masked_fill(~finite, -1e9)
        t_safe = t_logits.masked_fill(~finite, -1e9)
        per = F.kl_div(
            F.log_softmax(s_safe[step_valid] / tau, dim=-1),
            F.softmax(t_safe[step_valid] / tau, dim=-1),
            reduction="batchmean",
        ) * (tau * tau)
        terms.append(per)
    if not terms:
        return student_scores.new_tensor(0.0)
    return sum(terms) / float(len(terms))


def sequence_search_fraction_loss(scores: torch.Tensor, action_pairs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_action = action_pairs >= 0
    rows = pair_rows(action_pairs).clamp(min=0, max=MAXT)
    valid = mask.bool() & valid_action.any(dim=2)
    if not bool(valid.any()):
        return scores.new_tensor(0.0)
    losses = []
    for sensor in (0, 1):
        sensor_mask = mask * valid_action[:, :, sensor].to(mask.dtype)
        if not bool(sensor_mask.bool().any()):
            continue
        logits = scores[:, :, :, sensor]
        finite = torch.isfinite(logits) & (logits > -1e8)
        safe_logits = logits.masked_fill(~finite, -1e9)
        prob_search = torch.softmax(safe_logits, dim=-1)[:, :, 0]
        denom = sensor_mask.sum(dim=1).clamp_min(1.0)
        pred_frac = (prob_search * sensor_mask).sum(dim=1) / denom
        target_frac = (((rows[:, :, sensor] == 0).to(mask.dtype)) * sensor_mask).sum(dim=1) / denom
        losses.append(F.mse_loss(pred_frac, target_frac))
    if not losses:
        return scores.new_tensor(0.0)
    return sum(losses) / float(len(losses))


def sequence_action_count_loss(scores: torch.Tensor, action_pairs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Match aggregate search/track atom counts across a decoded window.

    Per-step cross entropy can copy noisy teacher choices. This auxiliary loss
    keeps the sequence-level service budget aligned: how many atoms search, how
    many atoms track, and how many unique targets are expected to be covered.
    """
    valid_action = action_pairs >= 0
    rows = pair_rows(action_pairs).clamp(min=0, max=MAXT)
    valid = mask[:, :, None].bool() & valid_action
    if not bool(valid.any()):
        return scores.new_tensor(0.0)
    pred_search = []
    pred_track_rows = []
    for sensor in (0, 1):
        logits = scores[:, :, :, sensor]
        finite = torch.isfinite(logits) & (logits > -1e8)
        safe_logits = logits.masked_fill(~finite, -1e9)
        probs = torch.softmax(safe_logits, dim=-1)
        sensor_mask = valid_action[:, :, sensor].to(scores.dtype)
        pred_search.append(probs[:, :, 0] * sensor_mask)
        pred_track_rows.append(probs[:, :, 1:] * sensor_mask[:, :, None])
    target_search = ((rows == 0).to(scores.dtype) * valid.to(scores.dtype)).sum(dim=(1, 2))
    pred_search_count = ((pred_search[0] + pred_search[1]) * mask).sum(dim=1)
    target_track = ((rows > 0).to(scores.dtype) * valid.to(scores.dtype)).sum(dim=(1, 2))
    pred_track_count = (((1.0 - pred_search[0]) * valid_action[:, :, 0].to(scores.dtype) + (1.0 - pred_search[1]) * valid_action[:, :, 1].to(scores.dtype)) * mask).sum(dim=1)

    p_track_row = torch.stack(pred_track_rows, dim=-1) * mask[:, :, None, None]
    pred_covered = 1.0 - (1.0 - p_track_row).clamp(1e-6, 1.0).prod(dim=3).prod(dim=1)
    target_rows = rows[:, :, :, None]
    target_ids = torch.arange(1, MAXT + 1, device=scores.device)[None, None, None, :]
    target_covered = ((target_rows == target_ids) & valid[:, :, :, None]).any(dim=(1, 2)).to(scores.dtype)
    pred_unique = pred_covered.sum(dim=1)
    target_unique = target_covered.sum(dim=1)

    denom_atoms = valid.to(scores.dtype).sum(dim=(1, 2)).clamp_min(1.0)
    search_loss = F.mse_loss(pred_search_count / denom_atoms, target_search / denom_atoms)
    track_loss = F.mse_loss(pred_track_count / denom_atoms, target_track / denom_atoms)
    unique_loss = F.mse_loss(pred_unique / float(MAXT), target_unique / float(MAXT))
    return search_loss + track_loss + 0.5 * unique_loss


def sequence_min_search_atoms_loss(
    scores: torch.Tensor,
    mask: torch.Tensor,
    min_atoms: float,
    active: torch.Tensor | None = None,
    active_threshold: float = 0.0,
) -> torch.Tensor:
    if float(min_atoms) <= 0.0 or not bool(mask.bool().any()):
        return scores.new_tensor(0.0)
    pred_search = []
    for sensor in (0, 1):
        logits = scores[:, :, :, sensor]
        finite = torch.isfinite(logits) & (logits > -1e8)
        safe_logits = logits.masked_fill(~finite, -1e9)
        pred_search.append(torch.softmax(safe_logits, dim=-1)[:, :, 0])
    expected_atoms = ((pred_search[0] + pred_search[1]) * mask).sum(dim=1)
    deficit = F.relu(scores.new_tensor(float(min_atoms)) - expected_atoms)
    if active is not None and float(active_threshold) > 0.0:
        weights = (active >= float(active_threshold)).to(scores.dtype)
        if not bool((weights > 0).any()):
            return scores.new_tensor(0.0)
        return ((deficit.square()) * weights).sum() / weights.sum().clamp_min(1.0)
    return deficit.square().mean()


def policy_search_profile_loss(
    scores: torch.Tensor,
    slot: torch.Tensor,
    low_active_threshold: float = 35.0,
    high_active_threshold: float = 35.0,
    low_max_search_atoms: float = 0.25,
    high_min_search_atoms: float = 1.0,
    high_until_search_atoms: float = 4.0,
) -> torch.Tensor:
    """Load-conditioned search calibration for the one-step latent policy head.

    The deployed tensor-loop repeatedly calls LatentG.policy_scores. Sequence
    losses can look good while this one-step head still over-searches light
    loads and under-searches medium/heavy loads. This loss directly shapes the
    expected number of search atoms for the current latent decision.
    """
    if scores.numel() == 0 or slot.numel() == 0:
        return scores.new_tensor(0.0)
    search_probs = []
    for sensor in (0, 1):
        logits = scores[:, :, sensor]
        finite = torch.isfinite(logits) & (logits > -1e8)
        safe_logits = logits.masked_fill(~finite, -1e9)
        search_probs.append(torch.softmax(safe_logits, dim=-1)[:, 0])
    expected_search_atoms = search_probs[0] + search_probs[1]
    active_count = slot[:, 4] * 100.0
    search_count = slot[:, 1] * 20.0

    losses = []
    if float(low_active_threshold) > 0.0:
        low_mask = active_count < float(low_active_threshold)
        if bool(low_mask.any()):
            excess = F.relu(expected_search_atoms[low_mask] - float(low_max_search_atoms))
            losses.append(excess.square().mean())
    if float(high_active_threshold) > 0.0 and float(high_min_search_atoms) > 0.0:
        high_mask = active_count >= float(high_active_threshold)
        if float(high_until_search_atoms) > 0.0:
            high_mask = high_mask & (search_count < float(high_until_search_atoms))
        if bool(high_mask.any()):
            deficit = F.relu(float(high_min_search_atoms) - expected_search_atoms[high_mask])
            losses.append(deficit.square().mean())
    if not losses:
        return scores.new_tensor(0.0)
    return sum(losses) / float(len(losses))


def sequence_joint_mix_loss(scores: torch.Tensor, action_pairs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_action = action_pairs >= 0
    rows = pair_rows(action_pairs).clamp(min=0, max=MAXT)
    valid_step = mask.bool() & valid_action.all(dim=2)
    if not bool(valid_step.any()):
        return scores.new_tensor(0.0)
    probs = []
    for sensor in (0, 1):
        logits = scores[:, :, :, sensor]
        finite = torch.isfinite(logits) & (logits > -1e8)
        safe_logits = logits.masked_fill(~finite, -1e9)
        probs.append(torch.softmax(safe_logits, dim=-1)[:, :, 0])
    p_s_search, p_x_search = probs
    pred_any_search = 1.0 - (1.0 - p_s_search) * (1.0 - p_x_search)
    pred_both_track = (1.0 - p_s_search) * (1.0 - p_x_search)
    target_s_search = (rows[:, :, 0] == 0).to(mask.dtype)
    target_x_search = (rows[:, :, 1] == 0).to(mask.dtype)
    target_any_search = ((target_s_search > 0) | (target_x_search > 0)).to(mask.dtype)
    target_both_track = ((target_s_search <= 0) & (target_x_search <= 0)).to(mask.dtype)
    step_mask = valid_step.to(mask.dtype)
    denom = step_mask.sum(dim=1).clamp_min(1.0)
    pred_any_frac = (pred_any_search * step_mask).sum(dim=1) / denom
    pred_both_track_frac = (pred_both_track * step_mask).sum(dim=1) / denom
    target_any_frac = (target_any_search * step_mask).sum(dim=1) / denom
    target_both_track_frac = (target_both_track * step_mask).sum(dim=1) / denom
    return 0.5 * (F.mse_loss(pred_any_frac, target_any_frac) + F.mse_loss(pred_both_track_frac, target_both_track_frac))


def sequence_target_coverage_loss(scores: torch.Tensor, action_pairs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_action = action_pairs >= 0
    rows = pair_rows(action_pairs).clamp(min=0, max=MAXT)
    valid = mask[:, :, None].bool() & valid_action
    if not bool(valid.any()):
        return scores.new_tensor(0.0)
    probs = []
    for sensor in (0, 1):
        logits = scores[:, :, :, sensor]
        finite = torch.isfinite(logits) & (logits > -1e8)
        safe_logits = logits.masked_fill(~finite, -1e9)
        sensor_mask = valid_action[:, :, sensor].to(scores.dtype)
        probs.append(torch.softmax(safe_logits, dim=-1)[:, :, 1:] * sensor_mask[:, :, None])
    p_track_row = torch.stack(probs, dim=-1) * mask[:, :, None, None]
    not_covered = (1.0 - p_track_row).clamp(1e-6, 1.0)
    pred_covered = 1.0 - not_covered.prod(dim=3).prod(dim=1)
    target_rows = rows[:, :, :, None]
    target_ids = torch.arange(1, MAXT + 1, device=scores.device)[None, None, None, :]
    target_covered = ((target_rows == target_ids) & valid[:, :, :, None]).any(dim=(1, 2)).to(scores.dtype)
    pred_count = pred_covered.sum(dim=1)
    target_count = target_covered.sum(dim=1)
    count_loss = F.mse_loss(pred_count / float(MAXT), target_count / float(MAXT))
    positive = target_covered > 0
    if bool(positive.any()):
        positive_loss = -torch.log(pred_covered[positive].clamp_min(1e-6)).mean()
    else:
        positive_loss = scores.new_tensor(0.0)
    return count_loss + 0.1 * positive_loss


def sequence_stop_loss(stop_scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_len = mask.sum(dim=1).long().clamp(min=0, max=mask.shape[1])
    stop_target = torch.zeros_like(stop_scores)
    stop_mask = torch.zeros_like(stop_scores)
    weights = torch.ones_like(stop_scores)
    for b in range(mask.shape[0]):
        n = int(valid_len[b].item())
        if n > 0:
            stop_mask[b, :n] = 1.0
        if n < mask.shape[1]:
            stop_target[b, n] = 1.0
            stop_mask[b, n] = 1.0
            weights[b, n] = float(max(1, n))
    if not bool((stop_mask > 0).any()):
        return stop_scores.new_tensor(0.0)
    per = F.binary_cross_entropy_with_logits(stop_scores, stop_target, reduction="none")
    weighted_mask = stop_mask * weights
    return (per * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)


class LatentG(nn.Module):
    def __init__(self, d_model: int = 48, max_actions: int = 2 * (MAXT + 1), seq_len: int = 40, ar_history_k: int = 0):
        super().__init__()
        self.d_model = int(d_model)
        self.seq_len = int(seq_len)
        self.ar_history_k = max(0, int(ar_history_k))
        self.input_proj = nn.Identity() if int(d_model) == 48 else nn.Linear(48, d_model)
        self.action_emb = nn.Embedding(max_actions, d_model)
        self.sensor_embed = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.seq_pos = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        self.slot_proj = nn.Sequential(nn.LayerNorm(SLOT_DIM), nn.Linear(SLOT_DIM, d_model), nn.GELU())
        self.cls_update = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.tok_update = nn.Sequential(nn.LayerNorm(5 * d_model), nn.Linear(5 * d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.reward_dt = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.service_head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 4))
        self.action_service_head = nn.Sequential(nn.LayerNorm(6 * d_model), nn.Linear(6 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 4))
        self.action_value_head = nn.Sequential(nn.LayerNorm(6 * d_model), nn.Linear(6 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.action_frame_head = nn.Sequential(nn.LayerNorm(6 * d_model), nn.Linear(6 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.service_gate = nn.Sequential(nn.Linear(SLOT_DIM + 2, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.type_policy = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.target_policy = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.policy_action_proj = nn.Linear(4 * d_model, d_model)
        self.policy_action_coupler = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=2 * d_model, batch_first=True, dropout=0.0, activation="gelu"),
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.policy_action_residual = nn.Linear(d_model, 1)
        self.policy_tiny_coupler = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=d_model, batch_first=True, dropout=0.0, activation="gelu"),
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.policy_tiny_residual = nn.Linear(d_model, 1)
        self.policy_light_residual = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.policy_action_mixer = "full"
        self.seq_type_policy = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.seq_target_policy = nn.Sequential(nn.LayerNorm(5 * d_model), nn.Linear(5 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.seq_stop_policy = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.seq_context_proj = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, d_model), nn.GELU())
        self.seq_context_coupler = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=2 * d_model, batch_first=True, dropout=0.0, activation="gelu"),
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.seq_context_type_residual = nn.Linear(d_model, 2)
        self.seq_context_target_proj = nn.Linear(d_model, d_model)
        self.seq_context_target_residual = nn.Linear(d_model, 1)
        self.ar_input = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, d_model), nn.GELU())
        if self.ar_history_k > 0:
            self.ar_history_proj = nn.Sequential(
                nn.LayerNorm(2 * self.ar_history_k * d_model),
                nn.Linear(2 * self.ar_history_k * d_model, d_model),
                nn.GELU(),
            )
        self.ar_cell = nn.GRUCell(d_model, d_model)
        self.ar_type_policy = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.ar_target_policy = nn.Sequential(nn.LayerNorm(5 * d_model), nn.Linear(5 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def _action_embedding(self, action: torch.Tensor) -> torch.Tensor:
        """Embed action ids, using a zero vector for disabled-sensor no-ops.

        The latent dynamics model was originally built around an S/X action
        pair. In single-sensor evaluation, forcing the disabled X branch to
        appear as an X-search action corrupts the latent transition. A negative
        action id is treated as an explicit no-op with zero embedding.
        """
        action = action.long()
        valid = action >= 0
        safe = action.clamp(min=0, max=self.action_emb.num_embeddings - 1)
        emb = self.action_emb(safe)
        return emb * valid.to(emb.dtype).unsqueeze(-1)

    def _ar_history_embedding(self, history_pairs: torch.Tensor) -> torch.Tensor:
        if self.ar_history_k <= 0:
            raise RuntimeError("AR history embedding requested with ar_history_k=0")
        safe = history_pairs.long().clamp(min=0, max=self.action_emb.num_embeddings - 1)
        emb = self.action_emb(safe).reshape(history_pairs.shape[0], -1)
        return self.ar_history_proj(emb)

    def _update_ar_history(self, history_pairs: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        if self.ar_history_k <= 0:
            return history_pairs
        return torch.cat([history_pairs[:, 1:, :], prev[:, None, :]], dim=1)

    def _project_state(self, cls: torch.Tensor, tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if cls.shape[-1] != self.d_model:
            cls = self.input_proj(cls)
        if tok.shape[-1] != self.d_model:
            tok = self.input_proj(tok)
        return cls, tok

    def forward(self, cls: torch.Tensor, tok: torch.Tensor, slot: torch.Tensor, action_pair: torch.Tensor):
        cls, tok = self._project_state(cls, tok)
        a0 = self._action_embedding(action_pair[:, 0])
        a1 = self._action_embedding(action_pair[:, 1])
        slot_e = self.slot_proj(slot)
        global_ctx = torch.cat([cls, slot_e, a0, a1], dim=-1)
        cls_next = cls + self.cls_update(global_ctx)
        bsz, rows, _ = tok.shape
        ctx = torch.cat(
            [
                tok,
                cls[:, None, :].expand(bsz, rows, -1),
                slot_e[:, None, :].expand(bsz, rows, -1),
                a0[:, None, :].expand(bsz, rows, -1),
                a1[:, None, :].expand(bsz, rows, -1),
            ],
            dim=-1,
        )
        tok_next = tok + self.tok_update(ctx)
        rd = self.reward_dt(global_ctx)
        return cls_next, tok_next, rd[:, 0], rd[:, 1]

    def predict_service(self, cls: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        cls, tok = self._project_state(cls, tok)
        pooled = tok.mean(dim=1)
        return self.service_head(torch.cat([cls, pooled], dim=-1))

    def predict_action_service(self, cls: torch.Tensor, tok: torch.Tensor, slot: torch.Tensor, action_pair: torch.Tensor) -> torch.Tensor:
        cls, tok = self._project_state(cls, tok)
        a0 = self._action_embedding(action_pair[:, 0])
        a1 = self._action_embedding(action_pair[:, 1])
        rows = pair_rows(action_pair).clamp(min=0, max=tok.shape[1] - 1)
        batch = torch.arange(tok.shape[0], device=tok.device)
        tok0 = tok[batch, rows[:, 0]]
        tok1 = tok[batch, rows[:, 1]]
        slot_e = self.slot_proj(slot)
        return self.action_service_head(torch.cat([cls, slot_e, a0, a1, tok0, tok1], dim=-1))

    def predict_action_value(self, cls: torch.Tensor, tok: torch.Tensor, slot: torch.Tensor, action_pair: torch.Tensor) -> torch.Tensor:
        cls, tok = self._project_state(cls, tok)
        a0 = self._action_embedding(action_pair[:, 0])
        a1 = self._action_embedding(action_pair[:, 1])
        rows = pair_rows(action_pair).clamp(min=0, max=tok.shape[1] - 1)
        batch = torch.arange(tok.shape[0], device=tok.device)
        tok0 = tok[batch, rows[:, 0]]
        tok1 = tok[batch, rows[:, 1]]
        slot_e = self.slot_proj(slot)
        return self.action_value_head(torch.cat([cls, slot_e, a0, a1, tok0, tok1], dim=-1)).squeeze(-1)

    def predict_action_frame(self, cls: torch.Tensor, tok: torch.Tensor, slot: torch.Tensor, action_pair: torch.Tensor) -> torch.Tensor:
        cls, tok = self._project_state(cls, tok)
        a0 = self._action_embedding(action_pair[:, 0])
        a1 = self._action_embedding(action_pair[:, 1])
        rows = pair_rows(action_pair).clamp(min=0, max=tok.shape[1] - 1)
        batch = torch.arange(tok.shape[0], device=tok.device)
        tok0 = tok[batch, rows[:, 0]]
        tok1 = tok[batch, rows[:, 1]]
        slot_e = self.slot_proj(slot)
        return self.action_frame_head(torch.cat([cls, slot_e, a0, a1, tok0, tok1], dim=-1)).squeeze(-1)

    def service_gate_logit(self, slot: torch.Tensor) -> torch.Tensor:
        active = slot[:, 4:5]
        gate_features = torch.cat([slot, active * active, torch.clamp(active - 0.45, min=0.0)], dim=-1)
        return self.service_gate(gate_features).squeeze(-1)

    def policy_scores(self, cls: torch.Tensor, tok: torch.Tensor, slot: torch.Tensor, selected: torch.Tensor, token_active: torch.Tensor) -> torch.Tensor:
        cls, tok = self._project_state(cls, tok)
        bsz, rows, dim = tok.shape
        slot_e = self.slot_proj(slot)
        sensor = self.sensor_embed[None, :, :].expand(bsz, -1, -1)
        cls_s = cls[:, None, :].expand(-1, 2, -1)
        slot_s = slot_e[:, None, :].expand(-1, 2, -1)
        type_logits = self.type_policy(torch.cat([cls_s, slot_s, sensor], dim=-1))
        type_logp = F.log_softmax(type_logits, dim=-1)
        tok_st = tok[:, :, None, :].expand(-1, -1, 2, -1)
        cls_st = cls[:, None, None, :].expand(-1, rows, 2, -1)
        slot_st = slot_e[:, None, None, :].expand(-1, rows, 2, -1)
        sensor_st = sensor[:, None, :, :].expand(-1, rows, -1, -1)
        target_logits = self.target_policy(torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)).squeeze(-1)
        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        target_logp = F.log_softmax(target_logits.masked_fill(~track_mask[:, :, None], -1e9), dim=1)
        scores = slot.new_full((bsz, rows, 2), -1e9)
        scores[:, 0, :] = type_logp[:, :, 0]
        scores[:, 1:, :] = (type_logp[:, None, :, 1] + target_logp)[:, 1:, :]
        row_is_search = torch.arange(rows, device=slot.device)[None, :, None] == 0
        valid = (track_mask[:, :, None] | row_is_search).expand(-1, -1, 2)
        action_ctx = torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)
        action_tokens = self.policy_action_proj(action_ctx).reshape(bsz, rows * 2, dim)
        mixer = str(getattr(self, "policy_action_mixer", "full"))
        if mixer == "light":
            valid_flat = valid.reshape(bsz, rows * 2)
            token_mask = valid_flat[:, :, None].to(action_tokens.dtype)
            pooled = (action_tokens * token_mask).sum(dim=1, keepdim=True) / token_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            light_in = torch.cat([action_tokens, pooled.expand_as(action_tokens)], dim=-1)
            scores = scores + self.policy_light_residual(light_in).reshape(bsz, rows, 2)
        elif mixer == "tiny":
            action_tokens = self.policy_tiny_coupler(action_tokens, src_key_padding_mask=~valid.reshape(bsz, rows * 2))
            scores = scores + self.policy_tiny_residual(action_tokens).reshape(bsz, rows, 2)
        elif mixer != "none":
            action_tokens = self.policy_action_coupler(action_tokens, src_key_padding_mask=~valid.reshape(bsz, rows * 2))
            scores = scores + self.policy_action_residual(action_tokens).reshape(bsz, rows, 2)
        return scores.masked_fill(~valid, -1e9)

    def sequence_scores(
        self,
        cls: torch.Tensor,
        tok: torch.Tensor,
        slot: torch.Tensor,
        selected: torch.Tensor,
        token_active: torch.Tensor,
        action_pairs: torch.Tensor | None = None,
        seq_slots: torch.Tensor | None = None,
        use_step_context: bool = False,
    ) -> torch.Tensor:
        cls, tok = self._project_state(cls, tok)
        bsz, rows, dim = tok.shape
        steps = self.seq_len
        if seq_slots is None:
            slot_e = self.slot_proj(slot)[:, None, :].expand(-1, steps, -1)
        else:
            slot_e = self.slot_proj(seq_slots.reshape(bsz * steps, -1)).reshape(bsz, steps, -1)
        sensor = self.sensor_embed[None, None, :, :].expand(bsz, steps, -1, -1)
        pos = self.seq_pos[None, :, :].expand(bsz, -1, -1)
        cls_sp = cls[:, None, None, :].expand(-1, steps, 2, -1)
        slot_sp = slot_e[:, :, None, :].expand(-1, -1, 2, -1)
        pos_sp = pos[:, :, None, :].expand(-1, -1, 2, -1)
        type_logits = self.seq_type_policy(torch.cat([cls_sp, slot_sp, pos_sp, sensor], dim=-1))
        if bool(use_step_context):
            seq_context = self.seq_context_proj(torch.cat([cls_sp, slot_sp, pos_sp, sensor], dim=-1))
            seq_context = self.seq_context_coupler(seq_context.reshape(bsz, steps * 2, dim)).reshape(bsz, steps, 2, dim)
            type_logits = type_logits + self.seq_context_type_residual(seq_context)
        else:
            seq_context = None
        tok_spt = tok[:, None, :, None, :].expand(-1, steps, -1, 2, -1)
        cls_spt = cls[:, None, None, None, :].expand(-1, steps, rows, 2, -1)
        slot_spt = slot_e[:, :, None, None, :].expand(-1, -1, rows, 2, -1)
        pos_spt = pos[:, :, None, None, :].expand(-1, -1, rows, 2, -1)
        sensor_spt = sensor[:, :, None, :, :].expand(-1, -1, rows, -1, -1)
        target_logits = self.seq_target_policy(torch.cat([tok_spt, cls_spt, slot_spt, pos_spt, sensor_spt], dim=-1)).squeeze(-1)
        if seq_context is not None:
            seq_ctx_spt = seq_context[:, :, None, :, :].expand(-1, -1, rows, -1, -1)
            target_logits = target_logits + self.seq_context_target_residual(tok_spt + self.seq_context_target_proj(seq_ctx_spt)).squeeze(-1)
        scores = slot.new_full((bsz, steps, rows, 2), -1e9)
        scores[:, :, 0, :] = type_logits[:, :, :, 0]
        scores[:, :, 1:, :] = (type_logits[:, :, None, :, 1] + target_logits)[:, :, 1:, :]
        row_is_search = torch.arange(rows, device=slot.device)[None, :, None] == 0
        if action_pairs is None:
            track_mask = token_active & ~selected
            track_mask[:, 0] = False
            valid = (track_mask[:, None, :, None] | row_is_search[:, None, :, :]).expand(-1, steps, -1, 2)
        else:
            selected_step = selected.clone()
            valid_steps = []
            for step in range(steps):
                track_mask = token_active & ~selected_step
                track_mask[:, 0] = False
                valid_steps.append((track_mask[:, :, None] | row_is_search).expand(-1, -1, 2))
                chosen_rows = pair_rows(action_pairs[:, step, :].clamp(min=0, max=self.action_emb.num_embeddings - 1)).clamp(min=0, max=MAXT)
                active_step = chosen_rows > 0
                if bool(active_step.any()):
                    batch_idx = torch.arange(bsz, device=slot.device)[:, None].expand_as(chosen_rows)
                    selected_step[batch_idx[active_step], chosen_rows[active_step]] = True
            valid = torch.stack(valid_steps, dim=1)
        return scores.masked_fill(~valid, -1e9)

    def sequence_factor_logits(
        self,
        cls: torch.Tensor,
        tok: torch.Tensor,
        slot: torch.Tensor,
        selected: torch.Tensor,
        token_active: torch.Tensor,
        seq_slots: torch.Tensor | None = None,
        use_step_context: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls, tok = self._project_state(cls, tok)
        bsz, rows, dim = tok.shape
        steps = self.seq_len
        if seq_slots is None:
            slot_e = self.slot_proj(slot)[:, None, :].expand(-1, steps, -1)
        else:
            slot_e = self.slot_proj(seq_slots.reshape(bsz * steps, -1)).reshape(bsz, steps, -1)
        sensor = self.sensor_embed[None, None, :, :].expand(bsz, steps, -1, -1)
        pos = self.seq_pos[None, :, :].expand(bsz, -1, -1)
        cls_sp = cls[:, None, None, :].expand(-1, steps, 2, -1)
        slot_sp = slot_e[:, :, None, :].expand(-1, -1, 2, -1)
        pos_sp = pos[:, :, None, :].expand(-1, -1, 2, -1)
        type_logits = self.seq_type_policy(torch.cat([cls_sp, slot_sp, pos_sp, sensor], dim=-1))
        if bool(use_step_context):
            seq_context = self.seq_context_proj(torch.cat([cls_sp, slot_sp, pos_sp, sensor], dim=-1))
            seq_context = self.seq_context_coupler(seq_context.reshape(bsz, steps * 2, dim)).reshape(bsz, steps, 2, dim)
            type_logits = type_logits + self.seq_context_type_residual(seq_context)
        else:
            seq_context = None
        tok_spt = tok[:, None, :, None, :].expand(-1, steps, -1, 2, -1)
        cls_spt = cls[:, None, None, None, :].expand(-1, steps, rows, 2, -1)
        slot_spt = slot_e[:, :, None, None, :].expand(-1, -1, rows, 2, -1)
        pos_spt = pos[:, :, None, None, :].expand(-1, -1, rows, 2, -1)
        sensor_spt = sensor[:, :, None, :, :].expand(-1, -1, rows, -1, -1)
        target_logits = self.seq_target_policy(torch.cat([tok_spt, cls_spt, slot_spt, pos_spt, sensor_spt], dim=-1)).squeeze(-1)
        if seq_context is not None:
            seq_ctx_spt = seq_context[:, :, None, :, :].expand(-1, -1, rows, -1, -1)
            target_logits = target_logits + self.seq_context_target_residual(tok_spt + self.seq_context_target_proj(seq_ctx_spt)).squeeze(-1)
        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        valid_target = track_mask[:, None, :, None].expand(-1, steps, -1, 2)
        target_logits = target_logits.masked_fill(~valid_target, -1e9)
        return type_logits, target_logits

    def sequence_stop_scores(
        self,
        cls: torch.Tensor,
        tok: torch.Tensor,
        slot: torch.Tensor,
        seq_slots: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cls, tok = self._project_state(cls, tok)
        bsz = cls.shape[0]
        steps = self.seq_len
        if seq_slots is None:
            slot_e = self.slot_proj(slot)[:, None, :].expand(-1, steps, -1)
        else:
            slot_e = self.slot_proj(seq_slots.reshape(bsz * steps, -1)).reshape(bsz, steps, -1)
        pos = self.seq_pos[None, :, :].expand(bsz, -1, -1)
        cls_s = cls[:, None, :].expand(-1, steps, -1)
        return self.seq_stop_policy(torch.cat([cls_s, slot_e, pos], dim=-1)).squeeze(-1)

    def ar_step_factor_logits(
        self,
        h: torch.Tensor,
        tok: torch.Tensor,
        slot_step: torch.Tensor,
        pos_step: torch.Tensor,
        selected: torch.Tensor,
        token_active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, tok = self._project_state(h, tok)
        bsz, rows, _dim = tok.shape
        slot_e = self.slot_proj(slot_step)
        sensor = self.sensor_embed[None, :, :].expand(bsz, -1, -1)
        h_s = h[:, None, :].expand(-1, 2, -1)
        slot_s = slot_e[:, None, :].expand(-1, 2, -1)
        pos_s = pos_step[:, None, :].expand(-1, 2, -1)
        type_logits = self.ar_type_policy(torch.cat([h_s, slot_s, pos_s, sensor], dim=-1))
        tok_st = tok[:, :, None, :].expand(-1, -1, 2, -1)
        h_st = h[:, None, None, :].expand(-1, rows, 2, -1)
        slot_st = slot_e[:, None, None, :].expand(-1, rows, 2, -1)
        pos_st = pos_step[:, None, None, :].expand(-1, rows, 2, -1)
        sensor_st = sensor[:, None, :, :].expand(-1, rows, -1, -1)
        target_logits = self.ar_target_policy(torch.cat([tok_st, h_st, slot_st, pos_st, sensor_st], dim=-1)).squeeze(-1)
        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        valid_target = track_mask[:, :, None].expand(-1, -1, 2)
        target_logits = target_logits.masked_fill(~valid_target, -1e9)
        return type_logits, target_logits

    def ar_step_scores(
        self,
        h: torch.Tensor,
        tok: torch.Tensor,
        slot_step: torch.Tensor,
        pos_step: torch.Tensor,
        selected: torch.Tensor,
        token_active: torch.Tensor,
    ) -> torch.Tensor:
        type_logits, target_logits = self.ar_step_factor_logits(h, tok, slot_step, pos_step, selected, token_active)
        bsz, rows, _sensors = target_logits.shape
        scores = slot_step.new_full((bsz, rows, 2), -1e9)
        scores[:, 0, :] = type_logits[:, :, 0]
        scores[:, 1:, :] = (type_logits[:, None, :, 1] + target_logits)[:, 1:, :]
        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        row_is_search = torch.arange(rows, device=slot_step.device)[None, :, None] == 0
        valid = (track_mask[:, :, None] | row_is_search).expand(-1, -1, 2)
        return scores.masked_fill(~valid, -1e9)

    def ar_sequence_factor_logits(
        self,
        cls: torch.Tensor,
        tok: torch.Tensor,
        slot: torch.Tensor,
        selected: torch.Tensor,
        token_active: torch.Tensor,
        action_pairs: torch.Tensor | None = None,
        seq_slots: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls, tok = self._project_state(cls, tok)
        bsz, _rows, _dim = tok.shape
        steps = self.seq_len
        if seq_slots is None:
            seq_slots = slot[:, None, :].expand(-1, steps, -1)
        h = cls
        prev = torch.zeros((bsz, 2), dtype=torch.long, device=slot.device)
        prev[:, 1] = 1
        history = None
        if self.ar_history_k > 0:
            history = prev[:, None, :].expand(-1, self.ar_history_k, -1).clone()
        # `selected` from the state encoder is an observation feature.  For an
        # autoregressive window decoder, the within-window exclusion set starts
        # empty and is updated only by actions decoded earlier in this window.
        selected_step = torch.zeros_like(selected)
        type_logits_out = []
        target_logits_out = []
        for step in range(steps):
            slot_step = seq_slots[:, step, :]
            pos_step = self.seq_pos[step][None, :].expand(bsz, -1)
            a0 = self.action_emb(prev[:, 0])
            a1 = self.action_emb(prev[:, 1])
            slot_e = self.slot_proj(slot_step)
            inp = self.ar_input(torch.cat([a0, a1, slot_e, pos_step], dim=-1))
            if history is not None:
                inp = inp + self._ar_history_embedding(history)
            h = self.ar_cell(inp, h)
            type_logits, target_logits = self.ar_step_factor_logits(h, tok, slot_step, pos_step, selected_step, token_active)
            type_logits_out.append(type_logits)
            target_logits_out.append(target_logits)
            if action_pairs is not None:
                prev = action_pairs[:, step, :].clone()
                prev[:, 1] = torch.where(prev[:, 1] < 0, torch.ones_like(prev[:, 1]), prev[:, 1])
                prev = prev.clamp(min=0, max=self.action_emb.num_embeddings - 1)
                chosen_rows = pair_rows(prev).clamp(min=0, max=MAXT)
                active_step = chosen_rows > 0
                if bool(active_step.any()):
                    batch_idx = torch.arange(bsz, device=slot.device)[:, None].expand_as(chosen_rows)
                    selected_step[batch_idx[active_step], chosen_rows[active_step]] = True
            if history is not None:
                history = self._update_ar_history(history, prev)
        return torch.stack(type_logits_out, dim=1), torch.stack(target_logits_out, dim=1)

    def ar_sequence_scores(
        self,
        cls: torch.Tensor,
        tok: torch.Tensor,
        slot: torch.Tensor,
        selected: torch.Tensor,
        token_active: torch.Tensor,
        action_pairs: torch.Tensor | None = None,
        seq_slots: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cls, tok = self._project_state(cls, tok)
        bsz, rows, _dim = tok.shape
        steps = self.seq_len
        if seq_slots is None:
            seq_slots = slot[:, None, :].expand(-1, steps, -1)
        h = cls
        prev = torch.zeros((bsz, 2), dtype=torch.long, device=slot.device)
        prev[:, 1] = 1
        history = None
        if self.ar_history_k > 0:
            history = prev[:, None, :].expand(-1, self.ar_history_k, -1).clone()
        # Start the in-window selected-target mask empty; do not treat the
        # observation's selected_i feature as already scheduled in this window.
        selected_step = torch.zeros_like(selected)
        scores = []
        for step in range(steps):
            slot_step = seq_slots[:, step, :]
            pos_step = self.seq_pos[step][None, :].expand(bsz, -1)
            a0 = self.action_emb(prev[:, 0])
            a1 = self.action_emb(prev[:, 1])
            slot_e = self.slot_proj(slot_step)
            inp = self.ar_input(torch.cat([a0, a1, slot_e, pos_step], dim=-1))
            if history is not None:
                inp = inp + self._ar_history_embedding(history)
            h = self.ar_cell(inp, h)
            scores.append(self.ar_step_scores(h, tok, slot_step, pos_step, selected_step, token_active))
            if action_pairs is not None:
                prev = action_pairs[:, step, :].clone()
                # In S-only data, the disabled X branch is encoded as -1.  Eval
                # uses action index 1 as the disabled/no-op X history token, so
                # keep teacher-forced AR history aligned with inference.
                prev[:, 1] = torch.where(prev[:, 1] < 0, torch.ones_like(prev[:, 1]), prev[:, 1])
                prev = prev.clamp(min=0, max=self.action_emb.num_embeddings - 1)
                chosen_rows = pair_rows(prev).clamp(min=0, max=MAXT)
                active_step = chosen_rows > 0
                if bool(active_step.any()):
                    batch_idx = torch.arange(bsz, device=slot.device)[:, None].expand_as(chosen_rows)
                    selected_step[batch_idx[active_step], chosen_rows[active_step]] = True
            if history is not None:
                history = self._update_ar_history(history, prev)
        return torch.stack(scores, dim=1)


def service_vector(metrics: dict) -> np.ndarray:
    return np.asarray(
        [
            float(metrics.get("active_targets", 0.0)) / 100.0,
            float(metrics.get("tracked_targets", 0.0)) / 100.0,
            float(metrics.get("drop_pct_active", 0.0)) / 100.0,
            float(metrics.get("mean_delay_active", 0.0)) / 1000.0,
        ],
        dtype=np.float32,
    )


def sequence_service_score(service: np.ndarray, tracked_weight: float = 1.0, drop_weight: float = 0.25, delay_weight: float = 0.002) -> float:
    tracked = float(service[1]) * 100.0
    drop_pct = float(service[2]) * 100.0
    delay_ms = float(service[3]) * 1000.0
    return float(tracked_weight * tracked - drop_weight * drop_pct - delay_weight * delay_ms)


def sequence_reward_service_adjustment(service: np.ndarray, spent_ms: float, args) -> float:
    active = float(service[0]) * 100.0
    drop_pct = float(service[2]) * 100.0
    drop_count = drop_pct * active / 100.0
    delay_ms = float(service[3]) * 1000.0
    drop_pct_weight = float(getattr(args, "seq_reward_drop_pct_penalty_weight", 0.0))
    drop_count_weight = float(getattr(args, "seq_reward_drop_count_penalty_weight", 0.0))
    delay_weight = float(getattr(args, "seq_reward_delay_penalty_weight", 0.0))
    underuse_weight = float(getattr(args, "seq_reward_underuse_penalty_weight", 0.0))
    underuse_target = float(getattr(args, "seq_reward_underuse_target_frac", 0.0))
    service_penalty = -(
        drop_pct_weight * drop_pct
        + drop_count_weight * drop_count
        + delay_weight * (delay_ms / 1000.0)
    )
    target_ms = max(0.0, min(1.0, underuse_target)) * 200.0
    underuse_penalty = -underuse_weight * max(0.0, target_ms - float(spent_ms)) / 200.0
    return float(service_penalty + underuse_penalty)


def filter_sequence_data_by_reward(seq_data: list[SequenceTransition], args) -> list[SequenceTransition]:
    q = float(getattr(args, "filter_seq_reward_quantile", 0.0))
    if not seq_data or q <= 0.0:
        return seq_data
    q = min(max(q, 0.0), 0.95)
    by_active = bool(getattr(args, "filter_seq_reward_by_active_bin", False))
    kept: list[SequenceTransition] = []
    groups: dict[int, list[SequenceTransition]] = {}
    if by_active:
        for item in seq_data:
            active_count = int(round(float(item.service_final[0]) * 100.0 / 10.0) * 10)
            groups.setdefault(active_count, []).append(item)
    else:
        groups[0] = list(seq_data)
    details = {}
    for key, items in groups.items():
        rewards = np.asarray([float(item.reward) for item in items], dtype=np.float32)
        threshold = float(np.quantile(rewards, q))
        group_keep = [item for item in items if float(item.reward) >= threshold]
        if not group_keep and items:
            group_keep = [items[int(np.argmax(rewards))]]
        kept.extend(group_keep)
        details[int(key)] = {
            "before": int(len(items)),
            "after": int(len(group_keep)),
            "reward_threshold": threshold,
            "reward_min": float(rewards.min()) if rewards.size else 0.0,
            "reward_mean": float(rewards.mean()) if rewards.size else 0.0,
            "reward_max": float(rewards.max()) if rewards.size else 0.0,
        }
    print(
        {
            "filter_seq_reward_quantile": q,
            "filter_seq_reward_by_active_bin": by_active,
            "seq_before": int(len(seq_data)),
            "seq_after": int(len(kept)),
            "groups": details,
        },
        flush=True,
    )
    return kept


def latent_scores(model, cls, tok, slot, selected, token_active):
    slot_emb = model.backbone.slot_proj(slot)
    bsz, rows, _ = tok.shape
    sensor = model.sensor_embed[None, :, :].expand(bsz, -1, -1)
    cls_s = cls[:, None, :].expand(-1, 2, -1)
    slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
    sensor_state = model.sensor_state_proj(torch.cat([cls_s, slot_s, sensor], dim=-1))
    coupled_sensor = model.sensor_coupler(sensor_state)
    type_ctx = torch.cat([cls_s, slot_s, coupled_sensor], dim=-1)
    type_logits = model.type_head(type_ctx)
    type_logp = F.log_softmax(type_logits, dim=-1)
    type_q = model.type_q_head(type_ctx)
    tok_st = tok[:, :, None, :].expand(-1, -1, 2, -1)
    cls_st = cls[:, None, None, :].expand(-1, rows, 2, -1)
    slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
    sensor_st = coupled_sensor[:, None, :, :].expand(bsz, rows, -1, -1)
    target_ctx = torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)
    target_logits = model.target_head(target_ctx).squeeze(-1)
    target_q = model.target_q_head(target_ctx).squeeze(-1)
    scores = slot.new_full((bsz, rows, 2), -1e9)
    q = slot.new_zeros((bsz, rows, 2))
    track_mask = token_active & ~selected
    track_mask[:, 0] = False
    target_logp = F.log_softmax(target_logits.masked_fill(~track_mask[:, :, None], -1e9), dim=1)
    scores[:, 0, :] = type_logp[:, :, 0]
    q[:, 0, :] = type_q[:, :, 0]
    scores[:, 1:, :] = (type_logp[:, None, :, 1] + target_logp)[:, 1:, :]
    q[:, 1:, :] = (type_q[:, None, :, 1] + target_q)[:, 1:, :]
    row_is_search = torch.arange(rows, device=slot.device)[None, :, None] == 0
    valid = (track_mask[:, :, None] | row_is_search).expand(-1, -1, 2)
    action_ctx = model.action_proj(target_ctx).reshape(bsz, rows * 2, -1)
    action_ctx = model.action_coupler(action_ctx, src_key_padding_mask=~valid.reshape(bsz, rows * 2))
    scores = scores + model.action_policy_residual(action_ctx).reshape(bsz, rows, 2)
    q = q + model.action_q_residual(action_ctx).reshape(bsz, rows, 2)
    return scores.masked_fill(~valid, -1e9), q.masked_fill(~valid, 0.0)


def collect(args, model, exact_args) -> list[Transition]:
    adapt = adapter()
    rows: list[Transition] = []
    episode_id = 0

    def teacher_plan(planner, eng, debt: float, obs: dict, budget_ms: float) -> list[int]:
        if hasattr(planner, "plan"):
            return list(planner.plan(obs, budget_ms=budget_ms))
        if hasattr(planner, "choose"):
            plan, _meta = planner.choose(eng, debt, obs)
            return [int(a) for a in plan]
        raise AttributeError(f"{type(planner).__name__} has neither plan() nor choose()")

    for seed in parse_ints(args.seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                collect_teacher = str(getattr(args, "collect_teacher", "direct"))
                single_sensor = bool(getattr(args, "single_sensor", False))
                env_cfg = env_cfg_for(float(rate), exact_args)
                env_cfg["enable_x_band"] = 0 if single_sensor else 1
                base = PhysicalHeadPlanner(
                    model,
                    args.variant,
                    env_cfg,
                    policy_weight=float(getattr(args, "teacher_policy_weight", 1.0)),
                    q_weight=float(getattr(args, "teacher_q_weight", 1.0)),
                    search_score_bias=float(getattr(args, "teacher_search_bias", -12.0)),
                )
                if single_sensor and collect_teacher == "exact_env_puct":
                    from current_sonly_exact_puct import ExactSOnlyPuctPlanner

                    planner = ExactSOnlyPuctPlanner(
                        str(args.base_state),
                        str(args.variant),
                        env_cfg,
                        device=str(args.device),
                        simulations=int(getattr(args, "teacher_puct_simulations", 4)),
                        expand_top_k=int(getattr(args, "teacher_puct_expand_top_k", 6)),
                        rollout_steps=int(getattr(args, "teacher_puct_rollout_steps", 2)),
                        rollout_windows=int(getattr(args, "teacher_puct_rollout_windows", 1)),
                        init_child_rollouts=int(getattr(args, "teacher_puct_init_child_rollouts", 0)),
                        c_puct=float(getattr(args, "teacher_puct_c", 1.25)),
                        discount=float(getattr(args, "teacher_puct_discount", 0.997)),
                        select_mode=str(getattr(args, "teacher_puct_select", "q")),
                        policy_weight=float(getattr(args, "teacher_policy_weight", 1.0)),
                        q_weight=float(getattr(args, "teacher_q_weight", 0.5)),
                        search_bias=float(getattr(args, "teacher_search_bias", 0.0)),
                        terminal_service_weight=float(getattr(args, "teacher_puct_terminal_service_weight", 0.0)),
                        terminal_search_frame_weight=float(getattr(args, "teacher_puct_terminal_search_frame_weight", 0.0)),
                        prior_uniform_mix=float(getattr(args, "teacher_puct_prior_uniform_mix", 0.0)),
                        root_dirichlet_alpha=float(getattr(args, "teacher_puct_root_dirichlet_alpha", 0.3)),
                        root_dirichlet_fraction=float(getattr(args, "teacher_puct_root_dirichlet_fraction", 0.0)),
                        progressive_widening_c=float(getattr(args, "teacher_puct_progressive_widening_c", 2.0)),
                        progressive_widening_alpha=float(getattr(args, "teacher_puct_progressive_widening_alpha", 0.5)),
                    )
                    puct_stepwise_sequence = True
                elif single_sensor and collect_teacher == "sband_tensor_loop":
                    from eval_action_attention_muzero_g import LatentG as EvalLatentG, LatentMuZeroPlanner

                    teacher_state = str(getattr(args, "hybrid_g_state", "") or getattr(args, "init_g_state", ""))
                    ckpt = torch.load(teacher_state, map_location=str(getattr(args, "hybrid_device", "cuda" if torch.cuda.is_available() else "cpu")))
                    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                    seq_len = int(state["seq_pos"].shape[0]) if isinstance(state, dict) and "seq_pos" in state else int(args.seq_len)
                    teacher_d_model = infer_latent_d_model(state, int(args.g_d_model))
                    g_fast = EvalLatentG(d_model=teacher_d_model, seq_len=seq_len).to(str(getattr(args, "hybrid_device", "cuda" if torch.cuda.is_available() else "cpu"))).eval()
                    g_fast.load_state_dict(state, strict=False)
                    planner = LatentMuZeroPlanner(
                        model,
                        g_fast,
                        env_cfg,
                        policy_weight=float(getattr(args, "teacher_policy_weight", 1.0)),
                        q_weight=float(getattr(args, "teacher_q_weight", 0.25)),
                        search_score_bias=float(getattr(args, "teacher_search_bias", 0.0)),
                        max_steps=int(getattr(args, "seq_len", 40)),
                        tensor_loop=True,
                        cuda_graph_tensor_loop=bool(torch.cuda.is_available() and str(getattr(args, "hybrid_device", "cuda")).startswith("cuda")),
                        cuda_graph_s_only_score=bool(torch.cuda.is_available() and str(getattr(args, "hybrid_device", "cuda")).startswith("cuda")),
                        single_sensor_noop_action=True,
                        device=str(getattr(args, "hybrid_device", "cuda" if torch.cuda.is_available() else "cpu")),
                    )
                    puct_stepwise_sequence = False
                elif single_sensor and collect_teacher in {"direct", "single_sensor_ar", "single_sensor_edf", "single_sensor_est"}:
                    from single_sensor_ar_action_attention import CachedSingleSensorActionAttentionAR, SingleSensorHeuristicAdapter

                    if collect_teacher == "single_sensor_edf":
                        planner = SingleSensorHeuristicAdapter(EDFPlanner(MAXT))
                    elif collect_teacher == "single_sensor_est":
                        planner = SingleSensorHeuristicAdapter(ESTPlanner(MAXT))
                    else:
                        planner = CachedSingleSensorActionAttentionAR(
                            base,
                            max_steps=int(getattr(args, "seq_len", 40)),
                            search_floor=int(getattr(args, "teacher_search_floor", 0)),
                            search_cap_frac=float(getattr(args, "teacher_search_cap_frac", 1.0)),
                            search_score_bias=float(getattr(args, "teacher_search_bias", 0.0)),
                            env_cfg=env_cfg,
                            single_sensor_action_only=bool(getattr(args, "teacher_single_sensor_action_only", False)),
                        )
                    puct_stepwise_sequence = False
                elif collect_teacher == "exact_workconserving":
                    learned = WorkConservingAsyncBeamPlanner(base, per_sensor_top=int(getattr(args, "teacher_per_sensor_top", 3)), beams=12, include_search_candidate=True)
                    planner = JointAwareLearnedProposalFairExact(
                        env_cfg,
                        [learned],
                        top_k=8,
                        score_horizon_ms=float(getattr(args, "collect_score_horizon_ms", 200.0)),
                        slots=96,
                        generator="structured",
                        seed=15008,
                        learned_extra_top_k=0,
                        force_learned_rescore=True,
                    )
                    puct_stepwise_sequence = False
                elif collect_teacher in {"hybrid_fullgate", "rootseq_repair"}:
                    from eval_action_attention_muzero_g import HybridRiskPlanner, LatentG as EvalLatentG, LatentMuZeroPlanner

                    teacher_state = str(getattr(args, "hybrid_g_state", "") or getattr(args, "init_g_state", ""))
                    ckpt = torch.load(teacher_state, map_location=str(getattr(args, "hybrid_device", "cpu")))
                    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                    seq_len = int(state["seq_pos"].shape[0]) if isinstance(state, dict) and "seq_pos" in state else int(args.seq_len)
                    teacher_d_model = infer_latent_d_model(state, int(args.g_d_model))
                    g_fast = EvalLatentG(d_model=teacher_d_model, seq_len=seq_len).to(str(getattr(args, "hybrid_device", "cpu"))).eval()
                    g_fast.load_state_dict(state, strict=False)
                    g_alt = None
                    alt_state_path = str(getattr(args, "hybrid_alt_g_state", ""))
                    if alt_state_path:
                        ckpt_alt = torch.load(alt_state_path, map_location=str(getattr(args, "hybrid_device", "cpu")))
                        alt_state = ckpt_alt["state_dict"] if isinstance(ckpt_alt, dict) and "state_dict" in ckpt_alt else ckpt_alt
                        alt_seq_len = int(alt_state["seq_pos"].shape[0]) if isinstance(alt_state, dict) and "seq_pos" in alt_state else seq_len
                        alt_d_model = infer_latent_d_model(alt_state, teacher_d_model)
                        g_alt = EvalLatentG(d_model=alt_d_model, seq_len=alt_seq_len).to(str(getattr(args, "hybrid_device", "cpu"))).eval()
                        g_alt.load_state_dict(alt_state, strict=False)
                    fast = LatentMuZeroPlanner(
                        model,
                        g_fast,
                        env_cfg,
                        policy_weight=float(getattr(args, "hybrid_policy_weight", 1.0)),
                        q_weight=float(getattr(args, "hybrid_q_weight", 1.0)),
                        search_score_bias=float(getattr(args, "hybrid_search_bias", -7.0)),
                        service_track_weight=float(getattr(args, "hybrid_service_track_weight", 1.0)),
                        service_search_weight=float(getattr(args, "hybrid_service_search_weight", 0.5)),
                        max_steps=int(getattr(args, "hybrid_max_steps", args.seq_len)),
                        lookahead_width=int(getattr(args, "hybrid_lookahead_width", 0)),
                        lookahead_leaf_weight=float(getattr(args, "hybrid_lookahead_leaf_weight", 0.25)),
                        service_critic_weight=float(getattr(args, "hybrid_service_critic_weight", 0.0)),
                        service_critic_active_weight=float(getattr(args, "hybrid_service_critic_active_weight", 0.25)),
                        service_critic_tracked_weight=float(getattr(args, "hybrid_service_critic_tracked_weight", 1.0)),
                        service_critic_drop_weight=float(getattr(args, "hybrid_service_critic_drop_weight", 1.5)),
                        service_critic_delay_weight=float(getattr(args, "hybrid_service_critic_delay_weight", 0.3)),
                        direct_action_value_weight=float(getattr(args, "hybrid_direct_action_value_weight", 0.0)),
                        direct_action_value_max_steps=int(getattr(args, "hybrid_direct_action_value_max_steps", 0)),
                        use_root_seq_policy=True,
                        max_window_search_frac=float(getattr(args, "hybrid_max_window_search_frac", 0.65)),
                        min_window_search_atoms=int(getattr(args, "hybrid_min_window_search_atoms", 0)),
                        service_sort_plan=bool(getattr(args, "hybrid_service_sort_plan", False)),
                        service_sort_search_prefix=int(getattr(args, "hybrid_service_sort_search_prefix", 0)),
                        pressure_repair_threshold=float(getattr(args, "hybrid_pressure_repair_threshold", 0.0)),
                        pressure_repair_max_atoms=int(getattr(args, "hybrid_pressure_repair_max_atoms", 0)),
                        g_alt=g_alt,
                        router_active_threshold=int(getattr(args, "hybrid_router_active_threshold", 0)),
                        router_alt_search_bias=float(getattr(args, "hybrid_router_alt_search_bias", getattr(args, "hybrid_search_bias", -7.0))),
                        router_alt_max_window_search_frac=float(getattr(args, "hybrid_router_alt_max_window_search_frac", getattr(args, "hybrid_max_window_search_frac", 0.65))),
                        router_alt_service_sort_search_prefix=int(getattr(args, "hybrid_router_alt_service_sort_search_prefix", getattr(args, "hybrid_service_sort_search_prefix", 0))),
                        device=str(getattr(args, "hybrid_device", "cpu")),
                    )
                    if collect_teacher == "rootseq_repair":
                        planner = fast
                    else:
                        full = WorkConservingAsyncCoupledPlanner(
                            base,
                            per_sensor_top=int(getattr(args, "teacher_per_sensor_top", 3)),
                            include_search_candidate=True,
                        )
                        planner = HybridRiskPlanner(
                            fast,
                            full,
                            active_threshold=int(getattr(args, "hybrid_active_threshold", 55)),
                            overdue_threshold=int(getattr(args, "hybrid_overdue_threshold", 0)),
                            pressure_threshold=float(getattr(args, "hybrid_pressure_threshold", 0.0)),
                            max_full_fraction=float(getattr(args, "hybrid_max_full_fraction", 0.35)),
                        )
                else:
                    planner = WorkConservingAsyncCoupledPlanner(base, per_sensor_top=int(getattr(args, "teacher_per_sensor_top", 3)), include_search_candidate=True)
                eng = build_env(planner, int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
                debt = 0.0
                cell_start = len(rows)
                cell_cap = int(getattr(args, "max_transitions_per_cell", 0))
                cell_both_search = 0
                try:
                    for _window in range(int(args.windows)):
                        spent = 0.0
                        selected = set()
                        search_count = track_count = 0
                        last = -1
                        plan = None
                        plan_pos = 0
                        while (
                            spent < 200.0
                            and len(rows) < int(args.max_transitions)
                            and (cell_cap <= 0 or len(rows) - cell_start < cell_cap)
                            and not bool(eng.term_buf[0])
                        ):
                            obs = attach_env_obs(get_obs(eng, debt), env_cfg, True, True)
                            x = tokenize(adapt, obs, selected=selected, search_count=search_count).astype(np.float32)
                            slot = slot_features(obs, spent, search_count, track_count, last, 200.0).astype(np.float32)
                            if str(getattr(args, "collect_mode", "receding")) == "full_window":
                                if plan is None:
                                    plan = teacher_plan(planner, eng, debt, obs, 200.0)
                                next_plan = [int(plan[plan_pos])] if plan_pos < len(plan) else []
                                plan_pos += 1
                            else:
                                if bool(locals().get("puct_stepwise_sequence", False)):
                                    next_plan = [
                                        int(
                                            planner.choose_action(
                                                eng,
                                                debt,
                                                selected,
                                                spent,
                                                search_count,
                                                track_count,
                                                last,
                                                200.0 - spent,
                                            )
                                        )
                                    ]
                                else:
                                    next_plan = teacher_plan(planner, eng, debt, obs, 200.0 - spent)
                            if not next_plan:
                                break
                            obs_before_reward = get_obs(eng, debt)
                            reward, dt, executed = execute_first_valid_action_joint(eng, [int(next_plan[0])], 200.0 - spent)
                            if executed is None or dt <= 0.0:
                                break
                            atoms = split_joint_action(int(executed)) if is_joint_action(int(executed)) else (int(executed),)
                            if any(xs_decode_action(int(a), MAXT)[0] == 0 for a in atoms):
                                debt = 0.0
                            else:
                                debt += float(dt)
                            for atom in atoms:
                                row, _sensor = xs_decode_action(int(atom), MAXT)
                                if int(row) == 0:
                                    search_count += 1
                                elif int(row) > 0:
                                    selected.add(int(row))
                                    track_count += 1
                                last = int(row)
                            spent += float(dt)
                            obs_after_reward = get_obs(eng, debt)
                            reward = shaped_step_reward(float(reward), float(dt), obs_before_reward, obs_after_reward, env_cfg, action=int(executed))
                            obs_next = attach_env_obs(obs_after_reward, env_cfg, True, True)
                            x_next = tokenize(adapt, obs_next, selected=selected, search_count=search_count).astype(np.float32)
                            slot_next = slot_features(obs_next, spent, search_count, track_count, last, 200.0).astype(np.float32)
                            service_next = service_vector(sample_state_metrics(eng, debt))
                            action_pair = training_action_pair(int(executed), single_sensor=single_sensor)
                            pair_rows = action_pair // 2
                            is_both_search = bool(np.all(pair_rows == 0))
                            search_frac = float(search_count) / max(1.0, float(search_count + track_count))
                            accept = True
                            max_search_frac = float(getattr(args, "collect_max_search_frac", 0.0))
                            max_drop_pct = float(getattr(args, "collect_max_drop_pct", 0.0))
                            max_delay_ms = float(getattr(args, "collect_max_delay_ms", 0.0))
                            max_both_search_frac = float(getattr(args, "collect_max_both_search_transition_frac", 0.0))
                            if max_search_frac > 0.0 and search_frac > max_search_frac:
                                accept = False
                            if max_drop_pct > 0.0 and float(service_next[2]) * 100.0 > max_drop_pct:
                                accept = False
                            if max_delay_ms > 0.0 and float(service_next[3]) * 1000.0 > max_delay_ms:
                                accept = False
                            if accept and max_both_search_frac > 0.0 and is_both_search:
                                quota_base = cell_cap if cell_cap > 0 else int(args.max_transitions)
                                allowed_both_search = max(1, int(max_both_search_frac * float(max(1, quota_base))))
                                if cell_both_search >= allowed_both_search:
                                    accept = False
                            if accept:
                                if is_both_search:
                                    cell_both_search += 1
                                rows.append(Transition(x, slot, action_pair, float(reward), float(dt), service_next, x_next, slot_next, episode_id))
                            if len(rows) >= int(args.max_transitions):
                                break
                        if cell_cap > 0 and len(rows) - cell_start >= cell_cap:
                            break
                finally:
                    eng.close()
                cell_n = len(rows) - cell_start
                print(
                    {
                        "collected": len(rows),
                        "cell_collected": cell_n,
                        "cell_both_search_frac": float(cell_both_search) / max(1.0, float(cell_n)),
                        "initial": int(initial),
                        "rate": float(rate),
                        "seed": int(seed),
                    },
                    flush=True,
                )
                if len(rows) >= int(args.max_transitions):
                    return rows
                episode_id += 1
    return rows


def build_return_action_value_data(data: list[Transition], args) -> list[ActionValueTransition]:
    if not data:
        return []
    horizon = max(1, int(getattr(args, "action_value_return_horizon", 1)))
    gamma = float(getattr(args, "action_value_return_discount", 1.0))
    out: list[ActionValueTransition] = []
    rewards = np.asarray([float(t.reward) for t in data], dtype=np.float32)
    for i, item in enumerate(data):
        total = 0.0
        discount = 1.0
        episode_id = int(getattr(item, "episode_id", 0))
        for j in range(i, min(len(data), i + horizon)):
            if int(getattr(data[j], "episode_id", 0)) != episode_id:
                break
            total += discount * float(rewards[j])
            discount *= gamma
        out.append(ActionValueTransition(item.x, item.slot, item.action_pair, float(total), item.service_next, 0.0))
    return out


def action_value_target(reward: float, service: np.ndarray, args, service_before: np.ndarray | None = None) -> float:
    mode = str(getattr(args, "action_value_target_mode", "absolute"))
    if mode == "delta":
        if service_before is None:
            raise ValueError("action_value_target_mode=delta requires service_before")
        before = np.asarray(service_before, dtype=np.float32)
        after = np.asarray(service, dtype=np.float32)
        active_before = float(before[0])
        active_after = float(after[0])
        tracked_delta = float(after[1] - before[1])
        drop_delta = float(after[2] - before[2])
        delay_delta = float(after[3] - before[3])
        drop_count_delta = active_after * float(after[2]) - active_before * float(before[2])
        return float(
            float(reward)
            + float(args.av_tracked_weight) * tracked_delta
            - float(args.av_drop_weight) * drop_delta
            - float(getattr(args, "av_drop_count_weight", 0.0)) * drop_count_delta
            - float(args.av_delay_weight) * delay_delta
        )
    active = float(service[0])
    tracked = float(service[1])
    drop = float(service[2])
    delay = float(service[3])
    drop_count = active * drop
    return float(
        float(reward)
        + float(args.av_tracked_weight) * tracked
        - float(args.av_drop_weight) * drop
        - float(getattr(args, "av_drop_count_weight", 0.0)) * drop_count
        - float(args.av_delay_weight) * delay
    )


def rollout_remaining_window_value(
    eng,
    planner,
    debt: float,
    env_cfg: dict,
    remaining_ms: float,
    args,
    future_windows: int = 0,
) -> tuple[float, np.ndarray]:
    total = 0.0
    local_debt = float(debt)
    remaining = float(max(0.0, remaining_ms))
    for horizon_idx in range(max(1, int(future_windows) + 1)):
        if bool(eng.term_buf[0]):
            break
        if horizon_idx > 0:
            remaining = 200.0
        while remaining > 0.0 and not bool(eng.term_buf[0]):
            obs = attach_env_obs(get_obs(eng, local_debt), env_cfg, True, True)
            plan = planner.plan(obs, budget_ms=max(1.0, remaining))
            if not plan:
                break
            obs_before = get_obs(eng, local_debt)
            reward, dt, executed = execute_first_valid_action_joint(eng, [int(plan[0])], remaining)
            if executed is None or dt <= 0.0:
                break
            atoms = split_joint_action(int(executed)) if is_joint_action(int(executed)) else (int(executed),)
            if any(xs_decode_action(int(a), MAXT)[0] == 0 for a in atoms):
                local_debt = 0.0
            else:
                local_debt += float(dt)
            obs_after = get_obs(eng, local_debt)
            total += shaped_step_reward(float(reward), float(dt), obs_before, obs_after, env_cfg, action=int(executed))
            remaining -= float(dt)
    return float(total), service_vector(sample_state_metrics(eng, local_debt))


def collect_action_value_counterfactuals(args, model, exact_args) -> list[ActionValueTransition]:
    adapt = adapter()
    rows: list[ActionValueTransition] = []
    for seed in parse_ints(args.seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                env_cfg = env_cfg_for(float(rate), exact_args)
                single_sensor = bool(getattr(args, "single_sensor", False))
                env_cfg["enable_x_band"] = 0 if single_sensor else 1
                base = PhysicalHeadPlanner(
                    model,
                    args.variant,
                    env_cfg,
                    policy_weight=float(getattr(args, "teacher_policy_weight", 1.0)),
                    q_weight=float(getattr(args, "teacher_q_weight", 1.0)),
                    search_score_bias=float(getattr(args, "teacher_search_bias", -12.0)),
                )
                planner = WorkConservingAsyncCoupledPlanner(base, per_sensor_top=int(getattr(args, "teacher_per_sensor_top", 3)), include_search_candidate=True)
                eng = build_env(planner, int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
                debt = 0.0
                cell_start = len(rows)
                cell_cap = int(getattr(args, "max_action_value_transitions_per_cell", 0))
                try:
                    group_seq = 0
                    for _window in range(int(args.windows)):
                        spent = 0.0
                        selected: set[int] = set()
                        search_count = track_count = 0
                        last = -1
                        while (
                            spent < 200.0
                            and len(rows) < int(args.max_action_value_transitions)
                            and (cell_cap <= 0 or len(rows) - cell_start < cell_cap)
                            and not bool(eng.term_buf[0])
                        ):
                            obs = attach_env_obs(get_obs(eng, debt), env_cfg, True, True)
                            x = tokenize(adapt, obs, selected=selected, search_count=search_count).astype(np.float32)
                            slot = slot_features(obs, spent, search_count, track_count, last, 200.0).astype(np.float32)
                            collect_transition = (
                                int(_window) >= int(getattr(args, "action_value_start_window", 0))
                                and (group_seq % max(1, int(getattr(args, "action_value_stride", 1)))) == 0
                            )
                            if collect_transition:
                                score = base.score_actions(obs, selected=selected, elapsed=spent, search_count=search_count, track_count=track_count, last=last)
                                score = np.asarray(score, dtype=np.float32)
                                service_before = service_vector(sample_state_metrics(eng, debt))
                                k = max(1, int(args.action_value_candidate_topk))
                                cand_s = np.argsort(-score[:, 0])[:k]
                                cand_x = np.asarray([-1], dtype=np.int64) if single_sensor else np.argsort(-score[:, 1])[:k]
                                if bool(args.action_value_include_search):
                                    cand_s = np.unique(np.concatenate([np.asarray([0], dtype=np.int64), cand_s.astype(np.int64)]))
                                    if not single_sensor:
                                        cand_x = np.unique(np.concatenate([np.asarray([0], dtype=np.int64), cand_x.astype(np.int64)]))
                                snapshot = binding.vec_snapshot(eng.env)
                                for s_row in cand_s:
                                    for x_row in cand_x:
                                        s_row = int(s_row)
                                        x_row = int(x_row)
                                        if s_row > 0 and s_row == x_row:
                                            continue
                                        action_pair = np.asarray([s_row * 2, -1 if single_sensor else x_row * 2 + 1], dtype=np.int64)
                                        action = joint_action_from_pair_indices(action_pair)
                                        binding.vec_restore(eng.env, snapshot)
                                        obs_before_reward = get_obs(eng, debt)
                                        reward, dt, executed = execute_first_valid_action_joint(eng, [int(action)], 200.0 - spent)
                                        if executed is None or dt <= 0.0:
                                            continue
                                        atoms = split_joint_action(int(executed)) if is_joint_action(int(executed)) else (int(executed),)
                                        next_debt = 0.0 if any(xs_decode_action(int(a), MAXT)[0] == 0 for a in atoms) else float(debt) + float(dt)
                                        obs_after_reward = get_obs(eng, next_debt)
                                        shaped = shaped_step_reward(float(reward), float(dt), obs_before_reward, obs_after_reward, env_cfg, action=int(executed))
                                        service_next = service_vector(sample_state_metrics(eng, next_debt))
                                        frame_next = float(search_frame_pressure_sum(obs_after_reward, env_cfg))
                                        if bool(getattr(args, "action_value_rollout_window", False)):
                                            future_reward, future_service = rollout_remaining_window_value(
                                                eng,
                                                planner,
                                                next_debt,
                                                env_cfg,
                                                float(200.0 - spent - float(dt)),
                                                args,
                                                future_windows=int(getattr(args, "action_value_rollout_future_windows", 0)),
                                            )
                                            shaped += future_reward
                                            service_next = future_service
                                        rows.append(ActionValueTransition(x, slot, training_action_pair(int(executed), single_sensor=single_sensor), action_value_target(shaped, service_next, args, service_before=service_before), service_next, frame_next))
                                        if len(rows) >= int(args.max_action_value_transitions):
                                            break
                                    if len(rows) >= int(args.max_action_value_transitions):
                                        break
                                binding.vec_restore(eng.env, snapshot)
                            group_seq += 1
                            plan = planner.plan(obs, budget_ms=200.0 - spent)
                            if not plan:
                                break
                            reward, dt, executed = execute_first_valid_action_joint(eng, [int(plan[0])], 200.0 - spent)
                            if executed is None or dt <= 0.0:
                                break
                            atoms = split_joint_action(int(executed)) if is_joint_action(int(executed)) else (int(executed),)
                            if any(xs_decode_action(int(a), MAXT)[0] == 0 for a in atoms):
                                debt = 0.0
                            else:
                                debt += float(dt)
                            for atom in atoms:
                                row, _sensor = xs_decode_action(int(atom), MAXT)
                                if int(row) == 0:
                                    search_count += 1
                                elif int(row) > 0:
                                    selected.add(int(row))
                                    track_count += 1
                                last = int(row)
                            spent += float(dt)
                        if cell_cap > 0 and len(rows) - cell_start >= cell_cap:
                            break
                finally:
                    eng.close()
                print({"action_value_collected": len(rows), "cell_collected": len(rows) - cell_start, "initial": int(initial), "rate": float(rate), "seed": int(seed)}, flush=True)
                if len(rows) >= int(args.max_action_value_transitions):
                    return rows
    return rows


def collect_action_value_groups(args, model, exact_args) -> list[ActionValueGroup]:
    adapt = adapter()
    groups: list[ActionValueGroup] = []
    max_pairs = max(1, int(args.action_value_group_max_pairs))
    max_seconds = float(getattr(args, "action_value_group_max_seconds", 0.0) or 0.0)
    progress_interval = max(0, int(getattr(args, "action_value_group_progress_interval", 0) or 0))
    start_time = time.perf_counter()
    eval_count = 0

    def timed_out() -> bool:
        return max_seconds > 0.0 and (time.perf_counter() - start_time) >= max_seconds

    for seed in parse_ints(args.seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                if timed_out():
                    print(
                        {
                            "action_value_group_timeout": True,
                            "groups": len(groups),
                            "elapsed_s": time.perf_counter() - start_time,
                        },
                        flush=True,
                    )
                    return groups
                env_cfg = env_cfg_for(float(rate), exact_args)
                single_sensor = bool(getattr(args, "single_sensor", False))
                env_cfg["enable_x_band"] = 0 if single_sensor else 1
                base = PhysicalHeadPlanner(
                    model,
                    args.variant,
                    env_cfg,
                    policy_weight=float(getattr(args, "teacher_policy_weight", 1.0)),
                    q_weight=float(getattr(args, "teacher_q_weight", 1.0)),
                    search_score_bias=float(getattr(args, "teacher_search_bias", -12.0)),
                )
                planner = WorkConservingAsyncCoupledPlanner(base, per_sensor_top=int(getattr(args, "teacher_per_sensor_top", 3)), include_search_candidate=True)
                eng = build_env(planner, int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
                debt = 0.0
                cell_start = len(groups)
                cell_cap = int(getattr(args, "max_action_value_groups_per_cell", 0))
                try:
                    group_seq = 0
                    for _window in range(int(args.windows)):
                        if timed_out():
                            print(
                                {
                                    "action_value_group_timeout": True,
                                    "groups": len(groups),
                                    "elapsed_s": time.perf_counter() - start_time,
                                },
                                flush=True,
                            )
                            return groups
                        spent = 0.0
                        selected: set[int] = set()
                        search_count = track_count = 0
                        last = -1
                        while (
                            spent < 200.0
                            and len(groups) < int(args.max_action_value_groups)
                            and (cell_cap <= 0 or len(groups) - cell_start < cell_cap)
                            and not bool(eng.term_buf[0])
                        ):
                            obs = attach_env_obs(get_obs(eng, debt), env_cfg, True, True)
                            x = tokenize(adapt, obs, selected=selected, search_count=search_count).astype(np.float32)
                            slot = slot_features(obs, spent, search_count, track_count, last, 200.0).astype(np.float32)
                            collect_group = (
                                int(_window) >= int(args.action_value_group_start_window)
                                and (group_seq % max(1, int(args.action_value_group_stride))) == 0
                            )
                            if collect_group:
                                if timed_out():
                                    print(
                                        {
                                            "action_value_group_timeout": True,
                                            "groups": len(groups),
                                            "elapsed_s": time.perf_counter() - start_time,
                                        },
                                        flush=True,
                                    )
                                    return groups
                                service_before = service_vector(sample_state_metrics(eng, debt))
                                score = np.asarray(base.score_actions(obs, selected=selected, elapsed=spent, search_count=search_count, track_count=track_count, last=last), dtype=np.float32)
                                k = max(1, int(args.action_value_candidate_topk))
                                cand_s = np.argsort(-score[:, 0])[:k]
                                cand_x = np.asarray([-1], dtype=np.int64) if single_sensor else np.argsort(-score[:, 1])[:k]
                                if bool(args.action_value_include_search):
                                    cand_s = np.unique(np.concatenate([np.asarray([0], dtype=np.int64), cand_s.astype(np.int64)]))
                                    if not single_sensor:
                                        cand_x = np.unique(np.concatenate([np.asarray([0], dtype=np.int64), cand_x.astype(np.int64)]))
                                snapshot = binding.vec_snapshot(eng.env)
                                pair_list = []
                                value_list = []
                                for s_row in cand_s:
                                    for x_row in cand_x:
                                        eval_count += 1
                                        if progress_interval > 0 and (eval_count % progress_interval) == 0:
                                            print(
                                                {
                                                    "action_value_group_progress": True,
                                                    "groups": len(groups),
                                                    "candidate_evals": eval_count,
                                                    "elapsed_s": time.perf_counter() - start_time,
                                                    "initial": int(initial),
                                                    "rate": float(rate),
                                                    "seed": int(seed),
                                                    "window": int(_window),
                                                },
                                                flush=True,
                                            )
                                        if timed_out():
                                            print(
                                                {
                                                    "action_value_group_timeout": True,
                                                    "groups": len(groups),
                                                    "candidate_evals": eval_count,
                                                    "elapsed_s": time.perf_counter() - start_time,
                                                },
                                                flush=True,
                                            )
                                            return groups
                                        s_row = int(s_row)
                                        x_row = int(x_row)
                                        if s_row > 0 and s_row == x_row:
                                            continue
                                        action_pair = np.asarray([s_row * 2, -1 if single_sensor else x_row * 2 + 1], dtype=np.int64)
                                        action = joint_action_from_pair_indices(action_pair)
                                        binding.vec_restore(eng.env, snapshot)
                                        obs_before_reward = get_obs(eng, debt)
                                        reward, dt, executed = execute_first_valid_action_joint(eng, [int(action)], 200.0 - spent)
                                        if executed is None or dt <= 0.0:
                                            continue
                                        atoms = split_joint_action(int(executed)) if is_joint_action(int(executed)) else (int(executed),)
                                        next_debt = 0.0 if any(xs_decode_action(int(a), MAXT)[0] == 0 for a in atoms) else float(debt) + float(dt)
                                        obs_after_reward = get_obs(eng, next_debt)
                                        shaped = shaped_step_reward(float(reward), float(dt), obs_before_reward, obs_after_reward, env_cfg, action=int(executed))
                                        service_next = service_vector(sample_state_metrics(eng, next_debt))
                                        if bool(args.action_value_group_rollout_window):
                                            future_reward, future_service = rollout_remaining_window_value(
                                                eng,
                                                planner,
                                                next_debt,
                                                env_cfg,
                                                float(200.0 - spent - float(dt)),
                                                args,
                                                future_windows=int(args.action_value_group_rollout_future_windows),
                                            )
                                            shaped += future_reward
                                            service_next = future_service
                                        pair_list.append(training_action_pair(int(executed), single_sensor=single_sensor))
                                        value_list.append(action_value_target(shaped, service_next, args, service_before=service_before))
                                binding.vec_restore(eng.env, snapshot)
                                if len(pair_list) >= 2:
                                    order = np.argsort(-np.asarray(value_list, dtype=np.float32))[:max_pairs]
                                    pairs = np.zeros((max_pairs, 2), dtype=np.int64)
                                    vals = np.zeros((max_pairs,), dtype=np.float32)
                                    mask = np.zeros((max_pairs,), dtype=np.float32)
                                    for out_i, src_i in enumerate(order):
                                        pairs[out_i] = pair_list[int(src_i)]
                                        vals[out_i] = float(value_list[int(src_i)])
                                        mask[out_i] = 1.0
                                    groups.append(ActionValueGroup(x, slot, pairs, vals, mask))
                            group_seq += 1
                            plan = planner.plan(obs, budget_ms=200.0 - spent)
                            if not plan:
                                break
                            reward, dt, executed = execute_first_valid_action_joint(eng, [int(plan[0])], 200.0 - spent)
                            if executed is None or dt <= 0.0:
                                break
                            atoms = split_joint_action(int(executed)) if is_joint_action(int(executed)) else (int(executed),)
                            if any(xs_decode_action(int(a), MAXT)[0] == 0 for a in atoms):
                                debt = 0.0
                            else:
                                debt += float(dt)
                            for atom in atoms:
                                row, _sensor = xs_decode_action(int(atom), MAXT)
                                if int(row) == 0:
                                    search_count += 1
                                elif int(row) > 0:
                                    selected.add(int(row))
                                    track_count += 1
                                last = int(row)
                            spent += float(dt)
                        if cell_cap > 0 and len(groups) - cell_start >= cell_cap:
                            break
                finally:
                    eng.close()
                print({"action_value_groups": len(groups), "cell_groups": len(groups) - cell_start, "initial": int(initial), "rate": float(rate), "seed": int(seed)}, flush=True)
                if len(groups) >= int(args.max_action_value_groups):
                    return groups
    return groups


def urgent_track_rows(obs: dict, limit: int = 2) -> list[int]:
    active = np.asarray(obs.get("active_mask", []), dtype=bool)
    desired = np.asarray(obs.get("t_desired", []), dtype=np.float32)
    deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
    n = min(len(active), len(desired), len(deadline))
    rows: list[tuple[float, int]] = []
    for idx in range(n):
        if not bool(active[idx]) or float(deadline[idx]) < 0.0:
            continue
        late = max(0.0, -float(desired[idx]) / 1000.0)
        risk = max(0.0, (1000.0 - float(deadline[idx])) / 1000.0)
        rows.append((late + 2.0 * risk, idx + 1))
    rows.sort(reverse=True)
    return [row for _score, row in rows[: max(1, int(limit))]]


def dedupe_action_pairs(pairs: list[np.ndarray]) -> list[np.ndarray]:
    seen: set[tuple[int, int]] = set()
    out: list[np.ndarray] = []
    for pair in pairs:
        key = (int(pair[0]), int(pair[1]))
        if key in seen:
            continue
        seen.add(key)
        out.append(np.asarray(pair, dtype=np.int64))
    return out


def collect_hard_negative_groups(args, model, exact_args) -> list[ActionValueGroup]:
    adapt = adapter()
    groups: list[ActionValueGroup] = []
    max_pairs = max(2, int(args.hard_negative_max_pairs))
    max_seconds = float(getattr(args, "hard_negative_max_seconds", 0.0) or 0.0)
    progress_interval = max(0, int(getattr(args, "hard_negative_progress_interval", 0) or 0))
    start_time = time.perf_counter()
    eval_count = 0

    def timed_out() -> bool:
        return max_seconds > 0.0 and (time.perf_counter() - start_time) >= max_seconds

    for seed in parse_ints(args.seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                if timed_out():
                    print(
                        {
                            "hard_negative_timeout": True,
                            "groups": int(len(groups)),
                            "elapsed_s": float(time.perf_counter() - start_time),
                        },
                        flush=True,
                    )
                    return groups
                env_cfg = env_cfg_for(float(rate), exact_args)
                single_sensor = bool(getattr(args, "single_sensor", False))
                env_cfg["enable_x_band"] = 0 if single_sensor else 1
                base = PhysicalHeadPlanner(
                    model,
                    args.variant,
                    env_cfg,
                    policy_weight=float(getattr(args, "teacher_policy_weight", 1.0)),
                    q_weight=float(getattr(args, "teacher_q_weight", 1.0)),
                    search_score_bias=float(getattr(args, "teacher_search_bias", -10.0)),
                )
                planner = WorkConservingAsyncCoupledPlanner(
                    base,
                    per_sensor_top=int(getattr(args, "teacher_per_sensor_top", 3)),
                    include_search_candidate=True,
                )
                eng = build_env(planner, int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
                debt = 0.0
                cell_start = len(groups)
                cell_cap = int(getattr(args, "max_hard_negative_groups_per_cell", 0))
                try:
                    for _window in range(int(args.windows)):
                        if bool(eng.term_buf[0]) or len(groups) >= int(args.max_hard_negative_groups):
                            break
                        if cell_cap > 0 and len(groups) - cell_start >= cell_cap:
                            break
                        if timed_out():
                            print(
                                {
                                    "hard_negative_timeout": True,
                                    "groups": int(len(groups)),
                                    "elapsed_s": float(time.perf_counter() - start_time),
                                    "initial": int(initial),
                                    "rate": float(rate),
                                    "seed": int(seed),
                                    "window": int(_window),
                                },
                                flush=True,
                            )
                            return groups
                        obs = attach_env_obs(get_obs(eng, debt), env_cfg, True, True)
                        active_n = int(np.sum(np.asarray(obs.get("active_mask", []), dtype=bool)))
                        plan = planner.plan(obs, budget_ms=200.0)
                        if active_n >= int(args.hard_negative_min_active) and plan:
                            x = tokenize(adapt, obs, selected=set(), search_count=0).astype(np.float32)
                            slot = slot_features(obs, 0.0, 0, 0, -1, 200.0).astype(np.float32)
                            urgent = urgent_track_rows(obs, limit=2)
                            row_a = urgent[0] if urgent else 1
                            row_b = urgent[1] if len(urgent) > 1 else row_a
                            if single_sensor:
                                candidates = dedupe_action_pairs(
                                    [
                                        training_action_pair(int(plan[0]), single_sensor=True),
                                        np.asarray([0, -1], dtype=np.int64),
                                        np.asarray([row_a * 2, -1], dtype=np.int64),
                                        np.asarray([row_b * 2, -1], dtype=np.int64),
                                    ]
                                )[:max_pairs]
                            else:
                                candidates = dedupe_action_pairs(
                                    [
                                        pair_indices(int(plan[0])),
                                        np.asarray([0, 1], dtype=np.int64),
                                        np.asarray([0, row_a * 2 + 1], dtype=np.int64),
                                        np.asarray([row_a * 2, 1], dtype=np.int64),
                                        np.asarray([row_a * 2, row_b * 2 + 1], dtype=np.int64),
                                    ]
                                )[:max_pairs]
                            snapshot = binding.vec_snapshot(eng.env)
                            pair_list: list[np.ndarray] = []
                            value_list: list[float] = []
                            for pair in candidates:
                                eval_count += 1
                                if progress_interval > 0 and (eval_count % progress_interval) == 0:
                                    print(
                                        {
                                            "hard_negative_progress": True,
                                            "groups": int(len(groups)),
                                            "candidate_evals": int(eval_count),
                                            "elapsed_s": float(time.perf_counter() - start_time),
                                            "initial": int(initial),
                                            "rate": float(rate),
                                            "seed": int(seed),
                                            "window": int(_window),
                                        },
                                        flush=True,
                                    )
                                if timed_out():
                                    print(
                                        {
                                            "hard_negative_timeout": True,
                                            "groups": int(len(groups)),
                                            "candidate_evals": int(eval_count),
                                            "elapsed_s": float(time.perf_counter() - start_time),
                                        },
                                        flush=True,
                                    )
                                    return groups
                                binding.vec_restore(eng.env, snapshot)
                                obs_before = get_obs(eng, debt)
                                action = joint_action_from_pair_indices(pair)
                                reward, dt, executed = execute_first_valid_action_joint(eng, [int(action)], 200.0)
                                if executed is None or dt <= 0.0:
                                    continue
                                atoms = split_joint_action(int(executed)) if is_joint_action(int(executed)) else (int(executed),)
                                next_debt = 0.0 if any(xs_decode_action(int(a), MAXT)[0] == 0 for a in atoms) else float(debt) + float(dt)
                                obs_after = get_obs(eng, next_debt)
                                shaped = shaped_step_reward(float(reward), float(dt), obs_before, obs_after, env_cfg, action=int(executed))
                                service_next = service_vector(sample_state_metrics(eng, next_debt))
                                future_windows = int(args.hard_negative_future_windows)
                                if future_windows > 0:
                                    future_reward, future_service = rollout_remaining_window_value(
                                        eng,
                                        planner,
                                        next_debt,
                                        env_cfg,
                                        max(0.0, 200.0 - float(dt)),
                                        args,
                                        future_windows=future_windows,
                                    )
                                    shaped += float(future_reward)
                                    service_next = future_service
                                pair_list.append(training_action_pair(int(executed), single_sensor=single_sensor))
                                value_list.append(action_value_target(shaped, service_next, args))
                            binding.vec_restore(eng.env, snapshot)
                            if len(pair_list) >= 2:
                                order = np.argsort(-np.asarray(value_list, dtype=np.float32))[:max_pairs]
                                pairs = np.zeros((max_pairs, 2), dtype=np.int64)
                                vals = np.zeros((max_pairs,), dtype=np.float32)
                                mask = np.zeros((max_pairs,), dtype=np.float32)
                                for out_i, src_i in enumerate(order):
                                    pairs[out_i] = pair_list[int(src_i)]
                                    vals[out_i] = float(value_list[int(src_i)])
                                    mask[out_i] = 1.0
                                groups.append(ActionValueGroup(x, slot, pairs, vals, mask))
                        if not plan:
                            break
                        reward, dt, executed = execute_first_valid_action_joint(eng, [int(plan[0])], 200.0)
                        if executed is None or dt <= 0.0:
                            break
                        atoms = split_joint_action(int(executed)) if is_joint_action(int(executed)) else (int(executed),)
                        debt = 0.0 if any(xs_decode_action(int(a), MAXT)[0] == 0 for a in atoms) else float(debt) + float(dt)
                finally:
                    eng.close()
                print(
                    {
                        "hard_negative_groups": int(len(groups)),
                        "cell_groups": int(len(groups) - cell_start),
                        "initial": int(initial),
                        "rate": float(rate),
                        "seed": int(seed),
                    },
                    flush=True,
                )
                if len(groups) >= int(args.max_hard_negative_groups):
                    return groups
    return groups


def collect_sequences(args, model, exact_args) -> list[SequenceTransition]:
    adapt = adapter()
    rows: list[SequenceTransition] = []
    seq_hybrid_search_biases = parse_floats(str(getattr(args, "collect_seq_hybrid_search_biases", "")))
    seq_hybrid_max_steps = parse_ints(str(getattr(args, "collect_seq_hybrid_max_steps", "")))
    seq_hybrid_search_fracs = parse_floats(str(getattr(args, "collect_seq_hybrid_search_fracs", "")))

    def apply_sequence_collection_variant(planner, idx: int) -> dict:
        changed = {}
        target = getattr(planner, "fast", planner)
        if seq_hybrid_search_biases and hasattr(target, "search_score_bias"):
            value = float(seq_hybrid_search_biases[idx % len(seq_hybrid_search_biases)])
            target.search_score_bias = value
            changed["search_bias"] = value
        if seq_hybrid_max_steps and hasattr(target, "max_steps"):
            value = int(seq_hybrid_max_steps[idx % len(seq_hybrid_max_steps)])
            target.max_steps = value
            changed["max_steps"] = value
        if seq_hybrid_search_fracs and hasattr(target, "max_window_search_frac"):
            value = float(seq_hybrid_search_fracs[idx % len(seq_hybrid_search_fracs)])
            target.max_window_search_frac = value
            changed["max_window_search_frac"] = value
        return changed

    for seed in parse_ints(args.seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                cell_start = len(rows)
                cell_cap = int(getattr(args, "max_sequences_per_cell", 0) or 0)
                single_sensor = bool(getattr(args, "single_sensor", False))
                env_cfg = env_cfg_for(float(rate), exact_args)
                env_cfg["enable_x_band"] = 0 if single_sensor else 1
                base = PhysicalHeadPlanner(
                    model,
                    args.variant,
                    env_cfg,
                    policy_weight=float(getattr(args, "teacher_policy_weight", 1.0)),
                    q_weight=float(getattr(args, "teacher_q_weight", 1.0)),
                    search_score_bias=float(getattr(args, "teacher_search_bias", -12.0)),
                )
                collect_teacher = str(getattr(args, "collect_teacher", "direct"))
                if single_sensor and collect_teacher == "exact_env_puct":
                    from current_sonly_exact_puct import ExactSOnlyPuctPlanner

                    planner = ExactSOnlyPuctPlanner(
                        str(args.base_state),
                        str(args.variant),
                        env_cfg,
                        device=str(args.device),
                        simulations=int(getattr(args, "teacher_puct_simulations", 4)),
                        expand_top_k=int(getattr(args, "teacher_puct_expand_top_k", 6)),
                        rollout_steps=int(getattr(args, "teacher_puct_rollout_steps", 2)),
                        rollout_windows=int(getattr(args, "teacher_puct_rollout_windows", 1)),
                        init_child_rollouts=int(getattr(args, "teacher_puct_init_child_rollouts", 0)),
                        c_puct=float(getattr(args, "teacher_puct_c", 1.25)),
                        discount=float(getattr(args, "teacher_puct_discount", 0.997)),
                        select_mode=str(getattr(args, "teacher_puct_select", "q")),
                        policy_weight=float(getattr(args, "teacher_policy_weight", 1.0)),
                        q_weight=float(getattr(args, "teacher_q_weight", 0.5)),
                        search_bias=float(getattr(args, "teacher_search_bias", 0.0)),
                        terminal_service_weight=float(getattr(args, "teacher_puct_terminal_service_weight", 0.0)),
                        terminal_search_frame_weight=float(getattr(args, "teacher_puct_terminal_search_frame_weight", 0.0)),
                        prior_uniform_mix=float(getattr(args, "teacher_puct_prior_uniform_mix", 0.0)),
                        root_dirichlet_alpha=float(getattr(args, "teacher_puct_root_dirichlet_alpha", 0.3)),
                        root_dirichlet_fraction=float(getattr(args, "teacher_puct_root_dirichlet_fraction", 0.0)),
                        progressive_widening_c=float(getattr(args, "teacher_puct_progressive_widening_c", 2.0)),
                        progressive_widening_alpha=float(getattr(args, "teacher_puct_progressive_widening_alpha", 0.5)),
                    )
                    # Use the same receding-step PUCT path as online
                    # evaluation.  The planner.choose() helper builds an
                    # internal full-window plan, but that path produced a
                    # different search/track distribution from the evaluated
                    # online PUCT teacher.  Sequence distillation must imitate
                    # the stepwise root planner, not the stale full-plan helper.
                    puct_stepwise_sequence = True
                elif single_sensor and collect_teacher == "sband_tensor_loop":
                    from eval_action_attention_muzero_g import LatentG as EvalLatentG, LatentMuZeroPlanner

                    teacher_state = str(getattr(args, "hybrid_g_state", "") or getattr(args, "init_g_state", ""))
                    teacher_device = str(getattr(args, "hybrid_device", "cuda" if torch.cuda.is_available() else "cpu"))
                    ckpt = torch.load(teacher_state, map_location=teacher_device)
                    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                    seq_len = int(state["seq_pos"].shape[0]) if isinstance(state, dict) and "seq_pos" in state else int(args.seq_len)
                    teacher_d_model = infer_latent_d_model(state, int(args.g_d_model))
                    g_fast = EvalLatentG(d_model=teacher_d_model, seq_len=seq_len).to(teacher_device).eval()
                    g_fast.load_state_dict(state, strict=False)
                    planner = LatentMuZeroPlanner(
                        model,
                        g_fast,
                        env_cfg,
                        policy_weight=float(getattr(args, "teacher_policy_weight", 1.0)),
                        q_weight=float(getattr(args, "teacher_q_weight", 0.25)),
                        search_score_bias=float(getattr(args, "teacher_search_bias", 0.0)),
                        max_steps=int(getattr(args, "seq_len", 40)),
                        tensor_loop=True,
                        cuda_graph_tensor_loop=bool(torch.cuda.is_available() and teacher_device.startswith("cuda")),
                        cuda_graph_s_only_score=bool(torch.cuda.is_available() and teacher_device.startswith("cuda")),
                        single_sensor_noop_action=True,
                        device=teacher_device,
                    )
                elif single_sensor and collect_teacher in {"direct", "single_sensor_ar", "single_sensor_edf", "single_sensor_est"}:
                    from single_sensor_ar_action_attention import CachedSingleSensorActionAttentionAR, SingleSensorHeuristicAdapter

                    if collect_teacher == "single_sensor_edf":
                        planner = SingleSensorHeuristicAdapter(EDFPlanner(MAXT))
                    elif collect_teacher == "single_sensor_est":
                        planner = SingleSensorHeuristicAdapter(ESTPlanner(MAXT))
                    else:
                        planner = CachedSingleSensorActionAttentionAR(
                            base,
                            max_steps=int(getattr(args, "seq_len", 40)),
                            search_floor=int(getattr(args, "teacher_search_floor", 0)),
                            search_cap_frac=float(getattr(args, "teacher_search_cap_frac", 1.0)),
                            search_score_bias=float(getattr(args, "teacher_search_bias", 0.0)),
                            env_cfg=env_cfg,
                            single_sensor_action_only=bool(getattr(args, "teacher_single_sensor_action_only", False)),
                        )
                elif collect_teacher == "exact_workconserving":
                    learned = WorkConservingAsyncBeamPlanner(
                        base,
                        per_sensor_top=int(getattr(args, "teacher_per_sensor_top", 3)),
                        beams=12,
                        include_search_candidate=True,
                    )
                    planner = JointAwareLearnedProposalFairExact(
                        env_cfg,
                        [learned],
                        top_k=8,
                        score_horizon_ms=float(getattr(args, "collect_score_horizon_ms", 200.0)),
                        slots=96,
                        generator="structured",
                        seed=15008,
                        learned_extra_top_k=0,
                        force_learned_rescore=True,
                    )
                elif collect_teacher in {"hybrid_fullgate", "rootseq_repair"}:
                    from eval_action_attention_muzero_g import HybridRiskPlanner, LatentG as EvalLatentG, LatentMuZeroPlanner

                    teacher_state = str(getattr(args, "hybrid_g_state", "") or getattr(args, "init_g_state", ""))
                    ckpt = torch.load(teacher_state, map_location=str(getattr(args, "hybrid_device", "cpu")))
                    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                    seq_len = int(state["seq_pos"].shape[0]) if isinstance(state, dict) and "seq_pos" in state else int(args.seq_len)
                    teacher_d_model = infer_latent_d_model(state, int(args.g_d_model))
                    g_fast = EvalLatentG(d_model=teacher_d_model, seq_len=seq_len).to(str(getattr(args, "hybrid_device", "cpu"))).eval()
                    g_fast.load_state_dict(state, strict=False)
                    g_alt = None
                    alt_state_path = str(getattr(args, "hybrid_alt_g_state", ""))
                    if alt_state_path:
                        ckpt_alt = torch.load(alt_state_path, map_location=str(getattr(args, "hybrid_device", "cpu")))
                        alt_state = ckpt_alt["state_dict"] if isinstance(ckpt_alt, dict) and "state_dict" in ckpt_alt else ckpt_alt
                        alt_seq_len = int(alt_state["seq_pos"].shape[0]) if isinstance(alt_state, dict) and "seq_pos" in alt_state else seq_len
                        alt_d_model = infer_latent_d_model(alt_state, teacher_d_model)
                        g_alt = EvalLatentG(d_model=alt_d_model, seq_len=alt_seq_len).to(str(getattr(args, "hybrid_device", "cpu"))).eval()
                        g_alt.load_state_dict(alt_state, strict=False)
                    fast = LatentMuZeroPlanner(
                        model,
                        g_fast,
                        env_cfg,
                        policy_weight=float(getattr(args, "hybrid_policy_weight", 1.0)),
                        q_weight=float(getattr(args, "hybrid_q_weight", 1.0)),
                        search_score_bias=float(getattr(args, "hybrid_search_bias", -7.0)),
                        service_track_weight=float(getattr(args, "hybrid_service_track_weight", 1.0)),
                        service_search_weight=float(getattr(args, "hybrid_service_search_weight", 0.5)),
                        max_steps=int(getattr(args, "hybrid_max_steps", args.seq_len)),
                        direct_action_value_weight=float(getattr(args, "hybrid_direct_action_value_weight", 0.0)),
                        direct_action_value_max_steps=int(getattr(args, "hybrid_direct_action_value_max_steps", 0)),
                        use_root_seq_policy=True,
                        max_window_search_frac=float(getattr(args, "hybrid_max_window_search_frac", 0.65)),
                        min_window_search_atoms=int(getattr(args, "hybrid_min_window_search_atoms", 0)),
                        service_sort_plan=bool(getattr(args, "hybrid_service_sort_plan", False)),
                        service_sort_search_prefix=int(getattr(args, "hybrid_service_sort_search_prefix", 0)),
                        pressure_repair_threshold=float(getattr(args, "hybrid_pressure_repair_threshold", 0.0)),
                        pressure_repair_max_atoms=int(getattr(args, "hybrid_pressure_repair_max_atoms", 0)),
                        g_alt=g_alt,
                        router_active_threshold=int(getattr(args, "hybrid_router_active_threshold", 0)),
                        router_alt_search_bias=float(getattr(args, "hybrid_router_alt_search_bias", getattr(args, "hybrid_search_bias", -7.0))),
                        router_alt_max_window_search_frac=float(getattr(args, "hybrid_router_alt_max_window_search_frac", getattr(args, "hybrid_max_window_search_frac", 0.65))),
                        router_alt_service_sort_search_prefix=int(getattr(args, "hybrid_router_alt_service_sort_search_prefix", getattr(args, "hybrid_service_sort_search_prefix", 0))),
                        device=str(getattr(args, "hybrid_device", "cpu")),
                    )
                    if collect_teacher == "rootseq_repair":
                        planner = fast
                    else:
                        full = WorkConservingAsyncCoupledPlanner(
                            base,
                            per_sensor_top=int(getattr(args, "teacher_per_sensor_top", 3)),
                            include_search_candidate=True,
                        )
                        planner = HybridRiskPlanner(
                            fast,
                            full,
                            active_threshold=int(getattr(args, "hybrid_active_threshold", 55)),
                            overdue_threshold=int(getattr(args, "hybrid_overdue_threshold", 0)),
                            pressure_threshold=float(getattr(args, "hybrid_pressure_threshold", 0.0)),
                            max_full_fraction=float(getattr(args, "hybrid_max_full_fraction", 0.35)),
                        )
                    puct_stepwise_sequence = False
                else:
                    planner = WorkConservingAsyncCoupledPlanner(
                        base,
                        per_sensor_top=int(getattr(args, "teacher_per_sensor_top", 3)),
                        include_search_candidate=True,
                    )
                    puct_stepwise_sequence = False
                eng = build_env(planner, int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
                debt = 0.0
                try:
                    for _window in range(int(args.windows)):
                        if bool(eng.term_buf[0]) or len(rows) >= int(args.max_sequences):
                            break
                        collect_start = int(getattr(args, "collect_seq_start_window", 0))
                        collect_stride = max(1, int(getattr(args, "collect_seq_stride", 1)))
                        record_window = int(_window) >= collect_start and ((int(_window) - collect_start) % collect_stride == 0)
                        if record_window and cell_cap > 0 and (len(rows) - cell_start) >= cell_cap:
                            break
                        obs0 = attach_env_obs(get_obs(eng, debt), env_cfg, True, True)
                        x0 = tokenize(adapt, obs0, selected=set(), search_count=0).astype(np.float32)
                        slot0 = slot_features(obs0, 0.0, 0, 0, -1, 200.0).astype(np.float32)
                        variant = apply_sequence_collection_variant(planner, int(_window))
                        if bool(locals().get("puct_stepwise_sequence", False)):
                            plan = []
                        elif hasattr(planner, "choose"):
                            plan, _meta = planner.choose(eng, debt, obs0)
                            plan = list(plan)
                        else:
                            plan = list(planner.plan(obs0, budget_ms=200.0))
                        spent = 0.0
                        search_count = 0
                        track_count = 0
                        last = -1
                        seq_reward = 0.0
                        selected: set[int] = set()
                        pairs = []
                        slots = []
                        policy_targets = []
                        while spent < 200.0 and len(pairs) < int(args.seq_len) and not bool(eng.term_buf[0]):
                            obs_now = attach_env_obs(get_obs(eng, debt), env_cfg, True, True)
                            puct_policy_target = None
                            if bool(locals().get("puct_stepwise_sequence", False)):
                                action, puct_rows, puct_probs, _puct_q, _puct_visits = planner.root_distribution(
                                        eng,
                                        debt,
                                        selected,
                                        spent,
                                        search_count,
                                        track_count,
                                        last,
                                        200.0 - spent,
                                        target=str(getattr(args, "teacher_puct_policy_target", "visits")),
                                        temperature=float(getattr(args, "teacher_puct_policy_temperature", 1.0)),
                                    )
                                action = int(action)
                                puct_policy_target = np.zeros((MAXT + 1,), dtype=np.float32)
                                puct_policy_target[np.asarray(puct_rows, dtype=np.int64)] = np.asarray(puct_probs, dtype=np.float32)
                            elif not plan:
                                if hasattr(planner, "choose"):
                                    plan, _meta = planner.choose(eng, debt, obs_now)
                                    plan = list(plan)
                                else:
                                    plan = list(planner.plan(obs_now, budget_ms=200.0 - spent))
                                if not plan:
                                    break
                                action = int(plan.pop(0))
                            else:
                                action = int(plan.pop(0))
                            reward, dt, executed = execute_first_valid_action_joint(eng, [action], 200.0 - spent)
                            if executed is None or dt <= 0.0:
                                continue
                            slots.append(slot_features(obs_now, spent, search_count, track_count, last, 200.0).astype(np.float32))
                            pairs.append(training_action_pair(int(executed), single_sensor=single_sensor))
                            if puct_policy_target is None:
                                puct_policy_target = np.zeros((MAXT + 1,), dtype=np.float32)
                                executed_row, _executed_sensor = xs_decode_action(int(executed), MAXT)
                                if 0 <= int(executed_row) <= MAXT:
                                    puct_policy_target[int(executed_row)] = 1.0
                            policy_targets.append(puct_policy_target)
                            atoms = split_joint_action(int(executed)) if is_joint_action(int(executed)) else (int(executed),)
                            next_debt = 0.0 if any(xs_decode_action(int(a), MAXT)[0] == 0 for a in atoms) else debt + float(dt)
                            obs_after = attach_env_obs(get_obs(eng, next_debt), env_cfg, True, True)
                            seq_reward += shaped_step_reward(float(reward), float(dt), obs_now, obs_after, env_cfg, action=int(executed))
                            spent += float(dt)
                            for atom in atoms:
                                row, _sensor = xs_decode_action(int(atom), MAXT)
                                row = int(row)
                                if row == 0:
                                    search_count += 1
                                elif row > 0:
                                    track_count += 1
                                    selected.add(row)
                                last = row
                            debt = float(next_debt)
                        if pairs and record_window:
                            y = np.zeros((int(args.seq_len), 2), dtype=np.int64)
                            m = np.zeros((int(args.seq_len),), dtype=np.float32)
                            ss = np.zeros((int(args.seq_len), SLOT_DIM), dtype=np.float32)
                            pt = np.zeros((int(args.seq_len), MAXT + 1), dtype=np.float32)
                            n = min(len(pairs), int(args.seq_len))
                            y[:n] = np.stack(pairs[:n], axis=0)
                            ss[:n] = np.stack(slots[:n], axis=0)
                            pt[:n] = np.stack(policy_targets[:n], axis=0)
                            m[:n] = 1.0
                            service_final = service_vector(sample_state_metrics(eng, debt))
                            seq_reward += sequence_reward_service_adjustment(service_final, spent, args)
                            accept = True
                            max_seq_drop_pct = float(getattr(args, "collect_seq_max_drop_pct", 0.0))
                            max_seq_delay_ms = float(getattr(args, "collect_seq_max_delay_ms", 0.0))
                            min_seq_drop_pct = float(getattr(args, "collect_seq_min_drop_pct", 0.0))
                            min_seq_delay_ms = float(getattr(args, "collect_seq_min_delay_ms", 0.0))
                            min_seq_tracked_frac = float(getattr(args, "collect_seq_min_tracked_frac", 0.0))
                            min_seq_service_score = float(getattr(args, "collect_seq_min_service_score", -1.0e30))
                            min_seq_reward = float(getattr(args, "collect_seq_min_reward", -1.0e30))
                            active_n = max(1.0, float(service_final[0]) * 100.0)
                            tracked_n = float(service_final[1]) * 100.0
                            drop_pct = float(service_final[2]) * 100.0
                            delay_ms = float(service_final[3]) * 1000.0
                            if max_seq_drop_pct > 0.0 and drop_pct > max_seq_drop_pct:
                                accept = False
                            if max_seq_delay_ms > 0.0 and delay_ms > max_seq_delay_ms:
                                accept = False
                            if min_seq_drop_pct > 0.0 and drop_pct < min_seq_drop_pct:
                                accept = False
                            if min_seq_delay_ms > 0.0 and delay_ms < min_seq_delay_ms:
                                accept = False
                            if min_seq_tracked_frac > 0.0 and tracked_n / active_n < min_seq_tracked_frac:
                                accept = False
                            if min_seq_service_score > -1.0e20 and sequence_service_score(service_final) < min_seq_service_score:
                                accept = False
                            if min_seq_reward > -1.0e20 and float(seq_reward) < min_seq_reward:
                                accept = False
                            if accept:
                                rows.append(SequenceTransition(x0, slot0, ss, y, m, service_final, float(seq_reward), pt))
                        if len(rows) >= int(args.max_sequences):
                            break
                        if record_window and cell_cap > 0 and (len(rows) - cell_start) >= cell_cap:
                            break
                finally:
                    eng.close()
                print(
                    {
                        "seq_collected": len(rows),
                        "initial": int(initial),
                        "rate": float(rate),
                        "seed": int(seed),
                        "collect_seq_variant_controls": {
                            "search_biases": seq_hybrid_search_biases,
                            "max_steps": seq_hybrid_max_steps,
                            "search_fracs": seq_hybrid_search_fracs,
                        },
                    },
                    flush=True,
                )
                if len(rows) >= int(args.max_sequences):
                    return rows
    return rows


def print_collection_stats(data: list[Transition]) -> None:
    if not data:
        return
    pairs = np.stack([t.action_pair for t in data])
    rows = pairs // 2
    service = np.stack([t.service_next for t in data])
    print(
        {
            "transitions": int(len(data)),
            "search_slot_frac": float((rows == 0).mean()),
            "joint_any_search_frac": float((rows == 0).any(axis=1).mean()),
            "joint_both_search_frac": float(((rows[:, 0] == 0) & (rows[:, 1] == 0)).mean()),
            "service_active_mean": float(service[:, 0].mean() * 100.0),
            "service_tracked_mean": float(service[:, 1].mean() * 100.0),
            "service_drop_pct_mean": float(service[:, 2].mean() * 100.0),
            "service_delay_mean": float(service[:, 3].mean() * 1000.0),
        },
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-state", default=str(ROOT / "CreateValid1" / "results" / "mixed_gate_distill_180_action_attention_step40_state.pt"))
    ap.add_argument("--variant", default="two_row_action_attention_qpolicy_factored_loss")
    ap.add_argument("--out", default=str(ROOT / "CreateValid1" / "results" / "action_attention_muzero_g.pt"))
    ap.add_argument("--init-g-state", default="")
    ap.add_argument("--initials", default="60")
    ap.add_argument("--rates", default="4")
    ap.add_argument("--seeds", default="916")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--max-transitions", type=int, default=256)
    ap.add_argument("--max-transitions-per-cell", type=int, default=0)
    ap.add_argument("--max-action-value-transitions", type=int, default=0)
    ap.add_argument("--max-action-value-transitions-per-cell", type=int, default=0)
    ap.add_argument("--max-action-value-groups", type=int, default=0)
    ap.add_argument("--max-action-value-groups-per-cell", type=int, default=0)
    ap.add_argument("--max-hard-negative-groups", type=int, default=0)
    ap.add_argument("--max-hard-negative-groups-per-cell", type=int, default=0)
    ap.add_argument("--hard-negative-min-active", type=int, default=40)
    ap.add_argument("--hard-negative-max-pairs", type=int, default=5)
    ap.add_argument("--hard-negative-future-windows", type=int, default=0)
    ap.add_argument("--hard-negative-max-seconds", type=float, default=0.0)
    ap.add_argument("--hard-negative-progress-interval", type=int, default=0)
    ap.add_argument("--action-value-start-window", type=int, default=0)
    ap.add_argument("--action-value-stride", type=int, default=1)
    ap.add_argument("--action-value-candidate-topk", type=int, default=3)
    ap.add_argument("--action-value-group-max-pairs", type=int, default=16)
    ap.add_argument("--action-value-include-search", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--action-value-rollout-window", action="store_true")
    ap.add_argument("--action-value-rollout-future-windows", type=int, default=0)
    ap.add_argument("--action-value-target-mode", choices=["absolute", "delta"], default="absolute")
    ap.add_argument("--action-value-from-returns", action="store_true")
    ap.add_argument("--action-value-return-horizon", type=int, default=32)
    ap.add_argument("--action-value-return-discount", type=float, default=1.0)
    ap.add_argument("--action-value-group-rollout-window", action="store_true")
    ap.add_argument("--action-value-group-rollout-future-windows", type=int, default=0)
    ap.add_argument("--action-value-group-start-window", type=int, default=0)
    ap.add_argument("--action-value-group-stride", type=int, default=1)
    ap.add_argument("--action-value-group-max-seconds", type=float, default=0.0)
    ap.add_argument("--action-value-group-progress-interval", type=int, default=0)
    ap.add_argument("--max-sequences", type=int, default=0)
    ap.add_argument("--max-sequences-per-cell", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=40)
    ap.add_argument("--ar-history-k", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--unroll-steps", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--collect-mode", choices=["receding", "full_window"], default="receding")
    ap.add_argument(
        "--collect-teacher",
        choices=["direct", "sband_tensor_loop", "exact_env_puct", "exact_workconserving", "hybrid_fullgate", "rootseq_repair", "single_sensor_ar", "single_sensor_edf", "single_sensor_est"],
        default="direct",
    )
    ap.add_argument("--single-sensor", action="store_true")
    ap.add_argument("--collect-score-horizon-ms", type=float, default=200.0)
    ap.add_argument("--teacher-policy-weight", type=float, default=1.0)
    ap.add_argument("--teacher-q-weight", type=float, default=1.0)
    ap.add_argument("--teacher-search-bias", type=float, default=0.0)
    ap.add_argument("--teacher-search-floor", type=int, default=0)
    ap.add_argument("--teacher-search-cap-frac", type=float, default=1.0)
    ap.add_argument("--teacher-single-sensor-action-only", action="store_true")
    ap.add_argument("--teacher-per-sensor-top", type=int, default=3)
    ap.add_argument("--teacher-puct-simulations", type=int, default=4)
    ap.add_argument("--teacher-puct-expand-top-k", type=int, default=6)
    ap.add_argument("--teacher-puct-rollout-steps", type=int, default=2)
    ap.add_argument("--teacher-puct-rollout-windows", type=int, default=1)
    ap.add_argument("--teacher-puct-init-child-rollouts", type=int, default=0)
    ap.add_argument("--teacher-puct-c", type=float, default=1.25)
    ap.add_argument("--teacher-puct-discount", type=float, default=0.997)
    ap.add_argument("--teacher-puct-select", choices=["visits", "q", "prior"], default="q")
    ap.add_argument("--teacher-puct-terminal-service-weight", type=float, default=0.0)
    ap.add_argument("--teacher-puct-terminal-search-frame-weight", type=float, default=0.0)
    ap.add_argument("--teacher-puct-prior-uniform-mix", type=float, default=0.0)
    ap.add_argument("--teacher-puct-root-dirichlet-alpha", type=float, default=0.3)
    ap.add_argument("--teacher-puct-root-dirichlet-fraction", type=float, default=0.0)
    ap.add_argument("--teacher-puct-progressive-widening-c", type=float, default=2.0)
    ap.add_argument("--teacher-puct-progressive-widening-alpha", type=float, default=0.5)
    ap.add_argument("--teacher-puct-policy-target", choices=["visits", "q_softmax", "prior"], default="visits")
    ap.add_argument("--teacher-puct-policy-temperature", type=float, default=1.0)
    ap.add_argument("--collect-max-search-frac", type=float, default=0.0)
    ap.add_argument("--collect-max-both-search-transition-frac", type=float, default=0.0)
    ap.add_argument("--collect-max-drop-pct", type=float, default=0.0)
    ap.add_argument("--collect-max-delay-ms", type=float, default=0.0)
    ap.add_argument("--collect-seq-max-drop-pct", type=float, default=0.0)
    ap.add_argument("--collect-seq-max-delay-ms", type=float, default=0.0)
    ap.add_argument("--collect-seq-min-drop-pct", type=float, default=0.0)
    ap.add_argument("--collect-seq-min-delay-ms", type=float, default=0.0)
    ap.add_argument("--collect-seq-min-tracked-frac", type=float, default=0.0)
    ap.add_argument("--collect-seq-min-service-score", type=float, default=-1.0e30)
    ap.add_argument("--collect-seq-min-reward", type=float, default=-1.0e30)
    ap.add_argument("--seq-reward-drop-pct-penalty-weight", type=float, default=0.0)
    ap.add_argument("--seq-reward-drop-count-penalty-weight", type=float, default=0.0)
    ap.add_argument("--seq-reward-delay-penalty-weight", type=float, default=0.0)
    ap.add_argument("--seq-reward-underuse-penalty-weight", type=float, default=0.0)
    ap.add_argument("--seq-reward-underuse-target-frac", type=float, default=0.0)
    ap.add_argument("--filter-seq-reward-quantile", type=float, default=0.0)
    ap.add_argument("--filter-seq-reward-by-active-bin", action="store_true")
    ap.add_argument("--collect-seq-start-window", type=int, default=0)
    ap.add_argument("--collect-seq-stride", type=int, default=1)
    ap.add_argument("--collect-seq-hybrid-search-biases", default="")
    ap.add_argument("--collect-seq-hybrid-max-steps", default="")
    ap.add_argument("--collect-seq-hybrid-search-fracs", default="")
    ap.add_argument("--hybrid-g-state", default="")
    ap.add_argument("--hybrid-alt-g-state", default="")
    ap.add_argument("--hybrid-device", default="cpu")
    ap.add_argument("--hybrid-policy-weight", type=float, default=1.0)
    ap.add_argument("--hybrid-q-weight", type=float, default=1.0)
    ap.add_argument("--hybrid-search-bias", type=float, default=-7.0)
    ap.add_argument("--hybrid-service-track-weight", type=float, default=1.0)
    ap.add_argument("--hybrid-service-search-weight", type=float, default=0.5)
    ap.add_argument("--hybrid-max-steps", type=int, default=40)
    ap.add_argument("--hybrid-lookahead-width", type=int, default=0)
    ap.add_argument("--hybrid-lookahead-leaf-weight", type=float, default=0.25)
    ap.add_argument("--hybrid-service-critic-weight", type=float, default=0.0)
    ap.add_argument("--hybrid-service-critic-active-weight", type=float, default=0.25)
    ap.add_argument("--hybrid-service-critic-tracked-weight", type=float, default=1.0)
    ap.add_argument("--hybrid-service-critic-drop-weight", type=float, default=1.5)
    ap.add_argument("--hybrid-service-critic-delay-weight", type=float, default=0.3)
    ap.add_argument("--hybrid-direct-action-value-weight", type=float, default=0.0)
    ap.add_argument("--hybrid-direct-action-value-max-steps", type=int, default=0)
    ap.add_argument("--hybrid-max-window-search-frac", type=float, default=0.65)
    ap.add_argument("--hybrid-min-window-search-atoms", type=int, default=0)
    ap.add_argument("--hybrid-service-sort-plan", action="store_true")
    ap.add_argument("--hybrid-service-sort-search-prefix", type=int, default=2)
    ap.add_argument("--hybrid-active-threshold", type=int, default=55)
    ap.add_argument("--hybrid-overdue-threshold", type=int, default=0)
    ap.add_argument("--hybrid-pressure-threshold", type=float, default=0.0)
    ap.add_argument("--hybrid-max-full-fraction", type=float, default=0.35)
    ap.add_argument("--hybrid-pressure-repair-threshold", type=float, default=0.0)
    ap.add_argument("--hybrid-pressure-repair-max-atoms", type=int, default=0)
    ap.add_argument("--hybrid-router-active-threshold", type=int, default=0)
    ap.add_argument("--hybrid-router-alt-search-bias", type=float, default=-7.0)
    ap.add_argument("--hybrid-router-alt-max-window-search-frac", type=float, default=0.65)
    ap.add_argument("--hybrid-router-alt-service-sort-search-prefix", type=int, default=0)
    ap.add_argument("--latent-loss-weight", type=float, default=1.0)
    ap.add_argument("--score-loss-weight", type=float, default=0.2)
    ap.add_argument("--rd-loss-weight", type=float, default=0.1)
    ap.add_argument("--service-loss-weight", type=float, default=0.5)
    ap.add_argument("--action-service-loss-weight", type=float, default=0.0)
    ap.add_argument("--action-value-loss-weight", type=float, default=0.0)
    ap.add_argument("--action-frame-loss-weight", type=float, default=0.0)
    ap.add_argument("--counterfactual-seq-loss-weight", type=float, default=0.0)
    ap.add_argument("--counterfactual-pairwise-loss-weight", type=float, default=0.0)
    ap.add_argument("--counterfactual-best-loss-weight", type=float, default=0.0)
    ap.add_argument("--counterfactual-best-min-gap", type=float, default=0.0)
    ap.add_argument("--counterfactual-best-max-search-frac", type=float, default=1.0)
    ap.add_argument("--counterfactual-policy-tau", type=float, default=0.05)
    ap.add_argument("--counterfactual-use-policy-scores", action="store_true")
    ap.add_argument("--action-loss-weight", type=float, default=0.5)
    ap.add_argument("--pair-factored-ce-weight", type=float, default=0.0)
    ap.add_argument("--pair-factored-type-weight", type=float, default=1.0)
    ap.add_argument("--pair-factored-target-weight", type=float, default=1.0)
    ap.add_argument("--gpolicy-search-profile-loss-weight", type=float, default=0.0)
    ap.add_argument("--gpolicy-distill-loss-weight", type=float, default=0.0)
    ap.add_argument("--gpolicy-distill-tau", type=float, default=1.0)
    ap.add_argument("--gpolicy-low-active-threshold", type=float, default=35.0)
    ap.add_argument("--gpolicy-high-active-threshold", type=float, default=35.0)
    ap.add_argument("--gpolicy-low-max-search-atoms", type=float, default=0.25)
    ap.add_argument("--gpolicy-high-min-search-atoms", type=float, default=1.0)
    ap.add_argument("--gpolicy-high-until-search-atoms", type=float, default=4.0)
    ap.add_argument("--root-seq-loss-weight", type=float, default=0.0)
    ap.add_argument("--root-seq-ce-weight", type=float, default=1.0)
    ap.add_argument("--root-seq-factored-ce-weight", type=float, default=0.0)
    ap.add_argument("--root-seq-puct-soft-loss-weight", type=float, default=0.0)
    ap.add_argument("--root-seq-factored-type-weight", type=float, default=1.0)
    ap.add_argument("--root-seq-factored-target-weight", type=float, default=1.0)
    ap.add_argument("--root-seq-search-weight", type=float, default=1.0)
    ap.add_argument("--root-seq-search-frac-loss-weight", type=float, default=0.0)
    ap.add_argument("--root-seq-action-count-loss-weight", type=float, default=0.0)
    ap.add_argument("--root-seq-min-search-loss-weight", type=float, default=0.0)
    ap.add_argument("--root-seq-min-search-atoms", type=float, default=0.0)
    ap.add_argument("--root-seq-min-search-active-threshold", type=float, default=0.0)
    ap.add_argument("--root-seq-joint-mix-loss-weight", type=float, default=0.0)
    ap.add_argument("--root-seq-target-coverage-loss-weight", type=float, default=0.0)
    ap.add_argument("--root-seq-stop-loss-weight", type=float, default=0.0)
    ap.add_argument("--root-seq-service-weighting", type=float, default=0.0)
    ap.add_argument("--root-seq-quality-tracked-weight", type=float, default=1.0)
    ap.add_argument("--root-seq-quality-drop-weight", type=float, default=2.0)
    ap.add_argument("--root-seq-quality-delay-weight", type=float, default=0.5)
    ap.add_argument("--root-seq-reward-weighting", type=float, default=0.0)
    ap.add_argument("--root-seq-distill-loss-weight", type=float, default=0.0)
    ap.add_argument("--root-seq-distill-tau", type=float, default=1.0)
    ap.add_argument("--seq-terminal-service-loss-weight", type=float, default=0.0)
    ap.add_argument("--seq-terminal-active-weight", type=float, default=0.25)
    ap.add_argument("--seq-terminal-tracked-weight", type=float, default=1.0)
    ap.add_argument("--seq-terminal-drop-weight", type=float, default=4.0)
    ap.add_argument("--seq-terminal-delay-weight", type=float, default=4.0)
    ap.add_argument("--service-gate-loss-weight", type=float, default=0.0)
    ap.add_argument("--service-gate-low-threshold", type=float, default=35.0)
    ap.add_argument("--service-gate-high-threshold", type=float, default=55.0)
    ap.add_argument("--av-tracked-weight", type=float, default=1.0)
    ap.add_argument("--av-drop-weight", type=float, default=1.5)
    ap.add_argument("--av-drop-count-weight", type=float, default=0.0)
    ap.add_argument("--av-delay-weight", type=float, default=0.3)
    ap.add_argument("--env-mode", default="current")
    ap.add_argument("--track-update-reward", type=float, default=0.30)
    ap.add_argument("--track-loss-penalty", type=float, default=8.0)
    ap.add_argument("--sector-staleness-weight", type=float, default=0.0)
    ap.add_argument("--search-frame-overdue-weight", type=float, default=0.20)
    ap.add_argument("--search-frame-desired-ms", type=float, default=3000.0)
    ap.add_argument("--search-frame-deadline-ms", type=float, default=4500.0)
    ap.add_argument("--search-frame-drop-penalty", type=float, default=0.0)
    ap.add_argument("--zero-action-rewards", action="store_true")
    ap.add_argument("--target-service-weight", type=float, default=0.0)
    ap.add_argument("--target-service-horizon-ms", type=float, default=3000.0)
    ap.add_argument("--tracked-target-ms-reward-weight", type=float, default=0.0)
    ap.add_argument("--on-time-target-ms-reward-weight", type=float, default=0.0)
    ap.add_argument("--service-pressure-delta-reward-weight", type=float, default=0.0)
    ap.add_argument("--service-pressure-state-penalty-weight", type=float, default=0.0)
    ap.add_argument("--search-frame-state-penalty-weight", type=float, default=0.0)
    ap.add_argument("--search-frame-delta-reward-weight", type=float, default=0.0)
    ap.add_argument("--track-pressure-reward-weight", type=float, default=0.0)
    ap.add_argument("--bounded-service-reward-weight", type=float, default=0.0)
    ap.add_argument("--serviced-target-reward-weight", type=float, default=0.0)
    ap.add_argument("--serviced-target-update-reward-weight", type=float, default=0.0)
    ap.add_argument("--serviced-pressure-delta-reward-weight", type=float, default=0.0)
    ap.add_argument("--serviced-pressure-improvement-reward-weight", type=float, default=0.0)
    ap.add_argument("--discovered-target-reward", type=float, default=0.0)
    ap.add_argument("--tracked-count-delta-reward-weight", type=float, default=0.0)
    ap.add_argument("--root-seq-dynamic-mask", action="store_true")
    ap.add_argument("--root-seq-step-context", action="store_true")
    ap.add_argument("--use-ar-seq-loss", action="store_true")
    ap.add_argument("--freeze-non-seq", action="store_true")
    ap.add_argument("--train-stop-only", action="store_true")
    ap.add_argument("--train-service-gate-only", action="store_true")
    ap.add_argument("--train-dynamics-only", action="store_true")
    ap.add_argument("--train-gpolicy-only", action="store_true")
    ap.add_argument("--train-light-mixer-only", action="store_true")
    ap.add_argument("--train-tiny-mixer-only", action="store_true")
    ap.add_argument("--train-action-service-only", action="store_true")
    ap.add_argument("--train-action-value-only", action="store_true")
    ap.add_argument("--train-action-frame-only", action="store_true")
    ap.add_argument("--train-counterfactual-seq-only", action="store_true")
    ap.add_argument("--train-terminal-service-only", action="store_true")
    ap.add_argument("--disable-policy-action-coupler", action="store_true")
    ap.add_argument("--policy-action-mixer", choices=["full", "light", "tiny", "none"], default="full")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-seed", type=int, default=123)
    ap.add_argument("--g-d-model", type=int, default=48)
    args = ap.parse_args()

    exact_args = make_exact_args(args)
    exact_args.enable_x_band = not bool(args.single_sensor)
    exact_args.single_sensor = bool(args.single_sensor)
    model_args = SimpleNamespace(
        targets=str(ROOT / "CreateValid1" / "results" / "edf_bootstrap_r3_lmh_1024_targets.pt"),
        model_seed=int(args.model_seed),
        q_loss_weight=0.25,
        value_loss_weight=0.25,
        use_arrival_token_feature=False,
        search_calibration_weight=0.0,
        non_strict_load=False,
    )
    base_checkpoint = torch.load(str(args.base_state), map_location="cpu", weights_only=False)
    if isinstance(base_checkpoint, dict) and "model_state_dict" in base_checkpoint:
        state = base_checkpoint["model_state_dict"]
        d_model = int(state["cls_token"].shape[0]) if "cls_token" in state else 48
        model = make_physical_model(
            str(args.variant),
            SimpleNamespace(d_model=d_model, nhead=4, nlayers=2),
        ).to(args.device).eval()
        model.load_state_dict(state, strict=True)
    else:
        model = train_variant(model_args, str(args.variant), str(args.base_state)).to(args.device).eval()
    data = collect(args, model, exact_args) if int(args.max_transitions) > 0 else []
    print_collection_stats(data)
    av_data = collect_action_value_counterfactuals(args, model, exact_args) if int(args.max_action_value_transitions) > 0 else []
    if bool(args.action_value_from_returns):
        av_data.extend(build_return_action_value_data(data, args))
    if av_data:
        vals = np.asarray([t.value for t in av_data], dtype=np.float32)
        args.action_value_target_mean = float(vals.mean())
        args.action_value_target_std = float(max(float(vals.std()), 1.0e-3))
        svc = np.stack([t.service_next for t in av_data])
        frame = np.asarray([t.frame_next for t in av_data], dtype=np.float32)
        pairs = np.stack([t.action_pair for t in av_data])
        print(
            {
                "action_value_transitions": int(len(av_data)),
                "action_value_mean": float(vals.mean()),
                "action_value_std": float(vals.std()),
                "action_value_return_horizon": int(args.action_value_return_horizon) if bool(args.action_value_from_returns) else 0,
                "action_value_return_discount": float(args.action_value_return_discount) if bool(args.action_value_from_returns) else 0.0,
                "action_value_search_slot_frac": float(((pairs // 2) == 0).mean()),
                "action_value_drop_pct_mean": float(svc[:, 2].mean() * 100.0),
                "action_value_delay_mean": float(svc[:, 3].mean() * 1000.0),
                "action_frame_pressure_mean": float(frame.mean()),
                "action_frame_pressure_std": float(frame.std()),
            },
            flush=True,
        )
    av_groups = collect_action_value_groups(args, model, exact_args) if int(args.max_action_value_groups) > 0 else []
    hard_groups = collect_hard_negative_groups(args, model, exact_args) if int(args.max_hard_negative_groups) > 0 else []
    if hard_groups:
        av_groups.extend(hard_groups)
        vals = np.concatenate([g.values[g.mask > 0.5] for g in hard_groups])
        masks = np.stack([g.mask for g in hard_groups])
        pairs = np.concatenate([g.action_pairs[g.mask > 0.5] for g in hard_groups], axis=0)
        print(
            {
                "hard_negative_groups": int(len(hard_groups)),
                "hard_negative_mean_pairs": float(masks.sum(axis=1).mean()),
                "hard_negative_value_mean": float(vals.mean()),
                "hard_negative_value_std": float(vals.std()),
                "hard_negative_search_slot_frac": float(((pairs // 2) == 0).mean()),
            },
            flush=True,
        )
    if av_groups:
        vals = np.concatenate([g.values[g.mask > 0.5] for g in av_groups])
        masks = np.stack([g.mask for g in av_groups])
        pairs = np.concatenate([g.action_pairs[g.mask > 0.5] for g in av_groups], axis=0)
        group_vals = np.stack([g.values for g in av_groups]).astype(np.float32)
        group_masks = np.stack([g.mask for g in av_groups]).astype(np.float32)
        masked_vals = np.where(group_masks > 0.5, group_vals, -1.0e9)
        sorted_vals = np.sort(masked_vals, axis=1)[:, ::-1]
        best_gap = sorted_vals[:, 0] - sorted_vals[:, 1]
        group_pairs = np.stack([g.action_pairs for g in av_groups])
        best_idx = np.argmax(masked_vals, axis=1)
        best_pairs = group_pairs[np.arange(group_pairs.shape[0]), best_idx]
        best_rows = best_pairs // 2
        best_search_frac = (best_rows <= 0).mean(axis=1)
        print(
            {
                "action_value_groups": int(len(av_groups)),
                "action_value_group_mean_pairs": float(masks.sum(axis=1).mean()),
                "action_value_group_value_mean": float(vals.mean()),
                "action_value_group_value_std": float(vals.std()),
                "action_value_group_search_slot_frac": float(((pairs // 2) == 0).mean()),
                "action_value_group_best_gap_p50": float(np.percentile(best_gap, 50)),
                "action_value_group_best_gap_p90": float(np.percentile(best_gap, 90)),
                "action_value_group_best_search_frac_mean": float(best_search_frac.mean()),
            },
            flush=True,
        )
    need_seq_data = (
        float(args.root_seq_loss_weight) > 0.0
        or float(args.service_gate_loss_weight) > 0.0
        or float(args.seq_terminal_service_loss_weight) > 0.0
    ) and int(args.max_sequences) > 0
    seq_data = collect_sequences(args, model, exact_args) if need_seq_data else []
    seq_data = filter_sequence_data_by_reward(seq_data, args)
    if seq_data:
        mask = np.stack([s.mask for s in seq_data])
        pairs = np.stack([s.action_pairs for s in seq_data])
        seq_rows = pairs // 2
        valid_pairs = (pairs >= 0) & (mask[:, :, None] > 0)
        valid_pair_count = float(valid_pairs.sum())
        service = np.stack([s.service_final for s in seq_data])
        seq_rewards = np.asarray([float(s.reward) for s in seq_data], dtype=np.float32)
        print(
            {
                "seq_sequences": int(len(seq_data)),
                "seq_mean_len": float(mask.sum(axis=1).mean()),
                "seq_search_slot_frac": float(((seq_rows == 0) & valid_pairs).sum() / max(1.0, valid_pair_count)),
                "seq_service_active_mean": float(service[:, 0].mean() * 100.0),
                "seq_service_tracked_mean": float(service[:, 1].mean() * 100.0),
                "seq_service_drop_pct_mean": float(service[:, 2].mean() * 100.0),
                "seq_service_drop_pct_max": float(service[:, 2].max() * 100.0),
                "seq_service_delay_mean": float(service[:, 3].mean() * 1000.0),
                "seq_service_delay_std": float(service[:, 3].std() * 1000.0),
                "seq_service_delay_max": float(service[:, 3].max() * 1000.0),
                "seq_reward_mean": float(seq_rewards.mean()),
                "seq_reward_std": float(seq_rewards.std()),
                "seq_reward_min": float(seq_rewards.min()),
                "seq_reward_max": float(seq_rewards.max()),
            },
            flush=True,
        )
    if not data and not seq_data and not av_data and not av_groups:
        raise RuntimeError("no transitions collected")
    g = LatentG(d_model=int(args.g_d_model), seq_len=int(args.seq_len), ar_history_k=int(args.ar_history_k)).to(args.device)
    g.policy_action_mixer = "none" if bool(args.disable_policy_action_coupler) else str(args.policy_action_mixer)
    if str(args.init_g_state):
        ckpt = torch.load(str(args.init_g_state), map_location=args.device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        missing, unexpected = g.load_state_dict(state, strict=False)
        print({"init_g_state": str(args.init_g_state), "missing": missing, "unexpected": unexpected}, flush=True)
    teacher_g = None
    if float(args.gpolicy_distill_loss_weight) > 0.0:
        teacher_g = LatentG(d_model=int(args.g_d_model), seq_len=int(args.seq_len), ar_history_k=int(args.ar_history_k)).to(args.device).eval()
        teacher_g.load_state_dict(g.state_dict(), strict=False)
        teacher_g.policy_action_mixer = "full"
        for p in teacher_g.parameters():
            p.requires_grad_(False)
        print({"gpolicy_distill_teacher": "full"}, flush=True)
    root_seq_teacher_g = None
    if float(args.root_seq_distill_loss_weight) > 0.0:
        root_seq_teacher_g = LatentG(d_model=int(args.g_d_model), seq_len=int(args.seq_len), ar_history_k=int(args.ar_history_k)).to(args.device).eval()
        root_seq_teacher_g.load_state_dict(g.state_dict(), strict=False)
        root_seq_teacher_g.policy_action_mixer = str(getattr(g, "policy_action_mixer", "full"))
        for p in root_seq_teacher_g.parameters():
            p.requires_grad_(False)
        print({"root_seq_distill_teacher": "init_state"}, flush=True)
    if bool(args.disable_policy_action_coupler) and hasattr(g, "policy_action_residual"):
        for p in g.policy_action_proj.parameters():
            p.requires_grad_(False)
        for p in g.policy_action_coupler.parameters():
            p.requires_grad_(False)
        for p in g.policy_action_residual.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            g.policy_action_residual.weight.zero_()
            g.policy_action_residual.bias.zero_()
    if bool(args.freeze_non_seq):
        for name, param in g.named_parameters():
            trainable = name.startswith("seq_")
            if bool(args.use_ar_seq_loss):
                trainable = trainable or name.startswith("ar_")
            param.requires_grad_(bool(trainable))
        print({"freeze_non_seq": True, "trainable": [name for name, p in g.named_parameters() if p.requires_grad]}, flush=True)
    if bool(args.train_stop_only):
        for name, param in g.named_parameters():
            param.requires_grad_(name.startswith("seq_stop_policy"))
        trainable = [name for name, p in g.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("train-stop-only selected but no seq_stop_policy parameters are trainable")
        print({"train_stop_only": True, "trainable": trainable}, flush=True)
    if bool(args.train_service_gate_only):
        for name, param in g.named_parameters():
            param.requires_grad_(name.startswith("service_gate"))
        trainable = [name for name, p in g.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("train-service-gate-only selected but no service_gate parameters are trainable")
        print({"train_service_gate_only": True, "trainable": trainable}, flush=True)
    if bool(args.train_action_service_only):
        for name, param in g.named_parameters():
            param.requires_grad_(name.startswith("action_service_head"))
        trainable = [name for name, p in g.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("train-action-service-only selected but no action_service_head parameters are trainable")
        print({"train_action_service_only": True, "trainable": trainable}, flush=True)
    if bool(args.train_action_value_only):
        for name, param in g.named_parameters():
            param.requires_grad_(name.startswith("action_value_head"))
        trainable = [name for name, p in g.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("train-action-value-only selected but no action_value_head parameters are trainable")
        print({"train_action_value_only": True, "trainable": trainable}, flush=True)
    if bool(args.train_action_frame_only):
        for name, param in g.named_parameters():
            param.requires_grad_(name.startswith("action_frame_head"))
        trainable = [name for name, p in g.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("train-action-frame-only selected but no action_frame_head parameters are trainable")
        print({"train_action_frame_only": True, "trainable": trainable}, flush=True)
    if bool(args.train_counterfactual_seq_only):
        for name, param in g.named_parameters():
            param.requires_grad_(name.startswith("seq_"))
        trainable = [name for name, p in g.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("train-counterfactual-seq-only selected but no seq parameters are trainable")
        print({"train_counterfactual_seq_only": True, "trainable": trainable}, flush=True)
    if bool(args.train_terminal_service_only):
        terminal_service_prefixes = ("input_proj", "cls_update", "tok_update", "service_head")
        for name, param in g.named_parameters():
            param.requires_grad_(name.startswith(terminal_service_prefixes))
        trainable = [name for name, p in g.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("train-terminal-service-only selected but no terminal service parameters are trainable")
        print({"train_terminal_service_only": True, "trainable": trainable}, flush=True)
    if bool(args.train_dynamics_only):
        dynamics_prefixes = ("input_proj", "cls_update", "tok_update", "reward_dt", "service_head", "action_service_head", "action_frame_head")
        for name, param in g.named_parameters():
            param.requires_grad_(name.startswith(dynamics_prefixes))
        trainable = [name for name, p in g.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("train-dynamics-only selected but no dynamics parameters are trainable")
        print({"train_dynamics_only": True, "trainable": trainable}, flush=True)
    if bool(args.train_gpolicy_only):
        if bool(args.disable_policy_action_coupler):
            gpolicy_prefixes = ("slot_proj", "type_policy", "target_policy")
        elif str(args.policy_action_mixer) == "light":
            gpolicy_prefixes = (
                "slot_proj",
                "type_policy",
                "target_policy",
                "policy_action_proj",
                "policy_light_residual",
            )
        elif str(args.policy_action_mixer) == "tiny":
            gpolicy_prefixes = (
                "slot_proj",
                "type_policy",
                "target_policy",
                "policy_action_proj",
                "policy_tiny_coupler",
                "policy_tiny_residual",
            )
        else:
            gpolicy_prefixes = (
                "slot_proj",
                "type_policy",
                "target_policy",
                "policy_action_proj",
                "policy_action_coupler",
                "policy_action_residual",
            )
        for name, param in g.named_parameters():
            param.requires_grad_(name.startswith(gpolicy_prefixes))
        trainable = [name for name, p in g.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("train-gpolicy-only selected but no gpolicy parameters are trainable")
        print({"train_gpolicy_only": True, "trainable": trainable}, flush=True)
    if bool(args.train_light_mixer_only):
        for name, param in g.named_parameters():
            param.requires_grad_(name.startswith(("policy_action_proj", "policy_light_residual")))
        trainable = [name for name, p in g.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("train-light-mixer-only selected but no light mixer parameters are trainable")
        print({"train_light_mixer_only": True, "trainable": trainable}, flush=True)
    if bool(args.train_tiny_mixer_only):
        for name, param in g.named_parameters():
            param.requires_grad_(name.startswith(("policy_action_proj", "policy_tiny_coupler", "policy_tiny_residual")))
        trainable = [name for name, p in g.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("train-tiny-mixer-only selected but no tiny mixer parameters are trainable")
        print({"train_tiny_mixer_only": True, "trainable": trainable}, flush=True)
    opt = torch.optim.AdamW([p for p in g.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=1e-4)
    rng = np.random.default_rng(123)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for step in range(int(args.steps)):
        if data:
            unroll = max(1, int(args.unroll_steps))
            max_start = max(1, len(data) - unroll)
            idx = rng.integers(0, max_start, size=min(int(args.batch_size), max_start))
            x = torch.from_numpy(np.stack([data[i].x for i in idx])).float().to(args.device)
            slot = torch.from_numpy(np.stack([data[i].slot for i in idx])).float().to(args.device)
            with torch.no_grad():
                cls, tok, sel, active = model.backbone.encode_tokens(x)
        else:
            unroll = 1
            idx = np.asarray([], dtype=np.int64)
            slot = torch.empty((0, SLOT_DIM), dtype=torch.float32, device=args.device)
        latent_loss = slot.new_tensor(0.0)
        score_loss = slot.new_tensor(0.0)
        rd_loss = slot.new_tensor(0.0)
        service_loss = slot.new_tensor(0.0)
        action_service_loss = slot.new_tensor(0.0)
        action_value_loss = slot.new_tensor(0.0)
        action_frame_loss = slot.new_tensor(0.0)
        action_loss = slot.new_tensor(0.0)
        gpolicy_search_profile_loss = slot.new_tensor(0.0)
        gpolicy_distill_loss = slot.new_tensor(0.0)
        service_gate_loss = slot.new_tensor(0.0)
        root_seq_loss = slot.new_tensor(0.0)
        root_seq_ce_loss = slot.new_tensor(0.0)
        root_seq_factored_ce_loss = slot.new_tensor(0.0)
        root_seq_puct_soft_loss = slot.new_tensor(0.0)
        root_seq_frac_loss = slot.new_tensor(0.0)
        root_seq_count_loss = slot.new_tensor(0.0)
        root_seq_min_search_loss = slot.new_tensor(0.0)
        root_seq_joint_loss = slot.new_tensor(0.0)
        root_seq_coverage_loss = slot.new_tensor(0.0)
        root_seq_stop_loss = slot.new_tensor(0.0)
        root_seq_distill_loss = slot.new_tensor(0.0)
        root_seq_s_search_frac = slot.new_tensor(0.0)
        root_seq_search_margin = slot.new_tensor(0.0)
        root_seq_track_margin = slot.new_tensor(0.0)
        seq_terminal_service_loss = slot.new_tensor(0.0)
        counterfactual_seq_loss = slot.new_tensor(0.0)
        counterfactual_pairwise_loss = slot.new_tensor(0.0)
        counterfactual_best_loss = slot.new_tensor(0.0)
        action_terms = 0
        if data:
            root_act = torch.from_numpy(np.stack([data[i].action_pair for i in idx])).long().to(args.device)
            root_scores = g.policy_scores(cls, tok, slot, sel, active)
            root_loss, root_terms = masked_pair_ce(root_scores, root_act)
            action_loss = action_loss + root_loss
            action_terms += root_terms
            if float(args.pair_factored_ce_weight) > 0.0:
                root_factored_loss, root_factored_terms = masked_pair_factored_ce(
                    root_scores,
                    root_act,
                    type_weight=float(args.pair_factored_type_weight),
                    target_weight=float(args.pair_factored_target_weight),
                )
                action_loss = action_loss + float(args.pair_factored_ce_weight) * root_factored_loss
                action_terms += root_factored_terms
            if teacher_g is not None and float(args.gpolicy_distill_loss_weight) > 0.0:
                with torch.no_grad():
                    teacher_scores = teacher_g.policy_scores(cls, tok, slot, sel, active)
                tau = max(1e-4, float(args.gpolicy_distill_tau))
                distill_terms = []
                for sensor_i in (0, 1):
                    t_logits = teacher_scores[:, :, sensor_i]
                    s_logits = root_scores[:, :, sensor_i]
                    valid = torch.isfinite(t_logits) & (t_logits > -1e8)
                    if bool(valid.any()):
                        t_safe = (t_logits / tau).masked_fill(~valid, -1e9)
                        s_safe = (s_logits / tau).masked_fill(~valid, -1e9)
                        target_probs = torch.softmax(t_safe, dim=1)
                        log_probs = torch.log_softmax(s_safe, dim=1)
                        per = -(target_probs * log_probs).sum(dim=1)
                        active_rows = valid.any(dim=1)
                        if bool(active_rows.any()):
                            distill_terms.append(per[active_rows].mean() * (tau * tau))
                if distill_terms:
                    gpolicy_distill_loss = sum(distill_terms) / float(len(distill_terms))
            if float(args.gpolicy_search_profile_loss_weight) > 0.0:
                gpolicy_search_profile_loss = gpolicy_search_profile_loss + policy_search_profile_loss(
                    root_scores,
                    slot,
                    low_active_threshold=float(args.gpolicy_low_active_threshold),
                    high_active_threshold=float(args.gpolicy_high_active_threshold),
                    low_max_search_atoms=float(args.gpolicy_low_max_search_atoms),
                    high_min_search_atoms=float(args.gpolicy_high_min_search_atoms),
                    high_until_search_atoms=float(args.gpolicy_high_until_search_atoms),
                )
            if float(args.action_service_loss_weight) > 0.0:
                root_service = torch.from_numpy(np.stack([data[i].service_next for i in idx])).float().to(args.device)
                action_service_loss = action_service_loss + F.smooth_l1_loss(g.predict_action_service(cls, tok, slot, root_act), root_service)
        if seq_data:
            sidx = rng.integers(0, len(seq_data), size=min(int(args.batch_size), len(seq_data)))
            sx = torch.from_numpy(np.stack([seq_data[i].x for i in sidx])).float().to(args.device)
            sslot = torch.from_numpy(np.stack([seq_data[i].slot for i in sidx])).float().to(args.device)
            sslots = torch.from_numpy(np.stack([seq_data[i].slots_seq for i in sidx])).float().to(args.device)
            spair = torch.from_numpy(np.stack([seq_data[i].action_pairs for i in sidx])).long().to(args.device)
            smask = torch.from_numpy(np.stack([seq_data[i].mask for i in sidx])).float().to(args.device)
            spolicy = None
            if all(seq_data[i].policy_targets is not None for i in sidx):
                spolicy = torch.from_numpy(np.stack([seq_data[i].policy_targets for i in sidx])).float().to(args.device)
            sservice = torch.from_numpy(np.stack([seq_data[i].service_final for i in sidx])).float().to(args.device)
            sreward = torch.tensor([float(seq_data[i].reward) for i in sidx], dtype=torch.float32, device=args.device)
            with torch.no_grad():
                scls, stok, ssel, sactive = model.backbone.encode_tokens(sx)
            if bool(args.use_ar_seq_loss):
                seq_scores = g.ar_sequence_scores(scls, stok, sslot, ssel, sactive, action_pairs=spair, seq_slots=sslots)
            else:
                seq_pairs_for_mask = spair if bool(args.root_seq_dynamic_mask) else None
                seq_scores = g.sequence_scores(
                    scls,
                    stok,
                    sslot,
                    ssel,
                    sactive,
                    action_pairs=seq_pairs_for_mask,
                    seq_slots=sslots,
                    use_step_context=bool(args.root_seq_step_context),
                )
            seq_weight = None
            if float(args.root_seq_service_weighting) > 0.0:
                quality = (
                    float(args.root_seq_quality_tracked_weight) * sservice[:, 1]
                    - float(args.root_seq_quality_drop_weight) * sservice[:, 2]
                    - float(args.root_seq_quality_delay_weight) * sservice[:, 3]
                )
                quality = (quality - quality.mean()) / quality.std(unbiased=False).clamp_min(1e-3)
                seq_weight = torch.exp(float(args.root_seq_service_weighting) * quality).clamp(0.25, 4.0)
            if float(args.root_seq_reward_weighting) > 0.0:
                reward_quality = (sreward - sreward.mean()) / sreward.std(unbiased=False).clamp_min(1e-3)
                reward_weight = torch.exp(float(args.root_seq_reward_weighting) * reward_quality).clamp(0.25, 4.0)
                seq_weight = reward_weight if seq_weight is None else (seq_weight * reward_weight).clamp(0.25, 4.0)
            root_seq_ce_loss, _seq_terms = masked_sequence_pair_ce(
                seq_scores,
                spair,
                smask,
                search_weight=float(args.root_seq_search_weight),
                seq_weight=seq_weight,
            )
            root_seq_loss = float(args.root_seq_ce_weight) * root_seq_ce_loss
            seq_type_logits = None
            seq_target_logits = None
            if float(args.root_seq_factored_ce_weight) > 0.0 or float(args.root_seq_puct_soft_loss_weight) > 0.0:
                if bool(args.use_ar_seq_loss) and hasattr(g, "ar_sequence_factor_logits"):
                    seq_type_logits, seq_target_logits = g.ar_sequence_factor_logits(
                        scls,
                        stok,
                        sslot,
                        ssel,
                        sactive,
                        action_pairs=spair,
                        seq_slots=sslots,
                    )
                    seq_rows_dbg = pair_rows(spair).clamp(min=0, max=MAXT)
                    seq_valid_dbg = smask.bool() & (spair[:, :, 0] >= 0)
                    seq_search_dbg = seq_valid_dbg & (seq_rows_dbg[:, :, 0] == 0)
                    seq_track_dbg = seq_valid_dbg & (seq_rows_dbg[:, :, 0] > 0)
                    if bool(seq_valid_dbg.any()):
                        root_seq_s_search_frac = seq_search_dbg.float().sum() / seq_valid_dbg.float().sum().clamp_min(1.0)
                    type_margin_dbg = seq_type_logits[:, :, 0, 0] - seq_type_logits[:, :, 0, 1]
                    if bool(seq_search_dbg.any()):
                        root_seq_search_margin = type_margin_dbg[seq_search_dbg].mean()
                    if bool(seq_track_dbg.any()):
                        root_seq_track_margin = type_margin_dbg[seq_track_dbg].mean()
                    root_seq_factored_ce_loss, _seq_factored_terms = masked_sequence_explicit_factor_ce(
                        seq_type_logits,
                        seq_target_logits,
                        spair,
                        smask,
                        type_weight=float(args.root_seq_factored_type_weight),
                        target_weight=float(args.root_seq_factored_target_weight),
                        search_weight=float(args.root_seq_search_weight),
                        seq_weight=seq_weight,
                    )
                elif hasattr(g, "sequence_factor_logits"):
                    seq_type_logits, seq_target_logits = g.sequence_factor_logits(
                        scls,
                        stok,
                        sslot,
                        ssel,
                        sactive,
                        seq_slots=sslots,
                        use_step_context=bool(args.root_seq_step_context),
                    )
                    seq_rows_dbg = pair_rows(spair).clamp(min=0, max=MAXT)
                    seq_valid_dbg = smask.bool() & (spair[:, :, 0] >= 0)
                    seq_search_dbg = seq_valid_dbg & (seq_rows_dbg[:, :, 0] == 0)
                    seq_track_dbg = seq_valid_dbg & (seq_rows_dbg[:, :, 0] > 0)
                    if bool(seq_valid_dbg.any()):
                        root_seq_s_search_frac = seq_search_dbg.float().sum() / seq_valid_dbg.float().sum().clamp_min(1.0)
                    type_margin_dbg = seq_type_logits[:, :, 0, 0] - seq_type_logits[:, :, 0, 1]
                    if bool(seq_search_dbg.any()):
                        root_seq_search_margin = type_margin_dbg[seq_search_dbg].mean()
                    if bool(seq_track_dbg.any()):
                        root_seq_track_margin = type_margin_dbg[seq_track_dbg].mean()
                    root_seq_factored_ce_loss, _seq_factored_terms = masked_sequence_explicit_factor_ce(
                        seq_type_logits,
                        seq_target_logits,
                        spair,
                        smask,
                        type_weight=float(args.root_seq_factored_type_weight),
                        target_weight=float(args.root_seq_factored_target_weight),
                        search_weight=float(args.root_seq_search_weight),
                        seq_weight=seq_weight,
                    )
                else:
                    root_seq_factored_ce_loss, _seq_factored_terms = masked_sequence_factored_ce(
                        seq_scores,
                        spair,
                        smask,
                        type_weight=float(args.root_seq_factored_type_weight),
                        target_weight=float(args.root_seq_factored_target_weight),
                        seq_weight=seq_weight,
                    )
                root_seq_loss = root_seq_loss + float(args.root_seq_factored_ce_weight) * root_seq_factored_ce_loss
                if (
                    spolicy is not None
                    and seq_type_logits is not None
                    and seq_target_logits is not None
                    and float(args.root_seq_puct_soft_loss_weight) > 0.0
                ):
                    root_seq_puct_soft_loss = masked_sequence_sonly_soft_puct_loss(
                        seq_type_logits,
                        seq_target_logits,
                        spolicy,
                        smask,
                    )
            if root_seq_teacher_g is not None and float(args.root_seq_distill_loss_weight) > 0.0:
                with torch.no_grad():
                    if bool(args.use_ar_seq_loss):
                        teacher_seq_scores = root_seq_teacher_g.ar_sequence_scores(
                            scls,
                            stok,
                            sslot,
                            ssel,
                            sactive,
                            action_pairs=spair,
                            seq_slots=sslots,
                        )
                    else:
                        teacher_seq_scores = root_seq_teacher_g.sequence_scores(
                            scls,
                            stok,
                            sslot,
                            ssel,
                            sactive,
                            action_pairs=seq_pairs_for_mask,
                            seq_slots=sslots,
                            use_step_context=bool(args.root_seq_step_context),
                        )
                root_seq_distill_loss = masked_sequence_logits_kl(
                    seq_scores,
                    teacher_seq_scores,
                    smask,
                    tau=float(args.root_seq_distill_tau),
                )
                root_seq_loss = root_seq_loss + float(args.root_seq_distill_loss_weight) * root_seq_distill_loss
            if float(args.root_seq_search_frac_loss_weight) > 0.0:
                root_seq_frac_loss = sequence_search_fraction_loss(seq_scores, spair, smask)
                root_seq_loss = root_seq_loss + float(args.root_seq_search_frac_loss_weight) * root_seq_frac_loss
            if float(args.root_seq_action_count_loss_weight) > 0.0:
                root_seq_count_loss = sequence_action_count_loss(seq_scores, spair, smask)
                root_seq_loss = root_seq_loss + float(args.root_seq_action_count_loss_weight) * root_seq_count_loss
            if float(args.root_seq_min_search_loss_weight) > 0.0:
                root_seq_min_search_loss = sequence_min_search_atoms_loss(
                    seq_scores,
                    smask,
                    min_atoms=float(args.root_seq_min_search_atoms),
                    active=sslot[:, 4] * 100.0,
                    active_threshold=float(args.root_seq_min_search_active_threshold),
                )
                root_seq_loss = root_seq_loss + float(args.root_seq_min_search_loss_weight) * root_seq_min_search_loss
            if float(args.root_seq_joint_mix_loss_weight) > 0.0:
                root_seq_joint_loss = sequence_joint_mix_loss(seq_scores, spair, smask)
                root_seq_loss = root_seq_loss + float(args.root_seq_joint_mix_loss_weight) * root_seq_joint_loss
            if float(args.root_seq_target_coverage_loss_weight) > 0.0:
                root_seq_coverage_loss = sequence_target_coverage_loss(seq_scores, spair, smask)
                root_seq_loss = root_seq_loss + float(args.root_seq_target_coverage_loss_weight) * root_seq_coverage_loss
            if float(args.root_seq_stop_loss_weight) > 0.0:
                stop_scores = g.sequence_stop_scores(scls, stok, sslot, seq_slots=sslots)
                root_seq_stop_loss = sequence_stop_loss(stop_scores, smask)
                root_seq_loss = root_seq_loss + float(args.root_seq_stop_loss_weight) * root_seq_stop_loss
            if float(args.service_gate_loss_weight) > 0.0:
                active_count = sslot[:, 4] * 100.0
                low = active_count < float(args.service_gate_low_threshold)
                high = active_count >= float(args.service_gate_high_threshold)
                target = (low | high).to(sslot.dtype)
                service_gate_loss = F.binary_cross_entropy_with_logits(g.service_gate_logit(sslot), target)
                synth_active = torch.linspace(0.0, 100.0, steps=101, device=args.device)
                synth_slot = torch.zeros((101, SLOT_DIM), dtype=sslot.dtype, device=args.device)
                synth_slot[:, 4] = synth_active / 100.0
                synth_slot[:, 5] = synth_active / 100.0
                synth_target = ((synth_active < float(args.service_gate_low_threshold)) | (synth_active >= float(args.service_gate_high_threshold))).to(sslot.dtype)
                service_gate_loss = service_gate_loss + F.binary_cross_entropy_with_logits(g.service_gate_logit(synth_slot), synth_target)
            if float(args.seq_terminal_service_loss_weight) > 0.0:
                term_cls, term_tok = scls, stok
                for seq_step in range(int(g.seq_len)):
                    step_mask = smask[:, seq_step].view(-1, 1)
                    if float(step_mask.sum().detach().cpu()) <= 0.0:
                        continue
                    next_cls, next_tok, _r_pred, _dt_pred = g(term_cls, term_tok, sslots[:, seq_step, :], spair[:, seq_step, :])
                    term_cls = torch.where(step_mask > 0.0, next_cls, term_cls)
                    term_tok = torch.where(step_mask[:, None, :] > 0.0, next_tok, term_tok)
                pred_service = g.predict_service(term_cls, term_tok)
                weights = torch.tensor(
                    [
                        float(args.seq_terminal_active_weight),
                        float(args.seq_terminal_tracked_weight),
                        float(args.seq_terminal_drop_weight),
                        float(args.seq_terminal_delay_weight),
                    ],
                    dtype=pred_service.dtype,
                    device=pred_service.device,
                )
                seq_terminal_service_loss = (F.smooth_l1_loss(pred_service, sservice, reduction="none") * weights[None, :]).mean()
        if av_data and float(args.action_value_loss_weight) > 0.0:
            aidx = rng.integers(0, len(av_data), size=min(int(args.batch_size), len(av_data)))
            ax = torch.from_numpy(np.stack([av_data[i].x for i in aidx])).float().to(args.device)
            aslot = torch.from_numpy(np.stack([av_data[i].slot for i in aidx])).float().to(args.device)
            aact = torch.from_numpy(np.stack([av_data[i].action_pair for i in aidx])).long().to(args.device)
            aval = torch.tensor([av_data[i].value for i in aidx], dtype=torch.float32, device=args.device)
            with torch.no_grad():
                acls, atok, _asel, _aactive = model.backbone.encode_tokens(ax)
            pred = g.predict_action_value(acls, atok, aslot, aact)
            mean = float(getattr(args, "action_value_target_mean", 0.0))
            std = max(1.0e-3, float(getattr(args, "action_value_target_std", 1.0)))
            target = (aval - mean) / std
            action_value_loss = F.smooth_l1_loss(pred, target)
        if av_data and float(args.action_service_loss_weight) > 0.0:
            aidx = rng.integers(0, len(av_data), size=min(int(args.batch_size), len(av_data)))
            ax = torch.from_numpy(np.stack([av_data[i].x for i in aidx])).float().to(args.device)
            aslot = torch.from_numpy(np.stack([av_data[i].slot for i in aidx])).float().to(args.device)
            aact = torch.from_numpy(np.stack([av_data[i].action_pair for i in aidx])).long().to(args.device)
            aservice = torch.from_numpy(np.stack([av_data[i].service_next for i in aidx])).float().to(args.device)
            with torch.no_grad():
                acls, atok, _asel, _aactive = model.backbone.encode_tokens(ax)
            action_service_loss = action_service_loss + F.smooth_l1_loss(
                g.predict_action_service(acls, atok, aslot, aact),
                aservice,
            )
        if av_data and float(args.action_frame_loss_weight) > 0.0:
            aidx = rng.integers(0, len(av_data), size=min(int(args.batch_size), len(av_data)))
            ax = torch.from_numpy(np.stack([av_data[i].x for i in aidx])).float().to(args.device)
            aslot = torch.from_numpy(np.stack([av_data[i].slot for i in aidx])).float().to(args.device)
            aact = torch.from_numpy(np.stack([av_data[i].action_pair for i in aidx])).long().to(args.device)
            aframe = torch.tensor([av_data[i].frame_next for i in aidx], dtype=torch.float32, device=args.device)
            with torch.no_grad():
                acls, atok, _asel, _aactive = model.backbone.encode_tokens(ax)
            action_frame_loss = F.smooth_l1_loss(g.predict_action_frame(acls, atok, aslot, aact), aframe)
        if av_groups and (
            float(args.counterfactual_seq_loss_weight) > 0.0
            or float(args.counterfactual_pairwise_loss_weight) > 0.0
            or float(args.counterfactual_best_loss_weight) > 0.0
        ):
            gidx = rng.integers(0, len(av_groups), size=min(int(args.batch_size), len(av_groups)))
            gx = torch.from_numpy(np.stack([av_groups[i].x for i in gidx])).float().to(args.device)
            gslot = torch.from_numpy(np.stack([av_groups[i].slot for i in gidx])).float().to(args.device)
            gpairs = torch.from_numpy(np.stack([av_groups[i].action_pairs for i in gidx])).long().to(args.device)
            gvals = torch.from_numpy(np.stack([av_groups[i].values for i in gidx])).float().to(args.device)
            gmask = torch.from_numpy(np.stack([av_groups[i].mask for i in gidx])).float().to(args.device)
            with torch.no_grad():
                gcls, gtok, gsel, gactive = model.backbone.encode_tokens(gx)
            if bool(args.train_action_value_only):
                flat_pairs = gpairs.reshape(-1, 2)
                flat_cls = gcls[:, None, :].expand(-1, gpairs.shape[1], -1).reshape(-1, gcls.shape[-1])
                flat_tok = gtok[:, None, :, :].expand(-1, gpairs.shape[1], -1, -1).reshape(-1, gtok.shape[1], gtok.shape[2])
                flat_slot = gslot[:, None, :].expand(-1, gpairs.shape[1], -1).reshape(-1, gslot.shape[-1])
                cand_logits = g.predict_action_value(flat_cls, flat_tok, flat_slot, flat_pairs).reshape(gpairs.shape[0], gpairs.shape[1])
                cand_logits = cand_logits.masked_fill(gmask <= 0.0, -1e9)
            elif bool(args.counterfactual_use_policy_scores):
                seq_scores = g.policy_scores(gcls, gtok, gslot, gsel, gactive)
                rows = pair_rows(gpairs).clamp(min=0, max=MAXT)
                bidx = torch.arange(seq_scores.shape[0], device=args.device)[:, None].expand_as(rows[:, :, 0])
                cand_logits = seq_scores[bidx, rows[:, :, 0], 0] + seq_scores[bidx, rows[:, :, 1], 1]
                cand_logits = cand_logits.masked_fill(gmask <= 0.0, -1e9)
            else:
                seq_slots = gslot[:, None, :].expand(-1, int(g.seq_len), -1)
                seq_scores = g.sequence_scores(
                    gcls,
                    gtok,
                    gslot,
                    gsel,
                    gactive,
                    seq_slots=seq_slots,
                    use_step_context=bool(args.root_seq_step_context),
                )[:, 0]
                rows = pair_rows(gpairs).clamp(min=0, max=MAXT)
                bidx = torch.arange(seq_scores.shape[0], device=args.device)[:, None].expand_as(rows[:, :, 0])
                cand_logits = seq_scores[bidx, rows[:, :, 0], 0] + seq_scores[bidx, rows[:, :, 1], 1]
                cand_logits = cand_logits.masked_fill(gmask <= 0.0, -1e9)
            tau = max(1e-4, float(args.counterfactual_policy_tau))
            if float(args.counterfactual_seq_loss_weight) > 0.0:
                target_logits = (gvals / tau).masked_fill(gmask <= 0.0, -1e9)
                target_probs = torch.softmax(target_logits, dim=1)
                log_probs = torch.log_softmax(cand_logits, dim=1)
                counterfactual_seq_loss = -(target_probs * log_probs).sum(dim=1).mean()
            if float(args.counterfactual_pairwise_loss_weight) > 0.0:
                valid_pair = (gmask[:, :, None] > 0.0) & (gmask[:, None, :] > 0.0)
                valid_pair = valid_pair & torch.triu(torch.ones_like(valid_pair, dtype=torch.bool), diagonal=1)
                val_diff = gvals[:, :, None] - gvals[:, None, :]
                logit_diff = cand_logits[:, :, None] - cand_logits[:, None, :]
                direction = torch.sign(val_diff)
                useful = valid_pair & (direction.abs() > 0.0)
                if bool(useful.any()):
                    margin = torch.clamp(val_diff.abs() / tau, min=0.0, max=10.0)
                    pair_loss = F.softplus(-direction * logit_diff)
                    counterfactual_pairwise_loss = (pair_loss * margin).masked_select(useful).mean()
            if float(args.counterfactual_best_loss_weight) > 0.0:
                masked_vals = gvals.masked_fill(gmask <= 0.0, -1e9)
                best_targets = torch.argmax(masked_vals, dim=1)
                top2_vals = torch.topk(masked_vals, k=min(2, masked_vals.shape[1]), dim=1).values
                if top2_vals.shape[1] > 1:
                    best_gap = top2_vals[:, 0] - top2_vals[:, 1]
                else:
                    best_gap = torch.full((masked_vals.shape[0],), float("inf"), dtype=masked_vals.dtype, device=masked_vals.device)
                best_pairs = gpairs[torch.arange(gpairs.shape[0], device=args.device), best_targets]
                best_rows = pair_rows(best_pairs).clamp(min=0, max=MAXT)
                best_search_frac = (best_rows <= 0).to(gvals.dtype).mean(dim=1)
                best_mask = best_gap >= float(args.counterfactual_best_min_gap)
                best_mask = best_mask & (best_search_frac <= float(args.counterfactual_best_max_search_frac))
                if bool(best_mask.any()):
                    best_loss_all = F.cross_entropy(cand_logits, best_targets, reduction="none")
                    counterfactual_best_loss = best_loss_all[best_mask].mean()
        for k in range(unroll if data else 0):
            ids = idx + k
            act = torch.from_numpy(np.stack([data[i].action_pair for i in ids])).long().to(args.device)
            reward = torch.tensor([data[i].reward for i in ids], dtype=torch.float32, device=args.device)
            dt = torch.tensor([data[i].dt / 200.0 for i in ids], dtype=torch.float32, device=args.device)
            service = torch.from_numpy(np.stack([data[i].service_next for i in ids])).float().to(args.device)
            slotn = torch.from_numpy(np.stack([data[i].slot_next for i in ids])).float().to(args.device)
            xn = torch.from_numpy(np.stack([data[i].x_next for i in ids])).float().to(args.device)
            if float(args.action_service_loss_weight) > 0.0:
                action_service_loss = action_service_loss + F.smooth_l1_loss(g.predict_action_service(cls, tok, slot, act), service)
            cls, tok, r_p, dt_p = g(cls, tok, slot, act)
            service_p = g.predict_service(cls, tok)
            with torch.no_grad():
                cls_t, tok_t, sel_t, active_t = model.backbone.encode_tokens(xn)
                score_t, q_t = model.forward_scores(xn, slotn)
            if int(args.g_d_model) == 48:
                score_p, q_p = latent_scores(model, cls, tok, slotn, sel_t, active_t)
            else:
                score_p = g.policy_scores(cls, tok, slotn, sel_t, active_t)
                q_p = None
            g_score_p = g.policy_scores(cls, tok, slotn, sel_t, active_t)
            if float(args.gpolicy_search_profile_loss_weight) > 0.0:
                gpolicy_search_profile_loss = gpolicy_search_profile_loss + policy_search_profile_loss(
                    g_score_p,
                    slotn,
                    low_active_threshold=float(args.gpolicy_low_active_threshold),
                    high_active_threshold=float(args.gpolicy_high_active_threshold),
                    low_max_search_atoms=float(args.gpolicy_low_max_search_atoms),
                    high_min_search_atoms=float(args.gpolicy_high_min_search_atoms),
                    high_until_search_atoms=float(args.gpolicy_high_until_search_atoms),
                )
            finite = torch.isfinite(score_t) & (score_t > -1e8)
            if int(args.g_d_model) == 48:
                cls_lat_t, tok_lat_t = cls_t, tok_t
            else:
                with torch.no_grad():
                    cls_lat_t, tok_lat_t = g._project_state(cls_t, tok_t)
            latent_loss = latent_loss + F.smooth_l1_loss(cls, cls_lat_t) + F.smooth_l1_loss(tok, tok_lat_t)
            score_loss = score_loss + F.smooth_l1_loss(score_p[finite], score_t[finite])
            if q_p is not None:
                score_loss = score_loss + 0.25 * F.smooth_l1_loss(q_p[finite], q_t[finite])
            rd_loss = rd_loss + F.smooth_l1_loss(r_p, reward) + F.smooth_l1_loss(dt_p, dt)
            service_loss = service_loss + F.smooth_l1_loss(service_p, service)
            if np.max(ids + 1) < len(data):
                next_act = torch.from_numpy(np.stack([data[i + 1].action_pair for i in ids])).long().to(args.device)
                next_loss, next_terms = masked_pair_ce(g_score_p, next_act)
                action_loss = action_loss + next_loss
                action_terms += next_terms
                if float(args.pair_factored_ce_weight) > 0.0:
                    next_factored_loss, next_factored_terms = masked_pair_factored_ce(
                        g_score_p,
                        next_act,
                        type_weight=float(args.pair_factored_type_weight),
                        target_weight=float(args.pair_factored_target_weight),
                    )
                    action_loss = action_loss + float(args.pair_factored_ce_weight) * next_factored_loss
                    action_terms += next_factored_terms
            slot = slotn
        latent_loss = latent_loss / float(unroll)
        score_loss = score_loss / float(unroll)
        rd_loss = rd_loss / float(unroll)
        service_loss = service_loss / float(unroll)
        action_service_loss = action_service_loss / float(max(1, unroll + (1 if data else 0)))
        if action_terms > 0:
            action_loss = action_loss / float(action_terms)
        if data and float(args.gpolicy_search_profile_loss_weight) > 0.0:
            gpolicy_search_profile_loss = gpolicy_search_profile_loss / float(unroll + 1)
        loss = (
            float(args.latent_loss_weight) * latent_loss
            + float(args.score_loss_weight) * score_loss
            + float(args.rd_loss_weight) * rd_loss
            + float(args.service_loss_weight) * service_loss
            + float(args.action_service_loss_weight) * action_service_loss
            + float(args.action_value_loss_weight) * action_value_loss
            + float(args.action_frame_loss_weight) * action_frame_loss
            + float(args.action_loss_weight) * action_loss
            + float(args.gpolicy_search_profile_loss_weight) * gpolicy_search_profile_loss
            + float(args.gpolicy_distill_loss_weight) * gpolicy_distill_loss
            + float(args.root_seq_loss_weight) * root_seq_loss
            + float(args.root_seq_puct_soft_loss_weight) * root_seq_puct_soft_loss
            + float(args.seq_terminal_service_loss_weight) * seq_terminal_service_loss
            + float(args.counterfactual_seq_loss_weight) * counterfactual_seq_loss
            + float(args.counterfactual_pairwise_loss_weight) * counterfactual_pairwise_loss
            + float(args.counterfactual_best_loss_weight) * counterfactual_best_loss
            + float(args.service_gate_loss_weight) * service_gate_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
        opt.step()
        if bool(args.disable_policy_action_coupler) and hasattr(g, "policy_action_residual"):
            with torch.no_grad():
                g.policy_action_residual.weight.zero_()
                g.policy_action_residual.bias.zero_()
        if step % 50 == 0 or step == int(args.steps) - 1:
                print({"step": step, "loss": float(loss.detach().cpu()), "latent": float(latent_loss.detach().cpu()), "score": float(score_loss.detach().cpu()), "rd": float(rd_loss.detach().cpu()), "service": float(service_loss.detach().cpu()), "action_service": float(action_service_loss.detach().cpu()), "action_value": float(action_value_loss.detach().cpu()), "action_frame": float(action_frame_loss.detach().cpu()), "action": float(action_loss.detach().cpu()), "gpolicy_search_profile": float(gpolicy_search_profile_loss.detach().cpu()), "gpolicy_distill": float(gpolicy_distill_loss.detach().cpu()), "root_seq": float(root_seq_loss.detach().cpu()), "root_seq_ce": float(root_seq_ce_loss.detach().cpu()), "root_seq_factored_ce": float(root_seq_factored_ce_loss.detach().cpu()), "root_seq_s_search_frac": float(root_seq_s_search_frac.detach().cpu()), "root_seq_search_margin": float(root_seq_search_margin.detach().cpu()), "root_seq_track_margin": float(root_seq_track_margin.detach().cpu()), "root_seq_distill": float(root_seq_distill_loss.detach().cpu()), "seq_terminal_service": float(seq_terminal_service_loss.detach().cpu()), "counterfactual_seq": float(counterfactual_seq_loss.detach().cpu()), "counterfactual_pairwise": float(counterfactual_pairwise_loss.detach().cpu()), "counterfactual_best": float(counterfactual_best_loss.detach().cpu()), "root_seq_frac": float(root_seq_frac_loss.detach().cpu()), "root_seq_count": float(root_seq_count_loss.detach().cpu()), "root_seq_min_search": float(root_seq_min_search_loss.detach().cpu()), "root_seq_joint": float(root_seq_joint_loss.detach().cpu()), "root_seq_coverage": float(root_seq_coverage_loss.detach().cpu()), "root_seq_stop": float(root_seq_stop_loss.detach().cpu()), "service_gate": float(service_gate_loss.detach().cpu())}, flush=True)
        if int(args.save_every) > 0 and ((step + 1) % int(args.save_every) == 0 or step == int(args.steps) - 1):
            ckpt = out.with_name(f"{out.stem}.step{step + 1}{out.suffix}")
            torch.save(
                {
                    "state_dict": g.state_dict(),
                    "transitions": len(data),
                    "action_value_target_mean": float(getattr(args, "action_value_target_mean", 0.0)),
                    "action_value_target_std": float(getattr(args, "action_value_target_std", 1.0)),
                },
                ckpt,
            )
    torch.save(
        {
            "state_dict": g.state_dict(),
            "transitions": len(data),
            "action_value_target_mean": float(getattr(args, "action_value_target_mean", 0.0)),
            "action_value_target_std": float(getattr(args, "action_value_target_std", 1.0)),
        },
        out,
    )
    # GPU latent loop speed: h once, then g+heads repeatedly.
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    sample = data[: min(64, len(data))]
    if sample:
        x = torch.from_numpy(np.stack([t.x for t in sample])).float().to(args.device)
        slot = torch.from_numpy(np.stack([t.slot for t in sample])).float().to(args.device)
        act = torch.from_numpy(np.stack([t.action_pair for t in sample])).long().to(args.device)
    else:
        ssample = seq_data[: min(64, len(seq_data))]
        if ssample:
            x = torch.from_numpy(np.stack([t.x for t in ssample])).float().to(args.device)
            slot = torch.from_numpy(np.stack([t.slot for t in ssample])).float().to(args.device)
            act = torch.zeros((len(ssample), 2), dtype=torch.long, device=args.device)
        else:
            asample = av_data[: min(64, len(av_data))]
            if asample:
                x = torch.from_numpy(np.stack([t.x for t in asample])).float().to(args.device)
                slot = torch.from_numpy(np.stack([t.slot for t in asample])).float().to(args.device)
                act = torch.from_numpy(np.stack([t.action_pair for t in asample])).long().to(args.device)
            else:
                gsample = av_groups[: min(64, len(av_groups))]
                x = torch.from_numpy(np.stack([t.x for t in gsample])).float().to(args.device)
                slot = torch.from_numpy(np.stack([t.slot for t in gsample])).float().to(args.device)
                act = torch.from_numpy(np.stack([t.action_pairs[0] for t in gsample])).long().to(args.device)
    if x.shape[0] > 0:
        with torch.inference_mode():
            cls, tok, sel, active = model.backbone.encode_tokens(x)
            for _ in range(10):
                cp, tp, _, _ = g(cls, tok, slot, act)
                if int(args.g_d_model) == 48:
                    latent_scores(model, cp, tp, slot, sel, active)
                else:
                    g.policy_scores(cp, tp, slot, sel, active)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            loops = 100
            for _ in range(loops):
                cp, tp, _, _ = g(cls, tok, slot, act)
                if int(args.g_d_model) == 48:
                    latent_scores(model, cp, tp, slot, sel, active)
                else:
                    g.policy_scores(cp, tp, slot, sel, active)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            ms = 1000.0 * (time.perf_counter() - t0) / loops
        print({"saved": str(out), "batch": int(x.shape[0]), "latent_g_plus_heads_ms": ms, "per_state_ms": ms / max(1, int(x.shape[0]))}, flush=True)
    else:
        print({"saved": str(out), "batch": 0}, flush=True)


if __name__ == "__main__":
    main()
