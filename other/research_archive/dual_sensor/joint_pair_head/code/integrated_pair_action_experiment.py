from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "CreateValid1" / "experiments" / "code" / "model_code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from best_model_joint_vs_seq_ablation import (
    DirectPlanAdapter,
    WorkConservingAsyncBeamPlanner,
    WorkConservingAsyncCoupledPlanner,
    encode_joint_action,
    execute_first_valid_action_joint,
    execute_plan_until_budget_joint_compatible,
    run_exact_rescore_grid_joint,
)
from exact_env_mutual import (
    EDFPlanner,
    ESTPlanner,
    MAXT,
    _DummyPlanner,
    attach_env_obs,
    engine_env_cfg,
    env_cfg_for,
    xs_decode_action,
    xs_s_search_action,
    xs_x_search_action,
)
from final_radar_campaign import get_obs, summarize_window_df
from foundation_mcts_fair_eval import physical_candidates, run_heuristic
from joint_action_experiment import is_joint_action, split_joint_action
from mutual_features import slot_features, tokenize
from penalty_window_quota_learner_eval import make_exact_args
from repaired_campaign_tools import build_env
from two_sensor_physical_head_eval import PhysicalHeadPlanner, make_physical_model, state_potential
from pufferlib.ocean.radarxs import binding


@dataclass
class PairGroup:
    tokens: np.ndarray
    slot: np.ndarray
    pairs: np.ndarray
    labels: np.ndarray


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def ranked_for_sensor(
    base_planner: PhysicalHeadPlanner,
    obs: dict,
    sensor: int,
    selected: set[int],
    per_sensor_top: int,
    force_search: bool,
    elapsed: float = 0.0,
    search_count: int = 0,
    track_count: int = 0,
    last: int = -1,
):
    scores = base_planner.score_actions(
        obs,
        selected=selected,
        elapsed=float(elapsed),
        search_count=int(search_count),
        track_count=int(track_count),
        last=int(last),
    )
    candidates = physical_candidates(obs, MAXT)
    ranked = []
    for action in candidates:
        base, sid = xs_decode_action(int(action), MAXT)
        if int(sid) != int(sensor):
            continue
        if int(base) > 0 and int(base) in selected:
            continue
        if int(base) >= scores.shape[0]:
            continue
        ranked.append((float(scores[int(base), int(sensor)]), int(base), int(action)))
    ranked.sort(reverse=True, key=lambda x: x[0])
    out = ranked[: max(1, int(per_sensor_top))]
    search_action = xs_s_search_action(MAXT) if int(sensor) == 0 else xs_x_search_action(MAXT)
    if bool(force_search) and not any(int(a) == int(search_action) for _score, _base, a in out):
        for item in ranked:
            if int(item[2]) == int(search_action):
                out.append(item)
                break
    return out


def candidate_pair_rows(
    base_planner: PhysicalHeadPlanner,
    obs: dict,
    selected: set[int],
    per_sensor_top: int,
    elapsed: float = 0.0,
    search_count: int = 0,
    track_count: int = 0,
    last: int = -1,
):
    obs = attach_env_obs(dict(obs), base_planner.env_cfg, True, True)
    s_ranked = ranked_for_sensor(base_planner, obs, 0, selected, per_sensor_top, True, elapsed, search_count, track_count, last)
    x_ranked = (
        ranked_for_sensor(base_planner, obs, 1, selected, per_sensor_top, True, elapsed, search_count, track_count, last)
        if int(obs.get("enable_x_band", 0))
        else []
    )
    out = []
    for _s_score, s_base, s_action in s_ranked:
        for _x_score, x_base, x_action in x_ranked:
            if int(s_base) > 0 and int(s_base) == int(x_base):
                continue
            out.append((int(s_base), int(s_action), int(x_base), int(x_action)))
    return out


def exact_pair_label(eng, plan: list[int], debt: float, horizon_ms: float, potential_weight: float) -> float:
    root = binding.vec_snapshot(eng.env)
    try:
        reward, _spent, next_debt, _executed, _searches, _rows = execute_plan_until_budget_joint_compatible(
            eng,
            [int(a) for a in plan],
            float(horizon_ms),
            float(debt),
            "integrated_pair_label",
            0,
            0,
        )
        label = float(reward) + float(potential_weight) * state_potential(eng, float(next_debt))
    finally:
        binding.vec_restore(eng.env, root)
    return float(label)


def tail_from_current_state(
    tail_planner: WorkConservingAsyncBeamPlanner,
    obs: dict,
    first_action: int,
    horizon_ms: float,
    selected: set[int],
    elapsed: float,
    search_count: int,
    track_count: int,
    last: int,
) -> list[int]:
    plan_obs = attach_env_obs(dict(obs), tail_planner.env_cfg, True, True)
    selected_tail = set(int(x) for x in selected)
    elapsed_tail = float(elapsed)
    end_elapsed = float(elapsed) + float(horizon_ms)
    search_count_tail = int(search_count)
    track_count_tail = int(track_count)
    last_tail = int(last)
    plan = [int(first_action)]
    selected_tail, elapsed_tail, search_count_tail, track_count_tail, last_tail = tail_planner._advance_synthetic(
        plan_obs,
        int(first_action),
        selected_tail,
        elapsed_tail,
        search_count_tail,
        track_count_tail,
        last_tail,
    )
    while elapsed_tail < end_elapsed and len(plan) < 256:
        action = tail_planner._choose_pair(
            plan_obs,
            selected_tail,
            elapsed_tail,
            search_count_tail,
            track_count_tail,
            last_tail,
        )
        if action is None:
            break
        plan.append(int(action))
        selected_tail, elapsed_tail, search_count_tail, track_count_tail, last_tail = tail_planner._advance_synthetic(
            plan_obs,
            int(action),
            selected_tail,
            elapsed_tail,
            search_count_tail,
            track_count_tail,
            last_tail,
        )
    return plan


def collect_pair_groups(args, base_planner: PhysicalHeadPlanner, exact_args) -> list[PairGroup]:
    groups: list[PairGroup] = []
    tail_planner = WorkConservingAsyncBeamPlanner(
        base_planner,
        per_sensor_top=int(args.per_sensor_top),
        beams=max(1, int(args.per_sensor_top) * int(args.per_sensor_top)),
        include_search_candidate=True,
    )
    adapt = base_planner.adapt
    for seed in parse_ints(args.train_seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                env_cfg = env_cfg_for(float(rate), exact_args)
                env_cfg["enable_x_band"] = 1
                base_planner.env_cfg = dict(env_cfg)
                tail_planner.env_cfg = dict(env_cfg)
                eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
                eng.reset(seed=int(seed))
                debt = 0.0
                selected: set[int] = set()
                try:
                    for _window in range(int(args.collect_windows)):
                        spent = 0.0
                        search_count = 0
                        track_count = 0
                        last = -1
                        selected.clear()
                        while spent < 200.0 and len(groups) < int(args.max_groups) and not bool(eng.term_buf[0]):
                            obs = attach_env_obs(get_obs(eng, debt), env_cfg, True, True)
                            pairs = candidate_pair_rows(
                                base_planner,
                                obs,
                                selected,
                                int(args.per_sensor_top),
                                spent,
                                search_count,
                                track_count,
                                last,
                            )[: int(args.max_pairs_per_state)]
                            if not pairs:
                                break
                            tok = tokenize(adapt, obs, selected=selected, search_count=search_count)
                            slot = slot_features(obs, spent, search_count, track_count, last, 200.0)
                            labels = []
                            scored = []
                            for s_base, s_action, x_base, x_action in pairs:
                                joint = encode_joint_action(int(s_action), int(x_action))
                                tail = tail_from_current_state(
                                    tail_planner,
                                    obs,
                                    int(joint),
                                    float(args.label_horizon_ms),
                                    selected,
                                    spent,
                                    search_count,
                                    track_count,
                                    last,
                                )
                                label = exact_pair_label(eng, tail, debt, float(args.label_horizon_ms), float(args.potential_weight))
                                labels.append(float(label))
                                scored.append((float(label), int(joint)))
                            groups.append(
                                PairGroup(
                                    tokens=tok.astype(np.float32),
                                    slot=slot.astype(np.float32),
                                    pairs=np.asarray([(p[0], p[2]) for p in pairs], dtype=np.int64),
                                    labels=np.asarray(labels, dtype=np.float32),
                                )
                            )
                            best_action = max(scored, key=lambda x: x[0])[1]
                            obs_before = dict(obs)
                            reward, dt, executed = execute_first_valid_action_joint(eng, [int(best_action)], 200.0 - spent)
                            if executed is None or float(dt) <= 0.0:
                                break
                            spent += float(dt)
                            atoms = split_joint_action(int(executed)) if is_joint_action(int(executed)) else (int(executed),)
                            is_search = False
                            for atom in atoms:
                                base, sid = xs_decode_action(int(atom), MAXT)
                                sensor_free = (
                                    (int(sid) == 0 and float(obs_before.get("s_band_busy_ms", 0.0)) <= 0.0)
                                    or (int(sid) == 1 and float(obs_before.get("x_band_busy_ms", 0.0)) <= 0.0)
                                )
                                if not sensor_free:
                                    continue
                                if int(base) == 0:
                                    is_search = True
                                    search_count += 1
                                elif int(base) > 0:
                                    selected.add(int(base))
                                    track_count += 1
                                last = int(base)
                            debt = 0.0 if is_search else float(debt) + float(dt)
                        if len(groups) >= int(args.max_groups):
                            break
                finally:
                    eng.close()
                print({"groups": len(groups), "initial": initial, "rate": rate, "seed": seed}, flush=True)
                if len(groups) >= int(args.max_groups):
                    break
            if len(groups) >= int(args.max_groups):
                break
        if len(groups) >= int(args.max_groups):
            break
    if not groups:
        raise RuntimeError("no pair groups collected")
    return groups


def train_integrated_pair_head(model, groups: list[PairGroup], args, device, reference_model=None):
    model.to(device).train()
    if reference_model is not None:
        reference_model.to(device).eval()
    for name, param in model.named_parameters():
        if str(args.pair_train_scope) == "pair":
            param.requires_grad = name.startswith("pair_")
        elif str(args.pair_train_scope) == "action":
            param.requires_grad = name.startswith("pair_") or name.startswith("action_")
        elif str(args.pair_train_scope) == "heads":
            param.requires_grad = (
                name.startswith("pair_")
                or name.startswith("action_")
                or name.startswith("type_")
                or name.startswith("target_")
                or name.startswith("sensor_")
            )
        elif str(args.pair_train_scope) == "all":
            param.requires_grad = True
        else:
            raise ValueError(f"unknown pair_train_scope: {args.pair_train_scope}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.pair_lr), weight_decay=1e-4)
    all_labels = np.concatenate([g.labels for g in groups])
    label_mean = float(np.mean(all_labels))
    label_std = max(1.0, float(np.std(all_labels)))
    rng = np.random.default_rng(int(args.model_seed))
    for step in range(int(args.pair_steps)):
        losses = []
        q_losses = []
        for gi in rng.integers(0, len(groups), size=min(int(args.batch_groups), len(groups))):
            g = groups[int(gi)]
            x = torch.from_numpy(g.tokens).float().unsqueeze(0).to(device)
            slot = torch.from_numpy(g.slot).float().unsqueeze(0).to(device)
            pairs = torch.from_numpy(g.pairs).long().to(device)
            labels = torch.from_numpy(g.labels).float().to(device)
            scores, _q, pair_scores, pair_q, _valid = model.forward_pair_candidates(x, slot, pairs[None, :, :])
            logits = pair_scores[0]
            pred_q = pair_q[0]
            with torch.inference_mode():
                if reference_model is not None:
                    ref_scores_for_target, ref_q_for_target = reference_model.forward_scores(x, slot)
                else:
                    ref_scores_for_target, ref_q_for_target = scores.detach(), _q.detach()
                ref_base_score = (
                    float(args.policy_weight)
                    * (ref_scores_for_target[0, pairs[:, 0], 0] + ref_scores_for_target[0, pairs[:, 1], 1])
                    + float(args.q_weight)
                    * (ref_q_for_target[0, pairs[:, 0], 0] + ref_q_for_target[0, pairs[:, 1], 1])
                )
                base_choice = int(torch.argmax(ref_base_score).item())
                exact_choice = int(torch.argmax(labels).item())
                label_gap = float((labels[exact_choice] - labels[base_choice]).detach().cpu())
                supervised_choice = exact_choice if label_gap > float(args.pair_advantage_margin) else base_choice
            if float(args.pair_target_temperature) > 0.0:
                target_dist = F.softmax(labels / float(args.pair_target_temperature), dim=0)
                if float(args.pair_advantage_margin) > 0.0 and supervised_choice == base_choice:
                    target_dist = torch.zeros_like(target_dist)
                    target_dist[base_choice] = 1.0
                losses.append(-(target_dist * F.log_softmax(logits, dim=0)).sum())
            else:
                target = torch.tensor([supervised_choice], dtype=torch.long, device=device)
                losses.append(F.cross_entropy(logits.reshape(1, -1), target))
            q_target = (labels - label_mean) / label_std
            base_logits = scores[0, pairs[:, 0], 0] + scores[0, pairs[:, 1], 1]
            residual_reg = F.smooth_l1_loss(logits - base_logits, torch.zeros_like(logits))
            aux_loss = F.smooth_l1_loss(pred_q, q_target) + float(args.pair_residual_reg) * residual_reg
            if reference_model is not None and float(args.base_distill_weight) > 0.0:
                with torch.inference_mode():
                    ref_scores, ref_q = reference_model.forward_scores(x, slot)
                cand_scores = torch.stack([scores[0, pairs[:, 0], 0], scores[0, pairs[:, 1], 1]], dim=1)
                ref_cand_scores = torch.stack([ref_scores[0, pairs[:, 0], 0], ref_scores[0, pairs[:, 1], 1]], dim=1)
                cand_q = torch.stack([_q[0, pairs[:, 0], 0], _q[0, pairs[:, 1], 1]], dim=1)
                ref_cand_q = torch.stack([ref_q[0, pairs[:, 0], 0], ref_q[0, pairs[:, 1], 1]], dim=1)
                aux_loss = aux_loss + float(args.base_distill_weight) * (
                    F.smooth_l1_loss(cand_scores, ref_cand_scores) + 0.25 * F.smooth_l1_loss(cand_q, ref_cand_q)
                )
            q_losses.append(aux_loss)
        policy_loss = torch.stack(losses).mean()
        q_loss = torch.stack(q_losses).mean()
        loss = policy_loss + float(args.pair_q_loss_weight) * q_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step in {0, int(args.pair_steps) - 1} or step % max(1, int(args.log_every)) == 0:
            print(
                {
                    "pair_step": step,
                    "loss": float(loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "q_loss": float(q_loss.detach().cpu()),
                    "groups": len(groups),
                    "label_mean": label_mean,
                    "label_std": label_std,
                },
                flush=True,
            )
    model.eval()
    return model


class IntegratedPairPlanner(WorkConservingAsyncCoupledPlanner):
    def __init__(self, base: PhysicalHeadPlanner, per_sensor_top: int = 4, pair_policy_weight: float = 1.0, pair_q_weight: float = 0.0):
        super().__init__(base, per_sensor_top=int(per_sensor_top), include_search_candidate=True)
        self.pair_policy_weight = float(pair_policy_weight)
        self.pair_q_weight = float(pair_q_weight)

    def _encoded_state(self, obs, selected, elapsed, search_count, track_count, last):
        obs = attach_env_obs(dict(obs), self.env_cfg, True, True)
        tok = tokenize(self.base.adapt, obs, selected=selected, search_count=int(search_count))
        slot = slot_features(obs, float(elapsed), int(search_count), int(track_count), int(last), 200.0)
        with torch.inference_mode():
            x = torch.from_numpy(tok).float().unsqueeze(0)
            s = torch.from_numpy(slot).float().unsqueeze(0)
            if hasattr(self.base.model, "_base_action_context"):
                scores, q, action_ctx, valid = self.base.model._base_action_context(x, s)
            else:
                scores, q = self.base.model.forward_scores(x, s)
                action_ctx = None
                valid = None
            marginal = (self.base.policy_weight * scores + self.base.q_weight * q).squeeze(0).cpu().numpy()
        marginal[0, :] += float(self.base.search_score_bias)
        return obs, marginal, scores, q, action_ctx, valid

    def _ranked_for_sensor_with_scores(self, obs: dict, scores: np.ndarray, sensor: int, selected: set[int]):
        ranked = []
        for action in physical_candidates(obs, MAXT):
            base, sid = xs_decode_action(int(action), MAXT)
            if int(sid) != int(sensor):
                continue
            if int(base) > 0 and int(base) in selected:
                continue
            if int(base) >= scores.shape[0]:
                continue
            ranked.append((float(scores[int(base), int(sensor)]), int(base), int(action)))
        ranked.sort(reverse=True, key=lambda x: x[0])
        out = ranked[: self.per_sensor_top]
        search_action = xs_s_search_action(MAXT) if int(sensor) == 0 else xs_x_search_action(MAXT)
        if not any(int(a) == int(search_action) for _score, _base, a in out):
            for item in ranked:
                if int(item[2]) == int(search_action):
                    out.append(item)
                    break
        return out

    def _candidate_pairs(self, obs: dict, selected: set[int], elapsed: float, search_count: int, track_count: int, last: int):
        obs, marginal, scores_t, q_t, action_ctx_t, valid_t = self._encoded_state(obs, selected, elapsed, search_count, track_count, last)
        s_busy = float(obs.get("s_band_busy_ms", 0.0))
        x_busy = float(obs.get("x_band_busy_ms", 0.0))
        x_enabled = bool(int(obs.get("enable_x_band", 0)))
        s_free = s_busy <= 0.0
        x_free = x_enabled and x_busy <= 0.0
        s_dummy = xs_s_search_action(MAXT)
        x_dummy = xs_x_search_action(MAXT)
        out = []
        if s_free and x_free:
            s_ranked = self._ranked_for_sensor_with_scores(obs, marginal, 0, selected)
            x_ranked = self._ranked_for_sensor_with_scores(obs, marginal, 1, selected)
            pair_items = []
            for _s_score, s_base, s_action in s_ranked:
                for _x_score, x_base, x_action in x_ranked:
                    if int(s_base) > 0 and int(s_base) == int(x_base):
                        continue
                    pair_items.append((int(s_base), int(s_action), int(x_base), int(x_action)))
            if pair_items:
                rows = torch.tensor([[(item[0], item[2]) for item in pair_items]], dtype=torch.long)
                with torch.inference_mode():
                    if action_ctx_t is not None and valid_t is not None and hasattr(self.base.model, "forward_pair_candidates_from_context"):
                        _scores, _q, pair_scores, pair_q, _valid = self.base.model.forward_pair_candidates_from_context(
                            scores_t,
                            q_t,
                            action_ctx_t,
                            valid_t,
                            rows,
                        )
                    else:
                        tok = tokenize(self.base.adapt, obs, selected=selected, search_count=int(search_count))
                        slot = slot_features(obs, float(elapsed), int(search_count), int(track_count), int(last), 200.0)
                        x = torch.from_numpy(tok).float().unsqueeze(0)
                        s = torch.from_numpy(slot).float().unsqueeze(0)
                        _scores, _q, pair_scores, pair_q, _valid = self.base.model.forward_pair_candidates(x, s, rows)
                    scores = scores_t.squeeze(0)
                    q = q_t.squeeze(0)
                    pair_scores = pair_scores.squeeze(0)
                    pair_q = pair_q.squeeze(0)
                    s_rows = rows[0, :, 0]
                    x_rows = rows[0, :, 1]
                    policy_base = scores[s_rows, 0] + scores[x_rows, 1]
                    q_base = q[s_rows, 0] + q[x_rows, 1]
                    policy_residual = pair_scores - policy_base
                    q_residual = pair_q - q_base
                    combo = (
                        self.base.policy_weight * policy_base
                        + self.base.q_weight * q_base
                        + self.pair_policy_weight * policy_residual
                        + self.pair_q_weight * q_residual
                    ).cpu().numpy()
                for item, score in zip(pair_items, combo):
                    s_base, s_action, x_base, x_action = item
                    bias = float(self.base.search_score_bias) * float(int(s_base == 0) + int(x_base == 0))
                    out.append((float(score) + bias, encode_joint_action(int(s_action), int(x_action))))
        elif s_free:
            for s_score, _s_base, s_action in self._ranked_for_sensor_with_scores(obs, marginal, 0, selected):
                out.append((float(s_score), encode_joint_action(int(s_action), int(x_dummy))))
        elif x_free:
            for x_score, _x_base, x_action in self._ranked_for_sensor_with_scores(obs, marginal, 1, selected):
                out.append((float(x_score), encode_joint_action(int(s_dummy), int(x_action))))
        elif s_busy > 0.0 or (x_enabled and x_busy > 0.0):
            out.append((-1e6, encode_joint_action(int(s_dummy), int(x_dummy))))
        deduped = {}
        for score, action in out:
            deduped[int(action)] = max(float(score), deduped.get(int(action), -np.inf))
        return sorted([(score, action) for action, score in deduped.items()], reverse=True, key=lambda x: x[0])


def eval_methods(args, direct_base: PhysicalHeadPlanner, pair_base: PhysicalHeadPlanner, exact_args):
    rows = []
    windows = []
    actions = []
    for initial in parse_ints(args.eval_initials):
        for rate in parse_floats(args.eval_rates):
            env_cfg = env_cfg_for(float(rate), exact_args)
            env_cfg["enable_x_band"] = 1
            direct_base.env_cfg = dict(env_cfg)
            pair_base.env_cfg = dict(env_cfg)
            pair_planner = IntegratedPairPlanner(
                pair_base,
                per_sensor_top=int(args.per_sensor_top),
                pair_policy_weight=float(args.pair_policy_weight),
                pair_q_weight=float(args.pair_q_weight),
            )
            direct_planner = WorkConservingAsyncCoupledPlanner(direct_base, per_sensor_top=int(args.per_sensor_top), include_search_candidate=True)
            planners = {
                "EDF": EDFPlanner(MAXT),
                "EST": ESTPlanner(MAXT),
                "Qstrong_direct": DirectPlanAdapter(direct_planner),
                "Integrated_pair_direct": DirectPlanAdapter(pair_planner),
            }
            for name, planner in planners.items():
                if name in {"EDF", "EST"}:
                    w, a = run_heuristic(planner, name, int(initial), int(args.eval_seed), int(args.eval_windows), engine_env_cfg(env_cfg))
                else:
                    w, a = run_exact_rescore_grid_joint(planner, name, int(initial), int(args.eval_seed), int(args.eval_windows), env_cfg)
                s = summarize_window_df(w, "fixed")
                denom = max(1.0, float(w["spent_ms"].sum()) if "spent_ms" in w else float(len(w) * 200))
                has_busy = not a.empty and {"s_busy_ms", "x_busy_ms"}.issubset(a.columns)
                s_util = float(a["s_busy_ms"].sum() / denom) if has_busy else np.nan
                x_util = float(a["x_busy_ms"].sum() / denom) if has_busy else np.nan
                row = {
                    "method": name,
                    "initial": int(initial),
                    "rate": float(rate),
                    "reward": float(s.get("reward_per_200ms_eq", np.nan)),
                    "search": float(s.get("search_fraction", np.nan)),
                    "tracked": float(s.get("mean_tracked_targets", np.nan)),
                    "drop": float(s.get("mean_drop_pct_active", np.nan)),
                    "delay": float(s.get("mean_delay_active", np.nan)),
                    "latency_ms": float(s.get("planning_ms_per_decision", np.nan)),
                    "s_util": s_util,
                    "x_util": x_util,
                    "mean_sensor_util": float(np.nanmean([s_util, x_util])) if has_busy else np.nan,
                }
                print(row, flush=True)
                rows.append(row)
                windows.append(w.assign(method=name, initial=int(initial), rate=float(rate)))
                if not a.empty:
                    actions.append(a.assign(method=name, initial=int(initial), rate=float(rate)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).groupby("method").agg(
        reward=("reward", "mean"),
        search=("search", "mean"),
        tracked=("tracked", "mean"),
        drop=("drop", "mean"),
        delay=("delay", "mean"),
        latency_ms=("latency_ms", "mean"),
        mean_sensor_util=("mean_sensor_util", "mean"),
        n=("reward", "size"),
    ).reset_index().sort_values("reward", ascending=False).to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    if windows:
        pd.concat(windows, ignore_index=True).to_csv(out.with_name(out.stem + "_windows.csv"), index=False)
    if actions:
        pd.concat(actions, ignore_index=True).to_csv(out.with_name(out.stem + "_actions.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-state", default="CreateValid1/results/factorized_qstrong_edfonly_state.pt")
    ap.add_argument("--pair-state-in", default="")
    ap.add_argument("--out", default="CreateValid1/results/integrated_pair_direct_hard_60_4_20w.csv")
    ap.add_argument("--initials", default="60")
    ap.add_argument("--rates", default="4")
    ap.add_argument("--train-seeds", default="916")
    ap.add_argument("--collect-windows", type=int, default=20)
    ap.add_argument("--max-groups", type=int, default=256)
    ap.add_argument("--max-pairs-per-state", type=int, default=16)
    ap.add_argument("--label-horizon-ms", type=float, default=800.0)
    ap.add_argument("--potential-weight", type=float, default=1.0)
    ap.add_argument("--pair-steps", type=int, default=600)
    ap.add_argument("--batch-groups", type=int, default=16)
    ap.add_argument("--pair-lr", type=float, default=3e-4)
    ap.add_argument("--pair-q-loss-weight", type=float, default=0.1)
    ap.add_argument("--pair-target-temperature", type=float, default=1.0)
    ap.add_argument("--pair-residual-reg", type=float, default=0.05)
    ap.add_argument("--pair-advantage-margin", type=float, default=0.0)
    ap.add_argument("--pair-train-scope", choices=["pair", "action", "heads", "all"], default="pair")
    ap.add_argument("--base-distill-weight", type=float, default=0.0)
    ap.add_argument("--per-sensor-top", type=int, default=4)
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--q-weight", type=float, default=1.5)
    ap.add_argument("--pair-policy-weight", type=float, default=1.0)
    ap.add_argument("--pair-q-weight", type=float, default=0.0)
    ap.add_argument("--search-bias", type=float, default=-12.0)
    ap.add_argument("--eval-initials", default="60")
    ap.add_argument("--eval-rates", default="4")
    ap.add_argument("--eval-seed", type=int, default=916)
    ap.add_argument("--eval-windows", type=int, default=20)
    ap.add_argument("--model-seed", type=int, default=123)
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()
    args.windows = int(args.eval_windows)

    torch.set_num_threads(1)
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = True
    exact_args.single_sensor = False
    env_cfg = env_cfg_for(float(parse_floats(args.rates)[0]), exact_args)
    env_cfg["enable_x_band"] = 1

    direct_model = make_physical_model("two_row_action_attention_factored_loss", args)
    pair_model = make_physical_model("two_row_pair_action_attention", args)
    state = torch.load(args.base_state, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    direct_model.load_state_dict(state, strict=True)
    missing, unexpected = pair_model.load_state_dict(state, strict=False)
    print({"pair_load_missing": len(missing), "pair_load_unexpected": len(unexpected)}, flush=True)

    direct_base = PhysicalHeadPlanner(
        direct_model.eval(),
        "two_row_action_attention_factored_loss",
        env_cfg,
        policy_weight=float(args.policy_weight),
        q_weight=float(args.q_weight),
        search_score_bias=float(args.search_bias),
    )
    pair_base = PhysicalHeadPlanner(
        pair_model.eval(),
        "two_row_pair_action_attention",
        env_cfg,
        policy_weight=float(args.policy_weight),
        q_weight=float(args.q_weight),
        search_score_bias=float(args.search_bias),
    )

    if str(args.pair_state_in).strip():
        pair_state = torch.load(args.pair_state_in, map_location="cpu", weights_only=False)
        if isinstance(pair_state, dict) and "state_dict" in pair_state:
            pair_state = pair_state["state_dict"]
        pair_model.load_state_dict(pair_state, strict=True)
        pair_model.eval()
        print({"loaded_pair_state": str(args.pair_state_in)}, flush=True)
    else:
        groups = collect_pair_groups(args, direct_base, exact_args)
        train_integrated_pair_head(pair_model, groups, args, torch.device("cpu"), reference_model=direct_model)
        state_out = Path(args.out).with_name(Path(args.out).stem + "_state.pt")
        torch.save(pair_model.state_dict(), state_out)
        print({"saved_state": str(state_out)}, flush=True)
    eval_methods(args, direct_base, pair_base, exact_args)


if __name__ == "__main__":
    main()
