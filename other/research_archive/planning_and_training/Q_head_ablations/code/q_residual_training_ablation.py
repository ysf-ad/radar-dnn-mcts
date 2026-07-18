"""Q-head target decomposition ablation.

Root cause tested here:
    The old Q training trained track_q(action) to the full action value while
    inference scored Track_i as type_q(track) + track_q(i). That double-counts
    the branch value and makes search-vs-track calibration fragile.

This script compares:
    old_full        type_q(track)=max/avg Q_track, track_q(i)=full Q_i
    residual_max    type_q(track)=max Q_track,     track_q(i)=Q_i-max Q_track
    residual_mean   type_q(track)=mean Q_track,    track_q(i)=Q_i-mean Q_track

The residual variants make the train-time target match the inference equation:
    Q_theta(s, track_i) = Q_type_theta(s, track) + Q_target_theta(s, i)
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from autoregressive_ablation_suite import ARQScorePlanner, plot_suite as plot_ablation
from final_radar_campaign import MAXT, run_fixed, seedall, summarize_window_df
from mutual_alpha_radar_loop import OUT as MODEL_OUT, collect_mutual_targets
from mutual_alpha_radar_loop import MutualArgmaxPolicyPlanner, MutualBatchQUrgencyPlanner, configured_env
from mutual_foundation import DEVICE, MutualRadarNet, SearchTarget
from repaired_campaign_tools import EDFPlanner, ESTPlanner


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CreateValid1" / "results" / "q_residual_training_ablation"
CLEAN = Path(r"C:\Users\yousi\Downloads\radar_outputs")
OUT.mkdir(parents=True, exist_ok=True)
CLEAN.mkdir(parents=True, exist_ok=True)


def base_args() -> SimpleNamespace:
    return SimpleNamespace(
        seed=91,
        env_mode="operational",
        search_refresh_tracked=0,
        search_refresh_gain=0.0,
        track_update_reward=0.30,
        track_loss_penalty=8.0,
        penalize_hidden_targets=1,
        search_debt_penalty_weight=0.00025,
        mcts_rollout_policy="edf",
        mcts_rollout_search_period_ms=100.0,
        mcts_prior_uniform_mix=0.8,
        c_puct=1.25,
        expand_top_k=100,
        q_scale=100.0,
        q_utility_weight=0.15,
        leaf_value_mix=0.25,
        belief_search_weight=0.25,
        belief_search_cap=4.0,
        episodes_per_iter=4,
        windows_per_episode=6,
        rollouts=8,
        train_mcts_mode="pq",
        train_initials="15,30,50,75,100",
        train_rates="1,2,3,4",
        seq_len=32,
        selfplay_replan_each_action=True,
        gamma=0.99,
    )


def load_model(path: Path | None = None) -> MutualRadarNet:
    ckpt = path or (MODEL_OUT / "mutual_alpha_model.pt")
    model = MutualRadarNet(d_model=96, nhead=4, nlayers=2).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


def collect_targets(model: MutualRadarNet, args, refresh: bool) -> List[SearchTarget]:
    rows: List[SearchTarget] = []
    old_refresh = args.search_refresh_tracked
    old_gain = args.search_refresh_gain
    if refresh:
        args.search_refresh_tracked = 1
        args.search_refresh_gain = 1.0
    else:
        args.search_refresh_tracked = 0
        args.search_refresh_gain = 0.0
    try:
        for it in range(1, 3):
            targets, _, windows = collect_mutual_targets(model, args, it)
            rows.extend(targets)
            print("collected", json.dumps({
                "iter": it,
                "refresh": refresh,
                "targets": len(targets),
                "mean_selfplay_reward": float(np.mean([w["reward"] for w in windows])) if windows else 0.0,
                "mean_search": float(np.mean([w["search_fraction"] for w in windows])) if windows else 0.0,
            }), flush=True)
    finally:
        args.search_refresh_tracked = old_refresh
        args.search_refresh_gain = old_gain
    return rows


def q_scale_for(rows: Iterable[SearchTarget]) -> float:
    vals = []
    for r in rows:
        vals.extend(np.abs(r.q[r.q_mask > 0.5]).tolist())
        vals.append(abs(float(r.ret)))
    return float(max(1.0, np.percentile(vals, 90))) if vals else 100.0


def set_trainable(model: MutualRadarNet, freeze_encoder: bool):
    for p in model.parameters():
        p.requires_grad = True
    if freeze_encoder:
        for name, p in model.named_parameters():
            if (
                name.startswith("token_proj")
                or name.startswith("slot_proj")
                or name.startswith("encoder")
                or name.startswith("cls_token")
            ):
                p.requires_grad = False


def train_q_step(model: MutualRadarNet, opt, rows: List[SearchTarget], batch_size: int, q_scale: float, mode: str):
    import random

    batch = random.sample(rows, min(int(batch_size), len(rows)))
    x = torch.from_numpy(np.stack([b.x for b in batch]).astype(np.float32)).to(DEVICE)
    slot = torch.from_numpy(np.stack([b.slot for b in batch]).astype(np.float32)).to(DEVICE)
    pi = torch.from_numpy(np.stack([b.pi for b in batch]).astype(np.float32)).to(DEVICE)
    q_raw = np.stack([b.q for b in batch]).astype(np.float32)
    qm_raw = np.stack([b.q_mask for b in batch]).astype(np.float32)
    q = torch.from_numpy(q_raw / q_scale).to(DEVICE)
    q_mask = torch.from_numpy(qm_raw).to(DEVICE)
    ret = torch.tensor([b.ret / q_scale for b in batch], dtype=torch.float32, device=DEVICE)

    type_logit, track_logits, value, type_q, track_q = model(x, slot)
    best_action = torch.argmax(pi, dim=1)
    type_loss = F.binary_cross_entropy_with_logits(type_logit, (best_action == 0).float())
    track_rows = best_action > 0
    rank_loss = F.cross_entropy(track_logits[track_rows], best_action[track_rows]) if bool(track_rows.any()) else torch.zeros((), device=DEVICE)
    v_loss = F.smooth_l1_loss(value, ret)

    search_q = q[:, 0]
    valid_track = q_mask[:, 1:] > 0.5
    track_q_targets = q[:, 1:].masked_fill(~valid_track, 0.0)
    any_track = valid_track.any(dim=1)
    if mode.endswith("mean"):
        denom = valid_track.float().sum(dim=1).clamp_min(1.0)
        branch = (track_q_targets * valid_track.float()).sum(dim=1) / denom
    else:
        branch = q[:, 1:].masked_fill(~valid_track, -1e9).amax(dim=1)
        branch = torch.where(any_track, branch, torch.zeros_like(branch))

    type_q_target = torch.stack([branch, search_q], dim=1)
    type_q_loss = F.smooth_l1_loss(type_q, type_q_target)

    full_valid = q_mask > 0.5
    full_valid[:, 0] = False
    if bool(full_valid.any()):
        if mode.startswith("residual"):
            residual_target = q - branch[:, None]
            track_q_loss = F.smooth_l1_loss(track_q[full_valid], residual_target[full_valid])
        else:
            track_q_loss = F.smooth_l1_loss(track_q[full_valid], q[full_valid])
    else:
        track_q_loss = torch.zeros((), device=DEVICE)

    # Q calibration is the point of this ablation, so weight Q losses higher
    # than the policy imitation term.
    loss = 0.25 * type_loss + 0.25 * rank_loss + 0.25 * v_loss + 1.0 * type_q_loss + 2.0 * track_q_loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    return {
        "loss": float(loss.detach().cpu()),
        "type": float(type_loss.detach().cpu()),
        "rank": float(rank_loss.detach().cpu()),
        "v": float(v_loss.detach().cpu()),
        "type_q": float(type_q_loss.detach().cpu()),
        "track_q": float(track_q_loss.detach().cpu()),
    }


def train_variant(base: MutualRadarNet, rows: List[SearchTarget], mode: str, freeze_encoder: bool, steps: int, lr: float):
    model = copy.deepcopy(base).to(DEVICE)
    set_trainable(model, freeze_encoder)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    q_scale = q_scale_for(rows)
    log = []
    model.train()
    for step in range(int(steps)):
        m = train_q_step(model, opt, rows, 96, q_scale, mode)
        if step % 25 == 0 or step == steps - 1:
            row = {"step": step, "mode": mode, "freeze_encoder": freeze_encoder, "q_scale": q_scale, **m}
            log.append(row)
            print("qtrain", json.dumps(row), flush=True)
    model.eval()
    tag = f"{mode}_{'frozen' if freeze_encoder else 'unfrozen'}"
    torch.save(model.cpu().state_dict(), OUT / f"{tag}.pt")
    model.to(DEVICE)
    pd.DataFrame(log).to_csv(OUT / f"{tag}_train_log.csv", index=False)
    return tag, model


def evaluate_variants(variants, windows: int, seeds: str, cells: str):
    all_raw = []
    all_win = []
    cell_list = []
    for item in cells.split(","):
        init, rate = item.split(":")
        cell_list.append((int(init), float(rate)))
    seed_list = [int(x) for x in seeds.split(",") if x]
    for tag, model in variants:
        rows = []
        wins = []
        env_cfg_args = base_args()
        for init, rate in cell_list:
            env = configured_env(float(rate), env_cfg_args)
            methods = {
                "BatchQScore_w8_o8_0r": lambda model=model: MutualBatchQUrgencyPlanner(model, deadline_weight=8.0, overdue_weight=8.0),
                "BatchQScore_w2_o2_0r": lambda model=model: MutualBatchQUrgencyPlanner(model, deadline_weight=2.0, overdue_weight=2.0),
                "AR_QScore_w2_o2_0r": lambda model=model: ARQScorePlanner(model, deadline_weight=2.0, overdue_weight=2.0),
                "SequentialQ_0r": lambda model=model: MutualArgmaxPolicyPlanner(model, mode="q"),
                "EDF": lambda: EDFPlanner(MAXT),
                "EST": lambda: ESTPlanner(MAXT),
            }
            for name, factory in methods.items():
                for seed in seed_list:
                    seedall(int(seed))
                    w, _ = run_fixed(factory(), name, int(init), MAXT, int(seed), int(windows), 200, env)
                    s = summarize_window_df(w, "fixed")
                    s.update(planner=name, initial_targets=int(init), rate=float(rate), seed=int(seed), variant=tag)
                    rows.append(s)
                    ww = w.copy()
                    ww["planner"] = name
                    ww["initial_targets"] = int(init)
                    ww["rate"] = float(rate)
                    ww["seed"] = int(seed)
                    ww["variant"] = tag
                    wins.append(ww)
                    print("qeval", tag, init, rate, seed, name, round(float(s["reward_per_200ms_eq"]), 4), round(float(s["planning_ms_per_200ms_eq"]), 2), flush=True)
        raw = pd.DataFrame(rows)
        win = pd.concat(wins, ignore_index=True)
        all_raw.append(raw)
        all_win.append(win)
    raw = pd.concat(all_raw, ignore_index=True)
    win = pd.concat(all_win, ignore_index=True)
    raw["planner_variant"] = raw["variant"] + "::" + raw["planner"]
    win["planner_variant"] = win["variant"] + "::" + win["planner"]
    raw2 = raw.copy()
    win2 = win.copy()
    raw2["planner"] = raw2["planner_variant"]
    win2["planner"] = win2["planner_variant"]
    summary, suite, cum, frontier = plot_ablation(raw2, win2, "q_residual_eval")
    raw.to_csv(OUT / "q_residual_eval_raw_by_variant.csv", index=False)
    win.to_csv(OUT / "q_residual_eval_windows_by_variant.csv", index=False)
    summary.to_csv(OUT / "q_residual_eval_summary.csv", index=False)
    for src in [OUT / "q_residual_eval_raw_by_variant.csv", OUT / "q_residual_eval_windows_by_variant.csv", OUT / "q_residual_eval_summary.csv"]:
        shutil.copy2(src, CLEAN / f"q_residual_{src.name}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=125)
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--seeds", default="82")
    ap.add_argument("--cells", default="15:1,30:2,50:3,75:4,100:4")
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()

    cfg = base_args()
    base = load_model()
    rows = collect_targets(base, cfg, refresh=args.refresh_data)
    print("target_stats", json.dumps({
        "targets": len(rows),
        "q_scale": q_scale_for(rows),
        "qmask_mean": float(np.mean([np.sum(r.q_mask) for r in rows])) if rows else 0.0,
        "search_pi_mean": float(np.mean([r.pi[0] for r in rows])) if rows else 0.0,
    }), flush=True)

    variants = [("base", base)]
    for mode, freeze in [
        ("old_full", True),
        ("residual_max", True),
        ("residual_mean", True),
        ("residual_max", False),
    ]:
        variants.append(train_variant(base, rows, mode, freeze, args.steps, lr=2e-4 if freeze else 8e-5))

    summary = evaluate_variants(variants, args.windows, args.seeds, args.cells)
    print(summary.head(30).to_string(index=False), flush=True)
    print("outputs", OUT, CLEAN, flush=True)


if __name__ == "__main__":
    main()
