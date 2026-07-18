from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from exact_env_mutual import (
    MAXT,
    ExactEnvMCTS,
    SnapshotSimulator,
    _DummyPlanner,
    build_env,
    choose_root_action,
    env_cfg_for,
    get_obs,
    load_model,
    xs_action_fractions,
    xs_decode_action,
    xs_s_search_action,
    xs_s_track_action,
    xs_x_search_action,
    xs_x_track_action,
)
from mutual_features import slot_features, tokenize
from repaired_campaign_tools import EDFPlanner, ESTPlanner, execute_first_valid_action
from realistic_reward_retrain import adapter
from pufferlib.ocean.radarxs import binding


OUT = Path(r"C:\Users\yousi\Downloads\radar_outputs\muzero_g0_prototype")
OUT.mkdir(parents=True, exist_ok=True)


class LatentDynamics(nn.Module):
    def __init__(self, d_model: int = 96, max_actions: int = 2 * (MAXT + 1)):
        super().__init__()
        self.action_emb = nn.Embedding(max_actions, d_model)
        self.cls_updater = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.tok_updater = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.reward_dt = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, cls: torch.Tensor, tok: torch.Tensor, action_idx: torch.Tensor):
        a = self.action_emb(action_idx)
        cls_delta = self.cls_updater(torch.cat([cls, a], dim=-1))
        a_rep = a[:, None, :].expand(-1, tok.shape[1], -1)
        cls_rep = cls[:, None, :].expand(-1, tok.shape[1], -1)
        tok_delta = self.tok_updater(torch.cat([tok, cls_rep, a_rep], dim=-1))
        pred = self.reward_dt(torch.cat([cls, a], dim=-1))
        return cls + cls_delta, tok + tok_delta, pred[:, 0], pred[:, 1]


class LatentPrediction(nn.Module):
    def __init__(self, d_model: int = 96, slot_dim: int = 11):
        super().__init__()
        self.slot_proj = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, d_model), nn.GELU())
        self.policy = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.q = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.value = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, cls: torch.Tensor, tok: torch.Tensor, slot: torch.Tensor):
        slot_emb = self.slot_proj(slot)
        cls_rep = cls[:, None, :].expand(-1, tok.shape[1], -1)
        slot_rep = slot_emb[:, None, :].expand(-1, tok.shape[1], -1)
        ctx = torch.cat([tok, cls_rep, slot_rep], dim=-1)
        return self.policy(ctx), self.q(ctx), self.value(torch.cat([cls, slot_emb], dim=-1)).squeeze(-1)


@dataclass
class Trans:
    x: np.ndarray
    slot: np.ndarray
    action: int
    reward: float
    dt: float
    x_next: np.ndarray
    slot_next: np.ndarray
    next_action: int = -1


def phys_to_index(action: int) -> int:
    base, sensor = xs_decode_action(int(action), MAXT)
    if sensor is None:
        sensor = 0
    base = int(np.clip(base, 0, MAXT))
    return base * 2 + int(sensor)


def index_to_phys(idx: int) -> int:
    base = int(idx) // 2
    sensor = int(idx) % 2
    if base <= 0:
        return xs_s_search_action(MAXT) if sensor == 0 else xs_x_search_action(MAXT)
    return xs_s_track_action(base, MAXT) if sensor == 0 else xs_x_track_action(base, MAXT)


def make_args(ckpt: str, device: str):
    return SimpleNamespace(
        ckpt=ckpt,
        device=device,
        d_model=96,
        nhead=4,
        nlayers=2,
        head_arch="baseline",
        enable_x_band=True,
        sensor_action_mode="explicit_head",
        env_mode="mcts_sched_v1",
        track_update_reward=0.30,
        track_loss_penalty=4.0,
        track_urgency_bonus_weight=-1.0,
        search_refresh_tracked=0,
        search_refresh_gain=0.0,
        search_debt_penalty_weight=0.0,
        sector_staleness_weight=0.0,
        searched_sector_reward_weight=0.25,
        search_frame_overdue_weight=0.05,
        search_frame_desired_ms=3000.0,
        search_frame_deadline_ms=4500.0,
        search_frame_drop_penalty=4.0,
        penalize_hidden_targets=1,
        disable_x_search=False,
    )


def teacher_action(model, sim, rollouts: int = 1) -> int:
    mcts = ExactEnvMCTS(
        model,
        sim,
        [],
        rollouts=rollouts,
        prior_mode="branch_corrected",
        head_mode="pq",
        q_utility_weight=1.0,
        sensor_action_mode="explicit_head",
        rollout_policy="model",
        expand_top_k=8,
        horizon_windows=4,
        skip_default_rollout_seed=True,
        eager_edge_depth=1,
    )
    root = mcts.run()
    return choose_root_action(root, "q")


def collect_transitions(model, args, n_trans: int, windows: int = 4) -> List[Trans]:
    adapt = adapter()
    rows: List[Trans] = []
    cfgs = [(15, 0.0), (15, 3.0), (50, 0.0), (50, 3.0)]
    seed = 100
    while len(rows) < n_trans:
        for init, rate in cfgs:
            env_cfg = env_cfg_for(rate, args)
            eng = build_env(_DummyPlanner(), init, MAXT, seed, 200, env_cfg)
            eng.reset(seed=seed)
            debt = 0.0
            try:
                for _ in range(windows):
                    used = 0.0
                    while used < 200.0 and len(rows) < n_trans:
                        obs = get_obs(eng, debt)
                        sim = SnapshotSimulator(eng, debt)
                        action = teacher_action(model, sim, rollouts=1)
                        x = tokenize(adapt, obs, selected=set(), search_count=0)
                        slot = slot_features(obs, 0.0, 0, 0, -1, 200.0)
                        reward, dt, debt, executed = sim.commit(action)
                        if executed is None or dt <= 0:
                            break
                        obs_next = get_obs(eng, debt)
                        x_next = tokenize(adapt, obs_next, selected=set(), search_count=0)
                        slot_next = slot_features(obs_next, 0.0, 0, 0, -1, 200.0)
                        next_action = -1
                        if not eng.term_buf[0]:
                            try:
                                cur_snap = binding.vec_snapshot(eng.env)
                                next_action = int(teacher_action(model, SnapshotSimulator(eng, debt), rollouts=1))
                                binding.vec_restore(eng.env, cur_snap)
                            except Exception:
                                next_action = -1
                        # Train on the commanded physical action.  The env can
                        # report collapsed logical actions (e.g. search -> 0),
                        # which is not the model's explicit S/X action space.
                        rows.append(Trans(x, slot, int(action), float(reward), float(dt), x_next, slot_next, next_action))
                        used += float(dt)
                        if len(rows) >= n_trans:
                            break
            finally:
                eng.close()
            seed += 1
            if len(rows) >= n_trans:
                break
    return rows


def encode_batch(model, xs: torch.Tensor, slots: torch.Tensor):
    cls, tok, selected, active = model.encode_tokens(xs)
    return cls, tok, selected, active


def _physical_ce(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    flat = logits.reshape(logits.shape[0], -1)
    return F.cross_entropy(flat, actions)


def train_g(model, dyn, pred, data: List[Trans], steps: int, batch: int, device):
    opt = torch.optim.AdamW(list(dyn.parameters()) + list(pred.parameters()), lr=3e-4, weight_decay=1e-4)
    model.eval()
    dyn.train()
    pred.train()
    for step in range(1, steps + 1):
        idx = np.random.randint(0, len(data), size=batch)
        xs = torch.from_numpy(np.stack([data[i].x for i in idx])).float().to(device)
        slots = torch.from_numpy(np.stack([data[i].slot for i in idx])).float().to(device)
        xn = torch.from_numpy(np.stack([data[i].x_next for i in idx])).float().to(device)
        slots_n = torch.from_numpy(np.stack([data[i].slot_next for i in idx])).float().to(device)
        act = torch.tensor([phys_to_index(data[i].action) for i in idx], dtype=torch.long, device=device)
        next_act = torch.tensor([phys_to_index(data[i].next_action if data[i].next_action >= 0 else data[i].action) for i in idx], dtype=torch.long, device=device)
        rew = torch.tensor([data[i].reward for i in idx], dtype=torch.float32, device=device)
        dt = torch.tensor([data[i].dt / 200.0 for i in idx], dtype=torch.float32, device=device)
        with torch.no_grad():
            cls, tok, selected, active = encode_batch(model, xs, slots)
            cls_t, tok_t, selected_t, active_t = encode_batch(model, xn, slots_n)
            real_heads = model.forward_with_sensor_from_latent(cls_t, tok_t, selected_t, active_t, slots_n)
        cls_p, tok_p, r_p, dt_p = dyn(cls, tok, act)
        pred_heads = model.forward_with_sensor_from_latent(cls_p, tok_p, selected_t, active_t, slots_n)
        logits0, q0, v0 = pred(cls, tok, slots)
        logits1, q1, v1 = pred(cls_p, tok_p, slots_n)
        latent_loss = F.smooth_l1_loss(cls_p, cls_t) + F.smooth_l1_loss(tok_p, tok_t)
        loss = 0.25 * latent_loss + 0.1 * F.smooth_l1_loss(r_p, rew) + 0.1 * F.smooth_l1_loss(dt_p, dt)
        loss = loss + 3.0 * _physical_ce(logits0, act) + 3.0 * _physical_ce(logits1, next_act)
        # Q head is trained as a simple advantage-shaped immediate action value
        # in this prototype.  The policy is the main decision signal.
        q0_flat = q0.reshape(q0.shape[0], -1).gather(1, act[:, None]).squeeze(1)
        q1_flat = q1.reshape(q1.shape[0], -1).gather(1, next_act[:, None]).squeeze(1)
        loss = loss + 0.05 * F.smooth_l1_loss(q0_flat, rew) + 0.05 * F.smooth_l1_loss(q1_flat, rew)
        # Value-equivalent consistency: the imagined latent does not need to
        # reconstruct radar state, but it must preserve the heads used for
        # planning.  This is the missing MuZero-like constraint in g0.
        for pred_head, real in zip(pred_heads, real_heads):
            if pred_head.shape == real.shape:
                finite = torch.isfinite(real) & (real > -1e8)
                if finite.any():
                    loss = loss + 0.05 * F.smooth_l1_loss(pred_head[finite], real[finite])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
        opt.step()
        if step % max(1, steps // 5) == 0:
            print(f"g step {step}/{steps} loss={float(loss.detach().cpu()):.4f}", flush=True)


def latent_choose(model, dyn, cls, tok, token_active, selected, slot, valid_actions, device):
    with torch.inference_mode():
        tl, tr, value, tq, trq, slog, sq = model.forward_with_sensor_from_latent(cls, tok, selected, token_active, slot) if hasattr(model, "forward_with_sensor_from_latent") else (None,) * 7
    raise RuntimeError("latent helper not patched")


def add_latent_helper(model):
    import types

    def helper(self, cls_out, tok_out, selected, token_active, slot):
        type_logit, track_logits, value, type_q, track_q = self.forward_heads(cls_out, tok_out, selected, token_active, slot)
        slot_emb = self.slot_proj(slot)
        cls_rep = cls_out.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
        slot_rep = slot_emb.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
        sensor_ctx = torch.cat([tok_out, cls_rep, slot_rep], dim=-1)
        sensor_logits = self.sensor_head(sensor_ctx)
        sensor_q = self.sensor_q_head(sensor_ctx)
        action_mask = token_active & ~selected
        action_mask[:, 0] = True
        sensor_logits = sensor_logits.masked_fill(~action_mask[:, :, None], -1e9)
        sensor_q = sensor_q.masked_fill(~action_mask[:, :, None], 0.0)
        return type_logit, track_logits, value, type_q, track_q, sensor_logits, sensor_q

    model.forward_with_sensor_from_latent = types.MethodType(helper, model)


def latent_action(model, cls, tok, selected, active, slot, valid, mode: str = "q"):
    with torch.inference_mode():
        tl, tr, _, tq, trq, slog, sq = model.forward_with_sensor_from_latent(cls, tok, selected, active, slot)
        p_search = float(torch.sigmoid(tl[0]).detach().cpu())
        tr_prob = torch.softmax(tr[0], dim=0).detach().cpu().numpy()
        slog_np = slog[0].detach().cpu().numpy()
    tq_np = tq[0].detach().cpu().numpy()
    trq_np = trq[0].detach().cpu().numpy()
    sq_np = sq[0].detach().cpu().numpy()
    best = int(valid[0])
    best_score = -1e30
    for a in valid:
        base, sensor = xs_decode_action(int(a), MAXT)
        if sensor is None:
            sensor = 0
        if base <= 0:
            q_score = float(tq_np[1]) + float(sq_np[0, sensor])
            sensor_p = float(np.exp(slog_np[0, sensor] - np.max(slog_np[0])) / max(1e-12, np.sum(np.exp(slog_np[0] - np.max(slog_np[0])))))
            p_score = float(np.log(max(1e-9, p_search * sensor_p)))
        else:
            q_score = float(tq_np[0]) + float(trq_np[base]) + float(sq_np[base, sensor])
            sensor_p = float(np.exp(slog_np[base, sensor] - np.max(slog_np[base])) / max(1e-12, np.sum(np.exp(slog_np[base] - np.max(slog_np[base])))))
            p_score = float(np.log(max(1e-9, (1.0 - p_search) * float(tr_prob[base]) * sensor_p)))
        if mode == "p":
            score = p_score
        elif mode == "pq":
            score = q_score + 0.25 * p_score
        else:
            score = q_score
        if score > best_score:
            best_score = score
            best = int(a)
    return best


def latent_action_f0(pred, cls, tok, slot, valid, mode: str = "p"):
    with torch.inference_mode():
        logits, q, _ = pred(cls, tok, slot)
    logits_np = logits[0].detach().cpu().numpy()
    q_np = q[0].detach().cpu().numpy()
    best = int(valid[0])
    best_score = -1e30
    for a in valid:
        idx = phys_to_index(int(a))
        base = idx // 2
        sensor = idx % 2
        p_score = float(logits_np[base, sensor])
        q_score = float(q_np[base, sensor])
        score = q_score if mode == "q" else (q_score + 0.25 * p_score if mode == "pq" else p_score)
        if score > best_score:
            best_score = score
            best = int(a)
    return best


def valid_from_obs(obs: Dict[str, np.ndarray], selected_targets: set[int]) -> List[int]:
    active = np.asarray(obs["active_mask"]).astype(bool)
    deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
    ranges = np.asarray(obs.get("target_range", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)
    s_free = float(obs.get("s_band_busy_ms", 0.0)) <= 0.0
    x_free = float(obs.get("x_band_busy_ms", 0.0)) <= 0.0
    valid: List[int] = []
    if s_free:
        valid.append(xs_s_search_action(MAXT))
    if x_free:
        valid.append(xs_x_search_action(MAXT))
    tracked = active & (deadline >= 0.0)
    for idx in np.where(tracked)[0].astype(int).tolist():
        base = idx + 1
        r = float(ranges[idx]) if idx < len(ranges) else 0.0
        if s_free and 10_000_000.0 < r < 184_000_000.0:
            valid.append(xs_s_track_action(base, MAXT))
        if x_free and 5_000_000.0 < r < 100_000_000.0:
            valid.append(xs_x_track_action(base, MAXT))
    return valid if valid else [xs_s_search_action(MAXT)]


def eval_muzero_window(model, dyn, args, windows: int = 4, mode: str = "q", pred=None):
    adapt = adapter()
    cfgs = [(15, 0.0), (15, 3.0), (50, 0.0), (50, 3.0)]
    rows = []
    device = next(model.parameters()).device
    for init, rate in cfgs:
        env_cfg = env_cfg_for(rate, args)
        eng = build_env(_DummyPlanner(), init, MAXT, 100, 200, env_cfg)
        eng.reset(seed=100)
        debt = 0.0
        cumulative = 0.0
        actions: List[int] = []
        t0 = time.perf_counter()
        try:
            for _ in range(windows):
                obs_cur = get_obs(eng, debt)
                x = tokenize(adapt, obs_cur, selected=set(), search_count=0)
                slot_np = slot_features(obs_cur, 0.0, 0, 0, -1, 200.0)
                tx = torch.from_numpy(x).float().unsqueeze(0).to(device)
                slot = torch.from_numpy(slot_np).float().unsqueeze(0).to(device)
                with torch.inference_mode():
                    cls, tok, selected, active = model.encode_tokens(tx)
                planned = []
                selected_targets: set[int] = set()
                used = 0.0
                search_count = 0
                track_count = 0
                last = -1
                while used < 200.0:
                    valid = valid_from_obs(obs_cur, selected_targets)
                    if pred is not None:
                        a = latent_action_f0(pred, cls, tok, slot, valid, mode=mode)
                    else:
                        a = latent_action(model, cls, tok, selected, active, slot, valid, mode=mode)
                    # Commit the same action to the real env.
                    reward, dt, debt, executed = SnapshotSimulator(eng, debt).commit(a)
                    if executed is None or dt <= 0:
                        break
                    planned.append(int(executed))
                    cumulative += float(reward)
                    used += float(dt)
                    base, _ = xs_decode_action(int(executed), MAXT)
                    last = int(base)
                    if base <= 0:
                        search_count += 1
                    else:
                        track_count += 1
                    obs_cur = get_obs(eng, debt)
                    slot_np = slot_features(obs_cur, used, search_count, track_count, last, 200.0)
                    slot = torch.from_numpy(slot_np).float().unsqueeze(0).to(device)
                    act_t = torch.tensor([phys_to_index(int(a))], dtype=torch.long, device=device)
                    with torch.inference_mode():
                        cls, tok, _, _ = dyn(cls, tok, act_t)
                    if used >= 200.0:
                        break
                actions.extend(planned)
            latency = (time.perf_counter() - t0) * 1000.0 / max(1, windows)
            obs = get_obs(eng, debt)
            active_np = np.asarray(obs["active_mask"]).astype(bool)
            tracked = active_np & (np.asarray(obs["t_deadline"], dtype=np.float32) >= 0.0)
            dropped = active_np & (np.asarray(obs["t_deadline"], dtype=np.float32) < 0.0)
            rows.append(
                {
                    "planner": f"MuZeroG0F0Window_{mode}" if pred is not None else f"MuZeroG0Window_{mode}",
                    "initial": init,
                    "rate": rate,
                    "reward": cumulative / windows,
                    "latency": latency,
                    "search": float(np.mean([xs_decode_action(a, MAXT)[0] == 0 for a in actions])) if actions else 0.0,
                    **xs_action_fractions(actions, MAXT),
                    "tracked": float(np.sum(tracked)),
                    "drop_pct": float(100.0 * np.sum(dropped) / max(1, np.sum(active_np))),
                }
            )
        finally:
            eng.close()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--transitions", type=int, default=160)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--eval-windows", type=int, default=4)
    ap.add_argument("--collect-windows", type=int, default=4)
    ap.add_argument("--dyn-ckpt", default="")
    args0 = ap.parse_args()
    args = make_args(args0.ckpt, args0.device)
    device = torch.device(args0.device)
    model = load_model(args).to(device).eval()
    add_latent_helper(model)
    dyn = LatentDynamics(96).to(device)
    pred = LatentPrediction(96).to(device)
    if args0.dyn_ckpt:
        ck = torch.load(args0.dyn_ckpt, map_location=device)
        if isinstance(ck, dict) and "dyn" in ck:
            dyn.load_state_dict(ck["dyn"])
            pred.load_state_dict(ck["pred"])
        else:
            dyn.load_state_dict(ck)
    else:
        print("collecting transitions", flush=True)
        data = collect_transitions(model, args, args0.transitions, windows=args0.collect_windows)
        print(f"collected {len(data)} transitions", flush=True)
        train_g(model, dyn, pred, data, args0.steps, min(64, len(data)), device)
        torch.save({"dyn": dyn.state_dict(), "pred": pred.state_dict()}, OUT / "muzero_g0f0.pt")
    frames = [eval_muzero_window(model, dyn.eval(), args, windows=args0.eval_windows, mode=m) for m in ("q", "pq", "p")]
    frames += [eval_muzero_window(model, dyn.eval(), args, windows=args0.eval_windows, mode=m, pred=pred.eval()) for m in ("q", "pq", "p")]
    df = pd.concat(frames, ignore_index=True)
    out_csv = OUT / "muzero_g0_eval.csv"
    df.to_csv(out_csv, index=False)
    print(df.groupby("planner").mean(numeric_only=True).reset_index().to_string(index=False), flush=True)
    print(out_csv, flush=True)


if __name__ == "__main__":
    main()
