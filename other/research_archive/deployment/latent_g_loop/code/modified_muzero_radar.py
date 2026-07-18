from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from exact_env_mutual import (
    MAXT,
    _DummyPlanner,
    build_env,
    env_cfg_for,
    get_obs,
    load_model,
    run_fixed,
    xs_decode_action,
    xs_s_search_action,
    xs_s_track_action,
    xs_x_search_action,
    xs_x_track_action,
)
from mutual_features import slot_features, tokenize
from mutual_foundation import MutualRadarDirectPlanner, MutualRadarNet
from realistic_reward_retrain import adapter
from repaired_campaign_tools import EDFPlanner, ESTPlanner, SEARCH_DWELL_MS, execute_first_valid_action


OUT = Path(r"C:\Users\yousi\Downloads\radar_outputs\modified_muzero_radar")
OUT.mkdir(parents=True, exist_ok=True)


def parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**31 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def make_base_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        ckpt=str(args.init_ckpt or ""),
        device=str(args.device),
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        nlayers=int(args.nlayers),
        head_arch=str(args.head_arch),
        enable_x_band=True,
        env_mode=str(args.env_mode),
        track_urgency_bonus_weight=-1.0,
    )


def flat_action_from_physical(action: int, max_trackers: int = MAXT) -> Tuple[int, int, int]:
    base, sensor = xs_decode_action(int(action), max_trackers)
    if base < 0:
        base = 0
    if sensor is None:
        sensor = 0
    base = int(np.clip(base, 0, max_trackers))
    sensor = int(np.clip(sensor, 0, 1))
    return base * 2 + sensor, base, sensor


def physical_from_base_sensor(base: int, sensor: int, max_trackers: int = MAXT) -> int:
    base = int(base)
    sensor = int(sensor)
    if base <= 0:
        return xs_x_search_action(max_trackers) if sensor == 1 else xs_s_search_action(max_trackers)
    return xs_x_track_action(base, max_trackers) if sensor == 1 else xs_s_track_action(base, max_trackers)


@dataclass
class Transition:
    tokens: np.ndarray
    slot: np.ndarray
    next_tokens: np.ndarray
    next_slot: np.ndarray
    action_flat: int
    base_action: int
    sensor: int
    reward: float
    ret: float
    done: float
    policy: str
    init: int
    rate: float
    seed: int


def model_first_action(planner: MutualRadarDirectPlanner, obs: dict, budget_ms: float = 200.0) -> int:
    plan = planner.plan(obs, budget_ms=budget_ms)
    return int(plan[0]) if plan else xs_s_search_action(MAXT)


def random_valid_action(obs: dict, p_search: float = 0.20) -> int:
    active = np.asarray(obs["active_mask"], dtype=bool)
    deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
    candidates = np.where(active & (deadline >= 0.0))[0] + 1
    if len(candidates) == 0 or random.random() < p_search:
        sensor = 1 if (obs.get("enable_x_band", 0) and random.random() < 0.35) else 0
        return physical_from_base_sensor(0, sensor)
    base = int(random.choice(candidates.tolist()))
    sensor = 1 if (obs.get("enable_x_band", 0) and random.random() < 0.35) else 0
    return physical_from_base_sensor(base, sensor)


def collect_episode(args, policy: str, init: int, rate: float, seed: int, model: Optional[MutualRadarNet], adapt) -> Tuple[List[Transition], pd.DataFrame]:
    env_args = make_base_args(args)
    env_cfg = env_cfg_for(float(rate), env_args)
    eng = build_env(_DummyPlanner(), int(init), MAXT, int(seed), int(args.window_ms), env_cfg)
    eng.reset(seed=int(seed))
    heuristic = None
    model_planner = None
    if policy == "edf":
        heuristic = EDFPlanner(MAXT)
    elif policy == "est":
        heuristic = ESTPlanner(MAXT)
    elif policy == "model" and model is not None:
        model_planner = MutualRadarDirectPlanner(
            model,
            direct_mode="branch",
            sensor_action_mode="explicit_head",
            cache_encoder=True,
            simulate_state=True,
        )

    rows: List[Transition] = []
    window_rows = []
    search_debt_ms = 0.0
    cumulative_reward = 0.0
    try:
        for window_idx in range(int(args.windows)):
            if bool(eng.term_buf[0]):
                break
            selected: set[int] = set()
            elapsed = 0.0
            search_count = 0
            track_count = 0
            last_action = -1
            window_reward = 0.0
            executed_count = 0
            search_actions = 0
            while elapsed < float(args.window_ms) and not bool(eng.term_buf[0]):
                obs = get_obs(eng, search_debt_ms)
                toks = tokenize(adapt, obs, selected=selected, search_count=search_count)
                slot = slot_features(obs, elapsed, search_count, track_count, last_action, float(args.window_ms))
                if policy in {"edf", "est"}:
                    plan = heuristic.plan(obs, budget_ms=float(args.window_ms) - elapsed)
                    action = int(plan[0]) if plan else xs_s_search_action(MAXT)
                elif policy == "model" and model is not None:
                    action = model_first_action(model_planner, obs, budget_ms=float(args.window_ms) - elapsed)
                    if random.random() < float(args.explore_eps):
                        action = random_valid_action(obs, p_search=float(args.random_search_prob))
                else:
                    action = random_valid_action(obs, p_search=float(args.random_search_prob))

                reward, dt, executed = execute_first_valid_action(eng, [action], float(args.window_ms) - elapsed)
                if executed is None or dt <= 0.0:
                    break
                executed = int(executed)
                base, sensor = xs_decode_action(executed, MAXT)
                if int(base) == 0:
                    search_debt_ms = 0.0
                    search_count += 1
                    search_actions += 1
                else:
                    search_debt_ms += float(dt)
                    selected.add(int(base))
                    track_count += 1
                elapsed += float(dt)
                cumulative_reward += float(reward)
                window_reward += float(reward)
                executed_count += 1
                last_action = int(base)
                next_obs = get_obs(eng, search_debt_ms)
                next_toks = tokenize(adapt, next_obs, selected=selected, search_count=search_count)
                next_slot = slot_features(next_obs, elapsed, search_count, track_count, last_action, float(args.window_ms))
                action_flat, base_action, sensor_id = flat_action_from_physical(executed, MAXT)
                rows.append(
                    Transition(
                        tokens=toks,
                        slot=slot,
                        next_tokens=next_toks,
                        next_slot=next_slot,
                        action_flat=action_flat,
                        base_action=base_action,
                        sensor=sensor_id,
                        reward=float(reward),
                        ret=0.0,
                        done=float(bool(eng.term_buf[0])),
                        policy=policy,
                        init=int(init),
                        rate=float(rate),
                        seed=int(seed),
                    )
                )
            obs_end = get_obs(eng, search_debt_ms)
            active = np.asarray(obs_end["active_mask"], dtype=bool)
            tracked = active & (np.asarray(obs_end["t_deadline"], dtype=np.float32) >= 0.0)
            window_rows.append(
                {
                    "method": policy,
                    "init": int(init),
                    "rate": float(rate),
                    "seed": int(seed),
                    "window": int(window_idx),
                    "window_reward": float(window_reward),
                    "cumulative_reward": float(cumulative_reward),
                    "search_fraction": float(search_actions / max(1, executed_count)),
                    "executed_actions": int(executed_count),
                    "active_targets": int(np.sum(active)),
                    "tracked_targets": int(np.sum(tracked)),
                    "search_debt_ms": float(search_debt_ms),
                }
            )
        G = 0.0
        for tr in reversed(rows):
            G = float(tr.reward) + float(args.gamma) * G
            tr.ret = G
    finally:
        eng.close()
    return rows, pd.DataFrame(window_rows)


def save_dataset(transitions: Sequence[Transition], path: Path) -> None:
    payload = {
        "tokens": np.stack([t.tokens for t in transitions]).astype(np.float32),
        "slot": np.stack([t.slot for t in transitions]).astype(np.float32),
        "next_tokens": np.stack([t.next_tokens for t in transitions]).astype(np.float32),
        "next_slot": np.stack([t.next_slot for t in transitions]).astype(np.float32),
        "action_flat": np.asarray([t.action_flat for t in transitions], dtype=np.int64),
        "base_action": np.asarray([t.base_action for t in transitions], dtype=np.int64),
        "sensor": np.asarray([t.sensor for t in transitions], dtype=np.int64),
        "reward": np.asarray([t.reward for t in transitions], dtype=np.float32),
        "ret": np.asarray([t.ret for t in transitions], dtype=np.float32),
        "done": np.asarray([t.done for t in transitions], dtype=np.float32),
        "policy": np.asarray([t.policy for t in transitions]),
        "init": np.asarray([t.init for t in transitions], dtype=np.int64),
        "rate": np.asarray([t.rate for t in transitions], dtype=np.float32),
        "seed": np.asarray([t.seed for t in transitions], dtype=np.int64),
    }
    torch.save(payload, path)


class LatentDynamics(nn.Module):
    def __init__(self, d_model: int = 96, action_count: int = 2 * (MAXT + 1), hidden: int = 256):
        super().__init__()
        self.action_emb = nn.Embedding(action_count, d_model)
        self.trunk = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.delta = nn.Linear(hidden, d_model)
        self.reward = nn.Linear(hidden, 1)
        self.value = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.policy = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, action_count))

    def forward(self, latent: torch.Tensor, action_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(torch.cat([latent, self.action_emb(action_flat)], dim=-1))
        next_latent = latent + 0.25 * self.delta(h)
        reward = self.reward(h).squeeze(-1)
        return next_latent, reward

    def root(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.policy(latent), self.value(latent).squeeze(-1)


class ModifiedMuZero(nn.Module):
    def __init__(self, base: MutualRadarNet, d_model: int = 96):
        super().__init__()
        self.base = base
        self.g = LatentDynamics(d_model=d_model)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        cls, _, _, _ = self.base.encode_tokens(tokens)
        return cls

    def predict(self, tokens: torch.Tensor, slot: torch.Tensor):
        return self.base.forward_with_sensor(tokens, slot)


def train_modified(args) -> Path:
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    base = load_model(make_base_args(args)).to(device)
    if bool(args.freeze_encoder):
        for name, p in base.named_parameters():
            if name.startswith("encoder.") or name.startswith("token_proj.") or name == "cls_token":
                p.requires_grad_(False)
    model = ModifiedMuZero(base, d_model=int(args.d_model)).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=float(args.weight_decay))
    n = int(data["tokens"].shape[0])
    ret = torch.as_tensor(data["ret"], dtype=torch.float32)
    reward = torch.as_tensor(data["reward"], dtype=torch.float32)
    value_scale = float(torch.quantile(ret.abs(), 0.90).clamp_min(1.0).item())
    reward_scale = float(torch.quantile(reward.abs(), 0.90).clamp_min(1.0).item())
    history = []
    for step in range(int(args.train_steps)):
        idx = torch.randint(0, n, (int(args.batch_size),))
        tokens = torch.as_tensor(data["tokens"][idx], dtype=torch.float32, device=device)
        slot = torch.as_tensor(data["slot"][idx], dtype=torch.float32, device=device)
        next_tokens = torch.as_tensor(data["next_tokens"][idx], dtype=torch.float32, device=device)
        action_flat = torch.as_tensor(data["action_flat"][idx], dtype=torch.long, device=device)
        base_action = torch.as_tensor(data["base_action"][idx], dtype=torch.long, device=device)
        sensor = torch.as_tensor(data["sensor"][idx], dtype=torch.long, device=device)
        r_t = torch.as_tensor(data["reward"][idx], dtype=torch.float32, device=device) / reward_scale
        ret_t = torch.as_tensor(data["ret"][idx], dtype=torch.float32, device=device) / value_scale

        type_logit, track_logits, value, _, _, sensor_logits, _ = model.predict(tokens, slot)
        is_search = (base_action == 0).float()
        type_loss = F.binary_cross_entropy_with_logits(type_logit, is_search)
        label_track_logits = track_logits[torch.arange(base_action.shape[0], device=device), base_action]
        valid_track_label = torch.isfinite(label_track_logits) & (label_track_logits > -1e8)
        has_track = (base_action > 0) & valid_track_label
        track_loss = torch.zeros((), device=device)
        if bool(has_track.any()):
            track_loss = F.cross_entropy(track_logits[has_track], base_action[has_track])
        sensor_loss = F.cross_entropy(sensor_logits[torch.arange(sensor.shape[0], device=device), base_action], sensor)
        value_loss = F.mse_loss(torch.tanh(value / value_scale), torch.tanh(ret_t))

        latent = model.encode(tokens)
        with torch.no_grad():
            next_latent_target = model.encode(next_tokens)
        next_latent_pred, reward_pred = model.g(latent, action_flat)
        latent_policy_logits, latent_value = model.g.root(latent)
        dyn_loss = F.mse_loss(F.normalize(next_latent_pred, dim=-1), F.normalize(next_latent_target, dim=-1))
        reward_loss = F.mse_loss(reward_pred, r_t)
        latent_policy_loss = F.cross_entropy(latent_policy_logits, action_flat)
        latent_value_loss = F.mse_loss(torch.tanh(latent_value), torch.tanh(ret_t))
        loss = (
            float(args.policy_weight) * (type_loss + track_loss + sensor_loss)
            + float(args.value_weight) * value_loss
            + float(args.dynamics_weight) * dyn_loss
            + float(args.reward_weight) * reward_loss
            + float(args.latent_policy_weight) * latent_policy_loss
            + float(args.latent_value_weight) * latent_value_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % max(1, int(args.log_every)) == 0 or step == int(args.train_steps) - 1:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "type": float(type_loss.detach().cpu()),
                "track": float(track_loss.detach().cpu()),
                "sensor": float(sensor_loss.detach().cpu()),
                "value": float(value_loss.detach().cpu()),
                "dyn": float(dyn_loss.detach().cpu()),
                "reward": float(reward_loss.detach().cpu()),
                "latent_policy": float(latent_policy_loss.detach().cpu()),
                "latent_value": float(latent_value_loss.detach().cpu()),
                "value_scale": value_scale,
                "reward_scale": reward_scale,
            }
            history.append(row)
            print(row, flush=True)
    out = OUT / f"{args.tag}_state.pt"
    torch.save({"model": model.base.state_dict(), "dynamics": model.g.state_dict(), "args": vars(args), "value_scale": value_scale, "reward_scale": reward_scale}, out)
    pd.DataFrame(history).to_csv(OUT / f"{args.tag}_train_log.csv", index=False)
    return out


def eval_checkpoint(args, ckpt: Path) -> pd.DataFrame:
    eval_args = make_base_args(args)
    eval_args.ckpt = str(ckpt)
    model = load_model(eval_args)
    model.eval()
    modified = None
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "dynamics" in payload:
        modified = ModifiedMuZero(model, d_model=int(args.d_model))
        modified.g.load_state_dict(payload["dynamics"], strict=False)
        modified.to(torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")).eval()
    rows = []
    planners = [
        ("ModifiedMuZero_PV_direct", lambda: MutualRadarDirectPlanner(model, direct_mode="branch", sensor_action_mode="explicit_head", cache_encoder=True)),
        ("EDF", lambda: EDFPlanner(MAXT)),
        ("EST", lambda: ESTPlanner(MAXT)),
    ]
    if modified is not None:
        planners.insert(1, ("ModifiedMuZero_latent1", lambda modified=modified, args=args: LatentOneStepPlanner(modified, args)))
    accepted_ckpt = str(getattr(args, "accepted_ckpt", "") or "")
    if accepted_ckpt:
        old_args = make_base_args(args)
        old_args.ckpt = accepted_ckpt
        old = load_model(old_args)
        planners.insert(1, ("Accepted_PV_direct", lambda old=old: MutualRadarDirectPlanner(old, direct_mode="branch", sensor_action_mode="explicit_head", cache_encoder=True)))
    for init in parse_ints(args.eval_initials):
        for rate in parse_floats(args.eval_rates):
            env_cfg = env_cfg_for(float(rate), eval_args)
            for seed in parse_ints(args.eval_seeds):
                for name, factory in planners:
                    t0 = time.perf_counter()
                    df, _ = run_fixed(factory(), name, int(init), MAXT, int(seed), int(args.windows), int(args.window_ms), env_cfg)
                    elapsed = time.perf_counter() - t0
                    if df.empty:
                        continue
                    rows.append(
                        {
                            "method": name,
                            "init": init,
                            "rate": rate,
                            "seed": seed,
                            "reward": float(df["window_reward"].mean()),
                            "total_reward": float(df["cumulative_reward"].iloc[-1]),
                            "search": float(df["search_fraction"].mean()),
                            "latency_ms": float(df["planning_ms_per_decision"].mean()),
                            "wall_s": float(elapsed),
                            "tracked_final": float(df["tracked_targets"].iloc[-1]),
                            "active_final": float(df["active_targets"].iloc[-1]),
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"{args.tag}_eval_raw.csv", index=False)
    summary = out.groupby("method").agg(
        reward=("reward", "mean"),
        total_reward=("total_reward", "mean"),
        search=("search", "mean"),
        latency_ms=("latency_ms", "mean"),
        tracked_final=("tracked_final", "mean"),
        active_final=("active_final", "mean"),
        n=("reward", "count"),
    ).reset_index().sort_values("reward", ascending=False)
    summary.to_csv(OUT / f"{args.tag}_eval_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    return summary


def legal_physical_actions(obs: dict, max_trackers: int = MAXT) -> List[int]:
    out = [xs_s_search_action(max_trackers)]
    if obs.get("enable_x_band", 0) and obs.get("x_band_busy_ms", 0.0) <= 0.0:
        out.append(xs_x_search_action(max_trackers))
    active = np.asarray(obs["active_mask"], dtype=bool)
    deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
    ranges = np.asarray(obs.get("target_range", np.zeros(max_trackers, dtype=np.float32)), dtype=np.float32)
    s_free = obs.get("s_band_busy_ms", 0.0) <= 0.0
    x_free = bool(obs.get("enable_x_band", 0)) and obs.get("x_band_busy_ms", 0.0) <= 0.0
    for i in np.where(active & (deadline >= 0.0))[0]:
        base = int(i) + 1
        r = float(ranges[i]) if i < len(ranges) else 0.0
        if s_free and 10_000_000.0 < r < 184_000_000.0:
            out.append(xs_s_track_action(base, max_trackers))
        if x_free and 5_000_000.0 < r < 100_000_000.0:
            out.append(xs_x_track_action(base, max_trackers))
    return out


class LatentOneStepPlanner:
    def __init__(self, model: ModifiedMuZero, args):
        self.model = model.eval()
        self.args = args
        self.adapt = adapter()

    def plan(self, obs, budget_ms=200):
        device = next(self.model.parameters()).device
        selected: set[int] = set()
        plan: List[int] = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last_action = -1
        while elapsed < float(budget_ms) and len(plan) < 64:
            tokens = torch.from_numpy(tokenize(self.adapt, obs, selected=selected, search_count=search_count)).float().unsqueeze(0).to(device)
            with torch.inference_mode():
                latent = self.model.encode(tokens)
                root_logits, _ = self.model.g.root(latent)
            actions = legal_physical_actions(obs, MAXT)
            if not actions:
                break
            flat = torch.tensor([flat_action_from_physical(a, MAXT)[0] for a in actions], dtype=torch.long, device=device)
            with torch.inference_mode():
                next_latent, pred_reward = self.model.g(latent.expand(flat.shape[0], -1), flat)
                _, next_value = self.model.g.root(next_latent)
                prior_bonus = float(getattr(self.args, "latent_prior_weight", 0.05)) * F.log_softmax(root_logits[0, flat], dim=0)
                score = pred_reward + float(getattr(self.args, "gamma", 0.997)) * next_value + prior_bonus
                best = int(torch.argmax(score).item())
            action = int(actions[best])
            base, _sensor = xs_decode_action(action, MAXT)
            plan.append(action)
            if int(base) == 0:
                dt = SEARCH_DWELL_MS
                search_count += 1
            else:
                dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
                dt = float(dwell[int(base) - 1]) if 1 <= int(base) <= len(dwell) else SEARCH_DWELL_MS
                selected.add(int(base))
                track_count += 1
            elapsed += max(1.0, float(dt))
            last_action = int(base)
        return plan


def collect_dataset(args) -> Path:
    seed_all(args.seed)
    adapt = adapter()
    model = load_model(make_base_args(args)) if args.init_ckpt else None
    if model is not None:
        model.eval()
    transitions: List[Transition] = []
    trace_frames = []
    policies = [x.strip().lower() for x in str(args.collect_policies).split(",") if x.strip()]
    for seed in parse_ints(args.train_seeds):
        for init in parse_ints(args.train_initials):
            for rate in parse_floats(args.train_rates):
                for policy in policies:
                    episode_seed = int(seed) + 1009 * policies.index(policy)
                    eps, trace = collect_episode(args, policy, init, rate, episode_seed, model, adapt)
                    transitions.extend(eps)
                    if not trace.empty:
                        trace_frames.append(trace)
                    print(f"collected policy={policy} init={init} rate={rate} seed={episode_seed} transitions={len(eps)}", flush=True)
    out = OUT / f"{args.tag}_dataset.pt"
    save_dataset(transitions, out)
    if trace_frames:
        pd.concat(trace_frames, ignore_index=True).to_csv(OUT / f"{args.tag}_collect_trace.csv", index=False)
    stats = pd.DataFrame(
        [{
            "transitions": len(transitions),
            "ret_mean": float(np.mean([t.ret for t in transitions])) if transitions else 0.0,
            "ret_p90_abs": float(np.percentile(np.abs([t.ret for t in transitions]), 90)) if transitions else 0.0,
            "search_frac": float(np.mean([t.base_action == 0 for t in transitions])) if transitions else 0.0,
        }]
    )
    stats.to_csv(OUT / f"{args.tag}_dataset_stats.csv", index=False)
    print(stats.to_string(index=False), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["collect", "train", "eval", "all"], default="all")
    ap.add_argument("--tag", default="modified_muzero_smoke")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--init-ckpt", default=r"C:\Users\yousi\Downloads\radar_outputs\alphazero_orthodox\paper_factorized_pv_current_best.pt")
    ap.add_argument("--accepted-ckpt", default=r"C:\Users\yousi\Downloads\radar_outputs\alphazero_orthodox\paper_factorized_pv_current_best.pt")
    ap.add_argument("--eval-ckpt", default="")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--head-arch", default="baseline")
    ap.add_argument("--env-mode", default="radarxs_original")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--window-ms", type=int, default=200)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--train-initials", default="20,60,100")
    ap.add_argument("--train-rates", default="0,4,8")
    ap.add_argument("--train-seeds", default="901")
    ap.add_argument("--collect-policies", default="model,edf,est,random")
    ap.add_argument("--explore-eps", type=float, default=0.10)
    ap.add_argument("--random-search-prob", type=float, default=0.20)
    ap.add_argument("--train-steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--dynamics-weight", type=float, default=1.0)
    ap.add_argument("--reward-weight", type=float, default=1.0)
    ap.add_argument("--latent-policy-weight", type=float, default=0.25)
    ap.add_argument("--latent-value-weight", type=float, default=1.0)
    ap.add_argument("--latent-prior-weight", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-initials", default="20,60,100")
    ap.add_argument("--eval-rates", default="0,4,8")
    ap.add_argument("--eval-seeds", default="931,932")
    args = ap.parse_args()
    seed_all(args.seed)
    dataset = Path(args.dataset) if args.dataset else OUT / f"{args.tag}_dataset.pt"
    if args.mode in {"collect", "all"}:
        dataset = collect_dataset(args)
        args.dataset = str(dataset)
    if args.mode in {"train", "all"}:
        if not args.dataset:
            args.dataset = str(dataset)
        ckpt = train_modified(args)
    else:
        ckpt = Path(args.eval_ckpt or args.init_ckpt)
    if args.mode in {"eval", "all"}:
        eval_checkpoint(args, ckpt)


if __name__ == "__main__":
    main()
