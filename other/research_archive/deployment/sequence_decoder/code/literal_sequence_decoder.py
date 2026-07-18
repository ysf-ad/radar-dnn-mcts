from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import literal_muzero_radar_smoke as lmz


class WindowSequenceDecoder(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, seq_len: int = 32, d_model: int = 128):
        super().__init__()
        self.seq_len = int(seq_len)
        self.action_dim = int(action_dim)
        self.obs_proj = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim > 2:
            obs = obs.view(obs.shape[0], -1)
        ctx = self.obs_proj(obs)
        pos = self.pos.unsqueeze(0).expand(obs.shape[0], -1, -1)
        ctx = ctx[:, None, :].expand(-1, self.seq_len, -1)
        return self.head(torch.cat([ctx, pos], dim=-1))


class AutoregressiveWindowDecoder(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, seq_len: int = 32, d_model: int = 128):
        super().__init__()
        self.seq_len = int(seq_len)
        self.action_dim = int(action_dim)
        self.obs_proj = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.action_emb = nn.Embedding(action_dim + 1, d_model)
        self.pos = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        self.gru = nn.GRUCell(2 * d_model, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
        )

    def forward(self, obs: torch.Tensor, teacher_actions: torch.Tensor | None = None) -> torch.Tensor:
        if obs.ndim > 2:
            obs = obs.view(obs.shape[0], -1)
        h = self.obs_proj(obs)
        batch = obs.shape[0]
        prev = torch.full((batch,), self.action_dim, dtype=torch.long, device=obs.device)
        logits = []
        for t in range(self.seq_len):
            inp = torch.cat([self.action_emb(prev), self.pos[t].unsqueeze(0).expand(batch, -1)], dim=-1)
            h = self.gru(inp, h)
            step_logits = self.head(h)
            logits.append(step_logits)
            if teacher_actions is not None:
                teacher = teacher_actions[:, t].clamp(min=0)
                prev = torch.where(teacher_actions[:, t] >= 0, teacher, step_logits.argmax(dim=-1))
            else:
                prev = step_logits.argmax(dim=-1)
        return torch.stack(logits, dim=1)


class ActionAwareOneStepDecoder(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, seq_len: int = 1, d_model: int = 128):
        super().__init__()
        self.seq_len = int(seq_len)
        self.action_dim = int(action_dim)
        self.max_targets = lmz.OBS_TARGETS
        self.target_dim = 7
        self.ctx_proj = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.target_proj = nn.Sequential(
            nn.LayerNorm(self.target_dim),
            nn.Linear(self.target_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.search_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.score = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim > 2:
            obs = obs.view(obs.shape[0], -1)
        ctx = self.ctx_proj(obs[:, :10])
        target_feats = obs[:, 10:].view(obs.shape[0], self.max_targets, self.target_dim)
        tgt = self.target_proj(target_feats[:, : self.action_dim - 1, :])
        search = self.search_token.view(1, 1, -1).expand(obs.shape[0], 1, -1)
        actions = torch.cat([search, tgt], dim=1)
        ctx_rep = ctx[:, None, :].expand(-1, actions.shape[1], -1)
        logits = self.score(torch.cat([ctx_rep, actions], dim=-1)).squeeze(-1)
        return logits[:, None, :]


class FactorizedOneStepDecoder(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, seq_len: int = 1, d_model: int = 128):
        super().__init__()
        self.seq_len = int(seq_len)
        self.action_dim = int(action_dim)
        self.max_targets = lmz.OBS_TARGETS
        self.target_dim = 7
        self.ctx_proj = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.target_proj = nn.Sequential(
            nn.LayerNorm(self.target_dim),
            nn.Linear(self.target_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.type_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        self.target_head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward_parts(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.ndim > 2:
            obs = obs.view(obs.shape[0], -1)
        ctx = self.ctx_proj(obs[:, :10])
        target_feats = obs[:, 10:].view(obs.shape[0], self.max_targets, self.target_dim)
        tgt = self.target_proj(target_feats[:, : self.action_dim - 1, :])
        type_logits = self.type_head(ctx)
        ctx_rep = ctx[:, None, :].expand(-1, tgt.shape[1], -1)
        target_logits = self.target_head(torch.cat([ctx_rep, tgt], dim=-1)).squeeze(-1)
        active = target_feats[:, : self.action_dim - 1, 0] > 0.5
        target_logits = target_logits.masked_fill(~active, -1e9)
        search_logit = type_logits[:, 0:1]
        track_logits = type_logits[:, 1:2] + target_logits
        logits = torch.cat([search_logit, track_logits], dim=1)
        return logits, type_logits, target_logits

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits, _type_logits, _target_logits = self.forward_parts(obs)
        return logits[:, None, :]


class ActionAttentionOneStepDecoder(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, seq_len: int = 1, d_model: int = 128):
        super().__init__()
        self.seq_len = int(seq_len)
        self.action_dim = int(action_dim)
        self.max_targets = lmz.OBS_TARGETS
        self.target_dim = 7
        self.ctx_proj = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.target_proj = nn.Sequential(
            nn.LayerNorm(self.target_dim),
            nn.Linear(self.target_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.search_token = nn.Parameter(torch.randn(d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=2 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mix = nn.TransformerEncoder(layer, num_layers=1)
        self.score = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim > 2:
            obs = obs.view(obs.shape[0], -1)
        ctx = self.ctx_proj(obs[:, :10])
        target_feats = obs[:, 10:].view(obs.shape[0], self.max_targets, self.target_dim)
        tgt = self.target_proj(target_feats[:, : self.action_dim - 1, :])
        search = self.search_token.view(1, 1, -1).expand(obs.shape[0], 1, -1)
        actions = torch.cat([search, tgt], dim=1)
        active = target_feats[:, : self.action_dim - 1, 0] > 0.5
        padding = torch.cat(
            [torch.zeros(obs.shape[0], 1, dtype=torch.bool, device=obs.device), ~active],
            dim=1,
        )
        mixed = self.mix(actions, src_key_padding_mask=padding)
        ctx_rep = ctx[:, None, :].expand(-1, mixed.shape[1], -1)
        logits = self.score(torch.cat([ctx_rep, mixed], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(padding, -1e9)
        return logits[:, None, :]


def make_decoder(args: argparse.Namespace, obs_dim: int, device: torch.device):
    decoder_type = str(getattr(args, "decoder_type", "independent"))
    if decoder_type == "gru":
        cls = AutoregressiveWindowDecoder
    elif decoder_type == "action_aware":
        cls = ActionAwareOneStepDecoder
    elif decoder_type == "factorized":
        cls = FactorizedOneStepDecoder
    elif decoder_type == "action_attention":
        cls = ActionAttentionOneStepDecoder
    else:
        cls = WindowSequenceDecoder
    return cls(obs_dim, args.action_ranks + 1, args.seq_len, args.d_model).to(device)


def make_args(ns: argparse.Namespace) -> argparse.Namespace:
    args = argparse.Namespace(**vars(ns))
    args.initial = 20
    args.rate = 2.0
    args.action_ranks = 64
    args.rank_action_space = True
    args.reward_scale = 10.0
    args.max_actions_per_window = ns.seq_len
    args.windows = ns.windows
    args.network = "factorized_fullyconnected"
    args.encoding_size = 64
    args.gamma = 0.997
    args.support_size = 20
    args.root_dirichlet_alpha = 0.25
    args.root_exploration_fraction = 0.25
    args.factorized_search_logit_offset = 0.0
    args.service_reward = 0.2
    args.window_service_reward = 0.5
    args.window_tracked_reward = 0.02
    args.discovery_reward = 0.5
    args.search_refresh_reward = 0.2
    args.search_debt_penalty = 0.5
    args.search_debt_penalty_mode = "time"
    args.terminal_search_debt_penalty = 0.0
    args.penalize_hidden_targets = 0
    args.simulations = 0
    args.temperature = 0.0
    return args


def load_teacher(args: argparse.Namespace):
    cfg = lmz.Config(args)
    model = lmz.models.MuZeroNetwork(cfg)
    payload = torch.load(Path(args.teacher_ckpt), map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model


def collect_dataset(args: argparse.Namespace, teacher, rollout_model=None) -> tuple[np.ndarray, np.ndarray]:
    device = None
    if rollout_model is not None:
        device = next(rollout_model.parameters()).device
    cache = Path(args.out_dir) / "sequence_teacher_data.npz"
    if rollout_model is None and cache.exists() and not args.recollect:
        data = np.load(cache)
        return data["x"], data["y"]
    xs, ys = [], []
    cells = [(i, r) for i in args.initials for r in args.rates]
    for ep in range(args.episodes):
        initial, rate = cells[ep % len(cells)]
        args.initial, args.rate = int(initial), float(rate)
        game = lmz.LiteralRadarGame(
            args.initial,
            args.rate,
            args.seed + 10000 + ep,
            args.windows,
            args.reward_scale,
            args.action_ranks,
            lmz.shaping_from_args(args),
            True,
        )
        obs = game.reset()
        try:
            while game.window_index < args.windows and not bool(game.eng.term_buf[0]):
                window_start = int(game.window_index)
                window_obs = []
                window_actions = []
                while (
                    game.window_index == window_start
                    and len(window_actions) < args.seq_len
                    and not bool(game.eng.term_buf[0])
                ):
                    window_obs.append(obs.copy())
                    legal = game.legal_actions()
                    label_action = lmz.select_direct_policy_action(teacher, obs, legal, 0.0)
                    if rollout_model is None:
                        action = int(label_action)
                    else:
                        action = int(decode_plan(rollout_model, obs, legal, args, device)[0])
                    window_actions.append(int(label_action))
                    obs, _reward, _done = game.step(action)
                    if _done:
                        break
                if window_actions:
                    starts = [0]
                    collect_chunk = int(getattr(args, "collect_chunk_size", 0) or 0)
                    if collect_chunk > 0:
                        starts = list(range(0, len(window_actions), collect_chunk))
                    for start in starts:
                        y = np.full((args.seq_len,), -100, dtype=np.int64)
                        future = window_actions[start : start + args.seq_len]
                        y[: len(future)] = np.asarray(future, dtype=np.int64)
                        xs.append(window_obs[start].reshape(-1).astype(np.float32))
                        ys.append(y)
            print("collect", ep, initial, rate, len(xs), flush=True)
        finally:
            game.close()
    x = np.stack(xs).astype(np.float32)
    y = np.stack(ys).astype(np.int64)
    if rollout_model is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, x=x, y=y)
    return x, y


def train_decoder(args: argparse.Namespace, x: np.ndarray, y: np.ndarray) -> WindowSequenceDecoder:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = make_decoder(args, x.shape[1], device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    logs = []
    n = x.shape[0]
    for step in range(args.train_steps):
        idx = np.random.randint(0, n, size=min(args.batch_size, n))
        xb = torch.from_numpy(x[idx]).float().to(device)
        yb = torch.from_numpy(y[idx]).long().to(device)
        logits = model(xb, yb) if isinstance(model, AutoregressiveWindowDecoder) else model(xb)
        flat_logits = logits.view(-1, args.action_ranks + 1)
        flat_y = yb.view(-1)
        search_weight = float(getattr(args, "search_loss_weight", 1.0))
        if search_weight != 1.0:
            class_weight = torch.ones(args.action_ranks + 1, dtype=torch.float32, device=device)
            class_weight[0] = search_weight
        else:
            class_weight = None
        loss = F.cross_entropy(flat_logits, flat_y, ignore_index=-100, weight=class_weight)
        if isinstance(model, FactorizedOneStepDecoder):
            _combined_logits, type_logits, target_logits = model.forward_parts(xb)
            labels = yb[:, 0]
            valid = labels >= 0
            if bool(valid.any()):
                type_targets = torch.where(labels[valid] == 0, torch.zeros_like(labels[valid]), torch.ones_like(labels[valid]))
                type_loss = F.cross_entropy(type_logits[valid], type_targets)
                loss = loss + float(getattr(args, "factorized_type_loss_weight", 0.5)) * type_loss
                track_valid = valid & (labels > 0)
                if bool(track_valid.any()):
                    target_loss = F.cross_entropy(target_logits[track_valid], labels[track_valid] - 1)
                    loss = loss + float(getattr(args, "factorized_target_loss_weight", 0.5)) * target_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        if step % 50 == 0:
            with torch.no_grad():
                pred = logits.argmax(-1)
                valid = yb >= 0
                acc = (pred[valid] == yb[valid]).float().mean().item() if bool(valid.any()) else 0.0
                search_pred = (pred[valid] == 0).float().mean().item() if bool(valid.any()) else 0.0
                search_label = (yb[valid] == 0).float().mean().item() if bool(valid.any()) else 0.0
            row = {"step": step, "loss": float(loss.detach().cpu()), "acc": acc, "search_pred": search_pred, "search_label": search_label}
            logs.append(row)
            print(row, flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(logs).to_csv(out / "sequence_train_log.csv", index=False)
    torch.save(model.state_dict(), out / "sequence_decoder.pt")
    model.eval()
    return model


def decode_plan(model: WindowSequenceDecoder, obs: np.ndarray, legal_actions: list[int], args, device) -> list[int]:
    x = torch.from_numpy(obs.reshape(1, -1)).float().to(device)
    with torch.inference_mode():
        logits = model(x)[0].detach().cpu().numpy()
    flat_obs = obs.reshape(-1)
    debt_norm = float(flat_obs[2]) if flat_obs.size > 2 else 0.0
    active_norm = float(flat_obs[0]) if flat_obs.size > 0 else 0.0
    tracked_norm = float(flat_obs[1]) if flat_obs.size > 1 else 0.0
    load_gap = max(0.0, active_norm - tracked_norm)
    adaptive_bias = (
        float(getattr(args, "search_bias", 0.0))
        + float(getattr(args, "search_debt_bias", 0.0)) * debt_norm
        + float(getattr(args, "search_load_gap_bias", 0.0)) * load_gap
    )
    logits[:, 0] += adaptive_bias
    plan = []
    legal = set(int(a) for a in legal_actions)
    for row in logits:
        action = int(np.argmax(row))
        plan.append(action if action in legal else 0)
    return plan


def eval_decoder(args: argparse.Namespace, model: WindowSequenceDecoder) -> pd.DataFrame:
    device = next(model.parameters()).device
    rows = []
    for initial in args.initials:
        for rate in args.rates:
            args.initial, args.rate = int(initial), float(rate)
            game = lmz.LiteralRadarGame(
                args.initial,
                args.rate,
                args.seed + 90000 + int(initial) * 10 + int(rate),
                args.windows,
                args.reward_scale,
                args.action_ranks,
                lmz.shaping_from_args(args),
                True,
            )
            obs = game.reset()
            total = 0.0
            plan_ms = []
            try:
                while game.window_index < args.windows and not bool(game.eng.term_buf[0]):
                    window_start = int(game.window_index)
                    executed_in_window = 0
                    while game.window_index == window_start and not bool(game.eng.term_buf[0]):
                        t0 = time.perf_counter()
                        plan = decode_plan(model, obs, game.legal_actions(), args, device)
                        chunk = int(getattr(args, "chunk_size", 0) or args.seq_len)
                        if chunk > 0:
                            plan = plan[:chunk]
                        plan_ms.append(1000.0 * (time.perf_counter() - t0))
                        if not plan:
                            break
                        for action in plan:
                            if game.window_index != window_start or bool(game.eng.term_buf[0]):
                                break
                            obs, reward, done = game.step(int(action))
                            total += float(reward)
                            executed_in_window += 1
                            if done:
                                break
                        if game.window_index != window_start or bool(game.eng.term_buf[0]):
                            break
                        if executed_in_window >= int(getattr(args, "eval_max_actions_per_window", args.seq_len)):
                            break
                metrics = game.metrics()
                rows.append(
                    {
                        "phase": "eval_only",
                        "initial": int(initial),
                        "rate": float(rate),
                        "scaled_total_reward": total,
                        "reward_per_window": total * args.reward_scale / max(1, args.windows),
                        "planning_ms_per_200ms_window": float(np.sum(plan_ms)) / max(1, args.windows),
                        "planning_ms_per_window": float(np.mean(plan_ms)) if plan_ms else 0.0,
                        "moves": int(game.step_count),
                        **metrics,
                    }
                )
            finally:
                game.close()
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out_dir) / "sequence_eval9_100w.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="CreateValid1/results/literal_sequence_decoder")
    ap.add_argument("--teacher-ckpt", default="CreateValid1/results/literal_muzero_balanced_teacher1000_searchsample15_dagger2_100w.pt")
    ap.add_argument("--episodes", type=int, default=27)
    ap.add_argument("--windows", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--action-ranks", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--decoder-type", choices=["independent", "gru", "action_aware", "factorized", "action_attention"], default="independent")
    ap.add_argument("--train-steps", type=int, default=600)
    ap.add_argument("--dagger-rounds", type=int, default=0)
    ap.add_argument("--dagger-steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--search-bias", type=float, default=0.0)
    ap.add_argument("--search-debt-bias", type=float, default=0.0)
    ap.add_argument("--search-load-gap-bias", type=float, default=0.0)
    ap.add_argument("--search-loss-weight", type=float, default=1.0)
    ap.add_argument("--factorized-type-loss-weight", type=float, default=0.5)
    ap.add_argument("--factorized-target-loss-weight", type=float, default=0.5)
    ap.add_argument("--chunk-size", type=int, default=0, help="Replan after this many decoded actions; 0 uses one full-window plan.")
    ap.add_argument("--collect-chunk-size", type=int, default=0, help="Add teacher samples from every N actions inside each window; 0 stores only window starts.")
    ap.add_argument("--eval-max-actions-per-window", type=int, default=0, help="Safety cap for executed actions per window; 0 uses seq_len.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--recollect", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--seq-ckpt", default="")
    ns = ap.parse_args()
    ns.initials = [20, 40, 60]
    ns.rates = [2.0, 3.0, 4.0]
    random.seed(ns.seed)
    np.random.seed(ns.seed)
    torch.manual_seed(ns.seed)
    args = make_args(ns)
    if int(getattr(args, "eval_max_actions_per_window", 0) or 0) <= 0:
        args.eval_max_actions_per_window = int(args.seq_len)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.eval_only:
        ckpt = Path(args.seq_ckpt) if args.seq_ckpt else out / "sequence_decoder.pt"
        model = make_decoder(args, lmz.OBS_DIM, device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
    else:
        teacher = load_teacher(args)
        x, y = collect_dataset(args, teacher)
        model = train_decoder(args, x, y)
        for round_idx in range(int(args.dagger_rounds)):
            rx, ry = collect_dataset(args, teacher, model)
            x = np.concatenate([x, rx], axis=0)
            y = np.concatenate([y, ry], axis=0)
            old_steps = int(args.train_steps)
            args.train_steps = int(args.dagger_steps)
            print({"dagger_round": round_idx, "dataset": int(x.shape[0]), "rollin": int(rx.shape[0])}, flush=True)
            model = train_decoder(args, x, y)
            args.train_steps = old_steps
    df = eval_decoder(args, model)
    summary = df.agg(
        {
            "reward_per_window": "mean",
            "search_ratio": "mean",
            "tracked_targets": "mean",
            "active_targets": "mean",
            "drop_pct_active": "mean",
            "mean_delay_active": "mean",
            "search_debt_end_ms": "mean",
            "planning_ms_per_200ms_window": "mean",
        }
    )
    summary.to_frame("value").to_csv(out / "sequence_summary.csv")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
