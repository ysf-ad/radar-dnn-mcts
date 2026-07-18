"""Compare factorized AlphaRadar against a flat-softmax AlphaRadar.

The factorized model asks:
    Search vs Track? then Which target?

The flat model asks:
    Which atomic action among [search, target_1, ..., target_N]?

Both use the same MCTS improvement loop and the same value/Q targets.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from alpharadar_foundation_train import (
    AlphaRadarDirectPlanner,
    AlphaRadarMCTSPlanner,
    AlphaRadarNet,
    RUN_OUT,
    ReplayBuffer,
    SearchTarget,
    eval_factory,
)
from continuous_utility_sweep import ContinuousUtilityPlanner
from final_radar_campaign import MAXT, build_env, get_obs, seedall
from hierarchical_policyq_train_eval import HierarchicalPolicyQTransformer, HierPolicyQPlanner
from hierarchical_sequence_transformer import SLOT_DIM, TOKEN_DIM, slot_features, tokenize
from hierarchical_window_transformer import HierarchicalWindowTransformer, HierWindowPlanner
from load_adaptive_train_eval import OUT, RES, load_model, make_env
from realistic_reward_retrain import adapter
from repaired_campaign_tools import EDFPlanner, ESTPlanner, SEARCH_DWELL_MS
from strict_window_report import execute_plan_until_budget


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)


class FlatAlphaRadarNet(nn.Module):
    """Flat action policy over [search, target_1, ..., target_N]."""

    def __init__(self, token_dim=TOKEN_DIM, slot_dim=SLOT_DIM, d_model=96, nhead=4, nlayers=2):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, d_model)
        self.slot_proj = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, d_model), nn.GELU())
        self.cls_token = nn.Parameter(torch.randn(d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.05,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.policy_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.q_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.value_head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, tokens: torch.Tensor, slot: torch.Tensor):
        token_active = tokens[:, :, 4] > 0.5
        token_active[:, 0] = True
        selected = tokens[:, :, 8] > 0.5
        valid = token_active & ~selected
        valid[:, 0] = True

        emb = self.token_proj(tokens)
        cls = self.cls_token.unsqueeze(0).unsqueeze(0).expand(tokens.shape[0], 1, -1)
        emb = torch.cat([cls, emb], dim=1)
        cls_valid = torch.ones((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device)
        out = self.encoder(emb, src_key_padding_mask=~torch.cat([cls_valid, token_active], dim=1))
        cls_out = out[:, 0, :]
        tok_out = out[:, 1:, :]
        slot_emb = self.slot_proj(slot)
        cls_rep = cls_out.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
        slot_rep = slot_emb.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
        ctx = torch.cat([tok_out, cls_rep, slot_rep], dim=-1)
        logits = self.policy_head(ctx).squeeze(-1).masked_fill(~valid, -1e9)
        q = self.q_head(ctx).squeeze(-1).masked_fill(~valid, 0.0)
        value = self.value_head(torch.cat([cls_out, slot_emb], dim=-1)).squeeze(-1)
        return logits, value, q


class FlatAlphaRadarMCTSPlanner(AlphaRadarMCTSPlanner):
    def __init__(self, model: FlatAlphaRadarNet, *args, q_scale=1.0, use_q_head=False, q_utility_weight=0.0, use_value_head=False, leaf_value_mix=0.0, **kwargs):
        super().__init__(
            AlphaRadarNet(),
            *args,
            q_scale=q_scale,
            use_q_head=use_q_head,
            q_utility_weight=q_utility_weight,
            use_value_head=use_value_head,
            leaf_value_mix=leaf_value_mix,
            **kwargs,
        )
        self.model = model.eval()

    def _priors_for_node(self, node, add_root_noise: bool = False):
        _, x, slot = self._features_for_node(node)
        with torch.inference_mode():
            logits, value, q = self.model(
                torch.from_numpy(x).float().unsqueeze(0),
                torch.from_numpy(slot).float().unsqueeze(0),
            )
        priors = torch.softmax(logits[0], dim=0).detach().cpu().numpy().astype(np.float32)
        node.nn_qvalues = q[0].detach().cpu().numpy().astype(np.float32) * self.q_scale
        node.nn_value = float(value[0].detach().cpu()) * self.q_scale
        valid = node.get_valid_actions()
        mask = np.zeros_like(priors)
        mask[np.asarray(valid, dtype=np.int64)] = 1.0
        priors *= mask
        if add_root_noise and len(valid) > 1:
            noise = np.random.dirichlet([self.root_dirichlet_alpha] * len(valid)).astype(np.float32)
            noisy = np.zeros_like(priors)
            noisy[np.asarray(valid, dtype=np.int64)] = noise
            priors = (1.0 - self.root_dirichlet_eps) * priors + self.root_dirichlet_eps * noisy
        s = float(np.sum(priors))
        if s <= 0:
            priors[np.asarray(valid, dtype=np.int64)] = 1.0 / max(1, len(valid))
        else:
            priors /= s
        node.nn_priors = priors
        return priors


class FlatAlphaRadarDirectPlanner:
    def __init__(self, model: FlatAlphaRadarNet, alpha=0.0):
        self.model = model.eval()
        self.adapt = adapter()
        self.alpha = float(alpha)

    def plan(self, obs, budget_ms=200):
        selected = set()
        plan: List[int] = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
        while elapsed < float(budget_ms) and len(plan) < 64:
            x = tokenize(self.adapt, obs, selected=selected, search_count=search_count)
            s = slot_features(obs, elapsed, search_count, track_count, last, float(budget_ms))
            with torch.inference_mode():
                logits, _, q = self.model(
                    torch.from_numpy(x).float().unsqueeze(0),
                    torch.from_numpy(s).float().unsqueeze(0),
                )
            score = logits[0].cpu().numpy() + self.alpha * q[0].cpu().numpy()
            a = int(np.argmax(score))
            if a == 0:
                dt = SEARCH_DWELL_MS
                search_count += 1
            else:
                selected.add(a)
                dt = float(dwell[a - 1]) if 1 <= a <= len(dwell) else SEARCH_DWELL_MS
                track_count += 1
            plan.append(a)
            elapsed += max(1.0, float(dt))
            last = a
        return plan if plan else [0]


def collect_flat_selfplay(model: FlatAlphaRadarNet, args):
    rows: List[SearchTarget] = []
    rewards = []
    init_choices = [int(x) for x in str(args.train_initials).split(",") if x]
    rate_choices = [float(x) for x in str(args.train_rates).split(",") if x]
    for ep in range(args.episodes_per_iter):
        init = random.choice(init_choices)
        rate = random.choice(rate_choices)
        seed = int(args.seed + 1000 * args.iteration + ep)
        seedall(seed)
        env = make_env(rate)
        planner = FlatAlphaRadarMCTSPlanner(model, env, rollouts=args.rollouts, c_puct=args.c_puct, expand_top_k=args.expand_top_k, training=True)
        eng = build_env(planner, init, MAXT, seed, 200, env)
        eng.reset(seed=seed)
        debt = 0.0
        traj: List[SearchTarget] = []
        for w in range(args.windows_per_episode):
            if eng.term_buf[0]:
                break
            obs = get_obs(eng, debt)
            plan, target = planner.plan_with_target(obs, 200)
            reward, spent, debt, executed, _, _ = execute_plan_until_budget(eng, plan, 200.0, debt, "FlatAlphaRadar_selfplay", seed, w)
            target.reward = float(reward)
            traj.append(target)
            rewards.append(float(reward))
            if executed <= 0 or spent <= 0:
                break
        eng.close()
        G = 0.0
        for target in reversed(traj):
            G = float(target.reward) + args.gamma * G
            target.ret = G
        rows.extend(traj)
    return rows, float(np.mean(rewards)) if rewards else 0.0


def train_flat_step(model, opt, replay: ReplayBuffer, batch_size: int, q_scale: float):
    if len(replay) < max(4, batch_size // 4):
        return {}
    batch = replay.sample(batch_size)
    x = torch.from_numpy(np.stack([b.x for b in batch]).astype(np.float32)).to(DEVICE)
    slot = torch.from_numpy(np.stack([b.slot for b in batch]).astype(np.float32)).to(DEVICE)
    pi = torch.from_numpy(np.stack([b.pi for b in batch]).astype(np.float32)).to(DEVICE)
    qtar = torch.from_numpy(np.stack([b.q for b in batch]).astype(np.float32) / q_scale).to(DEVICE)
    qmask = torch.from_numpy(np.stack([b.q_mask for b in batch]).astype(np.float32)).to(DEVICE)
    ret = torch.tensor([b.ret / q_scale for b in batch], dtype=torch.float32, device=DEVICE)
    logits, value, q = model(x, slot)
    policy_loss = F.kl_div(F.log_softmax(logits, dim=1), pi, reduction="batchmean")
    v_loss = F.smooth_l1_loss(value, ret)
    q_loss = F.smooth_l1_loss(q[qmask > 0.5], qtar[qmask > 0.5]) if bool(torch.any(qmask > 0.5)) else torch.zeros((), device=DEVICE)
    loss = policy_loss + 0.5 * v_loss + 0.5 * q_loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return {"loss": float(loss.detach().cpu()), "policy_loss": float(policy_loss.detach().cpu()), "v_loss": float(v_loss.detach().cpu()), "q_loss": float(q_loss.detach().cpu())}


def train_flat(args):
    ckpt = Path(args.flat_ckpt)
    model = FlatAlphaRadarNet().to(DEVICE)
    if ckpt.exists() and not args.force_retrain:
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        return model.cpu().eval(), float(args.eval_q_scale)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    replay = ReplayBuffer(args.replay_size)
    logs = []
    q_scale = float(args.eval_q_scale)
    for it in range(1, args.iterations + 1):
        args.iteration = it
        rows, mean_reward = collect_flat_selfplay(model.cpu().eval(), args)
        replay.extend(rows)
        vals = [abs(x.ret) for x in replay.items] + [abs(float(v)) for r in replay.items for v in r.q[r.q_mask > 0.5]]
        q_scale = float(max(1.0, np.percentile(vals, 90))) if vals else q_scale
        model.to(DEVICE).train()
        metrics = []
        for _ in range(args.train_steps):
            m = train_flat_step(model, opt, replay, args.batch_size, q_scale)
            if m:
                metrics.append(m)
        row = {"iteration": it, "collected": len(rows), "replay": len(replay), "mean_selfplay_reward": mean_reward, "q_scale": q_scale}
        if metrics:
            for k in metrics[0]:
                row[k] = float(np.mean([m[k] for m in metrics]))
        logs.append(row)
        print("flat_train", json.dumps(row), flush=True)
        torch.save(model.cpu().state_dict(), ckpt)
        pd.DataFrame(logs).to_csv(RUN_OUT / "flat_alpharadar_train_log.csv", index=False)
    return model.cpu().eval(), q_scale


def compare(args):
    flat, q_scale = train_flat(args)
    fact = AlphaRadarNet()
    fact.load_state_dict(torch.load(args.factorized_ckpt, map_location="cpu"))
    fact.eval()
    base = load_model(RES / "realistic_reward_retrain_20260508" / "realistic_return_q_policy.pt")
    oracle = HierarchicalWindowTransformer(); oracle.load_state_dict(torch.load(OUT / "hierarchical_window_oracle_teacher" / "oracle_hierwin_model.pt", map_location="cpu")); oracle.eval()
    oracleq = HierarchicalPolicyQTransformer(); oracleq.load_state_dict(torch.load(OUT / "hierarchical_oracle_q_utility" / "oracle_policy_q_model.pt", map_location="cpu")); oracleq.eval()

    cells = [(int(i), float(r)) for i in str(args.eval_initials).split(",") for r in str(args.eval_rates).split(",")]
    seeds = [int(x) for x in str(args.eval_seeds).split(",") if x]
    planners = {
        "FlatAlpha_DirectP": lambda: FlatAlphaRadarDirectPlanner(flat, alpha=0.0),
        "FlatAlpha_DirectPQ": lambda: FlatAlphaRadarDirectPlanner(flat, alpha=1.0),
        "FlatAlpha_MCTS_P": lambda: FlatAlphaRadarMCTSPlanner(flat, make_env(0.0), rollouts=args.eval_rollouts, c_puct=args.c_puct, expand_top_k=args.expand_top_k, q_scale=q_scale),
        "FlatAlpha_MCTS_PQ": lambda: FlatAlphaRadarMCTSPlanner(flat, make_env(0.0), rollouts=args.eval_rollouts, c_puct=args.c_puct, expand_top_k=args.expand_top_k, q_scale=q_scale, use_q_head=True, q_utility_weight=1.0),
        "FlatAlpha_MCTS_PV": lambda: FlatAlphaRadarMCTSPlanner(flat, make_env(0.0), rollouts=args.eval_rollouts, c_puct=args.c_puct, expand_top_k=args.expand_top_k, q_scale=q_scale, use_value_head=True, leaf_value_mix=0.5),
        "FlatAlpha_MCTS_PVQ": lambda: FlatAlphaRadarMCTSPlanner(flat, make_env(0.0), rollouts=args.eval_rollouts, c_puct=args.c_puct, expand_top_k=args.expand_top_k, q_scale=q_scale, use_q_head=True, q_utility_weight=1.0, use_value_head=True, leaf_value_mix=0.5),
        "FactAlpha_DirectP": lambda: AlphaRadarDirectPlanner(fact, alpha=0.0, beta=0.0, threshold=0.0),
        "FactAlpha_DirectPQ": lambda: AlphaRadarDirectPlanner(fact, alpha=1.0, beta=1.0, threshold=0.0),
        "FactAlpha_MCTS_P": lambda: AlphaRadarMCTSPlanner(fact, make_env(0.0), rollouts=args.eval_rollouts, c_puct=args.c_puct, expand_top_k=args.expand_top_k, q_scale=float(args.eval_q_scale)),
        "FactAlpha_MCTS_PQ": lambda: AlphaRadarMCTSPlanner(fact, make_env(0.0), rollouts=args.eval_rollouts, c_puct=args.c_puct, expand_top_k=args.expand_top_k, q_scale=float(args.eval_q_scale), use_q_head=True, q_utility_weight=1.0),
        "FactAlpha_MCTS_PV": lambda: AlphaRadarMCTSPlanner(fact, make_env(0.0), rollouts=args.eval_rollouts, c_puct=args.c_puct, expand_top_k=args.expand_top_k, q_scale=float(args.eval_q_scale), use_value_head=True, leaf_value_mix=0.5),
        "FactAlpha_MCTS_PVQ": lambda: AlphaRadarMCTSPlanner(fact, make_env(0.0), rollouts=args.eval_rollouts, c_puct=args.c_puct, expand_top_k=args.expand_top_k, q_scale=float(args.eval_q_scale), use_q_head=True, q_utility_weight=1.0, use_value_head=True, leaf_value_mix=0.5),
        "Old_HierPolicyQ": lambda: HierPolicyQPlanner(base, oracleq, alpha=1.0, beta=1.0, threshold=-1.238, timing=False),
        "Old_HierPolicyOnly": lambda: HierWindowPlanner(base, oracle, threshold=0.35),
        "Old_FlatPolicyQ": lambda: ContinuousUtilityPlanner(base, alpha=0.5),
        "Old_FlatPolicyOnly": lambda: ContinuousUtilityPlanner(base, alpha=0.0),
        "EDF": lambda: EDFPlanner(MAXT),
        "EST": lambda: ESTPlanner(MAXT),
    }
    rows = []
    for name, fac in planners.items():
        r = eval_factory(fac, name, cells, seeds, args.eval_windows)
        rows.extend(r)
        print("compare", name, float(pd.DataFrame(r)["reward_per_200ms_eq"].mean()), flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(RUN_OUT / "flat_vs_factorized_all_methods_raw.csv", index=False)
    summary = raw.groupby("planner").agg(
        reward=("reward_per_200ms_eq", "mean"),
        drop=("mean_drop_pct_active", "mean"),
        delay=("mean_delay_active", "mean"),
        search=("search_fraction", "mean"),
        latency=("planning_ms_per_200ms_eq", "mean"),
    ).reset_index().sort_values("reward", ascending=False)
    summary.to_csv(RUN_OUT / "flat_vs_factorized_all_methods_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=76)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--episodes-per-iter", type=int, default=4)
    ap.add_argument("--windows-per-episode", type=int, default=8)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--eval-rollouts", type=int, default=8)
    ap.add_argument("--expand-top-k", type=int, default=10)
    ap.add_argument("--c-puct", type=float, default=1.25)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--train-steps", type=int, default=16)
    ap.add_argument("--replay-size", type=int, default=50000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train-initials", default="5,15,25,35,50,75,100")
    ap.add_argument("--train-rates", default="0,1,2,5")
    ap.add_argument("--eval-initials", default="15,50")
    ap.add_argument("--eval-rates", default="0,2")
    ap.add_argument("--eval-seeds", default="100")
    ap.add_argument("--eval-windows", type=int, default=20)
    ap.add_argument("--eval-q-scale", type=float, default=3.78)
    ap.add_argument("--flat-ckpt", default=str(RUN_OUT / "flat_alpharadar_sanity.pt"))
    ap.add_argument("--factorized-ckpt", default=str(RUN_OUT / "alpharadar_foundation_sanity.pt"))
    ap.add_argument("--force-retrain", action="store_true")
    args = ap.parse_args()
    seedall(args.seed)
    compare(args)


if __name__ == "__main__":
    main()
