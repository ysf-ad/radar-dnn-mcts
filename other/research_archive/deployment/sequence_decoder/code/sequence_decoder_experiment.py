from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from final_radar_campaign import MAXT, build_env, get_obs, run_fixed, seedall, summarize_window_df
from load_adaptive_train_eval import make_env
from mutual_features import TOKEN_DIM, tokenize
from mutual_foundation import MutualRadarDirectPlanner, MutualRadarNet
from realistic_reward_retrain import adapter
from repaired_campaign_tools import EDFPlanner, ESTPlanner
from strict_window_report import execute_plan_until_budget


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(os.environ.get("MutualRadar_RUN_OUT", "CreateValid1/results/sequence_decoder_experiment"))
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)


class ParallelSequenceDecoder(nn.Module):
    """One-shot window decoder.

    It encodes the radar tokens once and emits a fixed sequence of factorized
    search-vs-track and target logits in parallel.  This is deliberately simpler
    than the autoregressive direct planner: the speed target is one forward pass
    per 200 ms window.
    """

    def __init__(self, seq_len: int = 32, token_dim: int = TOKEN_DIM, d_model: int = 96, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.seq_len = int(seq_len)
        self.token_proj = nn.Linear(token_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.05,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers, enable_nested_tensor=False, mask_check=False)
        self.pos = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        self.type_head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.track_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, tokens: torch.Tensor):
        token_active = tokens[:, :, 4] > 0.5
        token_active[:, 0] = True

        emb = self.token_proj(tokens)
        cls = self.cls_token.unsqueeze(0).unsqueeze(0).expand(tokens.shape[0], 1, -1)
        emb = torch.cat([cls, emb], dim=1)
        cls_valid = torch.ones((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device)
        out = self.encoder(emb, src_key_padding_mask=~torch.cat([cls_valid, token_active], dim=1))
        cls_out = out[:, 0, :]
        tok_out = out[:, 1:, :]

        pos = self.pos.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        cls_seq = cls_out.unsqueeze(1).expand(-1, self.seq_len, -1)
        type_logits = self.type_head(torch.cat([cls_seq, pos], dim=-1)).squeeze(-1)

        tok = tok_out.unsqueeze(1).expand(-1, self.seq_len, -1, -1)
        cls_rep = cls_out[:, None, None, :].expand(-1, self.seq_len, tok_out.shape[1], -1)
        pos_rep = pos[:, :, None, :].expand(-1, self.seq_len, tok_out.shape[1], -1)
        track_logits = self.track_head(torch.cat([tok, cls_rep, pos_rep], dim=-1)).squeeze(-1)
        track_mask = token_active.clone()
        track_mask[:, 0] = False
        track_logits = track_logits.masked_fill(~track_mask[:, None, :], -1e9)
        return type_logits, track_logits


class SequenceDirectPlanner:
    def __init__(self, model: ParallelSequenceDecoder, threshold: float = 0.5, mode: str = "branch", allow_retrack: bool = False):
        self.model = model.eval()
        self.adapt = adapter()
        self.threshold = float(threshold)
        self.mode = str(mode)
        self.allow_retrack = bool(allow_retrack)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def warmup(self, obs, budget_ms=200):
        _ = self.plan(obs, budget_ms=budget_ms)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def plan(self, obs, budget_ms=200):
        x = tokenize(self.adapt, obs, selected=set(), search_count=0)
        with torch.inference_mode():
            tokens = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            type_logits, track_logits = self.model(tokens)
            if self.mode == "flat":
                search_logits = type_logits[0]
                # Flat-like decode compares search logit against the best target logit at each slot.
                scores = track_logits[0]
                best_track = torch.argmax(scores, dim=-1)
                best_score = scores[torch.arange(track_logits.shape[1], device=self.device), best_track]
                choose_search = search_logits >= (best_score + self.threshold)
            else:
                p_search = torch.sigmoid(type_logits[0])
                scores = track_logits[0]
                best_track = torch.argmax(scores, dim=-1)
                choose_search = p_search >= self.threshold
            actions_t = torch.where(choose_search, torch.zeros_like(best_track), best_track)
            actions = actions_t.detach().cpu().numpy().astype(int)
            if not self.allow_retrack:
                scores_np = scores.detach().cpu().numpy()
                used = set()
                for i, a in enumerate(actions.tolist()):
                    if a <= 0:
                        continue
                    if a not in used:
                        used.add(int(a))
                        continue
                    row = scores_np[i].copy()
                    row[0] = -1e9
                    for u in used:
                        if 0 <= u < row.shape[0]:
                            row[u] = -1e9
                    repl = int(np.argmax(row))
                    actions[i] = repl if row[repl] > -1e8 else 0
                    if actions[i] > 0:
                        used.add(int(actions[i]))
        return [int(a) for a in actions]


def load_teacher(path: str, d_model: int, nhead: int, nlayers: int, args) -> MutualRadarDirectPlanner:
    model = MutualRadarNet(d_model=d_model, nhead=nhead, nlayers=nlayers)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return MutualRadarDirectPlanner(
        model,
        alpha=args.teacher_alpha,
        beta=args.teacher_beta,
        threshold=args.teacher_threshold,
        direct_mode=args.teacher_mode,
        allow_retrack=True,
        cache_encoder=True,
    )


def generate_teacher_data(args) -> Tuple[np.ndarray, np.ndarray]:
    teacher = load_teacher(args.ckpt, args.d_model, args.nhead, args.nlayers, args)
    adapt = adapter()
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    initials = [int(x) for x in args.train_initials.split(",")]
    rates = [float(x) for x in args.train_rates.split(",")]
    seeds = [int(x) for x in args.train_seeds.split(",")]
    for init in initials:
        for rate in rates:
            env_cfg = make_env(rate)
            for seed in seeds:
                eng = build_env(teacher, init, MAXT, seed, 200, env_cfg)
                eng.reset(seed=seed)
                search_debt = 0.0
                for w in range(args.train_windows):
                    obs = get_obs(eng, search_debt)
                    plan = teacher.plan(obs, budget_ms=200)
                    y = np.full((args.seq_len,), -100, dtype=np.int64)
                    n = min(args.seq_len, len(plan))
                    y[:n] = np.asarray(plan[:n], dtype=np.int64)
                    xs.append(tokenize(adapt, obs, selected=set(), search_count=0))
                    ys.append(y)
                    reward, spent_ms, search_debt, *_ = execute_plan_until_budget(eng, plan, 200.0, search_debt, "Teacher", seed, w)
                    if eng.term_buf[0]:
                        break
                eng.close()
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.int64)


def train_sequence(args) -> ParallelSequenceDecoder:
    seedall(args.seed)
    x, y = generate_teacher_data(args)
    np.savez_compressed(OUT / "sequence_teacher_data.npz", x=x, y=y)
    model = ParallelSequenceDecoder(seq_len=args.seq_len, d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    x_t = torch.from_numpy(x).to(DEVICE)
    y_t = torch.from_numpy(y).to(DEVICE)
    log = []
    for step in range(args.train_steps):
        idx = torch.randint(0, x_t.shape[0], (min(args.batch_size, x_t.shape[0]),), device=DEVICE)
        xb = x_t[idx]
        yb = y_t[idx]
        type_logits, track_logits = model(xb)
        valid = yb >= 0
        is_search = (yb == 0).float()
        type_loss = F.binary_cross_entropy_with_logits(type_logits[valid], is_search[valid]) if bool(valid.any()) else torch.zeros((), device=DEVICE)
        track_valid = valid & (yb > 0)
        if bool(track_valid.any()):
            track_loss = F.cross_entropy(track_logits[track_valid], yb[track_valid])
        else:
            track_loss = torch.zeros((), device=DEVICE)
        loss = type_loss + track_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.log_every) == 0 or step == args.train_steps - 1:
            log.append({"step": step, "loss": float(loss.detach().cpu()), "type_loss": float(type_loss.detach().cpu()), "track_loss": float(track_loss.detach().cpu())})
            print(log[-1], flush=True)
    pd.DataFrame(log).to_csv(OUT / "sequence_train_log.csv", index=False)
    torch.save(model.state_dict(), OUT / "sequence_decoder.pt")
    return model


def evaluate(model: ParallelSequenceDecoder, args):
    rows = []
    cells = [(int(i), float(r)) for i in args.eval_initials.split(",") for r in args.eval_rates.split(",")]
    seeds = [int(x) for x in args.eval_seeds.split(",")]
    teacher = load_teacher(args.ckpt, args.d_model, args.nhead, args.nlayers, args)
    for init, rate in cells:
        env_cfg = make_env(rate)
        for seed in seeds:
            planners = [
                ("SequenceDecoder", SequenceDirectPlanner(model, threshold=args.seq_threshold, mode=args.seq_mode, allow_retrack=not args.seq_no_retrack)),
                ("TeacherDirectPolicyQ", teacher),
                ("EDF", EDFPlanner(MAXT)),
                ("EST", ESTPlanner(MAXT)),
            ]
            for name, planner in planners:
                w, _ = run_fixed(planner, name, init, MAXT, seed, args.eval_windows, 200, env_cfg)
                s = summarize_window_df(w, "fixed")
                s.update(planner=name, initial_targets=init, rate=rate, seed=seed)
                rows.append(s)
                w.to_csv(OUT / f"{name}_windows_init{init}_rate{rate}_seed{seed}.csv", index=False)
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "sequence_eval_raw.csv", index=False)
    summary = raw.groupby("planner").agg(
        reward=("reward_per_200ms_eq", "mean"),
        delay=("mean_delay_active", "mean"),
        search=("search_fraction", "mean"),
        latency=("planning_ms_per_200ms_eq", "mean"),
    ).reset_index().sort_values("reward", ascending=False)
    summary.to_csv(OUT / "sequence_eval_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train_eval", "eval"], default="train_eval")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq-ckpt", default="")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--teacher-mode", choices=["prob", "branch", "flat", "q"], default="prob")
    ap.add_argument("--teacher-alpha", type=float, default=0.5)
    ap.add_argument("--teacher-beta", type=float, default=0.5)
    ap.add_argument("--teacher-threshold", type=float, default=0.0)
    ap.add_argument("--train-initials", default="8,15,30,50")
    ap.add_argument("--train-rates", default="0,1,2")
    ap.add_argument("--train-seeds", default="1,2,3,4,5,6")
    ap.add_argument("--train-windows", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-initials", default="15,50")
    ap.add_argument("--eval-rates", default="0,2")
    ap.add_argument("--eval-seeds", default="100")
    ap.add_argument("--eval-windows", type=int, default=5)
    ap.add_argument("--seq-threshold", type=float, default=0.5)
    ap.add_argument("--seq-mode", choices=["branch", "flat"], default="branch")
    ap.add_argument("--seq-no-retrack", action="store_true")
    args = ap.parse_args()
    if args.mode == "eval":
        model = ParallelSequenceDecoder(seq_len=args.seq_len, d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers).to(DEVICE)
        model.load_state_dict(torch.load(args.seq_ckpt, map_location=DEVICE))
        model.eval()
    else:
        model = train_sequence(args)
    evaluate(model, args)


if __name__ == "__main__":
    main()
