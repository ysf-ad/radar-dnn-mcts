from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from exact_env_mutual import MAXT, attach_env_obs, engine_env_cfg, env_cfg_for, xs_decode_action, xs_s_search_action, xs_s_track_action
from final_radar_campaign import run_fixed, summarize_window_df
from foundation_mcts_fair_eval import parse_floats, parse_ints, physical_candidates
from mutual_features import SLOT_DIM, slot_features, tokenize
from mutual_foundation import MutualRadarNet
from penalty_window_quota_learner_eval import make_exact_args
from realistic_reward_retrain import adapter
from repaired_campaign_tools import EDFPlanner, ESTPlanner
from window_plan_decoder_experiment import WindowSequence, action_to_class, load_sequences, sequence_weights


ROOT = Path(__file__).resolve().parents[4]


class GRUWindowDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 48,
        nhead: int = 4,
        enc_layers: int = 2,
        action_mixer_layers: int = 0,
        action_mixer_mode: str = "none",
        head_mode: str = "flat",
    ):
        super().__init__()
        self.action_mixer_layers = int(action_mixer_layers)
        self.action_mixer_mode = str(action_mixer_mode)
        self.head_mode = str(head_mode)
        self.legacy_flat = self.action_mixer_mode == "legacy_flat"
        self.backbone = MutualRadarNet(d_model=d_model, nhead=nhead, nlayers=enc_layers, head_arch="branch_context")
        self.slot_proj = nn.Sequential(nn.LayerNorm(SLOT_DIM), nn.Linear(SLOT_DIM, d_model), nn.GELU())
        self.prev_class = nn.Embedding(5, d_model)
        self.prev_row = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.init = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.Tanh())
        self.gru = nn.GRUCell(3 * d_model, d_model)
        self.sensor_embed = nn.Parameter(torch.randn(2, d_model) * 0.02)
        if self.legacy_flat:
            self.action_proj = None
        else:
            self.action_proj = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, d_model), nn.GELU())
        if self.action_mixer_mode == "transformer" and self.action_mixer_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=2 * d_model,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
            )
            self.action_mixer = nn.TransformerEncoder(layer, num_layers=self.action_mixer_layers, enable_nested_tensor=False, mask_check=False)
        else:
            self.action_mixer = None
        if self.action_mixer_mode == "summary":
            self.summary_mixer = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU())
        else:
            self.summary_mixer = None
        if self.legacy_flat:
            self.action_head = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        else:
            self.action_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        if self.head_mode == "factorized":
            self.type_proj = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU())
            self.type_head = nn.Linear(d_model, 2)
        else:
            self.type_proj = None
            self.type_head = None

    def encode_root(self, root_x: torch.Tensor):
        cls_out, tok_out, _selected, _active = self.backbone.encode_tokens(root_x)
        return cls_out, tok_out

    def score_from_hidden(self, hidden: torch.Tensor, cls_out: torch.Tensor, tok_out: torch.Tensor) -> torch.Tensor:
        bsz, rows, d_model = tok_out.shape
        h = hidden[:, None, None, :].expand(-1, rows, 2, -1)
        tok = tok_out[:, :, None, :].expand(-1, -1, 2, -1)
        cls = cls_out[:, None, None, :].expand(-1, rows, 2, -1)
        sensor = self.sensor_embed[None, None, :, :].expand(bsz, rows, -1, -1)
        raw_ctx = torch.cat([h, tok, cls, sensor], dim=-1)
        if self.legacy_flat:
            return self.action_head(raw_ctx).squeeze(-1)
        ctx = self.action_proj(raw_ctx)
        if self.action_mixer is not None:
            ctx = self.action_mixer(ctx.reshape(bsz, rows * 2, d_model)).reshape(bsz, rows, 2, d_model)
        elif self.summary_mixer is not None:
            flat = ctx.reshape(bsz, rows * 2, d_model)
            mean = flat.mean(dim=1, keepdim=True).expand(-1, rows * 2, -1)
            maxv = flat.max(dim=1, keepdim=True).values.expand(-1, rows * 2, -1)
            ctx = self.summary_mixer(torch.cat([flat, mean, maxv], dim=-1)).reshape(bsz, rows, 2, d_model)
        target_logits = self.action_head(ctx).squeeze(-1)
        if self.type_head is None:
            return target_logits
        cls2 = cls_out[:, None, :].expand(-1, 2, -1)
        h2 = hidden[:, None, :].expand(-1, 2, -1)
        sensor2 = self.sensor_embed[None, :, :].expand(bsz, -1, -1)
        type_logits = self.type_head(self.type_proj(torch.cat([h2, cls2, sensor2], dim=-1)))
        out = target_logits.new_full(target_logits.shape, -1e9)
        out[:, 0, :] = type_logits[:, :, 0]
        out[:, 1:, :] = type_logits[:, None, :, 1] + target_logits[:, 1:, :]
        return out

    def forward(self, root_x: torch.Tensor, slots: torch.Tensor, prev_classes: torch.Tensor, prev_rows: torch.Tensor) -> torch.Tensor:
        cls_out, tok_out = self.encode_root(root_x)
        h = self.init(cls_out)
        outs = []
        steps = int(slots.shape[1])
        for step in range(steps):
            inp = torch.cat(
                [
                    self.slot_proj(slots[:, step]),
                    self.prev_class(prev_classes[:, step].clamp(0, 4)),
                    self.prev_row(prev_rows[:, step : step + 1]),
                ],
                dim=-1,
            )
            h = self.gru(inp, h)
            outs.append(self.score_from_hidden(h, cls_out, tok_out))
        return torch.stack(outs, dim=1)


def collate(
    seqs: list[WindowSequence],
    idx: np.ndarray,
    max_seq: int,
    device: torch.device,
    seq_weights: np.ndarray | None = None,
    prefix_corrupt_prob: float = 0.0,
    rng: np.random.Generator | None = None,
):
    batch = [seqs[int(i)] for i in idx]
    bsz = len(batch)
    x = np.stack([b.root_x for b in batch], axis=0)
    slots = np.zeros((bsz, max_seq, SLOT_DIM), dtype=np.float32)
    prev_classes = np.zeros((bsz, max_seq), dtype=np.int64)
    prev_rows = np.zeros((bsz, max_seq), dtype=np.float32)
    labels = np.full((bsz, max_seq), -100, dtype=np.int64)
    weights = np.ones((bsz, max_seq), dtype=np.float32)
    for i, b in enumerate(batch):
        n = min(max_seq, len(b.labels))
        slots[i, :n] = b.slots[:n]
        prev_classes[i, :n] = b.prev_classes[:n]
        prev_rows[i, :n] = b.prev_rows[:n]
        labels[i, :n] = b.labels[:n]
        if seq_weights is not None:
            weights[i, :n] = float(seq_weights[int(idx[i])])
        corrupt_prob = float(prefix_corrupt_prob)
        if corrupt_prob > 0.0 and n > 1:
            local_rng = rng if rng is not None else np.random.default_rng()
            corrupt = local_rng.random(n) < corrupt_prob
            corrupt[0] = False
            for j in np.nonzero(corrupt)[0]:
                # Scheduled-sampling style robustness: perturb the previous
                # action context while keeping the supervised next action.
                if local_rng.random() < 0.35:
                    prev_classes[i, j] = 1
                    prev_rows[i, j] = 0.0
                else:
                    src = int(local_rng.integers(0, max(1, j)))
                    prev_label = int(labels[i, src]) if int(labels[i, src]) >= 0 else 0
                    prev_row = prev_label // 2
                    prev_sensor = prev_label % 2
                    prev_action = xs_s_search_action(MAXT) if prev_row <= 0 else xs_s_track_action(prev_row, MAXT)
                    prev_classes[i, j] = action_to_class(prev_action)
                    prev_rows[i, j] = float(prev_row) / float(MAXT)
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(slots).to(device),
        torch.from_numpy(prev_classes).to(device),
        torch.from_numpy(prev_rows).to(device),
        torch.from_numpy(labels).to(device),
        torch.from_numpy(weights).to(device),
    )


def train_decoder(seqs: list[WindowSequence], args, device: torch.device) -> GRUWindowDecoder:
    model = GRUWindowDecoder(
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        enc_layers=int(args.enc_layers),
        action_mixer_layers=int(getattr(args, "action_mixer_layers", 0)),
        action_mixer_mode=str(getattr(args, "action_mixer_mode", "none")),
        head_mode=str(getattr(args, "head_mode", "flat")),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(args.model_seed))
    seq_w = sequence_weights(seqs, str(getattr(args, "initial_weights", "")))
    sample_w = seq_w / seq_w.sum() if bool(args.weighted_sampling) and float(seq_w.sum()) > 0.0 else None
    for step in range(int(args.train_steps)):
        if sample_w is None:
            idx = rng.integers(0, len(seqs), size=int(args.batch_size))
        else:
            idx = rng.choice(len(seqs), size=int(args.batch_size), replace=True, p=sample_w)
        x, slots, prev_classes, prev_rows, labels, weights = collate(
            seqs,
            idx,
            int(args.max_seq),
            device,
            seq_w,
            prefix_corrupt_prob=float(getattr(args, "prefix_corrupt_prob", 0.0)),
            rng=rng,
        )
        logits = model(x, slots, prev_classes, prev_rows)
        flat_logits = logits.reshape(-1, (MAXT + 1) * 2)
        flat_labels = labels.reshape(-1)
        loss_all = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="none")
        valid = flat_labels >= 0
        flat_w = weights.reshape(-1)
        loss = (loss_all[valid] * flat_w[valid]).sum() / flat_w[valid].sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, int(args.log_every)) == 0 or step == int(args.train_steps) - 1:
            with torch.no_grad():
                pred = flat_logits.argmax(dim=1)
                top1 = (pred[valid] == flat_labels[valid]).float().mean().item() if bool(valid.any()) else 0.0
                search_acc = (((pred[valid] // 2) == 0) == ((flat_labels[valid] // 2) == 0)).float().mean().item() if bool(valid.any()) else 0.0
            print({"step": step, "loss": float(loss.detach().cpu()), "top1": top1, "search_acc": search_acc}, flush=True)
    return model.eval()


class GRUWindowPlanner:
    def __init__(self, model: GRUWindowDecoder, env_cfg: dict, max_seq: int = 32):
        self.model = model.eval()
        self.env_cfg = dict(env_cfg)
        self.max_seq = int(max_seq)
        self.adapt = adapter()

    def plan(self, obs, budget_ms=200):
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        root_x = tokenize(self.adapt, obs, selected=set(), search_count=0).astype(np.float32)
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            root_t = torch.from_numpy(root_x).float().unsqueeze(0).to(device)
            cls_out, tok_out = self.model.encode_root(root_t)
            h = self.model.init(cls_out)
        selected: set[int] = set()
        plan: list[int] = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        active = np.asarray(obs.get("active_mask", np.zeros((MAXT,), dtype=bool)), dtype=bool)
        deadline = np.asarray(obs.get("t_deadline", np.full((MAXT,), -1.0)), dtype=np.float32)
        ranges = np.asarray(obs.get("target_range", np.zeros_like(deadline)), dtype=np.float32)
        candidate_rows = [0]
        if float(obs.get("s_band_busy_ms", 0.0)) <= 0.0:
            ranked: list[tuple[float, int]] = []
            for i, ok in enumerate(active[:MAXT]):
                if not bool(ok) or i >= len(deadline) or float(deadline[i]) < 0.0:
                    continue
                rng = float(ranges[i]) if i < len(ranges) else 50_000_000.0
                if 10_000_000.0 < rng < 184_000_000.0:
                    ranked.append((float(deadline[i]), i + 1))
            ranked.sort(key=lambda item: (item[0], item[1]))
            candidate_rows.extend([row for _deadline, row in ranked[:MAXT]])
        candidate_rows = [int(r) for r in candidate_rows if 0 <= int(r) <= MAXT]
        candidate_t = torch.as_tensor(candidate_rows, dtype=torch.long, device=device)
        candidate_mask = torch.zeros((MAXT + 1,), dtype=torch.bool, device=device)
        if candidate_t.numel() > 0:
            candidate_mask.scatter_(0, candidate_t.clamp(0, MAXT), True)
        dwell_t = torch.as_tensor(dwell[:MAXT], dtype=torch.float32, device=device) if dwell.size else torch.ones((MAXT,), dtype=torch.float32, device=device) * 10.0
        selected_t = torch.zeros((MAXT + 1,), dtype=torch.bool, device=device)
        while elapsed < float(budget_ms) and len(plan) < self.max_seq:
            prev_action = plan[-1] if plan else None
            prev_cls = 0 if prev_action is None else action_to_class(int(prev_action))
            prev_base = 0 if prev_action is None else max(0, int(xs_decode_action(int(prev_action), MAXT)[0]))
            slot = slot_features(obs, elapsed, search_count, track_count, last, float(budget_ms)).astype(np.float32)
            with torch.inference_mode():
                inp = torch.cat(
                    [
                        self.model.slot_proj(torch.from_numpy(slot).float().to(device).unsqueeze(0)),
                        self.model.prev_class(torch.tensor([prev_cls], dtype=torch.long, device=device)),
                        self.model.prev_row(torch.tensor([[float(prev_base) / float(MAXT)]], dtype=torch.float32, device=device)),
                    ],
                    dim=-1,
                )
                h = self.model.gru(inp, h)
                scores_t = self.model.score_from_hidden(h, cls_out, tok_out)[0, :, 0]
            remaining = max(0.0, float(budget_ms) - float(elapsed))
            valid_t = candidate_mask & (~selected_t)
            if remaining < 10.0:
                valid_t[0] = False
            if dwell_t.numel() > 0:
                n = min(MAXT, int(dwell_t.numel()))
                valid_t[1 : n + 1] &= dwell_t[:n] <= float(remaining)
            if not bool(valid_t.any().item()):
                break
            best_row_t = torch.argmax(scores_t.masked_fill(~valid_t, -1e9))
            row = int(best_row_t.item())
            best_action = xs_s_search_action(MAXT) if row <= 0 else xs_s_track_action(row, MAXT)
            row, _sensor = xs_decode_action(best_action, MAXT)
            row = int(max(0, row))
            dt = 10.0 if row <= 0 else float(max(1.0, dwell[row - 1] if row - 1 < len(dwell) else 5.0))
            if elapsed + dt > float(budget_ms) and plan:
                break
            plan.append(xs_s_search_action(MAXT) if row <= 0 else xs_s_track_action(row, MAXT))
            if row <= 0:
                search_count += 1
                last = 0
            else:
                selected.add(row)
                selected_t[row] = True
                track_count += 1
                last = row
            elapsed += dt
        return plan if plan else [xs_s_search_action(MAXT)]


def eval_planners(model: GRUWindowDecoder, args, exact_args) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    windows = []
    for seed in parse_ints(args.eval_seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                env_cfg = env_cfg_for(float(rate), exact_args)
                env_cfg["enable_x_band"] = 0 if bool(args.single_sensor) else 1
                planners = {
                    "EDF": EDFPlanner(MAXT),
                    "EST": ESTPlanner(MAXT),
                    "GRUWindowDecoder": GRUWindowPlanner(model, env_cfg, max_seq=int(args.max_seq)),
                }
                for name, planner in planners.items():
                    t0 = time.perf_counter()
                    w, _ = run_fixed(planner, name, int(initial), MAXT, int(seed), int(args.eval_windows), 200, engine_env_cfg(env_cfg))
                    lat = 1000.0 * (time.perf_counter() - t0) / max(1, int(args.eval_windows))
                    w = w.assign(method=name, initial=int(initial), rate=float(rate), seed=int(seed), latency_ms_window=float(lat))
                    windows.append(w)
                    s = summarize_window_df(w, "fixed")
                    row = {
                        "method": name,
                        "initial": int(initial),
                        "rate": float(rate),
                        "seed": int(seed),
                        "reward": float(s.get("reward_per_200ms_eq", np.nan)),
                        "search": float(s.get("search_fraction", np.nan)),
                        "latency_ms_window": float(lat),
                    }
                    rows.append(row)
                    print(row, flush=True)
    return pd.DataFrame(rows), pd.concat(windows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="Path or comma-separated paths to plan-sequence .pt files")
    ap.add_argument("--out", default=str(ROOT / "CreateValid1" / "results" / "fast_gru_window_decoder_eval.csv"))
    ap.add_argument("--save-model", default="")
    ap.add_argument("--load-model", default="")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--single-sensor", action="store_true")
    ap.add_argument("--initials", default="20,40,60")
    ap.add_argument("--rates", default="2,3,4")
    ap.add_argument("--eval-seeds", default="916")
    ap.add_argument("--eval-windows", type=int, default=100)
    ap.add_argument("--max-seq", type=int, default=24)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--enc-layers", type=int, default=1)
    ap.add_argument("--action-mixer-layers", type=int, default=0)
    ap.add_argument("--action-mixer-mode", choices=["none", "summary", "transformer", "legacy_flat"], default="none")
    ap.add_argument("--head-mode", choices=["flat", "factorized"], default="flat")
    ap.add_argument("--train-steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log-every", type=int, default=60)
    ap.add_argument("--model-seed", type=int, default=123)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--weighted-sampling", action="store_true")
    ap.add_argument("--initial-weights", default="")
    ap.add_argument("--prefix-corrupt-prob", type=float, default=0.0)
    ap.add_argument("--env-mode", default="pufferlib_service")
    ap.add_argument("--track-update-reward", type=float, default=0.30)
    ap.add_argument("--track-loss-penalty", type=float, default=8.0)
    ap.add_argument("--search-frame-overdue-weight", type=float, default=0.5)
    ap.add_argument("--search-frame-drop-penalty", type=float, default=8.0)
    ap.add_argument("--search-frame-state-penalty-weight", type=float, default=2.0)
    ap.add_argument("--search-frame-delta-reward-weight", type=float, default=5.0)
    ap.add_argument("--service-pressure-delta-reward-weight", type=float, default=0.30)
    ap.add_argument("--serviced-pressure-improvement-reward-weight", type=float, default=0.15)
    ap.add_argument("--discovered-target-reward", type=float, default=0.08)
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(int(args.model_seed))
    np.random.seed(int(args.model_seed))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    else:
        device = torch.device(args.device)
    print({"device": str(device)}, flush=True)
    if not hasattr(args, "windows"):
        args.windows = int(args.eval_windows)
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = not bool(args.single_sensor)
    exact_args.single_sensor = bool(args.single_sensor)
    target_paths = [Path(p.strip()) for p in str(args.targets).split(",") if p.strip()]
    seqs = []
    for target_path in target_paths:
        seqs.extend(load_sequences(target_path, int(args.max_seq)))
    print({"sequences": len(seqs), "mean_len": float(np.mean([len(s.labels) for s in seqs]))}, flush=True)
    if args.load_model:
        state = torch.load(str(args.load_model), map_location="cpu", weights_only=False)
        cfg = state.get("config", {}) if isinstance(state, dict) else {}
        load_action_mixer_mode = str(cfg.get("action_mixer_mode", getattr(args, "action_mixer_mode", "flat")))
        if "action_mixer_mode" not in cfg and "head_mode" not in cfg:
            load_action_mixer_mode = "legacy_flat"
        model = GRUWindowDecoder(
            d_model=int(cfg.get("d_model", args.d_model)),
            nhead=int(cfg.get("nhead", args.nhead)),
            enc_layers=int(cfg.get("enc_layers", args.enc_layers)),
            action_mixer_layers=int(cfg.get("action_mixer_layers", getattr(args, "action_mixer_layers", 0))),
            action_mixer_mode=load_action_mixer_mode,
            head_mode=str(cfg.get("head_mode", getattr(args, "head_mode", "flat"))),
        ).eval().to(device)
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state, strict=True)
    elif args.eval_only:
        raise ValueError("--eval-only requires --load-model")
    else:
        model = train_decoder(seqs, args, device)
    if args.save_model:
        torch.save(
            {
                "model": model.state_dict(),
                "config": {
                    "d_model": int(args.d_model),
                    "nhead": int(args.nhead),
                    "enc_layers": int(args.enc_layers),
                    "action_mixer_layers": int(args.action_mixer_layers),
                    "action_mixer_mode": str(args.action_mixer_mode),
                    "head_mode": str(args.head_mode),
                    "max_seq": int(args.max_seq),
                    "targets": [str(p) for p in target_paths],
                },
            },
            str(args.save_model),
        )
    if args.train_only:
        return
    raw, windows = eval_planners(model, args, exact_args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out, index=False)
    windows.to_csv(out.with_name(out.stem + "_windows.csv"), index=False)
    summary = raw.groupby("method").agg(reward=("reward", "mean"), search=("search", "mean"), latency_ms_window=("latency_ms_window", "mean"), n=("reward", "size")).reset_index().sort_values("reward", ascending=False)
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
