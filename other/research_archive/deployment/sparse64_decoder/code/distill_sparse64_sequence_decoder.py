from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from collect_window_sequences_from_ar_teacher import collect_one
from eval_action_attention_muzero_g import run_plan_eval, summarize
from exact_env_mutual import (
    EDFPlanner,
    ESTPlanner,
    MAXT,
    attach_env_obs,
    env_cfg_for,
    xs_s_search_action,
    xs_s_track_action,
)
from foundation_mcts_fair_eval import parse_floats, parse_ints
from mutual_features import SLOT_DIM, TOKEN_DIM, slot_features, tokenize
from penalty_window_quota_learner_eval import make_exact_args
from realistic_reward_retrain import adapter
from single_sensor_ar_action_attention import CachedSingleSensorActionAttentionAR, load_action_attention_model
from two_sensor_physical_head_eval import PhysicalHeadPlanner


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_STATE = ROOT / "CreateValid1" / "results" / "single_sensor_fair_exact_action_attention_train_two_row_action_attention_qpolicy_factored_loss.pt"
DEFAULT_OUT = ROOT / "CreateValid1" / "results" / "sparse64_seqdistill"


class Sparse64SequenceDecoder(nn.Module):
    """Root-encoded AR decoder distilled from the sparse64 action-attention planner."""

    def __init__(self, d_model: int = 96, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.token_proj = nn.Linear(TOKEN_DIM, d_model)
        self.cls_token = nn.Parameter(torch.randn(d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers, enable_nested_tensor=False, mask_check=False)
        self.slot_proj = nn.Sequential(nn.LayerNorm(SLOT_DIM), nn.Linear(SLOT_DIM, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.prev_class_emb = nn.Embedding(5, d_model)
        self.prev_row_proj = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.step = nn.GRUCell(3 * d_model, d_model)
        self.type_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.target_head = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.value_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def encode(self, root_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active = root_x[:, :, 4] > 0.5
        active[:, 0] = True
        emb = self.token_proj(root_x)
        cls = self.cls_token.view(1, 1, -1).expand(root_x.shape[0], 1, -1)
        x = torch.cat([cls, emb], dim=1)
        mask = torch.cat([torch.ones(root_x.shape[0], 1, dtype=torch.bool, device=root_x.device), active], dim=1)
        out = self.encoder(x, src_key_padding_mask=~mask)
        return out[:, 0, :], out[:, 1:, :], active

    def score_step(
        self,
        cls: torch.Tensor,
        tok: torch.Tensor,
        slot: torch.Tensor,
        prev_class: torch.Tensor,
        prev_row: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inp = torch.cat(
            [
                self.slot_proj(slot),
                self.prev_class_emb(prev_class.clamp(0, 4)),
                self.prev_row_proj(prev_row.view(-1, 1).to(slot.dtype)),
            ],
            dim=-1,
        )
        h = self.step(inp, cls if hidden is None else hidden)
        type_logits = self.type_head(h)
        h_t = h[:, None, :].expand(-1, tok.shape[1], -1)
        cls_t = cls[:, None, :].expand(-1, tok.shape[1], -1)
        target_logits = self.target_head(torch.cat([tok, h_t, cls_t], dim=-1)).squeeze(-1)
        return type_logits, target_logits, h

    def forward(self, root_x: torch.Tensor, slots: torch.Tensor, prev_class: torch.Tensor, prev_row: torch.Tensor, return_hidden: bool = False):
        cls, tok, active = self.encode(root_x)
        logits_type = []
        logits_target = []
        hidden_states = []
        h = None
        for t in range(slots.shape[1]):
            tl, tr, h = self.score_step(cls, tok, slots[:, t, :], prev_class[:, t], prev_row[:, t], h)
            logits_type.append(tl)
            logits_target.append(tr)
            hidden_states.append(h)
        if return_hidden:
            return torch.stack(logits_type, dim=1), torch.stack(logits_target, dim=1), active, torch.stack(hidden_states, dim=1)
        return torch.stack(logits_type, dim=1), torch.stack(logits_target, dim=1), active


def action_to_prev_class(row: int) -> int:
    return 1 if int(row) <= 0 else 3


def load_sequences(path: Path) -> list[dict]:
    import __main__

    try:
        from train_sonly_puct_dagger_ar import DaggerWindow

        __main__.DaggerWindow = DaggerWindow
    except ImportError:
        pass
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "windows" in payload:
        seqs = []
        for window in payload["windows"]:
            valid = np.flatnonzero(np.asarray(window.mask).astype(bool))
            labels = np.asarray(window.target_pairs[valid, 0], dtype=np.int64)
            rows = np.maximum(0, labels // 2)
            prev_classes = np.zeros((len(rows),), dtype=np.int64)
            prev_rows = np.zeros((len(rows),), dtype=np.float32)
            if len(rows) > 1:
                prev_classes[1:] = np.asarray([action_to_prev_class(row) for row in rows[:-1]], dtype=np.int64)
                prev_rows[1:] = rows[:-1].astype(np.float32) / float(MAXT)
            seqs.append({
                "root_x": np.asarray(window.x, dtype=np.float32),
                "slots": np.asarray(window.slots[valid], dtype=np.float32),
                "labels": labels,
                "prev_classes": prev_classes,
                "prev_rows": prev_rows,
                "returns": np.asarray(window.returns[valid], dtype=np.float32),
                "teacher_reward": float(np.asarray(window.rewards[valid], dtype=np.float32).sum()),
                "meta": dict(window.meta or {}),
            })
        return [sequence for sequence in seqs if len(sequence["labels"]) > 0]
    seqs = payload["plan_sequences"] if isinstance(payload, dict) and "plan_sequences" in payload else payload
    return [s for s in seqs if len(s.get("labels", [])) > 0]


def collect_dataset(args: argparse.Namespace, out: Path) -> Path:
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = False
    exact_args.single_sensor = True
    model = load_action_attention_model(Path(args.base_state), str(args.device), "two_row_action_attention")
    all_sequences: list[dict] = []
    all_rows: list[dict] = []
    for initial in parse_ints(args.initials):
        for rate in parse_floats(args.rates):
            env_cfg = env_cfg_for(float(rate), exact_args)
            env_cfg["enable_x_band"] = 0
            base = PhysicalHeadPlanner(
                model,
                "two_row_action_attention",
                env_cfg,
                policy_weight=float(args.teacher_policy_weight),
                q_weight=float(args.teacher_q_weight),
                search_score_bias=0.0,
            )
            teacher = CachedSingleSensorActionAttentionAR(
                base,
                max_steps=int(args.max_steps),
                search_floor=0,
                search_cap_frac=1.0,
                env_cfg=env_cfg,
                action_coupler_top_k=int(args.action_coupler_top_k),
                sparse_residuals=bool(args.sparse_residuals),
            )
            for seed in parse_ints(args.seeds):
                seqs, rows = collect_one(teacher, env_cfg, int(initial), float(rate), int(seed), int(args.windows))
                all_sequences.extend(seqs)
                all_rows.extend(rows)
                print({"collect": len(all_sequences), "initial": int(initial), "rate": float(rate), "seed": int(seed)}, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"plan_sequences": all_sequences, "rows": all_rows}, out)
    pd.DataFrame(all_rows).to_csv(out.with_name(out.stem + "_teacher_rows.csv"), index=False)
    return out


def make_batch(seqs: list[dict], idx: np.ndarray, max_seq: int, device: torch.device) -> dict[str, torch.Tensor]:
    b = len(idx)
    root = np.zeros((b, MAXT + 1, TOKEN_DIM), dtype=np.float32)
    slots = np.zeros((b, max_seq, SLOT_DIM), dtype=np.float32)
    prev_class = np.zeros((b, max_seq), dtype=np.int64)
    prev_row = np.zeros((b, max_seq), dtype=np.float32)
    y_row = np.full((b, max_seq), -100, dtype=np.int64)
    y_type = np.full((b, max_seq), -100, dtype=np.int64)
    y_value = np.zeros((b, max_seq), dtype=np.float32)
    for n, j in enumerate(idx):
        s = seqs[int(j)]
        labels = np.asarray(s["labels"], dtype=np.int64)
        rows = np.maximum(0, labels // 2)
        length = min(max_seq, len(rows), len(s["slots"]))
        root[n] = np.asarray(s["root_x"], dtype=np.float32)
        slots[n, :length] = np.asarray(s["slots"], dtype=np.float32)[:length]
        prev_class[n, :length] = np.asarray(s.get("prev_classes", np.zeros(length)), dtype=np.int64)[:length]
        prev_row[n, :length] = np.asarray(s.get("prev_rows", np.zeros(length)), dtype=np.float32)[:length]
        y_row[n, :length] = rows[:length]
        y_type[n, :length] = (rows[:length] > 0).astype(np.int64)
        reward_mean = float(s.get("_reward_mean", 0.0))
        reward_std = max(1.0e-6, float(s.get("_reward_std", 1.0)))
        if "returns" in s:
            returns = np.asarray(s["returns"], dtype=np.float32)[:length]
            y_value[n, :length] = (returns - reward_mean) / reward_std
        else:
            y_value[n, :length] = (float(s.get("teacher_reward", 0.0)) - reward_mean) / reward_std
    return {
        "root": torch.from_numpy(root).to(device),
        "slots": torch.from_numpy(slots).to(device),
        "prev_class": torch.from_numpy(prev_class).to(device),
        "prev_row": torch.from_numpy(prev_row).to(device),
        "y_row": torch.from_numpy(y_row).to(device),
        "y_type": torch.from_numpy(y_type).to(device),
        "y_value": torch.from_numpy(y_value).to(device),
    }


def train(args: argparse.Namespace, data_path: Path, ckpt: Path) -> Sparse64SequenceDecoder:
    device = torch.device(args.device)
    seqs = load_sequences(data_path)
    value_targets = [
        np.asarray(s["returns"], dtype=np.float32).reshape(-1)
        if "returns" in s
        else np.asarray([float(s.get("teacher_reward", 0.0))], dtype=np.float32)
        for s in seqs
    ]
    rewards = np.concatenate(value_targets) if value_targets else np.zeros((0,), dtype=np.float32)
    reward_mean = float(rewards.mean()) if rewards.size else 0.0
    reward_std = float(max(1.0e-6, rewards.std())) if rewards.size else 1.0
    for s in seqs:
        s["_reward_mean"] = reward_mean
        s["_reward_std"] = reward_std
    model = Sparse64SequenceDecoder(d_model=int(args.d_model), nhead=int(args.nhead), nlayers=int(args.nlayers)).to(device)
    if str(getattr(args, "init_checkpoint", "")).strip():
        initial = torch.load(str(args.init_checkpoint), map_location=device, weights_only=False)
        state = initial.get("model", initial) if isinstance(initial, dict) else initial
        model.load_state_dict(state, strict=False)
        print({"init_checkpoint": str(args.init_checkpoint)}, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(args.seed))
    logs = []
    for step in range(int(args.train_steps)):
        idx = rng.integers(0, len(seqs), size=min(int(args.batch_size), len(seqs)))
        batch = make_batch(seqs, idx, int(args.max_steps), device)
        type_logits, target_logits, active, hidden_stack = model(
            batch["root"], batch["slots"], batch["prev_class"], batch["prev_row"], return_hidden=True
        )
        y_type = batch["y_type"]
        y_row = batch["y_row"]
        valid = y_type >= 0
        if bool(valid.any()) and float(args.type_balance_power) > 0.0:
            counts = torch.bincount(y_type[valid].clamp(0, 1), minlength=2).float()
            type_weight = (counts.sum().clamp_min(1.0) / (2.0 * counts.clamp_min(1.0))).pow(float(args.type_balance_power))
            loss_type = F.cross_entropy(type_logits[valid], y_type[valid], weight=type_weight)
        elif bool(valid.any()):
            loss_type = F.cross_entropy(type_logits[valid], y_type[valid])
        else:
            loss_type = torch.zeros((), device=device)
        track_valid = valid & (y_row > 0)
        if bool(track_valid.any()):
            loss_target = F.cross_entropy(target_logits[track_valid], y_row[track_valid])
        else:
            loss_target = torch.zeros((), device=device)
        if bool(valid.any()) and float(args.value_loss_weight) > 0.0:
            value_pred = model.value_head(hidden_stack).squeeze(-1)
            loss_value = F.mse_loss(value_pred[valid], batch["y_value"][valid])
        else:
            loss_value = torch.zeros((), device=device)
        loss = float(args.type_loss_weight) * loss_type + float(args.target_loss_weight) * loss_target + float(args.value_loss_weight) * loss_value
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, int(args.log_every)) == 0 or step == int(args.train_steps) - 1:
            with torch.no_grad():
                pred_type = type_logits.argmax(-1)
                type_acc = (pred_type[valid] == y_type[valid]).float().mean().item() if bool(valid.any()) else 0.0
                pred_row = target_logits.argmax(-1)
                target_acc = (pred_row[track_valid] == y_row[track_valid]).float().mean().item() if bool(track_valid.any()) else 0.0
            row = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "type_loss": float(loss_type.detach().cpu()),
                "target_loss": float(loss_target.detach().cpu()),
                "value_loss": float(loss_value.detach().cpu()),
                "type_acc": float(type_acc),
                "target_acc": float(target_acc),
            }
            logs.append(row)
            print(row, flush=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "reward_mean": reward_mean, "reward_std": reward_std}, ckpt)
    pd.DataFrame(logs).to_csv(ckpt.with_name(ckpt.stem + "_train_log.csv"), index=False)
    return model.eval()


class Sparse64SeqPlanner:
    def __init__(self, model: Sparse64SequenceDecoder, env_cfg: dict, max_steps: int = 32):
        self.model = model.eval()
        self.env_cfg = dict(env_cfg)
        self.max_steps = int(max_steps)
        self.adapt = adapter()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def warmup(self, obs, budget_ms=200.0):
        _ = self.plan(obs, budget_ms)

    def _slot_tensor_single(
        self,
        obs: dict,
        elapsed: float,
        search_count: int,
        track_count: int,
        last: int,
        budget_ms: float,
    ) -> torch.Tensor:
        if float(obs.get("use_grid_feature", 0.0)) > 0.5:
            slot = slot_features(obs, elapsed, search_count, track_count, last, float(budget_ms)).astype(np.float32)
            return torch.from_numpy(slot).float().unsqueeze(0).to(self.device)

        active_np = np.asarray(obs["active_mask"]).astype(bool)
        deadline_np = np.asarray(obs["t_deadline"], dtype=np.float32)
        dwell_np = np.asarray(obs["t_dwell"], dtype=np.float32)
        tracked_np = active_np & (deadline_np >= 0.0)
        workload = float(np.sum(dwell_np[tracked_np]) / max(1.0, float(budget_ms)))
        min_deadline = float(np.min(deadline_np[tracked_np & (deadline_np > 0)])) if np.any(tracked_np & (deadline_np > 0)) else 0.0
        arrival_feature = float(obs.get("enable_x_band", 0.0))
        if float(obs.get("use_arrival_feature", 0.0)) > 0.5:
            arrival_feature += np.clip(float(obs.get("arrival_rate", 0.0)) / 10.0, 0.0, 2.0)
        slot = torch.empty((1, SLOT_DIM), dtype=torch.float32, device=self.device)
        slot[0, 0] = float(elapsed) / float(budget_ms)
        slot[0, 1] = float(search_count) / 20.0
        slot[0, 2] = float(track_count) / 100.0
        slot[0, 3] = 1.0 if int(last) == 0 else 0.0
        slot[0, 4] = float(np.sum(active_np)) / 100.0
        slot[0, 5] = float(np.sum(tracked_np)) / 100.0
        slot[0, 6] = min(workload / 20.0, 2.0)
        slot[0, 7] = min_deadline / 3000.0
        slot[0, 8] = float(np.clip(float(obs.get("s_band_busy_ms", 0.0)) / 200.0, 0.0, 5.0))
        slot[0, 9] = float(np.clip(float(obs.get("x_band_busy_ms", 0.0)) / 200.0, 0.0, 5.0))
        slot[0, 10] = float(arrival_feature)
        return slot

    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        root = tokenize(self.adapt, obs, selected=set(), search_count=0).astype(np.float32)
        dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        use_fast_slot = float(obs.get("use_grid_feature", 0.0)) <= 0.5
        fast_slot_t = None
        if use_fast_slot:
            active_np = np.asarray(obs["active_mask"]).astype(bool)
            deadline_np = np.asarray(obs["t_deadline"], dtype=np.float32)
            dwell_np = np.asarray(obs["t_dwell"], dtype=np.float32)
            tracked_np = active_np & (deadline_np >= 0.0)
            workload = float(np.sum(dwell_np[tracked_np]) / max(1.0, float(budget_ms)))
            min_deadline = float(np.min(deadline_np[tracked_np & (deadline_np > 0)])) if np.any(tracked_np & (deadline_np > 0)) else 0.0
            arrival_feature = float(obs.get("enable_x_band", 0.0))
            if float(obs.get("use_arrival_feature", 0.0)) > 0.5:
                arrival_feature += np.clip(float(obs.get("arrival_rate", 0.0)) / 10.0, 0.0, 2.0)
            fast_slot_t = torch.empty((1, SLOT_DIM), dtype=torch.float32, device=self.device)
            fast_slot_t[0, 4] = float(np.sum(active_np)) / 100.0
            fast_slot_t[0, 5] = float(np.sum(tracked_np)) / 100.0
            fast_slot_t[0, 6] = min(workload / 20.0, 2.0)
            fast_slot_t[0, 7] = min_deadline / 3000.0
            fast_slot_t[0, 8] = float(np.clip(float(obs.get("s_band_busy_ms", 0.0)) / 200.0, 0.0, 5.0))
            fast_slot_t[0, 9] = float(np.clip(float(obs.get("x_band_busy_ms", 0.0)) / 200.0, 0.0, 5.0))
            fast_slot_t[0, 10] = float(arrival_feature)
        dwell_t = torch.ones(MAXT + 1, dtype=torch.float32, device=self.device)
        n_dwell = min(MAXT, int(dwell.size))
        if n_dwell > 0:
            dwell_t[1 : n_dwell + 1] = torch.from_numpy(np.maximum(dwell[:n_dwell], 1.0)).float().to(self.device)
        with torch.inference_mode():
            root_t = torch.from_numpy(root).float().unsqueeze(0).to(self.device)
            cls, tok, token_active = self.model.encode(root_t)
        selected: set[int] = set()
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        prev_class = 0
        prev_row = 0.0
        prev_class_t = torch.zeros(1, dtype=torch.long, device=self.device)
        prev_row_t = torch.zeros(1, dtype=torch.float32, device=self.device)
        hidden = None
        plan: list[int] = []
        while elapsed < float(budget_ms) and len(plan) < self.max_steps:
            if fast_slot_t is not None:
                fast_slot_t[0, 0] = float(elapsed) / float(budget_ms)
                fast_slot_t[0, 1] = float(search_count) / 20.0
                fast_slot_t[0, 2] = float(track_count) / 100.0
                fast_slot_t[0, 3] = 1.0 if int(last) == 0 else 0.0
                slot_t = fast_slot_t
            else:
                slot_t = self._slot_tensor_single(obs, elapsed, search_count, track_count, last, float(budget_ms))
            with torch.inference_mode():
                tl, tr, hidden = self.model.score_step(
                    cls,
                    tok,
                    slot_t,
                    prev_class_t,
                    prev_row_t,
                    hidden,
                )
                scores = torch.log_softmax(tr[0], dim=-1)
                type_logits = torch.log_softmax(tl[0], dim=-1)
            active = token_active[0].clone()
            active[0] = True
            for row in selected:
                if 0 <= int(row) < int(active.numel()):
                    active[int(row)] = False
            scores = scores.masked_fill(~active, -1e9)
            if dwell.size:
                remaining = max(0.0, float(budget_ms) - elapsed)
                n = min(MAXT, dwell.size, scores.numel() - 1)
                if n > 0:
                    too_long = dwell_t[1 : n + 1] > float(remaining)
                    scores[1 : n + 1] = scores[1 : n + 1].masked_fill(too_long, -1e9)
            if float(budget_ms) - elapsed < 10.0:
                search_score = torch.tensor(-1e9, device=self.device)
            else:
                search_score = type_logits[0]
            best_track = int(torch.argmax(scores).item())
            track_score = type_logits[1] + scores[best_track]
            row = 0 if float(search_score) >= float(track_score) else best_track
            if row <= 0:
                if elapsed + 10.0 > float(budget_ms) and plan:
                    break
                plan.append(xs_s_search_action(MAXT))
                elapsed += 10.0
                search_count += 1
                last = 0
                prev_class = 1
                prev_row = 0.0
                prev_class_t.fill_(1)
                prev_row_t.fill_(0.0)
            else:
                dt = float(max(1.0, dwell[row - 1] if row - 1 < len(dwell) else 5.0))
                if elapsed + dt > float(budget_ms) and plan:
                    break
                plan.append(xs_s_track_action(row, MAXT))
                selected.add(int(row))
                elapsed += dt
                track_count += 1
                last = int(row)
                prev_class = action_to_prev_class(row)
                prev_row = float(row) / float(MAXT)
                prev_class_t.fill_(int(prev_class))
                prev_row_t.fill_(float(prev_row))
        return plan if plan else [xs_s_search_action(MAXT)]


class Sparse64GraphModule(nn.Module):
    """Root encoder and complete autoregressive window decode in one graph."""

    def __init__(self, model: Sparse64SequenceDecoder, max_steps: int):
        super().__init__()
        self.model = model
        self.max_steps = int(max_steps)

    def forward(self, root, base_slot, dwell, budget):
        cls, tok, token_active = self.model.encode(root)
        selected = torch.zeros_like(token_active)
        elapsed = torch.zeros_like(budget)
        search_count = torch.zeros_like(budget)
        track_count = torch.zeros_like(budget)
        last_search = torch.zeros_like(budget)
        prev_class = torch.zeros((root.shape[0],), dtype=torch.long, device=root.device)
        prev_row = torch.zeros_like(budget)
        hidden = None
        rows = []
        for _ in range(self.max_steps):
            slot = torch.stack(
                [elapsed / budget.clamp_min(1.0), search_count / 20.0, track_count / 100.0, last_search,
                 base_slot[:, 4], base_slot[:, 5], base_slot[:, 6], base_slot[:, 7],
                 base_slot[:, 8], base_slot[:, 9], base_slot[:, 10]], dim=1
            )
            type_logits, target_logits, hidden = self.model.score_step(cls, tok, slot, prev_class, prev_row, hidden)
            remaining = budget - elapsed
            # Match the deployed soft-window decoder: the final action may
            # overshoot the nominal boundary, so dwell is not hard-masked by
            # the remaining budget.
            valid = token_active & ~selected
            valid[:, 0] = False
            target_logp = torch.log_softmax(target_logits, dim=-1).masked_fill(~valid, -1.0e9)
            best_track = target_logp.argmax(dim=-1)
            best_track_score = target_logp.gather(1, best_track[:, None]).squeeze(1)
            type_logp = torch.log_softmax(type_logits, dim=-1)
            search_score = type_logp[:, 0]
            track_score = type_logp[:, 1] + best_track_score
            active_step = elapsed < budget
            choose_track = active_step & valid.any(dim=-1) & (track_score > search_score)
            row = torch.where(choose_track, best_track, torch.zeros_like(best_track))
            rows.append(torch.where(active_step, row, torch.full_like(row, -1)))
            selected = selected | (F.one_hot(row.clamp(min=0), MAXT + 1).bool() & choose_track[:, None])
            track_dt = dwell.gather(1, row[:, None]).squeeze(1).clamp_min(1.0)
            dt = torch.where(choose_track, track_dt, torch.full_like(elapsed, 10.0))
            elapsed = elapsed + torch.where(active_step, dt, torch.zeros_like(dt))
            search_count = search_count + (active_step & ~choose_track).to(elapsed.dtype)
            track_count = track_count + choose_track.to(elapsed.dtype)
            last_search = (active_step & ~choose_track).to(elapsed.dtype)
            prev_class = torch.where(choose_track, torch.full_like(prev_class, 3), torch.ones_like(prev_class))
            prev_row = row.to(elapsed.dtype) / float(MAXT)
        return torch.stack(rows, dim=1)


class Sparse64CudaGraphRunner:
    def __init__(self, model: Sparse64SequenceDecoder, max_steps: int, device: torch.device):
        self.module = Sparse64GraphModule(model, max_steps).eval()
        self.root = torch.zeros((1, MAXT + 1, TOKEN_DIM), dtype=torch.float32, device=device)
        self.slot = torch.zeros((1, SLOT_DIM), dtype=torch.float32, device=device)
        self.dwell = torch.full((1, MAXT + 1), 1.0e9, dtype=torch.float32, device=device)
        self.budget = torch.full((1,), 200.0, dtype=torch.float32, device=device)
        with torch.inference_mode():
            for _ in range(6):
                self.rows = self.module(self.root, self.slot, self.dwell, self.budget)
            torch.cuda.synchronize(device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.rows = self.module(self.root, self.slot, self.dwell, self.budget)

    def __call__(self, root, slot, dwell, budget):
        self.root.copy_(root)
        self.slot.copy_(slot)
        self.dwell.copy_(dwell)
        self.budget.copy_(budget)
        self.graph.replay()
        return self.rows


class Sparse64CudaGraphPlanner(Sparse64SeqPlanner):
    def __init__(self, model: Sparse64SequenceDecoder, env_cfg: dict, max_steps: int = 32):
        super().__init__(model, env_cfg, max_steps)
        self.runner = Sparse64CudaGraphRunner(model, max_steps, self.device)

    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        root_np = tokenize(self.adapt, obs, selected=set(), search_count=0).astype(np.float32)
        root = torch.from_numpy(root_np[None]).to(self.device)
        slot = self._slot_tensor_single(obs, 0.0, 0, 0, -1, float(budget_ms))
        dwell_np = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        dwell_full = np.full((MAXT + 1,), 1.0e9, dtype=np.float32)
        count = min(MAXT, len(dwell_np))
        if count:
            dwell_full[1 : count + 1] = np.maximum(dwell_np[:count], 1.0)
        dwell = torch.from_numpy(dwell_full[None]).to(self.device)
        budget = torch.full((1,), float(budget_ms), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            decoded = self.runner(root, slot, dwell, budget)[0].detach().cpu().tolist()
        plan = [xs_s_search_action(MAXT) if int(row) <= 0 else xs_s_track_action(int(row), MAXT)
                for row in decoded if int(row) >= 0]
        return plan or [xs_s_search_action(MAXT)]


class Sparse64SeqBeamPlanner(Sparse64SeqPlanner):
    """Batched prefix beam decoder for the sequence-distilled policy.

    The beam uses policy likelihood only. It does not call the simulator for
    exact reranking and it does not impose search floors/caps.
    """

    def __init__(
        self,
        model: Sparse64SequenceDecoder,
        env_cfg: dict,
        max_steps: int = 32,
        beam_width: int = 4,
        target_k: int = 4,
        value_weight: float = 0.0,
        length_norm_alpha: float = 0.0,
    ):
        super().__init__(model, env_cfg, max_steps)
        self.beam_width = max(1, int(beam_width))
        self.target_k = max(1, int(target_k))
        self.value_weight = float(value_weight)
        self.length_norm_alpha = float(length_norm_alpha)

    def _rank_score(self, beam: dict) -> float:
        if self.length_norm_alpha == 0.0:
            return float(beam["score"])
        # Common sequence-decoder length normalization. It changes only beam
        # search ranking, not action validity or search/track quotas.
        length = max(1, len(beam.get("plan", [])))
        norm = ((5.0 + float(length)) / 6.0) ** self.length_norm_alpha
        return float(beam["score"]) / float(norm)

    def _beam_slots(self, obs: dict, beams: list[dict], budget_ms: float) -> np.ndarray:
        """Vectorized SLOT_DIM features for beam prefixes in a fixed window."""
        if float(obs.get("use_grid_feature", 0.0)) > 0.5:
            return np.stack(
                [
                    slot_features(
                        obs,
                        float(b["elapsed"]),
                        int(b["search_count"]),
                        int(b["track_count"]),
                        int(b["last"]),
                        float(budget_ms),
                    ).astype(np.float32)
                    for b in beams
                ],
                axis=0,
            )
        active = np.asarray(obs["active_mask"]).astype(bool)
        deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
        dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
        tracked = active & (deadline >= 0.0)
        workload = float(np.sum(dwell[tracked]) / max(1.0, float(budget_ms)))
        min_deadline = float(np.min(deadline[tracked & (deadline > 0)])) if np.any(tracked & (deadline > 0)) else 0.0
        s_busy = float(obs.get("s_band_busy_ms", 0.0))
        x_busy = float(obs.get("x_band_busy_ms", 0.0))
        arrival_feature = float(obs.get("enable_x_band", 0.0))
        if float(obs.get("use_arrival_feature", 0.0)) > 0.5:
            arrival_feature += np.clip(float(obs.get("arrival_rate", 0.0)) / 10.0, 0.0, 2.0)
        slots = np.empty((len(beams), SLOT_DIM), dtype=np.float32)
        active_norm = float(np.sum(active)) / 100.0
        tracked_norm = float(np.sum(tracked)) / 100.0
        workload_norm = min(workload / 20.0, 2.0)
        min_deadline_norm = min_deadline / 3000.0
        s_busy_norm = np.clip(s_busy / 200.0, 0.0, 5.0)
        x_busy_norm = np.clip(x_busy / 200.0, 0.0, 5.0)
        for i, b in enumerate(beams):
            slots[i, 0] = float(b["elapsed"]) / float(budget_ms)
            slots[i, 1] = float(b["search_count"]) / 20.0
            slots[i, 2] = float(b["track_count"]) / 100.0
            slots[i, 3] = 1.0 if int(b["last"]) == 0 else 0.0
            slots[i, 4] = active_norm
            slots[i, 5] = tracked_norm
            slots[i, 6] = workload_norm
            slots[i, 7] = min_deadline_norm
            slots[i, 8] = s_busy_norm
            slots[i, 9] = x_busy_norm
            slots[i, 10] = arrival_feature
        return slots

    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        root = tokenize(self.adapt, obs, selected=set(), search_count=0).astype(np.float32)
        dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        with torch.inference_mode():
            root_t = torch.from_numpy(root).float().unsqueeze(0).to(self.device)
            cls, tok, token_active = self.model.encode(root_t)

        beams = [
            {
                "score": 0.0,
                "plan": [],
                "selected": set(),
                "elapsed": 0.0,
                "search_count": 0,
                "track_count": 0,
                "last": -1,
                "prev_class": 0,
                "prev_row": 0.0,
                "hidden": None,
                "done": False,
            }
        ]
        for _step in range(self.max_steps):
            active_beams = [b for b in beams if not b["done"] and float(b["elapsed"]) < float(budget_ms)]
            if not active_beams:
                break
            slots = self._beam_slots(obs, active_beams, float(budget_ms))
            prev_class = torch.tensor([int(b["prev_class"]) for b in active_beams], dtype=torch.long, device=self.device)
            prev_row = torch.tensor([float(b["prev_row"]) for b in active_beams], dtype=torch.float32, device=self.device)
            hidden_items = [b["hidden"] for b in active_beams]
            hidden = None
            if all(h is not None for h in hidden_items):
                hidden = torch.cat(hidden_items, dim=0)
            with torch.inference_mode():
                cls_b = cls.expand(len(active_beams), -1)
                tok_b = tok.expand(len(active_beams), -1, -1)
                slot_b = torch.from_numpy(slots).float().to(self.device)
                type_logits, target_logits, next_hidden = self.model.score_step(cls_b, tok_b, slot_b, prev_class, prev_row, hidden)
                type_lp = torch.log_softmax(type_logits, dim=-1)
                value_pred = (
                    self.model.value_head(next_hidden).squeeze(-1)
                    if self.value_weight != 0.0
                    else None
                )

            candidates = [dict(b) for b in beams if b["done"] or float(b["elapsed"]) >= float(budget_ms)]
            for bi, b in enumerate(active_beams):
                remaining = max(0.0, float(budget_ms) - float(b["elapsed"]))
                base_score = float(b["score"])
                h_next = next_hidden[bi : bi + 1].detach()

                if remaining >= 10.0 or not b["plan"]:
                    nb = dict(b)
                    nb["score"] = base_score + float(type_lp[bi, 0].detach().cpu())
                    if value_pred is not None:
                        nb["score"] += self.value_weight * float(value_pred[bi].detach().cpu())
                    nb["plan"] = list(b["plan"]) + [xs_s_search_action(MAXT)]
                    nb["elapsed"] = float(b["elapsed"]) + 10.0
                    nb["search_count"] = int(b["search_count"]) + 1
                    nb["last"] = 0
                    nb["prev_class"] = 1
                    nb["prev_row"] = 0.0
                    nb["hidden"] = h_next
                    nb["done"] = bool(float(nb["elapsed"]) >= float(budget_ms))
                    candidates.append(nb)

                valid = token_active[0].clone()
                valid[0] = False
                for row in b["selected"]:
                    if 0 <= int(row) < int(valid.numel()):
                        valid[int(row)] = False
                if dwell.size:
                    n = min(MAXT, dwell.size, valid.numel() - 1)
                    if n > 0:
                        too_long_np = dwell[:n] > remaining
                        if bool(np.any(too_long_np)):
                            valid[1 : n + 1] = valid[1 : n + 1] & ~torch.from_numpy(too_long_np).to(self.device)
                masked_targets = target_logits[bi].masked_fill(~valid, -1e9)
                target_lp = torch.log_softmax(masked_targets, dim=-1)
                k = min(self.target_k, int(valid.sum().detach().cpu()))
                if k > 0:
                    top = torch.topk(target_lp, k=k).indices.detach().cpu().tolist()
                    for row in top:
                        row = int(row)
                        if row <= 0:
                            continue
                        dt = float(max(1.0, dwell[row - 1] if row - 1 < len(dwell) else 5.0))
                        if float(b["elapsed"]) + dt > float(budget_ms) and b["plan"]:
                            continue
                        nb = dict(b)
                        nb["score"] = base_score + float(type_lp[bi, 1].detach().cpu()) + float(target_lp[row].detach().cpu())
                        if value_pred is not None:
                            nb["score"] += self.value_weight * float(value_pred[bi].detach().cpu())
                        nb["plan"] = list(b["plan"]) + [xs_s_track_action(row, MAXT)]
                        nb["selected"] = set(b["selected"])
                        nb["selected"].add(row)
                        nb["elapsed"] = float(b["elapsed"]) + dt
                        nb["track_count"] = int(b["track_count"]) + 1
                        nb["last"] = row
                        nb["prev_class"] = action_to_prev_class(row)
                        nb["prev_row"] = float(row) / float(MAXT)
                        nb["hidden"] = h_next
                        nb["done"] = bool(float(nb["elapsed"]) >= float(budget_ms))
                        candidates.append(nb)
            if not candidates:
                break
            candidates.sort(key=lambda x: (self._rank_score(x), len(x["plan"])), reverse=True)
            beams = candidates[: self.beam_width]
        best = max(beams, key=lambda x: (self._rank_score(x), len(x["plan"]))) if beams else None
        if best is None or not best["plan"]:
            return [xs_s_search_action(MAXT)]
        return list(best["plan"])


class Sparse64SeqTensorBeamPlanner(Sparse64SeqBeamPlanner):
    """Tensorized beam decoder with the same policy ranking as Sparse64SeqBeamPlanner.

    This keeps selected-target masks, prefix rows, and candidate ranking in tensor
    form. It is intended to measure the deployable lower bound of the current
    prefix-search formulation without changing the learned model or adding
    heuristic gates.
    """

    def _slot_tensor(
        self,
        obs: dict,
        elapsed: torch.Tensor,
        search_count: torch.Tensor,
        track_count: torch.Tensor,
        last: torch.Tensor,
        budget_ms: float,
    ) -> torch.Tensor:
        if float(obs.get("use_grid_feature", 0.0)) > 0.5:
            beams = [
                {
                    "elapsed": float(elapsed[i].detach().cpu()),
                    "search_count": int(search_count[i].detach().cpu()),
                    "track_count": int(track_count[i].detach().cpu()),
                    "last": int(last[i].detach().cpu()),
                }
                for i in range(int(elapsed.numel()))
            ]
            return torch.from_numpy(self._beam_slots(obs, beams, budget_ms)).float().to(self.device)

        active_np = np.asarray(obs["active_mask"]).astype(bool)
        deadline_np = np.asarray(obs["t_deadline"], dtype=np.float32)
        dwell_np = np.asarray(obs["t_dwell"], dtype=np.float32)
        tracked_np = active_np & (deadline_np >= 0.0)
        workload = float(np.sum(dwell_np[tracked_np]) / max(1.0, float(budget_ms)))
        min_deadline = float(np.min(deadline_np[tracked_np & (deadline_np > 0)])) if np.any(tracked_np & (deadline_np > 0)) else 0.0
        arrival_feature = float(obs.get("enable_x_band", 0.0))
        if float(obs.get("use_arrival_feature", 0.0)) > 0.5:
            arrival_feature += np.clip(float(obs.get("arrival_rate", 0.0)) / 10.0, 0.0, 2.0)

        n = int(elapsed.numel())
        slots = torch.empty((n, SLOT_DIM), dtype=torch.float32, device=self.device)
        slots[:, 0] = elapsed / float(budget_ms)
        slots[:, 1] = search_count.float() / 20.0
        slots[:, 2] = track_count.float() / 100.0
        slots[:, 3] = (last == 0).float()
        slots[:, 4] = float(np.sum(active_np)) / 100.0
        slots[:, 5] = float(np.sum(tracked_np)) / 100.0
        slots[:, 6] = min(workload / 20.0, 2.0)
        slots[:, 7] = min_deadline / 3000.0
        slots[:, 8] = float(np.clip(float(obs.get("s_band_busy_ms", 0.0)) / 200.0, 0.0, 5.0))
        slots[:, 9] = float(np.clip(float(obs.get("x_band_busy_ms", 0.0)) / 200.0, 0.0, 5.0))
        slots[:, 10] = float(arrival_feature)
        return slots

    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        if self.value_weight != 0.0 or self.length_norm_alpha != 0.0:
            return super().plan(obs, budget_ms=budget_ms)

        obs = attach_env_obs(obs, self.env_cfg, True, True)
        if float(obs.get("use_grid_feature", 0.0)) > 0.5:
            return super().plan(obs, budget_ms=budget_ms)

        root = tokenize(self.adapt, obs, selected=set(), search_count=0).astype(np.float32)
        dwell_np = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        dwell = torch.ones(MAXT + 1, dtype=torch.float32, device=self.device)
        n_dwell = min(MAXT, int(dwell_np.size))
        if n_dwell > 0:
            dwell[1 : n_dwell + 1] = torch.from_numpy(np.maximum(dwell_np[:n_dwell], 1.0)).float().to(self.device)

        with torch.inference_mode():
            root_t = torch.from_numpy(root).float().unsqueeze(0).to(self.device)
            cls, tok, token_active = self.model.encode(root_t)

            bw = int(self.beam_width)
            max_steps = int(self.max_steps)
            score = torch.zeros(1, dtype=torch.float32, device=self.device)
            elapsed = torch.zeros(1, dtype=torch.float32, device=self.device)
            search_count = torch.zeros(1, dtype=torch.long, device=self.device)
            track_count = torch.zeros(1, dtype=torch.long, device=self.device)
            last = torch.full((1,), -1, dtype=torch.long, device=self.device)
            prev_class = torch.zeros(1, dtype=torch.long, device=self.device)
            prev_row = torch.zeros(1, dtype=torch.float32, device=self.device)
            hidden = None
            selected = torch.zeros((1, MAXT + 1), dtype=torch.bool, device=self.device)
            plan_rows = torch.full((1, max_steps), -1, dtype=torch.long, device=self.device)
            plan_len = torch.zeros(1, dtype=torch.long, device=self.device)

            for step in range(max_steps):
                bsz = int(score.numel())
                active_beam = elapsed < float(budget_ms)
                if not bool(active_beam.any().detach().cpu()):
                    break
                active_idx = torch.nonzero(active_beam, as_tuple=False).flatten()
                done_idx = torch.nonzero(~active_beam, as_tuple=False).flatten()

                slots = self._slot_tensor(
                    obs,
                    elapsed[active_idx],
                    search_count[active_idx],
                    track_count[active_idx],
                    last[active_idx],
                    float(budget_ms),
                )
                cls_b = cls.expand(int(active_idx.numel()), -1)
                tok_b = tok.expand(int(active_idx.numel()), -1, -1)
                hidden_b = hidden.index_select(0, active_idx) if hidden is not None else None
                type_logits, target_logits, next_hidden = self.model.score_step(
                    cls_b,
                    tok_b,
                    slots,
                    prev_class[active_idx],
                    prev_row[active_idx],
                    hidden_b,
                )
                type_lp = torch.log_softmax(type_logits, dim=-1)

                cand_score = []
                cand_elapsed = []
                cand_search = []
                cand_track = []
                cand_last = []
                cand_prev_class = []
                cand_prev_row = []
                cand_hidden = []
                cand_selected = []
                cand_plan = []
                cand_len = []

                if int(done_idx.numel()) > 0:
                    cand_score.append(score[done_idx])
                    cand_elapsed.append(elapsed[done_idx])
                    cand_search.append(search_count[done_idx])
                    cand_track.append(track_count[done_idx])
                    cand_last.append(last[done_idx])
                    cand_prev_class.append(prev_class[done_idx])
                    cand_prev_row.append(prev_row[done_idx])
                    if hidden is None:
                        d_h = cls.expand(int(done_idx.numel()), -1)
                    else:
                        d_h = hidden.index_select(0, done_idx)
                    cand_hidden.append(d_h)
                    cand_selected.append(selected.index_select(0, done_idx))
                    cand_plan.append(plan_rows.index_select(0, done_idx))
                    cand_len.append(plan_len[done_idx])

                # Search candidate for every active beam.
                remaining = float(budget_ms) - elapsed[active_idx]
                can_search = (remaining >= 10.0) | (plan_len[active_idx] == 0)
                if bool(can_search.any().detach().cpu()):
                    si = torch.nonzero(can_search, as_tuple=False).flatten()
                    src = active_idx.index_select(0, si)
                    p = plan_rows.index_select(0, src).clone()
                    p[torch.arange(int(si.numel()), device=self.device), plan_len[src].clamp(max=max_steps - 1)] = 0
                    cand_score.append(score[src] + type_lp[si, 0])
                    cand_elapsed.append(elapsed[src] + 10.0)
                    cand_search.append(search_count[src] + 1)
                    cand_track.append(track_count[src])
                    cand_last.append(torch.zeros(int(si.numel()), dtype=torch.long, device=self.device))
                    cand_prev_class.append(torch.ones(int(si.numel()), dtype=torch.long, device=self.device))
                    cand_prev_row.append(torch.zeros(int(si.numel()), dtype=torch.float32, device=self.device))
                    cand_hidden.append(next_hidden.index_select(0, si))
                    cand_selected.append(selected.index_select(0, src))
                    cand_plan.append(p)
                    cand_len.append((plan_len[src] + 1).clamp(max=max_steps))

                # Track candidates: top-k valid targets per active beam.
                valid = token_active[0, : MAXT + 1].view(1, -1).expand(int(active_idx.numel()), -1).clone()
                valid[:, 0] = False
                valid &= ~selected.index_select(0, active_idx)
                valid &= dwell.view(1, -1) <= remaining.view(-1, 1)
                target_lp = torch.log_softmax(target_logits[:, : MAXT + 1].masked_fill(~valid, -1e9), dim=-1)
                k = min(self.target_k, MAXT)
                top_vals, top_rows = torch.topk(target_lp, k=k, dim=-1)
                flat_rows = top_rows.reshape(-1)
                flat_vals = top_vals.reshape(-1)
                flat_src_local = torch.arange(int(active_idx.numel()), device=self.device).repeat_interleave(k)
                good = flat_rows > 0
                if bool(good.any().detach().cpu()):
                    si = flat_src_local[good]
                    rows = flat_rows[good]
                    src = active_idx.index_select(0, si)
                    dt = dwell.index_select(0, rows)
                    fits = (elapsed[src] + dt <= float(budget_ms)) | (plan_len[src] == 0)
                    if bool(fits.any().detach().cpu()):
                        si = si[fits]
                        rows = rows[fits]
                        src = src[fits]
                        vals = flat_vals[good][fits]
                        dt = dt[fits]
                        p = plan_rows.index_select(0, src).clone()
                        p[torch.arange(int(src.numel()), device=self.device), plan_len[src].clamp(max=max_steps - 1)] = rows
                        sel = selected.index_select(0, src).clone()
                        sel[torch.arange(int(src.numel()), device=self.device), rows] = True
                        cand_score.append(score[src] + type_lp[si, 1] + vals)
                        cand_elapsed.append(elapsed[src] + dt)
                        cand_search.append(search_count[src])
                        cand_track.append(track_count[src] + 1)
                        cand_last.append(rows)
                        cand_prev_class.append(torch.full((int(src.numel()),), 3, dtype=torch.long, device=self.device))
                        cand_prev_row.append(rows.float() / float(MAXT))
                        cand_hidden.append(next_hidden.index_select(0, si))
                        cand_selected.append(sel)
                        cand_plan.append(p)
                        cand_len.append((plan_len[src] + 1).clamp(max=max_steps))

                if not cand_score:
                    break

                score_all = torch.cat(cand_score, dim=0)
                elapsed_all = torch.cat(cand_elapsed, dim=0)
                search_all = torch.cat(cand_search, dim=0)
                track_all = torch.cat(cand_track, dim=0)
                last_all = torch.cat(cand_last, dim=0)
                prev_class_all = torch.cat(cand_prev_class, dim=0)
                prev_row_all = torch.cat(cand_prev_row, dim=0)
                hidden_all = torch.cat(cand_hidden, dim=0)
                selected_all = torch.cat(cand_selected, dim=0)
                plan_all = torch.cat(cand_plan, dim=0)
                len_all = torch.cat(cand_len, dim=0)

                rank = score_all + 1e-4 * len_all.float()
                keep = torch.topk(rank, k=min(bw, int(rank.numel())), dim=0).indices
                score = score_all.index_select(0, keep)
                elapsed = elapsed_all.index_select(0, keep)
                search_count = search_all.index_select(0, keep)
                track_count = track_all.index_select(0, keep)
                last = last_all.index_select(0, keep)
                prev_class = prev_class_all.index_select(0, keep)
                prev_row = prev_row_all.index_select(0, keep)
                hidden = hidden_all.index_select(0, keep)
                selected = selected_all.index_select(0, keep)
                plan_rows = plan_all.index_select(0, keep)
                plan_len = len_all.index_select(0, keep)

            if int(score.numel()) == 0:
                return [xs_s_search_action(MAXT)]
            best = int(torch.argmax(score + 1e-4 * plan_len.float()).detach().cpu())
            n = int(plan_len[best].detach().cpu())
            if n <= 0:
                return [xs_s_search_action(MAXT)]
            rows = plan_rows[best, :n].detach().cpu().tolist()
        out = []
        for row in rows:
            row = int(row)
            out.append(xs_s_search_action(MAXT) if row <= 0 else xs_s_track_action(row, MAXT))
        return out if out else [xs_s_search_action(MAXT)]


class SingleSensorHeuristicAdapter:
    def __init__(self, planner):
        self.planner = planner

    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        out = []
        for a in list(self.planner.plan(obs, budget_ms=budget_ms))[:64]:
            a = int(a)
            out.append(xs_s_search_action(MAXT) if a == 0 else xs_s_track_action(a, MAXT))
        return out


def evaluate(args: argparse.Namespace, ckpt: Path, out_csv: Path) -> pd.DataFrame:
    device = torch.device(args.device)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model = Sparse64SequenceDecoder(d_model=int(args.d_model), nhead=int(args.nhead), nlayers=int(args.nlayers)).to(device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print({"load_state_dict": "non_strict", "missing": list(missing), "unexpected": list(unexpected)}, flush=True)
    model.eval()
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = False
    exact_args.single_sensor = True
    windows = []
    summaries = []
    for initial in parse_ints(args.eval_initials):
        for rate in parse_floats(args.eval_rates):
            env_cfg = env_cfg_for(float(rate), exact_args)
            env_cfg["enable_x_band"] = 0
            direct_cls = Sparse64CudaGraphPlanner if bool(args.cuda_graph) and device.type == "cuda" else Sparse64SeqPlanner
            planners = [
                ("Sparse64SeqDistill", direct_cls(model, env_cfg, int(args.max_steps))),
                ("EDF", SingleSensorHeuristicAdapter(EDFPlanner(MAXT))),
                ("EST", SingleSensorHeuristicAdapter(ESTPlanner(MAXT))),
            ]
            if int(getattr(args, "beam_width", 1)) > 1:
                beam_cls = Sparse64SeqTensorBeamPlanner if bool(getattr(args, "tensor_beam", False)) else Sparse64SeqBeamPlanner
                beam_name = f"Sparse64SeqTensorBeam{int(args.beam_width)}" if bool(getattr(args, "tensor_beam", False)) else f"Sparse64SeqBeam{int(args.beam_width)}"
                planners.insert(
                    1,
                    (
                        beam_name,
                        beam_cls(
                            model,
                            env_cfg,
                            int(args.max_steps),
                            beam_width=int(args.beam_width),
                            target_k=int(args.beam_target_k),
                            value_weight=float(args.beam_value_weight),
                            length_norm_alpha=float(args.beam_length_norm_alpha),
                        ),
                    ),
                )
            for seed in parse_ints(args.eval_seeds):
                for name, planner in planners:
                    df, _actions = run_plan_eval(planner, name, int(initial), int(seed), int(args.eval_windows), env_cfg)
                    df = df.copy()
                    df["initial"] = int(initial)
                    df["rate"] = float(rate)
                    df["seed"] = int(seed)
                    windows.append(df)
                    summaries.append({"planner": name, "initial": int(initial), "rate": float(rate), "seed": int(seed), **summarize(df)})
                    print(summaries[-1], flush=True)
    all_w = pd.concat(windows, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    all_w.to_csv(out_csv, index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out_csv.with_name(out_csv.stem + "_summary.csv"), index=False)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["collect", "train", "eval", "all"], default="all")
    ap.add_argument("--base-state", default=str(DEFAULT_STATE))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--data", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--init-checkpoint", default="")
    ap.add_argument("--initials", default="20,40,60")
    ap.add_argument("--rates", default="2,3,4")
    ap.add_argument("--seeds", default="916")
    ap.add_argument("--windows", type=int, default=100)
    ap.add_argument("--eval-initials", default="20,40,60")
    ap.add_argument("--eval-rates", default="2,3,4")
    ap.add_argument("--eval-seeds", default="916")
    ap.add_argument("--eval-windows", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--train-steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--type-balance-power", type=float, default=0.0)
    ap.add_argument("--beam-width", type=int, default=1)
    ap.add_argument("--beam-target-k", type=int, default=4)
    ap.add_argument("--beam-value-weight", type=float, default=0.0)
    ap.add_argument("--beam-length-norm-alpha", type=float, default=0.0)
    ap.add_argument("--tensor-beam", action="store_true")
    ap.add_argument("--cuda-graph", action="store_true", help="Capture root encoding and the complete AR decode loop on CUDA.")
    ap.add_argument("--value-loss-weight", type=float, default=0.0)
    ap.add_argument("--type-loss-weight", type=float, default=1.0)
    ap.add_argument("--target-loss-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--teacher-policy-weight", type=float, default=1.0)
    ap.add_argument("--teacher-q-weight", type=float, default=0.5)
    ap.add_argument("--action-coupler-top-k", type=int, default=64)
    ap.add_argument("--sparse-residuals", action="store_true")
    ap.add_argument("--env-mode", default="pufferlib_service")
    ap.add_argument("--search-frame-overdue-weight", type=float, default=0.5)
    ap.add_argument("--search-frame-drop-penalty", type=float, default=8.0)
    ap.add_argument("--search-frame-state-penalty-weight", type=float, default=2.0)
    ap.add_argument("--search-frame-delta-reward-weight", type=float, default=5.0)
    ap.add_argument("--service-pressure-delta-reward-weight", type=float, default=0.30)
    ap.add_argument("--serviced-pressure-improvement-reward-weight", type=float, default=0.15)
    ap.add_argument("--discovered-target-reward", type=float, default=0.08)
    args = ap.parse_args()
    torch.set_num_threads(max(1, int(args.torch_threads)))
    torch.set_num_interop_threads(1)
    out_dir = Path(args.out_dir)
    data = Path(args.data) if args.data else out_dir / "teacher_sequences.pt"
    ckpt = Path(args.ckpt) if args.ckpt else out_dir / "seqdistill.pt"
    if args.mode in {"collect", "all"}:
        data = collect_dataset(args, data)
    if args.mode in {"train", "all"}:
        train(args, data, ckpt)
    if args.mode in {"eval", "all"}:
        summary = evaluate(args, ckpt, out_dir / "seqdistill_eval.csv")
        cols = ["reward_per_window", "drop_pct_active", "tracked_targets", "mean_delay_active", "search_fraction", "planning_ms_per_window"]
        print(summary.groupby("planner")[cols].mean().round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
