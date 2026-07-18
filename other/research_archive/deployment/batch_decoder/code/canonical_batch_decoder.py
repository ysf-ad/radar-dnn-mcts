from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_action_attention_muzero_g import run_plan_eval, summarize
from exact_env_mutual import MAXT, env_cfg_for, xs_s_search_action, xs_s_track_action
from foundation_mcts_fair_eval import parse_floats, parse_ints
from mutual_features import SLOT_DIM, TOKEN_DIM, slot_features, tokenize
from penalty_window_quota_learner_eval import make_exact_args
from realistic_reward_retrain import adapter


class BatchScheduleDecoder(nn.Module):
    """One root encoding and one parallel decode for the full 200 ms window."""

    def __init__(
        self,
        max_steps: int = 20,
        d_model: int = 64,
        nhead: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        refinement_passes: int = 1,
        use_target_bias: bool = True,
        decoder_kind: str = "attention",
    ):
        super().__init__()
        self.max_steps = int(max_steps)
        self.d_model = int(d_model)
        self.refinement_passes = max(1, int(refinement_passes))
        self.use_target_bias = bool(use_target_bias)
        self.decoder_kind = str(decoder_kind)
        self.token_proj = nn.Linear(TOKEN_DIM, d_model)
        self.cls_token = nn.Parameter(torch.randn(d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers, enable_nested_tensor=False)
        self.root_slot_proj = nn.Sequential(nn.LayerNorm(SLOT_DIM), nn.Linear(SLOT_DIM, d_model), nn.GELU())
        self.slot_pos = nn.Parameter(torch.randn(max_steps, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        if self.decoder_kind == "attention":
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
            self.independent_decoder = None
            self.cartesian_encoder = None
            self.action_encoder = None
        elif self.decoder_kind == "action_only":
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
            self.independent_decoder = None
            self.cartesian_encoder = None
            action_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=2 * d_model,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.action_encoder = nn.TransformerEncoder(
                action_layer,
                num_layers=decoder_layers,
                enable_nested_tensor=False,
            )
        elif self.decoder_kind == "independent":
            self.decoder = None
            self.independent_decoder = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, 2 * d_model),
                    nn.GELU(),
                    nn.Linear(2 * d_model, d_model),
                )
                for _ in range(decoder_layers)
            ])
            self.cartesian_encoder = None
            self.action_encoder = None
        elif self.decoder_kind == "cartesian":
            self.decoder = None
            self.independent_decoder = None
            self.action_encoder = None
            cartesian_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=2 * d_model,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.cartesian_encoder = nn.TransformerEncoder(
                cartesian_layer,
                num_layers=decoder_layers,
                enable_nested_tensor=False,
            )
            self.cartesian_type_embed = nn.Parameter(torch.randn(2, d_model) * 0.02)
            self.cartesian_pair_proj = nn.Sequential(
                nn.LayerNorm(3 * d_model),
                nn.Linear(3 * d_model, 2 * d_model),
                nn.GELU(),
                nn.Linear(2 * d_model, d_model),
            )
            self.cartesian_global_proj = nn.Sequential(
                nn.LayerNorm(2 * d_model),
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
            )
            self.cartesian_type_head = nn.Sequential(
                nn.LayerNorm(2 * d_model),
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2),
            )
            self.cartesian_target_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
        else:
            raise ValueError(f"unknown decoder kind: {self.decoder_kind}")
        self.type_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.slot_target = nn.Linear(d_model, d_model)
        self.token_target = nn.Linear(d_model, d_model)
        self.target_bias = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.feedback = nn.Sequential(
            nn.LayerNorm(2 * d_model + 2),
            nn.Linear(2 * d_model + 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def _heads(self, decoded: torch.Tensor, target_tokens: torch.Tensor, track_valid: torch.Tensor):
        type_logits = self.type_head(decoded)
        q = self.slot_target(decoded)
        k = self.token_target(target_tokens)
        target_logits = torch.einsum("bsd,brd->bsr", q, k) / math.sqrt(float(self.d_model))
        if self.use_target_bias:
            slot_rep = decoded[:, :, None, :].expand(-1, -1, target_tokens.shape[1], -1)
            token_rep = target_tokens[:, None, :, :].expand(-1, decoded.shape[1], -1, -1)
            target_logits = target_logits + self.target_bias(torch.cat([slot_rep, token_rep], dim=-1)).squeeze(-1)
        invalid_score = torch.finfo(target_logits.dtype).min
        target_logits = target_logits.masked_fill(~track_valid[:, None, :], invalid_score)
        return type_logits, target_logits

    def forward(self, x: torch.Tensor, root_slot: torch.Tensor, return_all: bool = False):
        token_active = x[:, :, 4] > 0.5
        sensor_valid = x[:, :, 10] > 0.5
        token_active = token_active & sensor_valid
        token_active[:, 0] = True
        token_emb = self.token_proj(x)
        cls = self.cls_token[None, None, :].expand(x.shape[0], 1, -1)
        memory_in = torch.cat([cls, token_emb], dim=1)
        memory_valid = torch.cat(
            [torch.ones((x.shape[0], 1), dtype=torch.bool, device=x.device), token_active],
            dim=1,
        )
        memory = self.encoder(memory_in, src_key_padding_mask=~memory_valid)
        cls_out = memory[:, 0]
        target_tokens = memory[:, 1:]
        if self.action_encoder is not None:
            memory = self.action_encoder(memory, src_key_padding_mask=~memory_valid)
            cls_out = memory[:, 0]
            target_tokens = memory[:, 1:]
        root_context = cls_out[:, None, :] + self.root_slot_proj(root_slot)[:, None, :]
        base_queries = root_context + self.slot_pos[None, :, :]
        track_valid = token_active.clone()
        track_valid[:, 0] = False
        if self.decoder_kind == "cartesian":
            bsz, rows, dim = target_tokens.shape
            steps = self.max_steps
            slot_tokens = self.slot_pos[None, :, None, :].expand(bsz, steps, rows, -1)
            action_tokens = target_tokens[:, None, :, :].expand(-1, steps, -1, -1)
            type_tokens = self.cartesian_type_embed[1][None, None, None, :].expand(bsz, steps, rows, -1).clone()
            type_tokens[:, :, 0, :] = self.cartesian_type_embed[0]
            pair_tokens = self.cartesian_pair_proj(torch.cat([slot_tokens, action_tokens, type_tokens], dim=-1))
            global_token = self.cartesian_global_proj(torch.cat([cls_out, self.root_slot_proj(root_slot)], dim=-1))
            candidate_valid = track_valid.clone()
            candidate_valid[:, 0] = True
            pair_valid = candidate_valid[:, None, :].expand(-1, steps, -1)
            flat_pairs = pair_tokens.reshape(bsz, steps * rows, dim)
            flat_valid = pair_valid.reshape(bsz, steps * rows)
            mixed = self.cartesian_encoder(
                torch.cat([global_token[:, None, :], flat_pairs], dim=1),
                src_key_padding_mask=~torch.cat([
                    torch.ones(bsz, 1, dtype=torch.bool, device=x.device),
                    flat_valid,
                ], dim=1),
            )[:, 1:, :].reshape(bsz, steps, rows, dim)
            search_repr = mixed[:, :, 0, :]
            track_weight = track_valid[:, None, :, None].to(mixed.dtype)
            track_repr = (mixed * track_weight).sum(dim=2) / track_weight.sum(dim=2).clamp_min(1.0)
            type_logits = self.cartesian_type_head(torch.cat([search_repr, track_repr], dim=-1))
            target_logits = self.cartesian_target_head(mixed).squeeze(-1)
            target_logits = target_logits.masked_fill(~track_valid[:, None, :], torch.finfo(target_logits.dtype).min)
            outputs = [(type_logits, target_logits)]
            if return_all:
                return outputs, track_valid
            return type_logits, target_logits, track_valid
        queries = base_queries
        outputs = []
        for refine_i in range(self.refinement_passes):
            if self.decoder is not None:
                decoded = self.decoder(queries, memory, memory_key_padding_mask=~memory_valid)
            else:
                decoded = queries
                for block in self.independent_decoder:
                    decoded = decoded + block(decoded)
            type_logits, target_logits = self._heads(decoded, target_tokens, track_valid)
            outputs.append((type_logits, target_logits))
            if refine_i + 1 < self.refinement_passes:
                type_probs = F.softmax(type_logits, dim=-1)
                target_probs = F.softmax(target_logits, dim=-1)
                expected_target = torch.einsum("bsr,brd->bsd", target_probs, target_tokens)
                feedback = self.feedback(torch.cat([decoded, expected_target, type_probs], dim=-1))
                queries = base_queries + feedback
        if return_all:
            return outputs, track_valid
        return outputs[-1][0], outputs[-1][1], track_valid


class CudaGraphBatchDecoder:
    """Capture the fixed-shape full-window decoder to remove launch overhead."""

    def __init__(self, model: BatchScheduleDecoder, x: torch.Tensor, slot: torch.Tensor):
        if x.device.type != "cuda" or slot.device.type != "cuda":
            raise ValueError("CUDA graph inputs must be CUDA tensors")
        self.model = model
        self.static_x = torch.empty_like(x)
        self.static_slot = torch.empty_like(slot)
        self.static_x.copy_(x)
        self.static_slot.copy_(slot)
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(5):
                self.outputs = self.model(self.static_x, self.static_slot)
        torch.cuda.current_stream(x.device).wait_stream(stream)
        torch.cuda.synchronize(x.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.outputs = self.model(self.static_x, self.static_slot)

    def __call__(self, x: torch.Tensor, slot: torch.Tensor):
        self.static_x.copy_(x)
        self.static_slot.copy_(slot)
        self.graph.replay()
        return self.outputs


class BatchGraphPlanModule(torch.nn.Module):
    """Decode a complete unique-target schedule without a CPU assignment pass."""

    def __init__(self, model: BatchScheduleDecoder):
        super().__init__()
        self.model = model

    def forward(self, x, slot, dwell, budget):
        type_logits, target_logits, valid = self.model(x, slot)
        batch = x.shape[0]
        used = torch.zeros((batch, MAXT + 1), dtype=torch.bool, device=x.device)
        spent = torch.zeros((batch,), dtype=x.dtype, device=x.device)
        rows = []
        row_ids = torch.arange(MAXT + 1, device=x.device)[None, :]
        base_valid = valid & (row_ids > 0)
        for step in range(self.model.max_steps):
            active = spent < budget
            track = type_logits[:, step].argmax(dim=-1) == 1
            candidates = target_logits[:, step].masked_fill(~(base_valid & ~used), -1.0e9)
            target_row = candidates.argmax(dim=-1)
            has_target = (base_valid & ~used).any(dim=-1)
            choose_track = active & track & has_target
            row = torch.where(choose_track, target_row, torch.zeros_like(target_row))
            rows.append(torch.where(active, row, torch.full_like(row, -1)))
            selected = torch.nn.functional.one_hot(row.clamp(min=0), MAXT + 1).bool()
            used = used | (selected & choose_track[:, None])
            track_dt = dwell.gather(1, (row - 1).clamp(min=0)[:, None]).squeeze(1).clamp_min(1.0)
            dt = torch.where(choose_track, track_dt, torch.full_like(track_dt, 10.0))
            spent = spent + torch.where(active, dt, torch.zeros_like(dt))
        return torch.stack(rows, dim=1)


class CudaGraphBatchPlanner:
    def __init__(self, model: BatchScheduleDecoder, x: torch.Tensor, slot: torch.Tensor):
        self.module = BatchGraphPlanModule(model).eval()
        self.static_x = torch.empty_like(x)
        self.static_slot = torch.empty_like(slot)
        self.static_dwell = torch.full((x.shape[0], MAXT), 10.0, dtype=x.dtype, device=x.device)
        self.static_budget = torch.full((x.shape[0],), 200.0, dtype=x.dtype, device=x.device)
        self.static_x.copy_(x)
        self.static_slot.copy_(slot)
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(5):
                self.rows = self.module(self.static_x, self.static_slot, self.static_dwell, self.static_budget)
        torch.cuda.current_stream(x.device).wait_stream(stream)
        torch.cuda.synchronize(x.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.rows = self.module(self.static_x, self.static_slot, self.static_dwell, self.static_budget)

    def __call__(self, x, slot, dwell, budget):
        self.static_x.copy_(x)
        self.static_slot.copy_(slot)
        self.static_dwell.copy_(dwell)
        self.static_budget.copy_(budget)
        self.graph.replay()
        return self.rows


class BatchSchedulePlanner:
    def __init__(
        self,
        model: BatchScheduleDecoder,
        device: str = "cuda",
        assignment: str = "hungarian",
        use_cuda_graph: bool = True,
    ):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.assignment = str(assignment)
        self.use_cuda_graph = bool(use_cuda_graph) and self.device.type == "cuda"
        self.graph_runner: CudaGraphBatchDecoder | None = None
        self.plan_graph_runner: CudaGraphBatchPlanner | None = None
        self.graph_error = ""
        self.dtype = next(self.model.parameters()).dtype
        self.adapt = adapter()

    def warmup(self, obs: dict, budget_ms: float = 200.0):
        self.plan(obs, budget_ms)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def plan(self, obs: dict, budget_ms: float = 200.0):
        x_np = tokenize(self.adapt, obs, selected=set(), search_count=0).astype(np.float32)
        slot_np = slot_features(obs, 0.0, 0, 0, -1, float(budget_ms)).astype(np.float32)
        x = torch.from_numpy(x_np[None]).to(self.device, dtype=self.dtype)
        slot = torch.from_numpy(slot_np[None]).to(self.device, dtype=self.dtype)
        dwell_np = np.asarray(obs.get("t_dwell", np.ones(MAXT, dtype=np.float32) * 10.0), dtype=np.float32)
        dwell_np = np.pad(dwell_np[:MAXT], (0, max(0, MAXT - len(dwell_np))), constant_values=10.0)
        dwell_tensor = torch.from_numpy(dwell_np[None]).to(self.device, dtype=self.dtype)
        budget_tensor = torch.full((1,), float(budget_ms), device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            if self.assignment == "gpu_greedy" and self.use_cuda_graph:
                if self.plan_graph_runner is None and not self.graph_error:
                    try:
                        self.plan_graph_runner = CudaGraphBatchPlanner(self.model, x, slot)
                    except (RuntimeError, ValueError) as exc:
                        self.graph_error = str(exc)
                if self.plan_graph_runner is not None:
                    decoded_rows = self.plan_graph_runner(x, slot, dwell_tensor, budget_tensor)[0].detach().cpu().tolist()
                    return [
                        xs_s_search_action(MAXT) if int(row) <= 0 else xs_s_track_action(int(row), MAXT)
                        for row in decoded_rows
                        if int(row) >= 0
                    ]
            if self.use_cuda_graph and self.graph_runner is None and not self.graph_error:
                try:
                    self.graph_runner = CudaGraphBatchDecoder(self.model, x, slot)
                except (RuntimeError, ValueError) as exc:
                    self.graph_error = str(exc)
            if self.graph_runner is not None:
                type_logits, target_logits, valid = self.graph_runner(x, slot)
            else:
                type_logits, target_logits, valid = self.model(x, slot)
        type_rows = type_logits[0].argmax(dim=-1).detach().cpu().numpy()
        target_scores = target_logits[0].detach().cpu().numpy()
        valid_rows = valid[0].detach().cpu().numpy()
        dwell = dwell_np
        assigned: dict[int, int] = {}
        track_steps = [int(step) for step in range(self.model.max_steps) if int(type_rows[step]) == 1]
        valid_targets = np.flatnonzero(valid_rows & (np.arange(len(valid_rows)) > 0)).astype(np.int64)
        if self.assignment == "hungarian" and track_steps and len(valid_targets) > 0:
            from scipy.optimize import linear_sum_assignment

            score_matrix = target_scores[np.asarray(track_steps)[:, None], valid_targets[None, :]]
            slot_idx, target_idx = linear_sum_assignment(-score_matrix)
            assigned = {track_steps[int(s)]: int(valid_targets[int(t)]) for s, t in zip(slot_idx, target_idx)}
        used: set[int] = set()
        actions = []
        spent = 0.0
        for step in range(self.model.max_steps):
            if int(type_rows[step]) == 0:
                row = 0
                dt = 10.0
            else:
                row = int(assigned.get(step, 0))
                if row <= 0 or row in used:
                    order = np.argsort(-target_scores[step])
                    row = next((int(candidate) for candidate in order if int(candidate) > 0 and valid_rows[int(candidate)] and int(candidate) not in used), 0)
                dt = 10.0 if row <= 0 else float(max(1.0, dwell[row - 1]))
            actions.append(xs_s_search_action(MAXT) if row <= 0 else xs_s_track_action(row, MAXT))
            if row > 0:
                used.add(row)
            spent += dt
            if spent >= float(budget_ms):
                break
        return actions


def _sequence_to_window(sequence: dict, max_steps: int):
    from train_sonly_puct_dagger_ar import DaggerWindow

    labels = np.asarray(sequence["labels"], dtype=np.int64)[:max_steps]
    length = int(labels.shape[0])
    slots_in = np.asarray(sequence["slots"], dtype=np.float32)[:max_steps]
    slots = np.zeros((max_steps, slots_in.shape[-1]), dtype=np.float32)
    slots[:length] = slots_in
    pairs = np.full((max_steps, 2), -1, dtype=np.int64)
    pairs[:length, 0] = labels
    mask = np.zeros((max_steps,), dtype=np.bool_)
    mask[:length] = True
    return DaggerWindow(
        x=np.asarray(sequence["root_x"], dtype=np.float32),
        slots=slots,
        student_pairs=pairs.copy(),
        target_pairs=pairs,
        type_probs=np.zeros((max_steps, 2), dtype=np.float32),
        row_probs=np.zeros((max_steps, MAXT + 1), dtype=np.float32),
        mask=mask,
        meta={
            "initial": int(sequence.get("initial", 0)),
            "rate": float(sequence.get("rate", 0.0)),
            "seed": int(sequence.get("seed", 0)),
            "window": int(sequence.get("window", 0)),
        },
    )


def _resize_window(window, max_steps: int):
    from dataclasses import replace

    def resize(value, fill=0):
        if value is None:
            return None
        array = np.asarray(value)
        shape = (max_steps, *array.shape[1:])
        out = np.full(shape, fill, dtype=array.dtype)
        count = min(max_steps, array.shape[0])
        out[:count] = array[:count]
        return out

    return replace(
        window,
        slots=resize(window.slots),
        student_pairs=resize(window.student_pairs, -1),
        target_pairs=resize(window.target_pairs, -1),
        type_probs=resize(window.type_probs),
        row_probs=resize(window.row_probs),
        mask=resize(window.mask, False),
        rewards=resize(window.rewards) if window.rewards is not None else None,
        returns=resize(window.returns) if window.returns is not None else None,
        root_q=resize(window.root_q) if window.root_q is not None else None,
        root_q_mask=resize(window.root_q_mask, False) if window.root_q_mask is not None else None,
        state_tokens=resize(window.state_tokens) if window.state_tokens is not None else None,
        next_state_tokens=resize(window.next_state_tokens) if window.next_state_tokens is not None else None,
    )


def load_windows(path: str, max_steps: int = 20):
    import __main__
    from train_sonly_puct_dagger_ar import DaggerWindow

    __main__.DaggerWindow = DaggerWindow
    windows = []
    for item in str(path).split(","):
        item = item.strip()
        if not item:
            continue
        payload = torch.load(item, map_location="cpu", weights_only=False)
        if "windows" in payload:
            windows.extend(_resize_window(window, int(max_steps)) for window in payload["windows"])
        elif "plan_sequences" in payload:
            windows.extend(_sequence_to_window(sequence, int(max_steps)) for sequence in payload["plan_sequences"])
        else:
            raise KeyError(f"Dataset {item} contains neither 'windows' nor 'plan_sequences'")
    return windows


def train(args) -> None:
    windows = load_windows(args.dataset, args.max_steps)
    grouped: dict[str, np.ndarray] = {}
    for initial in sorted({int(window.meta["initial"]) for window in windows}):
        grouped[str(initial)] = np.asarray([i for i, window in enumerate(windows) if int(window.meta["initial"]) == initial], dtype=np.int64)
    print({"windows": len(windows), "groups": {key: len(value) for key, value in grouped.items()}}, flush=True)
    model = BatchScheduleDecoder(
        args.max_steps,
        args.d_model,
        args.nhead,
        args.encoder_layers,
        args.decoder_layers,
        args.refinement_passes,
        not bool(args.disable_target_bias),
        args.decoder_kind,
    ).to(args.device)
    if str(args.init_checkpoint).strip():
        initial = torch.load(args.init_checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(initial["state_dict"], strict=True)
        print({"init_checkpoint": str(args.init_checkpoint)}, flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1.0e-4)
    rng = np.random.default_rng(int(args.seed))
    groups = list(grouped.values())
    model.train()
    for step in range(int(args.steps)):
        indices = []
        for group_i in rng.integers(0, len(groups), size=int(args.batch_size)):
            group = groups[int(group_i)]
            indices.append(int(group[rng.integers(0, len(group))]))
        batch = [windows[index] for index in indices]
        x = torch.from_numpy(np.stack([window.x for window in batch])).float().to(args.device)
        root_slot = torch.from_numpy(np.stack([window.slots[0] for window in batch])).float().to(args.device)
        pairs = torch.from_numpy(np.stack([window.target_pairs for window in batch])).long().to(args.device)
        mask = torch.from_numpy(np.stack([window.mask for window in batch])).bool().to(args.device)
        teacher_type_probs = torch.from_numpy(np.stack([window.type_probs for window in batch])).float().to(args.device)
        teacher_row_probs = torch.from_numpy(np.stack([window.row_probs for window in batch])).float().to(args.device)
        pairs = pairs[:, : model.max_steps]
        mask = mask[:, : model.max_steps]
        teacher_type_probs = teacher_type_probs[:, : model.max_steps]
        teacher_row_probs = teacher_row_probs[:, : model.max_steps]
        outputs, root_track_valid = model(x, root_slot, return_all=True)
        rows = torch.div(pairs[:, :, 0].clamp(min=0), 2, rounding_mode="floor").clamp(min=0, max=MAXT)
        valid_steps = mask & (pairs[:, :, 0] >= 0)
        type_targets = (rows > 0).long()
        track = valid_steps & (rows > 0)
        target_visible_at_root = root_track_valid.gather(1, rows.clamp(min=0, max=MAXT))
        target_supervised = track & target_visible_at_root
        loss = x.new_tensor(0.0)
        final_type_loss = x.new_tensor(0.0)
        final_target_loss = x.new_tensor(0.0)
        for output_i, (type_logits, target_logits) in enumerate(outputs):
            hard_type_loss = F.cross_entropy(type_logits[valid_steps], type_targets[valid_steps], reduction="none")
            type_mass = teacher_type_probs.sum(dim=-1)
            normalized_type = teacher_type_probs / type_mass.clamp_min(1.0e-8)[:, :, None]
            soft_type_loss = -(normalized_type * F.log_softmax(type_logits, dim=-1)).sum(dim=-1)
            use_soft_type = (type_mass[valid_steps] > 0.0) & (not bool(args.hard_targets))
            type_loss = torch.where(use_soft_type, soft_type_loss[valid_steps], hard_type_loss).mean()
            target_loss = type_loss.new_tensor(0.0)
            if bool(target_supervised.any()):
                hard_target_loss = F.cross_entropy(
                    target_logits[:, :, 1:][target_supervised],
                    rows[target_supervised] - 1,
                    reduction="none",
                )
                valid_target_rows = root_track_valid[:, None, 1:].expand_as(target_logits[:, :, 1:])
                row_teacher = teacher_row_probs[:, :, 1:] * valid_target_rows.to(teacher_row_probs.dtype)
                row_mass = row_teacher.sum(dim=-1)
                normalized_rows = row_teacher / row_mass.clamp_min(1.0e-8)[:, :, None]
                target_log_probs = F.log_softmax(target_logits[:, :, 1:], dim=-1)
                soft_terms = torch.where(normalized_rows > 0.0, normalized_rows * target_log_probs, torch.zeros_like(normalized_rows))
                soft_target_loss = -soft_terms.sum(dim=-1)
                use_soft_target = (row_mass[target_supervised] > 0.0) & (not bool(args.hard_targets))
                target_loss = torch.where(
                    use_soft_target,
                    soft_target_loss[target_supervised],
                    hard_target_loss,
                ).mean()
            pass_loss = float(args.type_weight) * type_loss + float(args.target_weight) * target_loss
            if output_i + 1 < len(outputs):
                pass_loss = float(args.deep_supervision_weight) * pass_loss
            loss = loss + pass_loss
            final_type_loss = type_loss
            final_target_loss = target_loss
        type_logits, target_logits = outputs[-1]
        step_weight = valid_steps.float()
        denom = step_weight.sum(dim=1).clamp_min(1.0)
        predicted_search = (F.softmax(type_logits, dim=-1)[:, :, 0] * step_weight).sum(dim=1) / denom
        target_search = ((type_targets == 0).float() * step_weight).sum(dim=1) / denom
        count_loss = F.mse_loss(predicted_search, target_search)
        track_prob = F.softmax(type_logits, dim=-1)[:, :, 1] * step_weight
        target_prob = F.softmax(target_logits, dim=-1)
        assignment_mass = (track_prob[:, :, None] * target_prob).sum(dim=1)
        competition_loss = F.relu(assignment_mass - 1.0).square().mean()
        loss = loss + float(args.count_weight) * count_loss + float(args.competition_weight) * competition_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == int(args.steps) - 1:
            type_acc = (type_logits.argmax(dim=-1)[valid_steps] == type_targets[valid_steps]).float().mean()
            row_pred = target_logits.argmax(dim=-1)
            target_acc = (
                (row_pred[target_supervised] == rows[target_supervised]).float().mean()
                if bool(target_supervised.any())
                else type_acc.new_tensor(0.0)
            )
            print(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "type_loss": float(final_type_loss.detach()),
                    "target_loss": float(final_target_loss.detach()),
                    "count_loss": float(count_loss.detach()),
                    "competition_loss": float(competition_loss.detach()),
                    "type_acc": float(type_acc),
                    "target_acc": float(target_acc),
                },
                flush=True,
            )
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": {
            "max_steps": args.max_steps,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "encoder_layers": args.encoder_layers,
            "decoder_layers": args.decoder_layers,
            "refinement_passes": args.refinement_passes,
            "use_target_bias": not bool(args.disable_target_bias),
            "decoder_kind": args.decoder_kind,
        },
        "dataset": str(args.dataset),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    print({"saved": str(args.out)}, flush=True)


def load_model(path: str, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = BatchScheduleDecoder(**cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def evaluate(args) -> None:
    model = load_model(args.checkpoint, args.device)
    if args.precision == "fp16":
        if str(args.device).startswith("cpu"):
            raise ValueError("fp16 evaluation requires CUDA")
        model = model.half()
    if bool(args.compile_model):
        if importlib.util.find_spec("triton") is None:
            print({"torch_compile": "disabled", "reason": "Triton is not installed"}, flush=True)
        else:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    if int(args.refinement_passes) > 0:
        model.refinement_passes = int(args.refinement_passes)
    if bool(args.disable_target_bias):
        model.use_target_bias = False
    all_windows = []
    summaries = []
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = False
    exact_args.single_sensor = True
    # The model shapes are fixed across load cells. Reuse one captured graph so
    # benchmark latency reflects steady-state deployment rather than charging
    # CUDA graph compilation once per evaluation cell.
    planner = BatchSchedulePlanner(
        model,
        args.device,
        assignment=args.assignment,
        use_cuda_graph=not bool(args.disable_cuda_graph),
    )
    for initial in parse_ints(args.initials):
        for rate in parse_floats(args.rates):
            env_cfg = env_cfg_for(float(rate), exact_args)
            env_cfg["enable_x_band"] = 0
            for seed in parse_ints(args.seeds):
                frame, _actions = run_plan_eval(planner, "Batch decoder", int(initial), int(seed), int(args.windows), env_cfg)
                frame["initial"] = int(initial)
                frame["rate"] = float(rate)
                frame["seed"] = int(seed)
                all_windows.append(frame)
                row = {"planner": "Batch decoder", "initial": int(initial), "rate": float(rate), "seed": int(seed), **summarize(frame)}
                summaries.append(row)
                print(row, flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_windows, ignore_index=True).to_csv(out, index=False)
    pd.DataFrame(summaries).to_csv(out.with_name(out.stem + "_summary.csv"), index=False)


def add_reward_args(parser):
    from canonical_scheduler_contract import add_canonical_reward_args

    add_canonical_reward_args(parser)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--dataset", required=True)
    train_parser.add_argument("--init-checkpoint", default="")
    train_parser.add_argument("--out", required=True)
    train_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_parser.add_argument("--steps", type=int, default=2000)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1.0e-4)
    train_parser.add_argument("--target-weight", type=float, default=1.0)
    train_parser.add_argument("--type-weight", type=float, default=1.0)
    train_parser.add_argument("--count-weight", type=float, default=0.0)
    train_parser.add_argument("--competition-weight", type=float, default=0.0)
    train_parser.add_argument("--deep-supervision-weight", type=float, default=0.25)
    train_parser.add_argument("--hard-targets", action="store_true", help="Train on the selected PUCT sequence instead of shallow visit-count distributions.")
    train_parser.add_argument("--seed", type=int, default=916)
    train_parser.add_argument("--max-steps", type=int, default=20)
    train_parser.add_argument("--d-model", type=int, default=64)
    train_parser.add_argument("--nhead", type=int, default=4)
    train_parser.add_argument("--encoder-layers", type=int, default=2)
    train_parser.add_argument("--decoder-layers", type=int, default=2)
    train_parser.add_argument(
        "--decoder-kind",
        choices=["attention", "action_only", "independent", "cartesian"],
        default="attention",
    )
    train_parser.add_argument("--refinement-passes", type=int, default=1)
    train_parser.add_argument("--disable-target-bias", action="store_true")
    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--out", required=True)
    eval_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    eval_parser.add_argument("--initials", default="20,40,60")
    eval_parser.add_argument("--rates", default="2,3,4")
    eval_parser.add_argument("--seeds", default="925")
    eval_parser.add_argument("--windows", type=int, default=100)
    eval_parser.add_argument("--assignment", choices=["greedy", "hungarian", "gpu_greedy"], default="hungarian")
    eval_parser.add_argument("--refinement-passes", type=int, choices=[0, 1, 2], default=0, help="Override checkpoint refinement passes; 0 keeps the checkpoint setting.")
    eval_parser.add_argument("--disable-cuda-graph", action="store_true")
    eval_parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    eval_parser.add_argument("--disable-target-bias", action="store_true")
    eval_parser.add_argument("--compile-model", action="store_true")
    add_reward_args(eval_parser)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
