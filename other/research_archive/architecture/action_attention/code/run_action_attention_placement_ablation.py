"""Depth-controlled ablation of where candidate-action attention is applied."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from shared_scheduler_family import (
    SchedulerOutput,
    SchedulerState,
    SharedActionScorer,
    masked_all_action_q_loss,
    soft_factorized_policy_loss,
)


def transformer(d_model: int, layers: int) -> nn.TransformerEncoder:
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=2 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        ),
        num_layers=layers,
        enable_nested_tensor=False,
    )


class RawStateEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, layers: int):
        super().__init__()
        self.input_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, d_model), nn.GELU())
        self.cls = nn.Parameter(torch.randn(d_model) * 0.02)
        self.encoder = transformer(d_model, layers)

    def forward(self, raw_x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = self.input_proj(raw_x)
        cls = self.cls[None, None, :].expand(raw_x.shape[0], 1, -1)
        encoded = self.encoder(
            torch.cat([cls, target], dim=1),
            src_key_padding_mask=torch.cat([
                torch.zeros(valid.shape[0], 1, dtype=torch.bool, device=valid.device),
                ~valid,
            ], dim=1),
        )
        return encoded[:, 0], encoded[:, 1:]


class LateActionAttention(nn.Module):
    """Two target-state layers followed by one action-set layer."""

    def __init__(self, input_dim: int, slot_dim: int, d_model: int):
        super().__init__()
        self.state_encoder = RawStateEncoder(input_dim, d_model, layers=2)
        self.scorer = SharedActionScorer(d_model, slot_dim, mixer="attention", builder="context_token")

    def forward(self, raw_x: torch.Tensor, slot: torch.Tensor, valid: torch.Tensor) -> SchedulerOutput:
        global_token, target_tokens = self.state_encoder(raw_x, valid)
        state = SchedulerState(global_token, target_tokens, torch.zeros_like(global_token))
        return self.scorer(state, slot, valid)


class NoActionMixing(nn.Module):
    """Three target-state layers followed by independent action scoring."""

    def __init__(self, input_dim: int, slot_dim: int, d_model: int):
        super().__init__()
        self.state_encoder = RawStateEncoder(input_dim, d_model, layers=3)
        self.scorer = SharedActionScorer(d_model, slot_dim, mixer="identity", builder="broadcast")

    def forward(self, raw_x: torch.Tensor, slot: torch.Tensor, valid: torch.Tensor) -> SchedulerOutput:
        global_token, target_tokens = self.state_encoder(raw_x, valid)
        state = SchedulerState(global_token, target_tokens, torch.zeros_like(global_token))
        return self.scorer(state, slot, valid)


class EarlyUnifiedActionEncoder(nn.Module):
    """Build actions from raw features first, then run one three-layer encoder."""

    def __init__(self, input_dim: int, slot_dim: int, d_model: int):
        super().__init__()
        self.raw_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, d_model), nn.GELU())
        self.slot_proj = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, d_model), nn.GELU())
        self.search_embedding = nn.Parameter(torch.randn(d_model) * 0.02)
        self.track_embedding = nn.Parameter(torch.randn(d_model) * 0.02)
        self.context_builder = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.action_builder = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.encoder = transformer(d_model, layers=3)
        self.type_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.target_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.q_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, raw_x: torch.Tensor, slot: torch.Tensor, valid: torch.Tensor) -> SchedulerOutput:
        bsz, rows, _ = raw_x.shape
        target = self.raw_proj(raw_x)
        kind = self.track_embedding[None, None, :].expand(bsz, rows, -1).clone()
        kind[:, 0] = self.search_embedding
        action = self.action_builder(torch.cat([target, kind], dim=-1))
        context = self.context_builder(self.slot_proj(slot))
        mixed = self.encoder(
            torch.cat([context[:, None, :], action], dim=1),
            src_key_padding_mask=torch.cat([
                torch.zeros(bsz, 1, dtype=torch.bool, device=valid.device),
                ~valid,
            ], dim=1),
        )
        context_out, action_out = mixed[:, 0], mixed[:, 1:]
        type_logits = self.type_head(context_out)
        target_logits = self.target_head(action_out).squeeze(-1).masked_fill(~valid, -torch.inf)
        type_logp = F.log_softmax(type_logits, dim=1)
        track_valid = valid.clone()
        track_valid[:, 0] = False
        target_logp = F.log_softmax(target_logits.masked_fill(~track_valid, -1e9), dim=1)
        policy = type_logp[:, 1, None] + target_logp
        policy[:, 0] = type_logp[:, 0]
        policy = policy.masked_fill(~valid, -torch.inf)
        q = self.q_head(action_out).squeeze(-1).masked_fill(~valid, -torch.inf)
        return SchedulerOutput(action_out, policy, q, valid, type_logits, target_logits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--q-weight", type=float, default=1.0)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(model: nn.Module, data: dict, idx: torch.Tensor, device: torch.device) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        output = model(data["raw_x"][idx].to(device), data["slot"][idx].to(device), data["valid_mask"][idx].to(device))
        policy_loss = soft_factorized_policy_loss(output, data["policy_target"][idx].to(device))
        q_mse = masked_all_action_q_loss(output, data["q_target"][idx].to(device), data["q_mask"][idx].to(device))
        accuracy = (output.policy_logits.argmax(1).cpu() == data["action"][idx]).float().mean()
    return {"policy_loss": float(policy_loss), "q_loss": float(q_mse), "accuracy": float(accuracy)}


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.samples, map_location="cpu", weights_only=True)
    count, input_dim = data["raw_x"].shape[0], data["raw_x"].shape[-1]
    slot_dim = data["slot"].shape[-1]
    order = torch.randperm(count, generator=torch.Generator().manual_seed(args.seed))
    val_n = max(1, round(0.2 * count))
    val_idx, train_idx = order[:val_n], order[val_n:]
    variants = {
        "late_action_attention": LateActionAttention(input_dim, slot_dim, 48),
        "early_unified": EarlyUnifiedActionEncoder(input_dim, slot_dim, 48),
        "no_action_mixing": NoActionMixing(input_dim, slot_dim, 48),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, model in variants.items():
        seed_all(args.seed)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        batch_rng = torch.Generator().manual_seed(args.seed)
        for _ in range(args.epochs):
            model.train()
            shuffled = train_idx[torch.randperm(train_idx.numel(), generator=batch_rng)]
            for start in range(0, shuffled.numel(), args.batch_size):
                idx = shuffled[start : start + args.batch_size]
                output = model(data["raw_x"][idx].to(device), data["slot"][idx].to(device), data["valid_mask"][idx].to(device))
                loss = soft_factorized_policy_loss(output, data["policy_target"][idx].to(device))
                loss = loss + args.q_weight * masked_all_action_q_loss(output, data["q_target"][idx].to(device), data["q_mask"][idx].to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        result = metrics(model, data, val_idx, device)
        result.update({"variant": name, "seed": args.seed, "parameters": sum(p.numel() for p in model.parameters())})
        rows.append(result)
        torch.save(model.state_dict(), args.output / f"{name}_seed{args.seed}.pt")
    with (args.output / f"placement_ablation_seed{args.seed}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
