"""Train and evaluate a boundary predictor with seed-disjoint partitions."""

from __future__ import annotations

import argparse
import json
import random
import copy
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from boundary_dataset import BoundaryTensorDataset, split_by_seed
from boundary_predictor import BoundaryStatePredictor


def causal_projection(midpoint: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Keep stochastic/discrete target identity fixed; forecast continuous pressures."""
    if midpoint.shape[-1] <= 4:
        return prediction
    projected = midpoint.clone()
    projected[:, 0, :8] = prediction[:, 0, :8]
    active = midpoint[:, 1:, 4] > 0.5
    for feature in (0, 1, 5, 6, 7):
        projected[:, 1:, feature] = torch.where(
            active, prediction[:, 1:, feature], midpoint[:, 1:, feature]
        )
    return projected


def forecast_mask(
    midpoint: torch.Tensor,
    target_features: tuple[int, ...] = (0, 1, 5, 6, 7),
    include_search_row: bool = True,
) -> torch.Tensor:
    """Fields observable at midpoint whose boundary evolution can be forecast."""
    if midpoint.shape[-1] <= 4:
        return torch.ones_like(midpoint, dtype=torch.bool)
    mask = torch.zeros_like(midpoint, dtype=torch.bool)
    if include_search_row:
        mask[:, 0, :8] = True
    active = midpoint[:, 1:, 4] > 0.5
    for feature in target_features:
        mask[:, 1:, feature] = active
    return mask


def load_frozen_planner(path: str | Path, device: torch.device):
    from distill_sparse64_sequence_decoder import Sparse64SequenceDecoder

    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload.get("args", {}) if isinstance(payload, dict) else {}
    model = Sparse64SequenceDecoder(
        d_model=int(cfg.get("d_model", 96)),
        nhead=int(cfg.get("nhead", 4)),
        nlayers=int(cfg.get("nlayers", 2)),
    ).to(device)
    model.load_state_dict(payload.get("model", payload), strict=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _planner_slot(
    root: torch.Tensor,
    elapsed: torch.Tensor,
    searches: torch.Tensor,
    tracks: torch.Tensor,
    last_is_search: torch.Tensor,
) -> torch.Tensor:
    root_active = root[..., 4] > 0.5
    root_active[:, 0] = False
    dwell = root[..., 2].clamp(0.0, 2.0) * 100.0
    deadline = root[..., 1] * 3000.0
    slot = root.new_zeros((root.shape[0], 11))
    slot[:, 0] = elapsed / 200.0
    slot[:, 1] = searches / 20.0
    slot[:, 2] = tracks / 100.0
    slot[:, 3] = last_is_search.to(root.dtype)
    slot[:, 4] = root_active.float().sum(dim=1) / 100.0
    slot[:, 5] = slot[:, 4]
    slot[:, 6] = (dwell * root_active).sum(dim=1).div(4000.0).clamp(max=2.0)
    positive = root_active & (deadline > 0)
    minimum = deadline.masked_fill(~positive, float("inf")).min(dim=1).values
    slot[:, 7] = torch.where(torch.isfinite(minimum), minimum / 3000.0, torch.zeros_like(minimum))
    slot[:, 8:11] = root[:, 0, 9:12]
    return slot


def planner_consistency_loss(
    planner,
    prediction: torch.Tensor,
    target: torch.Tensor,
    max_steps: int = 8,
) -> torch.Tensor:
    """Teacher-force the predicted boundary through the true-boundary plan."""
    batch, token_count = target.shape[:2]
    valid_base = target[..., 4] > 0.5
    valid_base[:, 0] = False

    def encode(root: torch.Tensor):
        return planner.encode(root)[:2]

    pred_cls, pred_tokens = encode(prediction)
    with torch.no_grad():
        true_cls, true_tokens = encode(target)

    selected = torch.zeros((batch, token_count), dtype=torch.bool, device=target.device)
    elapsed = target.new_zeros(batch)
    searches = target.new_zeros(batch)
    tracks = target.new_zeros(batch)
    last_is_search = torch.zeros(batch, dtype=torch.bool, device=target.device)
    prev_class = torch.zeros(batch, dtype=torch.long, device=target.device)
    prev_row = target.new_zeros(batch)
    true_hidden = pred_hidden = None
    losses: list[torch.Tensor] = []

    for _ in range(int(max_steps)):
        live = elapsed < 200.0
        if not bool(live.any()):
            break
        true_slot = _planner_slot(target, elapsed, searches, tracks, last_is_search)
        pred_slot = _planner_slot(prediction, elapsed, searches, tracks, last_is_search)
        with torch.no_grad():
            true_type, true_target, true_hidden = planner.score_step(
                true_cls, true_tokens, true_slot, prev_class, prev_row, true_hidden
            )
        pred_type, pred_target, pred_hidden = planner.score_step(
            pred_cls, pred_tokens, pred_slot, prev_class, prev_row, pred_hidden
        )
        valid = valid_base & ~selected
        masked_true_target = true_target.masked_fill(~valid, -1.0e9)
        best_row = masked_true_target.argmax(dim=-1)
        true_type_logp = nn.functional.log_softmax(true_type, dim=-1)
        true_target_logp = nn.functional.log_softmax(masked_true_target, dim=-1)
        has_target = valid.any(dim=-1)
        choose_track = has_target & (
            true_type_logp[:, 1] + true_target_logp.gather(1, best_row[:, None]).squeeze(1)
            > true_type_logp[:, 0]
        )
        row = torch.where(choose_track, best_row, torch.zeros_like(best_row))
        type_loss = nn.functional.cross_entropy(pred_type, choose_track.long(), reduction="none")
        losses.append(type_loss[live])
        tracked_live = live & choose_track
        if bool(tracked_live.any()):
            masked_pred_target = pred_target.masked_fill(~valid, -1.0e9)
            target_loss = nn.functional.cross_entropy(masked_pred_target, row, reduction="none")
            losses.append(target_loss[tracked_live])

        dwell = target[torch.arange(batch, device=target.device), row, 2].clamp(0.01, 2.0) * 100.0
        elapsed = elapsed + torch.where(choose_track, dwell, torch.full_like(dwell, 10.0))
        searches = searches + (~choose_track).to(searches.dtype)
        tracks = tracks + choose_track.to(tracks.dtype)
        selected[torch.arange(batch, device=target.device), row] |= choose_track
        last_is_search = ~choose_track
        prev_class = torch.where(choose_track, torch.full_like(row, 3), torch.ones_like(row))
        prev_row = row.to(target.dtype) / max(1, token_count - 1)

    return torch.cat([value.reshape(-1) for value in losses]).mean() if losses else prediction.sum() * 0.0


def planner_root_kl_loss(planner, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Diagnostic one-step distribution agreement."""
    active = target[..., 4] > 0.5
    active[:, 0] = False

    def logits(root: torch.Tensor):
        root_active = root[..., 4] > 0.5
        root_active[:, 0] = False
        dwell = root[..., 2].clamp(0.0, 2.0) * 100.0
        deadline = root[..., 1] * 3000.0
        slot = root.new_zeros((root.shape[0], 11))
        slot[:, 4] = root_active.float().sum(dim=1) / 100.0
        slot[:, 5] = slot[:, 4]
        slot[:, 6] = (dwell * root_active).sum(dim=1).div(4000.0).clamp(max=2.0)
        positive = root_active & (deadline > 0)
        minimum = deadline.masked_fill(~positive, float("inf")).min(dim=1).values
        slot[:, 7] = torch.where(torch.isfinite(minimum), minimum / 3000.0, torch.zeros_like(minimum))
        slot[:, 8:11] = root[:, 0, 9:12]
        cls, tokens, _ = planner.encode(root)
        previous_class = torch.zeros(root.shape[0], dtype=torch.long, device=root.device)
        previous_row = torch.zeros(root.shape[0], dtype=root.dtype, device=root.device)
        return planner.score_step(cls, tokens, slot, previous_class, previous_row)[:2]

    pred_type, pred_target = logits(prediction)
    with torch.no_grad():
        true_type, true_target = logits(target)
    type_loss = nn.functional.kl_div(
        nn.functional.log_softmax(pred_type, dim=-1),
        nn.functional.softmax(true_type, dim=-1),
        reduction="batchmean",
    )
    track_valid = active
    pred_target = pred_target.masked_fill(~track_valid, -1.0e9)
    true_target = true_target.masked_fill(~track_valid, -1.0e9)
    target_loss = nn.functional.kl_div(
        nn.functional.log_softmax(pred_target, dim=-1),
        nn.functional.softmax(true_target, dim=-1),
        reduction="batchmean",
    )
    return type_loss + target_loss


def token_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = prediction.float() - target.float()
    metrics = {
        "token_mae": float(error.abs().mean().item()),
        "token_rmse": float(error.square().mean().sqrt().item()),
    }
    if prediction.shape[-1] > 4:
        pred_active = prediction[..., 4] > 0.5
        true_active = target[..., 4] > 0.5
        metrics["active_accuracy"] = float((pred_active == true_active).float().mean().item())
        metrics["active_f1"] = float(
            (2 * (pred_active & true_active).sum().float()
             / (pred_active.sum() + true_active.sum()).clamp_min(1)).item()
        )
    return metrics


def forecast_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    midpoint: torch.Tensor,
    target_features: tuple[int, ...] = (0, 1, 5, 6, 7),
    include_search_row: bool = True,
) -> dict[str, float]:
    mask = forecast_mask(midpoint, target_features, include_search_row)
    error = (prediction.float() - target.float())[mask]
    return {
        "forecast_mae": float(error.abs().mean().item()),
        "forecast_rmse": float(error.square().mean().sqrt().item()),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    target_features: tuple[int, ...] = (0, 1, 5, 6, 7),
    include_search_row: bool = True,
) -> dict[str, dict[str, float]]:
    learned, naive, targets, midpoints = [], [], [], []
    model.eval()
    for batch in loader:
        inputs = {
            name: batch[name].to(device)
            for name in ("midpoint_tokens", "suffix_actions", "suffix_mask", "remaining_time_ms")
        }
        target = batch["boundary_tokens"].to(device)
        learned.append(causal_projection(inputs["midpoint_tokens"], model(**inputs)).cpu())
        naive.append(inputs["midpoint_tokens"].cpu())
        targets.append(target.cpu())
        midpoints.append(inputs["midpoint_tokens"].cpu())
    learned_tensor = torch.cat(learned)
    naive_tensor = torch.cat(naive)
    target_tensor = torch.cat(targets)
    midpoint_tensor = torch.cat(midpoints)
    learned_metrics = token_metrics(learned_tensor, target_tensor)
    naive_metrics = token_metrics(naive_tensor, target_tensor)
    learned_metrics.update(
        forecast_metrics(learned_tensor, target_tensor, midpoint_tensor, target_features, include_search_row)
    )
    naive_metrics.update(
        forecast_metrics(naive_tensor, target_tensor, midpoint_tensor, target_features, include_search_row)
    )
    return {
        "naive_midpoint": naive_metrics,
        "learned": learned_metrics,
        "mae_improvement_fraction": 1.0
        - learned_metrics["token_mae"] / max(naive_metrics["token_mae"], 1e-12),
    }


@torch.inference_mode()
def evaluate_planner_consistency(model, planner, loader, device: torch.device, max_steps: int) -> float:
    values = []
    model.eval()
    for batch in loader:
        midpoint = batch["midpoint_tokens"].to(device)
        target = batch["boundary_tokens"].to(device)
        prediction = causal_projection(
            midpoint,
            model(
                midpoint,
                batch["suffix_actions"].to(device),
                batch["suffix_mask"].to(device),
                batch["remaining_time_ms"].to(device),
            ),
        )
        values.append(float(planner_consistency_loss(planner, prediction, target, max_steps).item()))
    return float(np.mean(values))


def train_boundary_model(
    dataset: BoundaryTensorDataset,
    held_out_seeds: list[int] | None,
    *,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    d_model: int = 96,
    layers: int = 2,
    model_seed: int = 123,
    device: str = "cpu",
    change_weight: float = 12.0,
    unchanged_weight: float = 2.0,
    max_action_id: int | None = None,
    planner_checkpoint: str | Path | None = None,
    planner_loss_weight: float = 0.0,
    planner_steps: int = 8,
    target_features: tuple[int, ...] = (0, 1, 5, 6, 7),
    include_search_row: bool = True,
) -> tuple[BoundaryStatePredictor, dict]:
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    dev = torch.device(device)
    train_idx, eval_idx = split_by_seed(dataset, held_out_seeds)
    observed_max_action = int(dataset.arrays["suffix_actions"].max(initial=0))
    max_action_id = observed_max_action if max_action_id is None else max(observed_max_action, int(max_action_id))
    model = BoundaryStatePredictor(
        token_dim=int(dataset.arrays["midpoint_tokens"].shape[-1]),
        max_action_id=max_action_id,
        max_suffix=int(dataset.arrays["suffix_actions"].shape[1]),
        max_tokens=int(dataset.arrays["midpoint_tokens"].shape[1]),
        d_model=d_model,
        layers=layers,
    ).to(dev)
    planner = load_frozen_planner(planner_checkpoint, dev) if planner_checkpoint else None
    generator = torch.Generator().manual_seed(model_seed)
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()), batch_size=batch_size, shuffle=True, generator=generator
    )
    eval_loader = DataLoader(Subset(dataset, eval_idx.tolist()), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    for _ in range(int(epochs)):
        model.train()
        for batch in train_loader:
            target = batch["boundary_tokens"].to(dev)
            midpoint = batch["midpoint_tokens"].to(dev)
            prediction = model(
                midpoint_tokens=midpoint,
                suffix_actions=batch["suffix_actions"].to(dev),
                suffix_mask=batch["suffix_mask"].to(dev),
                remaining_time_ms=batch["remaining_time_ms"].to(dev),
            )
            prediction = causal_projection(midpoint, prediction)
            valid_forecast = forecast_mask(midpoint, target_features, include_search_row)
            delta = target - midpoint
            changed = delta.abs() > 1.0e-4
            weights = 1.0 + float(change_weight) * changed.to(target.dtype)
            weights = weights + float(unchanged_weight) * (~changed).to(target.dtype)
            per = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
            weights = weights * valid_forecast.to(weights.dtype)
            loss = (per * weights).sum() / weights.sum().clamp_min(1.0)
            if planner is not None and float(planner_loss_weight) > 0.0:
                loss = loss + float(planner_loss_weight) * planner_consistency_loss(
                    planner, prediction, target, planner_steps
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        current = evaluate(model, eval_loader, dev, target_features, include_search_row)
        current_mae = float(current["learned"]["forecast_mae"])
        planner_eval = (
            evaluate_planner_consistency(model, planner, eval_loader, dev, planner_steps)
            if planner is not None and float(planner_loss_weight) > 0.0
            else 0.0
        )
        score = current_mae + (0.01 * planner_eval if planner is not None else 0.0)
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    metrics = evaluate(model, eval_loader, dev, target_features, include_search_row)
    if planner is not None and float(planner_loss_weight) > 0.0:
        metrics["planner_sequence_loss"] = evaluate_planner_consistency(
            model, planner, eval_loader, dev, planner_steps
        )
    metrics["split"] = {
        "train_seeds": sorted(set(int(dataset.arrays["seed"][i]) for i in train_idx)),
        "held_out_seeds": sorted(set(int(dataset.arrays["seed"][i]) for i in eval_idx)),
        "train_samples": int(len(train_idx)),
        "held_out_samples": int(len(eval_idx)),
    }
    return model, metrics


def _parse_seeds(value: str) -> list[int] | None:
    return [int(part) for part in value.split(",") if part.strip()] or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, default=Path("boundary_predictor.pt"))
    parser.add_argument("--held-out-seeds", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--model-seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--change-weight", type=float, default=12.0)
    parser.add_argument("--unchanged-weight", type=float, default=2.0)
    parser.add_argument("--max-action-id", type=int, default=None)
    parser.add_argument("--planner-checkpoint", type=Path, default=None)
    parser.add_argument("--planner-loss-weight", type=float, default=0.0)
    parser.add_argument("--planner-steps", type=int, default=8)
    parser.add_argument("--target-features", default="0,1,5,6,7")
    parser.add_argument("--include-search-row", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    dataset = BoundaryTensorDataset.load(args.dataset)
    model, metrics = train_boundary_model(
        dataset,
        _parse_seeds(args.held_out_seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        layers=args.layers,
        model_seed=args.model_seed,
        device=args.device,
        change_weight=args.change_weight,
        unchanged_weight=args.unchanged_weight,
        max_action_id=args.max_action_id,
        planner_checkpoint=args.planner_checkpoint,
        planner_loss_weight=args.planner_loss_weight,
        planner_steps=args.planner_steps,
        target_features=tuple(int(value) for value in args.target_features.split(",") if value.strip()),
        include_search_row=bool(args.include_search_row),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "token_dim": model.token_dim,
            "max_suffix": model.max_suffix,
            "max_tokens": model.max_tokens,
            "use_row_embeddings": model.use_row_embeddings,
            "target_features": args.target_features,
            "include_search_row": bool(args.include_search_row),
            "max_action_id": model.action_embedding.num_embeddings - 1,
            "metrics": metrics,
        },
        args.out,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
