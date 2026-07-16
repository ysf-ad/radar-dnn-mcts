from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from radar_dnn_mcts.models.boundary import BoundaryPredictor
from radar_dnn_mcts.models.checkpoint import load_checkpoint, save_checkpoint
from radar_dnn_mcts.models.decoders import AutoregressiveDecoder, BatchDecoder
from radar_dnn_mcts.models.dynamics import LatentDynamics
from radar_dnn_mcts.models.scheduler import RadarSchedulerModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the asynchronous window-boundary predictor.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    core = RadarSchedulerModel().to(device)
    dynamics = LatentDynamics().to(device)
    ar = AutoregressiveDecoder(core).to(device)
    batch = BatchDecoder().to(device)
    metadata = load_checkpoint(
        args.checkpoint,
        {"core": core, "dynamics": dynamics, "ar": ar, "batch": batch},
        device,
    )
    boundary = BoundaryPredictor().to(device)
    arrays = np.load(args.data)
    required = {
        "tokens",
        "context",
        "suffix_rows",
        "suffix_mask",
        "remaining_ms",
        "boundary_tokens",
        "boundary_context",
    }
    missing = required - set(arrays.files)
    if missing:
        raise KeyError(f"dataset is missing arrays: {sorted(missing)}")
    data = {name: torch.as_tensor(arrays[name], device=device) for name in required}
    optimizer = torch.optim.AdamW(boundary.parameters(), lr=args.lr, weight_decay=1e-4)
    core.eval()
    for epoch in range(args.epochs):
        with torch.no_grad():
            state = core.encode(data["tokens"].float(), data["context"].float())
            target = core.encode(data["boundary_tokens"].float(), data["boundary_context"].float())
        prediction = boundary(
            state,
            data["suffix_rows"].long(),
            data["suffix_mask"].bool(),
            data["remaining_ms"].float(),
        )
        global_loss = (prediction.global_state - target.global_state).square().mean()
        valid = target.valid_rows.unsqueeze(-1).float()
        target_loss = ((prediction.target_states - target.target_states).square() * valid).sum() / valid.sum().clamp_min(1.0)
        loss = global_loss + target_loss / prediction.target_states.shape[-1]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(boundary.parameters(), 1.0)
        optimizer.step()
        print(f"epoch={epoch + 1} loss={float(loss.detach()):.6f}")
    metadata.update({"boundary_data": str(args.data), "boundary_epochs": args.epochs})
    save_checkpoint(
        args.out,
        {"core": core, "dynamics": dynamics, "ar": ar, "batch": batch, "boundary": boundary},
        metadata,
    )


if __name__ == "__main__":
    main()
