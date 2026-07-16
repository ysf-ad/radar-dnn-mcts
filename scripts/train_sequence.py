from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from radar_dnn_mcts.models.checkpoint import load_checkpoint, save_checkpoint
from radar_dnn_mcts.models.decoders import AutoregressiveDecoder, BatchDecoder
from radar_dnn_mcts.models.dynamics import LatentDynamics
from radar_dnn_mcts.models.scheduler import RadarSchedulerModel
from radar_dnn_mcts.training.losses import batch_sequence_loss


def ar_loss(model, tokens, context, actions, action_mask):
    state, prefix = model.initial(tokens, context)
    selected = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    previous = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
    losses = []
    for step in range(actions.shape[1]):
        output, prefix = model.step(state, prefix, context, previous, selected)
        ce = F.cross_entropy(output.policy_logits, actions[:, step], reduction="none")
        losses.append(ce * action_mask[:, step].float())
        row = actions[:, step]
        selected.scatter_(1, row[:, None], row[:, None] > 0)
        previous = row
    stacked = torch.stack(losses, dim=1)
    return stacked.sum() / action_mask.float().sum().clamp_min(1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="NPZ with tokens, context, actions, action_mask")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decoder", choices=["ar", "batch", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    core = RadarSchedulerModel().to(device)
    dynamics = LatentDynamics().to(device)
    ar = AutoregressiveDecoder(core).to(device)
    batch = BatchDecoder().to(device)
    metadata = load_checkpoint(args.checkpoint, {"core": core, "dynamics": dynamics, "ar": ar, "batch": batch}, device)
    arrays = np.load(args.data)
    data = {name: torch.as_tensor(arrays[name], device=device) for name in ("tokens", "context", "actions", "action_mask")}
    parameters = []
    if args.decoder in ("ar", "both"):
        parameters.extend(ar.parameters())
    if args.decoder in ("batch", "both"):
        parameters.extend(batch.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    count = data["tokens"].shape[0]
    for epoch in range(args.epochs):
        order = torch.randperm(count, device=device)
        values = []
        for start in range(0, count, args.batch_size):
            idx = order[start : start + args.batch_size]
            tokens = data["tokens"][idx].float()
            context = data["context"][idx].float()
            actions = data["actions"][idx].long()
            mask = data["action_mask"][idx].bool()
            loss = torch.zeros((), device=device)
            if args.decoder in ("ar", "both"):
                loss = loss + ar_loss(ar, tokens, context, actions, mask)
            if args.decoder in ("batch", "both"):
                batch_output = batch(tokens, context)
                batch_output.policy_logits = batch_output.policy_logits[:, : actions.shape[1]]
                batch_output.q_values = batch_output.q_values[:, : actions.shape[1]]
                batch_output.valid_rows = batch_output.valid_rows[:, : actions.shape[1]]
                loss = loss + batch_sequence_loss(batch_output, actions, mask)["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            values.append(float(loss.detach()))
        print(f"epoch={epoch + 1} loss={np.mean(values):.6f}")
    metadata.update({"sequence_data": str(args.data), "sequence_epochs": args.epochs})
    save_checkpoint(args.out, {"core": core, "dynamics": dynamics, "ar": ar, "batch": batch}, metadata)


if __name__ == "__main__":
    main()
