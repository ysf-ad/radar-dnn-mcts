from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from radar_dnn_mcts.models.checkpoint import load_checkpoint, save_checkpoint
from radar_dnn_mcts.models.decoders import AutoregressiveDecoder, BatchDecoder
from radar_dnn_mcts.models.dynamics import LatentDynamics
from radar_dnn_mcts.models.scheduler import RadarSchedulerModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="NPZ with tokens, context, actions, rewards, next_tokens, next_context, mask")
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
    metadata = load_checkpoint(args.checkpoint, {"core": core, "dynamics": dynamics, "ar": ar, "batch": batch}, device)
    arrays = np.load(args.data)
    data = {name: torch.as_tensor(arrays[name], device=device) for name in ("tokens", "context", "actions", "rewards", "next_tokens", "next_context", "mask")}
    optimizer = torch.optim.AdamW(dynamics.parameters(), lr=args.lr, weight_decay=1e-4)
    core.eval()
    for epoch in range(args.epochs):
        with torch.no_grad():
            state = core.encode(data["tokens"].float(), data["context"].float())
        total = torch.zeros((), device=device)
        valid_count = data["mask"].float().sum().clamp_min(1.0)
        for step in range(data["actions"].shape[1]):
            state, reward = dynamics(state, data["actions"][:, step].long())
            with torch.no_grad():
                target = core.encode(data["next_tokens"][:, step].float(), data["next_context"][:, step].float())
            mask = data["mask"][:, step].float()
            consistency = (state.global_state - target.global_state).square().mean(dim=-1)
            consistency += (state.target_states - target.target_states).square().mean(dim=(-1, -2))
            reward_loss = (reward - data["rewards"][:, step].float()).square()
            total = total + ((consistency + reward_loss) * mask).sum() / valid_count
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
        optimizer.step()
        print(f"epoch={epoch + 1} loss={float(total.detach()):.6f}")
    metadata.update({"dynamics_data": str(args.data), "dynamics_epochs": args.epochs})
    save_checkpoint(args.out, {"core": core, "dynamics": dynamics, "ar": ar, "batch": batch}, metadata)


if __name__ == "__main__":
    main()
