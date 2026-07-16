from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from radar_dnn_mcts.models.checkpoint import save_checkpoint
from radar_dnn_mcts.models.decoders import AutoregressiveDecoder, BatchDecoder
from radar_dnn_mcts.models.dynamics import LatentDynamics
from radar_dnn_mcts.models.scheduler import RadarSchedulerModel
from radar_dnn_mcts.training.losses import policy_q_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the common policy/Q model from PUCT-generated targets.")
    parser.add_argument("--data", type=Path, required=True, help="NPZ with tokens, context, policy, q, and q_mask")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=916)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    arrays = np.load(args.data)
    required = {"tokens", "context", "policy", "q", "q_mask"}
    missing = required - set(arrays.files)
    if missing:
        raise KeyError(f"dataset is missing arrays: {sorted(missing)}")
    device = torch.device(args.device)
    data = {name: torch.as_tensor(arrays[name], device=device) for name in required}
    core = RadarSchedulerModel().to(device)
    optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=1e-4)
    count = data["tokens"].shape[0]
    for epoch in range(args.epochs):
        permutation = torch.randperm(count, device=device)
        totals = []
        core.train()
        for start in range(0, count, args.batch_size):
            idx = permutation[start : start + args.batch_size]
            output = core(data["tokens"][idx].float(), data["context"][idx].float())
            losses = policy_q_loss(
                output,
                data["policy"][idx].float(),
                data["q"][idx].float(),
                data["q_mask"][idx].bool(),
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            optimizer.step()
            totals.append(float(losses["loss"].detach()))
        print(f"epoch={epoch + 1} loss={np.mean(totals):.6f}")

    dynamics = LatentDynamics().to(device)
    ar = AutoregressiveDecoder(core).to(device)
    batch = BatchDecoder().to(device)
    save_checkpoint(
        args.out,
        {"core": core, "dynamics": dynamics, "ar": ar, "batch": batch},
        {"seed": args.seed, "epochs": args.epochs, "source": str(args.data)},
    )


if __name__ == "__main__":
    main()
