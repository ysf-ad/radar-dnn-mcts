"""Train AR and batch decoders from grouped PUCT window trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from radar_dnn_mcts.models.backbone import EncodedState
from radar_dnn_mcts.models.checkpoint import load_checkpoint, save_checkpoint
from radar_dnn_mcts.models.decoders import AutoregressiveDecoder, BatchDecoder
from radar_dnn_mcts.models.dynamics import LatentDynamics
from radar_dnn_mcts.models.scheduler import PolicyQOutput, RadarSchedulerModel
from radar_dnn_mcts.training.losses import (
    LossWeights,
    batch_sequence_loss,
    policy_q_loss,
)


def ar_predictions(
    model: AutoregressiveDecoder,
    tokens: torch.Tensor,
    root_context: torch.Tensor,
    step_context: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    policies: torch.Tensor,
    returns: torch.Tensor,
) -> tuple[PolicyQOutput, torch.Tensor, torch.Tensor]:
    """Replay an MCTS trajectory and align each prefix with its PUCT targets."""
    state, _ = model.initial(tokens, root_context)
    # Replaying MCTS-selected earlier actions reconstructs the self-play prefix;
    # the current policy target remains the soft PUCT visit distribution.
    prefix_deltas = model.decode_prefix_deltas(state, actions[:, :-1])
    rows = policies.shape[-1]
    selected_actions = F.one_hot(actions, num_classes=rows).bool()
    selected_actions[:, :, 0] = False
    selected_before = selected_actions.long().cumsum(dim=1) - selected_actions.long()

    batch_index, step_index = torch.nonzero(action_mask, as_tuple=True)
    # Flatten real positions so padding contributes neither policy nor value
    # loss, then recover each position's prefix-conditioned representation.
    delta = prefix_deltas[batch_index, step_index]
    context_delta = model.context_delta(delta)
    context_delta = torch.where(
        (step_index > 0)[:, None], context_delta, torch.zeros_like(context_delta)
    )
    conditioned = EncodedState(
        state.global_state[batch_index] + delta,
        state.target_states[batch_index],
        state.valid_rows[batch_index],
    )
    unavailable = selected_before[batch_index, step_index].bool()
    output = model.backbone.predict(
        conditioned,
        step_context[batch_index, step_index] + context_delta,
        unavailable,
    )
    return (
        output,
        policies[batch_index, step_index],
        returns[batch_index, step_index],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train AR and batch decoders directly from grouped PUCT windows."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decoder", choices=["ar", "batch", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
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
    arrays = np.load(args.data)
    names = (
        "tokens",
        "context",
        "actions",
        "policy",
        "returns",
        "action_mask",
    )
    missing = set(names) - set(arrays.files)
    if missing:
        raise KeyError(f"dataset is missing arrays: {sorted(missing)}")
    data = {name: torch.as_tensor(arrays[name], device=device) for name in names}

    parameters: list[torch.nn.Parameter] = []
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
            index = order[start : start + args.batch_size]
            step_context = data["context"][index].float()
            actions = data["actions"][index].long()
            mask = data["action_mask"][index].bool()
            policies = data["policy"][index].float()
            returns = data["returns"][index].float()
            active_steps = int(mask.sum(dim=1).max().item())
            # Trim unused tail padding before running either decoder.
            tokens = data["tokens"][index, 0].float()
            root_context = step_context[:, 0]
            step_context = step_context[:, :active_steps]
            actions = actions[:, :active_steps]
            mask = mask[:, :active_steps]
            policies = policies[:, :active_steps]
            returns = returns[:, :active_steps]

            loss = torch.zeros((), device=device)
            if args.decoder in ("ar", "both"):
                # AR predicts each position conditioned on recorded earlier
                # actions and learns from its PUCT policy/value targets.
                output, policy_target, value_target = ar_predictions(
                    ar,
                    tokens,
                    root_context,
                    step_context,
                    actions,
                    mask,
                    policies,
                    returns,
                )
                loss = loss + policy_q_loss(
                    output,
                    policy_target,
                    value_target,
                    weights=LossWeights(q=args.value_loss_weight),
                )["loss"]
            if args.decoder in ("batch", "both"):
                # Batch predicts all positions in parallel and is supervised
                # against the same grouped trajectory.
                output = batch(tokens, root_context)
                output.policy_logits = output.policy_logits[:, :active_steps]
                output.q_values = output.q_values[:, :active_steps]
                output.valid_rows = output.valid_rows[:, :active_steps]
                loss = loss + batch_sequence_loss(
                    output,
                    actions,
                    mask,
                    q_target=returns,
                    weights=LossWeights(q=args.value_loss_weight),
                )["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            values.append(float(loss.detach()))
        print(f"epoch={epoch + 1} loss={np.mean(values):.6f}")

    metadata.update(
        {
            "sequence_data": str(args.data),
            "sequence_epochs": args.epochs,
            "ar_policy_target": "soft_puct_visits",
            "batch_policy_target": "selected_window_trajectory",
        }
    )
    save_checkpoint(
        args.out,
        {"core": core, "dynamics": dynamics, "ar": ar, "batch": batch},
        metadata,
    )


if __name__ == "__main__":
    main()
