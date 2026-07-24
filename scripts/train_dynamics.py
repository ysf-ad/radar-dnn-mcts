from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from radar_dnn_mcts.models.checkpoint import load_checkpoint, save_checkpoint
from radar_dnn_mcts.models.decoders import AutoregressiveDecoder, BatchDecoder
from radar_dnn_mcts.models.dynamics import LatentDynamics
from radar_dnn_mcts.models.scheduler import PolicyQOutput, RadarSchedulerModel
from radar_dnn_mcts.training.losses import LossWeights, policy_q_loss


def select_output(output: PolicyQOutput, indices: torch.Tensor) -> PolicyQOutput:
    """Select active roots without changing the prediction contract."""
    return PolicyQOutput(
        output.policy_logits[indices],
        output.q_values[indices],
        output.valid_rows[indices],
        None if output.type_logits is None else output.type_logits[indices],
        None if output.target_logits is None else output.target_logits[indices],
    )


def main() -> None:
    """Jointly optimize MuZero representation, dynamics, and prediction."""
    parser = argparse.ArgumentParser(
        description="Jointly train MuZero h/g/f with recurrent policy, value, and reward targets."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--unroll-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--core-lr-scale", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--reward-loss-weight", type=float, default=32.0)
    parser.add_argument("--return-scale", type=float, default=32.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    arrays = np.load(args.data)
    required = {
        "tokens",
        "context",
        "policy",
        "actions",
        "rewards",
        "returns",
        "action_mask",
    }
    missing = required - set(arrays.files)
    if missing:
        raise KeyError(f"dataset is missing arrays: {sorted(missing)}")

    device = torch.device(args.device)
    data = {
        name: torch.as_tensor(arrays[name], device=device)
        for name in required
    }
    core = RadarSchedulerModel().to(device)
    dynamics = LatentDynamics().to(device)
    ar = AutoregressiveDecoder(core).to(device)
    batch = BatchDecoder().to(device)
    metadata = load_checkpoint(
        args.checkpoint,
        {"core": core, "dynamics": dynamics, "ar": ar, "batch": batch},
        device,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": core.parameters(), "lr": args.lr * args.core_lr_scale},
            {"params": dynamics.parameters(), "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    parameters = list(core.parameters()) + list(dynamics.parameters())
    positions = torch.nonzero(data["action_mask"].bool())
    count = positions.shape[0]
    sequence_steps = data["actions"].shape[1]
    max_steps = min(sequence_steps, args.unroll_steps)
    rows = data["policy"].shape[-1]
    selected_actions = F.one_hot(data["actions"].long(), num_classes=rows).bool()
    selected_actions[:, :, 0] = False
    selected_before = (
        selected_actions.long().cumsum(dim=1) - selected_actions.long()
    ).bool()
    weights = LossWeights(q=args.value_loss_weight)

    for epoch in range(args.epochs):
        order = torch.randperm(count, device=device)
        epoch_losses = []
        core.train()
        dynamics.train()
        for start in range(0, count, args.batch_size):
            roots = positions[order[start : start + args.batch_size]]
            windows = roots[:, 0]
            root_steps = roots[:, 1]
            root_context = data["context"][windows, root_steps].float()
            state = core.encode(
                data["tokens"][windows, root_steps].float(), root_context
            )
            selected = selected_before[windows, root_steps].clone()
            recurrent_context = root_context.clone()
            loss = torch.zeros((), device=device)
            terms = 0
            for offset in range(max_steps):
                # Shorter trajectory suffixes become inactive during unrolling.
                steps = root_steps + offset
                in_bounds = steps < sequence_steps
                safe_steps = steps.clamp_max(sequence_steps - 1)
                active_mask = (
                    in_bounds
                    & data["action_mask"][windows, safe_steps].bool()
                )
                active = torch.nonzero(active_mask, as_tuple=True)[0]
                if not active.numel():
                    break
                recurrent_context = recurrent_context.clone()
                step_context = data["context"][
                    windows, safe_steps
                ].float()
                recurrent_context[:, :4] = step_context[:, :4]
                output = core.predict(state, recurrent_context, selected)
                prediction_loss = policy_q_loss(
                    select_output(output, active),
                    data["policy"][
                        windows[active], safe_steps[active]
                    ].float(),
                    data["returns"][
                        windows[active], safe_steps[active]
                    ].float(),
                    weights,
                )["loss"]

                chosen = data["actions"][windows, safe_steps].long()
                chosen = torch.where(in_bounds, chosen, torch.zeros_like(chosen))
                next_state, predicted_reward = dynamics(state, chosen)
                reward_target = (
                    data["rewards"][windows, safe_steps].float()
                    / args.return_scale
                )
                reward_loss = F.smooth_l1_loss(
                    predicted_reward[active],
                    reward_target[active],
                )
                loss = loss + prediction_loss + args.reward_loss_weight * reward_loss
                terms += 1

                selected = selected.clone()
                selected.scatter_(1, chosen[:, None], (chosen > 0)[:, None])
                # MuZero General trainer.py scales each recurrent hidden-state
                # gradient without changing its forward value.
                next_state.global_state.register_hook(lambda grad: grad * 0.5)
                next_state.target_states.register_hook(lambda grad: grad * 0.5)
                state = next_state

            loss = loss / max(terms, 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        print(f"epoch={epoch + 1} loss={np.mean(epoch_losses):.6f}")

    metadata.update(
        {
            "dynamics_data": str(args.data),
            "dynamics_epochs": args.epochs,
            "muzero_unroll_steps": max_steps,
            "muzero_objective": "joint_recurrent_policy_value_reward",
            "muzero_core_lr_scale": args.core_lr_scale,
            "muzero_recurrent_gradient_scale": 0.5,
            "muzero_policy_target": "soft_puct_visits",
            "muzero_value_target": "episode_return",
            "muzero_reward_target": "executed_transition_reward",
            "muzero_training_roots": "all_executed_decisions",
            "muzero_next_state_source": "recurrent_latent_dynamics",
            "muzero_finetuned_core": True,
        }
    )
    save_checkpoint(
        args.out,
        {"core": core, "dynamics": dynamics, "ar": ar, "batch": batch},
        metadata,
    )


if __name__ == "__main__":
    main()
