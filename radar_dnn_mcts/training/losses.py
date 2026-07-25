from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from radar_dnn_mcts.models.decoders import SequenceOutput
from radar_dnn_mcts.models.scheduler import PolicyQOutput


@dataclass(frozen=True)
class LossWeights:
    policy: float = 1.0
    q: float = 1.0
    dynamics_reward: float = 1.0
    consistency: float = 0.25


def policy_q_loss(
    output: PolicyQOutput,
    policy_target: torch.Tensor,
    q_target: torch.Tensor,
    weights: LossWeights = LossWeights(),
) -> dict[str, torch.Tensor]:
    """Train policy by cross-entropy and state value by mean-squared return error."""
    target = policy_target / policy_target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if output.type_logits is None or output.target_logits is None:
        policy_loss = F.cross_entropy(output.policy_logits, target)
        type_loss = torch.zeros((), device=policy_loss.device)
        target_loss = policy_loss
    else:
        # Marginalize complete-action targets into type and target factors.
        search_mass = target[:, 0]
        track_mass = target[:, 1:].sum(dim=-1)
        type_target = torch.stack([search_mass, track_mass], dim=-1)
        type_loss = F.cross_entropy(output.type_logits, type_target)

        conditional = target[:, 1:] / track_mass[:, None].clamp_min(1e-8)
        target_ce = F.cross_entropy(
            output.target_logits[:, 1:].masked_fill(
                ~output.valid_rows[:, 1:], -1e9
            ),
            conditional,
            reduction="none",
        )
        target_loss = (target_ce * track_mass).mean()
        policy_loss = type_loss + target_loss

    if q_target.ndim == 1:
        value_prediction = output.q_values[:, 0]
    else:
        value_prediction = output.q_values
    q_loss = F.mse_loss(value_prediction, q_target)
    total = weights.policy * policy_loss + weights.q * q_loss
    return {
        "loss": total,
        "policy": policy_loss,
        "type": type_loss,
        "target": target_loss,
        "q": q_loss,
    }


def batch_sequence_loss(
    output: SequenceOutput,
    action_rows: torch.Tensor,
    action_mask: torch.Tensor,
    q_target: torch.Tensor | None = None,
    policy_target: torch.Tensor | None = None,
    weights: LossWeights = LossWeights(),
) -> dict[str, torch.Tensor]:
    """Average slot policy/value losses over executed schedule positions."""
    logits = output.policy_logits.flatten(0, 1)
    mask = action_mask.flatten().float()
    if policy_target is None:
        ce = F.cross_entropy(logits, action_rows.flatten(), reduction="none")
    else:
        target = policy_target.flatten(0, 1)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        ce = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    policy = (ce * mask).sum() / mask.sum().clamp_min(1.0)
    q = torch.zeros((), device=logits.device)
    if q_target is not None:
        selected_q = output.q_values.gather(-1, action_rows.unsqueeze(-1)).squeeze(-1)
        q_error = F.mse_loss(selected_q, q_target, reduction="none")
        q = (q_error * action_mask.float()).sum() / action_mask.float().sum().clamp_min(1.0)
    return {"loss": weights.policy * policy + weights.q * q, "policy": policy, "q": q}
