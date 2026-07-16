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
    q_mask: torch.Tensor,
    weights: LossWeights = LossWeights(),
) -> dict[str, torch.Tensor]:
    target = policy_target / policy_target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    search_mass = target[:, 0]
    track_mass = target[:, 1:].sum(dim=-1)
    type_target = torch.stack([search_mass, track_mass], dim=-1)
    type_log_prob = F.log_softmax(output.type_logits, dim=-1)
    type_loss = -(type_target * type_log_prob).sum(dim=-1).mean()

    conditional = target[:, 1:] / track_mass[:, None].clamp_min(1e-8)
    target_log_prob = F.log_softmax(output.target_logits[:, 1:].masked_fill(~output.valid_rows[:, 1:], -1e9), dim=-1)
    target_ce = -(conditional * target_log_prob).sum(dim=-1)
    target_loss = (target_ce * (track_mass > 0).float()).sum() / (track_mass > 0).float().sum().clamp_min(1.0)
    policy_loss = type_loss + target_loss

    q_error = (output.q_values - q_target).square() * q_mask.float()
    q_loss = q_error.sum() / q_mask.float().sum().clamp_min(1.0)
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
    weights: LossWeights = LossWeights(),
) -> dict[str, torch.Tensor]:
    logits = output.policy_logits.flatten(0, 1)
    targets = action_rows.flatten()
    mask = action_mask.flatten().float()
    ce = F.cross_entropy(logits, targets, reduction="none")
    policy = (ce * mask).sum() / mask.sum().clamp_min(1.0)
    q = torch.zeros((), device=logits.device)
    if q_target is not None:
        selected_q = output.q_values.gather(-1, action_rows.unsqueeze(-1)).squeeze(-1)
        q = ((selected_q - q_target).square() * action_mask.float()).sum() / action_mask.float().sum().clamp_min(1.0)
    return {"loss": weights.policy * policy + weights.q * q, "policy": policy, "q": q}
