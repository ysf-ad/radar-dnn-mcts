from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from radar_dnn_mcts.env.actions import action_duration_ms
class Scheduler(Protocol):
    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]: ...


def device_of(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def choose_row(
    policy_logits: torch.Tensor,
    q_values: torch.Tensor,
    valid: torch.Tensor,
    policy_weight: float,
    q_weight: float,
) -> int:
    utility = policy_weight * policy_logits + q_weight * q_values
    utility = utility.masked_fill(~valid, -1e9)
    return int(torch.argmax(utility, dim=-1).item())


def selected_mask(
    rows: int,
    selected: set[int],
    device: torch.device,
    obs: dict | None = None,
    elapsed_ms: float = 0.0,
) -> torch.Tensor:
    mask = torch.zeros(1, rows, dtype=torch.bool, device=device)
    if selected:
        mask[0, torch.as_tensor(sorted(selected), device=device)] = True
    if obs is not None and rows > 1:
        deadline = np.asarray(obs["t_deadline"], dtype=np.float32)[: rows - 1]
        unavailable = np.flatnonzero(deadline <= float(elapsed_ms)) + 1
        if unavailable.size:
            mask[0, torch.as_tensor(unavailable, device=device)] = True
    return mask


def update_prefix(row: int, obs: dict, elapsed: float, searches: int, tracks: int, selected: set[int]):
    duration = action_duration_ms(obs, row)
    if row == 0:
        searches += 1
    else:
        tracks += 1
        selected.add(row)
    return elapsed + duration, searches, tracks


def tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array).unsqueeze(0).to(device)
