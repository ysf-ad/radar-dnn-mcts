from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Action:
    """One physical action. Target row zero denotes search."""

    sensor: int
    row: int

    @property
    def is_search(self) -> bool:
        return self.row == 0


def valid_action_mask(
    active: np.ndarray | torch.Tensor,
    deadline: np.ndarray | torch.Tensor,
    selected: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray | torch.Tensor:
    """Return validity for rows [search, target_1, ..., target_N]."""

    if isinstance(active, torch.Tensor):
        active = active.bool()
        deadline = deadline.to(active.device)
        chosen = torch.zeros_like(active) if selected is None else selected.bool().to(active.device)
        tracks = active & (deadline >= 0) & ~chosen
        return torch.cat([torch.ones_like(tracks[..., :1]), tracks], dim=-1)
    active_np = np.asarray(active, dtype=bool)
    deadline_np = np.asarray(deadline)
    chosen_np = np.zeros_like(active_np) if selected is None else np.asarray(selected, dtype=bool)
    return np.concatenate([np.ones_like(active_np[..., :1], dtype=bool), active_np & (deadline_np >= 0) & ~chosen_np], axis=-1)


def action_duration_ms(obs: dict, row: int, search_dwell_ms: float = 10.0) -> float:
    if int(row) == 0:
        return float(search_dwell_ms)
    dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
    return float(dwell[int(row) - 1])


def fill_budget(rows: list[int], obs: dict, budget_ms: float = 200.0, search_dwell_ms: float = 10.0) -> list[int]:
    """Keep a legal prefix that fills the soft window without starting beyond it."""

    selected: set[int] = set()
    elapsed = 0.0
    output: list[int] = []
    for row in rows:
        row = int(row)
        if row > 0 and row in selected:
            continue
        duration = action_duration_ms(obs, row, search_dwell_ms)
        if elapsed >= budget_ms:
            break
        output.append(row)
        elapsed += duration
        if row > 0:
            selected.add(row)
    return output
