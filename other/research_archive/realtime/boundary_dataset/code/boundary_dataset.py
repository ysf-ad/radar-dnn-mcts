"""Honest, fixed-shape datasets for predicting a scheduled-plan boundary state.

The observation at the end of the suffix is deliberately a label.  It is never
returned by :meth:`BoundaryTensorDataset.model_inputs`, which is the preferred
inference/training input boundary for callers that want to avoid accidental
future-arrival leakage.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


Array = np.ndarray
Observation = Mapping[str, Any]
Tokenizer = Callable[[Observation], Array]
ExecuteSuffix = Callable[[Sequence[int], float], Observation]


@dataclass(frozen=True)
class BoundaryRecord:
    """One causal midpoint-to-boundary transition.

    ``boundary_tokens`` may contain the effects of stochastic arrivals realized
    while executing the suffix.  Those effects are a prediction target and are
    intentionally absent from every other field.
    """

    midpoint_tokens: Array
    suffix_actions: Array
    suffix_length: int
    remaining_time_ms: float
    boundary_tokens: Array
    seed: int
    episode: int = 0
    midpoint_step: int = 0


def _fixed_tokens(value: Array, token_shape: tuple[int, int], name: str) -> Array:
    out = np.asarray(value, dtype=np.float32)
    if out.shape != token_shape:
        raise ValueError(f"{name} has shape {out.shape}; expected {token_shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.array(out, dtype=np.float32, copy=True)


class BoundaryCollector:
    """Collect labels by executing known suffixes from copied midpoint state.

    The environment-specific caller owns positioning the simulator at the
    midpoint. ``execute_suffix`` must execute only the supplied scheduled
    actions (up to ``remaining_time_ms``) and return the resulting observation.
    """

    def __init__(self, tokenizer: Tokenizer, token_shape: tuple[int, int], max_suffix: int):
        if max_suffix <= 0:
            raise ValueError("max_suffix must be positive")
        self.tokenizer = tokenizer
        self.token_shape = tuple(int(v) for v in token_shape)
        self.max_suffix = int(max_suffix)
        self.records: list[BoundaryRecord] = []

    def collect(
        self,
        midpoint_observation: Observation,
        remaining_scheduled_actions: Sequence[int],
        remaining_time_ms: float,
        execute_suffix: ExecuteSuffix,
        *,
        seed: int,
        episode: int = 0,
        midpoint_step: int = 0,
    ) -> BoundaryRecord:
        if remaining_time_ms < 0 or not np.isfinite(remaining_time_ms):
            raise ValueError("remaining_time_ms must be finite and non-negative")
        actions = np.asarray(remaining_scheduled_actions, dtype=np.int64).reshape(-1)
        if len(actions) > self.max_suffix:
            raise ValueError(f"suffix has {len(actions)} actions; max_suffix={self.max_suffix}")
        if np.any(actions < 0):
            raise ValueError("scheduled actions must be non-negative integer IDs")

        # Tokenize an isolated copy before executing. This protects against an
        # environment returning mutable observation views that change in-place.
        midpoint = copy.deepcopy(dict(midpoint_observation))
        midpoint_tokens = _fixed_tokens(
            self.tokenizer(midpoint), self.token_shape, "midpoint_tokens"
        )
        boundary_observation = execute_suffix(actions.tolist(), float(remaining_time_ms))
        boundary_tokens = _fixed_tokens(
            self.tokenizer(copy.deepcopy(dict(boundary_observation))),
            self.token_shape,
            "boundary_tokens",
        )
        padded = np.zeros(self.max_suffix, dtype=np.int64)
        padded[: len(actions)] = actions
        record = BoundaryRecord(
            midpoint_tokens=midpoint_tokens,
            suffix_actions=padded,
            suffix_length=int(len(actions)),
            remaining_time_ms=float(remaining_time_ms),
            boundary_tokens=boundary_tokens,
            seed=int(seed),
            episode=int(episode),
            midpoint_step=int(midpoint_step),
        )
        self.records.append(record)
        return record

    def save(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
        save_records(path, self.records, metadata=metadata)


def save_records(
    path: str | Path,
    records: Sequence[BoundaryRecord],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    if not records:
        raise ValueError("cannot save an empty boundary dataset")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        midpoint_tokens=np.stack([r.midpoint_tokens for r in records]),
        suffix_actions=np.stack([r.suffix_actions for r in records]),
        suffix_length=np.asarray([r.suffix_length for r in records], dtype=np.int64),
        remaining_time_ms=np.asarray([r.remaining_time_ms for r in records], dtype=np.float32),
        boundary_tokens=np.stack([r.boundary_tokens for r in records]),
        seed=np.asarray([r.seed for r in records], dtype=np.int64),
        episode=np.asarray([r.episode for r in records], dtype=np.int64),
        midpoint_step=np.asarray([r.midpoint_step for r in records], dtype=np.int64),
        metadata_json=np.asarray(json.dumps(dict(metadata or {}), sort_keys=True)),
        format_version=np.asarray(1, dtype=np.int64),
    )


class BoundaryTensorDataset(Dataset):
    REQUIRED = (
        "midpoint_tokens",
        "suffix_actions",
        "suffix_length",
        "remaining_time_ms",
        "boundary_tokens",
        "seed",
    )

    def __init__(self, arrays: Mapping[str, Array]):
        missing = [name for name in self.REQUIRED if name not in arrays]
        if missing:
            raise ValueError(f"missing boundary dataset arrays: {missing}")
        self.arrays = {name: np.asarray(arrays[name]) for name in self.REQUIRED}
        n = len(self.arrays["seed"])
        if any(len(value) != n for value in self.arrays.values()):
            raise ValueError("boundary dataset arrays have inconsistent lengths")
        if self.arrays["midpoint_tokens"].shape != self.arrays["boundary_tokens"].shape:
            raise ValueError("midpoint and boundary token shapes differ")
        width = self.arrays["suffix_actions"].shape[1]
        lengths = self.arrays["suffix_length"]
        if np.any(lengths < 0) or np.any(lengths > width):
            raise ValueError("suffix_length is outside the padded suffix width")

    @classmethod
    def load(cls, path: str | Path) -> "BoundaryTensorDataset":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls({name: np.array(data[name], copy=True) for name in cls.REQUIRED})

    def __len__(self) -> int:
        return len(self.arrays["seed"])

    def model_inputs(self, index: int) -> dict[str, torch.Tensor]:
        """Return causal inputs only; the true boundary is never included."""
        length = int(self.arrays["suffix_length"][index])
        width = self.arrays["suffix_actions"].shape[1]
        return {
            "midpoint_tokens": torch.as_tensor(self.arrays["midpoint_tokens"][index], dtype=torch.float32),
            "suffix_actions": torch.as_tensor(self.arrays["suffix_actions"][index], dtype=torch.long),
            "suffix_mask": torch.arange(width, dtype=torch.long) < length,
            "remaining_time_ms": torch.as_tensor(self.arrays["remaining_time_ms"][index], dtype=torch.float32),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.model_inputs(index)
        item["boundary_tokens"] = torch.as_tensor(
            self.arrays["boundary_tokens"][index], dtype=torch.float32
        )
        item["seed"] = torch.as_tensor(self.arrays["seed"][index], dtype=torch.long)
        return item


def split_by_seed(
    dataset: BoundaryTensorDataset,
    held_out_seeds: Sequence[int] | None = None,
    held_out_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint train/eval indices, grouping every sample by seed."""
    seeds = np.asarray(dataset.arrays["seed"], dtype=np.int64)
    unique = np.unique(seeds)
    if len(unique) < 2:
        raise ValueError("at least two distinct seeds are required for a held-out split")
    if held_out_seeds is None:
        count = min(len(unique) - 1, max(1, int(round(len(unique) * held_out_fraction))))
        held = unique[-count:]
    else:
        held = np.asarray(sorted(set(int(v) for v in held_out_seeds)), dtype=np.int64)
        unknown = np.setdiff1d(held, unique)
        if len(unknown):
            raise ValueError(f"held-out seeds absent from dataset: {unknown.tolist()}")
    eval_mask = np.isin(seeds, held)
    train_idx = np.flatnonzero(~eval_mask)
    eval_idx = np.flatnonzero(eval_mask)
    if not len(train_idx) or not len(eval_idx):
        raise ValueError("seed split produced an empty train or evaluation partition")
    return train_idx, eval_idx
