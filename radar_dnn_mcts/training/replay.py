from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import torch


@dataclass
class TrainingSample:
    tokens: torch.Tensor
    context: torch.Tensor
    policy_target: torch.Tensor
    q_target: torch.Tensor
    q_mask: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.data: deque[TrainingSample] = deque(maxlen=int(capacity))

    def extend(self, samples: list[TrainingSample]) -> None:
        self.data.extend(samples)

    def sample(self, batch_size: int) -> list[TrainingSample]:
        return random.sample(self.data, min(int(batch_size), len(self.data)))

    def __len__(self) -> int:
        return len(self.data)
