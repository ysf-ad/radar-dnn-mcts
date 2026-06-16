from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perf_fast_planner import BatchedActionAttentionScorer, BatchedRootProposals


@dataclass
class DenseRootSearchState:
    """MCTX-inspired dense root tree state for radar root expansions.

    This is intentionally limited to the root for now. It gives us the tensor
    layout needed for batched MCTS work:

    - batch dimension = independent radar states/windows
    - action dimension = top-K valid root actions per state
    - dense arrays hold priors/scores/visits/values
    """

    actions: np.ndarray
    prior_scores: np.ndarray
    q_values: np.ndarray
    visits: np.ndarray
    value_sums: np.ndarray
    valid: np.ndarray

    @property
    def batch_size(self) -> int:
        return int(self.actions.shape[0])

    @property
    def num_actions(self) -> int:
        return int(self.actions.shape[1])

    def puct_scores(self, c_puct: float = 1.25) -> np.ndarray:
        parent_visits = np.maximum(self.visits.sum(axis=1, keepdims=True), 1)
        q = self.value_sums / np.maximum(self.visits, 1)
        explore = float(c_puct) * self.prior_scores * np.sqrt(parent_visits) / (1.0 + self.visits)
        score = q + explore
        return np.where(self.valid, score, -np.inf)

    def select(self, c_puct: float = 1.25) -> np.ndarray:
        return np.argmax(self.puct_scores(c_puct), axis=1).astype(np.int64)

    def selected_actions(self, c_puct: float = 1.25) -> np.ndarray:
        idx = self.select(c_puct)
        return self.actions[np.arange(self.batch_size), idx]

    def update_selected(self, returns: np.ndarray, c_puct: float = 1.25) -> None:
        idx = self.select(c_puct)
        rows = np.arange(self.batch_size)
        self.visits[rows, idx] += 1
        self.value_sums[rows, idx] += np.asarray(returns, dtype=np.float32)
        self.q_values[rows, idx] = self.value_sums[rows, idx] / np.maximum(self.visits[rows, idx], 1)


def build_dense_root_state(
    scorer: BatchedActionAttentionScorer,
    observations: list[dict],
    top_k: int = 16,
    **score_kwargs,
) -> DenseRootSearchState:
    proposals: BatchedRootProposals = scorer.topk_root_proposals(observations, k=int(top_k), **score_kwargs)
    visits = np.zeros_like(proposals.scores, dtype=np.int32)
    value_sums = np.zeros_like(proposals.scores, dtype=np.float32)
    q_values = np.zeros_like(proposals.scores, dtype=np.float32)
    return DenseRootSearchState(
        actions=proposals.actions,
        prior_scores=proposals.scores,
        q_values=q_values,
        visits=visits,
        value_sums=value_sums,
        valid=proposals.valid,
    )
