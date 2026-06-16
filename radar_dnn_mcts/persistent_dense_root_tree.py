from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from batched_branch_sim import BranchStepResult
from persistent_root_search import PersistentRootSearch, RootSearchWave


@dataclass
class DenseRootTreeUpdate:
    """A single persistent root expansion/update wave."""

    actions: np.ndarray
    prior_scores: np.ndarray
    rewards: np.ndarray
    executed: np.ndarray
    indices: np.ndarray
    inserted: int
    updated: int


class PersistentDenseRootTree:
    """Dense root tree state around ``PersistentRootSearch``.

    The tree is deliberately root-only for the performance lab. It keeps the
    MCTS-style arrays we need for a fast root selector while delegating neural
    proposals and exact C branch simulation to ``PersistentRootSearch``.
    """

    def __init__(self, search: PersistentRootSearch, capacity: int = 256):
        self.search = search
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")

        self.actions = np.full((self.capacity,), -1, dtype=np.int32)
        self.prior_scores = np.full((self.capacity,), -np.inf, dtype=np.float32)
        self.visits = np.zeros((self.capacity,), dtype=np.int32)
        self.value_sums = np.zeros((self.capacity,), dtype=np.float32)
        self.reward_sums = np.zeros((self.capacity,), dtype=np.float32)
        self.elapsed_ms = np.zeros((self.capacity,), dtype=np.float32)
        self.executed = np.full((self.capacity,), -1, dtype=np.int32)
        self.valid = np.zeros((self.capacity,), dtype=bool)
        self._action_to_index: dict[int, int] = {}
        self.size = 0

    @property
    def total_visits(self) -> int:
        return int(np.sum(self.visits[self.valid]))

    def update_from_wave(self, wave: RootSearchWave) -> DenseRootTreeUpdate:
        actions = np.asarray(wave.actions, dtype=np.int32)
        prior_scores = np.asarray(wave.scores, dtype=np.float32)
        rewards = np.asarray(wave.sim.rewards, dtype=np.float32)
        executed = np.asarray(wave.sim.executed, dtype=np.int32)
        dt_ms = np.asarray(wave.sim.dt_ms, dtype=np.float32)

        indices = np.full((actions.size,), -1, dtype=np.int32)
        inserted = 0
        updated = 0
        for row, action in enumerate(actions.tolist()):
            action = int(action)
            if action < 0:
                continue
            idx = self._action_to_index.get(action)
            if idx is None:
                if self.size >= self.capacity:
                    continue
                idx = self.size
                self.size += 1
                self._action_to_index[action] = idx
                self.actions[idx] = action
                self.valid[idx] = True
                inserted += 1
            else:
                updated += 1

            indices[row] = idx
            self.prior_scores[idx] = max(float(self.prior_scores[idx]), float(prior_scores[row]))
            self.visits[idx] += 1
            self.value_sums[idx] += float(rewards[row])
            self.reward_sums[idx] += float(rewards[row])
            self.elapsed_ms[idx] = float(dt_ms[row])
            self.executed[idx] = int(executed[row])

        return DenseRootTreeUpdate(
            actions=actions,
            prior_scores=prior_scores,
            rewards=rewards,
            executed=executed,
            indices=indices,
            inserted=int(inserted),
            updated=int(updated),
        )

    def expand_root(self, top_k: int) -> DenseRootTreeUpdate:
        return self.update_from_wave(self.search.search_wave(top_k=int(top_k)))

    def expand_root_cached(self, top_k: int, only_new: bool = True) -> DenseRootTreeUpdate:
        exclude = set(self._action_to_index) if only_new else None
        actions, scores = self.search.propose_cached(top_k=int(top_k), exclude=exclude)
        if actions.size == 0:
            empty_sim = BranchStepResult(
                rewards=np.empty((0,), dtype=np.float32),
                dt_ms=np.empty((0,), dtype=np.float32),
                executed=np.empty((0,), dtype=np.int32),
                terminals=np.empty((0,), dtype=np.uint8),
                observations=[],
            )
            return self.update_from_wave(RootSearchWave(actions=actions, scores=scores, sim=empty_sim))
        result = self.search.simulate(actions)
        return self.update_from_wave(RootSearchWave(actions=actions, scores=scores, sim=result))

    def q_values(self) -> np.ndarray:
        q = np.full((self.capacity,), -np.inf, dtype=np.float32)
        q[self.valid] = self.value_sums[self.valid] / np.maximum(self.visits[self.valid], 1)
        return q

    def prior_probabilities(self) -> np.ndarray:
        probs = np.zeros((self.capacity,), dtype=np.float32)
        valid_scores = self.prior_scores[self.valid]
        if valid_scores.size == 0:
            return probs
        centered = valid_scores - np.max(valid_scores)
        exp_scores = np.exp(centered, dtype=np.float32)
        denom = float(np.sum(exp_scores))
        if denom <= 0.0 or not np.isfinite(denom):
            probs[self.valid] = 1.0 / float(valid_scores.size)
        else:
            probs[self.valid] = exp_scores / denom
        return probs

    def puct_scores(self, c_puct: float = 1.25) -> np.ndarray:
        q = self.q_values()
        prior = self.prior_probabilities()
        parent_visits = max(self.total_visits, 1)
        explore = float(c_puct) * prior * np.sqrt(float(parent_visits)) / (1.0 + self.visits)
        score = q + explore
        return np.where(self.valid, score, -np.inf)

    def select_index(self, c_puct: float = 1.25) -> int:
        scores = self.puct_scores(c_puct=float(c_puct))
        if not np.isfinite(scores).any():
            return -1
        return int(np.nanargmax(scores))

    def select_action(self, c_puct: float = 1.25) -> int:
        idx = self.select_index(c_puct=float(c_puct))
        return int(self.actions[idx]) if idx >= 0 else -1

    def close(self) -> None:
        self.search.close()
