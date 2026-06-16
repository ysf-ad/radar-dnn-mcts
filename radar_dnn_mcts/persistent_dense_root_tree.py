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

    def __init__(self, search: PersistentRootSearch, capacity: int = 256, maintain_action_index: bool = True):
        self.search = search
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self.maintain_action_index = bool(maintain_action_index)

        self.actions = np.full((self.capacity,), -1, dtype=np.int32)
        self.prior_scores = np.full((self.capacity,), -np.inf, dtype=np.float32)
        self.visits = np.zeros((self.capacity,), dtype=np.int32)
        self.value_sums = np.zeros((self.capacity,), dtype=np.float32)
        self.reward_sums = np.zeros((self.capacity,), dtype=np.float32)
        self.elapsed_ms = np.zeros((self.capacity,), dtype=np.float32)
        self.executed = np.full((self.capacity,), -1, dtype=np.int32)
        self.valid = np.zeros((self.capacity,), dtype=bool)
        self._q_live_cache = np.zeros((self.capacity,), dtype=np.float32)
        self._visit_inv_cache = np.zeros((self.capacity,), dtype=np.float32)
        self._action_to_index: dict[int, int] = {}
        self._root_action_cursor = 0
        self._total_visits = 0
        self._prior_dirty = True
        self._prior_live_cache = np.empty((0,), dtype=np.float32)
        self._prior_live_size = 0
        self._puct_scratch = np.empty((self.capacity,), dtype=np.float32)
        self.size = 0

    @property
    def total_visits(self) -> int:
        return int(self._total_visits)

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
                if self.maintain_action_index:
                    self._action_to_index[action] = idx
                self.actions[idx] = action
                self.valid[idx] = True
                inserted += 1
            else:
                updated += 1

            indices[row] = idx
            old_prior = float(self.prior_scores[idx])
            new_prior = max(old_prior, float(prior_scores[row]))
            if new_prior != old_prior:
                self.prior_scores[idx] = new_prior
                self._prior_dirty = True
            self.visits[idx] += 1
            self._total_visits += 1
            self.value_sums[idx] += float(rewards[row])
            self.reward_sums[idx] += float(rewards[row])
            visits = max(int(self.visits[idx]), 1)
            self._q_live_cache[idx] = self.value_sums[idx] / visits
            self._visit_inv_cache[idx] = 1.0 / float(1 + visits)
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

    def append_new_from_wave(self, wave: RootSearchWave) -> DenseRootTreeUpdate:
        """Append a wave known to contain previously unseen actions.

        Cursor-based root expansion slices a sorted action table exactly once,
        so duplicate checking is unnecessary on the hot path. This preserves the
        action index map for callers that still need it, but moves array writes
        into one bulk slice assignment.
        """
        actions = np.asarray(wave.actions, dtype=np.int32)
        prior_scores = np.asarray(wave.scores, dtype=np.float32)
        rewards = np.asarray(wave.sim.rewards, dtype=np.float32)
        executed = np.asarray(wave.sim.executed, dtype=np.int32)
        dt_ms = np.asarray(wave.sim.dt_ms, dtype=np.float32)
        take = min(int(actions.size), max(0, self.capacity - self.size))
        indices = np.full((actions.size,), -1, dtype=np.int32)
        if take <= 0:
            return DenseRootTreeUpdate(
                actions=actions,
                prior_scores=prior_scores,
                rewards=rewards,
                executed=executed,
                indices=indices,
                inserted=0,
                updated=0,
            )

        start = int(self.size)
        stop = start + int(take)
        idx = np.arange(start, stop, dtype=np.int32)
        self.actions[start:stop] = actions[:take]
        self.prior_scores[start:stop] = prior_scores[:take]
        self.visits[start:stop] = 1
        self._total_visits += int(take)
        self.value_sums[start:stop] = rewards[:take]
        self.reward_sums[start:stop] = rewards[:take]
        self._q_live_cache[start:stop] = rewards[:take]
        self._visit_inv_cache[start:stop] = 0.5
        self.elapsed_ms[start:stop] = dt_ms[:take]
        self.executed[start:stop] = executed[:take]
        self.valid[start:stop] = True
        indices[:take] = idx
        if self.maintain_action_index:
            for action, action_idx in zip(actions[:take].tolist(), idx.tolist()):
                self._action_to_index[int(action)] = int(action_idx)
        self.size = stop
        self._prior_dirty = True
        return DenseRootTreeUpdate(
            actions=actions,
            prior_scores=prior_scores,
            rewards=rewards,
            executed=executed,
            indices=indices,
            inserted=int(take),
            updated=0,
        )

    def expand_root(self, top_k: int) -> DenseRootTreeUpdate:
        return self.update_from_wave(self.search.search_wave(top_k=int(top_k)))

    def expand_root_cached(self, top_k: int, only_new: bool = True) -> DenseRootTreeUpdate:
        exclude = set(self._action_to_index) if only_new else None
        if only_new and not self.maintain_action_index:
            raise RuntimeError("expand_root_cached(only_new=True) requires maintain_action_index=True")
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
        wave = RootSearchWave(actions=actions, scores=scores, sim=result)
        return self.append_new_from_wave(wave) if only_new else self.update_from_wave(wave)

    def propose_cached_cursor(self, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        actions, scores = self.search.propose_cached_view(top_k=int(top_k), offset=int(self._root_action_cursor))
        self._root_action_cursor += int(actions.size)
        return actions, scores

    def expand_root_cached_cursor(self, top_k: int) -> DenseRootTreeUpdate:
        actions, scores = self.propose_cached_cursor(top_k=int(top_k))
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
        return self.append_new_from_wave(RootSearchWave(actions=actions, scores=scores, sim=result))

    def expand_root_cached_cursor_bulk(self, total_k: int) -> DenseRootTreeUpdate:
        """Expand many cached cursor actions in one simulator batch.

        Root cursor actions are independent one-step branches from the same
        root snapshot. Simulating them together exposes more parallel work to
        the C vector environment and avoids per-wave restore/step/update
        overhead in the performance lab root-search path.
        """
        actions, scores = self.propose_cached_cursor(top_k=int(total_k))
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
        return self.append_new_from_wave(RootSearchWave(actions=actions, scores=scores, sim=result))

    def q_values(self) -> np.ndarray:
        q = np.full((self.capacity,), -np.inf, dtype=np.float32)
        q[: self.size] = self._q_live_cache[: self.size]
        return q

    def prior_probabilities(self) -> np.ndarray:
        probs = np.zeros((self.capacity,), dtype=np.float32)
        prior_live = self._live_prior()
        if prior_live.size == 0:
            return probs
        probs[: self.size] = prior_live
        return probs

    def puct_scores(self, c_puct: float = 1.25) -> np.ndarray:
        q_live, prior_live = self._live_q_prior()
        scores = np.full((self.capacity,), -np.inf, dtype=np.float32)
        if self.size <= 0:
            return scores
        parent_visits = max(self.total_visits, 1)
        visits = self.visits[: self.size]
        explore = float(c_puct) * prior_live * np.sqrt(float(parent_visits)) / (1.0 + visits)
        scores[: self.size] = q_live + explore
        return scores

    def select_index(self, c_puct: float = 1.25) -> int:
        if self.size <= 0:
            return -1
        q_live, prior_live = self._live_q_prior()
        parent_visits = max(self.total_visits, 1)
        live_scores = self._puct_scratch[: self.size]
        np.multiply(prior_live, float(c_puct) * np.sqrt(float(parent_visits)), out=live_scores)
        np.multiply(live_scores, self._visit_inv_cache[: self.size], out=live_scores)
        np.add(live_scores, q_live, out=live_scores)
        return int(np.argmax(live_scores))

    def select_action(self, c_puct: float = 1.25) -> int:
        idx = self.select_index(c_puct=float(c_puct))
        return int(self.actions[idx]) if idx >= 0 else -1

    def close(self) -> None:
        self.search.close()

    def _live_q_prior(self) -> tuple[np.ndarray, np.ndarray]:
        return self._q_live_cache[: self.size], self._live_prior()

    def _live_prior(self) -> np.ndarray:
        if self.size <= 0:
            return np.empty((0,), dtype=np.float32)
        if not self._prior_dirty and self._prior_live_size == self.size:
            return self._prior_live_cache[: self.size]
        prior_scores = self.prior_scores[: self.size]
        centered = prior_scores - np.max(prior_scores)
        exp_scores = np.exp(centered, dtype=np.float32)
        denom = float(np.sum(exp_scores))
        if denom <= 0.0 or not np.isfinite(denom):
            prior = np.full_like(exp_scores, 1.0 / float(exp_scores.size), dtype=np.float32)
        else:
            prior = exp_scores / denom
        self._prior_live_cache = prior.astype(np.float32, copy=False)
        self._prior_live_size = int(self.size)
        self._prior_dirty = False
        return self._prior_live_cache[: self.size]
