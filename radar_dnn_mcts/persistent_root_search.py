from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from batched_branch_sim import BatchedRootBranchSimulator, BranchStepResult
from batched_window_expansion import BatchedWindowExpansionScorer, BranchPrefix
from final_radar_campaign import get_obs
from perf_fast_planner import FastActionAttentionPlanner


@dataclass
class RootSearchWave:
    actions: np.ndarray
    scores: np.ndarray
    sim: BranchStepResult

    @property
    def reward_sum(self) -> float:
        return float(np.sum(self.sim.rewards))

    @property
    def executed_count(self) -> int:
        return int(np.sum(self.sim.executed >= 0))


class PersistentRootSearch:
    """Reusable exact root-search primitive with cached neural root state.

    The object owns:

    - one root C environment snapshot;
    - one vectorized C branch simulator;
    - one cached action-attention root scorer.

    It is intentionally root-scoped. Deeper tree reuse needs additional state
    snapshots per node, but root reuse already removes the measured dominant
    setup cost from repeated root expansion waves.
    """

    def __init__(
        self,
        planner: FastActionAttentionPlanner,
        initial_targets: int,
        seed: int,
        env_cfg: dict,
        batch_size: int,
        budget_ms: float = 200.0,
        max_trackers: int | None = None,
        window_ms: int = 200,
    ):
        kwargs = {}
        if max_trackers is not None:
            kwargs["max_trackers"] = int(max_trackers)
        self.sim = BatchedRootBranchSimulator(
            initial_targets=int(initial_targets),
            seed=int(seed),
            env_cfg=dict(env_cfg),
            batch_size=int(batch_size),
            window_ms=int(window_ms),
            **kwargs,
        )
        self.root_snapshot = self.sim.snapshot_root()
        self.root_obs = get_obs(self.sim.root_eng, 0.0)
        self.scorer = BatchedWindowExpansionScorer(planner, self.root_obs, budget_ms=float(budget_ms))

    def propose(self, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        prefixes = self.scorer.expand_prefixes([BranchPrefix()], top_k=int(top_k))
        actions = np.asarray([p.actions[-1] for p in prefixes], dtype=np.int32)
        scores = np.asarray([p.score_sum for p in prefixes], dtype=np.float32)
        return actions, scores

    def simulate(self, actions: np.ndarray) -> BranchStepResult:
        return self.sim.step_actions(np.asarray(actions, dtype=np.int32), snapshot=self.root_snapshot)

    def search_wave(self, top_k: int) -> RootSearchWave:
        actions, scores = self.propose(top_k=int(top_k))
        result = self.simulate(actions)
        return RootSearchWave(actions=actions, scores=scores, sim=result)

    def close(self) -> None:
        self.sim.close()


def sync_device(device: torch.device | str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
