from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pufferlib.ocean.radarxs import binding
from pufferlib.ocean.radarxs.engine import FEATURES_PER_TRACKER, GRID_SIZE, get_obs_from_buf
from repaired_campaign_tools import build_env
from exact_env_mutual import _DummyPlanner, engine_env_cfg
from two_sensor_physical_head_eval import MAXT


@dataclass
class BranchStepResult:
    rewards: np.ndarray
    dt_ms: np.ndarray
    executed: np.ndarray
    terminals: np.ndarray
    observations: list[dict]


class BatchedRootBranchSimulator:
    """Vectorized one-step branch simulator for root/top-K actions.

    It uses the radar C binding's vector environment support:

    1. build one root environment;
    2. snapshot the root;
    3. restore that snapshot into every vector env;
    4. assign one candidate action per env;
    5. call one vectorized C step.

    This is the simulator-side counterpart to batched neural root scoring.
    """

    def __init__(
        self,
        initial_targets: int,
        seed: int,
        env_cfg: dict,
        batch_size: int,
        max_trackers: int = MAXT,
        window_ms: int = 200,
    ):
        self.initial_targets = int(initial_targets)
        self.seed = int(seed)
        self.env_cfg = engine_env_cfg(dict(env_cfg))
        self.batch_size = int(batch_size)
        self.max_trackers = int(max_trackers)
        self.window_ms = int(window_ms)
        self.obs_size = GRID_SIZE + self.max_trackers * FEATURES_PER_TRACKER + 1

        self.root_eng = build_env(_DummyPlanner(), self.initial_targets, self.max_trackers, self.seed, self.window_ms, self.env_cfg)
        self.root_eng.reset(seed=self.seed)
        self.obs_buf = np.zeros((self.batch_size, self.obs_size), dtype=np.float32)
        self.act_buf = np.zeros((self.batch_size,), dtype=np.int32)
        self.rew_buf = np.zeros((self.batch_size,), dtype=np.float32)
        self.term_buf = np.zeros((self.batch_size,), dtype=np.uint8)
        self.trunc_buf = np.zeros((self.batch_size,), dtype=np.uint8)
        self.dt_buf = np.zeros((self.batch_size,), dtype=np.float32)
        self.executed_buf = np.full((self.batch_size,), -1, dtype=np.int32)
        self.env = binding.vec_init(
            self.obs_buf,
            self.act_buf,
            self.rew_buf,
            self.term_buf,
            self.trunc_buf,
            self.batch_size,
            self.seed,
            initial_targets=self.initial_targets,
            max_trackers=self.max_trackers,
            **self.env_cfg,
        )

    def snapshot_root(self):
        return binding.vec_snapshot(self.root_eng.env)

    def restore_root(self, snapshot=None, count: int | None = None) -> None:
        snap = self.snapshot_root() if snapshot is None else snapshot
        if count is not None and hasattr(binding, "vec_restore_n"):
            binding.vec_restore_n(self.env, snap, int(count))
        else:
            binding.vec_restore_all(self.env, snap)

    def step_actions(self, actions: np.ndarray, snapshot=None, include_observations: bool = True) -> BranchStepResult:
        actions = np.asarray(actions, dtype=np.int32).reshape(-1)
        if actions.size > self.batch_size:
            raise ValueError(f"actions has {actions.size} items, batch_size={self.batch_size}")
        if actions.size == 0:
            return BranchStepResult(
                rewards=np.empty((0,), dtype=np.float32),
                dt_ms=np.empty((0,), dtype=np.float32),
                executed=np.empty((0,), dtype=np.int32),
                terminals=np.empty((0,), dtype=np.uint8),
                observations=[],
            )
        if hasattr(binding, "vec_restore_step_validated_into"):
            snap = self.snapshot_root() if snapshot is None else snapshot
            binding.vec_restore_step_validated_into(self.env, snap, actions, self.dt_buf, self.executed_buf, int(actions.size))
            dt = self.dt_buf[: actions.size].copy()
            executed = self.executed_buf[: actions.size].copy()
        elif hasattr(binding, "vec_step_validated_into"):
            self.restore_root(snapshot, count=int(actions.size))
            self.act_buf[: actions.size] = actions
            binding.vec_step_validated_into(self.env, self.dt_buf, self.executed_buf, int(actions.size))
            dt = self.dt_buf[: actions.size].copy()
            executed = self.executed_buf[: actions.size].copy()
        else:
            self.restore_root(snapshot, count=int(actions.size))
            self.act_buf[actions.size :] = -1
            self.act_buf[: actions.size] = actions
            info = binding.vec_step_validated(self.env)
            dt = np.asarray(info["dt"], dtype=np.float32)[: actions.size]
            executed = np.asarray(info["executed"], dtype=np.int32)[: actions.size]
        rewards = np.asarray(self.rew_buf[: actions.size], dtype=np.float32).copy()
        terminals = np.asarray(self.term_buf[: actions.size], dtype=np.uint8).copy()
        observations = (
            [get_obs_from_buf(self.obs_buf[i], max_trackers=self.max_trackers) for i in range(actions.size)]
            if include_observations
            else []
        )
        return BranchStepResult(rewards=rewards, dt_ms=dt, executed=executed, terminals=terminals, observations=observations)

    def step_actions_legacy(self, actions: np.ndarray, snapshot=None, include_observations: bool = True) -> BranchStepResult:
        actions = np.asarray(actions, dtype=np.int32).reshape(-1)
        if actions.size > self.batch_size:
            raise ValueError(f"actions has {actions.size} items, batch_size={self.batch_size}")
        if actions.size == 0:
            return BranchStepResult(
                rewards=np.empty((0,), dtype=np.float32),
                dt_ms=np.empty((0,), dtype=np.float32),
                executed=np.empty((0,), dtype=np.int32),
                terminals=np.empty((0,), dtype=np.uint8),
                observations=[],
            )
        snap = self.snapshot_root() if snapshot is None else snapshot
        binding.vec_restore_all(self.env, snap)
        self.act_buf.fill(-1)
        self.act_buf[: actions.size] = actions
        info = binding.vec_step_validated(self.env)
        dt = np.asarray(info["dt"], dtype=np.float32)[: actions.size]
        executed = np.asarray(info["executed"], dtype=np.int32)[: actions.size]
        rewards = np.asarray(self.rew_buf[: actions.size], dtype=np.float32).copy()
        terminals = np.asarray(self.term_buf[: actions.size], dtype=np.uint8).copy()
        observations = (
            [get_obs_from_buf(self.obs_buf[i], max_trackers=self.max_trackers) for i in range(actions.size)]
            if include_observations
            else []
        )
        return BranchStepResult(rewards=rewards, dt_ms=dt, executed=executed, terminals=terminals, observations=observations)

    def close(self) -> None:
        if getattr(self, "env", None) is not None:
            binding.vec_close(self.env)
            self.env = None
        if getattr(self, "root_eng", None) is not None:
            self.root_eng.close()
            self.root_eng = None


class BatchedMultiRootBranchSimulator:
    """Vectorized one-step simulator for branches from many different roots."""

    def __init__(
        self,
        initial_targets: int,
        seeds: list[int],
        env_cfg: dict,
        batch_size: int,
        max_trackers: int = MAXT,
        window_ms: int = 200,
    ):
        self.initial_targets = int(initial_targets)
        self.seeds = [int(seed) for seed in seeds]
        self.env_cfg = engine_env_cfg(dict(env_cfg))
        self.batch_size = int(batch_size)
        self.max_trackers = int(max_trackers)
        self.window_ms = int(window_ms)
        self.obs_size = GRID_SIZE + self.max_trackers * FEATURES_PER_TRACKER + 1
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")

        self.root_engs = [
            build_env(_DummyPlanner(), self.initial_targets, self.max_trackers, seed, self.window_ms, self.env_cfg)
            for seed in self.seeds
        ]
        for eng, seed in zip(self.root_engs, self.seeds):
            eng.reset(seed=seed)
        self.root_snapshots = [binding.vec_snapshot(eng.env) for eng in self.root_engs]

        self.obs_buf = np.zeros((self.batch_size, self.obs_size), dtype=np.float32)
        self.act_buf = np.zeros((self.batch_size,), dtype=np.int32)
        self.rew_buf = np.zeros((self.batch_size,), dtype=np.float32)
        self.term_buf = np.zeros((self.batch_size,), dtype=np.uint8)
        self.trunc_buf = np.zeros((self.batch_size,), dtype=np.uint8)
        self.dt_buf = np.zeros((self.batch_size,), dtype=np.float32)
        self.executed_buf = np.full((self.batch_size,), -1, dtype=np.int32)
        self.env = binding.vec_init(
            self.obs_buf,
            self.act_buf,
            self.rew_buf,
            self.term_buf,
            self.trunc_buf,
            self.batch_size,
            self.seeds[0],
            initial_targets=self.initial_targets,
            max_trackers=self.max_trackers,
            **self.env_cfg,
        )

    def root_observations(self) -> list[dict]:
        return [get_obs_from_buf(eng.obs_buf, max_trackers=self.max_trackers) for eng in self.root_engs]

    def step_root_actions(
        self,
        root_indices: np.ndarray,
        actions: np.ndarray,
        include_observations: bool = False,
    ) -> BranchStepResult:
        root_indices = np.asarray(root_indices, dtype=np.int32).reshape(-1)
        actions = np.asarray(actions, dtype=np.int32).reshape(-1)
        if actions.size != root_indices.size:
            raise ValueError("actions and root_indices must have the same size")
        if actions.size > self.batch_size:
            raise ValueError(f"actions has {actions.size} items, batch_size={self.batch_size}")
        if actions.size == 0:
            return BranchStepResult(
                rewards=np.empty((0,), dtype=np.float32),
                dt_ms=np.empty((0,), dtype=np.float32),
                executed=np.empty((0,), dtype=np.int32),
                terminals=np.empty((0,), dtype=np.uint8),
                observations=[],
            )
        if np.any(root_indices < 0) or np.any(root_indices >= len(self.root_snapshots)):
            raise ValueError("root index out of range")
        if not hasattr(binding, "vec_restore_many"):
            raise RuntimeError("binding.vec_restore_many is required for multi-root branch simulation")

        count = int(actions.size)
        snapshots = [self.root_snapshots[int(idx)] for idx in root_indices.tolist()]
        binding.vec_restore_many(self.env, snapshots, count)
        self.act_buf[:count] = actions
        binding.vec_step_validated_into(self.env, self.dt_buf, self.executed_buf, count)
        rewards = np.asarray(self.rew_buf[:count], dtype=np.float32).copy()
        terminals = np.asarray(self.term_buf[:count], dtype=np.uint8).copy()
        observations = (
            [get_obs_from_buf(self.obs_buf[i], max_trackers=self.max_trackers) for i in range(count)]
            if include_observations
            else []
        )
        return BranchStepResult(
            rewards=rewards,
            dt_ms=self.dt_buf[:count].copy(),
            executed=self.executed_buf[:count].copy(),
            terminals=terminals,
            observations=observations,
        )

    def close(self) -> None:
        if getattr(self, "env", None) is not None:
            binding.vec_close(self.env)
            self.env = None
        for eng in getattr(self, "root_engs", []):
            eng.close()
        self.root_engs = []
