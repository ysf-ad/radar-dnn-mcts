"""Exact-environment mutual MCTS experiments for radar scheduling.

This is the principled AlphaZero-style scaffold we need before doing more
architecture work.  The existing tree search in ``models/mcts.py`` simulates a
simplified observation-derived state.  This file instead evaluates tree actions
by replaying into the actual C radar environment from the episode seed and
action history.  It is slower, but it makes MCTS targets faithful to the same
environment/reward used at evaluation.

The first implementation uses deterministic reset+replay instead of C-level
snapshot/restore.  That gives exact cloned states without touching the binding;
if this proves useful, the next speed step is exposing snapshot/restore in C.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from final_radar_campaign import MAXT, build_env, get_obs, run_fixed, seedall, summarize_window_df
from load_adaptive_train_eval import OUT, make_env
def configured_env(rate, args):
    """Optional clean-loop environment config.

    The full mutual-alpha loop is not imported at module load time because it
    pulls in additional training-only dependencies. Most evaluation paths use
    make_env(rate); env_mode-specific runs can still import the full helper.
    """
    from mutual_alpha_radar_loop import configured_env as _configured_env

    return _configured_env(rate, args)
from mutual_features import slot_features, tokenize
from mutual_foundation import (
    DEVICE,
    MutualRadarDirectPlanner,
    MutualRadarNet,
    ReplayBuffer,
    SearchTarget,
    action_priors_from_logits,
    train_step,
    train_step_branch_balanced,
    train_step_branch_max,
)
from realistic_reward_retrain import adapter
from repaired_campaign_tools import EDFPlanner, ESTPlanner, SEARCH_DWELL_MS, execute_first_valid_action
from pufferlib.ocean.radarxs import binding


def stable_seed(*parts) -> int:
    text = "|".join(str(p) for p in parts)
    return int(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)


RUN_OUT = Path(os.environ.get("EXACT_MUTUAL_RUN_OUT", str(OUT / "exact_env_mutual")))
RUN_OUT.mkdir(parents=True, exist_ok=True)
_SHARED_ADAPTER = None


def xs_s_search_action(max_trackers: int = MAXT) -> int:
    return int(max_trackers) + 3


def xs_x_search_action(max_trackers: int = MAXT) -> int:
    return int(max_trackers) + 4


def xs_s_track_action(target_action: int, max_trackers: int = MAXT) -> int:
    return int(max_trackers) + 5 + (int(target_action) - 1)


def xs_x_track_action(target_action: int, max_trackers: int = MAXT) -> int:
    return int(max_trackers) + 5 + int(max_trackers) + (int(target_action) - 1)


def xs_decode_action(action: int, max_trackers: int = MAXT) -> Tuple[int, Optional[int]]:
    a = int(action)
    if a == int(max_trackers) + 1 or a == int(max_trackers) + 2:
        return -1, None
    if a == xs_s_search_action(max_trackers):
        return 0, 0
    if a == xs_x_search_action(max_trackers):
        return 0, 1
    s_base = int(max_trackers) + 5
    x_base = int(max_trackers) + 5 + int(max_trackers)
    if s_base <= a < s_base + int(max_trackers):
        return (a - s_base) + 1, 0
    if x_base <= a < x_base + int(max_trackers):
        return (a - x_base) + 1, 1
    return a, None


def xs_action_fractions(history: Sequence[int], max_trackers: int = MAXT) -> Dict[str, float]:
    if not history:
        return {
            "s_search_fraction": 0.0,
            "x_search_fraction": 0.0,
            "s_track_fraction": 0.0,
            "x_track_fraction": 0.0,
        }
    counts = {"s_search": 0, "x_search": 0, "s_track": 0, "x_track": 0}
    for action in history:
        base_action, sensor = xs_decode_action(int(action), max_trackers)
        if sensor == 0 and int(base_action) == 0:
            counts["s_search"] += 1
        elif sensor == 1 and int(base_action) == 0:
            counts["x_search"] += 1
        elif sensor == 0:
            counts["s_track"] += 1
        elif sensor == 1:
            counts["x_track"] += 1
    denom = float(len(history))
    return {
        "s_search_fraction": counts["s_search"] / denom,
        "x_search_fraction": counts["x_search"] / denom,
        "s_track_fraction": counts["s_track"] / denom,
        "x_track_fraction": counts["x_track"] / denom,
    }


def env_cfg_for(rate: float, args) -> Dict[str, float]:
    """Use the same reward/environment knobs as the clean mutual loop."""
    env = configured_env(float(rate), args) if hasattr(args, "env_mode") else make_env(rate)
    env["tracked_target_ms_reward_weight"] = float(getattr(args, "tracked_target_ms_reward_weight", 0.0))
    env["discovered_target_reward"] = float(getattr(args, "discovered_target_reward", 0.0))
    if hasattr(args, "enable_x_band") and args.enable_x_band:
        env["enable_x_band"] = 1
    if hasattr(args, "track_urgency_bonus_weight") and args.track_urgency_bonus_weight >= 0.0:
        env["track_urgency_bonus_weight"] = float(args.track_urgency_bonus_weight)
    return env


def shaped_step_reward(
    base_reward: float,
    dt_ms: float,
    obs_before: Dict[str, np.ndarray],
    obs_after: Dict[str, np.ndarray],
    env_cfg: Dict[str, float],
) -> float:
    """Optional dense reward terms; disabled by default for backwards compatibility."""
    reward = float(base_reward)
    tracked_ms_weight = float(env_cfg.get("tracked_target_ms_reward_weight", 0.0))
    if tracked_ms_weight != 0.0 and dt_ms > 0.0:
        active_after = np.asarray(obs_after["active_mask"]).astype(bool)
        tracked_after = active_after & (np.asarray(obs_after["t_deadline"], dtype=np.float32) >= 0.0)
        reward += tracked_ms_weight * (float(dt_ms) / 1000.0) * float(np.sum(tracked_after))
    discovered_reward = float(env_cfg.get("discovered_target_reward", 0.0))
    if discovered_reward != 0.0:
        before_n = float(np.sum(np.asarray(obs_before["active_mask"]).astype(bool)))
        after_n = float(np.sum(np.asarray(obs_after["active_mask"]).astype(bool)))
        reward += discovered_reward * max(0.0, after_n - before_n)
    return reward


def engine_env_cfg(env_cfg: Dict[str, float]) -> Dict[str, float]:
    """Drop Python-side reward-shaping keys before constructing the C engine."""
    out = dict(env_cfg)
    out.pop("tracked_target_ms_reward_weight", None)
    out.pop("discovered_target_reward", None)
    return out


def attach_env_obs(
    obs: Dict[str, np.ndarray],
    env_cfg: Dict[str, float],
    use_arrival_feature: bool = False,
    use_grid_feature: bool = False,
) -> Dict[str, np.ndarray]:
    out = dict(obs)
    out["arrival_rate"] = float(env_cfg.get("arrival_rate", env_cfg.get("poisson_rate_per_second", 0.0)))
    out["use_arrival_feature"] = 1.0 if bool(use_arrival_feature) else 0.0
    out["use_grid_feature"] = 1.0 if bool(use_grid_feature) else 0.0
    return out


def get_shared_adapter():
    global _SHARED_ADAPTER
    if _SHARED_ADAPTER is None:
        _SHARED_ADAPTER = adapter()
    return _SHARED_ADAPTER


class _DummyPlanner:
    def plan(self, obs, budget_ms=200):
        return [0]


@dataclass
class ReplayState:
    obs: Dict[str, np.ndarray]
    debt_ms: float
    reward: float
    dt_ms: float
    terminal: bool


class ExactReplaySimulator:
    """Clones radar state by reset+replay of the actual C environment."""

    def __init__(self, initial_targets: int, seed: int, env_cfg: Dict[str, float], max_trackers: int = MAXT):
        self.initial_targets = int(initial_targets)
        self.seed = int(seed)
        self.env_cfg = dict(env_cfg)
        self.max_trackers = int(max_trackers)

    def _new_env(self):
        eng = build_env(_DummyPlanner(), self.initial_targets, self.max_trackers, self.seed, 200, engine_env_cfg(self.env_cfg))
        eng.reset(seed=self.seed)
        return eng

    @staticmethod
    def _is_search_action(action: int) -> bool:
        return xs_decode_action(int(action), MAXT)[0] == 0

    def replay(self, actions: Sequence[int]) -> ReplayState:
        eng = self._new_env()
        debt = 0.0
        total_reward = 0.0
        total_dt = 0.0
        terminal = False
        try:
            for a in actions:
                obs_before = attach_env_obs(get_obs(eng, debt), self.env_cfg)
                reward, dt, executed = execute_first_valid_action(eng, [int(a)], 1e9)
                if executed is None or dt <= 0.0:
                    terminal = True
                    break
                if self._is_search_action(int(executed)):
                    debt = 0.0
                else:
                    debt += float(dt)
                obs_after = attach_env_obs(get_obs(eng, debt), self.env_cfg)
                total_reward += shaped_step_reward(float(reward), float(dt), obs_before, obs_after, self.env_cfg)
                total_dt += float(dt)
                terminal = bool(eng.term_buf[0])
                if terminal:
                    break
            obs = attach_env_obs(get_obs(eng, debt), self.env_cfg)
            return ReplayState(obs=obs, debt_ms=debt, reward=total_reward, dt_ms=total_dt, terminal=terminal)
        finally:
            eng.close()

    def evaluate_plan_sequence(self, actions: Sequence[int], budget_ms: float) -> Tuple[Tuple[int, ...], float, float]:
        """Execute a proposed plan from the root, skipping stale invalid actions.

        This matches the fixed-window heuristic baseline: a planner may emit a
        long ordered list, and actions that are no longer valid by the time they
        are reached are skipped rather than terminating the window.
        """
        eng = self._new_env()
        executed: List[int] = []
        total_reward = 0.0
        elapsed = 0.0
        try:
            for a in actions:
                if elapsed >= float(budget_ms) or bool(eng.term_buf[0]):
                    break
                reward, dt, actual = execute_first_valid_action(eng, [int(a)], max(0.0, float(budget_ms) - elapsed))
                if actual is None or dt <= 0.0:
                    continue
                executed.append(int(actual))
                total_reward += float(reward)
                elapsed += float(dt)
            return tuple(executed), float(total_reward), float(elapsed)
        finally:
            eng.close()

    def step_from(self, actions: Sequence[int], action: int) -> Tuple[ReplayState, float, float, Optional[int]]:
        before = self.replay(actions)
        after = self.replay([*actions, int(action)])
        return after, float(after.reward - before.reward), float(after.dt_ms - before.dt_ms), int(action)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - float(np.max(z))
    p = np.exp(np.clip(z, -60.0, 60.0))
    return (p / max(float(np.sum(p)), 1e-12)).astype(np.float32)


def _seq_halving_considered_visits(max_num_considered_actions: int, num_simulations: int) -> Tuple[int, ...]:
    """Python port of MCTX seq_halving.get_sequence_of_considered_visits."""
    max_num_considered_actions = int(max_num_considered_actions)
    num_simulations = int(num_simulations)
    if max_num_considered_actions <= 1:
        return tuple(range(num_simulations))
    log2max = int(math.ceil(math.log2(max_num_considered_actions)))
    sequence: List[int] = []
    visits = [0] * max_num_considered_actions
    num_considered = max_num_considered_actions
    while len(sequence) < num_simulations:
        num_extra_visits = max(1, int(num_simulations / (log2max * num_considered)))
        for _ in range(num_extra_visits):
            sequence.extend(visits[:num_considered])
            for i in range(num_considered):
                visits[i] += 1
        num_considered = max(2, num_considered // 2)
    return tuple(sequence[:num_simulations])


class SnapshotSimulator:
    """Fast exact clone using C snapshot/restore of one live environment."""

    def __init__(
        self,
        eng,
        root_debt_ms: float,
        env_cfg: Optional[Dict[str, float]] = None,
        use_arrival_feature: bool = False,
        use_grid_feature: bool = False,
        seed: int = 0,
    ):
        self.eng = eng
        self.root_debt_ms = float(root_debt_ms)
        self.env_cfg = dict(env_cfg or {})
        self.use_arrival_feature = bool(use_arrival_feature)
        self.use_grid_feature = bool(use_grid_feature)
        self.seed = int(seed)
        if not hasattr(binding, "vec_snapshot") or not hasattr(binding, "vec_restore"):
            raise RuntimeError("radarxs binding does not expose vec_snapshot/vec_restore; rebuild binding first")
        self.root = binding.vec_snapshot(self.eng.env)
        root_obs = attach_env_obs(get_obs(self.eng, self.root_debt_ms), self.env_cfg, self.use_arrival_feature, self.use_grid_feature)
        self._cache: Dict[Tuple[int, ...], ReplayState] = {
            (): ReplayState(obs=root_obs, debt_ms=self.root_debt_ms, reward=0.0, dt_ms=0.0, terminal=bool(self.eng.term_buf[0]))
        }
        self._snap_cache: Dict[Tuple[int, ...], object] = {(): self.root}

    @staticmethod
    def _is_search_action(action: int) -> bool:
        return xs_decode_action(int(action), MAXT)[0] == 0

    def replay(self, actions: Sequence[int]) -> ReplayState:
        key = tuple(int(a) for a in actions)
        if key in self._cache:
            return self._cache[key]
        prefix_len = 0
        for n in range(len(key), -1, -1):
            if key[:n] in self._snap_cache:
                prefix_len = n
                break
        prefix = key[:prefix_len]
        binding.vec_restore(self.eng.env, self._snap_cache[prefix])
        # Defensive copy: some binding snapshot objects behave like mutable
        # handles after restore/step. Preserve the prefix snapshot before
        # advancing the live environment.
        self._snap_cache[prefix] = binding.vec_snapshot(self.eng.env)
        prefix_state = self._cache[prefix]
        debt = float(prefix_state.debt_ms)
        total_reward = float(prefix_state.reward)
        total_dt = float(prefix_state.dt_ms)
        terminal = bool(prefix_state.terminal)
        cur_prefix = list(prefix)
        for a in key[prefix_len:]:
            if terminal:
                break
            obs_before = attach_env_obs(get_obs(self.eng, debt), self.env_cfg, self.use_arrival_feature, self.use_grid_feature)
            reward, dt, executed = execute_first_valid_action(self.eng, [int(a)], 1e9)
            if executed is None or dt <= 0.0:
                terminal = True
                break
            debt = 0.0 if self._is_search_action(int(executed)) else debt + float(dt)
            obs_after = attach_env_obs(get_obs(self.eng, debt), self.env_cfg, self.use_arrival_feature, self.use_grid_feature)
            total_reward += shaped_step_reward(float(reward), float(dt), obs_before, obs_after, self.env_cfg)
            total_dt += float(dt)
            terminal = bool(self.eng.term_buf[0])
            cur_prefix.append(int(a))
            cur_key = tuple(cur_prefix)
            obs = obs_after
            out = ReplayState(obs=obs, debt_ms=debt, reward=total_reward, dt_ms=total_dt, terminal=terminal)
            self._cache[cur_key] = out
            self._snap_cache[cur_key] = binding.vec_snapshot(self.eng.env)
            if terminal:
                break
        if key in self._cache:
            return self._cache[key]
        obs = attach_env_obs(get_obs(self.eng, debt), self.env_cfg, self.use_arrival_feature, self.use_grid_feature)
        out = ReplayState(obs=obs, debt_ms=debt, reward=total_reward, dt_ms=total_dt, terminal=terminal)
        self._cache[key] = out
        return out

    def commit(self, action: int) -> Tuple[float, float, float, Optional[int]]:
        binding.vec_restore(self.eng.env, self.root)
        obs_before = attach_env_obs(get_obs(self.eng, self.root_debt_ms), self.env_cfg, self.use_arrival_feature, self.use_grid_feature)
        reward, dt, executed = execute_first_valid_action(self.eng, [int(action)], 1e9)
        if executed is None or dt <= 0.0:
            return 0.0, 0.0, self.root_debt_ms, None
        debt = 0.0 if self._is_search_action(int(executed)) else self.root_debt_ms + float(dt)
        obs_after = attach_env_obs(get_obs(self.eng, debt), self.env_cfg, self.use_arrival_feature, self.use_grid_feature)
        return shaped_step_reward(float(reward), float(dt), obs_before, obs_after, self.env_cfg), float(dt), debt, int(executed)

    def commit_first_valid(self, actions: Sequence[int], budget_ms: float) -> Tuple[float, float, float, Optional[int]]:
        binding.vec_restore(self.eng.env, self.root)
        obs_before = attach_env_obs(get_obs(self.eng, self.root_debt_ms), self.env_cfg, self.use_arrival_feature, self.use_grid_feature)
        reward, dt, executed = execute_first_valid_action(self.eng, [int(a) for a in actions], max(0.0, float(budget_ms)))
        if executed is None or dt <= 0.0:
            return 0.0, 0.0, self.root_debt_ms, None
        debt = 0.0 if self._is_search_action(int(executed)) else self.root_debt_ms + float(dt)
        obs_after = attach_env_obs(get_obs(self.eng, debt), self.env_cfg, self.use_arrival_feature, self.use_grid_feature)
        return shaped_step_reward(float(reward), float(dt), obs_before, obs_after, self.env_cfg), float(dt), debt, int(executed)

    def commit_sequence(self, actions: Sequence[int], budget_ms: float = 200.0) -> Tuple[List[Tuple[float, float, int]], float]:
        """Restore the root snapshot once, then execute a planned window path."""
        binding.vec_restore(self.eng.env, self.root)
        debt = self.root_debt_ms
        elapsed = 0.0
        out: List[Tuple[float, float, int]] = []
        for action in actions:
            if elapsed >= float(budget_ms) or bool(self.eng.term_buf[0]):
                break
            obs_before = attach_env_obs(get_obs(self.eng, debt), self.env_cfg, self.use_arrival_feature, self.use_grid_feature)
            reward, dt, executed = execute_first_valid_action(
                self.eng, [int(action)], max(0.0, float(budget_ms) - elapsed)
            )
            if executed is None or dt <= 0.0:
                continue
            debt = 0.0 if self._is_search_action(int(executed)) else debt + float(dt)
            obs_after = attach_env_obs(get_obs(self.eng, debt), self.env_cfg, self.use_arrival_feature, self.use_grid_feature)
            out.append((shaped_step_reward(float(reward), float(dt), obs_before, obs_after, self.env_cfg), float(dt), int(executed)))
            elapsed += float(dt)
        return out, debt

    def evaluate_plan_sequence(self, actions: Sequence[int], budget_ms: float) -> Tuple[Tuple[int, ...], float, float]:
        """Score a root-relative proposed plan, skipping stale invalid actions."""
        binding.vec_restore(self.eng.env, self.root)
        root_copy = binding.vec_snapshot(self.eng.env)
        executed_actions: List[int] = []
        total_reward = 0.0
        elapsed = 0.0
        debt = self.root_debt_ms
        try:
            for action in actions:
                if elapsed >= float(budget_ms) or bool(self.eng.term_buf[0]):
                    break
                obs_before = attach_env_obs(get_obs(self.eng, debt), self.env_cfg, self.use_arrival_feature, self.use_grid_feature)
                reward, dt, executed = execute_first_valid_action(self.eng, [int(action)], max(0.0, float(budget_ms) - elapsed))
                if executed is None or dt <= 0.0:
                    continue
                executed_actions.append(int(executed))
                debt = 0.0 if self._is_search_action(int(executed)) else debt + float(dt)
                obs_after = attach_env_obs(get_obs(self.eng, debt), self.env_cfg, self.use_arrival_feature, self.use_grid_feature)
                total_reward += shaped_step_reward(float(reward), float(dt), obs_before, obs_after, self.env_cfg)
                elapsed += float(dt)
            return tuple(executed_actions), float(total_reward), float(elapsed)
        finally:
            binding.vec_restore(self.eng.env, root_copy)
            self.root = binding.vec_snapshot(self.eng.env)
            self._snap_cache[()] = self.root
            self._cache[()] = ReplayState(
                obs=attach_env_obs(get_obs(self.eng, self.root_debt_ms), self.env_cfg, self.use_arrival_feature, self.use_grid_feature),
                debt_ms=self.root_debt_ms,
                reward=0.0,
                dt_ms=0.0,
                terminal=bool(self.eng.term_buf[0]),
            )


@dataclass
class ExactNode:
    seq: Tuple[int, ...]
    prior: float = 1.0
    parent: Optional["ExactNode"] = None
    action: int = -1
    prior_logit: float = 0.0
    root_gumbel: float = 0.0
    raw_value: float = 0.0
    edge_reward: float = 0.0
    edge_dt_ms: float = 0.0
    edge_evaluated: bool = False
    visits: int = 0
    total_value: float = 0.0
    children: List["ExactNode"] = field(default_factory=list)
    expanded: bool = False

    @property
    def mean_value(self) -> float:
        return self.total_value / max(1, self.visits)


class ExactEnvMCTS:
    """PUCT over actual radar env dynamics using reset+replay cloning."""

    def __init__(
        self,
        model: MutualRadarNet,
        sim: ExactReplaySimulator,
        prefix_actions: Sequence[int],
        rollouts: int = 16,
        c_puct: float = 1.25,
        expand_top_k: int = 12,
        horizon_windows: int = 10,
        rollout_policy: str = "model",
        prior_mode: str = "factorized",
        q_scale: float = 100.0,
        epsilon: float = 0.10,
        policy_target: str = "visits",
        policy_tau: float = 1.0,
        branch_rollout_threshold: float = 0.65,
        search_alg: str = "puct",
        max_num_considered_actions: int = 16,
        gumbel_scale: float = 0.0,
        mctx_value_scale: float = 0.1,
        mctx_maxvisit_init: float = 50.0,
        eager_edge_depth: int = 1,
        prior_uniform_mix: float = 0.0,
        root_dirichlet_alpha: float = 0.0,
        root_dirichlet_frac: float = 0.0,
        rollout_est_prob: float = 0.5,
        mask_selected: bool = True,
        stateless_tree_context: bool = False,
        head_mode: str = "p",
        q_utility_weight: float = 0.0,
        q_utility_normalize: bool = False,
        puct_q_transform: str = "raw",
        leaf_value_mix: float = 1.0,
        seed_rollout_policies: Sequence[str] = (),
        fast_zero_rollout: bool = False,
        skip_default_rollout_seed: bool = False,
        complete_root_q_with_value: bool = False,
        visit_unvisited_first: bool = False,
        duration_normalize_q: bool = False,
        prior_q_beta: float = 0.0,
        prior_search_bias: float = 0.0,
        adaptive_search_bias: float = 0.0,
        adaptive_search_target_load: float = 0.75,
        forbidden_actions: Sequence[int] = (),
        sensor_action_mode: str = "implicit",
        disable_x_search: bool = False,
        canonical_search_only: bool = False,
    ):
        self.model = model.eval()
        self.sim = sim
        self.prefix = tuple(int(a) for a in prefix_actions)
        self.rollouts = int(rollouts)
        self.c_puct = float(c_puct)
        self.expand_top_k = int(expand_top_k)
        self.horizon_ms = float(horizon_windows) * 200.0
        self.rollout_policy = str(rollout_policy)
        self.prior_mode = str(prior_mode)
        self.q_scale = float(max(1e-6, q_scale))
        self.epsilon = float(epsilon)
        self.policy_target = str(policy_target)
        self.policy_tau = float(max(1e-6, policy_tau))
        self.branch_rollout_threshold = float(branch_rollout_threshold)
        self.search_alg = str(search_alg)
        self.max_num_considered_actions = int(max_num_considered_actions)
        self.gumbel_scale = float(gumbel_scale)
        self.mctx_value_scale = float(mctx_value_scale)
        self.mctx_maxvisit_init = float(mctx_maxvisit_init)
        self.eager_edge_depth = int(eager_edge_depth)
        self.prior_uniform_mix = float(np.clip(prior_uniform_mix, 0.0, 1.0))
        self.root_dirichlet_alpha = float(max(0.0, root_dirichlet_alpha))
        self.root_dirichlet_frac = float(np.clip(root_dirichlet_frac, 0.0, 1.0))
        self.rollout_est_prob = float(np.clip(rollout_est_prob, 0.0, 1.0))
        self.mask_selected = bool(mask_selected)
        self.stateless_tree_context = bool(stateless_tree_context)
        self.head_mode = str(head_mode).lower()
        self.use_q_head = "q" in self.head_mode
        self.use_value_head = "v" in self.head_mode
        self.q_utility_weight = float(q_utility_weight)
        self.q_utility_normalize = bool(q_utility_normalize)
        self.puct_q_transform = str(puct_q_transform)
        self.leaf_value_mix = float(np.clip(leaf_value_mix, 0.0, 1.0))
        self.seed_rollout_policies = tuple(str(p).strip() for p in seed_rollout_policies if str(p).strip())
        self.fast_zero_rollout = bool(fast_zero_rollout)
        self.skip_default_rollout_seed = bool(skip_default_rollout_seed)
        self.complete_root_q_with_value = bool(complete_root_q_with_value)
        self.visit_unvisited_first = bool(visit_unvisited_first)
        self.duration_normalize_q = bool(duration_normalize_q)
        self.prior_q_beta = float(prior_q_beta)
        self.prior_search_bias = float(prior_search_bias)
        self.adaptive_search_bias = float(adaptive_search_bias)
        self.adaptive_search_target_load = float(np.clip(adaptive_search_target_load, 0.0, 1.0))
        self.forbidden_actions = {int(a) for a in forbidden_actions if int(a) != 0}
        self.sensor_action_mode = str(sensor_action_mode)
        self.disable_x_search = bool(disable_x_search)
        self.canonical_search_only = bool(canonical_search_only)
        self._considered_visits = _seq_halving_considered_visits(
            max(1, self.max_num_considered_actions), max(1, self.rollouts)
        )
        self.adapt = get_shared_adapter()
        self._state_cache: Dict[Tuple[int, ...], ReplayState] = {}
        self._net_cache = {}
        self._branch_prob_cache: Dict[Tuple[int, ...], float] = {}
        self.best_seq: Tuple[int, ...] = ()
        self.best_seq_value: float = -float("inf")
        rng_seed = stable_seed(getattr(self.sim, "seed", 0), self.prefix, self.rollouts, self.expand_top_k, self.horizon_ms, "rollout")
        self.rng = random.Random(int(rng_seed))

    def _effective_search_bias(self, obs: Dict[str, np.ndarray]) -> float:
        bias = float(self.prior_search_bias)
        if self.adaptive_search_bias != 0.0:
            active = np.asarray(obs.get("active_mask", []), dtype=bool)
            active_frac = float(np.mean(active)) if active.size else 0.0
            bias += float(self.adaptive_search_bias) * (self.adaptive_search_target_load - active_frac)
        return bias

    def _search_logit_offset(self, type_q: np.ndarray, obs: Dict[str, np.ndarray]) -> float:
        q_shift = float(type_q[1] - type_q[0]) if len(type_q) >= 2 else 0.0
        return float(self._effective_search_bias(obs) + self.prior_q_beta * q_shift)

    def _calibrate_search_prior(self, priors: np.ndarray, type_q: np.ndarray, obs: Dict[str, np.ndarray]) -> np.ndarray:
        effective_bias = self._effective_search_bias(obs)
        if self.prior_q_beta == 0.0 and effective_bias == 0.0:
            return priors
        current_search = float(np.clip(priors[0], 1e-6, 1.0 - 1e-6))
        logit = math.log(current_search / (1.0 - current_search))
        q_shift = float(type_q[1] - type_q[0])
        p_search = float(1.0 / (1.0 + math.exp(-np.clip(logit + self.prior_q_beta * q_shift + effective_bias, -30.0, 30.0))))
        track_sum = float(np.sum(priors[1:]))
        if track_sum > 1e-12:
            priors[1:] = priors[1:] * ((1.0 - p_search) / track_sum)
        priors[0] = p_search
        return priors

    def _raw_child_q(self, child: ExactNode) -> float:
        q = float(child.edge_reward + child.mean_value)
        if self.duration_normalize_q and child.edge_evaluated and child.edge_dt_ms > 0.0:
            q /= max(0.05, float(child.edge_dt_ms) / 200.0)
        return q

    def _puct_q_lookup(self, children: Sequence[ExactNode]) -> Dict[int, float]:
        if self.puct_q_transform in {"mctx", "completed", "completed_mix"}:
            parent = children[0].parent if children else None
            if parent is not None:
                return self._completed_qvalues(parent)
        raw = np.asarray([self._raw_child_q(c) for c in children], dtype=np.float64)
        mode = self.puct_q_transform
        if mode == "scale":
            vals = raw / max(1e-6, float(self.q_scale))
        elif mode == "minmax":
            q_min = float(np.min(raw)) if raw.size else 0.0
            q_max = float(np.max(raw)) if raw.size else 0.0
            vals = (raw - q_min) / (q_max - q_min) if q_max > q_min + 1e-8 else np.zeros_like(raw)
        else:
            vals = raw
        return {int(c.action): float(v) for c, v in zip(children, vals)}

    @property
    def device(self):
        return next(self.model.parameters()).device

    def state(self, seq: Sequence[int]) -> ReplayState:
        key = tuple(int(a) for a in seq)
        if key not in self._state_cache:
            self._state_cache[key] = self.sim.replay([*self.prefix, *key])
        return self._state_cache[key]

    def _net(self, obs: Dict[str, np.ndarray], seq: Sequence[int]):
        key = tuple(int(a) for a in seq)
        if key in self._net_cache:
            return self._net_cache[key]
        elapsed = 0.0
        search_count = 0
        track_count = 0
        selected = set()
        last = -1
        dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
        feature_seq = () if self.stateless_tree_context else tuple(seq)
        for a in feature_seq:
            base_a, _ = xs_decode_action(int(a), MAXT)
            last = int(base_a)
            if int(base_a) < 0:
                elapsed += SEARCH_DWELL_MS
            elif int(base_a) == 0:
                elapsed += SEARCH_DWELL_MS
                search_count += 1
            else:
                if self.mask_selected:
                    selected.add(int(base_a))
                idx = int(base_a) - 1
                elapsed += max(1.0, float(dwell[idx]) if 0 <= idx < len(dwell) else SEARCH_DWELL_MS)
                track_count += 1
        x = tokenize(self.adapt, obs, selected=selected, search_count=search_count)
        slot = slot_features(obs, elapsed, search_count, track_count, last, 200.0)
        with torch.inference_mode():
            tx = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            ts = torch.from_numpy(slot).float().unsqueeze(0).to(self.device)
            if self.prior_mode == "true_physical_flat" and hasattr(self.model, "forward_physical_flat"):
                phys_logits, phys_q, value = self.model.forward_physical_flat(tx, ts)
                full_len = xs_x_track_action(MAXT, MAXT) + 1
                flat_logits = np.full((full_len,), -1e9, dtype=np.float64)
                flat_q = np.zeros((full_len,), dtype=np.float32)
                pl = phys_logits[0].detach().cpu().numpy()
                pq = phys_q[0].detach().cpu().numpy()
                for base_a in range(MAXT + 1):
                    actions = [xs_s_search_action(MAXT), xs_x_search_action(MAXT)] if base_a == 0 else [xs_s_track_action(base_a, MAXT), xs_x_track_action(base_a, MAXT)]
                    for sensor_idx, explicit_a in enumerate(actions):
                        flat_logits[explicit_a] = float(pl[base_a, sensor_idx])
                        flat_q[explicit_a] = float(pq[base_a, sensor_idx]) * self.q_scale
                z = flat_logits - float(np.max(flat_logits))
                p = np.exp(np.clip(z, -60.0, 60.0))
                priors = (p / max(float(np.sum(p)), 1e-12)).astype(np.float32)
                q = flat_q
                out = (priors, float(value[0].detach().cpu()) * self.q_scale, q, x, slot)
                self._net_cache[key] = out
                return out
            elif self.sensor_action_mode == "explicit_head" and hasattr(self.model, "forward_with_sensor"):
                tl, tr, value, type_q, track_q, sensor_logits, sensor_q = self.model.forward_with_sensor(tx, ts)
            else:
                tl, tr, value, type_q, track_q = self.model(tx, ts)
                sensor_logits = sensor_q = None
        logical_prior_mode = "flat" if self.prior_mode == "physical_flat" else self.prior_mode
        self._branch_prob_cache[key] = float(torch.sigmoid(tl[0]).detach().cpu().item())
        priors = action_priors_from_logits(tl[0], tr[0], logical_prior_mode)
        q = np.zeros((MAXT + 1,), dtype=np.float32)
        tq = type_q[0].detach().cpu().numpy()
        tq_track = track_q[0].detach().cpu().numpy()
        priors = self._calibrate_search_prior(priors, tq, obs)
        q[0] = float(tq[1] * self.q_scale)
        q[1:] = float(tq[0] * self.q_scale) + tq_track[1:] * self.q_scale
        if self.sensor_action_mode == "explicit_head" and sensor_logits is not None and sensor_q is not None:
            full_len = xs_x_track_action(MAXT, MAXT) + 1
            priors_full = np.zeros((full_len,), dtype=np.float32)
            q_full = np.zeros((full_len,), dtype=np.float32)
            slog = sensor_logits[0].detach().cpu()
            sq = sensor_q[0].detach().cpu().numpy()
            if self.prior_mode == "physical_flat":
                phys_logits = np.full((full_len,), -1e9, dtype=np.float64)
                logical_logits = np.full((MAXT + 1,), -1e9, dtype=np.float64)
                logical_logits[0] = float(tl[0].detach().cpu()) + self._search_logit_offset(tq, obs)
                tr_np = tr[0].detach().cpu().numpy()
                finite_np = np.isfinite(tr_np) & (tr_np > -1e8)
                logical_logits[1:][finite_np[1:]] = tr_np[1:][finite_np[1:]]
                for base_a in range(0, MAXT + 1):
                    if logical_logits[base_a] < -1e8:
                        continue
                    actions = [xs_s_search_action(MAXT), xs_x_search_action(MAXT)] if base_a == 0 else [xs_s_track_action(base_a, MAXT), xs_x_track_action(base_a, MAXT)]
                    sensor_logits_np = slog[base_a].numpy().astype(np.float64)
                    for sensor_idx, explicit_a in enumerate(actions):
                        phys_logits[explicit_a] = logical_logits[base_a] + sensor_logits_np[sensor_idx]
                z = phys_logits - float(np.max(phys_logits))
                p = np.exp(np.clip(z, -60.0, 60.0))
                priors_full = (p / max(float(np.sum(p)), 1e-12)).astype(np.float32)
                for base_a in range(0, MAXT + 1):
                    actions = [xs_s_search_action(MAXT), xs_x_search_action(MAXT)] if base_a == 0 else [xs_s_track_action(base_a, MAXT), xs_x_track_action(base_a, MAXT)]
                    for sensor_idx, explicit_a in enumerate(actions):
                        q_full[explicit_a] = float(q[base_a]) + float(sq[base_a, sensor_idx]) * self.q_scale
                priors = priors_full
                q = q_full
                out = (priors, float(value[0].detach().cpu()) * self.q_scale, q, x, slot)
                self._net_cache[key] = out
                return out
            for base_a in range(0, MAXT + 1):
                sensor_p = torch.softmax(slog[base_a], dim=0).numpy().astype(np.float32)
                if base_a == 0:
                    actions = [xs_s_search_action(MAXT), xs_x_search_action(MAXT)]
                else:
                    actions = [xs_s_track_action(base_a, MAXT), xs_x_track_action(base_a, MAXT)]
                for sensor_idx, explicit_a in enumerate(actions):
                    priors_full[explicit_a] = float(priors[base_a]) * float(sensor_p[sensor_idx])
                    q_full[explicit_a] = float(q[base_a]) + float(sq[base_a, sensor_idx]) * self.q_scale
            priors = priors_full
            q = q_full
        out = (priors, float(value[0].detach().cpu()) * self.q_scale, q, x, slot)
        self._net_cache[key] = out
        return out

    def _net_many(self, states_and_seqs: Sequence[Tuple[ReplayState, Sequence[int]]]):
        """Batched equivalent of ``_net`` for sibling leaf evaluation.

        Low-rollout PUCT tends to spend all visits on one high-prior child.
        For in-window planning, evaluating sibling child states in one
        transformer batch is both faster on GPU and less myopic.
        """
        pending = []
        results = []
        for st, seq in states_and_seqs:
            key = tuple(int(a) for a in seq)
            cached = self._net_cache.get(key)
            if cached is None:
                pending.append((key, st.obs, tuple(int(a) for a in seq)))
                results.append(None)
            else:
                results.append(cached)
        if pending:
            xs = []
            slots = []
            meta = []
            for key, obs, seq in pending:
                elapsed = 0.0
                search_count = 0
                track_count = 0
                selected = set()
                last = -1
                dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
                feature_seq = () if self.stateless_tree_context else tuple(seq)
                for a in feature_seq:
                    base_a, _ = xs_decode_action(int(a), MAXT)
                    last = int(base_a)
                    if int(base_a) < 0:
                        elapsed += SEARCH_DWELL_MS
                    elif int(base_a) == 0:
                        elapsed += SEARCH_DWELL_MS
                        search_count += 1
                    else:
                        if self.mask_selected:
                            selected.add(int(base_a))
                        idx = int(base_a) - 1
                        elapsed += max(1.0, float(dwell[idx]) if 0 <= idx < len(dwell) else SEARCH_DWELL_MS)
                        track_count += 1
                xs.append(tokenize(self.adapt, obs, selected=selected, search_count=search_count))
                slots.append(slot_features(obs, elapsed, search_count, track_count, last, 200.0))
                meta.append(key)
            with torch.inference_mode():
                tx = torch.from_numpy(np.stack(xs).astype(np.float32)).to(self.device)
                ts = torch.from_numpy(np.stack(slots).astype(np.float32)).to(self.device)
                if self.prior_mode == "true_physical_flat" and hasattr(self.model, "forward_physical_flat"):
                    phys_logits, phys_q, value = self.model.forward_physical_flat(tx, ts)
                    tl = tr = type_q = track_q = sensor_logits = sensor_q = None
                elif self.sensor_action_mode == "explicit_head" and hasattr(self.model, "forward_with_sensor"):
                    tl, tr, value, type_q, track_q, sensor_logits, sensor_q = self.model.forward_with_sensor(tx, ts)
                else:
                    tl, tr, value, type_q, track_q = self.model(tx, ts)
                    sensor_logits = sensor_q = None
            for i, key in enumerate(meta):
                if self.prior_mode == "true_physical_flat":
                    full_len = xs_x_track_action(MAXT, MAXT) + 1
                    flat_logits = np.full((full_len,), -1e9, dtype=np.float64)
                    flat_q = np.zeros((full_len,), dtype=np.float32)
                    pl = phys_logits[i].detach().cpu().numpy()
                    pq = phys_q[i].detach().cpu().numpy()
                    for base_a in range(MAXT + 1):
                        actions = [xs_s_search_action(MAXT), xs_x_search_action(MAXT)] if base_a == 0 else [xs_s_track_action(base_a, MAXT), xs_x_track_action(base_a, MAXT)]
                        for sensor_idx, explicit_a in enumerate(actions):
                            flat_logits[explicit_a] = float(pl[base_a, sensor_idx])
                            flat_q[explicit_a] = float(pq[base_a, sensor_idx]) * self.q_scale
                    z = flat_logits - float(np.max(flat_logits))
                    p = np.exp(np.clip(z, -60.0, 60.0))
                    priors = (p / max(float(np.sum(p)), 1e-12)).astype(np.float32)
                    q = flat_q
                    out = (priors, float(value[i].detach().cpu()) * self.q_scale, q, xs[i], slots[i])
                    self._net_cache[key] = out
                    continue
                logical_prior_mode = "flat" if self.prior_mode == "physical_flat" else self.prior_mode
                self._branch_prob_cache[key] = float(torch.sigmoid(tl[i]).detach().cpu().item())
                priors = action_priors_from_logits(tl[i], tr[i], logical_prior_mode)
                q = np.zeros((MAXT + 1,), dtype=np.float32)
                tq = type_q[i].detach().cpu().numpy()
                tq_track = track_q[i].detach().cpu().numpy()
                priors = self._calibrate_search_prior(priors, tq, states_and_seqs[i][0].obs)
                q[0] = float(tq[1] * self.q_scale)
                q[1:] = float(tq[0] * self.q_scale) + tq_track[1:] * self.q_scale
                if self.sensor_action_mode == "explicit_head" and sensor_logits is not None and sensor_q is not None:
                    full_len = xs_x_track_action(MAXT, MAXT) + 1
                    priors_full = np.zeros((full_len,), dtype=np.float32)
                    q_full = np.zeros((full_len,), dtype=np.float32)
                    slog = sensor_logits[i].detach().cpu()
                    sq = sensor_q[i].detach().cpu().numpy()
                    if self.prior_mode == "physical_flat":
                        phys_logits = np.full((full_len,), -1e9, dtype=np.float64)
                        logical_logits = np.full((MAXT + 1,), -1e9, dtype=np.float64)
                        logical_logits[0] = float(tl[i].detach().cpu()) + self._search_logit_offset(tq, states_and_seqs[i][0].obs)
                        tr_np = tr[i].detach().cpu().numpy()
                        finite_np = np.isfinite(tr_np) & (tr_np > -1e8)
                        logical_logits[1:][finite_np[1:]] = tr_np[1:][finite_np[1:]]
                        for base_a in range(0, MAXT + 1):
                            if logical_logits[base_a] < -1e8:
                                continue
                            actions = [xs_s_search_action(MAXT), xs_x_search_action(MAXT)] if base_a == 0 else [xs_s_track_action(base_a, MAXT), xs_x_track_action(base_a, MAXT)]
                            sensor_logits_np = slog[base_a].numpy().astype(np.float64)
                            for sensor_idx, explicit_a in enumerate(actions):
                                phys_logits[explicit_a] = logical_logits[base_a] + sensor_logits_np[sensor_idx]
                        z = phys_logits - float(np.max(phys_logits))
                        p = np.exp(np.clip(z, -60.0, 60.0))
                        priors_full = (p / max(float(np.sum(p)), 1e-12)).astype(np.float32)
                        for base_a in range(0, MAXT + 1):
                            actions = [xs_s_search_action(MAXT), xs_x_search_action(MAXT)] if base_a == 0 else [xs_s_track_action(base_a, MAXT), xs_x_track_action(base_a, MAXT)]
                            for sensor_idx, explicit_a in enumerate(actions):
                                q_full[explicit_a] = float(q[base_a]) + float(sq[base_a, sensor_idx]) * self.q_scale
                        priors = priors_full
                        q = q_full
                        out = (priors, float(value[i].detach().cpu()) * self.q_scale, q, xs[i], slots[i])
                        self._net_cache[key] = out
                        continue
                    for base_a in range(0, MAXT + 1):
                        sensor_p = torch.softmax(slog[base_a], dim=0).numpy().astype(np.float32)
                        actions = [xs_s_search_action(MAXT), xs_x_search_action(MAXT)] if base_a == 0 else [xs_s_track_action(base_a, MAXT), xs_x_track_action(base_a, MAXT)]
                        for sensor_idx, explicit_a in enumerate(actions):
                            priors_full[explicit_a] = float(priors[base_a]) * float(sensor_p[sensor_idx])
                            q_full[explicit_a] = float(q[base_a]) + float(sq[base_a, sensor_idx]) * self.q_scale
                    priors = priors_full
                    q = q_full
                out = (priors, float(value[i].detach().cpu()) * self.q_scale, q, xs[i], slots[i])
                self._net_cache[key] = out
        final = []
        for st, seq in states_and_seqs:
            final.append(self._net_cache[tuple(int(a) for a in seq)])
        return final

    def valid_actions(self, obs: Dict[str, np.ndarray]) -> List[int]:
        active = np.asarray(obs["active_mask"]).astype(bool)
        deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
        if self.sensor_action_mode in {"explicit", "explicit_head"}:
            ranges = np.asarray(obs.get("target_range", np.zeros_like(deadline)), dtype=np.float32)
            s_free = float(obs.get("s_band_busy_ms", 0.0)) <= 0.0
            x_enabled = bool(int(obs.get("enable_x_band", 0))) and not self.disable_x_search and not self.canonical_search_only
            x_free = x_enabled and float(obs.get("x_band_busy_ms", 0.0)) <= 0.0
            valid: List[int] = []
            if s_free:
                valid.append(xs_s_search_action(MAXT))
            if x_free:
                valid.append(xs_x_search_action(MAXT))
            tracked = active & (deadline >= 0.0)
            for idx in np.where(tracked)[0].astype(int).tolist():
                base_a = int(idx) + 1
                if base_a in self.forbidden_actions:
                    continue
                r = float(ranges[idx]) if idx < len(ranges) else 0.0
                if s_free and 10_000_000.0 < r < 184_000_000.0:
                    valid.append(xs_s_track_action(base_a, MAXT))
                if x_free and 5_000_000.0 < r < 100_000_000.0:
                    valid.append(xs_x_track_action(base_a, MAXT))
            return valid if valid else [MAXT + 1]
        valid = [xs_s_search_action(MAXT)]
        valid.extend(
            [
                int(a)
                for a in (np.where(active & (deadline >= 0.0))[0] + 1).astype(int).tolist()
                if int(a) not in self.forbidden_actions
            ]
        )
        return valid

    def _prior_for_action(self, priors: np.ndarray, action: int) -> float:
        """Map logical or explicit X/S action ids to the network prior.

        In explicit-head mode ``_net`` already expands priors to physical
        S/X actions.  Logical track ids can still appear when X-band is
        disabled, so map those ids back to their available physical heads
        instead of reading the zero-filled logical slot.
        """
        a = int(action)
        if self.sensor_action_mode == "explicit_head":
            base_a, sensor = xs_decode_action(a, MAXT)
            if sensor is not None and 0 <= a < len(priors):
                return float(priors[a])
            if int(base_a) == 0:
                s_idx = xs_s_search_action(MAXT)
                x_idx = xs_x_search_action(MAXT)
                total = float(priors[s_idx]) if 0 <= s_idx < len(priors) else 0.0
                if not self.disable_x_search and 0 <= x_idx < len(priors):
                    total += float(priors[x_idx])
                return total
            if int(base_a) > 0:
                s_idx = xs_s_track_action(int(base_a), MAXT)
                x_idx = xs_x_track_action(int(base_a), MAXT)
                total = float(priors[s_idx]) if 0 <= s_idx < len(priors) else 0.0
                if 0 <= x_idx < len(priors):
                    total += float(priors[x_idx])
                return total
        base_a, sensor = xs_decode_action(a, MAXT)
        p = float(priors[int(base_a)]) if 0 <= int(base_a) < len(priors) else 0.0
        return 0.5 * p if sensor is not None else p

    def _q_for_action(self, q: np.ndarray, action: int) -> float:
        a = int(action)
        base_a, _ = xs_decode_action(a, MAXT)
        if self.sensor_action_mode == "explicit_head" and int(base_a) > 0 and a == int(base_a):
            vals = []
            for idx in (xs_s_track_action(int(base_a), MAXT), xs_x_track_action(int(base_a), MAXT)):
                if 0 <= idx < len(q):
                    vals.append(float(q[idx]))
            if vals:
                return max(vals)
        if 0 <= a < len(q):
            return float(q[a])
        return float(q[int(base_a)]) if 0 <= int(base_a) < len(q) else 0.0

    def _q_bonus_lookup(self, q: Optional[np.ndarray], children: Sequence[ExactNode]) -> Dict[int, float]:
        if q is None or not children:
            return {}
        raw = np.asarray([self._q_for_action(q, int(c.action)) for c in children], dtype=np.float64)
        if self.q_utility_normalize:
            finite = np.isfinite(raw)
            if np.any(finite):
                lo = float(np.min(raw[finite]))
                hi = float(np.max(raw[finite]))
                if hi > lo + 1e-8:
                    raw = (raw - lo) / (hi - lo)
                else:
                    raw = np.zeros_like(raw)
            else:
                raw = np.zeros_like(raw)
        return {int(c.action): float(v) for c, v in zip(children, raw)}

    def _coverage_actions(self, obs: Dict[str, np.ndarray], valid: Sequence[int], cap: int) -> List[int]:
        """Mandatory root/action coverage before learned-prior pruning.

        The learned prior is useful, but with small exact rollout budgets it can
        prune every useful urgent track before exact edge rewards are measured.
        Preserve search actions and a few earliest-deadline track actions, then
        let the prior fill the rest of the expansion budget.
        """
        cap = max(1, int(cap))
        valid_list = [int(a) for a in valid]
        out: List[int] = []
        seen: set[int] = set()

        def add(action: int) -> None:
            if len(out) >= cap:
                return
            a = int(action)
            if a not in seen:
                seen.add(a)
                out.append(a)

        for action in valid_list:
            base, _ = xs_decode_action(int(action), MAXT)
            if int(base) == 0:
                add(int(action))

        active = np.asarray(obs["active_mask"]).astype(bool)
        deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
        tracked = active & (deadline >= 0.0)

        urgent = []
        for action in valid_list:
            base, _ = xs_decode_action(int(action), MAXT)
            idx = int(base) - 1
            if int(base) > 0 and 0 <= idx < len(deadline) and tracked[idx]:
                urgent.append((float(deadline[idx]), int(action)))
        for _, action in sorted(urgent, key=lambda x: (x[0], x[1])):
            add(action)
            if len(out) >= cap:
                break
        return out

    def expand(self, node: ExactNode):
        st = self.state(node.seq)
        if st.terminal:
            node.expanded = True
            return
        priors, value, _, _, _ = self._net(st.obs, node.seq)
        node.raw_value = float(value)
        valid = self.valid_actions(st.obs)
        valid_arr = np.asarray(valid, dtype=np.int64)
        prior_scores_all = np.asarray([self._prior_for_action(priors, int(a)) for a in valid_arr], dtype=np.float64)
        branch_mass_all: Dict[int, float] = {}
        if self.sensor_action_mode in {"explicit", "explicit_head"}:
            for a, p in zip(valid_arr.tolist(), prior_scores_all.tolist()):
                base, _ = xs_decode_action(int(a), MAXT)
                branch_key = 0 if int(base) == 0 else 1
                branch_mass_all[branch_key] = branch_mass_all.get(branch_key, 0.0) + max(0.0, float(p))
            total_branch_mass = float(sum(branch_mass_all.values()))
            if total_branch_mass > 1e-12:
                branch_mass_all = {k: float(v) / total_branch_mass for k, v in branch_mass_all.items()}
        logits_valid = np.log(np.clip(prior_scores_all, 1e-12, 1.0))
        gumbel_valid = np.zeros_like(prior_scores_all, dtype=np.float32)
        if len(node.seq) == 0 and self.search_alg == "gumbel":
            rng_seed = stable_seed(self.sim.seed if hasattr(self.sim, "seed") else 0, self.prefix, self.rollouts, self.expand_top_k, "gumbel")
            rng = np.random.default_rng(rng_seed)
            gumbel_valid = (self.gumbel_scale * rng.gumbel(size=prior_scores_all.shape)).astype(np.float32)
        if len(valid_arr) > self.expand_top_k:
            if len(node.seq) == 0 and self.search_alg == "gumbel":
                scores = logits_valid + gumbel_valid
                cap = min(self.expand_top_k, self.max_num_considered_actions)
            else:
                scores = prior_scores_all
                cap = self.expand_top_k
            coverage = self._coverage_actions(st.obs, valid_arr.tolist(), cap)
            action_to_idx = {int(a): i for i, a in enumerate(valid_arr.tolist())}
            keep_list = [action_to_idx[int(a)] for a in coverage if int(a) in action_to_idx]
            for idx in np.argsort(scores)[::-1].tolist():
                if len(keep_list) >= cap:
                    break
                if int(idx) not in keep_list:
                    keep_list.append(int(idx))
            keep = np.asarray(keep_list, dtype=np.int64)
            valid_arr = valid_arr[keep]
            prior_scores_all = prior_scores_all[keep]
            logits_valid = logits_valid[keep]
            gumbel_valid = gumbel_valid[keep]
            if self.sensor_action_mode not in {"explicit", "explicit_head"} and 0 not in valid_arr:
                valid_arr = np.unique(np.concatenate([np.asarray([0], dtype=np.int64), valid_arr]))
                prior_scores_all = np.asarray([self._prior_for_action(priors, int(a)) for a in valid_arr], dtype=np.float64)
                logits_valid = np.log(np.clip(prior_scores_all, 1e-12, 1.0))
                gumbel_valid = np.zeros_like(prior_scores_all, dtype=np.float32)
        masked = prior_scores_all.astype(np.float64)
        if branch_mass_all:
            branch_sum_selected: Dict[int, float] = {}
            for a, p in zip(valid_arr.tolist(), masked.tolist()):
                base, _ = xs_decode_action(int(a), MAXT)
                branch_key = 0 if int(base) == 0 else 1
                branch_sum_selected[branch_key] = branch_sum_selected.get(branch_key, 0.0) + max(0.0, float(p))
            adjusted = np.zeros_like(masked, dtype=np.float64)
            for i, (a, p) in enumerate(zip(valid_arr.tolist(), masked.tolist())):
                base, _ = xs_decode_action(int(a), MAXT)
                branch_key = 0 if int(base) == 0 else 1
                denom = float(branch_sum_selected.get(branch_key, 0.0))
                if denom > 1e-12:
                    adjusted[i] = max(0.0, float(p)) * float(branch_mass_all.get(branch_key, 0.0)) / denom
            masked = adjusted
        if float(masked.sum()) <= 0.0:
            masked[:] = 1.0 / len(masked)
        else:
            masked /= float(masked.sum())
        if self.prior_uniform_mix > 0.0 and len(masked) > 0:
            masked = (1.0 - self.prior_uniform_mix) * masked + self.prior_uniform_mix / float(len(masked))
        if (
            len(node.seq) == 0
            and self.root_dirichlet_alpha > 0.0
            and self.root_dirichlet_frac > 0.0
            and len(masked) > 1
        ):
            rng_seed = stable_seed(getattr(self.sim, "seed", 0), self.prefix, self.rollouts, self.expand_top_k, "root_dirichlet")
            rng = np.random.default_rng(rng_seed)
            noise = rng.dirichlet(np.full(len(masked), self.root_dirichlet_alpha, dtype=np.float64))
            masked = (1.0 - self.root_dirichlet_frac) * masked + self.root_dirichlet_frac * noise
            masked /= max(float(masked.sum()), 1e-12)
        children = []
        logit_lookup = {int(a): float(l) for a, l in zip(valid_arr.tolist(), logits_valid.tolist())}
        gumbel_lookup = {int(a): float(g) for a, g in zip(valid_arr.tolist(), gumbel_valid.tolist())}
        for a, p in zip(valid_arr.tolist(), masked.tolist()):
            child_seq = (*node.seq, int(a))
            child = (
                ExactNode(
                    seq=child_seq,
                    prior=float(p),
                    parent=node,
                    action=int(a),
                    prior_logit=float(logit_lookup.get(int(a), math.log(max(float(p), 1e-12)))),
                    root_gumbel=float(gumbel_lookup.get(int(a), 0.0)),
                )
            )
            if len(node.seq) < self.eager_edge_depth:
                self.eval_edge(child)
            children.append(child)
        node.children = children
        node.expanded = True

    def expand_model_only(self, node: ExactNode):
        """Root-only expansion for true zero-rollout deployment.

        This deliberately does not replay child transitions.  It is the cheap
        deployment path: one encoder/head call, mask invalid physical actions,
        then choose by learned Q/prior.  Training/search can still use exact
        edge evaluation; this path exists to measure the real inference floor.
        """
        st = self.state(node.seq)
        if st.terminal:
            node.expanded = True
            return
        priors, value, q, _, _ = self._net(st.obs, node.seq)
        node.raw_value = float(value)
        valid = self.valid_actions(st.obs)
        valid_arr = np.asarray(valid, dtype=np.int64)
        prior_scores = np.asarray([self._prior_for_action(priors, int(a)) for a in valid_arr], dtype=np.float64)
        if float(prior_scores.sum()) <= 0.0:
            prior_scores[:] = 1.0 / max(1, len(prior_scores))
        else:
            prior_scores /= float(prior_scores.sum())
        children: List[ExactNode] = []
        for a, p in zip(valid_arr.tolist(), prior_scores.tolist()):
            qa = self._q_for_action(q, int(a)) if self.use_q_head else 0.0
            child = ExactNode(
                seq=(*node.seq, int(a)),
                prior=float(p),
                parent=node,
                action=int(a),
                prior_logit=float(math.log(max(float(p), 1e-12))),
                total_value=float(qa),
            )
            children.append(child)
        node.children = children
        node.expanded = True

    def eval_edge(self, child: ExactNode):
        """Evaluate a child transition lazily.

        The previous implementation simulated every child during expansion.
        That is exact, but very expensive: most children are never visited by
        low-rollout search.  MCTS only needs the transition for selected/visited
        edges, so we defer the exact C replay until the edge is actually used.
        """
        if child.edge_evaluated:
            return
        parent_seq = child.parent.seq if child.parent is not None else ()
        parent_state = self.state(parent_seq)
        child_state = self.state(child.seq)
        child.edge_reward = float(child_state.reward - parent_state.reward)
        child.edge_dt_ms = float(child_state.dt_ms - parent_state.dt_ms)
        child.edge_evaluated = True

    def select(self, node: ExactNode) -> ExactNode:
        if self.search_alg == "gumbel":
            if node.parent is None:
                return self._select_root_gumbel(node)
            return self._select_interior_gumbel(node)
        if self.search_alg == "hierarchical":
            return self._select_hierarchical(node)
        if self.visit_unvisited_first:
            unvisited = [c for c in node.children if c.visits == 0]
            if unvisited:
                # Standard low-budget MCTS safeguard: before spending repeat
                # visits on one high-prior child, evaluate distinct siblings.
                # This is not radar-specific; it fixes first-play uncertainty.
                pred_q = None
                if self.use_q_head and self.q_utility_weight != 0.0:
                    _, _, pred_q, _, _ = self._net(self.state(node.seq).obs, node.seq)
                q_bonus = self._q_bonus_lookup(pred_q, unvisited)
                best = unvisited[0]
                best_score = -float("inf")
                for child in unvisited:
                    score = float(child.prior)
                    if child.edge_evaluated:
                        score += float(child.edge_reward)
                    if pred_q is not None:
                        score += self.q_utility_weight * q_bonus.get(int(child.action), 0.0)
                    if score > best_score:
                        best_score = score
                        best = child
                return best
        best = node.children[0]
        best_score = -float("inf")
        pred_q = None
        if self.use_q_head and self.q_utility_weight != 0.0:
            _, _, pred_q, _, _ = self._net(self.state(node.seq).obs, node.seq)
        q_bonus = self._q_bonus_lookup(pred_q, node.children)
        puct_q = self._puct_q_lookup(node.children)
        for child in node.children:
            q = puct_q.get(int(child.action), 0.0)
            u = self.c_puct * child.prior * math.sqrt(node.visits + 1.0) / (1.0 + child.visits)
            score = q + u
            if pred_q is not None:
                score += self.q_utility_weight * q_bonus.get(int(child.action), 0.0)
            if score > best_score:
                best = child
                best_score = score
        return best

    def _select_hierarchical(self, node: ExactNode) -> ExactNode:
        """Two-stage PUCT: first search-vs-track, then target-within-track.

        The flat PUCT selector makes the single search action compete directly
        against every target action.  That is the right baseline, but it does
        not test whether separate type/target heads specialize.  This selector
        preserves the two-head semantics: branch mass controls search-vs-track,
        conditional target mass controls which target to track.
        """
        search_children = []
        track_children = []
        for c in node.children:
            base, _ = xs_decode_action(int(c.action), MAXT)
            if int(base) == 0:
                search_children.append(c)
            else:
                track_children.append(c)
        if not search_children or not track_children:
            return self.select_puct_flat(node)

        pred_q = None
        if self.use_q_head and self.q_utility_weight != 0.0:
            _, _, pred_q, _, _ = self._net(self.state(node.seq).obs, node.seq)
        q_bonus = self._q_bonus_lookup(pred_q, node.children)
        puct_q = self._puct_q_lookup(node.children)

        def base_score(child: ExactNode) -> float:
            q = puct_q.get(int(child.action), 0.0)
            if pred_q is not None:
                q += self.q_utility_weight * q_bonus.get(int(child.action), 0.0)
            return q

        search = max(search_children, key=lambda c: float(c.prior))
        p_search = max(float(sum(c.prior for c in search_children)), 1e-12)
        p_track = max(float(sum(c.prior for c in track_children)), 1e-12)
        z = p_search + p_track
        p_search /= z
        p_track /= z

        n_search = float(sum(c.visits for c in search_children))
        n_track = float(sum(c.visits for c in track_children))
        parent_visits = float(node.visits + 1.0)

        best_search = search_children[0]
        best_search_score = -float("inf")
        search_prior_sum = max(float(sum(c.prior for c in search_children)), 1e-12)
        for child in search_children:
            p_cond = max(float(child.prior) / search_prior_sum, 1e-12)
            score = base_score(child) + self.c_puct * p_cond * math.sqrt(n_search + 1.0) / (1.0 + float(child.visits))
            if score > best_search_score:
                best_search_score = score
                best_search = child
        search_score = best_search_score + self.c_puct * p_search * math.sqrt(parent_visits) / (1.0 + n_search)

        # Choose a target conditionally inside the track branch.
        best_track = track_children[0]
        best_track_score = -float("inf")
        track_prior_sum = max(float(sum(c.prior for c in track_children)), 1e-12)
        for child in track_children:
            p_cond = max(float(child.prior) / track_prior_sum, 1e-12)
            score = base_score(child) + self.c_puct * p_cond * math.sqrt(n_track + 1.0) / (1.0 + float(child.visits))
            if score > best_track_score:
                best_track_score = score
                best_track = child

        track_score = best_track_score + self.c_puct * p_track * math.sqrt(parent_visits) / (1.0 + n_track)
        return best_track if track_score > search_score else best_search

    def select_puct_flat(self, node: ExactNode) -> ExactNode:
        best = node.children[0]
        best_score = -float("inf")
        pred_q = None
        if self.use_q_head and self.q_utility_weight != 0.0:
            _, _, pred_q, _, _ = self._net(self.state(node.seq).obs, node.seq)
        q_bonus = self._q_bonus_lookup(pred_q, node.children)
        puct_q = self._puct_q_lookup(node.children)
        for child in node.children:
            q = puct_q.get(int(child.action), 0.0)
            u = self.c_puct * child.prior * math.sqrt(node.visits + 1.0) / (1.0 + child.visits)
            score = q + u
            if pred_q is not None:
                score += self.q_utility_weight * q_bonus.get(int(child.action), 0.0)
            if score > best_score:
                best = child
                best_score = score
        return best

    def _completed_qvalues(self, node: ExactNode) -> Dict[int, float]:
        if not node.children:
            return {}
        q_raw = []
        prior_probs = []
        visited_mask = []
        for child in node.children:
            visited = child.visits > 0
            visited_mask.append(visited)
            q_raw.append(float(child.edge_reward + child.mean_value) if visited else 0.0)
            prior_probs.append(max(float(child.prior), 1e-12))
        q_raw_arr = np.asarray(q_raw, dtype=np.float64)
        prior_arr = np.asarray(prior_probs, dtype=np.float64)
        visited = np.asarray(visited_mask, dtype=bool)
        if np.any(visited):
            denom = max(float(np.sum(prior_arr[visited])), 1e-12)
            weighted_q = float(np.sum(prior_arr[visited] * q_raw_arr[visited] / denom))
        else:
            weighted_q = float(node.raw_value)
        mixed_value = (float(node.raw_value) + float(np.sum(visited)) * weighted_q) / (float(np.sum(visited)) + 1.0)
        completed = np.where(visited, q_raw_arr, mixed_value)
        q_min = float(np.min(completed))
        q_max = float(np.max(completed))
        if q_max > q_min + 1e-8:
            completed = (completed - q_min) / (q_max - q_min)
        else:
            completed = np.zeros_like(completed)
        visit_scale = self.mctx_maxvisit_init + max([c.visits for c in node.children] + [0])
        completed = visit_scale * self.mctx_value_scale * completed
        return {int(c.action): float(v) for c, v in zip(node.children, completed)}

    def _select_root_gumbel(self, node: ExactNode) -> ExactNode:
        sim_index = min(int(sum(c.visits for c in node.children)), len(self._considered_visits) - 1)
        considered_visit = self._considered_visits[sim_index]
        completed = self._completed_qvalues(node)
        eligible = [c for c in node.children if c.visits == considered_visit]
        if not eligible:
            eligible = node.children
        max_logit = max(float(c.prior_logit) for c in node.children)
        best = eligible[0]
        best_score = -float("inf")
        for child in eligible:
            score = float(child.root_gumbel) + (float(child.prior_logit) - max_logit) + completed.get(int(child.action), 0.0)
            if score > best_score:
                best_score = score
                best = child
        return best

    def _select_interior_gumbel(self, node: ExactNode) -> ExactNode:
        completed = self._completed_qvalues(node)
        logits = np.asarray([float(c.prior_logit) + completed.get(int(c.action), 0.0) for c in node.children], dtype=np.float64)
        probs = _softmax_np(logits)
        total_visits = float(sum(c.visits for c in node.children))
        scores = probs - np.asarray([c.visits for c in node.children], dtype=np.float64) / (1.0 + total_visits)
        return node.children[int(np.argmax(scores))]

    def rollout_trace(self, node: ExactNode, budget_ms: Optional[float] = None) -> Tuple[float, Tuple[int, ...]]:
        actions = [*self.prefix, *node.seq]
        st = self.state(node.seq)
        total = 0.0
        elapsed = 0.0
        limit_ms = self.horizon_ms if budget_ms is None else max(0.0, float(budget_ms))
        local_seq = list(node.seq)
        while elapsed < limit_ms and not st.terminal:
            valid = self.valid_actions(st.obs)
            if not valid:
                break
            if self.rollout_policy == "random" or self.rng.random() < self.epsilon:
                a = int(self.rng.choice(valid))
            elif self.rollout_policy == "edf" or (
                self.rollout_policy == "mixed" and self.rng.random() >= self.rollout_est_prob
            ):
                active = np.asarray(st.obs["active_mask"]).astype(bool)
                deadline = np.asarray(st.obs["t_deadline"], dtype=np.float32)
                tracked = active & (deadline >= 0.0)
                search_debt = float(st.obs.get("search_debt_ms", 0.0))
                search_actions = [int(v) for v in valid if int(xs_decode_action(int(v), MAXT)[0]) == 0]
                if search_actions and (not np.any(tracked) or search_debt >= 200.0):
                    a = int(search_actions[0])
                else:
                    candidates = [
                        (float(deadline[int(base_v) - 1]), int(v))
                        for v in valid
                        for base_v, _sensor in [xs_decode_action(int(v), MAXT)]
                        if int(base_v) > 0 and 0 <= int(base_v) - 1 < len(deadline) and tracked[int(base_v) - 1]
                    ]
                    a = min(candidates, key=lambda x: x[0])[1] if candidates else (int(search_actions[0]) if search_actions else int(valid[0]))
            elif self.rollout_policy in {"est", "mixed"}:
                active = np.asarray(st.obs["active_mask"]).astype(bool)
                desired = np.asarray(st.obs["t_desired"], dtype=np.float32)
                deadline = np.asarray(st.obs["t_deadline"], dtype=np.float32)
                tracked = active & (deadline >= 0.0)
                search_debt = float(st.obs.get("search_debt_ms", 0.0))
                search_actions = [int(v) for v in valid if int(xs_decode_action(int(v), MAXT)[0]) == 0]
                if search_actions and (not np.any(tracked) or search_debt >= 300.0):
                    a = int(search_actions[0])
                else:
                    candidates = [
                        (float(desired[int(base_v) - 1]), int(v))
                        for v in valid
                        for base_v, _sensor in [xs_decode_action(int(v), MAXT)]
                        if int(base_v) > 0 and 0 <= int(base_v) - 1 < len(desired) and tracked[int(base_v) - 1]
                    ]
                    a = min(candidates, key=lambda x: x[0])[1] if candidates else (int(search_actions[0]) if search_actions else int(valid[0]))
            elif self.rollout_policy in {"q", "pq"}:
                priors, _, pred_q, _, _ = self._net(st.obs, local_seq)
                best_score = -float("inf")
                a = int(valid[0])
                for v in valid:
                    qv = float(self._q_for_action(pred_q, int(v)))
                    if self.rollout_policy == "pq":
                        qv += self.c_puct * float(self._prior_for_action(priors, int(v)))
                    if qv > best_score:
                        best_score = qv
                        a = int(v)
            elif self.rollout_policy in {"branch", "branch_margin"}:
                priors, _, _, _, _ = self._net(st.obs, local_seq)
                search_actions = []
                track_actions = []
                for v in valid:
                    base_v, _sensor_v = xs_decode_action(int(v), MAXT)
                    if int(base_v) == 0:
                        search_actions.append(int(v))
                    else:
                        track_actions.append(int(v))
                p_search = float(self._branch_prob_cache.get(tuple(int(a) for a in local_seq), 0.0))
                if self.rollout_policy == "branch_margin" and search_actions and track_actions:
                    best_search = max(search_actions, key=lambda v: float(self._prior_for_action(priors, int(v))))
                    best_track = max(track_actions, key=lambda v: float(self._prior_for_action(priors, int(v))))
                    search_score = float(self._prior_for_action(priors, int(best_search)))
                    track_score = float(self._prior_for_action(priors, int(best_track)))
                    if search_score >= float(self.branch_rollout_threshold) * max(track_score, 1e-12):
                        a = int(best_search)
                    else:
                        a = int(best_track)
                elif search_actions and (not track_actions or p_search >= self.branch_rollout_threshold):
                    a = max(search_actions, key=lambda v: float(self._prior_for_action(priors, int(v))))
                elif track_actions:
                    a = max(track_actions, key=lambda v: float(self._prior_for_action(priors, int(v))))
                else:
                    a = int(valid[0])
            elif self.rollout_policy == "edge":
                active = np.asarray(st.obs["active_mask"]).astype(bool)
                deadline = np.asarray(st.obs["t_deadline"], dtype=np.float32)
                desired = np.asarray(st.obs["t_desired"], dtype=np.float32)
                dwell = np.asarray(st.obs["t_dwell"], dtype=np.float32)
                priority = np.asarray(st.obs["priority"], dtype=np.float32)
                best_score = -float("inf")
                a = int(valid[0])
                for v in valid:
                    base_v, _sensor_v = xs_decode_action(int(v), MAXT)
                    if int(base_v) == 0:
                        # Small positive pressure to keep surveillance alive;
                        # exact env reward after stepping will still decide.
                        score = 0.02 * max(0.0, float(st.obs.get("search_debt_ms", 0.0)) / 200.0)
                    else:
                        i = int(base_v) - 1
                        if not (0 <= i < len(deadline)):
                            score = -float("inf")
                        else:
                            slack = float(deadline[i] - max(1.0, dwell[i]))
                            tardy = max(0.0, -float(desired[i]))
                            score = 0.30 + 0.002 * tardy * (1.0 + 2.0 * float(priority[i])) - 0.0005 * max(0.0, 100.0 - slack)
                    if score > best_score:
                        best_score = score
                        a = int(v)
            else:
                priors, _, _, _, _ = self._net(st.obs, local_seq)
                prior_scores = np.asarray([self._prior_for_action(priors, int(v)) for v in valid], dtype=np.float64)
                a = int(valid[int(np.argmax(prior_scores))])
            prev = st
            local_seq.append(a)
            st = self.sim.replay([*self.prefix, *local_seq])
            dr = float(st.reward - prev.reward)
            dt = max(1.0, float(st.dt_ms - prev.dt_ms))
            total += dr
            elapsed += dt
        return total, tuple(local_seq)

    def rollout(self, node: ExactNode) -> float:
        value, _ = self.rollout_trace(node)
        return value

    def update_best_seq(self, seq: Sequence[int]):
        raw = tuple(int(a) for a in seq)
        if not raw:
            return
        if hasattr(self.sim, "evaluate_plan_sequence"):
            executed, value, elapsed = self.sim.evaluate_plan_sequence(raw, self.horizon_ms)
            if executed and elapsed > 0.0 and value > self.best_seq_value:
                self.best_seq_value = float(value)
                self.best_seq = tuple(int(a) for a in executed)
            return
        root_state = self.state(())
        key_list: List[int] = []
        prev = root_state
        elapsed = 0.0
        for action in raw:
            candidate = tuple([*key_list, int(action)])
            st = self.state(candidate)
            dt = float(st.dt_ms - prev.dt_ms)
            if dt <= 0.0 or elapsed >= self.horizon_ms:
                break
            key_list.append(int(action))
            elapsed += dt
            prev = st
            if elapsed >= self.horizon_ms:
                break
        if not key_list:
            return
        key = tuple(key_list)
        st = self.state(key)
        value = float(st.reward - root_state.reward)
        if value > self.best_seq_value:
            self.best_seq_value = value
            self.best_seq = key

    def backprop(self, node: ExactNode, value: float):
        cur = node
        v = float(value)
        while cur is not None:
            cur.visits += 1
            cur.total_value += v
            v += cur.edge_reward
            cur = cur.parent

    def run(self) -> ExactNode:
        root = ExactNode(seq=())
        self.root = root
        if self.fast_zero_rollout and self.rollouts <= 0:
            self.expand_model_only(root)
            return root
        if self.seed_rollout_policies:
            old_policy = self.rollout_policy
            old_epsilon = self.epsilon
            try:
                self.epsilon = 0.0
                for policy in self.seed_rollout_policies:
                    if policy in {"planner_edf", "planner_est"}:
                        st = self.state(())
                        planner = EDFPlanner(MAXT) if policy == "planner_edf" else ESTPlanner(MAXT)
                        candidate = tuple(int(a) for a in planner.plan(st.obs, int(max(200.0, self.horizon_ms)))[:96])
                        self.update_best_seq(candidate)
                        continue
                    if policy == "planner_edf_fast":
                        st = self.state(())
                        planner = EDFPlanner(MAXT)
                        candidate = tuple(int(a) for a in planner.plan(st.obs, int(max(200.0, self.horizon_ms)))[:96])
                        self.best_seq = candidate
                        self.best_seq_value = float("inf")
                        continue
                    if policy == "planner_est_fast":
                        st = self.state(())
                        planner = ESTPlanner(MAXT)
                        candidate = tuple(int(a) for a in planner.plan(st.obs, int(max(200.0, self.horizon_ms)))[:96])
                        self.best_seq = candidate
                        self.best_seq_value = float("inf")
                        continue
                    self.rollout_policy = policy
                    _, seed_seq = self.rollout_trace(root)
                    self.update_best_seq(seed_seq)
            finally:
                self.rollout_policy = old_policy
                self.epsilon = old_epsilon
        if self.rollout_policy not in {"edge", "value"} and not self.skip_default_rollout_seed:
            _, seed_seq = self.rollout_trace(root)
            self.update_best_seq(seed_seq)
        if not (self.fast_zero_rollout and self.rollouts <= 0 and self.seed_rollout_policies):
            self.expand(root)
        for _ in range(self.rollouts):
            node = root
            while node.expanded and node.children:
                node = self.select(node)
                self.eval_edge(node)
            if not node.expanded and self.rollout_policy != "edge":
                self.expand(node)
            if self.state(node.seq).terminal:
                value = 0.0
            elif self.rollout_policy == "edge":
                value = 0.0
                self.update_best_seq(node.seq)
            elif self.rollout_policy == "value":
                value = float(node.raw_value)
                self.update_best_seq(node.seq)
            else:
                rollout_value, full_seq = self.rollout_trace(node)
                if self.use_value_head:
                    value = (1.0 - self.leaf_value_mix) * float(rollout_value) + self.leaf_value_mix * float(node.raw_value)
                else:
                    value = float(rollout_value)
                self.update_best_seq(full_seq)
            self.backprop(node, value)
        if not self.best_seq:
            visited = [c.seq for c in root.children if c.visits > 0]
            for seq in visited:
                self.update_best_seq(seq)
        return root

    def target_from_root(self, root: ExactNode) -> SearchTarget:
        st = self.state(())
        _, _, _, x, slot = self._net(st.obs, ())
        pi = np.zeros((MAXT + 1,), dtype=np.float32)
        q = np.zeros((MAXT + 1,), dtype=np.float32)
        q_mask = np.zeros((MAXT + 1,), dtype=np.float32)
        sensor_pi = np.zeros((MAXT + 1, 2), dtype=np.float32)
        sensor_q = np.zeros((MAXT + 1, 2), dtype=np.float32)
        sensor_q_mask = np.zeros((MAXT + 1, 2), dtype=np.float32)
        use_sensor_targets = self.sensor_action_mode in {"explicit", "explicit_head"}
        sensor_pi = np.zeros((MAXT + 1, 2), dtype=np.float32)
        sensor_q = np.zeros((MAXT + 1, 2), dtype=np.float32)
        sensor_q_mask = np.zeros((MAXT + 1, 2), dtype=np.float32)
        total_visits = max(1, sum(c.visits for c in root.children))
        use_sensor_targets = self.sensor_action_mode in {"explicit", "explicit_head"}
        for child in root.children:
            self.eval_edge(child)
            base_action, sensor_id = xs_decode_action(int(child.action), MAXT)
            target_sensor_id = 0 if sensor_id is None and int(base_action) >= 0 else sensor_id
            if child.visits > 0:
                qv = float(child.edge_reward + child.mean_value)
                if self.duration_normalize_q and child.edge_dt_ms > 0.0:
                    qv /= max(0.05, float(child.edge_dt_ms) / 200.0)
                if 0 <= int(base_action) < len(q):
                    q[int(base_action)] = max(float(q[int(base_action)]), qv) if q_mask[int(base_action)] > 0.5 else qv
                    q_mask[int(base_action)] = 1.0
                if use_sensor_targets and target_sensor_id is not None and 0 <= int(base_action) <= MAXT:
                    sensor_q[int(base_action), int(target_sensor_id)] = qv
                    sensor_q_mask[int(base_action), int(target_sensor_id)] = 1.0
            elif getattr(self, "complete_root_q_with_value", False):
                _, child_value, _, _, _ = self._net(self.state(child.seq).obs, child.seq)
                qv = float(child.edge_reward + child_value)
                if self.duration_normalize_q and child.edge_dt_ms > 0.0:
                    qv /= max(0.05, float(child.edge_dt_ms) / 200.0)
                if 0 <= int(base_action) < len(q):
                    q[int(base_action)] = max(float(q[int(base_action)]), qv) if q_mask[int(base_action)] > 0.5 else qv
                    q_mask[int(base_action)] = 1.0
                if use_sensor_targets and target_sensor_id is not None and 0 <= int(base_action) <= MAXT:
                    sensor_q[int(base_action), int(target_sensor_id)] = qv
                    sensor_q_mask[int(base_action), int(target_sensor_id)] = 1.0
            elif child.edge_evaluated:
                qv = float(child.edge_reward)
                if self.duration_normalize_q and child.edge_dt_ms > 0.0:
                    qv /= max(0.05, float(child.edge_dt_ms) / 200.0)
                if 0 <= int(base_action) < len(q):
                    q[int(base_action)] = max(float(q[int(base_action)]), qv) if q_mask[int(base_action)] > 0.5 else qv
                    q_mask[int(base_action)] = 1.0
                if use_sensor_targets and target_sensor_id is not None and 0 <= int(base_action) <= MAXT:
                    sensor_q[int(base_action), int(target_sensor_id)] = qv
                    sensor_q_mask[int(base_action), int(target_sensor_id)] = 1.0
            if 0 <= int(base_action) < len(pi):
                pi[int(base_action)] += float(child.visits / total_visits)
            if use_sensor_targets and target_sensor_id is not None and 0 <= int(base_action) <= MAXT:
                sensor_pi[int(base_action), int(target_sensor_id)] += float(child.visits / total_visits)
        if self.policy_target == "mctx":
            completed = self._completed_qvalues(root)
            logits = np.full((MAXT + 1,), -1e9, dtype=np.float64)
            for child in root.children:
                base_action, _ = xs_decode_action(int(child.action), MAXT)
                if 0 <= int(base_action) < len(logits):
                    score = float(child.prior_logit) + completed.get(int(child.action), 0.0)
                    logits[int(base_action)] = max(float(logits[int(base_action)]), score)
            pi = _softmax_np(logits)
        elif self.policy_target in {"branch_q_softmax", "branch_future_softmax"} and root.children:
            branch_score_by_base: Dict[int, float] = {}
            future_score_by_base: Dict[int, float] = {}
            for child in root.children:
                self.eval_edge(child)
                base_action, _ = xs_decode_action(int(child.action), MAXT)
                if not (0 <= int(base_action) < len(pi)):
                    continue
                if child.visits > 0:
                    edge_plus_value = float(child.edge_reward + child.mean_value)
                    future_only = float(child.mean_value)
                elif child.edge_evaluated and self.policy_target != "branch_future_softmax":
                    edge_plus_value = float(child.edge_reward)
                    future_only = 0.0
                else:
                    continue
                branch_score_by_base[int(base_action)] = max(
                    branch_score_by_base.get(int(base_action), -1e30),
                    edge_plus_value,
                )
                future_score_by_base[int(base_action)] = max(
                    future_score_by_base.get(int(base_action), -1e30),
                    future_only,
                )
            score_by_base = future_score_by_base if self.policy_target == "branch_future_softmax" else branch_score_by_base
            search_q = score_by_base.get(0, None)
            track_actions = np.asarray([a for a in score_by_base.keys() if int(a) > 0], dtype=np.int64)
            track_scores = np.asarray([score_by_base[int(a)] for a in track_actions], dtype=np.float64)
            pi = np.zeros_like(pi)
            if search_q is not None and len(track_scores) > 0:
                branch_logits = np.asarray([float(search_q), float(np.max(track_scores))], dtype=np.float64) / self.policy_tau
                branch_logits -= float(np.max(branch_logits))
                branch_probs = np.exp(np.clip(branch_logits, -60.0, 60.0))
                branch_probs /= max(float(np.sum(branch_probs)), 1e-12)
                pi[0] = float(branch_probs[0])
                track_logits = track_scores / self.policy_tau
                track_logits -= float(np.max(track_logits))
                track_probs = np.exp(np.clip(track_logits, -60.0, 60.0))
                track_probs /= max(float(np.sum(track_probs)), 1e-12)
                pi[track_actions] = (float(branch_probs[1]) * track_probs).astype(np.float32)
            elif search_q is not None:
                pi[0] = 1.0
            elif len(track_scores) > 0:
                track_logits = track_scores / self.policy_tau
                track_logits -= float(np.max(track_logits))
                track_probs = np.exp(np.clip(track_logits, -60.0, 60.0))
                track_probs /= max(float(np.sum(track_probs)), 1e-12)
                pi[track_actions] = track_probs.astype(np.float32)
        elif self.policy_target in {"q_softmax", "mixed"} and float(np.sum(q_mask)) > 0.0:
            actions = np.where(q_mask > 0.5)[0]
            logits = q[actions].astype(np.float64) / self.policy_tau
            logits -= float(np.max(logits))
            probs = np.exp(np.clip(logits, -60.0, 60.0))
            probs /= max(float(np.sum(probs)), 1e-12)
            q_pi = np.zeros_like(pi)
            q_pi[actions] = probs.astype(np.float32)
            if self.policy_target == "q_softmax":
                pi = q_pi
            else:
                pi = 0.5 * pi + 0.5 * q_pi
        if use_sensor_targets and self.policy_target in {"q_softmax", "mixed", "mctx", "branch_q_softmax", "branch_future_softmax"}:
            # Keep the factorized policy target consistent with transformed
            # base-action pi.  The training path consumes sensor_pi, so changing
            # only pi silently leaves the heads trained on the old visit target.
            old_sensor_pi = sensor_pi
            sensor_pi = np.zeros((MAXT + 1, 2), dtype=np.float32)
            for base_action in np.where(pi > 0.0)[0]:
                mass = float(pi[int(base_action)])
                sensor_mask = sensor_q_mask[int(base_action)] > 0.5
                if np.any(sensor_mask):
                    sensor_ids = np.where(sensor_mask)[0]
                    logits = sensor_q[int(base_action), sensor_ids].astype(np.float64) / self.policy_tau
                    logits -= float(np.max(logits))
                    probs = np.exp(np.clip(logits, -60.0, 60.0))
                    probs /= max(float(np.sum(probs)), 1e-12)
                    sensor_pi[int(base_action), sensor_ids] = (mass * probs).astype(np.float32)
                elif float(np.sum(old_sensor_pi[int(base_action)])) > 0.0:
                    sensor_pi[int(base_action)] = (
                        mass
                        * old_sensor_pi[int(base_action)]
                        / max(float(np.sum(old_sensor_pi[int(base_action)])), 1e-12)
                    )
                else:
                    sensor_pi[int(base_action), 0] = mass
        return SearchTarget(
            x=x,
            slot=slot,
            pi=pi,
            q=q,
            q_mask=q_mask,
            search_count=0,
            track_count=0,
            sensor_pi=sensor_pi if use_sensor_targets else None,
            sensor_q=sensor_q if use_sensor_targets else None,
            sensor_q_mask=sensor_q_mask if use_sensor_targets else None,
        )

    def target_from_node(self, node: ExactNode) -> SearchTarget:
        """Extract a policy/Q target at an arbitrary in-window tree node."""
        st = self.state(node.seq)
        _, _, _, x, slot = self._net(st.obs, node.seq)
        pi = np.zeros((MAXT + 1,), dtype=np.float32)
        q = np.zeros((MAXT + 1,), dtype=np.float32)
        q_mask = np.zeros((MAXT + 1,), dtype=np.float32)
        sensor_pi = np.zeros((MAXT + 1, 2), dtype=np.float32)
        sensor_q = np.zeros((MAXT + 1, 2), dtype=np.float32)
        sensor_q_mask = np.zeros((MAXT + 1, 2), dtype=np.float32)
        use_sensor_targets = self.sensor_action_mode in {"explicit", "explicit_head"}
        total_visits = max(1, sum(c.visits for c in node.children))
        for child in node.children:
            self.eval_edge(child)
            base_action, sensor_id = xs_decode_action(int(child.action), MAXT)
            if int(base_action) < 0:
                continue
            target_sensor_id = 0 if sensor_id is None else int(sensor_id)
            if 0 <= int(base_action) < len(pi):
                pi[int(base_action)] += float(child.visits / total_visits)
                if child.visits > 0 or child.edge_evaluated:
                    qv = float(child.edge_reward + (child.mean_value if child.visits > 0 else 0.0))
                    q[int(base_action)] = max(float(q[int(base_action)]), qv) if q_mask[int(base_action)] > 0.5 else qv
                    q_mask[int(base_action)] = 1.0
            if use_sensor_targets and 0 <= int(base_action) <= MAXT:
                sensor_pi[int(base_action), target_sensor_id] += float(child.visits / total_visits)
                if child.visits > 0 or child.edge_evaluated:
                    qv = float(child.edge_reward + (child.mean_value if child.visits > 0 else 0.0))
                    sensor_q[int(base_action), target_sensor_id] = qv
                    sensor_q_mask[int(base_action), target_sensor_id] = 1.0

        if self.policy_target in {"q_softmax", "mixed", "mctx", "branch_q_softmax", "branch_future_softmax"} and float(np.sum(q_mask)) > 0.0:
            actions = np.where(q_mask > 0.5)[0]
            logits = q[actions].astype(np.float64) / self.policy_tau
            logits -= float(np.max(logits))
            probs = np.exp(np.clip(logits, -60.0, 60.0))
            probs /= max(float(np.sum(probs)), 1e-12)
            q_pi = np.zeros_like(pi)
            q_pi[actions] = probs.astype(np.float32)
            pi = 0.5 * pi + 0.5 * q_pi if self.policy_target == "mixed" else q_pi
            if use_sensor_targets:
                sensor_pi[:] = 0.0
                for base_action in actions:
                    smask = sensor_q_mask[int(base_action)] > 0.5
                    if not np.any(smask):
                        continue
                    sensor_ids = np.where(smask)[0]
                    slogits = sensor_q[int(base_action), sensor_ids].astype(np.float64) / self.policy_tau
                    slogits -= float(np.max(slogits))
                    sp = np.exp(np.clip(slogits, -60.0, 60.0))
                    sp /= max(float(np.sum(sp)), 1e-12)
                    sensor_pi[int(base_action), sensor_ids] = (float(pi[int(base_action)]) * sp).astype(np.float32)

        search_count = 0
        track_count = 0
        for action in node.seq:
            base_action, _ = xs_decode_action(int(action), MAXT)
            if int(base_action) == 0:
                search_count += 1
            elif int(base_action) > 0:
                track_count += 1
        return SearchTarget(
            x=x,
            slot=slot,
            pi=pi,
            q=q,
            q_mask=q_mask,
            search_count=int(search_count),
            track_count=int(track_count),
            sensor_pi=sensor_pi if use_sensor_targets else None,
            sensor_q=sensor_q if use_sensor_targets else None,
            sensor_q_mask=sensor_q_mask if use_sensor_targets else None,
        )

    def target_counterfactual_branch_q(
        self,
        root: ExactNode,
        top_k: int = 8,
        mode: str = "value",
        subrollouts: int = 8,
        candidate_mode: str = "prior",
    ) -> SearchTarget:
        """Train branch heads from forced first-action counterfactuals.

        Root visit counts can be a poor type target: a low-rollout tree may
        never test search and track siblings fairly.  This target explicitly
        evaluates "force search first" and "force candidate track first", then
        stores those returns in q/q_mask.  ``train_step_branch_max`` converts
        those Q values into the factorized type target:
        Q(search) vs max_i Q(track_i).
        """
        st = self.state(())
        priors, _, _, x, slot = self._net(st.obs, ())
        valid = self.valid_actions(st.obs)
        search_actions = [int(a) for a in valid if int(xs_decode_action(int(a), MAXT)[0]) == 0]
        tracks = [int(a) for a in valid if int(xs_decode_action(int(a), MAXT)[0]) > 0]
        if len(tracks) > int(top_k):
            if str(candidate_mode) == "urgent":
                deadline = np.asarray(st.obs.get("t_deadline", []), dtype=np.float32)
                tracks = sorted(
                    tracks,
                    key=lambda a: (
                        float(deadline[xs_decode_action(int(a), MAXT)[0] - 1])
                        if xs_decode_action(int(a), MAXT)[0] > 0 and xs_decode_action(int(a), MAXT)[0] - 1 < len(deadline)
                        else 1e9,
                        int(a),
                    ),
                )[: int(top_k)]
            else:
                tracks = sorted(tracks, key=lambda a: float(self._prior_for_action(priors, int(a))), reverse=True)[: int(top_k)]
        candidates = [*search_actions, *tracks]

        pi = np.zeros((MAXT + 1,), dtype=np.float32)
        q = np.zeros((MAXT + 1,), dtype=np.float32)
        q_mask = np.zeros((MAXT + 1,), dtype=np.float32)
        sensor_pi = np.zeros((MAXT + 1, 2), dtype=np.float32)
        sensor_q = np.zeros((MAXT + 1, 2), dtype=np.float32)
        sensor_q_mask = np.zeros((MAXT + 1, 2), dtype=np.float32)
        use_sensor_targets = self.sensor_action_mode in {"explicit", "explicit_head"}
        root_state = self.state(())
        old_policy = self.rollout_policy

        def rollout_candidates(obs: Dict[str, np.ndarray], seq: Tuple[int, ...]) -> List[int]:
            valid_next = self.valid_actions(obs)
            next_search = [int(a) for a in valid_next if int(xs_decode_action(int(a), MAXT)[0]) == 0]
            next_tracks = [int(a) for a in valid_next if int(xs_decode_action(int(a), MAXT)[0]) > 0]
            cap = max(1, int(top_k))
            if len(next_tracks) > cap:
                if str(candidate_mode) == "urgent":
                    deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
                    next_tracks = sorted(
                        next_tracks,
                        key=lambda a: (
                            float(deadline[xs_decode_action(int(a), MAXT)[0] - 1])
                            if xs_decode_action(int(a), MAXT)[0] > 0 and xs_decode_action(int(a), MAXT)[0] - 1 < len(deadline)
                            else 1e9,
                            int(a),
                        ),
                    )[:cap]
                else:
                    local_priors, _, _, _, _ = self._net(obs, seq)
                    next_tracks = sorted(
                        next_tracks,
                        key=lambda a: float(self._prior_for_action(local_priors, int(a))),
                        reverse=True,
                    )[:cap]
            return [*next_search, *next_tracks]

        def terminal_penalty_potential(obs: Dict[str, np.ndarray]) -> float:
            active = np.asarray(obs.get("active_mask", []), dtype=bool)
            deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
            desired = np.asarray(obs.get("t_desired", np.zeros_like(deadline)), dtype=np.float32)
            n = min(len(active), len(deadline), len(desired), MAXT)
            target_cost = 0.0
            if n > 0:
                active_n = active[:n]
                if np.any(active_n):
                    target_cost += float(self.sim.env_cfg.get("track_loss_penalty", 4.0)) * float(
                        np.sum(active_n & (deadline[:n] < 0.0))
                    )
                    target_cost += float(np.sum(np.maximum(0.0, -desired[:n][active_n]) / 200.0))

            grid = np.asarray(obs.get("grid", []), dtype=np.float32)
            frame_cost = 0.0
            desired_ms = float(self.sim.env_cfg.get("search_frame_desired_ms", 3000.0))
            deadline_ms = float(self.sim.env_cfg.get("search_frame_deadline_ms", 4500.0))
            frame_weight = float(self.sim.env_cfg.get("search_frame_overdue_weight", 0.0))
            if grid.size > 0 and desired_ms > 0.0 and frame_weight > 0.0:
                age = 3000.0 - grid
                overdue = np.maximum(0.0, age - desired_ms)
                frame_terms = np.square(overdue / max(1e-6, desired_ms))
                drop_penalty = float(self.sim.env_cfg.get("search_frame_drop_penalty", 0.0))
                if deadline_ms > 0.0 and drop_penalty > 0.0:
                    frame_terms = frame_terms + drop_penalty * (age > deadline_ms)
                frame_cost = frame_weight * float(np.mean(frame_terms))
            return -float(target_cost + frame_cost)

        def greedy_edge_future(seq: Tuple[int, ...], budget_ms: float) -> Tuple[float, ReplayState]:
            total = 0.0
            local_seq = tuple(int(a) for a in seq)
            prev = self.state(local_seq)
            elapsed = 0.0
            while elapsed < float(budget_ms) and not prev.terminal:
                valid_next = rollout_candidates(prev.obs, local_seq)
                best_action = None
                best_score = -float("inf")
                best_state = None
                best_dr = 0.0
                best_dt = 0.0
                for cand in valid_next:
                    cand_seq = (*local_seq, int(cand))
                    cand_state = self.state(cand_seq)
                    dr = float(cand_state.reward - prev.reward)
                    dt = float(cand_state.dt_ms - prev.dt_ms)
                    if dt <= 0.0:
                        continue
                    score = dr / max(1.0, dt)
                    if score > best_score:
                        best_score = score
                        best_action = int(cand)
                        best_state = cand_state
                        best_dr = dr
                        best_dt = dt
                if best_action is None or best_state is None:
                    break
                total += float(best_dr)
                elapsed += float(best_dt)
                local_seq = (*local_seq, int(best_action))
                prev = best_state
            return total, prev

        for action in candidates:
            child = ExactNode(
                seq=(int(action),),
                prior=float(self._prior_for_action(priors, int(action))),
                parent=root,
                action=int(action),
            )
            self.eval_edge(child)
            child_state = self.state(child.seq)
            if child.edge_dt_ms <= 0.0:
                continue
            if str(mode) == "edge_density":
                value = float(child.edge_reward)
            elif str(mode) == "edge_greedy_rollout":
                remaining = max(0.0, self.horizon_ms - float(child.edge_dt_ms))
                future, _ = greedy_edge_future(tuple(child.seq), remaining)
                value = float(child.edge_reward + future)
            elif str(mode) == "edge_greedy_potential":
                remaining = max(0.0, self.horizon_ms - float(child.edge_dt_ms))
                future, final_state = greedy_edge_future(tuple(child.seq), remaining)
                value = float(child.edge_reward + future + terminal_penalty_potential(final_state.obs))
            elif str(mode) == "rollout":
                remaining = max(0.0, self.horizon_ms - float(child.edge_dt_ms))
                future, _ = self.rollout_trace(child, budget_ms=remaining)
                value = float(child.edge_reward + future)
            elif str(mode) == "model_rollout":
                self.rollout_policy = "model"
                try:
                    remaining = max(0.0, self.horizon_ms - float(child.edge_dt_ms))
                    future, _ = self.rollout_trace(child, budget_ms=remaining)
                    value = float(child.edge_reward + future)
                finally:
                    self.rollout_policy = old_policy
            elif str(mode) == "subtree":
                remaining = max(0.0, self.horizon_ms - float(child.edge_dt_ms))
                sub = ExactEnvMCTS(
                    self.model,
                    self.sim,
                    [*self.prefix, int(action)],
                    rollouts=max(0, int(subrollouts)),
                    c_puct=self.c_puct,
                    expand_top_k=self.expand_top_k,
                    horizon_windows=1,
                    rollout_policy=self.rollout_policy,
                    prior_mode=self.prior_mode,
                    q_scale=self.q_scale,
                    epsilon=self.epsilon,
                    policy_target=self.policy_target,
                    policy_tau=self.policy_tau,
                    branch_rollout_threshold=self.branch_rollout_threshold,
                    search_alg=self.search_alg,
                    max_num_considered_actions=self.max_num_considered_actions,
                    gumbel_scale=self.gumbel_scale,
                    mctx_value_scale=self.mctx_value_scale,
                    mctx_maxvisit_init=self.mctx_maxvisit_init,
                    eager_edge_depth=self.eager_edge_depth,
                    prior_uniform_mix=self.prior_uniform_mix,
                    rollout_est_prob=self.rollout_est_prob,
                    mask_selected=self.mask_selected,
                    stateless_tree_context=self.stateless_tree_context,
                    head_mode=self.head_mode,
                    q_utility_weight=self.q_utility_weight,
                    q_utility_normalize=self.q_utility_normalize,
                    leaf_value_mix=self.leaf_value_mix,
                    seed_rollout_policies=(),
                    fast_zero_rollout=False,
                    skip_default_rollout_seed=True,
                    complete_root_q_with_value=self.complete_root_q_with_value,
                    visit_unvisited_first=self.visit_unvisited_first,
                    duration_normalize_q=self.duration_normalize_q,
                    prior_q_beta=self.prior_q_beta,
                    prior_search_bias=self.prior_search_bias,
                    adaptive_search_bias=self.adaptive_search_bias,
                    adaptive_search_target_load=self.adaptive_search_target_load,
                    forbidden_actions=self.forbidden_actions,
                    sensor_action_mode=self.sensor_action_mode,
                    disable_x_search=self.disable_x_search,
                )
                sub.horizon_ms = remaining
                sub_root = sub.run()
                future = float(sub.best_seq_value) if sub.best_seq else 0.0
                if not sub.best_seq and sub_root.children:
                    future = max(float(c.edge_reward + c.mean_value) for c in sub_root.children)
                value = float(child.edge_reward + future)
            else:
                _, child_value, _, _, _ = self._net(child_state.obs, child.seq)
                value = float(child.edge_reward + child_value)
            if self.duration_normalize_q or str(mode) == "edge_density":
                value /= max(0.05, float(child.edge_dt_ms) / 200.0)
            base_action, sensor_id = xs_decode_action(int(action), MAXT)
            if 0 <= int(base_action) < len(q):
                q[int(base_action)] = max(float(q[int(base_action)]), float(value)) if q_mask[int(base_action)] > 0.5 else float(value)
                q_mask[int(base_action)] = 1.0
            if use_sensor_targets and sensor_id is not None and 0 <= int(base_action) <= MAXT:
                sensor_q[int(base_action), int(sensor_id)] = float(value)
                sensor_q_mask[int(base_action), int(sensor_id)] = 1.0

        if float(np.sum(q_mask)) > 0.0:
            actions = np.where(q_mask > 0.5)[0]
            logits = q[actions].astype(np.float64) / self.policy_tau
            logits -= float(np.max(logits))
            probs = np.exp(np.clip(logits, -60.0, 60.0))
            probs /= max(float(np.sum(probs)), 1e-12)
            pi[actions] = probs.astype(np.float32)
        if use_sensor_targets:
            for base_action in range(MAXT + 1):
                if np.any(sensor_q_mask[base_action] > 0.5):
                    slogits = sensor_q[base_action].astype(np.float64)
                    slogits[sensor_q_mask[base_action] <= 0.5] = -1e9
                    slogits -= float(np.max(slogits))
                    sp = np.exp(np.clip(slogits, -60.0, 60.0))
                    sp /= max(float(np.sum(sp)), 1e-12)
                    sensor_pi[base_action] = sp.astype(np.float32) * float(pi[base_action] if base_action < len(pi) else 1.0)
        return SearchTarget(
            x=x,
            slot=slot,
            pi=pi,
            q=q,
            q_mask=q_mask,
            search_count=0,
            track_count=0,
            sensor_pi=sensor_pi if use_sensor_targets else None,
            sensor_q=sensor_q if use_sensor_targets else None,
            sensor_q_mask=sensor_q_mask if use_sensor_targets else None,
        )


def _child_branch(child: ExactNode) -> int:
    base, _ = xs_decode_action(int(child.action), MAXT)
    return 0 if int(base) == 0 else 1


def _best_child_in_branch(children: Sequence[ExactNode], mode: str) -> ExactNode:
    if str(mode) == "q":
        visited = [c for c in children if int(c.visits) > 0]
        pool = visited if visited else list(children)
        return max(pool, key=lambda c: c.edge_reward + c.mean_value)
    return max(children, key=lambda c: (c.visits, c.edge_reward + c.mean_value, c.prior))


def choose_root_action(root: ExactNode, mode: str = "visits") -> int:
    if not root.children:
        return 0
    if mode == "prior":
        return int(max(root.children, key=lambda c: (c.prior, c.visits, c.edge_reward + c.mean_value)).action)
    if mode in {"branch_visits", "branch_q"}:
        branches: Dict[int, List[ExactNode]] = {0: [], 1: []}
        for child in root.children:
            branches[_child_branch(child)].append(child)
        nonempty = {k: v for k, v in branches.items() if v}
        if not nonempty:
            return int(max(root.children, key=lambda c: (c.visits, c.edge_reward + c.mean_value)).action)
        if mode == "branch_q":
            def branch_best_q(k: int) -> float:
                visited = [c for c in nonempty[k] if int(c.visits) > 0]
                pool = visited if visited else nonempty[k]
                return max(c.edge_reward + c.mean_value for c in pool)

            branch = max(nonempty, key=branch_best_q)
            return int(_best_child_in_branch(nonempty[branch], "q").action)
        branch = max(
            nonempty,
            key=lambda k: (
                sum(int(c.visits) for c in nonempty[k]),
                max(c.edge_reward + c.mean_value for c in nonempty[k]),
            ),
        )
        return int(_best_child_in_branch(nonempty[branch], "visits").action)
    if mode == "q":
        # Unvisited children have no backed-up value yet; treating their
        # missing value as zero lets them win spuriously in negative-reward
        # states.  Use backed-up Q when available, falling back only if the
        # root genuinely has no visits.
        visited = [c for c in root.children if int(c.visits) > 0]
        pool = visited if visited else root.children
        return int(max(pool, key=lambda c: c.edge_reward + c.mean_value).action)
    return int(max(root.children, key=lambda c: (c.visits, c.edge_reward + c.mean_value)).action)


def choose_root_action_load_gated(root: ExactNode, obs: Dict[str, np.ndarray], threshold: int) -> int:
    active = np.asarray(obs.get("active_mask", []), dtype=bool)
    active_count = int(np.sum(active)) if active.size else 0
    mode = "prior" if active_count <= int(threshold) else "visits"
    return choose_root_action(root, mode)


def sample_root_action(root: ExactNode, tau: float, seed: int) -> int:
    if not root.children:
        return 0
    tau = float(tau)
    if tau <= 0.0:
        return choose_root_action(root, "visits")
    visits = np.asarray([max(0, int(c.visits)) for c in root.children], dtype=np.float64)
    if float(visits.sum()) <= 0.0:
        probs = np.asarray([max(0.0, float(c.prior)) for c in root.children], dtype=np.float64)
    else:
        probs = np.power(visits, 1.0 / max(1e-6, tau))
    if float(probs.sum()) <= 0.0:
        probs = np.full(len(root.children), 1.0 / max(1, len(root.children)), dtype=np.float64)
    else:
        probs /= float(probs.sum())
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    idx = int(rng.choice(len(root.children), p=probs))
    return int(root.children[idx].action)


def _node_tree_value(mcts: ExactEnvMCTS, child: ExactNode) -> float:
    q = float(child.edge_reward + child.mean_value)
    if mcts.duration_normalize_q and child.edge_evaluated and child.edge_dt_ms > 0.0:
        q /= max(0.05, float(child.edge_dt_ms) / 200.0)
    return q


def greedy_tree_plan(mcts: ExactEnvMCTS, root: ExactNode, mode: str = "visits", budget_ms: float = 200.0) -> List[int]:
    """Extract a planned path from the already-built tree."""
    plan: List[int] = []
    elapsed = 0.0
    node = root
    while node.children and elapsed < float(budget_ms):
        if mode == "q":
            pool = [c for c in node.children if c.visits > 0] or node.children
            child = max(pool, key=lambda c: _node_tree_value(mcts, c))
        else:
            child = max(node.children, key=lambda c: (c.visits, _node_tree_value(mcts, c)))
        mcts.eval_edge(child)
        if child.edge_dt_ms <= 0.0:
            break
        plan.append(int(child.action))
        elapsed += float(child.edge_dt_ms)
        node = child
    return plan


def greedy_tree_fill_plan(mcts: ExactEnvMCTS, root: ExactNode, mode: str = "visits", budget_ms: float = 200.0) -> List[int]:
    """Extract from the searched tree, then keep filling the 200ms window.

    Low-rollout MCTS often builds only the first one or two levels.  Stopping
    there wastes radar time and made light-load results look much worse than
    the actual model.  The tail fill remains transformer-only: when the searched
    tree ends, expand the current model state and select the next action using
    the learned prior/Q plus exact one-step reward.
    """
    plan: List[int] = []
    elapsed = 0.0
    node = root
    while elapsed < float(budget_ms):
        if not node.expanded:
            mcts.expand(node)
        if not node.children:
            break
        visited = [c for c in node.children if c.visits > 0]
        if visited:
            pool = visited
            if mode == "q":
                child = max(pool, key=lambda c: _node_tree_value(mcts, c))
            else:
                child = max(pool, key=lambda c: (c.visits, _node_tree_value(mcts, c)))
        else:
            child = mcts.select(node)
        mcts.eval_edge(child)
        if child.edge_dt_ms <= 0.0:
            break
        plan.append(int(child.action))
        elapsed += float(child.edge_dt_ms)
        node = child
    return plan


def best_window_plan(mcts: ExactEnvMCTS, root: ExactNode, mode: str = "visits", budget_ms: float = 200.0) -> List[int]:
    """Prefer the best rollout trajectory, but always fill the window budget.

    Value-only / Gumbel-style searches can identify a strong short prefix
    without producing a full rollout trace.  Returning that short prefix makes
    evaluation look artificially good because the radar only executes a few ms
    in a 200ms window.  Treat ``best_seq`` as a prefix and fill the tail from
    the searched/model-expanded tree.
    """
    seq = list(mcts.best_seq)
    if seq:
        return [int(a) for a in seq]
    seq = greedy_tree_fill_plan(mcts, root, mode, budget_ms)
    out: List[int] = []
    prev = mcts.state(())
    elapsed = 0.0
    prefix: List[int] = []
    for action in seq:
        prefix.append(int(action))
        st = mcts.state(prefix)
        dt = float(st.dt_ms - prev.dt_ms)
        if dt <= 0.0 or elapsed >= float(budget_ms):
            break
        out.append(int(action))
        elapsed += dt
        prev = st
        if elapsed >= float(budget_ms):
            break
    if elapsed < float(budget_ms):
        node = root
        found = True
        for action in prefix:
            matches = [c for c in node.children if int(c.action) == int(action)]
            if not matches:
                found = False
                break
            node = matches[0]
        if not found:
            node = ExactNode(seq=tuple(prefix), expanded=False)
        tail = greedy_tree_fill_plan(mcts, node, mode, max(0.0, float(budget_ms) - elapsed))
        out.extend(int(a) for a in tail)
    if not out and root.children:
        out = greedy_tree_fill_plan(mcts, root, mode, budget_ms) or [choose_root_action(root, mode)]
    return out


def fill_window_plan(
    mcts: ExactEnvMCTS,
    root: ExactNode,
    plan: Sequence[int],
    mode: str = "visits",
    budget_ms: float = 200.0,
) -> List[int]:
    """Append model/tree actions until the plan covers the requested budget."""
    out = [int(a) for a in plan]
    prev = mcts.state(())
    elapsed = 0.0
    prefix: List[int] = []
    for action in out:
        prefix.append(int(action))
        st = mcts.state(prefix)
        dt = float(st.dt_ms - prev.dt_ms)
        if dt <= 0.0:
            break
        elapsed += dt
        prev = st
        if elapsed >= float(budget_ms):
            return out

    node = root
    found = True
    for action in prefix:
        matches = [c for c in node.children if int(c.action) == int(action)]
        if not matches:
            found = False
            break
        node = matches[0]
    if not found:
        node = ExactNode(seq=tuple(prefix), expanded=False)
    out.extend(greedy_tree_fill_plan(mcts, node, mode, max(0.0, float(budget_ms) - elapsed)))
    return out


def set_selected_action_target(target: SearchTarget, action: int) -> None:
    """Make a factorized one-hot target from a physical X/S action."""
    base_action, sensor_id = xs_decode_action(int(action), MAXT)
    if sensor_id is None and int(base_action) >= 0:
        sensor_id = 0
    target.pi[:] = 0.0
    if 0 <= int(base_action) < len(target.pi):
        target.pi[int(base_action)] = 1.0
    if getattr(target, "sensor_pi", None) is not None:
        target.sensor_pi[:] = 0.0
        if sensor_id is not None and 0 <= int(base_action) < target.sensor_pi.shape[0]:
            target.sensor_pi[int(base_action), int(sensor_id)] = 1.0
    if getattr(target, "sensor_q_mask", None) is not None:
        target.sensor_q_mask[:] = 0.0
        if sensor_id is not None and 0 <= int(base_action) < target.sensor_q_mask.shape[0]:
            target.sensor_q_mask[int(base_action), int(sensor_id)] = 1.0


def best_sequence_prefix_targets(mcts: ExactEnvMCTS, plan: Sequence[int]) -> List[SearchTarget]:
    """Train every decision in the MCTS-selected window sequence.

    Root-only AlphaZero targets are too sparse for our deployment planner,
    which decodes a full 200ms sequence.  These targets are still generated by
    MCTS/self-play: each prefix state gets the next action from the best
    MCTS-backed sequence.
    """
    out: List[SearchTarget] = []
    prefix: List[int] = []
    node = mcts.root if hasattr(mcts, "root") else None
    for action in plan:
        st = mcts.state(prefix)
        if node is not None and node.children:
            target = mcts.target_from_node(node)
            out.append(target)
            matches = [c for c in node.children if int(c.action) == int(action)]
            node = matches[0] if matches else None
            prefix.append(int(action))
            continue

        _, _, _, x, slot = mcts._net(st.obs, prefix)
        pi = np.zeros((MAXT + 1,), dtype=np.float32)
        q = np.zeros((MAXT + 1,), dtype=np.float32)
        q_mask = np.zeros((MAXT + 1,), dtype=np.float32)
        sensor_pi = np.zeros((MAXT + 1, 2), dtype=np.float32)
        sensor_q = np.zeros((MAXT + 1, 2), dtype=np.float32)
        sensor_q_mask = np.zeros((MAXT + 1, 2), dtype=np.float32)
        a = int(action)
        base_action, sensor_id = xs_decode_action(a, MAXT)
        if sensor_id is None and int(base_action) >= 0:
            sensor_id = 0
        if 0 <= int(base_action) <= MAXT:
            q_val = float(mcts.state([*prefix, a]).reward - st.reward)
            pi[int(base_action)] = 1.0
            q_mask[int(base_action)] = 1.0
            q[int(base_action)] = q_val
            if sensor_id is not None:
                sensor_pi[int(base_action), int(sensor_id)] = 1.0
                sensor_q[int(base_action), int(sensor_id)] = q_val
                sensor_q_mask[int(base_action), int(sensor_id)] = 1.0
        out.append(
            SearchTarget(
                x=x,
                slot=slot,
                pi=pi,
                q=q,
                q_mask=q_mask,
                search_count=int(sum(1 for pa in prefix if int(xs_decode_action(int(pa), MAXT)[0]) == 0)),
                track_count=int(sum(1 for pa in prefix if int(xs_decode_action(int(pa), MAXT)[0]) > 0)),
                sensor_pi=sensor_pi,
                sensor_q=sensor_q,
                sensor_q_mask=sensor_q_mask,
            )
        )
        prefix.append(a)
    return out


def greedy_expand_window_plan(mcts: ExactEnvMCTS, root: ExactNode, budget_ms: float = 200.0) -> List[int]:
    """Build one full-window path inside one tree, expanding only the chosen branch."""
    plan: List[int] = []
    elapsed = 0.0
    node = root
    while elapsed < float(budget_ms):
        if not node.expanded:
            mcts.expand(node)
        if not node.children:
            break
        best = None
        best_score = -float("inf")
        pred_q = None
        q_bonus = {}
        if mcts.use_q_head and mcts.q_utility_weight != 0.0:
            _, _, pred_q, _, _ = mcts._net(mcts.state(node.seq).obs, node.seq)
            q_bonus = mcts._q_bonus_lookup(pred_q, node.children)
        for child in node.children:
            mcts.eval_edge(child)
            if child.edge_dt_ms <= 0.0:
                continue
            # Match one fresh PUCT selection at this node: exact edge reward
            # plus the learned prior bonus.  This is equivalent to doing the
            # cheap one-rollout edge planner repeatedly, but inside one
            # window tree instead of re-instantiating MCTS per action.
            score = float(child.edge_reward) + mcts.c_puct * float(child.prior)
            if pred_q is not None:
                score += mcts.q_utility_weight * q_bonus.get(int(child.action), 0.0)
            if score > best_score:
                best = child
                best_score = score
        if best is None:
            break
        plan.append(int(best.action))
        elapsed += float(best.edge_dt_ms)
        node = best
    return plan


def batched_value_window_plan(mcts: ExactEnvMCTS, root: ExactNode, budget_ms: float = 200.0) -> List[int]:
    """Plan a full 200ms window by batched exact child-state value lookahead.

    This is a transformer-only planner: no EDF/EST ordering is injected.  At
    each slot it expands the current node, exact-steps every candidate child,
    evaluates those child observations in one transformer batch, then commits
    the child with the highest exact immediate reward plus learned value.
    """
    plan: List[int] = []
    elapsed = 0.0
    node = root
    while elapsed < float(budget_ms):
        if not node.expanded:
            mcts.expand(node)
        if not node.children:
            break
        candidates = []
        for child in node.children:
            mcts.eval_edge(child)
            if child.edge_dt_ms > 0.0:
                candidates.append(child)
        if not candidates:
            break
        evals = mcts._net_many([(mcts.state(child.seq), child.seq) for child in candidates])
        parent_q = None
        q_bonus = {}
        if mcts.use_q_head and mcts.q_utility_weight != 0.0:
            _, _, parent_q, _, _ = mcts._net(mcts.state(node.seq).obs, node.seq)
            q_bonus = mcts._q_bonus_lookup(parent_q, candidates)
        best_child = candidates[0]
        best_score = -float("inf")
        value_mix = float(np.clip(mcts.leaf_value_mix, 0.0, 1.0))
        for child, (_, child_value, _, _, _) in zip(candidates, evals):
            score = float(child.edge_reward) + value_mix * float(child_value) + mcts.c_puct * float(child.prior)
            if parent_q is not None:
                score += mcts.q_utility_weight * q_bonus.get(int(child.action), 0.0)
            if score > best_score:
                best_score = score
                best_child = child
        plan.append(int(best_child.action))
        elapsed += float(best_child.edge_dt_ms)
        node = best_child
    return plan


def model_q_window_plan(mcts: ExactEnvMCTS, root: ExactNode, budget_ms: float = 200.0) -> List[int]:
    """Autoregressive model-Q window planner with exact state updates.

    This is the low-latency bridge between full atomic MCTS and a static batch
    decoder: after each chosen action, the exact simulator state is advanced
    before the next model call.  It does not expand/evaluate all root children;
    it only scores currently valid actions with the learned prior/Q.
    """
    del root  # The prefix is represented by node.seq below.
    plan: List[int] = []
    elapsed = 0.0
    while elapsed < float(budget_ms):
        node = ExactNode(seq=tuple(plan))
        mcts.expand_model_only(node)
        if not node.children:
            break
        candidates = sorted(node.children, key=lambda c: c.edge_reward + c.mean_value, reverse=True)
        child = None
        for cand in candidates:
            mcts.eval_edge(cand)
            if cand.edge_dt_ms > 0.0:
                child = cand
                break
        if child is None:
            break
        plan.append(int(child.action))
        elapsed += float(child.edge_dt_ms)
    return plan


def edge_q_window_plan(mcts: ExactEnvMCTS, root: ExactNode, budget_ms: float = 200.0) -> List[int]:
    """Autoregressive one-step exact edge + learned-Q planner.

    This keeps the state update correctness of atomic planning, but removes
    tree rollouts.  At each slot it scores valid actions by exact immediate
    reward plus the parent Q head, then commits the best valid edge.
    """
    del root
    plan: List[int] = []
    elapsed = 0.0
    while elapsed < float(budget_ms):
        node = ExactNode(seq=tuple(plan))
        mcts.expand_model_only(node)
        if not node.children:
            break
        best_child = None
        best_score = -float("inf")
        for child in node.children:
            mcts.eval_edge(child)
            if child.edge_dt_ms <= 0.0:
                continue
            score = float(child.edge_reward) + float(child.mean_value)
            if score > best_score:
                best_score = score
                best_child = child
        if best_child is None:
            break
        plan.append(int(best_child.action))
        elapsed += float(best_child.edge_dt_ms)
    return plan


def run_exact_episode(model: MutualRadarNet, args, initial_targets: int, rate: float, seed: int, train: bool = False):
    env_cfg = env_cfg_for(rate, args)
    sim = ExactReplaySimulator(initial_targets, seed, env_cfg, MAXT)
    history: List[int] = []
    rows: List[Dict[str, float]] = []
    targets: List[SearchTarget] = []
    cumulative = 0.0
    for window in range(int(args.windows)):
        window_reward = 0.0
        window_ms = 0.0
        window_actions: List[int] = []
        window_forbidden: set[int] = set()
        while window_ms < 200.0:
            mcts = ExactEnvMCTS(
                model,
                sim,
                history,
                q_scale=float(getattr(args, "q_scale", 100.0)),
                rollouts=args.rollouts,
                c_puct=args.c_puct,
                expand_top_k=args.expand_top_k,
                horizon_windows=args.horizon_windows,
                rollout_policy=args.rollout_policy,
                prior_mode=args.prior_mode,
                epsilon=args.epsilon,
                policy_target=args.policy_target,
                policy_tau=args.policy_tau,
                branch_rollout_threshold=getattr(args, "branch_rollout_threshold", 0.65),
                search_alg=args.search_alg,
                max_num_considered_actions=args.max_num_considered_actions,
                gumbel_scale=args.gumbel_scale,
                mctx_value_scale=args.mctx_value_scale,
                mctx_maxvisit_init=args.mctx_maxvisit_init,
                eager_edge_depth=args.eager_edge_depth,
                prior_uniform_mix=args.prior_uniform_mix,
                root_dirichlet_alpha=getattr(args, "root_dirichlet_alpha", 0.0),
                root_dirichlet_frac=getattr(args, "root_dirichlet_frac", 0.0),
                rollout_est_prob=args.rollout_est_prob,
                mask_selected=not args.allow_retrack_in_window,
                stateless_tree_context=args.stateless_tree_context,
                head_mode=args.head_mode,
                q_utility_weight=args.q_utility_weight,
                q_utility_normalize=args.q_utility_normalize,
                puct_q_transform=getattr(args, "puct_q_transform", "raw"),
                leaf_value_mix=args.leaf_value_mix,
                seed_rollout_policies=args.seed_rollout_policies.split(",") if args.seed_rollout_policies else (),
                fast_zero_rollout=args.fast_zero_rollout,
                skip_default_rollout_seed=args.skip_default_rollout_seed,
                complete_root_q_with_value=args.complete_root_q_with_value,
                visit_unvisited_first=args.visit_unvisited_first,
                duration_normalize_q=args.duration_normalize_q,
                prior_q_beta=args.prior_q_beta,
                prior_search_bias=args.prior_search_bias,
                adaptive_search_bias=getattr(args, "adaptive_search_bias", 0.0),
                adaptive_search_target_load=getattr(args, "adaptive_search_target_load", 0.75),
                forbidden_actions=window_forbidden if args.forbid_retrack_within_window else (),
                sensor_action_mode=args.sensor_action_mode,
                disable_x_search=args.disable_x_search,
                canonical_search_only=args.canonical_search_only,
            )
            root = mcts.run()
            target = (
                mcts.target_counterfactual_branch_q(
                    root,
                    args.counterfactual_top_k,
                    args.counterfactual_mode,
                    args.counterfactual_subrollouts,
                    args.counterfactual_candidate_mode,
                )
                if train and args.counterfactual_branch_q
                else (mcts.target_from_root(root) if train else None)
            )
            if args.plan_mode in {"window", "first_window"}:
                if args.window_extract == "tree":
                    plan = greedy_tree_plan(mcts, root, args.select_mode, 200.0 - window_ms)
                elif args.window_extract == "tree_fill":
                    plan = greedy_tree_fill_plan(mcts, root, args.select_mode, 200.0 - window_ms)
                elif args.window_extract == "greedy_expand":
                    plan = greedy_expand_window_plan(mcts, root, 200.0 - window_ms)
                elif args.window_extract == "batched_value":
                    plan = batched_value_window_plan(mcts, root, 200.0 - window_ms)
                elif args.window_extract == "model_q":
                    plan = model_q_window_plan(mcts, root, 200.0 - window_ms)
                elif args.window_extract == "edge_q":
                    plan = edge_q_window_plan(mcts, root, 200.0 - window_ms)
                else:
                    plan = best_window_plan(mcts, root, args.select_mode, 200.0 - window_ms)
            else:
                plan = [choose_root_action(root, args.select_mode)]
            if args.plan_mode == "first_window":
                plan = plan[:1] if plan else [choose_root_action(root, args.select_mode)]
            if train and args.plan_mode == "atomic" and getattr(args, "self_play_sample_tau", 0.0) > 0.0:
                sample_seed = stable_seed(getattr(args, "seed", 0), seed, window, len(history), "replay_sample")
                plan = [sample_root_action(root, args.self_play_sample_tau, sample_seed)]
            plan_reward = 0.0
            for action in plan:
                before = sim.replay(history)
                after = sim.replay([*history, action])
                dr = float(after.reward - before.reward)
                dt = float(after.dt_ms - before.dt_ms)
                if dt <= 0.0 or after.terminal:
                    break
                if train and target is not None and args.target_selected_action:
                    set_selected_action_target(target, int(action))
                if train and target is not None and args.plan_mode in {"atomic", "first_window"}:
                    target.reward = dr
                    targets.append(target)
                else:
                    plan_reward += dr
                history.append(action)
                window_actions.append(int(action))
                base_action, _ = xs_decode_action(int(action), MAXT)
                if args.forbid_retrack_within_window and int(base_action) > 0:
                    window_forbidden.add(int(base_action))
                window_reward += dr
                window_ms += dt
                if window_ms >= 200.0:
                    break
            if train and target is not None and args.plan_mode == "window" and plan_reward != 0.0:
                target.reward = plan_reward
                if args.add_prefix_targets:
                    prefix_targets = best_sequence_prefix_targets(mcts, plan)
                    per = plan_reward / max(1, len(prefix_targets))
                    if args.counterfactual_branch_q:
                        target.reward = float(per)
                        targets.append(target)
                    for pt in prefix_targets:
                        pt.reward = float(per)
                    targets.extend(prefix_targets)
                else:
                    targets.append(target)
            if train and len(targets) >= int(args.max_targets_per_episode):
                break
            if args.plan_mode == "window":
                break
        cumulative += window_reward
        st = sim.replay(history)
        obs = st.obs
        active = np.asarray(obs["active_mask"]).astype(bool)
        tracked = active & (np.asarray(obs["t_deadline"], dtype=np.float32) >= 0.0)
        dropped = active & (np.asarray(obs["t_deadline"], dtype=np.float32) < 0.0)
        rows.append(
            {
                "window": window + 1,
                "window_reward": float(window_reward),
                "cumulative_reward": float(cumulative),
                "window_ms_used": float(window_ms),
                "actions": int(len(history)),
                "search_fraction": float(np.mean([xs_decode_action(a, MAXT)[0] == 0 for a in window_actions])) if window_actions else 0.0,
                **xs_action_fractions(window_actions, MAXT),
                "active_targets": float(np.sum(active)),
                "tracked_targets": float(np.sum(tracked)),
                "drop_pct_active": float(100.0 * np.sum(dropped) / max(1, np.sum(active))) if np.any(active) else 0.0,
                "mean_delay_active": float(np.mean(np.maximum(0.0, -obs["t_desired"][active]))) if np.any(active) else 0.0,
            }
        )
        if train and len(targets) >= int(args.max_targets_per_episode):
            break
    G = 0.0
    for target in reversed(targets):
        G = float(target.reward) + float(args.gamma) * G
        target.ret = G
    return pd.DataFrame(rows), targets


def run_snapshot_exact_episode(model: MutualRadarNet, args, initial_targets: int, rate: float, seed: int, train: bool = False):
    env_cfg = env_cfg_for(rate, args)
    eng = build_env(_DummyPlanner(), initial_targets, MAXT, seed, 200, engine_env_cfg(env_cfg))
    eng.reset(seed=seed)
    rows: List[Dict[str, float]] = []
    targets: List[SearchTarget] = []
    history: List[int] = []
    step_rewards: List[float] = []
    cumulative = 0.0
    debt = 0.0
    max_train_targets = int(getattr(args, "max_targets_per_episode", 0))
    try:
        for window in range(int(args.windows)):
            window_reward = 0.0
            window_ms = 0.0
            window_actions: List[int] = []
            window_forbidden: set[int] = set()
            target_start_window = max(1, int(getattr(args, "target_start_window", 1)))
            target_stride = max(1, int(getattr(args, "target_stride", 1)))
            collect_window_target = (
                train
                and int(window + 1) >= target_start_window
                and ((int(window + 1) - target_start_window) % target_stride == 0)
                and (max_train_targets <= 0 or len(targets) < max_train_targets)
            )
            while window_ms < 200.0 and not eng.term_buf[0]:
                sim = SnapshotSimulator(
                    eng,
                    debt,
                    env_cfg,
                    bool(getattr(args, "use_arrival_feature", False)),
                    bool(getattr(args, "use_grid_feature", False)),
                    int(seed),
                )
                mcts = ExactEnvMCTS(
                    model,
                    sim,
                    [],
                    q_scale=float(getattr(args, "q_scale", 100.0)),
                    rollouts=args.rollouts,
                    c_puct=args.c_puct,
                    expand_top_k=args.expand_top_k,
                    horizon_windows=args.horizon_windows,
                    rollout_policy=args.rollout_policy,
                    prior_mode=args.prior_mode,
                    epsilon=args.epsilon,
                    policy_target=args.policy_target,
                    policy_tau=args.policy_tau,
                    branch_rollout_threshold=getattr(args, "branch_rollout_threshold", 0.65),
                    search_alg=args.search_alg,
                    max_num_considered_actions=args.max_num_considered_actions,
                    gumbel_scale=args.gumbel_scale,
                    mctx_value_scale=args.mctx_value_scale,
                    mctx_maxvisit_init=args.mctx_maxvisit_init,
                    eager_edge_depth=args.eager_edge_depth,
                    prior_uniform_mix=args.prior_uniform_mix,
                    root_dirichlet_alpha=getattr(args, "root_dirichlet_alpha", 0.0),
                    root_dirichlet_frac=getattr(args, "root_dirichlet_frac", 0.0),
                    rollout_est_prob=args.rollout_est_prob,
                    mask_selected=not args.allow_retrack_in_window,
                    stateless_tree_context=args.stateless_tree_context,
                    head_mode=args.head_mode,
                    q_utility_weight=args.q_utility_weight,
                    q_utility_normalize=args.q_utility_normalize,
                    leaf_value_mix=args.leaf_value_mix,
                    seed_rollout_policies=args.seed_rollout_policies.split(",") if args.seed_rollout_policies else (),
                    fast_zero_rollout=args.fast_zero_rollout,
                    skip_default_rollout_seed=args.skip_default_rollout_seed,
                    complete_root_q_with_value=args.complete_root_q_with_value,
                    visit_unvisited_first=args.visit_unvisited_first,
                    duration_normalize_q=args.duration_normalize_q,
                    prior_q_beta=args.prior_q_beta,
                    prior_search_bias=args.prior_search_bias,
                    adaptive_search_bias=getattr(args, "adaptive_search_bias", 0.0),
                    adaptive_search_target_load=getattr(args, "adaptive_search_target_load", 0.75),
                    forbidden_actions=window_forbidden if args.forbid_retrack_within_window else (),
                    sensor_action_mode=args.sensor_action_mode,
                    disable_x_search=args.disable_x_search,
                    canonical_search_only=args.canonical_search_only,
                )
                root = mcts.run()
                target = None
                if collect_window_target:
                    target = (
                        mcts.target_counterfactual_branch_q(
                            root,
                            args.counterfactual_top_k,
                            args.counterfactual_mode,
                            args.counterfactual_subrollouts,
                            args.counterfactual_candidate_mode,
                        )
                        if args.counterfactual_branch_q
                        else mcts.target_from_root(root)
                    )
                if args.plan_mode in {"window", "first_window"}:
                    if args.window_extract == "tree":
                        plan = greedy_tree_plan(mcts, root, args.select_mode, 200.0 - window_ms)
                    elif args.window_extract == "tree_fill":
                        plan = greedy_tree_fill_plan(mcts, root, args.select_mode, 200.0 - window_ms)
                    elif args.window_extract == "greedy_expand":
                        plan = greedy_expand_window_plan(mcts, root, 200.0 - window_ms)
                    elif args.window_extract == "batched_value":
                        plan = batched_value_window_plan(mcts, root, 200.0 - window_ms)
                    elif args.window_extract == "model_q":
                        plan = model_q_window_plan(mcts, root, 200.0 - window_ms)
                    elif args.window_extract == "edge_q":
                        plan = edge_q_window_plan(mcts, root, 200.0 - window_ms)
                    else:
                        plan = best_window_plan(mcts, root, args.select_mode, 200.0 - window_ms)
                    if args.plan_mode == "first_window":
                        plan = plan[:1] if plan else [choose_root_action(root, args.select_mode)]
                    else:
                        _checked_actions, _checked_reward, checked_ms = sim.evaluate_plan_sequence(plan, 200.0 - window_ms)
                        if checked_ms < (200.0 - window_ms) - 1e-6:
                            plan = fill_window_plan(mcts, root, plan, args.select_mode, 200.0 - window_ms)
                    executed_steps, debt = sim.commit_sequence(plan, 200.0 - window_ms)
                    if not executed_steps:
                        debt += max(0.0, 200.0 - window_ms)
                        window_ms = 200.0
                        break
                    if target is not None and (max_train_targets <= 0 or len(targets) < max_train_targets):
                        target.reward = float(sum(r for r, _, _ in executed_steps))
                        target.initial = int(initial_targets)
                        target.rate = float(rate)
                        target.seed = int(seed)
                        target.window = int(window + 1)
                        target.action_index = int(len(history))
                        if args.plan_mode == "first_window":
                            if args.target_selected_action:
                                set_selected_action_target(target, int(executed_steps[0][2]))
                            targets.append(target)
                        elif args.add_prefix_targets:
                            prefix_targets = best_sequence_prefix_targets(mcts, [int(e) for _, _, e in executed_steps])
                            denom = len(prefix_targets) + (1 if args.counterfactual_branch_q else 0)
                            per = target.reward / max(1, denom)
                            if args.counterfactual_branch_q:
                                target.reward = float(per)
                                targets.append(target)
                            for offset, pt in enumerate(prefix_targets):
                                pt.reward = float(per)
                                pt.initial = int(initial_targets)
                                pt.rate = float(rate)
                                pt.seed = int(seed)
                                pt.window = int(window + 1)
                                pt.action_index = int(len(history) + offset)
                            targets.extend(prefix_targets)
                        else:
                            targets.append(target)
                    for reward, dt, executed in executed_steps:
                        history.append(int(executed))
                        step_rewards.append(float(reward))
                        window_actions.append(int(executed))
                        base_action, _ = xs_decode_action(int(executed), MAXT)
                        if args.forbid_retrack_within_window and int(base_action) > 0:
                            window_forbidden.add(int(base_action))
                        window_reward += float(reward)
                        window_ms += float(dt)
                        if window_ms >= 200.0:
                            break
                    if args.plan_mode == "window":
                        break
                else:
                    if train and getattr(args, "self_play_sample_tau", 0.0) > 0.0:
                        sample_seed = stable_seed(getattr(args, "seed", 0), seed, window, len(history), "snapshot_sample")
                        action = sample_root_action(root, args.self_play_sample_tau, sample_seed)
                    elif str(args.select_mode) == "load_gated_prior":
                        action = choose_root_action_load_gated(
                            root,
                            sim._cache[()].obs,
                            int(getattr(args, "load_gated_prior_threshold", 80)),
                        )
                    else:
                        action = choose_root_action(root, args.select_mode)
                    # A single stale/busy model action should not terminate the
                    # episode. Execute the selected action first, then fall back
                    # through the current valid set within the remaining window.
                    fallback_actions = [int(a) for a in mcts.valid_actions(sim._cache[()].obs) if int(a) != int(action)]
                    fallback_actions.extend([MAXT + 1, MAXT + 2])
                    reward, dt, debt, executed = sim.commit_first_valid([int(action), *fallback_actions], 200.0 - window_ms)
                    if executed is None or dt <= 0.0:
                        debt += max(0.0, 200.0 - window_ms)
                        window_ms = 200.0
                        break
                    if target is not None and args.target_selected_action:
                        set_selected_action_target(target, int(action))
                    if target is not None and (max_train_targets <= 0 or len(targets) < max_train_targets):
                        target.reward = reward
                        target.initial = int(initial_targets)
                        target.rate = float(rate)
                        target.seed = int(seed)
                        target.window = int(window + 1)
                        target.action_index = int(len(history))
                        targets.append(target)
                    history.append(int(executed))
                    step_rewards.append(float(reward))
                    window_actions.append(int(executed))
                    window_reward += float(reward)
                    window_ms += float(dt)
            if window_ms <= 0.0 and not window_actions and not eng.term_buf[0]:
                debt += 200.0
                window_ms = 200.0
            cumulative += window_reward
            obs = get_obs(eng, debt)
            active = np.asarray(obs["active_mask"]).astype(bool)
            tracked = active & (np.asarray(obs["t_deadline"], dtype=np.float32) >= 0.0)
            dropped = active & (np.asarray(obs["t_deadline"], dtype=np.float32) < 0.0)
            rows.append(
                {
                    "window": window + 1,
                    "window_reward": float(window_reward),
                    "cumulative_reward": float(cumulative),
                    "window_ms_used": float(window_ms),
                    "actions": int(len(history)),
                    "search_fraction": float(np.mean([xs_decode_action(a, MAXT)[0] == 0 for a in window_actions])) if window_actions else 0.0,
                    **xs_action_fractions(window_actions, MAXT),
                    "active_targets": float(np.sum(active)),
                    "tracked_targets": float(np.sum(tracked)),
                    "drop_pct_active": float(100.0 * np.sum(dropped) / max(1, np.sum(active))) if np.any(active) else 0.0,
                    "mean_delay_active": float(np.mean(np.maximum(0.0, -obs["t_desired"][active]))) if np.any(active) else 0.0,
                }
            )
        suffix_returns = [0.0 for _ in range(len(step_rewards) + 1)]
        G = 0.0
        gamma = float(args.gamma)
        for idx in range(len(step_rewards) - 1, -1, -1):
            G = float(step_rewards[idx]) + gamma * G
            suffix_returns[idx] = G
        for target in targets:
            action_idx = int(getattr(target, "action_index", 0))
            if 0 <= action_idx < len(suffix_returns):
                target.ret = float(suffix_returns[action_idx])
            else:
                target.ret = float(target.reward)
        return pd.DataFrame(rows), targets
    finally:
        eng.close()


def train_exact(model: MutualRadarNet, args):
    replay = ReplayBuffer(args.replay_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    log = []
    accepted_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    accepted_score = -float("inf")
    if args.accept_gate:
        accepted_score = float(evaluate_gate(model.eval(), args, "gate_initial")["reward"])
        torch.save(accepted_state, RUN_OUT / "exact_mutual_accepted.pt")
    for it in range(1, args.iterations + 1):
        start_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        iter_targets: List[SearchTarget] = []
        rewards = []
        t0 = time.perf_counter()
        initials = [int(x) for x in args.train_initials.split(",") if x]
        rates = [float(x) for x in args.train_rates.split(",") if x]
        if args.train_grid:
            configs = [(init, rate, int(args.seed + it * 1000 + ep)) for ep in range(args.episodes_per_iter) for init in initials for rate in rates]
        else:
            configs = [
                (random.choice(initials), random.choice(rates), int(args.seed + it * 1000 + ep))
                for ep in range(args.episodes_per_iter)
            ]
        for init, rate, seed in configs:
            runner = run_snapshot_exact_episode if args.clone_mode == "snapshot" else run_exact_episode
            df, targets = runner(model.eval(), args, init, rate, seed, train=True)
            iter_targets.extend(targets)
            if not df.empty:
                rewards.append(float(df["window_reward"].mean()))
        replay.extend(iter_targets)
        abs_vals = [abs(x.ret) for x in replay.items]
        abs_vals.extend(abs(float(v)) for r in replay.items for v in r.q[r.q_mask > 0.5])
        q_scale = float(max(1.0, np.percentile(abs_vals, 90))) if abs_vals else 100.0
        metrics = []
        model.train()
        for _ in range(args.train_steps):
            if getattr(args, "branch_target", "standard") == "max":
                m = train_step_branch_max(
                    model,
                    opt,
                    replay,
                    args.batch_size,
                    q_scale,
                    args.policy_tau,
                    args.type_loss_weight,
                    args.rank_loss_weight,
                    args.value_loss_weight,
                    args.type_q_loss_weight,
                    args.track_q_loss_weight,
                )
            elif getattr(args, "branch_balanced_policy", False):
                m = train_step_branch_balanced(model, opt, replay, args.batch_size, q_scale, args.policy_tau)
            else:
                m = train_step(model, opt, replay, args.batch_size, q_scale)
            if m:
                metrics.append(m)
        model.eval()
        row = {
            "iteration": it,
            "targets": len(iter_targets),
            "replay": len(replay),
            "selfplay_reward": float(np.mean(rewards)) if rewards else 0.0,
            "q_scale": q_scale,
            "seconds": time.perf_counter() - t0,
        }
        if iter_targets:
            search_q = []
            best_track_q = []
            branch_pref = []
            search_pi = []
            for tgt in iter_targets:
                if float(tgt.q_mask[0]) <= 0.5:
                    continue
                track_mask = np.asarray(tgt.q_mask[1:] > 0.5)
                if not np.any(track_mask):
                    continue
                sq = float(tgt.q[0])
                bt = float(np.max(tgt.q[1:][track_mask]))
                search_q.append(sq)
                best_track_q.append(bt)
                branch_pref.append(1.0 if sq > bt else 0.0)
                search_pi.append(float(tgt.pi[0]))
            if search_q:
                row["cf_search_q_mean"] = float(np.mean(search_q))
                row["cf_best_track_q_mean"] = float(np.mean(best_track_q))
                row["cf_search_minus_track_mean"] = float(np.mean(np.asarray(search_q) - np.asarray(best_track_q)))
                row["cf_search_preferred_frac"] = float(np.mean(branch_pref))
                row["cf_search_pi_mean"] = float(np.mean(search_pi))
        if metrics:
            for k in metrics[0]:
                row[k] = float(np.mean([m[k] for m in metrics]))
        if args.accept_gate:
            candidate_score = float(evaluate_gate(model.eval(), args, f"gate_iter{it:03d}")["reward"])
            accept = candidate_score >= accepted_score + float(args.gate_min_delta)
            row["candidate_gate_reward"] = candidate_score
            row["accepted_gate_reward_before"] = accepted_score
            row["accepted"] = bool(accept)
            if accept:
                accepted_score = candidate_score
                accepted_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(accepted_state, RUN_OUT / "exact_mutual_accepted.pt")
            else:
                # Revert to the previous accepted model. This keeps exact MCTS
                # self-improvement monotone on the validation objective.
                model.load_state_dict(accepted_state)
                model.to(DEVICE)
        else:
            accepted_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if getattr(args, "eval_each_iter", False):
            eval_summary = evaluate_exact(model.eval(), args, f"iter{it:03d}")
            exact_rows = eval_summary[eval_summary["planner"].str.startswith("ExactEnvMutual")]
            if not exact_rows.empty:
                row["eval_reward"] = float(exact_rows["reward"].mean())
                row["eval_search_fraction"] = float(exact_rows["search"].mean())
                if "drop" in exact_rows:
                    row["eval_drop_pct_active"] = float(exact_rows["drop"].mean())
                if "tracked" in exact_rows:
                    row["eval_tracked_targets"] = float(exact_rows["tracked"].mean())
                row["eval_planning_ms"] = float(exact_rows["latency"].mean())
        log.append(row)
        print("exact_mutual_train", json.dumps(row), flush=True)
        torch.save(model.cpu().state_dict(), RUN_OUT / "exact_mutual_latest.pt")
        if getattr(args, "save_each_iter", False):
            torch.save(model.cpu().state_dict(), RUN_OUT / f"exact_mutual_iter{it:03d}.pt")
        model.to(DEVICE)
        pd.DataFrame(log).to_csv(RUN_OUT / "exact_mutual_train_log.csv", index=False)
    return model


def evaluate_gate(model: MutualRadarNet, args, tag: str) -> Dict[str, float]:
    old_windows = args.windows
    old_initials = args.eval_initials
    old_rates = args.eval_rates
    old_seeds = args.eval_seeds
    try:
        args.windows = int(args.gate_windows)
        args.eval_initials = str(args.gate_initials)
        args.eval_rates = str(args.gate_rates)
        args.eval_seeds = str(args.gate_seeds)
        summary = evaluate_exact(model, args, tag)
        row = summary[summary["planner"].str.startswith("ExactEnvMutual")].iloc[0].to_dict()
        return {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in row.items()}
    finally:
        args.windows = old_windows
        args.eval_initials = old_initials
        args.eval_rates = old_rates
        args.eval_seeds = old_seeds


class ArrivalAwarePlanner:
    def __init__(self, planner, env_cfg: Dict[str, float], enabled: bool):
        self.planner = planner
        self.env_cfg = env_cfg
        self.enabled = bool(enabled)

    def warmup(self, obs, budget_ms=200):
        obs2 = attach_env_obs(obs, self.env_cfg, self.enabled)
        if hasattr(self.planner, "warmup"):
            return self.planner.warmup(obs2, budget_ms=budget_ms)
        return self.planner.plan(obs2, budget_ms=budget_ms)

    def plan(self, obs, budget_ms=200):
        return self.planner.plan(attach_env_obs(obs, self.env_cfg, self.enabled), budget_ms=budget_ms)


def evaluate_exact(model: MutualRadarNet, args, tag: str):
    rows = []
    cells = [(int(i), float(r)) for i in args.eval_initials.split(",") for r in args.eval_rates.split(",")]
    seeds = [int(x) for x in args.eval_seeds.split(",") if x]
    for init, rate in cells:
        env_cfg = env_cfg_for(rate, args)
        for seed in seeds:
            t0 = time.perf_counter()
            runner = run_snapshot_exact_episode if args.clone_mode == "snapshot" else run_exact_episode
            df, _ = runner(model.eval(), args, init, rate, seed, train=False)
            latency = (time.perf_counter() - t0) * 1000.0 / max(1, len(df))
            s = {
                "planner": f"ExactEnvMutual_r{args.rollouts}",
                "initial_targets": init,
                "rate": rate,
                "seed": seed,
                "reward_per_200ms_eq": float(df["window_reward"].mean()) if not df.empty else 0.0,
                "total_reward": float(df["window_reward"].sum()) if not df.empty else 0.0,
                "mean_delay_active": float(df["mean_delay_active"].mean()) if not df.empty else 0.0,
                "search_fraction": float(df["search_fraction"].iloc[-1]) if not df.empty else 0.0,
                "s_search_fraction": float(df["s_search_fraction"].iloc[-1]) if "s_search_fraction" in df and not df.empty else 0.0,
                "x_search_fraction": float(df["x_search_fraction"].iloc[-1]) if "x_search_fraction" in df and not df.empty else 0.0,
                "s_track_fraction": float(df["s_track_fraction"].iloc[-1]) if "s_track_fraction" in df and not df.empty else 0.0,
                "x_track_fraction": float(df["x_track_fraction"].iloc[-1]) if "x_track_fraction" in df and not df.empty else 0.0,
                "mean_active_targets": float(df["active_targets"].mean()) if "active_targets" in df and not df.empty else 0.0,
                "mean_tracked_targets": float(df["tracked_targets"].mean()) if "tracked_targets" in df and not df.empty else 0.0,
                "mean_drop_pct_active": float(df["drop_pct_active"].mean()) if "drop_pct_active" in df and not df.empty else 0.0,
                "mean_window_ms_used": float(df["window_ms_used"].mean()) if "window_ms_used" in df and not df.empty else 0.0,
                "planning_ms_per_200ms_eq": latency,
            }
            rows.append(s)
            if not getattr(args, "compact_output", False):
                df.to_csv(RUN_OUT / f"{tag}_exact_windows_init{init}_rate{rate}_seed{seed}.csv", index=False)
            if not args.skip_direct_eval:
                for name, planner in [
                    (
                        "DirectPolicy",
                        ArrivalAwarePlanner(
                            MutualRadarDirectPlanner(
                                model,
                                alpha=0.0,
                                beta=0.0,
                                threshold=args.direct_threshold,
                                direct_mode=args.direct_mode,
                                allow_retrack=args.direct_allow_retrack,
                                stateless_context=args.direct_stateless_context,
                                cache_encoder=args.direct_cache_encoder,
                                sensor_action_mode=args.sensor_action_mode,
                                disable_x_search=args.disable_x_search,
                            ),
                            env_cfg,
                            bool(getattr(args, "use_arrival_feature", False)),
                        ),
                    ),
                    (
                        "DirectPolicyQ",
                        ArrivalAwarePlanner(
                            MutualRadarDirectPlanner(
                                model,
                                alpha=args.direct_q_alpha,
                                beta=args.direct_q_beta,
                                threshold=args.direct_threshold,
                                direct_mode=args.direct_mode,
                                allow_retrack=args.direct_allow_retrack,
                                stateless_context=args.direct_stateless_context,
                                cache_encoder=args.direct_cache_encoder,
                                sensor_action_mode=args.sensor_action_mode,
                                disable_x_search=args.disable_x_search,
                            ),
                            env_cfg,
                            bool(getattr(args, "use_arrival_feature", False)),
                        ),
                    ),
                ]:
                    w, _ = run_fixed(planner, name, init, MAXT, seed, int(args.windows), 200, engine_env_cfg(env_cfg))
                    ds = summarize_window_df(w, "fixed")
                    ds.update(planner=name, initial_targets=init, rate=rate, seed=seed)
                    rows.append(ds)
            for name, planner in [("EDF", EDFPlanner(MAXT)), ("EST", ESTPlanner(MAXT))]:
                w, _ = run_fixed(planner, name, init, MAXT, seed, int(args.windows), 200, engine_env_cfg(env_cfg))
                bs = summarize_window_df(w, "fixed")
                bs.update(planner=name, initial_targets=init, rate=rate, seed=seed)
                rows.append(bs)
    raw = pd.DataFrame(rows)
    raw.to_csv(RUN_OUT / f"{tag}_eval_raw.csv", index=False)
    summary = raw.groupby("planner").agg(
        reward=("reward_per_200ms_eq", "mean"),
        delay=("mean_delay_active", "mean"),
        search=("search_fraction", "mean"),
        s_search=("s_search_fraction", "mean"),
        x_search=("x_search_fraction", "mean"),
        s_track=("s_track_fraction", "mean"),
        x_track=("x_track_fraction", "mean"),
        tracked=("mean_tracked_targets", "mean"),
        drop=("mean_drop_pct_active", "mean"),
        util=("mean_window_ms_used", "mean"),
        latency=("planning_ms_per_200ms_eq", "mean"),
    ).reset_index().sort_values("reward", ascending=False)
    summary.to_csv(RUN_OUT / f"{tag}_eval_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    return summary


def load_model(args) -> MutualRadarNet:
    model = MutualRadarNet(d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers, head_arch=args.head_arch)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        current = model.state_dict()
        compatible = {}
        padded = []
        for k, v in state.items():
            if k not in current:
                continue
            if tuple(current[k].shape) == tuple(v.shape):
                compatible[k] = v
                continue
            # Backward-compatible feature extension: old checkpoints used
            # fewer token/slot features.  Preserve their learned weights and
            # initialize only the newly added feature columns neutrally.
            if k in {"token_proj.weight", "slot_proj.1.weight"} and v.ndim == 2 and current[k].ndim == 2:
                if v.shape[0] == current[k].shape[0] and v.shape[1] < current[k].shape[1]:
                    nv = current[k].clone()
                    nv.zero_()
                    nv[:, : v.shape[1]] = v
                    compatible[k] = nv
                    padded.append(k)
                    continue
            if k == "slot_proj.0.weight" and v.ndim == 1 and current[k].ndim == 1:
                if v.shape[0] < current[k].shape[0]:
                    nv = torch.ones_like(current[k])
                    nv[: v.shape[0]] = v
                    compatible[k] = nv
                    padded.append(k)
                    continue
            if k == "slot_proj.0.bias" and v.ndim == 1 and current[k].ndim == 1:
                if v.shape[0] < current[k].shape[0]:
                    nv = torch.zeros_like(current[k])
                    nv[: v.shape[0]] = v
                    compatible[k] = nv
                    padded.append(k)
                    continue
        skipped = sorted(set(state.keys()) - set(compatible.keys()))
        model.load_state_dict(compatible, strict=False)
        if padded:
            print(f"load_model padded {len(padded)} extended feature keys: {', '.join(padded)}", flush=True)
        if skipped:
            print(f"load_model skipped {len(skipped)} incompatible keys for head_arch={args.head_arch}", flush=True)
    target_device = getattr(args, "device", "auto")
    if target_device == "auto":
        target_device = str(DEVICE)
    model.to(torch.device(target_device)).eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["diagnose", "train", "train_eval"], default="diagnose")
    ap.add_argument("--seed", type=int, default=76)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--episodes-per-iter", type=int, default=1)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--max-targets-per-episode", type=int, default=16)
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--horizon-windows", type=int, default=2)
    ap.add_argument("--expand-top-k", type=int, default=8)
    ap.add_argument("--c-puct", type=float, default=1.25)
    ap.add_argument("--epsilon", type=float, default=0.10)
    ap.add_argument("--rollout-policy", choices=["model", "branch", "branch_margin", "q", "pq", "random", "value", "edge", "edf", "est", "mixed"], default="model")
    ap.add_argument("--branch-rollout-threshold", type=float, default=0.65)
    ap.add_argument("--seed-rollout-policies", default="")
    ap.add_argument("--clone-mode", choices=["snapshot", "replay"], default="snapshot")
    ap.add_argument("--plan-mode", choices=["atomic", "window", "first_window"], default="atomic")
    ap.add_argument("--window-extract", choices=["best", "tree", "tree_fill", "greedy_expand", "batched_value", "model_q", "edge_q"], default="best")
    ap.add_argument("--allow-retrack-in-window", action="store_true")
    ap.add_argument("--forbid-retrack-within-window", action="store_true")
    ap.add_argument("--stateless-tree-context", action="store_true")
    ap.add_argument("--head-mode", choices=["p", "pq", "pv", "pvq"], default="p")
    ap.add_argument("--q-utility-weight", type=float, default=0.0)
    ap.add_argument("--q-utility-normalize", action="store_true")
    ap.add_argument("--leaf-value-mix", type=float, default=1.0)
    ap.add_argument("--select-mode", choices=["visits", "q", "prior", "branch_visits", "branch_q", "load_gated_prior"], default="visits")
    ap.add_argument("--load-gated-prior-threshold", type=int, default=80)
    ap.add_argument("--self-play-sample-tau", type=float, default=0.0)
    ap.add_argument("--policy-target", choices=["visits", "q_softmax", "branch_q_softmax", "branch_future_softmax", "mixed", "mctx"], default="q_softmax")
    ap.add_argument("--policy-tau", type=float, default=1.0)
    ap.add_argument("--branch-balanced-policy", action="store_true")
    ap.add_argument("--branch-target", choices=["standard", "logsum", "max"], default="standard")
    ap.add_argument("--counterfactual-branch-q", action="store_true")
    ap.add_argument("--counterfactual-top-k", type=int, default=8)
    ap.add_argument("--counterfactual-mode", choices=["value", "rollout", "model_rollout", "subtree", "edge_density", "edge_greedy_rollout"], default="value")
    ap.add_argument("--counterfactual-candidate-mode", choices=["prior", "urgent"], default="prior")
    ap.add_argument("--counterfactual-subrollouts", type=int, default=8)
    ap.add_argument("--type-loss-weight", type=float, default=1.0)
    ap.add_argument("--rank-loss-weight", type=float, default=1.0)
    ap.add_argument("--value-loss-weight", type=float, default=0.5)
    ap.add_argument("--type-q-loss-weight", type=float, default=0.25)
    ap.add_argument("--track-q-loss-weight", type=float, default=0.5)
    ap.add_argument("--search-alg", choices=["puct", "gumbel", "hierarchical"], default="puct")
    ap.add_argument("--max-num-considered-actions", type=int, default=16)
    ap.add_argument("--gumbel-scale", type=float, default=0.0)
    ap.add_argument("--mctx-value-scale", type=float, default=0.1)
    ap.add_argument("--mctx-maxvisit-init", type=float, default=50.0)
    ap.add_argument("--eager-edge-depth", type=int, default=1)
    ap.add_argument("--prior-uniform-mix", type=float, default=0.0)
    ap.add_argument("--root-dirichlet-alpha", type=float, default=0.0)
    ap.add_argument("--root-dirichlet-frac", type=float, default=0.0)
    ap.add_argument("--rollout-est-prob", type=float, default=0.5)
    ap.add_argument("--prior-mode", choices=["factorized", "flat", "branch_corrected", "physical_flat", "true_physical_flat"], default="factorized")
    ap.add_argument("--direct-mode", choices=["prob", "flat", "branch", "q"], default="branch")
    ap.add_argument("--direct-threshold", type=float, default=0.5)
    ap.add_argument("--direct-q-alpha", type=float, default=1.0)
    ap.add_argument("--direct-q-beta", type=float, default=1.0)
    ap.add_argument("--direct-allow-retrack", action="store_true")
    ap.add_argument("--direct-stateless-context", action="store_true")
    ap.add_argument("--direct-cache-encoder", action="store_true")
    ap.add_argument("--skip-direct-eval", action="store_true")
    ap.add_argument("--accept-gate", action="store_true")
    ap.add_argument("--gate-min-delta", type=float, default=0.0)
    ap.add_argument("--gate-windows", type=int, default=1)
    ap.add_argument("--gate-initials", default="15,50")
    ap.add_argument("--gate-rates", default="0,2")
    ap.add_argument("--gate-seeds", default="100")
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--head-arch", choices=["baseline", "specialized", "branch_context"], default="baseline")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--replay-size", type=int, default=50000)
    ap.add_argument("--train-steps", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train-initials", default="15,50")
    ap.add_argument("--train-rates", default="0,2")
    ap.add_argument("--train-grid", action="store_true")
    ap.add_argument("--add-prefix-targets", action="store_true")
    ap.add_argument("--target-selected-action", action="store_true")
    ap.add_argument("--eval-initials", default="15,50")
    ap.add_argument("--eval-rates", default="0,2")
    ap.add_argument("--eval-seeds", default="100")
    ap.add_argument(
        "--env-mode",
        choices=[
            "current",
            "no_refresh",
            "operational",
            "original_reward",
            "radarxs_original",
            "radarxs_original_global",
            "radarxs_balanced",
            "radarxs_mission_delta",
            "penalty_only_frame",
            "repaired_stress_reward",
            "balanced_linear",
            "staleness_potential",
            "searched_sector_frame",
            "ding_moo_frame",
            "mcts_sched_v1",
        ],
        default="searched_sector_frame",
    )
    ap.add_argument("--track-update-reward", type=float, default=0.30)
    ap.add_argument("--track-loss-penalty", type=float, default=4.0)
    ap.add_argument("--track-urgency-bonus-weight", type=float, default=-1.0)
    ap.add_argument("--target-service-weight", type=float, default=1.0)
    ap.add_argument("--target-service-horizon-ms", type=float, default=1000.0)
    ap.add_argument("--search-refresh-tracked", type=int, default=0)
    ap.add_argument("--search-refresh-gain", type=float, default=0.0)
    ap.add_argument("--enable-x-band", action="store_true")
    ap.add_argument("--use-arrival-feature", action="store_true")
    ap.add_argument("--sensor-action-mode", choices=["implicit", "explicit", "explicit_head"], default="implicit")
    ap.add_argument("--disable-x-search", action="store_true")
    ap.add_argument("--canonical-search-only", action="store_true")
    ap.add_argument("--search-debt-penalty-weight", type=float, default=0.0)
    ap.add_argument("--sector-staleness-weight", type=float, default=0.0)
    ap.add_argument("--searched-sector-reward-weight", type=float, default=0.25)
    ap.add_argument("--search-frame-overdue-weight", type=float, default=0.05)
    ap.add_argument("--search-frame-desired-ms", type=float, default=3000.0)
    ap.add_argument("--search-frame-deadline-ms", type=float, default=4500.0)
    ap.add_argument("--search-frame-drop-penalty", type=float, default=4.0)
    ap.add_argument("--penalize-hidden-targets", type=int, default=1)
    ap.add_argument("--compact-output", action="store_true")
    ap.add_argument("--eval-each-iter", action="store_true")
    ap.add_argument("--save-each-iter", action="store_true")
    ap.add_argument("--fast-zero-rollout", action="store_true")
    ap.add_argument("--skip-default-rollout-seed", action="store_true")
    ap.add_argument("--complete-root-q-with-value", action="store_true")
    ap.add_argument("--visit-unvisited-first", action="store_true")
    ap.add_argument("--duration-normalize-q", action="store_true")
    ap.add_argument("--prior-q-beta", type=float, default=0.0)
    ap.add_argument("--prior-search-bias", type=float, default=0.0)
    args = ap.parse_args()
    seedall(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = load_model(args)
    if args.mode in {"train", "train_eval"}:
        model = train_exact(model, args)
    if args.mode in {"diagnose", "train_eval"}:
        evaluate_exact(model, args, "diagnose" if args.mode == "diagnose" else "train_eval")


if __name__ == "__main__":
    main()
