from __future__ import annotations

import argparse
import math
import random
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "CreateValid1" / "experiments" / "code" / "model_code"
MUZERO = CODE / "third_party" / "muzero-general"
for p in (CODE, MUZERO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import ray  # noqa: F401
except Exception:
    fake_ray = types.ModuleType("ray")

    def _remote(obj=None, **_kwargs):
        if obj is None:
            return lambda x: x
        return obj

    fake_ray.remote = _remote
    sys.modules["ray"] = fake_ray

import models  # type: ignore
from self_play import GameHistory, MCTS, MinMaxStats, Node  # type: ignore

from exact_env_mutual import EDFPlanner, ESTPlanner, MAXT, _DummyPlanner, attach_env_obs, engine_env_cfg, env_cfg_for, shaped_step_reward
from final_radar_campaign import get_obs
from penalty_window_quota_learner_eval import make_exact_args
from repaired_campaign_tools import build_env, execute_first_valid_action
from strict_window_report import sample_state_metrics


OBS_TARGETS = MAXT
OBS_DIM = 10 + OBS_TARGETS * 7


def parse_int_list(text: str, default: list[int]) -> list[int]:
    values = [int(x) for x in str(text).split(",") if x.strip()]
    return values or list(default)


def parse_float_list(text: str, default: list[float]) -> list[float]:
    values = [float(x) for x in str(text).split(",") if x.strip()]
    return values or list(default)


def curriculum_cells(args) -> list[tuple[int, float]]:
    initials = parse_int_list(getattr(args, "initials", ""), [int(args.initial)])
    rates = parse_float_list(getattr(args, "rates", ""), [float(args.rate)])
    return [(int(initial), float(rate)) for initial in initials for rate in rates]


def set_cell(args, cell: tuple[int, float]) -> None:
    args.initial = int(cell[0])
    args.rate = float(cell[1])


def s_search() -> int:
    return MAXT + 3


def s_track(row: int) -> int:
    return MAXT + 5 + int(row) - 1


def ranked_rows(obs: dict) -> list[int]:
    active = np.asarray(obs.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
    deadline = np.asarray(obs.get("t_deadline", np.full(MAXT, 1e9, dtype=np.float32)), dtype=np.float32)
    desired = np.asarray(obs.get("t_desired", np.full(MAXT, 1e9, dtype=np.float32)), dtype=np.float32)
    dwell = np.asarray(obs.get("t_dwell", np.ones(MAXT, dtype=np.float32) * 10.0), dtype=np.float32)
    valid = active & (deadline >= 0.0)
    rows = np.nonzero(valid)[0] + 1
    if rows.size == 0:
        return []
    score = deadline[rows - 1] + 0.35 * desired[rows - 1] + 0.02 * dwell[rows - 1]
    return [int(r) for r in rows[np.argsort(score)]]


def valid_target_rows(obs: dict) -> list[int]:
    active = np.asarray(obs.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
    deadline = np.asarray(obs.get("t_deadline", np.full(MAXT, 1e9, dtype=np.float32)), dtype=np.float32)
    rows = np.nonzero(active & (deadline >= 0.0))[0] + 1
    return [int(r) for r in rows]


def discrete_to_s_action(action: int, obs: dict, rank_action_space: bool = False) -> int:
    action = int(action)
    if action <= 0:
        return s_search()
    if rank_action_space:
        rows = ranked_rows(obs)
        if 1 <= action <= len(rows):
            return s_track(rows[action - 1])
        return s_search()
    rows = set(valid_target_rows(obs))
    if action not in rows:
        return s_search()
    return s_track(action)


def observation_vector(obs: dict, debt: float, window_spent_ms: float, window_index: int, total_windows: int) -> np.ndarray:
    active = np.asarray(obs.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
    deadline = np.asarray(obs.get("t_deadline", np.full(MAXT, 1e9, dtype=np.float32)), dtype=np.float32)
    desired = np.asarray(obs.get("t_desired", np.full(MAXT, 1e9, dtype=np.float32)), dtype=np.float32)
    dwell = np.asarray(obs.get("t_dwell", np.ones(MAXT, dtype=np.float32) * 10.0), dtype=np.float32)
    priority = np.asarray(obs.get("priority", np.ones(MAXT, dtype=np.float32)), dtype=np.float32)
    rng = np.asarray(obs.get("target_range", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)
    rows = ranked_rows(obs)[:OBS_TARGETS]
    tracked = active & (deadline >= 0.0)
    base = [
        float(np.sum(active)) / 100.0,
        float(np.sum(tracked)) / 100.0,
        float(debt) / 3000.0,
        float(obs.get("enable_x_band", 1.0)),
        float(np.min(deadline[tracked])) / 3000.0 if np.any(tracked) else 0.0,
        float(np.mean(np.maximum(0.0, -desired[tracked]) / 3000.0)) if np.any(tracked) else 0.0,
        float(np.mean(dwell[tracked]) / 50.0) if np.any(tracked) else 0.0,
        1.0,
        float(window_spent_ms) / 200.0,
        float(window_index) / max(1.0, float(total_windows)),
    ]
    feats = []
    for r in rows:
        j = r - 1
        feats.extend(
            [
                1.0,
                float(np.clip(deadline[j] / 3000.0, -2.0, 4.0)),
                float(np.clip(desired[j] / 3000.0, -2.0, 4.0)),
                float(np.clip(dwell[j] / 50.0, 0.0, 4.0)),
                float(np.clip(priority[j] / 5.0, 0.0, 2.0)),
                float(np.clip(rng[j] / 184_000_000.0, 0.0, 2.0)),
                float(r) / float(MAXT),
            ]
        )
    while len(feats) < OBS_TARGETS * 7:
        feats.extend([0.0] * 7)
    vec = np.asarray(base + feats[: OBS_TARGETS * 7], dtype=np.float32)
    return vec.reshape(1, 1, OBS_DIM)


@dataclass
class RewardShaping:
    env_mode: str = "radarxs_original"
    track_update_reward: float = 0.30
    track_loss_penalty: float = 4.0
    searched_sector_reward_weight: float = 0.0
    search_frame_overdue_weight: float = 0.05
    search_frame_desired_ms: float = 3000.0
    search_frame_deadline_ms: float = 4500.0
    search_frame_drop_penalty: float = 4.0
    search_frame_state_penalty_weight: float = 0.0
    search_frame_delta_reward_weight: float = 0.0
    service_pressure_delta_reward_weight: float = 0.0
    serviced_pressure_delta_reward_weight: float = 0.0
    serviced_pressure_improvement_reward_weight: float = 0.0
    discovered_target_reward: float = 0.0
    tracked_count_delta_reward_weight: float = 0.0
    tracked_target_ms_reward_weight: float = 0.0
    service_reward: float = 0.0
    discovery_reward: float = 0.0
    search_refresh_reward: float = 0.0
    search_debt_penalty: float = 0.0
    search_debt_penalty_mode: str = "step"
    terminal_search_debt_penalty: float = 0.0
    window_service_reward: float = 0.0
    window_tracked_reward: float = 0.0
    window_on_time_reward: float = 0.0
    window_on_time_ratio_reward: float = 0.0
    penalize_hidden_targets: int = 0


class LiteralRadarGame:
    def __init__(
        self,
        initial: int,
        rate: float,
        seed: int,
        windows: int,
        reward_scale: float,
        action_ranks: int,
        shaping: RewardShaping | None = None,
        rank_action_space: bool = False,
    ):
        self.initial = int(initial)
        self.rate = float(rate)
        self.seed = int(seed)
        self.windows = int(windows)
        self.reward_scale = float(reward_scale)
        self.action_ranks = int(action_ranks)
        self.shaping = shaping or RewardShaping()
        self.rank_action_space = bool(rank_action_space)
        self.exact_args = make_exact_args(
            argparse.Namespace(
                windows=windows,
                env_mode=str(getattr(self.shaping, "env_mode", "radarxs_original")),
                track_update_reward=float(getattr(self.shaping, "track_update_reward", 0.30)),
                track_loss_penalty=float(getattr(self.shaping, "track_loss_penalty", 4.0)),
                track_urgency_bonus_weight=-1.0,
                target_service_weight=0.0,
                target_service_horizon_ms=1000.0,
                search_refresh_tracked=0,
                search_refresh_gain=0.0,
                search_debt_penalty_weight=0.0,
                sector_staleness_weight=0.0,
                searched_sector_reward_weight=float(getattr(self.shaping, "searched_sector_reward_weight", 0.0)),
                search_frame_overdue_weight=float(getattr(self.shaping, "search_frame_overdue_weight", 0.05)),
                search_frame_desired_ms=float(getattr(self.shaping, "search_frame_desired_ms", 3000.0)),
                search_frame_deadline_ms=float(getattr(self.shaping, "search_frame_deadline_ms", 4500.0)),
                search_frame_drop_penalty=float(getattr(self.shaping, "search_frame_drop_penalty", 4.0)),
                search_frame_state_penalty_weight=float(getattr(self.shaping, "search_frame_state_penalty_weight", 0.0)),
                search_frame_delta_reward_weight=float(getattr(self.shaping, "search_frame_delta_reward_weight", 0.0)),
                service_pressure_delta_reward_weight=float(getattr(self.shaping, "service_pressure_delta_reward_weight", 0.0)),
                serviced_pressure_delta_reward_weight=float(getattr(self.shaping, "serviced_pressure_delta_reward_weight", 0.0)),
                serviced_pressure_improvement_reward_weight=float(getattr(self.shaping, "serviced_pressure_improvement_reward_weight", 0.0)),
                discovered_target_reward=float(getattr(self.shaping, "discovered_target_reward", 0.0)),
                tracked_count_delta_reward_weight=float(getattr(self.shaping, "tracked_count_delta_reward_weight", 0.0)),
                tracked_target_ms_reward_weight=float(getattr(self.shaping, "tracked_target_ms_reward_weight", 0.0)),
                penalize_hidden_targets=int(getattr(shaping, "penalize_hidden_targets", 0)),
                disable_x_search=False,
            )
        )
        self.exact_args.enable_x_band = False
        self.exact_args.single_sensor = True
        self.env_cfg = env_cfg_for(self.rate, self.exact_args)
        self.env_cfg["enable_x_band"] = 0
        self.eng = None
        self.debt = 0.0
        self.step_count = 0
        self.search_count = 0
        self.track_count = 0
        self.shaping_total = 0.0
        self.window_index = 0
        self.window_spent_ms = 0.0

    def reset(self):
        if self.eng is not None:
            self.eng.close()
        self.eng = build_env(_DummyPlanner(), self.initial, MAXT, self.seed, 200, engine_env_cfg(self.env_cfg))
        self.eng.reset(seed=self.seed)
        self.debt = 0.0
        self.step_count = 0
        self.search_count = 0
        self.track_count = 0
        self.shaping_total = 0.0
        self.window_index = 0
        self.window_spent_ms = 0.0
        return self._obs()

    def _obs(self):
        assert self.eng is not None
        obs = attach_env_obs(get_obs(self.eng, self.debt), self.env_cfg, True, True)
        return observation_vector(obs, self.debt, self.window_spent_ms, self.window_index, self.windows)

    def obs_dict(self) -> dict:
        assert self.eng is not None
        return attach_env_obs(get_obs(self.eng, self.debt), self.env_cfg, True, True)

    def legal_actions(self):
        assert self.eng is not None
        obs = attach_env_obs(get_obs(self.eng, self.debt), self.env_cfg, True, True)
        remaining = max(0.0, 200.0 - float(self.window_spent_ms))
        dwell = np.asarray(obs.get("t_dwell", np.ones(MAXT, dtype=np.float32) * 10.0), dtype=np.float32)
        def fits(row: int) -> bool:
            j = int(row) - 1
            if j < 0 or j >= int(dwell.shape[0]):
                return False
            return float(dwell[j]) <= remaining
        if self.rank_action_space:
            rows = [r for r in ranked_rows(obs) if fits(r)][: self.action_ranks]
            actions = list(range(1, len(rows) + 1))
        else:
            actions = [r for r in valid_target_rows(obs) if r <= self.action_ranks and fits(r)]
        if remaining >= 10.0:
            return [0] + actions
        return actions or [0]

    def to_play(self):
        return 0

    def step(self, action: int):
        assert self.eng is not None
        obs = attach_env_obs(get_obs(self.eng, self.debt), self.env_cfg, True, True)
        active_before = np.asarray(obs.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
        deadline_before = np.asarray(obs.get("t_deadline", np.full(MAXT, -1.0, dtype=np.float32)), dtype=np.float32)
        old_debt = float(self.debt)
        radar_action = discrete_to_s_action(int(action), obs, self.rank_action_space)
        remaining = max(1.0, 200.0 - float(self.window_spent_ms))
        reward, dt, executed = execute_first_valid_action(self.eng, [radar_action, s_search()], remaining)
        if executed is None or dt <= 0.0:
            reward, dt, executed = execute_first_valid_action(self.eng, [s_search()], remaining)
        executed_i = int(executed or s_search())
        is_search = executed_i == s_search()
        if is_search:
            self.search_count += 1
        else:
            self.track_count += 1
        self.debt = 0.0 if is_search else self.debt + float(dt)
        shaped = 0.0
        if not is_search and self.shaping.service_reward:
            row = int(executed_i - (MAXT + 5) + 1)
            if 1 <= row <= MAXT:
                j = row - 1
                if bool(active_before[j]) and float(deadline_before[j]) >= float(dt):
                    shaped += float(self.shaping.service_reward)
        if is_search and self.shaping.search_refresh_reward:
            shaped += float(self.shaping.search_refresh_reward) * float(np.clip(old_debt / 3000.0, 0.0, 1.0))
        if is_search and self.shaping.discovery_reward:
            obs_after = attach_env_obs(get_obs(self.eng, self.debt), self.env_cfg, True, True)
            active_after = np.asarray(obs_after.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
            newly_active = int(np.sum(active_after & ~active_before))
            shaped += float(self.shaping.discovery_reward) * float(newly_active)
        if self.shaping.search_debt_penalty and self.shaping.search_debt_penalty_mode == "step":
            shaped -= float(self.shaping.search_debt_penalty) * float(np.clip(self.debt / 3000.0, 0.0, 2.0))
        if self.shaping.search_debt_penalty and self.shaping.search_debt_penalty_mode == "time":
            shaped -= (
                float(self.shaping.search_debt_penalty)
                * float(np.clip(self.debt / 3000.0, 0.0, 2.0))
                * float(max(dt, 0.0) / 200.0)
            )
        obs_after_for_reward = attach_env_obs(get_obs(self.eng, self.debt), self.env_cfg, True, True)
        reward = shaped_step_reward(float(reward), float(dt), obs, obs_after_for_reward, self.env_cfg, action=int(executed_i))
        self.window_spent_ms += float(dt)
        self.step_count += 1
        if self.window_spent_ms >= 200.0:
            if self.shaping.search_debt_penalty and self.shaping.search_debt_penalty_mode == "window":
                shaped -= float(self.shaping.search_debt_penalty) * float(np.clip(self.debt / 3000.0, 0.0, 2.0))
            if self.shaping.window_service_reward:
                obs_after = attach_env_obs(get_obs(self.eng, self.debt), self.env_cfg, True, True)
                active_after = np.asarray(obs_after.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
                deadline_after = np.asarray(obs_after.get("t_deadline", np.full(MAXT, -1.0, dtype=np.float32)), dtype=np.float32)
                active_n = int(np.sum(active_after))
                if active_n:
                    tracked_n = int(np.sum(active_after & (deadline_after >= 0.0)))
                    shaped += float(self.shaping.window_service_reward) * float(tracked_n) / float(active_n)
            if self.shaping.window_tracked_reward:
                obs_after = attach_env_obs(get_obs(self.eng, self.debt), self.env_cfg, True, True)
                active_after = np.asarray(obs_after.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
                deadline_after = np.asarray(obs_after.get("t_deadline", np.full(MAXT, -1.0, dtype=np.float32)), dtype=np.float32)
                tracked_n = int(np.sum(active_after & (deadline_after >= 0.0)))
                shaped += float(self.shaping.window_tracked_reward) * float(tracked_n)
            if self.shaping.window_on_time_reward:
                obs_after = attach_env_obs(get_obs(self.eng, self.debt), self.env_cfg, True, True)
                active_after = np.asarray(obs_after.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
                deadline_after = np.asarray(obs_after.get("t_deadline", np.full(MAXT, -1.0, dtype=np.float32)), dtype=np.float32)
                desired_after = np.asarray(obs_after.get("t_desired", np.full(MAXT, -1.0, dtype=np.float32)), dtype=np.float32)
                on_time_n = int(np.sum(active_after & (deadline_after >= 0.0) & (desired_after >= 0.0)))
                shaped += float(self.shaping.window_on_time_reward) * float(on_time_n)
            if self.shaping.window_on_time_ratio_reward:
                obs_after = attach_env_obs(get_obs(self.eng, self.debt), self.env_cfg, True, True)
                active_after = np.asarray(obs_after.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
                deadline_after = np.asarray(obs_after.get("t_deadline", np.full(MAXT, -1.0, dtype=np.float32)), dtype=np.float32)
                desired_after = np.asarray(obs_after.get("t_desired", np.full(MAXT, -1.0, dtype=np.float32)), dtype=np.float32)
                active_n = int(np.sum(active_after))
                if active_n:
                    on_time_n = int(np.sum(active_after & (deadline_after >= 0.0) & (desired_after >= 0.0)))
                    shaped += float(self.shaping.window_on_time_ratio_reward) * float(on_time_n) / float(active_n)
            self.window_index += 1
            self.window_spent_ms = 0.0
        done = self.window_index >= self.windows or bool(self.eng.term_buf[0])
        if done and self.shaping.terminal_search_debt_penalty:
            shaped -= float(self.shaping.terminal_search_debt_penalty) * float(np.clip(self.debt / 3000.0, 0.0, 2.0))
        reward = float(reward) + shaped
        self.shaping_total += shaped
        return self._obs(), float(reward) / self.reward_scale, bool(done)

    def metrics(self):
        assert self.eng is not None
        metrics = sample_state_metrics(self.eng, self.debt)
        metrics["search_actions"] = int(self.search_count)
        metrics["track_actions"] = int(self.track_count)
        metrics["search_ratio"] = float(self.search_count) / max(1, int(self.search_count + self.track_count))
        metrics["shaping_reward_total"] = float(self.shaping_total)
        return metrics

    def close(self):
        if self.eng is not None:
            self.eng.close()
            self.eng = None


class Config:
    def __init__(self, args):
        self.seed = int(args.seed)
        self.max_num_gpus = 0
        self.observation_shape = (1, 1, OBS_DIM)
        self.action_space = list(range(int(args.action_ranks) + 1))
        self.players = [0]
        self.stacked_observations = 0
        self.muzero_player = 0
        self.opponent = None
        self.num_simulations = int(args.simulations)
        self.discount = float(args.gamma)
        self.td_steps = int(getattr(args, "td_steps", 10))
        self.value_target_mode = str(getattr(args, "value_target_mode", "td"))
        self.root_dirichlet_alpha = float(getattr(args, "root_dirichlet_alpha", 0.25))
        self.root_exploration_fraction = float(getattr(args, "root_exploration_fraction", 0.25))
        self.pb_c_base = float(getattr(args, "pb_c_base", 19652.0))
        self.pb_c_init = float(getattr(args, "pb_c_init", 1.25))
        self.network = str(getattr(args, "network", "fullyconnected"))
        self.factorized_search_logit_offset = float(getattr(args, "factorized_search_logit_offset", 0.0))
        self.factorized_puct_prior = str(getattr(args, "factorized_puct_prior", "standard"))
        self.legal_head = bool(getattr(args, "legal_head", False))
        self.legal_mask_loss_weight = float(getattr(args, "legal_mask_loss_weight", 0.0))
        self.support_size = int(args.support_size)
        self.encoding_size = int(args.encoding_size)
        self.fc_representation_layers = [128, 128]
        self.fc_dynamics_layers = [128]
        self.fc_reward_layers = [64]
        self.fc_value_layers = [64]
        self.fc_policy_layers = [64]
        self.value_loss_weight = 0.25
        self.PER = False


class RadarMCTS(MCTS):
    """MuZero MCTS with learned recurrent legality for radar actions.

    The reference MuZero implementation expands every recurrent node with the
    full fixed action space. That is fine for board games, but radar actions
    become invalid as the window budget and target set change. If the model has
    a legal head, use it to restrict latent child expansion after recurrent
    inference while keeping the root constrained by the true environment.
    """

    def _latent_legal_actions(self, model, hidden_state, fallback_actions: list[int]) -> list[int]:
        if not hasattr(model, "legal_prediction"):
            return fallback_actions
        legal_logits = model.legal_prediction(hidden_state)
        if legal_logits is None:
            return fallback_actions
        scores = legal_logits[0].detach().float()
        legal = torch.nonzero(scores >= 0.0, as_tuple=False).flatten().tolist()
        legal = [int(a) for a in legal if int(a) in fallback_actions]
        if 0 in fallback_actions and 0 not in legal:
            legal.append(0)
        if legal:
            return sorted(set(legal))
        k = min(8, len(fallback_actions))
        top = torch.topk(scores, k=k).indices.detach().cpu().tolist()
        legal = [int(a) for a in top if int(a) in fallback_actions]
        if 0 in fallback_actions and 0 not in legal:
            legal.append(0)
        return sorted(set(legal)) or fallback_actions

    def _factorized_probs(self, model, node: Node):
        if not hasattr(model, "prediction_type_network") or not hasattr(model, "prediction_target_network"):
            return None, None
        if node.hidden_state is None:
            return None, None
        with torch.no_grad():
            type_probs = torch.softmax(model.prediction_type_network(node.hidden_state), dim=1)[0].detach().float()
            target_probs = torch.softmax(model.prediction_target_network(node.hidden_state), dim=1)[0].detach().float()
        return type_probs, target_probs

    def _apply_factorized_priors(self, model, node: Node) -> None:
        mode = str(getattr(self.config, "factorized_puct_prior", "standard"))
        if mode not in {"type_balanced", "hierarchical"}:
            return
        if node.hidden_state is None or not node.children:
            return
        type_probs, target_probs = self._factorized_probs(model, node)
        if type_probs is None or target_probs is None:
            return
        track_actions = [int(a) for a in node.children if int(a) > 0 and int(a) - 1 < int(target_probs.shape[0])]
        if not track_actions:
            return
        track_norm = sum(float(target_probs[int(a) - 1].item()) for a in track_actions)
        track_norm = max(track_norm, 1e-8)
        max_track = max(float(target_probs[int(a) - 1].item()) for a in track_actions)
        max_track = max(max_track, 1e-8)
        search_prior = float(type_probs[0].item())
        track_type_prior = float(type_probs[1].item())
        for action, child in node.children.items():
            action = int(action)
            if action == 0:
                child.prior = search_prior
            elif action - 1 < int(target_probs.shape[0]):
                target_prior = float(target_probs[action - 1].item())
                if mode == "hierarchical":
                    child.prior = target_prior / track_norm
                else:
                    child.prior = track_type_prior * target_prior / max_track

    def _puct_pb_c(self, parent: Node, child: Node) -> float:
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        return float(pb_c)

    def _child_value_score(self, child: Node, min_max_stats: MinMaxStats) -> float:
        if child.visit_count <= 0:
            return 0.0
        return float(min_max_stats.normalize(child.reward + self.config.discount * child.value()))

    def _branch_pb_c(self, parent: Node, branch_visit_count: int) -> float:
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (branch_visit_count + 1)
        return float(pb_c)

    def select_child(self, node: Node, min_max_stats: MinMaxStats):
        if str(getattr(self.config, "factorized_puct_prior", "standard")) != "hierarchical":
            return super().select_child(node, min_max_stats)
        model = getattr(self, "_active_model", None)
        if model is None or 0 not in node.children:
            return super().select_child(node, min_max_stats)
        type_probs, target_probs = self._factorized_probs(model, node)
        if type_probs is None or target_probs is None:
            return super().select_child(node, min_max_stats)
        track_actions = [int(a) for a in node.children if int(a) > 0 and int(a) - 1 < int(target_probs.shape[0])]
        if not track_actions:
            return 0, node.children[0]

        search_child = node.children[0]
        search_score = self._branch_pb_c(node, search_child.visit_count) * float(type_probs[0].item())
        search_score += self._child_value_score(search_child, min_max_stats)

        track_visits = sum(node.children[a].visit_count for a in track_actions)
        track_value = max(self._child_value_score(node.children[a], min_max_stats) for a in track_actions)
        track_score = self._branch_pb_c(node, track_visits) * float(type_probs[1].item()) + track_value

        if search_score > track_score:
            return 0, search_child
        if track_score == search_score and np.random.random() < 0.5:
            return 0, search_child

        max_target_score = max(
            self._puct_pb_c(node, node.children[a]) * node.children[a].prior
            + self._child_value_score(node.children[a], min_max_stats)
            for a in track_actions
        )
        best_actions = [
            a
            for a in track_actions
            if self._puct_pb_c(node, node.children[a]) * node.children[a].prior
            + self._child_value_score(node.children[a], min_max_stats)
            == max_target_score
        ]
        action = int(np.random.choice(best_actions))
        return action, node.children[action]

    def run(self, model, observation, legal_actions, to_play, add_exploration_noise, override_root_with=None):
        self._active_model = model
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            root = Node(0)
            observation = torch.tensor(observation).float().unsqueeze(0).to(next(model.parameters()).device)
            root_predicted_value, reward, policy_logits, hidden_state = model.initial_inference(observation)
            root_predicted_value = models.support_to_scalar(root_predicted_value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            assert legal_actions, f"Legal actions should not be an empty array. Got {legal_actions}."
            assert set(legal_actions).issubset(set(self.config.action_space)), "Legal actions should be a subset of the action space."
            root.expand(legal_actions, to_play, reward, policy_logits, hidden_state)
            self._apply_factorized_priors(model, root)

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()
        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            parent = search_path[-2]
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                torch.tensor([[action]]).to(parent.hidden_state.device),
            )
            value = models.support_to_scalar(value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            latent_actions = self._latent_legal_actions(model, hidden_state, list(self.config.action_space))
            node.expand(latent_actions, virtual_to_play, reward, policy_logits, hidden_state)
            self._apply_factorized_priors(model, node)
            self.backpropagate(search_path, value, virtual_to_play, min_max_stats)
            max_tree_depth = max(max_tree_depth, current_tree_depth)

        return root, {"max_tree_depth": max_tree_depth, "root_predicted_value": root_predicted_value}


def select_action(root, temperature: float, factorized: bool = False) -> int:
    if factorized and 0 in root.children:
        track_actions = [int(a) for a in root.children if int(a) > 0]
        if track_actions:
            search_visits = float(root.children[0].visit_count)
            track_visits = float(sum(root.children[a].visit_count for a in track_actions))
            if temperature <= 0:
                choose_track = track_visits > search_visits
            else:
                branch = np.asarray([search_visits, track_visits], dtype=np.float64)
                branch = branch ** (1.0 / temperature)
                if float(branch.sum()) <= 0.0:
                    branch = np.ones_like(branch)
                branch = branch / float(branch.sum())
                choose_track = bool(np.random.choice([0, 1], p=branch))
            if not choose_track:
                return 0
            visits = np.asarray([root.children[a].visit_count for a in track_actions], dtype=np.float64)
            if temperature <= 0:
                return int(track_actions[int(np.argmax(visits))])
            probs = visits ** (1.0 / temperature)
            if float(probs.sum()) <= 0.0:
                probs = np.ones_like(probs)
            probs = probs / float(probs.sum())
            return int(np.random.choice(track_actions, p=probs))
    visits = np.asarray([ch.visit_count for ch in root.children.values()], dtype=np.float64)
    actions = list(root.children.keys())
    if temperature <= 0:
        return int(actions[int(np.argmax(visits))])
    probs = visits ** (1.0 / temperature)
    probs = probs / max(1e-12, float(probs.sum()))
    return int(np.random.choice(actions, p=probs))


def legal_action_mask(cfg: Config, legal_actions: list[int]) -> list[float]:
    legal = {int(a) for a in legal_actions}
    return [1.0 if int(a) in legal else 0.0 for a in cfg.action_space]


def select_direct_policy_action(model, obs: np.ndarray, legal_actions: list[int], temperature: float) -> int:
    device = next(model.parameters()).device
    x = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, 1, 1, OBS_DIM)
    with torch.no_grad():
        _value, _reward, policy, hidden = model.initial_inference(x)
        factored = select_factorized_policy_action(model, hidden, legal_actions, temperature)
        if factored is not None:
            return int(factored)
        logits = policy[0].detach().float().cpu().numpy()
    return select_from_logits(logits, legal_actions, temperature)


def select_direct_q1_action(model, cfg: Config, obs: np.ndarray, legal_actions: list[int]) -> int:
    legal = [int(a) for a in legal_actions if int(a) in set(cfg.action_space)]
    if not legal:
        return 0
    device = next(model.parameters()).device
    x = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, 1, 1, OBS_DIM)
    with torch.no_grad():
        _value, _reward, _policy, hidden = model.initial_inference(x)
        action_t = torch.as_tensor([[a] for a in legal], dtype=torch.long, device=device)
        hidden_b = hidden.repeat(len(legal), 1)
        child_value, child_reward, _child_policy, _child_hidden = model.recurrent_inference(hidden_b, action_t)
        v = models.support_to_scalar(child_value, cfg.support_size).reshape(-1)
        r = models.support_to_scalar(child_reward, cfg.support_size).reshape(-1)
        score = r + float(cfg.discount) * v
        return int(legal[int(torch.argmax(score).item())])


def select_direct_accum_action(model, obs: np.ndarray, legal_actions: list[int], credit: float) -> tuple[int, float]:
    legal = [int(a) for a in legal_actions if int(a) >= 0]
    if not legal:
        return 0, float(credit)
    device = next(model.parameters()).device
    x = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, 1, 1, OBS_DIM)
    with torch.no_grad():
        _value, _reward, _policy, hidden = model.initial_inference(x)
        if not hasattr(model, "prediction_type_network") or not hasattr(model, "prediction_target_network"):
            return select_direct_policy_action(model, obs, legal, 0.0), float(credit)
        type_probs = torch.softmax(model.prediction_type_network(hidden)[0], dim=0)
        target_logits = model.prediction_target_network(hidden)[0].detach().float()
        track_legal = [a for a in legal if a > 0 and a - 1 < int(target_logits.shape[0])]
        p_search = float(type_probs[0].detach().cpu().item()) if 0 in legal else 0.0
        next_credit = float(credit) + p_search
        if 0 in legal and (next_credit >= 1.0 or not track_legal):
            return 0, next_credit - 1.0
        if track_legal:
            vals = torch.stack([target_logits[int(a) - 1] for a in track_legal])
            return int(track_legal[int(torch.argmax(vals).item())]), next_credit
    return (0 if 0 in legal else int(legal[0])), float(credit)


def select_factorized_policy_action(model, hidden, legal_actions: list[int], temperature: float) -> int | None:
    if not hasattr(model, "prediction_type_network") or not hasattr(model, "prediction_target_network"):
        return None
    legal = [int(a) for a in legal_actions if int(a) >= 0]
    if not legal:
        return 0
    with torch.no_grad():
        type_logits = model.prediction_type_network(hidden)[0].detach().float()
        target_logits = model.prediction_target_network(hidden)[0].detach().float()
    track_legal = [a for a in legal if a > 0 and a - 1 < int(target_logits.shape[0])]
    can_search = 0 in legal
    if not track_legal:
        return 0 if can_search else int(legal[0])
    if not can_search:
        vals = torch.stack([target_logits[a - 1] for a in track_legal])
        return int(track_legal[int(torch.argmax(vals).item())])
    if str(getattr(model, "factorized_direct_select", "type_then_target")) == "joint_product":
        type_logp = F.log_softmax(type_logits, dim=0)
        target_logp = F.log_softmax(target_logits, dim=0)
        action_scores = [type_logp[0]]
        action_ids = [0]
        for a in track_legal:
            action_ids.append(int(a))
            action_scores.append(type_logp[1] + target_logp[int(a) - 1])
        scores = torch.stack(action_scores)
        if temperature <= 0:
            return int(action_ids[int(torch.argmax(scores).item())])
        probs = torch.softmax(scores / max(1e-6, float(temperature)), dim=0)
        return int(action_ids[int(torch.multinomial(probs, 1).item())])
    if temperature <= 0:
        want_search = int(torch.argmax(type_logits).item()) == 0
    else:
        probs = torch.softmax(type_logits / max(1e-6, float(temperature)), dim=0)
        want_search = int(torch.multinomial(probs, 1).item()) == 0
    if want_search:
        return 0
    vals = torch.stack([target_logits[a - 1] for a in track_legal])
    if temperature <= 0:
        return int(track_legal[int(torch.argmax(vals).item())])
    probs = torch.softmax(vals / max(1e-6, float(temperature)), dim=0)
    return int(track_legal[int(torch.multinomial(probs, 1).item())])


def select_from_logits(logits, legal_actions: list[int], temperature: float) -> int:
    logits = np.asarray(logits, dtype=np.float64)
    legal = [int(a) for a in legal_actions if 0 <= int(a) < logits.shape[0]]
    if not legal:
        return 0
    vals = np.asarray([logits[a] for a in legal], dtype=np.float64)
    if temperature <= 0:
        return int(legal[int(np.argmax(vals))])
    vals = vals - float(np.max(vals))
    probs = np.exp(vals / max(1e-6, float(temperature)))
    probs = probs / max(1e-12, float(probs.sum()))
    return int(np.random.choice(legal, p=probs))


def service_score_tuple(metrics: dict, reward_first: bool = False) -> tuple[float, ...]:
    reward = float(metrics.get("reward_per_window", 0.0))
    drop = float(metrics.get("drop_pct_active", 0.0))
    delay = float(metrics.get("mean_delay_active", 0.0))
    tracked = float(metrics.get("tracked_targets", 0.0))
    debt = float(metrics.get("search_debt_end_ms", 0.0))
    latency = float(metrics.get("planning_ms_per_200ms_window", 0.0))
    service_score = float(metrics.get("service_score", tracked - 0.25 * drop - 0.002 * delay - 0.005 * debt))
    if reward_first:
        return (reward, -drop, tracked, -delay, -debt, -latency)
    return (service_score, -drop, -debt, -delay, tracked, reward, -latency)


def annotate_service_score(metrics: dict) -> dict:
    # Human-readable scalar only; checkpoint ranking uses service_score_tuple.
    drop = float(metrics.get("drop_pct_active", 0.0))
    delay = float(metrics.get("mean_delay_active", 0.0))
    tracked = float(metrics.get("tracked_targets", 0.0))
    debt = float(metrics.get("search_debt_end_ms", 0.0))
    metrics["service_score"] = tracked - 0.25 * drop - 0.002 * delay - 0.005 * debt
    return metrics


def shaping_from_args(args) -> RewardShaping:
    return RewardShaping(
        env_mode=str(getattr(args, "env_mode", "radarxs_original")),
        track_update_reward=float(getattr(args, "track_update_reward", 0.30)),
        track_loss_penalty=float(getattr(args, "track_loss_penalty", 4.0)),
        searched_sector_reward_weight=float(getattr(args, "searched_sector_reward_weight", 0.0)),
        search_frame_overdue_weight=float(getattr(args, "search_frame_overdue_weight", 0.05)),
        search_frame_desired_ms=float(getattr(args, "search_frame_desired_ms", 3000.0)),
        search_frame_deadline_ms=float(getattr(args, "search_frame_deadline_ms", 4500.0)),
        search_frame_drop_penalty=float(getattr(args, "search_frame_drop_penalty", 4.0)),
        search_frame_state_penalty_weight=float(getattr(args, "search_frame_state_penalty_weight", 0.0)),
        search_frame_delta_reward_weight=float(getattr(args, "search_frame_delta_reward_weight", 0.0)),
        service_pressure_delta_reward_weight=float(getattr(args, "service_pressure_delta_reward_weight", 0.0)),
        serviced_pressure_delta_reward_weight=float(getattr(args, "serviced_pressure_delta_reward_weight", 0.0)),
        serviced_pressure_improvement_reward_weight=float(getattr(args, "serviced_pressure_improvement_reward_weight", 0.0)),
        discovered_target_reward=float(getattr(args, "discovered_target_reward", 0.0)),
        tracked_count_delta_reward_weight=float(getattr(args, "tracked_count_delta_reward_weight", 0.0)),
        tracked_target_ms_reward_weight=float(getattr(args, "tracked_target_ms_reward_weight", 0.0)),
        service_reward=float(getattr(args, "service_reward", 0.0)),
        discovery_reward=float(getattr(args, "discovery_reward", 0.0)),
        search_refresh_reward=float(getattr(args, "search_refresh_reward", 0.0)),
        search_debt_penalty=float(getattr(args, "search_debt_penalty", 0.0)),
        search_debt_penalty_mode=str(getattr(args, "search_debt_penalty_mode", "step")),
        terminal_search_debt_penalty=float(getattr(args, "terminal_search_debt_penalty", 0.0)),
        window_service_reward=float(getattr(args, "window_service_reward", 0.0)),
        window_tracked_reward=float(getattr(args, "window_tracked_reward", 0.0)),
        window_on_time_reward=float(getattr(args, "window_on_time_reward", 0.0)),
        window_on_time_ratio_reward=float(getattr(args, "window_on_time_ratio_reward", 0.0)),
        penalize_hidden_targets=int(getattr(args, "penalize_hidden_targets", 0)),
    )


def play_episode(model, cfg: Config, args, seed: int, train: bool) -> tuple[GameHistory, dict]:
    game = LiteralRadarGame(
        args.initial,
        args.rate,
        seed,
        args.windows,
        args.reward_scale,
        args.action_ranks,
        shaping_from_args(args),
        bool(getattr(args, "rank_action_space", False)),
    )
    hist = GameHistory()
    hist.legal_history = []
    obs = game.reset()
    hist.action_history.append(0)
    hist.observation_history.append(obs)
    hist.reward_history.append(0.0)
    hist.to_play_history.append(0)
    total = 0.0
    planning_ms = []
    done = False
    try:
        max_moves = int(args.windows) * int(args.max_actions_per_window)
        latent_hidden = None
        latent_policy = None
        accum_credit = 0.0
        while not done and len(hist.action_history) <= max_moves:
            t0 = time.perf_counter()
            legal_actions = game.legal_actions()
            hist.legal_history.append(legal_action_mask(cfg, legal_actions))
            root = None
            if int(args.simulations) <= 0 and bool(getattr(args, "latent_window_rollout", False)):
                device = next(model.parameters()).device
                with torch.no_grad():
                    if latent_hidden is None or float(game.window_spent_ms) <= 1e-6 or latent_policy is None:
                        x = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, 1, 1, OBS_DIM)
                        _value, _reward, policy, latent_hidden = model.initial_inference(x)
                        latent_policy = policy
                    action = select_from_logits(
                        latent_policy[0].detach().float().cpu().numpy(),
                        legal_actions,
                        args.temperature if train else 0.0,
                    )
                    action_t = torch.tensor([[int(action)]], dtype=torch.long, device=device)
                    _value, _reward, latent_policy, latent_hidden = model.recurrent_inference(latent_hidden, action_t)
            elif int(args.simulations) <= 0:
                root = None
                mode = str(getattr(args, "direct_action_mode", "policy"))
                if float(game.window_spent_ms) <= 1e-6:
                    accum_credit = 0.0
                if mode == "accum":
                    action, accum_credit = select_direct_accum_action(model, obs, legal_actions, accum_credit)
                elif mode == "q1" and not train:
                    action = select_direct_q1_action(model, cfg, obs, legal_actions)
                else:
                    action = select_direct_policy_action(model, obs, legal_actions, args.temperature if train else 0.0)
            else:
                use_radar_mcts = bool(getattr(args, "legal_head", False)) or str(
                    getattr(args, "factorized_puct_prior", "standard")
                ) in {"type_balanced", "hierarchical"}
                mcts_cls = RadarMCTS if use_radar_mcts else MCTS
                root, _info = mcts_cls(cfg).run(model, obs, legal_actions, 0, train)
                factorized_select = str(getattr(args, "factorized_puct_prior", "standard")) in {"type_balanced", "hierarchical"}
                action = select_action(root, args.temperature if train else 0.0, factorized=factorized_select)
            planning_ms.append(1000.0 * (time.perf_counter() - t0))
            obs, reward, done = game.step(action)
            if root is not None:
                hist.store_search_statistics(root, cfg.action_space)
            else:
                policy = [0.0] * len(cfg.action_space)
                if 0 <= int(action) < len(policy):
                    policy[int(action)] = 1.0
                else:
                    policy[0] = 1.0
                hist.child_visits.append(policy)
                hist.root_values.append(0.0)
            hist.action_history.append(action)
            hist.observation_history.append(obs)
            hist.reward_history.append(float(reward))
            hist.to_play_history.append(0)
            total += float(reward)
        metrics = game.metrics()
        metrics["scaled_total_reward"] = total
        metrics["reward_per_window"] = total * float(args.reward_scale) / max(1, int(args.windows))
        metrics["planning_ms_per_window"] = float(np.mean(planning_ms)) if planning_ms else 0.0
        metrics["planning_ms_per_200ms_window"] = float(np.sum(planning_ms)) / max(1, int(args.windows))
        metrics["moves"] = int(game.step_count)
        annotate_service_score(metrics)
    finally:
        game.close()
    return hist, metrics


def _logical_action_to_game_action(logical_action: int, game: LiteralRadarGame, legal: set[int], obs: dict) -> int | None:
    logical_action = int(logical_action)
    if logical_action <= 0:
        return 0 if 0 in legal else None
    if bool(game.rank_action_space):
        ranked = ranked_rows(obs)
        if logical_action in ranked:
            action = int(ranked.index(logical_action) + 1)
            return action if action in legal else None
        return None
    return logical_action if logical_action in legal else None


def heuristic_teacher_action(name: str, obs: dict, game: LiteralRadarGame, legal: set[int]) -> int:
    planner = EDFPlanner(MAXT) if name == "edf" else ESTPlanner(MAXT) if name == "est" else None
    if planner is None:
        raise ValueError(name)
    budget_ms = max(1.0, 200.0 - float(game.window_spent_ms))
    for logical_action in planner.plan(obs, budget_ms=budget_ms):
        action = _logical_action_to_game_action(int(logical_action), game, legal, obs)
        if action is not None:
            return int(action)
    return 0 if 0 in legal else int(next(iter(legal)))


def teacher_action(obs: dict, game: LiteralRadarGame, args, legal: set[int] | None = None) -> int:
    legal = set(game.legal_actions()) if legal is None else set(legal)
    mode = str(getattr(args, "teacher_mode", "debt_edf")).lower()
    if mode in {"est", "edf"}:
        return heuristic_teacher_action(mode, obs, game, legal)
    period = int(args.bootstrap_search_period)
    if period > 0 and (game.search_count + game.track_count) % period == 0:
        return 0
    rows = [r for r in ranked_rows(obs) if r <= int(args.action_ranks)]
    if str(getattr(args, "teacher_mode", "debt_edf")) == "service_gate":
        if not rows:
            return 0
        active = np.asarray(obs.get("active_mask", np.zeros(MAXT, dtype=bool))).astype(bool)
        deadline = np.asarray(obs.get("t_deadline", np.full(MAXT, 1e9, dtype=np.float32)), dtype=np.float32)
        desired = np.asarray(obs.get("t_desired", np.full(MAXT, 1e9, dtype=np.float32)), dtype=np.float32)
        dwell = np.asarray(obs.get("t_dwell", np.ones(MAXT, dtype=np.float32) * 10.0), dtype=np.float32)
        valid = active & (deadline >= 0.0)
        if np.any(valid):
            slack = deadline[valid] - dwell[valid]
            min_slack = float(np.min(slack))
            late_frac = float(np.mean(desired[valid] < 0.0))
        else:
            min_slack = 1e9
            late_frac = 0.0
        debt = float(game.debt)
        hard = float(getattr(args, "teacher_search_hard_debt_ms", 1800.0))
        soft = float(getattr(args, "teacher_search_soft_debt_ms", float(args.bootstrap_search_debt_ms)))
        min_safe = float(getattr(args, "teacher_search_min_slack_ms", 180.0))
        max_late = float(getattr(args, "teacher_search_max_late_frac", 0.35))
        if debt >= hard:
            return 0
        if debt >= soft and min_slack >= min_safe and late_frac <= max_late:
            return 0
    elif float(game.debt) >= float(args.bootstrap_search_debt_ms):
        return 0
    if bool(getattr(args, "rank_action_space", False)):
        return 1 if rows else 0
    return int(rows[0]) if rows else 0


def play_teacher_episode(cfg: Config, args, seed: int) -> tuple[GameHistory, dict]:
    game = LiteralRadarGame(
        args.initial,
        args.rate,
        seed,
        args.windows,
        args.reward_scale,
        args.action_ranks,
        shaping_from_args(args),
        bool(getattr(args, "rank_action_space", False)),
    )
    hist = GameHistory()
    hist.legal_history = []
    obs = game.reset()
    hist.action_history.append(0)
    hist.observation_history.append(obs)
    hist.reward_history.append(0.0)
    hist.to_play_history.append(0)
    total = 0.0
    done = False
    try:
        max_moves = int(args.windows) * int(args.max_actions_per_window)
        while not done and len(hist.action_history) <= max_moves:
            od = game.obs_dict()
            legal = set(game.legal_actions())
            hist.legal_history.append(legal_action_mask(cfg, list(legal)))
            action = teacher_action(od, game, args, legal)
            if action not in legal:
                action = 0
            policy = [0.0] * len(cfg.action_space)
            if 0 <= int(action) < len(policy):
                policy[int(action)] = 1.0
            else:
                policy[0] = 1.0
                action = 0
            obs, reward, done = game.step(action)
            hist.child_visits.append(policy)
            hist.root_values.append(0.0)
            hist.action_history.append(int(action))
            hist.observation_history.append(obs)
            hist.reward_history.append(float(reward))
            hist.to_play_history.append(0)
            total += float(reward)
        metrics = game.metrics()
        metrics["scaled_total_reward"] = total
        metrics["reward_per_window"] = total * float(args.reward_scale) / max(1, int(args.windows))
        metrics["planning_ms_per_window"] = 0.0
        metrics["planning_ms_per_200ms_window"] = 0.0
        metrics["moves"] = int(game.step_count)
        annotate_service_score(metrics)
    finally:
        game.close()
    return hist, metrics


def play_dagger_episode(model, cfg: Config, args, seed: int) -> tuple[GameHistory, dict]:
    """Roll in with the current policy, but label visited states with the teacher."""
    game = LiteralRadarGame(
        args.initial,
        args.rate,
        seed,
        args.windows,
        args.reward_scale,
        args.action_ranks,
        shaping_from_args(args),
        bool(getattr(args, "rank_action_space", False)),
    )
    hist = GameHistory()
    hist.legal_history = []
    obs = game.reset()
    hist.action_history.append(0)
    hist.observation_history.append(obs)
    hist.reward_history.append(0.0)
    hist.to_play_history.append(0)
    total = 0.0
    done = False
    planning_ms = []
    try:
        max_moves = int(args.windows) * int(args.max_actions_per_window)
        while not done and len(hist.action_history) <= max_moves:
            legal = game.legal_actions()
            hist.legal_history.append(legal_action_mask(cfg, legal))
            od = game.obs_dict()
            teacher = int(teacher_action(od, game, args, set(legal)))
            if teacher not in set(legal):
                teacher = 0
            policy = [0.0] * len(cfg.action_space)
            if 0 <= teacher < len(policy):
                policy[teacher] = 1.0
            else:
                policy[0] = 1.0

            t0 = time.perf_counter()
            action = select_direct_policy_action(
                model,
                obs,
                legal,
                args.temperature if bool(getattr(args, "dagger_explore", False)) else 0.0,
            )
            planning_ms.append(1000.0 * (time.perf_counter() - t0))
            obs, reward, done = game.step(action)
            hist.child_visits.append(policy)
            hist.root_values.append(0.0)
            hist.action_history.append(int(action))
            hist.observation_history.append(obs)
            hist.reward_history.append(float(reward))
            hist.to_play_history.append(0)
            total += float(reward)
        metrics = game.metrics()
        metrics["scaled_total_reward"] = total
        metrics["reward_per_window"] = total * float(args.reward_scale) / max(1, int(args.windows))
        metrics["planning_ms_per_window"] = float(np.mean(planning_ms)) if planning_ms else 0.0
        metrics["planning_ms_per_200ms_window"] = float(np.sum(planning_ms)) / max(1, int(args.windows))
        metrics["moves"] = int(game.step_count)
        annotate_service_score(metrics)
    finally:
        game.close()
    return hist, metrics


def make_targets(hist: GameHistory, start: int, cfg: Config, unroll: int):
    obs = hist.observation_history[start]
    actions = []
    values = []
    rewards = []
    policies = []
    legal_masks = []
    observations = []
    legal_history = getattr(hist, "legal_history", None)
    for i in range(unroll + 1):
        idx = start + i
        if idx < len(hist.observation_history):
            observations.append(hist.observation_history[idx])
        else:
            observations.append(np.zeros_like(obs))
        if idx < len(hist.reward_history):
            g = 0.0
            if str(getattr(cfg, "value_target_mode", "td")) == "mc":
                bootstrap_idx = len(hist.reward_history) - 1
            else:
                bootstrap_idx = idx + max(1, int(getattr(cfg, "td_steps", 10)))
                if bootstrap_idx < len(hist.root_values):
                    root_value = hist.root_values[bootstrap_idx]
                    if root_value is not None:
                        g = float(root_value) * (cfg.discount ** max(1, int(getattr(cfg, "td_steps", 10))))
            # Match reference MuZero target construction in TD mode: value
            # from the current state is future rewards plus an optional
            # td-step root bootstrap. MC mode disables the root bootstrap and
            # sums the observed episode return, useful for offline teacher
            # histories whose root_values are placeholders.
            stop = min(bootstrap_idx, len(hist.reward_history) - 1)
            p = 1.0
            for j in range(idx + 1, stop + 1):
                g += p * float(hist.reward_history[j])
                p *= cfg.discount
            values.append(g)
            rewards.append(float(hist.reward_history[idx]))
            policies.append(hist.child_visits[idx] if idx < len(hist.child_visits) else [1.0 / len(cfg.action_space)] * len(cfg.action_space))
            if legal_history is not None and idx < len(legal_history):
                legal_masks.append(legal_history[idx])
            else:
                legal_masks.append([1.0] * len(cfg.action_space))
        else:
            values.append(0.0)
            rewards.append(0.0)
            policies.append([1.0 / len(cfg.action_space)] * len(cfg.action_space))
            legal_masks.append([0.0] * len(cfg.action_space))
        if idx < len(hist.action_history):
            actions.append(int(hist.action_history[idx]))
        else:
            actions.append(0)
    return obs, actions, values, rewards, policies, legal_masks, observations


def policy_branch_masses(policy) -> tuple[float, float]:
    arr = np.asarray(policy, dtype=np.float32)
    if arr.size <= 0:
        return 0.0, 0.0
    search_mass = float(arr[0])
    track_mass = float(arr[1:].sum()) if arr.size > 1 else 0.0
    norm = max(1.0e-8, search_mass + track_mass)
    return search_mass / norm, track_mass / norm


def train_batch(model, opt, cfg: Config, replay: list[GameHistory], args):
    device = next(model.parameters()).device
    search_frac = float(getattr(args, "search_sample_frac", 0.0))
    track_frac = float(getattr(args, "track_sample_frac", 0.0))
    search_positions = []
    track_positions = []
    if search_frac > 0.0:
        cache_len = int(getattr(args, "_search_sample_cache_len", -1))
        if cache_len != len(replay):
            cached = []
            for hist_idx, hist in enumerate(replay):
                max_start = max(1, len(hist.action_history) - 1)
                for start in range(min(max_start, len(hist.child_visits))):
                    policy = hist.child_visits[start]
                    search_mass, track_mass = policy_branch_masses(policy)
                    if policy and search_mass >= track_mass:
                        cached.append((hist_idx, start))
            setattr(args, "_search_sample_cache", cached)
            setattr(args, "_search_sample_cache_len", len(replay))
        search_positions = list(getattr(args, "_search_sample_cache", []))
    if track_frac > 0.0:
        cache_len = int(getattr(args, "_track_sample_cache_len", -1))
        if cache_len != len(replay):
            cached = []
            for hist_idx, hist in enumerate(replay):
                max_start = max(1, len(hist.action_history) - 1)
                for start in range(min(max_start, len(hist.child_visits))):
                    policy = hist.child_visits[start]
                    search_mass, track_mass = policy_branch_masses(policy)
                    if policy and track_mass > search_mass:
                        cached.append((hist_idx, start))
            setattr(args, "_track_sample_cache", cached)
            setattr(args, "_track_sample_cache_len", len(replay))
        track_positions = list(getattr(args, "_track_sample_cache", []))
    batch = []
    starts = []
    for _ in range(args.batch_size):
        draw = random.random()
        if track_positions and draw < track_frac:
            hist_idx, start = random.choice(track_positions)
            h = replay[hist_idx]
        elif search_positions and draw < track_frac + search_frac:
            hist_idx, start = random.choice(search_positions)
            h = replay[hist_idx]
        else:
            h = random.choice(replay)
            start = random.randrange(max(1, len(h.action_history) - 1))
        batch.append(h)
        starts.append(start)
    samples = [make_targets(h, s, cfg, args.unroll_steps) for h, s in zip(batch, starts)]
    obs = torch.tensor(np.stack([s[0] for s in samples]), dtype=torch.float32, device=device)
    actions = torch.tensor([[a for a in s[1]] for s in samples], dtype=torch.long, device=device)
    target_v = torch.tensor([[v for v in s[2]] for s in samples], dtype=torch.float32, device=device)
    target_r = torch.tensor([[r for r in s[3]] for s in samples], dtype=torch.float32, device=device)
    target_p = torch.tensor([[p for p in s[4]] for s in samples], dtype=torch.float32, device=device)
    target_legal = torch.tensor([[m for m in s[5]] for s in samples], dtype=torch.float32, device=device)
    target_obs = torch.tensor(np.stack([np.stack(s[6]) for s in samples]), dtype=torch.float32, device=device)
    target_v_support = models.scalar_to_support(target_v, cfg.support_size).to(device)
    target_r_support = models.scalar_to_support(target_r, cfg.support_size).to(device)
    value, reward, policy, hidden = model.initial_inference(obs)
    losses = []
    use_reference_loss_scale = bool(getattr(args, "reference_loss_scale", False))

    def append_loss(per_sample_loss, step_idx: int):
        scale = 1.0
        if use_reference_loss_scale and int(step_idx) > 0:
            scale = 1.0 / max(1, int(args.unroll_steps))
        losses.append(float(scale) * per_sample_loss)

    def add_policy_losses(policy_logits, policy_target, step_idx: int, encoded_state=None):
        is_factorized = str(getattr(args, "network", "")) == "factorized_fullyconnected"
        policy_loss_mode = str(getattr(args, "factorized_policy_loss", "joint"))
        if is_factorized and policy_loss_mode == "separate":
            if encoded_state is not None and hasattr(model, "prediction_type_network"):
                pred_type_logp = F.log_softmax(model.prediction_type_network(encoded_state), dim=1)
                target_logits = model.prediction_target_network(encoded_state)
            else:
                pred_logp = F.log_softmax(policy_logits, dim=1)
                pred_type_logp = torch.stack(
                    [pred_logp[:, 0], torch.logsumexp(pred_logp[:, 1:], dim=1)],
                    dim=1,
                )
                target_logits = policy_logits[:, 1:]
            hard_factorized = bool(getattr(args, "hard_factorized_targets", False))
            track_mass = torch.clamp(policy_target[:, 1:].sum(dim=1), min=0.0, max=1.0)
            if hard_factorized:
                search_mass = torch.clamp(policy_target[:, 0], min=0.0, max=1.0)
                target_is_track = track_mass >= search_mass
                target_type = torch.stack([(~target_is_track).float(), target_is_track.float()], dim=1)
            else:
                target_type = torch.stack(
                    [
                        policy_target[:, 0],
                        track_mass,
                    ],
                    dim=1,
                )
                target_type = target_type / torch.clamp(target_type.sum(dim=1, keepdim=True), min=1e-6)
            smooth = float(getattr(args, "type_label_smoothing", 0.0))
            if smooth > 0.0:
                target_type = (1.0 - smooth) * target_type + smooth * 0.5
            if bool(getattr(args, "balanced_type_loss", False)):
                class_mass = target_type.detach().sum(dim=0).clamp_min(1e-6)
                type_weight = target_type.new_tensor(float(target_type.shape[0]) / 2.0) / class_mass
                type_weight = type_weight / type_weight.mean().clamp_min(1e-6)
            else:
                type_weight = pred_type_logp.new_tensor(
                    [float(getattr(args, "type_search_weight", 1.0)), 1.0]
                )
            type_loss = -((target_type * type_weight) * pred_type_logp).sum(1)

            if hard_factorized:
                track_policy = policy_target[:, 1:]
                action_target = torch.argmax(track_policy, dim=1).clamp(min=0) + 1
                track_mask = (action_target > 0) & (action_target <= target_logits.shape[1])
                target_loss = F.cross_entropy(
                    target_logits,
                    torch.clamp(action_target - 1, min=0, max=target_logits.shape[1] - 1),
                    reduction="none",
                )
                target_loss = target_loss * track_mask.float() * (track_mass > 1e-6).float()
            else:
                target_dist = policy_target[:, 1:] / torch.clamp(track_mass[:, None], min=1e-6)
                target_loss = -(target_dist * F.log_softmax(target_logits, dim=1)).sum(1)
                if bool(getattr(args, "track_mass_target_loss", False)):
                    target_loss = target_loss * track_mass
                else:
                    target_loss = target_loss * (track_mass > 1e-6).float()
            append_loss(type_loss + float(getattr(args, "target_loss_weight", 1.0)) * target_loss, step_idx)
        else:
            per_policy = -(policy_target * F.log_softmax(policy_logits, dim=1)).sum(1)
            search_weight = float(getattr(args, "search_policy_weight", 1.0))
            if search_weight != 1.0:
                target_action = torch.argmax(policy_target, dim=1)
                per_policy = per_policy * torch.where(
                    target_action == 0,
                    per_policy.new_full(per_policy.shape, search_weight),
                    per_policy.new_ones(per_policy.shape),
                )
            append_loss(per_policy, step_idx)
        if float(getattr(args, "type_loss_weight", 0.0)) > 0.0 and is_factorized:
            if encoded_state is not None and hasattr(model, "prediction_type_network"):
                pred_type_logp = F.log_softmax(model.prediction_type_network(encoded_state), dim=1)
            else:
                pred_logp = F.log_softmax(policy_logits, dim=1)
                pred_type_logp = torch.stack(
                    [pred_logp[:, 0], torch.logsumexp(pred_logp[:, 1:], dim=1)],
                    dim=1,
                )
            target_type = torch.stack(
                [
                    policy_target[:, 0],
                    torch.clamp(policy_target[:, 1:].sum(dim=1), min=0.0, max=1.0),
                ],
                dim=1,
            )
            target_type = target_type / torch.clamp(target_type.sum(dim=1, keepdim=True), min=1e-6)
            smooth = float(getattr(args, "type_label_smoothing", 0.0))
            if smooth > 0.0:
                target_type = (1.0 - smooth) * target_type + smooth * 0.5
            if bool(getattr(args, "balanced_type_loss", False)):
                class_mass = target_type.detach().sum(dim=0).clamp_min(1e-6)
                type_weight = target_type.new_tensor(float(target_type.shape[0]) / 2.0) / class_mass
                type_weight = type_weight / type_weight.mean().clamp_min(1e-6)
            else:
                type_weight = pred_type_logp.new_tensor(
                    [float(getattr(args, "type_search_weight", 1.0)), 1.0]
                )
            append_loss(float(args.type_loss_weight) * (-((target_type * type_weight) * pred_type_logp).sum(1)), step_idx)

    policy_only = bool(getattr(args, "policy_only", False))
    skip_policy_loss = bool(getattr(args, "skip_policy_loss", False))
    if not policy_only:
        append_loss(float(getattr(args, "value_loss_weight", 0.25)) * (-(target_v_support[:, 0] * F.log_softmax(value, dim=1)).sum(1)), 0)
    if not skip_policy_loss:
        add_policy_losses(policy, target_p[:, 0], 0, hidden)
    if float(getattr(args, "debt_gate_loss_weight", 0.0)) > 0.0:
        debt_feature = obs[:, 0, 0, 2]
        debt_threshold = float(getattr(args, "debt_gate_threshold_ms", getattr(args, "bootstrap_search_debt_ms", 600.0))) / 3000.0
        debt_search = (debt_feature >= float(debt_threshold)).long()
        pred_logp = F.log_softmax(policy, dim=1)
        pred_type_logp = torch.stack(
            [pred_logp[:, 0], torch.logsumexp(pred_logp[:, 1:], dim=1)],
            dim=1,
        )
        gate_loss = F.nll_loss(pred_type_logp, debt_search, reduction="none")
        append_loss(float(args.debt_gate_loss_weight) * gate_loss, 0)
    recurrent_policy = not bool(getattr(args, "initial_policy_only", False))
    legal_mask_loss_weight = float(getattr(args, "legal_mask_loss_weight", 0.0))
    consistency_loss_weight = float(getattr(args, "consistency_loss_weight", 0.0))
    type_consistency_loss_weight = float(getattr(args, "type_consistency_loss_weight", 0.0))
    search_prob_consistency_loss_weight = float(getattr(args, "search_prob_consistency_loss_weight", 0.0))
    if legal_mask_loss_weight > 0.0 and hasattr(model, "legal_prediction"):
        legal_logits = model.legal_prediction(hidden)
        if legal_logits is not None:
            append_loss(legal_mask_loss_weight * F.binary_cross_entropy_with_logits(legal_logits, target_legal[:, 0], reduction="none").mean(1), 0)
    search_q_rank_weight = float(getattr(args, "search_q_rank_loss_weight", 0.0))
    if search_q_rank_weight > 0.0 and not policy_only:
        root_legal = target_legal[:, 0]
        has_search = root_legal[:, 0] > 0.5
        has_track = root_legal[:, 1:].sum(dim=1) > 0.5
        valid_rank = has_search & has_track
        if bool(valid_rank.any()):
            action_ids = torch.arange(len(cfg.action_space), dtype=torch.long, device=device).view(-1, 1)
            hidden_b = hidden[valid_rank].repeat_interleave(len(cfg.action_space), dim=0)
            action_b = action_ids.repeat(int(valid_rank.sum().item()), 1)
            child_v, child_r, _child_p, _child_h = model.recurrent_inference(hidden_b, action_b)
            q = models.support_to_scalar(child_r, cfg.support_size).reshape(-1) + float(cfg.discount) * models.support_to_scalar(child_v, cfg.support_size).reshape(-1)
            q = q.view(int(valid_rank.sum().item()), len(cfg.action_space))
            legal = root_legal[valid_rank].bool()
            track_q = q[:, 1:].masked_fill(~legal[:, 1:], -1.0e9).max(dim=1).values
            search_q = q[:, 0]
            margin = float(getattr(args, "search_q_rank_margin", 0.0))
            rank_loss = torch.zeros(root_legal.shape[0], dtype=q.dtype, device=device)
            rank_loss[valid_rank] = F.relu(search_q - track_q + margin)
            append_loss(search_q_rank_weight * rank_loss, 0)
    for i in range(1, args.unroll_steps + 1):
        value, reward, policy, hidden = model.recurrent_inference(hidden, actions[:, i : i + 1])
        if use_reference_loss_scale and hidden.requires_grad:
            hidden.register_hook(lambda grad: grad * 0.5)
        if consistency_loss_weight > 0.0:
            with torch.no_grad():
                target_hidden = model.representation(target_obs[:, i])
            append_loss(consistency_loss_weight * F.smooth_l1_loss(hidden, target_hidden, reduction="none").mean(1), i)
        if type_consistency_loss_weight > 0.0 and hasattr(model, "prediction_type_network"):
            with torch.no_grad():
                target_hidden = model.representation(target_obs[:, i])
                target_type = F.softmax(model.prediction_type_network(target_hidden), dim=1)
            pred_type_logp = F.log_softmax(model.prediction_type_network(hidden), dim=1)
            append_loss(type_consistency_loss_weight * (-(target_type * pred_type_logp).sum(1)), i)
        if search_prob_consistency_loss_weight > 0.0 and hasattr(model, "prediction_type_network"):
            with torch.no_grad():
                target_hidden = model.representation(target_obs[:, i])
                target_search = F.softmax(model.prediction_type_network(target_hidden), dim=1)[:, 0]
            pred_search = F.softmax(model.prediction_type_network(hidden), dim=1)[:, 0]
            append_loss(search_prob_consistency_loss_weight * F.smooth_l1_loss(pred_search, target_search, reduction="none"), i)
        if not policy_only:
            append_loss(float(getattr(args, "value_loss_weight", 0.25)) * (-(target_v_support[:, i] * F.log_softmax(value, dim=1)).sum(1)), i)
            append_loss(float(getattr(args, "reward_loss_weight", 1.0)) * (-(target_r_support[:, i] * F.log_softmax(reward, dim=1)).sum(1)), i)
        if recurrent_policy and not skip_policy_loss:
            add_policy_losses(policy, target_p[:, i], i, hidden)
        if legal_mask_loss_weight > 0.0 and hasattr(model, "legal_prediction"):
            legal_logits = model.legal_prediction(hidden)
            if legal_logits is not None:
                append_loss(legal_mask_loss_weight * F.binary_cross_entropy_with_logits(legal_logits, target_legal[:, i], reduction="none").mean(1), i)
    loss = torch.stack(losses).sum(0).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    return float(loss.detach().cpu())


def teacher_policy_diagnostics(model, replay: list[GameHistory], cfg: Config, max_examples: int = 2048) -> dict:
    if not replay:
        return {}
    obs_rows = []
    target_actions = []
    search_branch_masses = []
    hard_branch_search_targets = []
    for hist in replay:
        n = min(len(hist.child_visits), len(hist.observation_history))
        for i in range(n):
            policy = hist.child_visits[i]
            if not policy:
                continue
            policy_arr = np.asarray(policy, dtype=np.float32)
            search_mass = float(policy_arr[0]) if policy_arr.shape[0] else 0.0
            track_mass = float(policy_arr[1:].sum()) if policy_arr.shape[0] > 1 else 0.0
            obs_rows.append(hist.observation_history[i])
            target_actions.append(int(np.argmax(policy)))
            search_branch_masses.append(search_mass / max(1.0e-8, search_mass + track_mass))
            hard_branch_search_targets.append(int(search_mass >= track_mass))
            if len(obs_rows) >= max_examples:
                break
        if len(obs_rows) >= max_examples:
            break
    if not obs_rows:
        return {}
    device = next(model.parameters()).device
    correct1 = 0
    correct5 = 0
    type_correct = 0
    total = 0
    search_targets = 0
    pred_search = 0
    factored_type_correct = 0
    factored_pred_search = 0
    factored_seen = 0
    track_examples = 0
    flat_track_top1 = 0
    flat_track_top5 = 0
    factored_track_top1 = 0
    factored_track_top5 = 0
    with torch.no_grad():
        for start in range(0, len(obs_rows), 256):
            batch_obs = torch.tensor(np.stack(obs_rows[start : start + 256]), dtype=torch.float32, device=device)
            targets = np.asarray(target_actions[start : start + 256], dtype=np.int64)
            _value, _reward, policy, hidden = model.initial_inference(batch_obs)
            logits = policy.detach().float().cpu().numpy()
            pred = np.argmax(logits, axis=1)
            topk = np.argsort(logits, axis=1)[:, -5:]
            correct1 += int(np.sum(pred == targets))
            correct5 += int(sum(int(t) in row for t, row in zip(targets, topk)))
            type_correct += int(np.sum((pred == 0) == (targets == 0)))
            search_targets += int(np.sum(targets == 0))
            pred_search += int(np.sum(pred == 0))
            track_mask = targets > 0
            if np.any(track_mask):
                track_examples += int(np.sum(track_mask))
                order_desc = np.argsort(logits, axis=1)[:, ::-1]
                for tgt, row in zip(targets[track_mask], order_desc[track_mask]):
                    track_row = [int(a) for a in row if int(a) > 0]
                    flat_track_top1 += int(bool(track_row) and track_row[0] == int(tgt))
                    flat_track_top5 += int(int(tgt) in track_row[:5])
            if hasattr(model, "prediction_type_network"):
                type_pred = torch.argmax(model.prediction_type_network(hidden), dim=1).detach().cpu().numpy()
                factored_type_correct += int(np.sum((type_pred == 0) == (targets == 0)))
                factored_pred_search += int(np.sum(type_pred == 0))
                factored_seen += int(targets.shape[0])
                if np.any(track_mask) and hasattr(model, "prediction_target_network"):
                    target_logits = model.prediction_target_network(hidden).detach().float().cpu().numpy()
                    target_order = np.argsort(target_logits, axis=1)[:, ::-1] + 1
                    for tgt, row in zip(targets[track_mask], target_order[track_mask]):
                        factored_track_top1 += int(int(row[0]) == int(tgt))
                        factored_track_top5 += int(int(tgt) in [int(a) for a in row[:5]])
            total += int(targets.shape[0])
    return {
        "teacher_policy_examples": total,
        "teacher_policy_top1": correct1 / max(1, total),
        "teacher_policy_top5": correct5 / max(1, total),
        "teacher_policy_type_acc": type_correct / max(1, total),
        "teacher_policy_search_target_frac": search_targets / max(1, total),
        "teacher_policy_search_branch_mass": float(np.mean(search_branch_masses)) if search_branch_masses else 0.0,
        "teacher_policy_hard_branch_search_frac": float(np.mean(hard_branch_search_targets)) if hard_branch_search_targets else 0.0,
        "teacher_policy_pred_search_frac": pred_search / max(1, total),
        "teacher_policy_factored_type_acc": factored_type_correct / max(1, factored_seen),
        "teacher_policy_factored_pred_search_frac": factored_pred_search / max(1, factored_seen),
        "teacher_track_examples": track_examples,
        "teacher_flat_track_top1": flat_track_top1 / max(1, track_examples),
        "teacher_flat_track_top5": flat_track_top5 / max(1, track_examples),
        "teacher_factored_track_top1": factored_track_top1 / max(1, track_examples),
        "teacher_factored_track_top5": factored_track_top5 / max(1, track_examples),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "CreateValid1" / "results" / "literal_muzero_radar_smoke.csv"))
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--initial", type=int, default=40)
    ap.add_argument("--rate", type=float, default=3.0)
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--action-ranks", type=int, default=MAXT)
    ap.add_argument("--rank-action-space", action="store_true", help="Interpret track actions as ranked target-token slots instead of physical row ids.")
    ap.add_argument("--max-actions-per-window", type=int, default=32)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--episodes-per-iter", type=int, default=2)
    ap.add_argument("--train-steps-per-iter", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--unroll-steps", type=int, default=5)
    ap.add_argument("--td-steps", type=int, default=10)
    ap.add_argument("--value-target-mode", choices=["td", "mc"], default="td")
    ap.add_argument("--simulations", type=int, default=8)
    ap.add_argument("--direct-action-mode", choices=["policy", "q1", "accum"], default="policy")
    ap.add_argument("--latent-window-rollout", action="store_true", help="Encode once per 200 ms window and use recurrent dynamics for subsequent direct actions.")
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--support-size", type=int, default=20)
    ap.add_argument("--encoding-size", type=int, default=64)
    ap.add_argument("--network", choices=["fullyconnected", "factorized_fullyconnected"], default="fullyconnected")
    ap.add_argument("--factorized-search-logit-offset", type=float, default=0.0)
    ap.add_argument("--factorized-puct-prior", choices=["standard", "type_balanced", "hierarchical"], default="standard")
    ap.add_argument("--type-loss-weight", type=float, default=0.0)
    ap.add_argument("--policy-only", action="store_true")
    ap.add_argument("--skip-policy-loss", action="store_true", help="Train value/reward/dynamics losses without applying policy imitation losses.")
    ap.add_argument("--train-dynamics-only", action="store_true", help="Freeze representation and prediction heads; train only MuZero dynamics/reward networks.")
    ap.add_argument("--initial-policy-only", action="store_true")
    ap.add_argument("--search-policy-weight", type=float, default=1.0)
    ap.add_argument("--factorized-policy-loss", choices=["joint", "separate"], default="joint")
    ap.add_argument("--hard-factorized-targets", action="store_true", help="For separate factorized policy loss, train type/target heads from the executed action instead of soft policy mass.")
    ap.add_argument("--soft-factorized-targets", dest="hard_factorized_targets", action="store_false", help="For separate factorized policy loss, keep soft policy mass targets, e.g. PUCT visit distributions.")
    ap.add_argument("--type-search-weight", type=float, default=1.0)
    ap.add_argument("--balanced-type-loss", action="store_true")
    ap.add_argument("--type-label-smoothing", type=float, default=0.0)
    ap.add_argument("--target-loss-weight", type=float, default=1.0)
    ap.add_argument("--track-mass-target-loss", action="store_true", help="For soft factorized policy targets, weight target CE by P(track).")
    ap.add_argument("--legal-head", action="store_true")
    ap.add_argument("--legal-mask-loss-weight", type=float, default=0.0)
    ap.add_argument("--search-q-rank-loss-weight", type=float, default=0.0)
    ap.add_argument("--search-q-rank-margin", type=float, default=0.0)
    ap.add_argument("--debt-gate-loss-weight", type=float, default=0.0)
    ap.add_argument("--debt-gate-threshold-ms", type=float, default=600.0)
    ap.add_argument("--search-sample-frac", type=float, default=0.0)
    ap.add_argument("--track-sample-frac", type=float, default=0.0)
    ap.add_argument("--value-loss-weight", type=float, default=0.25)
    ap.add_argument("--reward-loss-weight", type=float, default=1.0)
    ap.add_argument("--consistency-loss-weight", type=float, default=0.0, help="Optional latent dynamics consistency loss ||g(h,a)-h(s')|| using replay observations.")
    ap.add_argument("--type-consistency-loss-weight", type=float, default=0.0, help="Optional recurrent type-head consistency loss against h(s') type distribution.")
    ap.add_argument("--search-prob-consistency-loss-weight", type=float, default=0.0, help="Optional recurrent P(search) calibration loss against h(s') P(search).")
    ap.add_argument("--reference-loss-scale", action="store_true", help="Apply reference MuZero recurrent loss and hidden-state gradient scaling.")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--reward-scale", type=float, default=10.0)
    ap.add_argument("--env-mode", default="radarxs_original")
    ap.add_argument("--track-update-reward", type=float, default=0.30)
    ap.add_argument("--track-loss-penalty", type=float, default=4.0)
    ap.add_argument("--searched-sector-reward-weight", type=float, default=0.0)
    ap.add_argument("--search-frame-overdue-weight", type=float, default=0.05)
    ap.add_argument("--search-frame-desired-ms", type=float, default=3000.0)
    ap.add_argument("--search-frame-deadline-ms", type=float, default=4500.0)
    ap.add_argument("--search-frame-drop-penalty", type=float, default=4.0)
    ap.add_argument("--search-frame-state-penalty-weight", type=float, default=0.0)
    ap.add_argument("--search-frame-delta-reward-weight", type=float, default=0.0)
    ap.add_argument("--service-pressure-delta-reward-weight", type=float, default=0.0)
    ap.add_argument("--serviced-pressure-delta-reward-weight", type=float, default=0.0)
    ap.add_argument("--serviced-pressure-improvement-reward-weight", type=float, default=0.0)
    ap.add_argument("--discovered-target-reward", type=float, default=0.0)
    ap.add_argument("--tracked-count-delta-reward-weight", type=float, default=0.0)
    ap.add_argument("--tracked-target-ms-reward-weight", type=float, default=0.0)
    ap.add_argument("--service-reward", type=float, default=0.0)
    ap.add_argument("--discovery-reward", type=float, default=0.0)
    ap.add_argument("--search-refresh-reward", type=float, default=0.0)
    ap.add_argument("--search-debt-penalty", type=float, default=0.0)
    ap.add_argument("--search-debt-penalty-mode", choices=["step", "time", "window"], default="step")
    ap.add_argument("--terminal-search-debt-penalty", type=float, default=0.0)
    ap.add_argument("--window-service-reward", type=float, default=0.0)
    ap.add_argument("--window-tracked-reward", type=float, default=0.0)
    ap.add_argument("--window-on-time-reward", type=float, default=0.0)
    ap.add_argument("--window-on-time-ratio-reward", type=float, default=0.0)
    ap.add_argument("--penalize-hidden-targets", type=int, choices=[0, 1], default=0)
    ap.add_argument("--root-dirichlet-alpha", type=float, default=0.25)
    ap.add_argument("--root-exploration-fraction", type=float, default=0.25)
    ap.add_argument("--pb-c-base", type=float, default=19652.0)
    ap.add_argument("--pb-c-init", type=float, default=1.25)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--teacher-mode", choices=["debt_edf", "service_gate", "est", "edf"], default="debt_edf")
    ap.add_argument("--teacher-search-soft-debt-ms", type=float, default=600.0)
    ap.add_argument("--teacher-search-hard-debt-ms", type=float, default=1800.0)
    ap.add_argument("--teacher-search-min-slack-ms", type=float, default=180.0)
    ap.add_argument("--teacher-search-max-late-frac", type=float, default=0.35)
    ap.add_argument("--bootstrap-episodes", type=int, default=0)
    ap.add_argument("--bootstrap-grid-eval", action="store_true", help="Evaluate bootstrapped checkpoint on the full grid and allow it to be selected as best.")
    ap.add_argument("--bootstrap-search-debt-ms", type=float, default=600.0)
    ap.add_argument("--bootstrap-search-period", type=int, default=10)
    ap.add_argument("--dagger-iterations", type=int, default=0)
    ap.add_argument("--dagger-episodes-per-iter", type=int, default=3)
    ap.add_argument("--dagger-train-steps-per-iter", type=int, default=60)
    ap.add_argument("--dagger-explore", action="store_true")
    ap.add_argument("--dagger-grid-eval", action="store_true", help="Select DAgger best checkpoint using the full eval grid.")
    ap.add_argument("--selfplay-grid-eval", action="store_true", help="Select self-play best checkpoint using the full eval grid.")
    ap.add_argument("--selfplay-eval-every", type=int, default=1, help="Run the deterministic self-play validation episode every N iterations.")
    ap.add_argument("--grid-eval-simulations", type=int, default=-1, help="Override --simulations during grid checkpoint evaluation; use 0 for deployable direct evaluation.")
    ap.add_argument("--grid-eval-direct-action-mode", choices=["", "policy", "q1", "accum"], default="", help="Override direct action mode during grid checkpoint evaluation.")
    ap.add_argument("--grid-eval-latent-window-rollout", action="store_true", help="Use latent-window rollout during grid checkpoint evaluation.")
    ap.add_argument("--checkpoint-score", choices=["service", "reward"], default="service")
    ap.add_argument("--save-state", default="", help="Optional path for saving the trained MuZero network state.")
    ap.add_argument("--load-state", default="", help="Optional checkpoint path to load before training/evaluation.")
    ap.add_argument("--load-state-key", default="model", choices=["model", "best_model"], help="State dict key to load from --load-state.")
    ap.add_argument("--restore-args-from-state", action="store_true", help="Restore model/action-space/reward contract args stored in --load-state unless explicitly overridden.")
    ap.add_argument("--eval-only", action="store_true", help="Skip training and only run deterministic evaluations.")
    ap.add_argument("--eval-planner", choices=["model", "teacher"], default="model", help="Planner used by --eval-only.")
    ap.add_argument("--evals", type=int, default=1, help="Number of deterministic eval episodes when --eval-only is set.")
    ap.add_argument("--initials", default="", help="Comma-separated initial target counts for eval-only grid mode.")
    ap.add_argument("--rates", default="", help="Comma-separated arrival rates for eval-only grid mode.")
    args = ap.parse_args()
    payload = None
    if str(args.load_state):
        payload = torch.load(Path(args.load_state), map_location="cpu")
        if bool(getattr(args, "restore_args_from_state", False)) and isinstance(payload, dict) and isinstance(payload.get("args"), dict):
            cli_flags = {tok for tok in sys.argv[1:] if tok.startswith("--")}
            restore_keys = {
                "action_ranks",
                "rank_action_space",
                "max_actions_per_window",
                "gamma",
                "support_size",
                "encoding_size",
                "network",
                "factorized_search_logit_offset",
                "factorized_policy_loss",
                "factorized_puct_prior",
                "hard_factorized_targets",
                "balanced_type_loss",
                "track_mass_target_loss",
                "value_target_mode",
                "legal_head",
                "legal_mask_loss_weight",
                "consistency_loss_weight",
                "type_consistency_loss_weight",
                "search_prob_consistency_loss_weight",
                "root_dirichlet_alpha",
                "root_exploration_fraction",
                "pb_c_base",
                "pb_c_init",
                "reward_scale",
                "env_mode",
                "track_update_reward",
                "track_loss_penalty",
                "searched_sector_reward_weight",
                "search_frame_overdue_weight",
                "search_frame_desired_ms",
                "search_frame_deadline_ms",
                "search_frame_drop_penalty",
                "search_frame_state_penalty_weight",
                "search_frame_delta_reward_weight",
                "service_pressure_delta_reward_weight",
                "serviced_pressure_delta_reward_weight",
                "serviced_pressure_improvement_reward_weight",
                "discovered_target_reward",
                "service_reward",
                "discovery_reward",
                "search_refresh_reward",
                "search_debt_penalty",
                "search_debt_penalty_mode",
                "terminal_search_debt_penalty",
                "window_service_reward",
                "window_tracked_reward",
                "window_on_time_reward",
                "window_on_time_ratio_reward",
                "penalize_hidden_targets",
            }
            saved_args = payload.get("args", {})
            for key in restore_keys:
                flag = "--" + key.replace("_", "-")
                if key == "hard_factorized_targets" and "--soft-factorized-targets" in cli_flags:
                    continue
                if key in saved_args and flag not in cli_flags:
                    setattr(args, key, saved_args[key])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    cfg = Config(args)
    model = models.MuZeroNetwork(cfg)
    if str(args.load_state):
        if payload is None:
            payload = torch.load(Path(args.load_state), map_location="cpu")
        state = payload[str(args.load_state_key)] if isinstance(payload, dict) and str(args.load_state_key) in payload else payload
        model.load_state_dict(state)
    if bool(getattr(args, "train_dynamics_only", False)):
        trainable_prefixes = ("dynamics_encoded_state_network", "dynamics_reward_network")
        trainable = 0
        frozen = 0
        for name, param in model.named_parameters():
            keep = name.startswith(trainable_prefixes)
            param.requires_grad_(keep)
            trainable += int(keep)
            frozen += int(not keep)
        if trainable <= 0:
            raise RuntimeError("train-dynamics-only selected but no dynamics parameters were found")
        print({"train_dynamics_only": True, "trainable_tensors": trainable, "frozen_tensors": frozen}, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    replay: list[GameHistory] = []
    rows = []
    best_eval = None
    best_state = None
    cells = curriculum_cells(args)

    def grid_eval_rows(phase: str, it: int, loss: float, diag: dict | None, seed_base: int):
        base_initial, base_rate = int(args.initial), float(args.rate)
        base_simulations = int(args.simulations)
        base_direct_mode = str(getattr(args, "direct_action_mode", "policy"))
        base_latent_window = bool(getattr(args, "latent_window_rollout", False))
        eval_simulations = int(getattr(args, "grid_eval_simulations", -1))
        if eval_simulations >= 0:
            args.simulations = eval_simulations
        eval_direct_mode = str(getattr(args, "grid_eval_direct_action_mode", ""))
        if eval_direct_mode:
            args.direct_action_mode = eval_direct_mode
        if bool(getattr(args, "grid_eval_latent_window_rollout", False)):
            args.latent_window_rollout = True
        eval_rows = []
        try:
            for ci, cell in enumerate(cells):
                set_cell(args, cell)
                _hist, metrics = play_episode(model, cfg, args, seed_base + ci, train=False)
                row = {
                    "phase": phase,
                    "iter": it,
                    "episode": ci,
                    "initial": int(args.initial),
                    "rate": float(args.rate),
                    "loss": loss,
                    **(diag or {}),
                    **metrics,
                }
                rows.append(row)
                eval_rows.append(row)
        finally:
            args.initial, args.rate = base_initial, base_rate
            args.simulations = base_simulations
            args.direct_action_mode = base_direct_mode
            args.latent_window_rollout = base_latent_window
        if not eval_rows:
            return {}, ()
        keys = [
            "reward_per_window",
            "drop_pct_active",
            "tracked_targets",
            "mean_delay_active",
            "search_debt_end_ms",
            "planning_ms_per_200ms_window",
            "service_score",
        ]
        avg = {k: float(np.mean([float(r.get(k, 0.0)) for r in eval_rows])) for k in keys}
        score = service_score_tuple(avg, reward_first=str(args.checkpoint_score) == "reward")
        return avg, score
    if bool(args.eval_only):
        base_initial, base_rate = int(args.initial), float(args.rate)
        for ci, (initial, rate) in enumerate(cells):
            set_cell(args, (initial, rate))
            for ep in range(int(args.evals)):
                eval_seed = args.seed + 9000 + ci * 100 + ep
                if str(getattr(args, "eval_planner", "model")) == "teacher":
                    hist, metrics = play_teacher_episode(cfg, args, eval_seed)
                else:
                    hist, metrics = play_episode(model, cfg, args, eval_seed, train=False)
                rows.append(
                    {
                        "phase": "eval_only",
                        "iter": 0,
                        "episode": ep,
                        "initial": int(initial),
                        "rate": float(rate),
                        "loss": np.nan,
                        **metrics,
                    }
                )
                print(rows[-1], flush=True)
        args.initial, args.rate = base_initial, base_rate
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print({"out": str(out), "state": ""}, flush=True)
        return
    for ep in range(int(args.bootstrap_episodes)):
        cell = cells[ep % len(cells)]
        set_cell(args, cell)
        hist, metrics = play_teacher_episode(cfg, args, args.seed + 70000 + ep)
        replay.append(hist)
        rows.append({"phase": "bootstrap", "iter": -1, "episode": ep, "initial": cell[0], "rate": cell[1], "loss": np.nan, **metrics})
    if replay and int(args.train_steps_per_iter) > 0:
        losses = []
        for _ in range(max(1, int(args.bootstrap_episodes)) * int(args.train_steps_per_iter)):
            losses.append(train_batch(model, opt, cfg, replay, args))
        diag = teacher_policy_diagnostics(model, replay, cfg)
        bootstrap_loss = float(np.mean(losses)) if losses else np.nan
        rows.append(
            {
                "phase": "bootstrap_train",
                "iter": -1,
                "episode": -1,
                "loss": bootstrap_loss,
                **diag,
            }
        )
        score = None
        if bool(getattr(args, "bootstrap_grid_eval", False)):
            metrics, score = grid_eval_rows("bootstrap_grid_eval", -1, bootstrap_loss, diag, args.seed + 88000)
            rows.append(
                {
                    "phase": "bootstrap_grid_summary",
                    "iter": -1,
                    "episode": -1,
                    "initial": -1,
                    "rate": -1.0,
                    "loss": bootstrap_loss,
                    **diag,
                    **{f"grid_{k}": v for k, v in metrics.items()},
                }
            )
        if score is not None and (best_eval is None or score > best_eval[0]):
            best_eval = (score, dict(rows[-1]))
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if str(args.load_state) and bool(getattr(args, "selfplay_grid_eval", False)) and not bool(getattr(args, "eval_only", False)):
        metrics, score = grid_eval_rows("loaded_grid_eval", -1, float("nan"), None, args.seed + 8500)
        rows.append(
            {
                "phase": "loaded_grid_summary",
                "iter": -1,
                "episode": -1,
                "initial": -1,
                "rate": -1.0,
                "loss": float("nan"),
                **{f"grid_{k}": v for k, v in metrics.items()},
            }
        )
        if best_eval is None or score > best_eval[0]:
            best_eval = (score, dict(rows[-1]))
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for it in range(int(args.dagger_iterations)):
        for ep in range(int(args.dagger_episodes_per_iter)):
            cell = cells[(it * int(args.dagger_episodes_per_iter) + ep) % len(cells)]
            set_cell(args, cell)
            hist, metrics = play_dagger_episode(model, cfg, args, args.seed + 80000 + it * 100 + ep)
            replay.append(hist)
            rows.append({"phase": "dagger_rollin", "iter": it, "episode": ep, "initial": cell[0], "rate": cell[1], "loss": np.nan, **metrics})
        losses = []
        if replay:
            for _ in range(int(args.dagger_train_steps_per_iter)):
                losses.append(train_batch(model, opt, cfg, replay, args))
        diag = teacher_policy_diagnostics(model, replay, cfg)
        loss_value = float(np.mean(losses)) if losses else np.nan
        if bool(getattr(args, "dagger_grid_eval", False)):
            metrics, score = grid_eval_rows("dagger_grid_eval", it, loss_value, diag, args.seed + 95000 + it * 100)
            rows.append(
                {
                    "phase": "dagger_grid_summary",
                    "iter": it,
                    "episode": -1,
                    "initial": -1,
                    "rate": -1.0,
                    "loss": loss_value,
                    **diag,
                    **{f"grid_{k}": v for k, v in metrics.items()},
                }
            )
        else:
            set_cell(args, cells[it % len(cells)])
            _hist, metrics = play_episode(model, cfg, args, args.seed + 95000 + it, train=False)
            rows.append(
                {
                    "phase": "dagger_eval",
                    "iter": it,
                    "episode": -1,
                    "initial": int(args.initial),
                    "rate": float(args.rate),
                    "loss": loss_value,
                    **diag,
                    **metrics,
                }
            )
            score = (
                service_score_tuple(metrics, reward_first=str(args.checkpoint_score) == "reward")
            )
        if best_eval is None or score > best_eval[0]:
            best_eval = (score, dict(rows[-1]))
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for it in range(args.iterations):
        for ep in range(args.episodes_per_iter):
            cell = cells[(it * int(args.episodes_per_iter) + ep) % len(cells)]
            set_cell(args, cell)
            hist, metrics = play_episode(model, cfg, args, args.seed + it * 100 + ep, train=True)
            replay.append(hist)
            rows.append({"phase": "selfplay", "iter": it, "episode": ep, "initial": cell[0], "rate": cell[1], "loss": np.nan, **metrics})
        losses = []
        if replay:
            for _ in range(args.train_steps_per_iter):
                losses.append(train_batch(model, opt, cfg, replay, args))
        diag = teacher_policy_diagnostics(model, replay, cfg)
        loss_value = float(np.mean(losses)) if losses else np.nan
        score = None
        if bool(getattr(args, "selfplay_grid_eval", False)):
            metrics, score = grid_eval_rows("selfplay_grid_eval", it, loss_value, diag, args.seed + 9000 + it * 100)
            rows.append(
                {
                    "phase": "selfplay_grid_summary",
                    "iter": it,
                    "episode": -1,
                    "initial": -1,
                    "rate": -1.0,
                    "loss": loss_value,
                    **diag,
                    **{f"grid_{k}": v for k, v in metrics.items()},
                }
            )
        else:
            evaluate_now = (it % max(1, int(args.selfplay_eval_every)) == 0) or (it + 1 == int(args.iterations))
            if evaluate_now:
                set_cell(args, cells[it % len(cells)])
                hist, metrics = play_episode(model, cfg, args, args.seed + 9000 + it, train=False)
                rows.append(
                    {
                        "phase": "eval",
                        "iter": it,
                        "episode": -1,
                        "initial": int(args.initial),
                        "rate": float(args.rate),
                        "loss": loss_value,
                        **diag,
                        **metrics,
                    }
                )
                score = service_score_tuple(metrics, reward_first=str(args.checkpoint_score) == "reward")
            else:
                rows.append(
                    {
                        "phase": "selfplay_train",
                        "iter": it,
                        "episode": -1,
                        "initial": -1,
                        "rate": -1.0,
                        "loss": loss_value,
                        **diag,
                    }
                )
        if score is not None and (best_eval is None or score > best_eval[0]):
            best_eval = (score, dict(rows[-1]))
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(rows[-1], flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    saved_state = ""
    if str(args.save_state):
        save_state = Path(args.save_state)
        save_state.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": model.state_dict(), "args": vars(args), "best_eval": best_eval[1] if best_eval else None}
        if best_state is not None:
            payload["best_model"] = best_state
        torch.save(payload, save_state)
        saved_state = str(save_state)
    print({"out": str(out), "state": saved_state}, flush=True)


if __name__ == "__main__":
    main()
