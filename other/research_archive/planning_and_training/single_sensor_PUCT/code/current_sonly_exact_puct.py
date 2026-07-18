from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "CreateValid1" / "experiments" / "code" / "model_code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from exact_env_mutual import (  # noqa: E402
    EDFPlanner,
    ESTPlanner,
    MAXT,
    _DummyPlanner,
    active_target_count,
    build_env,
    engine_env_cfg,
    env_cfg_for,
    get_obs,
    search_frame_pressure_sum,
    service_pressure_sum,
    shaped_step_reward,
    xs_decode_action,
    xs_s_search_action,
    xs_s_track_action,
)
from final_radar_campaign import summarize_window_df  # noqa: E402
from penalty_window_quota_learner_eval import make_exact_args  # noqa: E402
from pufferlib.ocean.radarxs import binding  # noqa: E402
from repaired_campaign_tools import execute_first_valid_action  # noqa: E402
from single_sensor_ar_action_attention import load_action_attention_model  # noqa: E402
from strict_window_report import sample_state_metrics  # noqa: E402
from two_sensor_physical_head_eval import PhysicalHeadPlanner  # noqa: E402
from exact_env_mutual import attach_env_obs  # noqa: E402
from mutual_features import slot_features, tokenize  # noqa: E402
from realistic_reward_retrain import adapter  # noqa: E402
from train_action_attention_muzero_g import LatentG, infer_latent_d_model  # noqa: E402


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


class MatchedPUCTPolicyValue:
    """Expose a matched AR/batch checkpoint as PUCT policy and action-Q heads."""

    def __init__(self, checkpoint_path: Path, env_cfg: dict, *, device: str, policy_weight: float, q_weight: float):
        from puct_pq_ar_batch import MatchedPUCTScheduler

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = dict(checkpoint["model_config"])
        family = str(checkpoint.get("family", "batch"))
        self.model = MatchedPUCTScheduler(
            family=family,
            d_model=int(cfg["d_model"]),
            slot_dim=int(cfg["slot_dim"]),
            mixer=str(cfg["mixer"]),
            builder=str(cfg["builder"]),
            max_steps=int(cfg["max_steps"]),
            raw_feature_dim=int(cfg.get("raw_feature_dim", 0)),
            max_rows=int(cfg.get("max_rows", MAXT + 1)),
            encoder_layers=int(cfg.get("encoder_layers", 2)),
            encoder_nhead=int(cfg.get("encoder_heads", 4)),
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.env_cfg = dict(env_cfg)
        self.device = torch.device(device)
        self.policy_weight = float(policy_weight)
        self.q_weight = float(q_weight)
        self.max_steps = int(cfg["max_steps"])
        self._adapt = adapter()

    @torch.inference_mode()
    def _outputs(self, obs: dict, *, selected: set[int], elapsed: float, search_count: int, track_count: int, last: int):
        from puct_pq_ar_batch import RawDaggerView, SchedulerState

        attached = attach_env_obs(dict(obs), self.env_cfg, True, True)
        x_np = tokenize(self._adapt, attached, selected=selected, search_count=search_count)
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(self.device)
        valid = RawDaggerView._base_valid(x[0]).unsqueeze(0)
        state = self.model.encode_root(x, valid)
        slot_np = slot_features(attached, elapsed, search_count, track_count, last, 200.0)
        slot = torch.from_numpy(slot_np).float().unsqueeze(0).to(self.device)
        if self.model.family == "batch":
            step = min(self.max_steps - 1, max(0, int(search_count) + int(track_count)))
            contexts = self.model.adapter(state, step + 1)
            state = SchedulerState(state.global_token, state.target_tokens, contexts[:, step])
        output = self.model.scorer(state, slot, valid)
        return output

    def score_actions(self, obs: dict, **kwargs) -> np.ndarray:
        output = self._outputs(obs, **kwargs)
        # PUCT priors must come from policy alone; learned Q initializes child
        # values separately and must not be counted twice in the exploration prior.
        return (self.policy_weight * output.policy_logits[0]).detach().cpu().numpy()[:, None]

    def action_values(self, obs: dict, **kwargs) -> np.ndarray:
        output = self._outputs(obs, **kwargs)
        return (self.q_weight * output.q_values[0]).detach().cpu().numpy()


@dataclass
class Node:
    snapshot: object
    debt: float
    selected: set[int]
    elapsed: float
    search_count: int
    track_count: int
    last: int
    prior: float = 1.0
    reward: float = 0.0
    dt: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)
    pending: list[tuple[int, float]] = field(default_factory=list)
    candidates_ready: bool = False

    def value(self) -> float:
        return self.value_sum / max(1, self.visits)


class ExactSOnlyPuctPlanner:
    def __init__(
        self,
        model_path: str,
        variant: str,
        env_cfg: dict,
        *,
        device: str,
        simulations: int,
        expand_top_k: int,
        rollout_steps: int,
        c_puct: float,
        discount: float,
        select_mode: str,
        policy_weight: float,
        q_weight: float,
        search_bias: float,
        terminal_service_weight: float = 0.0,
        terminal_search_frame_weight: float = 0.0,
        rollout_windows: int = 1,
        init_child_rollouts: bool = False,
        leaf_g_state: str = "",
        leaf_value_weight: float = 0.0,
        leaf_value_top_k: int = 16,
        direct_child_value_weight: float = 0.0,
        prior_uniform_mix: float = 0.0,
        root_dirichlet_alpha: float = 0.3,
        root_dirichlet_fraction: float = 0.0,
        progressive_widening_c: float = 2.0,
        progressive_widening_alpha: float = 0.5,
        stratify_root_types: bool = True,
        matched_checkpoint: str = "",
    ):
        if str(matched_checkpoint).strip():
            self.base = MatchedPUCTPolicyValue(
                Path(matched_checkpoint),
                env_cfg,
                device=str(device),
                policy_weight=float(policy_weight),
                q_weight=float(q_weight),
            )
        else:
            model = load_action_attention_model(Path(model_path), device)
            self.base = PhysicalHeadPlanner(
                model,
                str(variant),
                env_cfg,
                policy_weight=float(policy_weight),
                q_weight=float(q_weight),
                search_score_bias=float(search_bias),
            )
        self.env_cfg = dict(env_cfg)
        self.simulations = int(simulations)
        self.expand_top_k = int(expand_top_k)
        self.rollout_steps = int(rollout_steps)
        self.c_puct = float(c_puct)
        self.discount = float(discount)
        self.select_mode = str(select_mode)
        self.terminal_service_weight = float(terminal_service_weight)
        self.terminal_search_frame_weight = float(terminal_search_frame_weight)
        self.rollout_windows = max(1, int(rollout_windows))
        self.init_child_rollouts = bool(init_child_rollouts)
        self.leaf_value_weight = float(leaf_value_weight)
        self.leaf_value_top_k = int(leaf_value_top_k)
        self.direct_child_value_weight = float(direct_child_value_weight)
        self.prior_uniform_mix = float(np.clip(prior_uniform_mix, 0.0, 1.0))
        self.root_dirichlet_alpha = max(1.0e-4, float(root_dirichlet_alpha))
        self.root_dirichlet_fraction = float(np.clip(root_dirichlet_fraction, 0.0, 1.0))
        self.progressive_widening_c = max(1.0, float(progressive_widening_c))
        self.progressive_widening_alpha = float(np.clip(progressive_widening_alpha, 0.0, 1.0))
        self.stratify_root_types = bool(stratify_root_types)
        self.rng = np.random.default_rng(0)
        self._adapt = adapter()
        self.leaf_g = None
        self.leaf_value_mean = 0.0
        self.leaf_value_std = 1.0
        if str(leaf_g_state).strip() and (self.leaf_value_weight != 0.0 or self.direct_child_value_weight != 0.0):
            ckpt = torch.load(str(leaf_g_state), map_location=device)
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            d_model = infer_latent_d_model(state, 48)
            history_k = 4 if any(str(k).startswith("ar_history_proj.") for k in state.keys()) else 0
            leaf_g = LatentG(d_model=d_model, seq_len=40, ar_history_k=history_k).to(device)
            leaf_g.load_state_dict(state, strict=False)
            leaf_g.eval()
            self.leaf_g = leaf_g
            self.leaf_value_mean = float(ckpt.get("action_value_target_mean", 0.0)) if isinstance(ckpt, dict) else 0.0
            self.leaf_value_std = max(1.0e-3, float(ckpt.get("action_value_target_std", 1.0))) if isinstance(ckpt, dict) else 1.0

    @staticmethod
    def _action_pair_from_row(row: int) -> torch.Tensor:
        row = max(0, int(row))
        return torch.tensor([[row * 2, -1]], dtype=torch.long)

    def _terminal_value(self, obs: dict) -> float:
        value = 0.0
        if self.terminal_service_weight != 0.0:
            denom = float(max(1, active_target_count(obs)))
            value -= self.terminal_service_weight * service_pressure_sum(obs) / denom
        if self.terminal_search_frame_weight != 0.0:
            value -= self.terminal_search_frame_weight * search_frame_pressure_sum(obs, self.env_cfg)
        return float(value)

    def _learned_leaf_value(self, obs: dict, selected: set[int], elapsed: float, search_count: int, track_count: int, last: int, remaining: float) -> float:
        if self.leaf_g is None or self.leaf_value_weight == 0.0:
            return 0.0
        scores = self._valid_scores(obs, selected, elapsed, search_count, track_count, last, remaining)
        valid = np.where(np.isfinite(scores) & (scores > -1.0e17))[0]
        if len(valid) == 0:
            return 0.0
        order = valid[np.argsort(scores[valid])[::-1]]
        if self.leaf_value_top_k > 0:
            order = order[: max(1, self.leaf_value_top_k)]
        obs_attached = attach_env_obs(dict(obs), self.env_cfg, True, True)
        x = tokenize(self._adapt, obs_attached, selected=selected, search_count=search_count).astype(np.float32)
        slot = slot_features(obs_attached, float(elapsed), int(search_count), int(track_count), int(last), 200.0).astype(np.float32)
        device = next(self.base.model.parameters()).device
        with torch.inference_mode():
            xt = torch.from_numpy(x).float().unsqueeze(0).to(device)
            st = torch.from_numpy(slot).float().unsqueeze(0).to(device)
            cls, tok, _sel, _active = self.base.model.backbone.encode_tokens(xt)
            values = []
            for row in order:
                pair = self._action_pair_from_row(int(row)).to(device)
                pred = self.leaf_g.predict_action_value(cls, tok, st, pair)
                denorm = pred * self.leaf_value_std + self.leaf_value_mean
                values.append(float(denorm.detach().cpu()[0]))
        if not values:
            return 0.0
        return float(self.leaf_value_weight * max(values))

    def _learned_action_value(self, obs: dict, row: int, selected: set[int], elapsed: float, search_count: int, track_count: int, last: int) -> float:
        if hasattr(self.base, "action_values") and self.direct_child_value_weight != 0.0:
            values = self.base.action_values(
                obs,
                selected=selected,
                elapsed=float(elapsed),
                search_count=int(search_count),
                track_count=int(track_count),
                last=int(last),
            )
            if 0 <= int(row) < len(values) and np.isfinite(values[int(row)]):
                return float(self.direct_child_value_weight * float(values[int(row)]))
        if self.leaf_g is None or self.direct_child_value_weight == 0.0:
            return 0.0
        obs_attached = attach_env_obs(dict(obs), self.env_cfg, True, True)
        x = tokenize(self._adapt, obs_attached, selected=selected, search_count=search_count).astype(np.float32)
        slot = slot_features(obs_attached, float(elapsed), int(search_count), int(track_count), int(last), 200.0).astype(np.float32)
        device = next(self.base.model.parameters()).device
        with torch.inference_mode():
            xt = torch.from_numpy(x).float().unsqueeze(0).to(device)
            st = torch.from_numpy(slot).float().unsqueeze(0).to(device)
            pair = self._action_pair_from_row(int(row)).to(device)
            cls, tok, _sel, _active = self.base.model.backbone.encode_tokens(xt)
            pred = self.leaf_g.predict_action_value(cls, tok, st, pair)
            denorm = pred * self.leaf_value_std + self.leaf_value_mean
        return float(self.direct_child_value_weight * float(denorm.detach().cpu()[0]))

    def _valid_scores(self, obs: dict, selected: set[int], elapsed: float, search_count: int, track_count: int, last: int, remaining: float) -> np.ndarray:
        scores = np.asarray(
            self.base.score_actions(
                obs,
                selected=selected,
                elapsed=float(elapsed),
                search_count=int(search_count),
                track_count=int(track_count),
                last=int(last),
            )[:, 0],
            dtype=np.float64,
        ).copy()
        active = np.asarray(obs.get("active_mask", np.zeros(MAXT, dtype=bool)), dtype=bool)
        deadline = np.asarray(obs.get("t_deadline", np.full(MAXT, -1.0, dtype=np.float32)), dtype=np.float32)
        dwell = np.asarray(obs.get("t_dwell", np.ones(MAXT, dtype=np.float32) * 10.0), dtype=np.float32)
        if remaining < 10.0:
            scores[0] = -1.0e18
        n = min(MAXT, len(active), len(deadline), len(dwell), len(scores) - 1)
        for row in range(1, n + 1):
            if (not bool(active[row - 1])) or float(deadline[row - 1]) < 0.0 or row in selected:
                scores[row] = -1.0e18
            elif float(dwell[row - 1]) > float(remaining) and selected:
                scores[row] = -1.0e18
        if n + 1 < len(scores):
            scores[n + 1 :] = -1.0e18
        return scores

    def _candidate_actions(self, obs: dict, node: Node, remaining: float, *, is_root: bool = False) -> list[tuple[int, float]]:
        scores = self._valid_scores(obs, node.selected, node.elapsed, node.search_count, node.track_count, node.last, remaining)
        valid = np.where(np.isfinite(scores) & (scores > -1.0e17))[0]
        if len(valid) == 0:
            return []
        order = valid[np.argsort(scores[valid])[::-1]]
        if self.expand_top_k > 0:
            order = order[: max(1, self.expand_top_k)]
        logits = scores[order]
        logits = logits - float(np.max(logits))
        probs = np.exp(np.clip(logits, -60.0, 60.0))
        probs = probs / max(1.0e-12, float(np.sum(probs)))
        if self.prior_uniform_mix > 0.0:
            uniform = np.full_like(probs, 1.0 / float(len(probs)))
            probs = (1.0 - self.prior_uniform_mix) * probs + self.prior_uniform_mix * uniform
        if is_root and self.root_dirichlet_fraction > 0.0 and len(probs) > 1:
            noise = self.rng.dirichlet(np.full((len(probs),), self.root_dirichlet_alpha, dtype=np.float64))
            probs = (1.0 - self.root_dirichlet_fraction) * probs + self.root_dirichlet_fraction * noise
        out = []
        for row, prior in zip(order, probs):
            action = xs_s_search_action(MAXT) if int(row) == 0 else xs_s_track_action(int(row), MAXT)
            out.append((int(action), float(prior)))
        return out

    def _transition(self, eng, node: Node, action: int, remaining: float) -> Node | None:
        binding.vec_restore(eng.env, node.snapshot)
        obs_before = get_obs(eng, node.debt)
        reward, dt, executed = execute_first_valid_action(eng, [int(action)], float(remaining))
        if executed is None or dt <= 0.0:
            return None
        base_row, _sensor = xs_decode_action(int(executed), MAXT)
        next_debt = 0.0 if int(base_row) == 0 else float(node.debt) + float(dt)
        obs_after = get_obs(eng, next_debt)
        shaped = shaped_step_reward(float(reward), float(dt), obs_before, obs_after, self.env_cfg, action=int(executed))
        selected = set(node.selected)
        search_count = int(node.search_count)
        track_count = int(node.track_count)
        last = int(base_row)
        if int(base_row) == 0:
            search_count += 1
        else:
            selected.add(int(base_row))
            track_count += 1
        return Node(
            snapshot=binding.vec_snapshot(eng.env),
            debt=float(next_debt),
            selected=selected,
            elapsed=float(node.elapsed) + float(dt),
            search_count=search_count,
            track_count=track_count,
            last=last,
            reward=float(shaped),
            dt=float(dt),
        )

    def _rollout_value(self, eng, node: Node, remaining: float) -> float:
        if self.rollout_steps <= 0 or (remaining <= 0.0 and self.rollout_windows <= 1):
            binding.vec_restore(eng.env, node.snapshot)
            obs0 = get_obs(eng, node.debt)
            return self._terminal_value(obs0) + self._learned_leaf_value(obs0, node.selected, node.elapsed, node.search_count, node.track_count, node.last, remaining)
        binding.vec_restore(eng.env, node.snapshot)
        debt = float(node.debt)
        selected = set(node.selected)
        elapsed = float(node.elapsed)
        search_count = int(node.search_count)
        track_count = int(node.track_count)
        last = int(node.last)
        total = 0.0
        gamma = 1.0
        rem = float(remaining)
        steps = 0
        windows_used = 1
        while steps < self.rollout_steps * self.rollout_windows:
            if bool(eng.term_buf[0]):
                break
            if rem <= 0.0:
                if windows_used >= self.rollout_windows:
                    break
                windows_used += 1
                selected = set()
                elapsed = 0.0
                search_count = 0
                track_count = 0
                last = -1
                rem = 200.0
            obs = get_obs(eng, debt)
            scores = self._valid_scores(obs, selected, elapsed, search_count, track_count, last, rem)
            if not np.any(np.isfinite(scores) & (scores > -1.0e17)):
                break
            row = int(np.nanargmax(scores))
            action = xs_s_search_action(MAXT) if row == 0 else xs_s_track_action(row, MAXT)
            obs_before = get_obs(eng, debt)
            reward, dt, executed = execute_first_valid_action(eng, [int(action)], rem)
            if executed is None or dt <= 0.0:
                break
            base_row, _sensor = xs_decode_action(int(executed), MAXT)
            next_debt = 0.0 if int(base_row) == 0 else debt + float(dt)
            obs_after = get_obs(eng, next_debt)
            shaped = shaped_step_reward(float(reward), float(dt), obs_before, obs_after, self.env_cfg, action=int(executed))
            total += gamma * float(shaped)
            gamma *= self.discount
            debt = next_debt
            rem -= float(dt)
            elapsed += float(dt)
            if int(base_row) == 0:
                search_count += 1
            else:
                selected.add(int(base_row))
                track_count += 1
            last = int(base_row)
            steps += 1
        obs_leaf = get_obs(eng, debt)
        total += gamma * (
            self._terminal_value(obs_leaf)
            + self._learned_leaf_value(obs_leaf, selected, elapsed, search_count, track_count, last, rem)
        )
        return float(total)

    def _prepare_candidates(self, eng, node: Node, remaining: float, *, is_root: bool = False) -> None:
        if node.candidates_ready:
            return
        binding.vec_restore(eng.env, node.snapshot)
        obs = get_obs(eng, node.debt)
        node.pending = self._candidate_actions(obs, node, remaining, is_root=is_root)
        node.pending.sort(key=lambda item: item[1], reverse=True)
        if is_root and self.stratify_root_types and len(node.pending) > 1:
            # Progressive widening must evaluate both structured action types.
            # Otherwise one search action can be excluded by dozens of track
            # candidates before its value is ever observed.
            search = [item for item in node.pending if xs_decode_action(int(item[0]), MAXT)[0] == 0]
            track = [item for item in node.pending if xs_decode_action(int(item[0]), MAXT)[0] > 0]
            if search and track:
                first_search = search[0]
                first_track = track[0]
                used = {int(first_search[0]), int(first_track[0])}
                rest = [item for item in node.pending if int(item[0]) not in used]
                node.pending = [first_search, first_track, *rest]
        node.candidates_ready = True

    def _widening_limit(self, node: Node) -> int:
        total = len(node.children) + len(node.pending)
        if total <= 0:
            return 0
        width = int(math.ceil(self.progressive_widening_c * ((node.visits + 1.0) ** self.progressive_widening_alpha)))
        return min(total, max(1, width))

    def _materialize_child(self, eng, node: Node, remaining: float) -> Node | None:
        while node.pending:
            action, prior = node.pending.pop(0)
            binding.vec_restore(eng.env, node.snapshot)
            obs = get_obs(eng, node.debt)
            parent_row, _ = xs_decode_action(int(action), MAXT)
            direct_q = self._learned_action_value(obs, int(parent_row), node.selected, node.elapsed, node.search_count, node.track_count, node.last)
            child = self._transition(eng, node, int(action), remaining)
            if child is None:
                continue
            child.prior = float(prior)
            if self.init_child_rollouts:
                child_remaining = max(0.0, float(remaining) - float(child.dt))
                child_value = self._rollout_value(eng, child, child_remaining)
                child.visits = 1
                child.value_sum = float(child_value)
            elif direct_q != 0.0:
                child.visits = 1
                child.value_sum = float((direct_q - child.reward) / max(1.0e-6, self.discount))
            node.children[int(action)] = child
            return child
        return None

    def _ucb(self, parent: Node, child: Node) -> float:
        q = child.reward + self.discount * child.value() if child.visits > 0 else 0.0
        u = self.c_puct * child.prior * math.sqrt(parent.visits + 1.0) / (1.0 + child.visits)
        return float(q + u)

    def _run_search(self, eng, root: Node, remaining: float) -> None:
        self._prepare_candidates(eng, root, remaining, is_root=True)
        for _ in range(max(1, self.simulations)):
            node = root
            path = [node]
            rem = float(remaining)
            while True:
                self._prepare_candidates(eng, node, rem, is_root=(node is root))
                if node.pending and len(node.children) < self._widening_limit(node):
                    child = self._materialize_child(eng, node, rem)
                    if child is not None:
                        node = child
                        path.append(node)
                        rem = max(0.0, rem - float(node.dt))
                    break
                if not node.children:
                    break
                _action, node = max(node.children.items(), key=lambda kv: self._ucb(path[-1], kv[1]))
                path.append(node)
                rem = max(0.0, rem - float(node.dt))
                if node.visits == 0:
                    break
            value = self._rollout_value(eng, node, rem)
            for n in reversed(path):
                n.visits += 1
                n.value_sum += value
                value = n.reward + self.discount * value

    def _select(self, root: Node) -> int:
        if self.select_mode == "q":
            visited = [(action, child) for action, child in root.children.items() if child.visits > 0]
            pool = visited if visited else list(root.children.items())
            return int(max(pool, key=lambda kv: (kv[1].reward + self.discount * kv[1].value(), kv[1].visits, kv[1].prior))[0])
        if self.select_mode == "prior":
            return int(max(root.children.items(), key=lambda kv: (kv[1].prior, kv[1].visits))[0])
        return int(max(root.children.items(), key=lambda kv: (kv[1].visits, kv[1].reward + self.discount * kv[1].value()))[0])

    def choose_action(self, eng, debt: float, selected: set[int], elapsed: float, search_count: int, track_count: int, last: int, remaining: float) -> int:
        root_snapshot = binding.vec_snapshot(eng.env)
        root = Node(root_snapshot, float(debt), set(selected), float(elapsed), int(search_count), int(track_count), int(last))
        self._run_search(eng, root, remaining)
        binding.vec_restore(eng.env, root_snapshot)
        if not root.children:
            return xs_s_search_action(MAXT)
        return self._select(root)

    def root_distribution(
        self,
        eng,
        debt: float,
        selected: set[int],
        elapsed: float,
        search_count: int,
        track_count: int,
        last: int,
        remaining: float,
        *,
        target: str = "visits",
        temperature: float = 0.5,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the PUCT-improved root distribution over S-only action rows.

        Rows use the S-only convention: row 0 is search, row i>0 tracks
        target i. The environment is restored before returning so callers can
        use this as an offline teacher target without consuming the real step.
        """
        root_snapshot = binding.vec_snapshot(eng.env)
        root = Node(root_snapshot, float(debt), set(selected), float(elapsed), int(search_count), int(track_count), int(last))
        self._run_search(eng, root, remaining)
        binding.vec_restore(eng.env, root_snapshot)
        if not root.children:
            rows = np.asarray([0], dtype=np.int64)
            probs = np.asarray([1.0], dtype=np.float32)
            q_values = np.asarray([0.0], dtype=np.float32)
            visits = np.asarray([0.0], dtype=np.float32)
            return xs_s_search_action(MAXT), rows, probs, q_values, visits

        entries: list[tuple[int, int, float, float, float]] = []
        for action, child in root.children.items():
            row, _sensor = xs_decode_action(int(action), MAXT)
            q = float(child.reward + self.discount * child.value())
            entries.append((int(action), int(row), float(child.visits), q, float(child.prior)))

        selected_action = self._select(root)
        rows = np.asarray([e[1] for e in entries], dtype=np.int64)
        visits = np.asarray([e[2] for e in entries], dtype=np.float32)
        q_values = np.asarray([e[3] for e in entries], dtype=np.float32)
        priors = np.asarray([e[4] for e in entries], dtype=np.float32)
        mode = str(target)
        if mode == "q_softmax":
            tau = max(1.0e-4, float(temperature))
            logits = (q_values - float(np.max(q_values))) / tau
            mass = np.exp(np.clip(logits, -60.0, 60.0)).astype(np.float32)
        elif mode == "prior":
            mass = priors.astype(np.float32)
        else:
            mass = visits.astype(np.float32)
        total = float(np.sum(mass))
        if not np.isfinite(total) or total <= 0.0:
            mass = priors.astype(np.float32)
            total = float(np.sum(mass))
        if not np.isfinite(total) or total <= 0.0:
            mass = np.ones_like(visits, dtype=np.float32)
            total = float(np.sum(mass))
        probs = (mass / max(total, 1.0e-12)).astype(np.float32)
        return int(selected_action), rows, probs, q_values, visits

    def choose(self, eng, debt: float, obs: dict):
        root_snapshot = binding.vec_snapshot(eng.env)
        selected: set[int] = set()
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        plan: list[int] = []
        for _ in range(32):
            if elapsed >= 200.0 or bool(eng.term_buf[0]):
                break
            action = self.choose_action(eng, debt, selected, elapsed, search_count, track_count, last, 200.0 - elapsed)
            plan.append(int(action))
            reward, dt, executed = execute_first_valid_action(eng, [int(action)], 200.0 - elapsed)
            if executed is None or dt <= 0.0:
                break
            row, _sensor = xs_decode_action(int(executed), MAXT)
            if int(row) == 0:
                search_count += 1
                debt = 0.0
            else:
                selected.add(int(row))
                track_count += 1
                debt = float(debt) + float(dt)
            elapsed += float(dt)
            last = int(row)
        binding.vec_restore(eng.env, root_snapshot)
        return (plan if plan else [xs_s_search_action(MAXT)]), {"teacher": "exact_env_puct"}


def run_exact_puct_eval(planner: ExactSOnlyPuctPlanner, name: str, initial: int, seed: int, windows: int, env_cfg: dict) -> pd.DataFrame:
    eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
    eng.reset(seed=int(seed))
    debt = 0.0
    cumulative = 0.0
    rows = []
    try:
        for window in range(int(windows)):
            if bool(eng.term_buf[0]):
                break
            spent = 0.0
            reward_total = 0.0
            executed_n = 0
            search_n = 0
            selected: set[int] = set()
            search_count = 0
            track_count = 0
            last = -1
            t0 = time.perf_counter()
            while spent < 200.0 and not bool(eng.term_buf[0]):
                remaining = 200.0 - spent
                action = planner.choose_action(eng, debt, selected, spent, search_count, track_count, last, remaining)
                obs_before = get_obs(eng, debt)
                reward, dt, executed = execute_first_valid_action(eng, [int(action)], remaining)
                if executed is None or dt <= 0.0:
                    break
                row, _sensor = xs_decode_action(int(executed), MAXT)
                next_debt = 0.0 if int(row) == 0 else float(debt) + float(dt)
                obs_after = get_obs(eng, next_debt)
                shaped = shaped_step_reward(float(reward), float(dt), obs_before, obs_after, env_cfg, action=int(executed))
                reward_total += float(shaped)
                spent += float(dt)
                debt = float(next_debt)
                executed_n += 1
                if int(row) == 0:
                    search_n += 1
                    search_count += 1
                else:
                    selected.add(int(row))
                    track_count += 1
                last = int(row)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            plan_ms = 1000.0 * (time.perf_counter() - t0)
            metrics = sample_state_metrics(eng, debt)
            cumulative += float(reward_total)
            rows.append(
                {
                    "planner": name,
                    "seed": int(seed),
                    "window": int(window),
                    "window_reward": float(reward_total),
                    "cumulative_reward": float(cumulative),
                    "search_fraction": float(search_n / max(1, executed_n)),
                    "planning_ms_per_window": float(plan_ms),
                    "planning_ms_per_decision": float(plan_ms),
                    "executed_actions": int(executed_n),
                    "spent_ms": float(spent),
                    **metrics,
                }
            )
    finally:
        eng.close()
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-state", default=str(ROOT / "CreateValid1" / "results" / "single_sensor_fair_exact_action_attention_train_two_row_action_attention_qpolicy_factored_loss.pt"))
    ap.add_argument("--variant", default="two_row_action_attention")
    ap.add_argument("--out", default=str(ROOT / "CreateValid1" / "results" / "current_sonly_exact_puct.csv"))
    ap.add_argument("--initials", default="40")
    ap.add_argument("--rates", default="3")
    ap.add_argument("--seeds", default="916")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--simulations", type=int, default=16)
    ap.add_argument("--expand-top-k", type=int, default=8)
    ap.add_argument("--rollout-steps", type=int, default=4)
    ap.add_argument("--rollout-windows", type=int, default=1)
    ap.add_argument("--init-child-rollouts", action="store_true")
    ap.add_argument("--leaf-g-state", default="")
    ap.add_argument("--leaf-value-weight", type=float, default=0.0)
    ap.add_argument("--leaf-value-top-k", type=int, default=16)
    ap.add_argument("--direct-child-value-weight", type=float, default=0.0)
    ap.add_argument("--prior-uniform-mix", type=float, default=0.0)
    ap.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    ap.add_argument("--root-dirichlet-fraction", type=float, default=0.0)
    ap.add_argument("--progressive-widening-c", type=float, default=2.0)
    ap.add_argument("--progressive-widening-alpha", type=float, default=0.5)
    ap.add_argument("--stratify-root-types", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--c-puct", type=float, default=1.25)
    ap.add_argument("--discount", type=float, default=0.997)
    ap.add_argument("--select-mode", choices=["visits", "q", "prior"], default="q")
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--q-weight", type=float, default=0.5)
    ap.add_argument("--search-bias", type=float, default=0.0)
    ap.add_argument("--terminal-service-weight", type=float, default=0.0)
    ap.add_argument("--terminal-search-frame-weight", type=float, default=0.0)
    ap.add_argument("--include-baselines", action="store_true")
    ap.add_argument("--env-mode", default="pufferlib_service")
    ap.add_argument("--search-frame-overdue-weight", type=float, default=0.5)
    ap.add_argument("--search-frame-drop-penalty", type=float, default=8.0)
    ap.add_argument("--search-frame-state-penalty-weight", type=float, default=2.0)
    ap.add_argument("--search-frame-delta-reward-weight", type=float, default=5.0)
    ap.add_argument("--service-pressure-delta-reward-weight", type=float, default=0.30)
    ap.add_argument("--serviced-pressure-improvement-reward-weight", type=float, default=0.15)
    ap.add_argument("--discovered-target-reward", type=float, default=0.08)
    ap.add_argument("--tracked-count-delta-reward-weight", type=float, default=0.0)
    args = ap.parse_args()
    torch.set_num_threads(1)
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = False
    exact_args.single_sensor = True
    exact_args.serviced_pressure_improvement_reward_weight = float(args.serviced_pressure_improvement_reward_weight)
    frames = []
    for initial in parse_ints(args.initials):
        for rate in parse_floats(args.rates):
            env_cfg = env_cfg_for(float(rate), exact_args)
            env_cfg["enable_x_band"] = 0
            for seed in parse_ints(args.seeds):
                planner = ExactSOnlyPuctPlanner(
                    str(args.base_state),
                    str(args.variant),
                    env_cfg,
                    device=str(args.device),
                    simulations=int(args.simulations),
                    expand_top_k=int(args.expand_top_k),
                    rollout_steps=int(args.rollout_steps),
                    c_puct=float(args.c_puct),
                    discount=float(args.discount),
                    select_mode=str(args.select_mode),
                    policy_weight=float(args.policy_weight),
                    q_weight=float(args.q_weight),
                    search_bias=float(args.search_bias),
                        terminal_service_weight=float(args.terminal_service_weight),
                        terminal_search_frame_weight=float(args.terminal_search_frame_weight),
                        rollout_windows=int(args.rollout_windows),
                        init_child_rollouts=bool(args.init_child_rollouts),
                        leaf_g_state=str(args.leaf_g_state),
                        leaf_value_weight=float(args.leaf_value_weight),
                        leaf_value_top_k=int(args.leaf_value_top_k),
                        direct_child_value_weight=float(args.direct_child_value_weight),
                        prior_uniform_mix=float(args.prior_uniform_mix),
                        root_dirichlet_alpha=float(args.root_dirichlet_alpha),
                        root_dirichlet_fraction=float(args.root_dirichlet_fraction),
                    progressive_widening_c=float(args.progressive_widening_c),
                    progressive_widening_alpha=float(args.progressive_widening_alpha),
                    stratify_root_types=bool(args.stratify_root_types),
                    )
                df = run_exact_puct_eval(planner, f"ExactEnvPUCT_s{args.simulations}_k{args.expand_top_k}", int(initial), int(seed), int(args.windows), env_cfg)
                df["init"] = int(initial)
                df["rate"] = float(rate)
                frames.append(df)
                if bool(args.include_baselines):
                    from eval_action_attention_muzero_g import run_plan_eval

                    for name, bp in {"EDF": EDFPlanner(MAXT), "EST": ESTPlanner(MAXT)}.items():
                        bdf, _ = run_plan_eval(bp, name, int(initial), int(seed), int(args.windows), env_cfg)
                        bdf["init"] = int(initial)
                        bdf["rate"] = float(rate)
                        frames.append(bdf)
    out = pd.concat(frames, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    summary = (
        out.groupby("planner")
        .agg(
            reward=("window_reward", "mean"),
            total_reward=("cumulative_reward", "last"),
            drop_pct=("drop_pct_active", "mean"),
            tracked=("tracked_targets", "mean"),
            delay=("mean_delay_active", "mean"),
            search=("search_fraction", "mean"),
            latency_ms=("planning_ms_per_window", "mean"),
            n=("window_reward", "count"),
        )
        .reset_index()
        .sort_values("reward", ascending=False)
    )
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False), flush=True)
    print({"out": str(out_path), "summary": str(summary_path)}, flush=True)


if __name__ == "__main__":
    main()
