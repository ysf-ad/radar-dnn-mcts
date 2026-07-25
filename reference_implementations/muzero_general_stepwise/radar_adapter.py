"""Radar interfaces for the otherwise unchanged MuZero-General stepwise MCTS."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from radar_dnn_mcts.env.actions import action_duration_ms
from radar_dnn_mcts.env.features import FeatureBuilder
from radar_dnn_mcts.models.backbone import EncodedState
from radar_dnn_mcts.models.dynamics import LatentDynamics
from radar_dnn_mcts.models.scheduler import RadarSchedulerModel

from . import models
from .self_play import MCTS, SelfPlay


@dataclass
class RadarHiddenState:
    """Metadata carried beside MuZero's learned hidden state."""

    latent: EncodedState
    elapsed: float = 0.0
    searches: int = 0
    tracks: int = 0
    last: int = -1
    selected: set[int] = field(default_factory=set)
    prefix: list[int] = field(default_factory=list)

    @property
    def device(self) -> torch.device:
        return self.latent.global_state.device


class RadarMuZeroConfig:
    """Only the fields read by the vendored MuZero-General MCTS."""

    def __init__(
        self,
        action_count: int,
        simulations: int,
        discount: float,
        pb_c_base: float,
        pb_c_init: float,
        support_size: int,
    ):
        self.action_space = list(range(action_count))
        self.players = [0]
        self.num_simulations = int(simulations)
        self.discount = float(discount)
        self.pb_c_base = float(pb_c_base)
        self.pb_c_init = float(pb_c_init)
        self.support_size = int(support_size)
        self.root_dirichlet_alpha = 0.03
        self.root_exploration_fraction = 0.0


class RadarInferenceAdapter(nn.Module):
    """Expose local h/f/g through MuZero-General's inference contract."""

    def __init__(
        self,
        model: RadarSchedulerModel,
        dynamics: LatentDynamics,
        obs: dict,
        features: FeatureBuilder,
        budget_ms: float,
        support_size: int,
        max_steps: int,
    ):
        super().__init__()
        self.model = model
        self.dynamics = dynamics
        self.obs = obs
        self.features = features
        self.budget_ms = float(budget_ms)
        self.support_size = int(support_size)
        self.max_steps = int(max_steps)
        self.g_calls = 0

    def _support_logits(self, value: torch.Tensor) -> torch.Tensor:
        distribution = models.scalar_to_support(
            value.reshape(-1, 1), self.support_size
        ).squeeze(1)
        return distribution.clamp_min(1e-12).log()

    def _context(self, state: RadarHiddenState, *, boundary: bool = False):
        values = (
            self.features.context(self.obs, 0.0, 0, 0, -1)
            if boundary
            else self.features.context(
                self.obs,
                state.elapsed,
                state.searches,
                state.tracks,
                state.last,
            )
        )
        return torch.from_numpy(values).unsqueeze(0).to(state.device)

    def legal_actions(self, state: RadarHiddenState) -> list[int]:
        if state.elapsed >= self.budget_ms or len(state.prefix) >= self.max_steps:
            return []
        active = np.asarray(self.obs["active_mask"], dtype=bool)
        deadline = np.asarray(self.obs["t_deadline"], dtype=np.float32)
        valid = np.asarray(state.latent.valid_rows[0].cpu(), dtype=bool)
        tracks = [
            row
            for row in range(1, min(len(active) + 1, len(valid)))
            if valid[row]
            and active[row - 1]
            and deadline[row - 1] > state.elapsed
            and row not in state.selected
        ]
        return [0, *tracks]

    def _predict(self, state: RadarHiddenState):
        legal = self.legal_actions(state)
        unavailable = torch.ones_like(state.latent.valid_rows)
        if legal:
            unavailable[0, torch.as_tensor(legal, device=state.device)] = False
        output = self.model.predict(
            state.latent,
            self._context(state, boundary=not legal),
            unavailable,
        )
        policy = output.policy_logits.clone()
        if not legal:
            policy.fill_(-1e9)
        value = self._support_logits(output.q_values[:, 0])
        return value, policy

    def initial_inference(self, _observation):
        device = next(self.model.parameters()).device
        tokens = torch.from_numpy(self.features.tokens(self.obs)).unsqueeze(0).to(device)
        context = torch.from_numpy(
            self.features.context(self.obs, 0.0, 0, 0, -1)
        ).unsqueeze(0).to(device)
        state = RadarHiddenState(self.model.encode(tokens, context))
        value, policy = self._predict(state)
        reward = self._support_logits(torch.zeros(1, device=device))
        return value, reward, policy, state

    def recurrent_inference(self, state: RadarHiddenState, action: torch.Tensor):
        row = int(action.reshape(-1)[0])
        latent, reward = self.dynamics(
            state.latent,
            torch.as_tensor([row], dtype=torch.long, device=state.device),
        )
        self.g_calls += 1
        next_state = RadarHiddenState(
            latent=latent,
            elapsed=state.elapsed + action_duration_ms(self.obs, row),
            searches=state.searches + int(row == 0),
            tracks=state.tracks + int(row > 0),
            last=row,
            selected=set(state.selected),
            prefix=[*state.prefix, row],
        )
        if row > 0:
            next_state.selected.add(row)
        value, policy = self._predict(next_state)
        return value, self._support_logits(reward), policy, next_state


class MuZeroGeneralStepwiseScheduler:
    """Run the vendored stop-at-first-leaf MCTS before each planned action."""

    def __init__(
        self,
        model: RadarSchedulerModel,
        dynamics: LatentDynamics,
        features: FeatureBuilder | None = None,
        simulations: int = 8,
        discount: float = 0.99,
        pb_c_base: float = 19652.0,
        pb_c_init: float = 1.25,
        support_size: int = 20,
        max_steps: int = 32,
        random_seed: int = 0,
    ):
        self.model = model.eval()
        self.dynamics = dynamics.eval()
        self.features = features or FeatureBuilder()
        self.simulations = int(simulations)
        self.discount = float(discount)
        self.pb_c_base = float(pb_c_base)
        self.pb_c_init = float(pb_c_init)
        self.support_size = int(support_size)
        self.max_steps = int(max_steps)
        self.random_seed = int(random_seed)
        self.last_g_calls = 0
        self.last_observations = 1
        self.g_call_history: list[int] = []

    @torch.inference_mode()
    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        np.random.seed(self.random_seed)
        config = RadarMuZeroConfig(
            self.features.max_targets + 1,
            self.simulations,
            self.discount,
            self.pb_c_base,
            self.pb_c_init,
            self.support_size,
        )
        adapter = RadarInferenceAdapter(
            self.model,
            self.dynamics,
            obs,
            self.features,
            budget_ms,
            self.support_size,
            self.max_steps,
        )
        root = None
        plan: list[int] = []
        while len(plan) < self.max_steps:
            if root is None:
                active = np.asarray(obs["active_mask"], dtype=bool)
                deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
                legal = [
                    0,
                    *((np.flatnonzero(active & (deadline >= 0.0)) + 1).tolist()),
                ]
            else:
                legal = adapter.legal_actions(root.hidden_state)
            if not legal:
                break
            root, _ = MCTS(config).run(
                adapter,
                np.zeros(1, dtype=np.float32),
                legal,
                0,
                False,
                override_root_with=root,
            )
            action = int(SelfPlay.select_action(root, 0))
            plan.append(action)
            parent = root
            root = parent.children[action]
            if root.hidden_state is None:
                _, reward, policy, hidden = adapter.recurrent_inference(
                    parent.hidden_state,
                    torch.as_tensor([[action]], device=next(adapter.parameters()).device),
                )
                root.expand(adapter.legal_actions(hidden), 0, reward, policy, hidden)
            if root.hidden_state.elapsed >= budget_ms:
                break
        self.last_g_calls = adapter.g_calls
        self.g_call_history.append(adapter.g_calls)
        return plan
