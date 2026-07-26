from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from radar_dnn_mcts.env.actions import action_duration_ms
from radar_dnn_mcts.env.features import FeatureBuilder
from radar_dnn_mcts.models.backbone import EncodedState
from radar_dnn_mcts.models.dynamics import LatentDynamics
from radar_dnn_mcts.models.scheduler import RadarSchedulerModel
from radar_dnn_mcts.search.puct import PUCT, PUCTConfig, SearchResult


@dataclass
class DynamicsState:
    """One branch-local latent state used by the shared PUCT algorithm."""

    obs: dict
    features: FeatureBuilder
    dynamics: LatentDynamics
    latent: EncodedState
    budget_ms: float
    max_steps: int
    counter: list[int]
    cache: dict[tuple[int, ...], tuple[EncodedState, float]]
    elapsed: float = 0.0
    searches: int = 0
    tracks: int = 0
    last: int = -1
    selected: set[int] = field(default_factory=set)
    prefix: list[int] = field(default_factory=list)

    def clone(self) -> "DynamicsState":
        state = copy(self)
        state.selected = set(self.selected)
        state.prefix = list(self.prefix)
        return state

    def legal_actions(self) -> list[int]:
        if self.elapsed >= self.budget_ms or len(self.prefix) >= self.max_steps:
            return []
        active = np.asarray(self.obs["active_mask"], dtype=bool)
        tracked = np.asarray(
            self.obs.get("tracked_mask", active),
            dtype=bool,
        )
        deadline = np.asarray(self.obs["t_deadline"], dtype=np.float32)
        tracks = [
            row
            for row in range(1, len(active) + 1)
            if active[row - 1]
            and tracked[row - 1]
            and deadline[row - 1] > self.elapsed
            and row not in self.selected
        ]
        return [0, *tracks]

    def step(self, action: int) -> tuple[float, bool]:
        key = (*self.prefix, int(action))
        cached = self.cache.get(key)
        if cached is None:
            device = self.latent.global_state.device
            latent, reward = self.dynamics(
                self.latent,
                torch.as_tensor([action], dtype=torch.long, device=device),
            )
            cached = (latent, float(reward[0]))
            self.cache[key] = cached
            self.counter[0] += 1
        self.latent, reward = cached
        self.elapsed += action_duration_ms(self.obs, action)
        if action == 0:
            self.searches += 1
        else:
            self.tracks += 1
            self.selected.add(int(action))
        self.last = int(action)
        self.prefix.append(int(action))
        return reward, not self.legal_actions()


class DynamicsPUCT(PUCT):
    """Shared PUCT whose state transition is learned dynamics g."""

    def __init__(
        self,
        model: RadarSchedulerModel,
        dynamics: LatentDynamics,
        config: PUCTConfig,
        features: FeatureBuilder | None = None,
        max_steps: int = 32,
    ):
        self.model = model.eval()
        self.dynamics = dynamics.eval()
        self.features = features or FeatureBuilder()
        self.max_steps = int(max_steps)
        self.last_g_calls = 0
        self.last_observations = 1
        self.last_h_calls = 0
        self.last_f_calls = 0
        self.window_g_calls = 0
        self.window_h_calls = 0
        self.window_f_calls = 0
        self.g_call_history: list[int] = []
        super().__init__(self._evaluate, config)

    def reset_window_counters(self) -> None:
        self.window_g_calls = 0
        self.window_h_calls = 0
        self.window_f_calls = 0

    @torch.inference_mode()
    def _evaluate(
        self, state: DynamicsState, legal_actions: list[int]
    ) -> tuple[np.ndarray, float]:
        self.window_f_calls += 1
        device = state.latent.global_state.device
        context_values = (
            state.features.context(
                state.obs, state.elapsed, state.searches, state.tracks, state.last
            )
            if legal_actions
            else state.features.context(state.obs, 0.0, 0, 0, -1)
        )
        context = torch.from_numpy(context_values).unsqueeze(0).to(device)
        unavailable = torch.zeros_like(state.latent.valid_rows)
        if legal_actions and state.selected:
            rows = torch.as_tensor(sorted(state.selected), device=device)
            unavailable[0, rows] = True
        output = self.model.predict(state.latent, context, unavailable)
        value = float(output.q_values[0, 0])
        if not legal_actions:
            return np.empty(0, dtype=np.float32), value
        legal = torch.as_tensor(legal_actions, device=device)
        policy = torch.softmax(output.policy_logits[0, legal], dim=-1)
        return policy.cpu().numpy(), value

    @torch.inference_mode()
    def initial_state(
        self,
        obs: dict,
        budget_ms: float = 200.0,
    ) -> DynamicsState:
        """Encode one radar observation into a reusable latent root."""
        self.window_h_calls += 1
        device = next(self.model.parameters()).device
        tokens = torch.from_numpy(self.features.tokens(obs)).unsqueeze(0).to(device)
        context = torch.from_numpy(
            self.features.context(obs, 0.0, 0, 0, -1)
        ).unsqueeze(0).to(device)
        counter = [0]
        return DynamicsState(
            obs=obs,
            features=self.features,
            dynamics=self.dynamics,
            latent=self.model.encode(tokens, context),
            budget_ms=float(budget_ms),
            max_steps=self.max_steps,
            counter=counter,
            cache={},
        )

    @torch.inference_mode()
    def run_state(
        self,
        root: DynamicsState,
        *,
        training: bool = False,
    ) -> SearchResult:
        """Search from an existing latent prefix without re-running h."""
        before = root.counter[0]
        result = super().run(root, training=training)
        self.last_g_calls = root.counter[0] - before
        self.window_g_calls += self.last_g_calls
        self.last_h_calls = self.window_h_calls
        self.last_f_calls = self.window_f_calls
        self.g_call_history.append(self.last_g_calls)
        return result

    @torch.inference_mode()
    def run_observation(
        self,
        obs: dict,
        budget_ms: float = 200.0,
        *,
        training: bool = False,
    ) -> SearchResult:
        self.reset_window_counters()
        return self.run_state(
            self.initial_state(obs, budget_ms),
            training=training,
        )

    def run(self, root_state, *, training: bool = False) -> SearchResult:
        """Match PUCT.run while replacing its shadow state with a latent root."""
        if isinstance(root_state, DynamicsState):
            return self.run_state(root_state, training=training)
        return self.run_observation(
            root_state.obs,
            root_state.budget_ms,
            training=training,
        )
