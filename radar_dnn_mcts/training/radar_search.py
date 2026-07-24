from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from radar_dnn_mcts.env.config import RewardConfig
from radar_dnn_mcts.env.features import FeatureBuilder
from radar_dnn_mcts.env.reward import transition_reward
from radar_dnn_mcts.env.transition import RadarObservationTransition
from radar_dnn_mcts.models.scheduler import RadarSchedulerModel


def _device_of(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array).unsqueeze(0).to(device)


def _selected_mask(rows: int, selected: set[int], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(1, rows, dtype=torch.bool, device=device)
    if selected:
        mask[0, torch.as_tensor(sorted(selected), device=device)] = True
    return mask


class RadarWindowSearchState:
    """Cloneable shadow state for searching one scheduling window."""

    def __init__(
        self,
        obs: dict,
        features: FeatureBuilder,
        reward: RewardConfig,
        transition: RadarObservationTransition | None = None,
        budget_ms: float = 200.0,
    ):
        self.obs = deepcopy(obs)
        self.features = features
        self.reward_config = reward
        self.transition = transition or RadarObservationTransition()
        self.budget_ms = float(budget_ms)
        self.root_obs = deepcopy(obs)
        self.elapsed = 0.0
        self.searches = 0
        self.tracks = 0
        self.last = -1
        self.selected: set[int] = set()
        self.prefix: list[int] = []

    def clone(self):
        return deepcopy(self)

    def legal_actions(self) -> list[int]:
        """Return search plus active, unexpired, unselected targets."""
        if self.elapsed >= self.budget_ms:
            return []
        active = np.asarray(self.obs["active_mask"], dtype=bool)
        deadline = np.asarray(self.obs["t_deadline"], dtype=np.float32)
        tracks = [idx + 1 for idx in np.where(active & (deadline >= 0.0))[0] if idx + 1 not in self.selected]
        return [0, *tracks]

    def step(self, action: int) -> tuple[float, bool]:
        """Advance the shadow state and return its transition reward."""
        before = self.obs
        self.obs, duration = self.transition.step(self.obs, action)
        self.elapsed += duration
        if action == 0:
            self.searches += 1
        else:
            self.tracks += 1
            self.selected.add(int(action))
        self.last = int(action)
        self.prefix.append(int(action))
        return transition_reward(before, self.obs, action, self.reward_config), self.elapsed >= self.budget_ms

    def network_input(self):
        """Build model features at the current searched action prefix."""
        return (
            self.features.tokens(self.obs, self.selected, self.searches),
            self.features.context(self.obs, self.elapsed, self.searches, self.tracks, self.last),
        )

    def root_network_input(self):
        return (
            self.features.tokens(self.root_obs),
            self.features.context(self.root_obs, 0.0, 0, 0, -1),
        )

    def boundary_network_input(self):
        return (
            self.features.tokens(self.obs),
            self.features.context(self.obs, 0.0, 0, 0, -1),
        )


class RadarModelEvaluator:
    """Adapt the shared policy/value model to the PUCT interface."""

    def __init__(self, model: RadarSchedulerModel):
        self.model = model.eval()

    @torch.inference_mode()
    def __call__(self, state: RadarWindowSearchState, legal_actions: list[int]):
        device = _device_of(self.model)
        inputs = state.network_input() if legal_actions else state.boundary_network_input()
        tokens_np, context_np = inputs
        tokens = _tensor(tokens_np, device)
        context = _tensor(context_np, device)
        selected = _selected_mask(tokens.shape[1], state.selected, device)
        output = self.model(tokens, context, selected)
        value = float(output.q_values[0, 0].cpu())
        if not legal_actions:
            return np.empty(0, dtype=np.float32), value
        legal = torch.as_tensor(legal_actions, device=device)
        policy = torch.softmax(output.policy_logits[0, legal], dim=-1)
        return policy.cpu().numpy(), value


class RadarAREvaluator:
    """Evaluate a PUCT prefix through the causal sequence decoder."""

    def __init__(self, decoder):
        self.decoder = decoder.eval()

    @torch.inference_mode()
    def __call__(self, state: RadarWindowSearchState, legal_actions: list[int]):
        device = _device_of(self.decoder)
        root_tokens_np, root_context_np = state.root_network_input()
        root_tokens = _tensor(root_tokens_np, device)
        root_context = _tensor(root_context_np, device)
        encoded, _ = self.decoder.initial(root_tokens, root_context)
        prefix = torch.as_tensor(state.prefix, dtype=torch.long, device=device).unsqueeze(0)
        context_np = (
            state.network_input()[1]
            if legal_actions
            else state.boundary_network_input()[1]
        )
        context = _tensor(context_np, device)
        output = self.decoder.step(
            encoded,
            prefix,
            context,
            _selected_mask(root_tokens.shape[1], state.selected, device),
        )
        value = float(output.q_values[0, 0].cpu())
        if not legal_actions:
            return np.empty(0, dtype=np.float32), value
        legal = torch.as_tensor(legal_actions, device=device)
        policy = torch.softmax(output.policy_logits[0, legal], dim=-1)
        return policy.cpu().numpy(), value
