from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from radar_dnn_mcts.env.config import RewardConfig
from radar_dnn_mcts.env.features import FeatureBuilder
from radar_dnn_mcts.env.reward import transition_reward
from radar_dnn_mcts.env.transition import RadarObservationTransition
from radar_dnn_mcts.models.scheduler import RadarSchedulerModel
from radar_dnn_mcts.schedulers.base import device_of, selected_mask, tensor


class RadarWindowSearchState:
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
        self.elapsed = 0.0
        self.searches = 0
        self.tracks = 0
        self.last = -1
        self.selected: set[int] = set()

    def clone(self):
        return deepcopy(self)

    def legal_actions(self) -> list[int]:
        if self.elapsed >= self.budget_ms:
            return []
        active = np.asarray(self.obs["active_mask"], dtype=bool)
        deadline = np.asarray(self.obs["t_deadline"], dtype=np.float32)
        tracks = [idx + 1 for idx in np.where(active & (deadline >= 0.0))[0] if idx + 1 not in self.selected]
        return [0, *tracks]

    def step(self, action: int) -> tuple[float, bool]:
        before = self.obs
        self.obs, duration = self.transition.step(self.obs, action)
        self.elapsed += duration
        if action == 0:
            self.searches += 1
        else:
            self.tracks += 1
            self.selected.add(int(action))
        self.last = int(action)
        return transition_reward(before, self.obs, action, self.reward_config), self.elapsed >= self.budget_ms

    def network_input(self):
        return (
            self.features.tokens(self.obs, self.selected, self.searches),
            self.features.context(self.obs, self.elapsed, self.searches, self.tracks, self.last),
        )


class RadarModelEvaluator:
    def __init__(self, model: RadarSchedulerModel, policy_weight: float = 1.0, q_weight: float = 1.0):
        self.model = model.eval()
        self.policy_weight = float(policy_weight)
        self.q_weight = float(q_weight)

    @torch.inference_mode()
    def __call__(self, state: RadarWindowSearchState, legal_actions: list[int]):
        device = device_of(self.model)
        tokens_np, context_np = state.network_input()
        tokens = tensor(tokens_np, device)
        context = tensor(context_np, device)
        selected = selected_mask(tokens.shape[1], state.selected, device)
        output = self.model(tokens, context, selected)
        legal = torch.as_tensor(legal_actions, device=device)
        policy = torch.softmax(output.policy_logits[0, legal], dim=-1)
        q = output.q_values[0, legal]
        return policy.cpu().numpy(), q.cpu().numpy()
