"""Radar runner for the minimally modified MuZero-General windowed MCTS."""

from __future__ import annotations

import numpy as np
import torch

from reference_implementations.muzero_general_stepwise.radar_adapter import (
    RadarInferenceAdapter,
    RadarMuZeroConfig,
)

from .self_play import MCTS, SelfPlay


class MuZeroGeneralWindowedScheduler:
    """Run complete-window simulations in the vendored MuZero-General tree."""

    def __init__(
        self,
        model,
        dynamics,
        features=None,
        simulations: int = 8,
        discount: float = 0.99,
        pb_c_base: float = 19652.0,
        pb_c_init: float = 1.25,
        support_size: int = 20,
        max_steps: int = 32,
        random_seed: int = 0,
    ):
        from radar_dnn_mcts.env.features import FeatureBuilder

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
        active = np.asarray(obs["active_mask"], dtype=bool)
        deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
        legal = [0, *((np.flatnonzero(active & (deadline >= 0.0)) + 1).tolist())]
        root, _ = MCTS(config).run(
            adapter,
            np.zeros(1, dtype=np.float32),
            legal,
            0,
            False,
        )

        # WINDOWED CHANGE: extract one schedule after all complete rollouts.
        plan: list[int] = []
        node = root
        while node.children and len(plan) < self.max_steps:
            action = int(SelfPlay.select_action(node, 0))
            child = node.children[action]
            if child.hidden_state is None:
                _, reward, policy, hidden = adapter.recurrent_inference(
                    node.hidden_state,
                    torch.as_tensor(
                        [[action]], device=next(adapter.parameters()).device
                    ),
                )
                child.expand(
                    adapter.legal_actions(hidden),
                    0,
                    reward,
                    policy,
                    hidden,
                )
            plan.append(action)
            node = child
            if node.hidden_state.elapsed >= budget_ms:
                break

        self.last_g_calls = adapter.g_calls
        self.g_call_history.append(adapter.g_calls)
        return plan
