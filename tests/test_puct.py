from copy import copy

import numpy as np

from radar_dnn_mcts.search import PUCT, PUCTConfig


class ToyState:
    def __init__(self, depth=0, total=0, elapsed=0.0):
        self.depth = depth
        self.total = total
        self.elapsed = elapsed

    def clone(self):
        return copy(self)

    def legal_actions(self):
        return [] if self.depth >= 3 else [0, 1]

    def step(self, action):
        self.depth += 1
        self.total += action
        self.elapsed += 1.0
        return float(action), self.depth >= 3

    def network_input(self):
        return np.zeros((2, 1), np.float32), np.zeros((1,), np.float32)


def evaluator(state, legal):
    return np.full(len(legal), 1.0 / max(1, len(legal))), 0.0


def test_puct_prefers_higher_return_action():
    result = PUCT(evaluator, PUCTConfig(rollouts=64, c_puct=1.0)).run(ToyState())
    assert result.trajectory[0].action == 1
    assert abs(sum(result.trajectory[0].policy.values()) - 1.0) < 1e-6


class OneStepState(ToyState):
    def legal_actions(self):
        return [] if self.depth else [0, 1]


def zero_evaluator(state, legal):
    return np.full(len(legal), 1.0 / max(1, len(legal))), 0.0


def test_root_q_includes_immediate_edge_reward():
    result = PUCT(
        zero_evaluator,
        PUCTConfig(rollouts=16, c_puct=0.5, reward_scale=1.0),
    ).run(OneStepState())
    assert result.trajectory[0].action == 1


class EqualRewardState(OneStepState):
    def step(self, action):
        self.depth += 1
        self.total += action
        self.elapsed += 1.0
        return 0.0, True


def boundary_value_evaluator(state, legal):
    return np.full(len(legal), 1.0 / max(1, len(legal))), float(state.total)


def test_puct_uses_boundary_state_value():
    result = PUCT(
        boundary_value_evaluator,
        PUCTConfig(rollouts=16, c_puct=1.0),
    ).run(EqualRewardState())
    assert result.trajectory[0].action == 1


def test_puct_returns_complete_trajectory():
    result = PUCT(evaluator, PUCTConfig(rollouts=16)).run(ToyState())
    assert len(result.trajectory) == 3
