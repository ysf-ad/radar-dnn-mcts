from copy import copy

import numpy as np

from radar_dnn_mcts.search import PUCT, PUCTConfig


class ToyState:
    def __init__(self, depth=0, total=0):
        self.depth = depth
        self.total = total

    def clone(self):
        return copy(self)

    def legal_actions(self):
        return [] if self.depth >= 3 else [0, 1]

    def step(self, action):
        self.depth += 1
        self.total += action
        return float(action), self.depth >= 3

    def network_input(self):
        return np.zeros((2, 1), np.float32), np.zeros((1,), np.float32)


def evaluator(state, legal):
    return np.asarray([0.5, 0.5]), np.asarray([0.0, 0.5])


def test_puct_prefers_higher_return_action():
    result = PUCT(evaluator, PUCTConfig(simulations=64, c_puct=1.0)).run(ToyState())
    assert result.action == 1
    assert abs(sum(result.policy.values()) - 1.0) < 1e-6


class OneStepState(ToyState):
    def legal_actions(self):
        return [] if self.depth else [0, 1]


def zero_evaluator(state, legal):
    return np.full(len(legal), 1.0 / len(legal)), np.zeros(len(legal))


def test_root_q_includes_immediate_edge_reward():
    result = PUCT(zero_evaluator, PUCTConfig(simulations=16, c_puct=0.5)).run(OneStepState())
    assert result.action == 1
    assert result.q_values[1] == 1.0


class EqualRewardState(OneStepState):
    def step(self, action):
        self.depth += 1
        return 0.0, True


def action_q_evaluator(state, legal):
    return np.full(len(legal), 1.0 / len(legal)), np.asarray([0.0, 1.0])


def test_default_puct_uses_predicted_action_q():
    result = PUCT(action_q_evaluator, PUCTConfig(simulations=8, c_puct=0.0)).run(EqualRewardState())
    assert result.action == 1
