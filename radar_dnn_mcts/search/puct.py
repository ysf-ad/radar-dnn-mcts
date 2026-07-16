from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class SearchState(Protocol):
    def clone(self) -> "SearchState": ...
    def legal_actions(self) -> list[int]: ...
    def step(self, action: int) -> tuple[float, bool]: ...
    def network_input(self): ...


class PolicyQEvaluator(Protocol):
    def __call__(self, state: SearchState, legal_actions: list[int]) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class PUCTConfig:
    simulations: int = 32
    c_puct: float = 1.5
    discount: float = 0.997
    learned_q_weight: float = 1.0
    temperature: float = 1.0


@dataclass
class Node:
    prior: float = 1.0
    initial_q: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    reward: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class SearchResult:
    action: int
    policy: dict[int, float]
    q_values: dict[int, float]
    root_value: float


class PUCT:
    """AlphaZero/MuZero-style tree search with optional learned action-Q guidance."""

    def __init__(self, evaluator: PolicyQEvaluator, config: PUCTConfig = PUCTConfig()):
        self.evaluator = evaluator
        self.config = config

    def _expand(self, node: Node, state: SearchState) -> float:
        legal = state.legal_actions()
        if not legal:
            return 0.0
        priors, q_values = self.evaluator(state, legal)
        priors = np.asarray(priors, dtype=np.float64)
        q_values = np.asarray(q_values, dtype=np.float64)
        priors = np.maximum(priors, 0.0)
        priors = priors / max(priors.sum(), 1e-12)
        for action, prior, q_value in zip(legal, priors, q_values):
            node.children[int(action)] = Node(prior=float(prior), initial_q=float(q_value))
        return self.config.learned_q_weight * float(np.dot(priors, q_values))

    def _select(self, node: Node) -> tuple[int, Node]:
        root_scale = np.sqrt(node.visits + 1.0)
        return max(
            node.children.items(),
            key=lambda item: (
                item[1].reward
                + self.config.discount * item[1].value
                + self.config.learned_q_weight * item[1].initial_q
                + self.config.c_puct * item[1].prior * root_scale / (1 + item[1].visits)
            ),
        )

    def run(self, root_state: SearchState) -> SearchResult:
        root = Node()
        self._expand(root, root_state)
        for _ in range(self.config.simulations):
            state = root_state.clone()
            path = [root]
            node = root
            terminal = False
            while node.children:
                action, child = self._select(node)
                reward, terminal = state.step(action)
                child.reward = float(reward)
                node = child
                path.append(node)
                if terminal:
                    break
            leaf_value = 0.0 if terminal else self._expand(node, state)
            value = leaf_value
            for current in reversed(path):
                current.visits += 1
                current.value_sum += value
                value = current.reward + self.config.discount * value

        actions = list(root.children)
        visits = np.asarray([root.children[a].visits for a in actions], dtype=np.float64)
        if self.config.temperature <= 1e-6:
            probabilities = np.zeros_like(visits)
            probabilities[int(np.argmax(visits))] = 1.0
        else:
            probabilities = visits ** (1.0 / self.config.temperature)
            probabilities /= max(probabilities.sum(), 1e-12)
        best = actions[int(np.argmax(visits))]
        return SearchResult(
            action=best,
            policy={a: float(p) for a, p in zip(actions, probabilities)},
            q_values={
                a: root.children[a].reward + self.config.discount * root.children[a].value
                for a in actions
            },
            root_value=root.value,
        )
