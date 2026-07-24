from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class SearchState(Protocol):
    elapsed: float

    def clone(self) -> "SearchState": ...
    def legal_actions(self) -> list[int]: ...
    def step(self, action: int) -> tuple[float, bool]: ...


class PolicyValueEvaluator(Protocol):
    def __call__(self, state: SearchState, legal_actions: list[int]) -> tuple[np.ndarray, float]: ...


@dataclass(frozen=True)
class PUCTConfig:
    rollouts: int = 32
    c_puct: float = 1.5
    discount: float = 0.99
    window_ms: float = 200.0
    reward_scale: float = 32.0
    temperature: float = 1.0
    dirichlet_alpha: float = 0.03
    dirichlet_fraction: float = 0.25
    random_seed: int = 0


@dataclass
class Node:
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    reward: float = 0.0
    discount: float = 1.0
    children: dict[int, "Node"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class SearchDecision:
    action: int
    policy: dict[int, float]
    value: float


@dataclass(frozen=True)
class SearchResult:
    trajectory: tuple[SearchDecision, ...]
    root_value: float


class PUCT:
    """AlphaZero-style PUCT over complete scheduling-window rollouts."""

    def __init__(self, evaluator: PolicyValueEvaluator, config: PUCTConfig = PUCTConfig()):
        self.evaluator = evaluator
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def _expand(self, node: Node, state: SearchState, root_noise: bool = False) -> float:
        legal = state.legal_actions()
        priors, value = self.evaluator(state, legal)
        if not legal:
            return float(value)
        priors = np.maximum(np.asarray(priors, dtype=np.float64), 0.0)
        priors /= max(priors.sum(), 1e-12)
        if root_noise and self.config.dirichlet_fraction > 0.0 and len(legal) > 1:
            noise = self.rng.dirichlet(
                np.full(len(legal), max(self.config.dirichlet_alpha, 1e-6))
            )
            fraction = float(np.clip(self.config.dirichlet_fraction, 0.0, 1.0))
            priors = (1.0 - fraction) * priors + fraction * noise
        for action, prior in zip(legal, priors):
            node.children[int(action)] = Node(prior=float(prior))
        return float(value)

    def _edge_value(self, child: Node) -> float:
        if not child.visits:
            return 0.0
        return child.reward + child.discount * child.value

    def _select(self, node: Node) -> tuple[int, Node]:
        scale = np.sqrt(node.visits + 1.0)
        return max(
            node.children.items(),
            key=lambda item: (
                self._edge_value(item[1])
                + self.config.c_puct
                * item[1].prior
                * scale
                / (1 + item[1].visits)
            ),
        )

    def _step(self, state: SearchState, action: int) -> tuple[float, float, bool]:
        before_elapsed = state.elapsed
        reward, terminal = state.step(action)
        discount = self.config.discount ** (
            (state.elapsed - before_elapsed) / self.config.window_ms
        )
        return float(reward) / max(self.config.reward_scale, 1e-6), discount, terminal

    def _complete_trajectory(
        self,
        state: SearchState,
        node: Node,
        path: list[Node],
        terminal: bool,
    ) -> float:
        while not terminal:
            legal = state.legal_actions()
            if not legal:
                break
            if not node.children:
                self._expand(node, state)
            actions = list(node.children)
            probabilities = np.asarray(
                [node.children[action].prior for action in actions],
                dtype=np.float64,
            )
            probabilities /= max(probabilities.sum(), 1e-12)
            index = int(np.argmax(probabilities))
            child = node.children[int(actions[index])]
            reward, discount, terminal = self._step(state, int(actions[index]))
            child.reward = reward
            child.discount = discount
            node = child
            path.append(node)

        _, value = self.evaluator(state, [])
        return float(value)

    def _distribution(self, node: Node) -> tuple[list[int], np.ndarray]:
        actions = list(node.children)
        visits = np.asarray([node.children[action].visits for action in actions], dtype=np.float64)
        if visits.sum() <= 0.0:
            visits = np.asarray([node.children[action].prior for action in actions], dtype=np.float64)
        if self.config.temperature <= 1e-6:
            probabilities = np.zeros_like(visits)
            probabilities[int(np.argmax(visits))] = 1.0
        else:
            probabilities = visits ** (1.0 / self.config.temperature)
            probabilities /= max(probabilities.sum(), 1e-12)
        return actions, probabilities

    def _extract_trajectory(
        self, root: Node, root_state: SearchState
    ) -> tuple[SearchDecision, ...]:
        state = root_state.clone()
        node: Node | None = root
        decisions: list[SearchDecision] = []
        while state.legal_actions():
            if node is not None and node.children:
                actions, probabilities = self._distribution(node)
                action = actions[int(np.argmax(probabilities))]
                value = node.value
                next_node = node.children[action]
            else:
                actions = state.legal_actions()
                priors, value = self.evaluator(state, actions)
                probabilities = np.maximum(np.asarray(priors, dtype=np.float64), 0.0)
                probabilities /= max(probabilities.sum(), 1e-12)
                action = actions[int(np.argmax(probabilities))]
                next_node = None
            decisions.append(
                SearchDecision(
                    action=int(action),
                    policy={
                        candidate: float(probability)
                        for candidate, probability in zip(actions, probabilities)
                    },
                    value=float(value),
                )
            )
            _, terminal = state.step(int(action))
            node = next_node
            if terminal:
                break
        return tuple(decisions)

    def run(self, root_state: SearchState, *, training: bool = False) -> SearchResult:
        root = Node()
        self._expand(root, root_state, root_noise=training)
        for _ in range(self.config.rollouts):
            state = root_state.clone()
            path = [root]
            node = root
            terminal = False
            while node.children and not terminal:
                action, child = self._select(node)
                reward, discount, terminal = self._step(state, action)
                child.reward = reward
                child.discount = discount
                node = child
                path.append(node)
                if node.visits == 0:
                    break
            value = self._complete_trajectory(state, node, path, terminal)
            for current in reversed(path):
                current.visits += 1
                current.value_sum += value
                value = current.reward + current.discount * value

        trajectory = self._extract_trajectory(root, root_state)
        if not trajectory:
            raise RuntimeError("PUCT produced no scheduling trajectory")
        return SearchResult(
            trajectory=trajectory,
            root_value=root.value,
        )
