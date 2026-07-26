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
    """Search and duration-aware return settings for one scheduling window."""

    rollouts: int = 32
    c_puct: float = 1.5
    discount: float = 0.99
    window_ms: float = 200.0
    reward_scale: float = 32.0
    temperature: float = 1.0
    dirichlet_alpha: float = 0.03
    dirichlet_fraction: float = 0.25
    random_seed: int = 0
    factorized_policy_first: bool = False
    factorized_search_logit_bias: float = 0.0
    sample_training_actions: bool = True


@dataclass
class Node:
    """One action edge and its backed-up subtree statistics."""

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
    policy_visits: int = 0
    policy_from_visits: bool = False


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
        # A fresh state has no backed-up tree statistics, so start from P.
        if node.visits == 0:
            if self.config.factorized_policy_first and 0 in node.children:
                search = node.children[0]
                tracks = [
                    item for item in node.children.items() if item[0] != 0
                ]
                track_mass = sum(child.prior for _, child in tracks)
                adjusted_search = search.prior * np.exp(
                    self.config.factorized_search_logit_bias
                )
                if tracks and track_mass > adjusted_search:
                    return max(tracks, key=lambda item: item[1].prior)
                return 0, search
            return max(node.children.items(), key=lambda item: item[1].prior)
        scale = np.sqrt(node.visits)

        def score(child: Node) -> float:
            return (
                self._edge_value(child)
                + self.config.c_puct
                * child.prior
                * scale
                / (1 + child.visits)
            )

        return max(
            node.children.items(),
            key=lambda item: score(item[1]),
        )

    def _step(self, state: SearchState, action: int) -> tuple[float, float, bool]:
        before_elapsed = state.elapsed
        reward, terminal = state.step(action)
        discount = self.config.discount ** (
            (state.elapsed - before_elapsed) / self.config.window_ms
        )
        return float(reward) / max(self.config.reward_scale, 1e-6), discount, terminal

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
        self,
        root: Node,
        root_state: SearchState,
        *,
        sample_actions: bool = False,
    ) -> tuple[SearchDecision, ...]:
        """Extract one complete schedule from the searched tree."""
        state = root_state.clone()
        node: Node | None = root
        decisions: list[SearchDecision] = []
        while state.legal_actions():
            if node is not None and node.children:
                policy_visits = sum(
                    child.visits for child in node.children.values()
                )
                policy_from_visits = policy_visits > 0
                actions, probabilities = self._distribution(node)
                index = (
                    int(self.rng.choice(len(actions), p=probabilities))
                    if sample_actions
                    else int(np.argmax(probabilities))
                )
                action = actions[index]
                value = node.value
                next_node = node.children[action]
            else:
                policy_visits = 0
                policy_from_visits = False
                actions = state.legal_actions()
                priors, value = self.evaluator(state, actions)
                probabilities = np.maximum(np.asarray(priors, dtype=np.float64), 0.0)
                probabilities /= max(probabilities.sum(), 1e-12)
                if self.config.temperature <= 1e-6:
                    tempered = np.zeros_like(probabilities)
                    tempered[int(np.argmax(probabilities))] = 1.0
                else:
                    tempered = probabilities ** (
                        1.0 / self.config.temperature
                    )
                    tempered /= max(tempered.sum(), 1e-12)
                probabilities = tempered
                index = (
                    int(self.rng.choice(len(actions), p=probabilities))
                    if sample_actions
                    else int(np.argmax(probabilities))
                )
                action = actions[index]
                next_node = None
            decisions.append(
                SearchDecision(
                    action=int(action),
                    policy={
                        candidate: float(probability)
                        for candidate, probability in zip(actions, probabilities)
                    },
                    value=float(value),
                    policy_visits=int(policy_visits),
                    policy_from_visits=bool(policy_from_visits),
                )
            )
            _, terminal = state.step(int(action))
            node = next_node
            if terminal:
                break
        return tuple(decisions)

    def run(self, root_state: SearchState, *, training: bool = False) -> SearchResult:
        """Run complete-window rollouts and return their principal trajectory."""
        prepare = getattr(self.evaluator, "prepare", None)
        if callable(prepare):
            prepare(root_state)
        root = Node()
        self._expand(root, root_state, root_noise=training)
        for _ in range(self.config.rollouts):
            state = root_state.clone()
            path = [root]
            node = root
            terminal = False
            value = 0.0
            while not terminal:
                if not node.children:
                    value = self._expand(node, state)
                    if not node.children:
                        break
                action, child = self._select(node)
                reward, discount, terminal = self._step(state, action)
                child.reward = reward
                child.discount = discount
                node = child
                path.append(node)
            if terminal:
                _, value = self.evaluator(state, [])
            # Back up edge rewards followed by the boundary value at the leaf.
            for current in reversed(path):
                current.visits += 1
                current.value_sum += value
                value = current.reward + current.discount * value
        trajectory = self._extract_trajectory(
            root,
            root_state,
            sample_actions=(
                training and self.config.sample_training_actions
            ),
        )
        if not trajectory:
            raise RuntimeError("PUCT produced no scheduling trajectory")
        return SearchResult(
            trajectory=trajectory,
            root_value=root.value,
        )
