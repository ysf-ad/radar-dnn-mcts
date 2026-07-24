from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pufferlib.ocean.radarxs.models.edf import EDFPlanner
from pufferlib.ocean.radarxs.models.est import ESTPlanner
from radar_dnn_mcts.env.config import RewardConfig
from radar_dnn_mcts.env.factory import create_engine
from radar_dnn_mcts.env.features import FeatureBuilder
from radar_dnn_mcts.models.checkpoint import load_checkpoint
from radar_dnn_mcts.models.decoders import AutoregressiveDecoder
from radar_dnn_mcts.models.scheduler import RadarSchedulerModel
from radar_dnn_mcts.search import PUCT, PUCTConfig
from radar_dnn_mcts.training.radar_search import (
    RadarAREvaluator,
    RadarModelEvaluator,
    RadarWindowSearchState,
)


class WindowTrajectoryCollector:
    def __init__(
        self,
        search: PUCT | None,
        features: FeatureBuilder,
        reward: RewardConfig,
        teacher=None,
    ):
        self.search = search
        self.features = features
        self.reward = reward
        self.teacher = teacher
        self.windows: list[list[dict]] = []
        self.pending: list[dict] = []
        self.executed: list[dict] = []
        self.cursor = 0

    def _puct_records(self, obs: dict, budget_ms: float) -> list[dict]:
        state = RadarWindowSearchState(
            obs, self.features, self.reward, budget_ms=budget_ms
        )
        result = self.search.run(state, training=True)
        return self._records(
            state,
            (
                (decision.action, decision.policy)
                for decision in result.trajectory
            ),
        )

    def _teacher_records(self, obs: dict, budget_ms: float) -> list[dict]:
        state = RadarWindowSearchState(
            obs, self.features, self.reward, budget_ms=budget_ms
        )
        actions = (
            (int(action), {int(action): 1.0})
            for action in self.teacher.plan(obs, budget_ms=budget_ms)
        )
        return self._records(state, actions, skip_invalid=True)

    @staticmethod
    def _records(state, actions, *, skip_invalid: bool = False) -> list[dict]:
        records = []
        for action, probabilities in actions:
            action = int(action)
            if action not in state.legal_actions():
                if skip_invalid:
                    continue
                break
            tokens, context = state.network_input()
            policy = np.zeros(tokens.shape[0], dtype=np.float32)
            for candidate, probability in probabilities.items():
                policy[int(candidate)] = probability
            records.append(
                {
                    "tokens": tokens,
                    "context": context,
                    "policy": policy,
                    "action": action,
                }
            )
            _, terminal = state.step(action)
            if terminal:
                break
        return records

    def plan(self, obs: dict, budget_ms: float) -> list[int]:
        self.pending = (
            self._teacher_records(obs, budget_ms)
            if self.teacher is not None
            else self._puct_records(obs, budget_ms)
        )
        self.executed = []
        self.cursor = 0
        return [record["action"] for record in self.pending]

    def observe_transition(
        self,
        action: int,
        reward: float,
        _before: dict,
        _after: dict,
        duration_ms: float,
    ) -> None:
        while (
            self.cursor < len(self.pending)
            and self.pending[self.cursor]["action"] != action
        ):
            self.cursor += 1
        if self.cursor >= len(self.pending):
            return
        record = self.pending[self.cursor]
        record["reward"] = reward
        record["duration_ms"] = duration_ms
        self.executed.append(record)
        self.cursor += 1

    def finish_window(self) -> None:
        if self.executed:
            self.windows.append(self.executed)
        self.pending = []
        self.executed = []
        self.cursor = 0

    def backfill_episode_returns(self, discount: float, window_ms: float, scale: float) -> None:
        value = 0.0
        for window in reversed(self.windows):
            for record in reversed(window):
                step_discount = discount ** (
                    float(record["duration_ms"]) / max(window_ms, 1e-6)
                )
                value = float(record["reward"]) / max(scale, 1e-6) + step_discount * value
                record["return"] = value

    def arrays(self) -> dict[str, np.ndarray]:
        if not self.windows:
            raise RuntimeError("collector produced no executed window trajectories")
        window_count = len(self.windows)
        max_steps = max(len(window) for window in self.windows)
        example = self.windows[0][0]
        rows, token_dim = example["tokens"].shape
        context_dim = example["context"].shape[0]
        shapes = {
            "tokens": (rows, token_dim),
            "context": (context_dim,),
            "policy": (rows,),
            "actions": (),
            "rewards": (),
            "returns": (),
            "durations_ms": (),
        }
        dtypes = {"actions": np.int64}
        prefix = (window_count, max_steps)
        arrays = {
            key: np.zeros(prefix + shape, dtype=dtypes.get(key, np.float32))
            for key, shape in shapes.items()
        }
        arrays["action_mask"] = np.zeros(prefix, dtype=bool)
        for window_index, window in enumerate(self.windows):
            for step, record in enumerate(window):
                arrays["tokens"][window_index, step] = record["tokens"]
                arrays["context"][window_index, step] = record["context"]
                arrays["policy"][window_index, step] = record["policy"]
                arrays["actions"][window_index, step] = record["action"]
                arrays["rewards"][window_index, step] = record["reward"]
                arrays["returns"][window_index, step] = record["return"]
                arrays["durations_ms"][window_index, step] = record["duration_ms"]
                arrays["action_mask"][window_index, step] = True
        return arrays


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect grouped full-window PUCT trajectories."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=50)
    parser.add_argument(
        "--rollouts",
        type=int,
        default=32,
        help="Complete scheduling-window PUCT rollouts.",
    )
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--return-scale", type=float, default=32.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.03)
    parser.add_argument("--dirichlet-fraction", type=float, default=0.25)
    parser.add_argument("--teacher", choices=["puct", "edf", "est"], default="puct")
    parser.add_argument("--search-model", choices=["core", "ar"], default="core")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--arrival-rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--max-targets", type=int, default=100)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    features = FeatureBuilder(max_targets=args.max_targets)
    reward = RewardConfig()
    core = RadarSchedulerModel().to(device)
    ar = AutoregressiveDecoder(core).to(device)
    if args.checkpoint:
        modules = {"core": core}
        if args.search_model == "ar":
            modules["ar"] = ar
        load_checkpoint(args.checkpoint, modules, device)
    elif args.teacher == "puct":
        parser.error("--teacher puct requires --checkpoint")

    teacher = None
    search = None
    if args.teacher == "edf":
        teacher = EDFPlanner(args.max_targets)
    elif args.teacher == "est":
        teacher = ESTPlanner(args.max_targets)
    else:
        evaluator = (
            RadarAREvaluator(ar)
            if args.search_model == "ar"
            else RadarModelEvaluator(core)
        )
        search = PUCT(
            evaluator,
            PUCTConfig(
                rollouts=args.rollouts,
                c_puct=args.c_puct,
                discount=args.discount,
                reward_scale=args.return_scale,
                temperature=args.temperature,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_fraction=args.dirichlet_fraction,
                random_seed=args.seed,
            ),
        )
    collector = WindowTrajectoryCollector(search, features, reward, teacher)
    engine = create_engine(
        collector,
        initial_targets=args.initial_targets,
        max_targets=args.max_targets,
        seed=args.seed,
        arrival_rate=args.arrival_rate,
        reward=reward,
    )
    try:
        for _ in range(args.windows):
            engine.step_window()
            if engine.term_buf[0]:
                break
    finally:
        engine.close()

    collector.backfill_episode_returns(
        args.discount, 200.0, args.return_scale
    )
    arrays = collector.arrays()
    arrays.update(
        puct_rollouts=np.asarray(args.rollouts),
        puct_c=np.asarray(args.c_puct),
        puct_discount=np.asarray(args.discount),
        return_scale=np.asarray(args.return_scale),
        dirichlet_alpha=np.asarray(args.dirichlet_alpha),
        dirichlet_fraction=np.asarray(args.dirichlet_fraction),
        teacher=np.asarray(args.teacher),
        search_model=np.asarray(args.search_model),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(
        f"saved {int(arrays['action_mask'].sum())} decisions "
        f"in {arrays['action_mask'].shape[0]} grouped windows to {args.out}"
    )


if __name__ == "__main__":
    main()
