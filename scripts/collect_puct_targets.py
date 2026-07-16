from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from radar_dnn_mcts.env.config import RewardConfig
from radar_dnn_mcts.env.factory import create_engine
from radar_dnn_mcts.env.features import FeatureBuilder
from radar_dnn_mcts.models.scheduler import RadarSchedulerModel
from radar_dnn_mcts.search import PUCT, PUCTConfig
from radar_dnn_mcts.training.radar_search import RadarModelEvaluator, RadarWindowSearchState


class PUCTWindowPlanner:
    def __init__(self, model: RadarSchedulerModel, config: PUCTConfig, max_targets: int):
        self.features = FeatureBuilder(max_targets=max_targets)
        self.reward = RewardConfig()
        self.search = PUCT(RadarModelEvaluator(model), config)
        self.records: list[dict] = []

    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        state = RadarWindowSearchState(obs, self.features, self.reward, budget_ms=budget_ms)
        plan: list[int] = []
        pending: list[dict] = []
        while state.legal_actions():
            tokens, context = state.network_input()
            result = self.search.run(state)
            policy = np.zeros(tokens.shape[0], dtype=np.float32)
            q = np.zeros_like(policy)
            q_mask = np.zeros_like(policy, dtype=bool)
            for action, probability in result.policy.items():
                policy[action] = probability
                q[action] = result.q_values[action]
                q_mask[action] = True
            pending.append({"tokens": tokens, "context": context, "policy": policy, "q": q, "q_mask": q_mask})
            plan.append(result.action)
            _, terminal = state.step(result.action)
            if terminal:
                break
        self.records.extend(pending)
        return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--discount", type=float, default=0.997)
    parser.add_argument("--learned-q-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--arrival-rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--max-targets", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = RadarSchedulerModel()
    search_config = PUCTConfig(
        simulations=args.simulations,
        c_puct=args.c_puct,
        discount=args.discount,
        learned_q_weight=args.learned_q_weight,
        temperature=args.temperature,
    )
    planner = PUCTWindowPlanner(model, search_config, args.max_targets)
    engine = create_engine(
        planner,
        initial_targets=args.initial_targets,
        max_targets=args.max_targets,
        seed=args.seed,
        arrival_rate=args.arrival_rate,
        reward=planner.reward,
    )
    try:
        for _ in range(args.windows):
            engine.step_window()
            if engine.term_buf[0]:
                break
    finally:
        engine.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.stack([record[key] for record in planner.records]) for key in planner.records[0]}
    arrays.update(
        puct_simulations=np.asarray(search_config.simulations),
        puct_c=np.asarray(search_config.c_puct),
        puct_discount=np.asarray(search_config.discount),
        puct_learned_q_weight=np.asarray(search_config.learned_q_weight),
        puct_temperature=np.asarray(search_config.temperature),
    )
    np.savez_compressed(args.out, **arrays)
    print(f"saved {len(planner.records)} PUCT targets to {args.out}")


if __name__ == "__main__":
    main()
