from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "radar_dnn_mcts"))


@dataclass(frozen=True)
class ModeSpec:
    name: str
    proposal_mode: str
    maintain_action_index: bool
    bulk: bool = False


MODES = [
    ModeSpec("recompute", "recompute", True),
    ModeSpec("cached_unique", "cached", True),
    ModeSpec("cached_cursor", "cached_cursor", True),
    ModeSpec("cached_cursor_bulk_with_map", "cached_cursor_bulk", True, bulk=True),
    ModeSpec("cached_cursor_bulk_best", "cached_cursor_bulk", False, bulk=True),
]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "p99_ms": 0.0}
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def empty_branch_result():
    from batched_branch_sim import BranchStepResult

    return BranchStepResult(
        rewards=np.empty((0,), dtype=np.float32),
        dt_ms=np.empty((0,), dtype=np.float32),
        executed=np.empty((0,), dtype=np.int32),
        terminals=np.empty((0,), dtype=np.uint8),
        observations=[],
    )


def build_search(args: argparse.Namespace, mode: ModeSpec):
    from perf_fast_planner import FastActionAttentionPlanner
    from persistent_root_search import PersistentRootSearch
    from repaired_campaign_tools import env_preset_cfg
    from two_sensor_physical_head_eval import ActionAttentionFactorizedNet

    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1

    model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    planner = FastActionAttentionPlanner(
        model,
        env_cfg,
        device=torch.device(args.device),
        use_amp=bool(args.amp),
        use_compile=bool(args.compile),
    )
    batch_size = int(args.top_k) * int(args.waves) if mode.bulk else int(args.top_k)
    search = PersistentRootSearch(
        planner,
        initial_targets=int(args.initial_targets),
        seed=int(args.seed),
        env_cfg=env_cfg,
        batch_size=batch_size,
        budget_ms=float(args.budget_ms),
    )
    return search, batch_size


def run_mode(args: argparse.Namespace, mode: ModeSpec) -> dict[str, object]:
    from persistent_dense_root_tree import PersistentDenseRootTree
    from persistent_root_search import RootSearchWave

    device = torch.device(args.device)
    sync(device)
    t_setup = time.perf_counter()
    search, batch_size = build_search(args, mode)
    sync(device)
    setup_ms = (time.perf_counter() - t_setup) * 1000.0

    propose_times: list[float] = []
    sim_times: list[float] = []
    update_times: list[float] = []
    select_times: list[float] = []
    combined_times: list[float] = []
    final_sizes: list[int] = []
    final_visits: list[int] = []
    selected_actions: list[int] = []
    reward_sums: list[float] = []

    try:
        total_iters = int(args.warmup) + int(args.iters)
        for i in range(total_iters):
            tree = PersistentDenseRootTree(
                search,
                capacity=int(args.capacity),
                maintain_action_index=bool(mode.maintain_action_index),
            )
            iter_reward = 0.0
            t_prop_total = 0.0
            t_sim_total = 0.0
            t_update_total = 0.0
            t_select_total = 0.0

            sync(device)
            t_iter = time.perf_counter()
            if mode.bulk:
                sync(device)
                t0 = time.perf_counter()
                actions, scores = tree.propose_cached_cursor(int(args.top_k) * int(args.waves))
                sync(device)
                t_prop_total += (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                sim = search.simulate(actions) if actions.size else empty_branch_result()
                t_sim_total += (time.perf_counter() - t1) * 1000.0

                t2 = time.perf_counter()
                update = tree.append_new_from_wave(RootSearchWave(actions=actions, scores=scores, sim=sim))
                t_update_total += (time.perf_counter() - t2) * 1000.0
                iter_reward += float(np.sum(update.rewards))
            else:
                for _ in range(int(args.waves)):
                    if mode.proposal_mode == "cached_cursor":
                        sync(device)
                        t0 = time.perf_counter()
                        actions, scores = tree.propose_cached_cursor(int(args.top_k))
                        sync(device)
                        t_prop_total += (time.perf_counter() - t0) * 1000.0
                    elif mode.proposal_mode == "cached":
                        sync(device)
                        t0 = time.perf_counter()
                        actions, scores = search.propose_cached(int(args.top_k), exclude=set(tree._action_to_index))
                        sync(device)
                        t_prop_total += (time.perf_counter() - t0) * 1000.0
                    else:
                        sync(device)
                        t0 = time.perf_counter()
                        actions, scores = search.propose(int(args.top_k))
                        sync(device)
                        t_prop_total += (time.perf_counter() - t0) * 1000.0

                    t1 = time.perf_counter()
                    sim = search.simulate(actions) if actions.size else empty_branch_result()
                    t_sim_total += (time.perf_counter() - t1) * 1000.0

                    t2 = time.perf_counter()
                    wave = RootSearchWave(actions=actions, scores=scores, sim=sim)
                    if mode.proposal_mode == "cached_cursor":
                        update = tree.append_new_from_wave(wave)
                    else:
                        update = tree.update_from_wave(wave)
                    t_update_total += (time.perf_counter() - t2) * 1000.0
                    iter_reward += float(np.sum(update.rewards))

            t3 = time.perf_counter()
            selected = tree.select_action()
            t_select_total += (time.perf_counter() - t3) * 1000.0

            sync(device)
            iter_ms = (time.perf_counter() - t_iter) * 1000.0
            if i >= int(args.warmup):
                propose_times.append(t_prop_total)
                sim_times.append(t_sim_total)
                update_times.append(t_update_total)
                select_times.append(t_select_total)
                combined_times.append(iter_ms)
                final_sizes.append(int(tree.size))
                final_visits.append(int(tree.total_visits))
                selected_actions.append(int(selected))
                reward_sums.append(float(iter_reward))
    finally:
        search.close()

    mean_combined = float(np.mean(combined_times)) if combined_times else 0.0
    return {
        "mode": mode.name,
        "proposal_mode": mode.proposal_mode,
        "search_batch_size": int(batch_size),
        "maintain_action_index": bool(mode.maintain_action_index),
        "one_time_setup_ms": float(setup_ms),
        "persistent_neural_propose_total": stats(propose_times),
        "exact_branch_sim_total": stats(sim_times),
        "dense_tree_update_total": stats(update_times),
        "puct_select_total": stats(select_times),
        "combined_iteration": stats(combined_times),
        "mean_unique_actions": float(np.mean(final_sizes)) if final_sizes else 0.0,
        "mean_total_visits": float(np.mean(final_visits)) if final_visits else 0.0,
        "mean_reward_sum": float(np.mean(reward_sums)) if reward_sums else 0.0,
        "last_selected_action": int(selected_actions[-1]) if selected_actions else -1,
        "propose_fraction": float(np.mean(propose_times) / max(mean_combined, 1e-12)) if propose_times else 0.0,
        "sim_fraction": float(np.mean(sim_times) / max(mean_combined, 1e-12)) if sim_times else 0.0,
        "update_fraction": float(np.mean(update_times) / max(mean_combined, 1e-12)) if update_times else 0.0,
        "selection_fraction": float(np.mean(select_times) / max(mean_combined, 1e-12)) if select_times else 0.0,
    }


def print_table(results: list[dict[str, object]]) -> None:
    headers = ["mode", "combined", "propose", "sim", "update", "select", "unique", "reward"]
    print("\t".join(headers))
    for row in results:
        print(
            "\t".join(
                [
                    str(row["mode"]),
                    f"{row['combined_iteration']['mean_ms']:.3f}",
                    f"{row['persistent_neural_propose_total']['mean_ms']:.3f}",
                    f"{row['exact_branch_sim_total']['mean_ms']:.3f}",
                    f"{row['dense_tree_update_total']['mean_ms']:.3f}",
                    f"{row['puct_select_total']['mean_ms']:.3f}",
                    f"{row['mean_unique_actions']:.1f}",
                    f"{row['mean_reward_sum']:.3f}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--waves", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--budget-ms", type=float, default=200.0)
    parser.add_argument("--modes", nargs="*", default=[mode.name for mode in MODES])
    parser.add_argument("--out", type=Path, default=Path("results/perf_lab_root_expansion_compare.json"))
    args = parser.parse_args()

    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)

    requested = set(args.modes)
    specs = [mode for mode in MODES if mode.name in requested]
    if not specs:
        raise SystemExit(f"No valid modes selected. Choices: {', '.join(mode.name for mode in MODES)}")

    results = [run_mode(args, mode) for mode in specs]
    report = {
        "device": str(args.device),
        "cuda_available": bool(torch.cuda.is_available()),
        "amp": bool(args.amp),
        "compile": bool(args.compile),
        "initial_targets": int(args.initial_targets),
        "rate": float(args.rate),
        "seed": int(args.seed),
        "waves_per_iteration": int(args.waves),
        "top_k": int(args.top_k),
        "capacity": int(args.capacity),
        "iters": int(args.iters),
        "warmup": int(args.warmup),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_table(results)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
