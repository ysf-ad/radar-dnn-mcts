from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "radar_dnn_mcts"))


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--waves", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--proposal-mode", choices=["recompute", "cached", "cached_cursor", "cached_cursor_bulk"], default="recompute")
    parser.add_argument("--maintain-action-index", action="store_true", help="Keep the Python action->index map even for cursor-bulk mode.")
    parser.add_argument("--select-every-wave", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("perf_lab_persistent_dense_root_tree.json"))
    args = parser.parse_args()

    from perf_fast_planner import FastActionAttentionPlanner
    from persistent_dense_root_tree import PersistentDenseRootTree
    from persistent_root_search import PersistentRootSearch, RootSearchWave
    from repaired_campaign_tools import env_preset_cfg
    from two_sensor_physical_head_eval import ActionAttentionFactorizedNet

    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)
    device = torch.device(args.device)

    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1

    model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    planner = FastActionAttentionPlanner(model, env_cfg, device=device, use_amp=bool(args.amp), use_compile=bool(args.compile))

    sync(device)
    t_setup = time.perf_counter()
    search_batch_size = int(args.top_k) * int(args.waves) if str(args.proposal_mode) == "cached_cursor_bulk" else int(args.top_k)
    search = PersistentRootSearch(
        planner,
        initial_targets=args.initial_targets,
        seed=args.seed,
        env_cfg=env_cfg,
        batch_size=search_batch_size,
        budget_ms=200.0,
    )
    maintain_action_index = bool(args.maintain_action_index or str(args.proposal_mode) != "cached_cursor_bulk")
    tree = PersistentDenseRootTree(search, capacity=int(args.capacity), maintain_action_index=maintain_action_index)
    sync(device)
    setup_ms = (time.perf_counter() - t_setup) * 1000.0

    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "amp": bool(args.amp),
        "compile": bool(args.compile),
        "initial_targets": int(args.initial_targets),
        "rate": float(args.rate),
        "seed": int(args.seed),
        "waves_per_iteration": int(args.waves),
        "top_k": int(args.top_k),
        "search_batch_size": int(search_batch_size),
        "capacity": int(args.capacity),
        "proposal_mode": str(args.proposal_mode),
        "maintain_action_index": bool(maintain_action_index),
        "select_every_wave": bool(args.select_every_wave),
        "one_time_setup_ms": float(setup_ms),
    }

    propose_times = []
    sim_times = []
    update_times = []
    select_times = []
    combined_times = []
    final_sizes = []
    final_visits = []
    selected_actions = []
    reward_sums = []

    try:
        for i in range(int(args.warmup) + int(args.iters)):
            # Use a fresh dense tree per iteration so every timing run has the
            # same root and same number of updates.
            tree = PersistentDenseRootTree(search, capacity=int(args.capacity), maintain_action_index=maintain_action_index)
            iter_reward = 0.0

            sync(device)
            t_iter = time.perf_counter()
            t_prop_total = 0.0
            t_sim_total = 0.0
            t_update_total = 0.0
            t_select_total = 0.0
            if str(args.proposal_mode) == "cached_cursor_bulk":
                sync(device)
                t0 = time.perf_counter()
                actions, scores = tree.propose_cached_cursor(int(args.top_k) * int(args.waves))
                sync(device)
                t_prop_total += (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                if actions.size:
                    sim = search.simulate(actions)
                else:
                    from batched_branch_sim import BranchStepResult

                    sim = BranchStepResult(
                        rewards=np.empty((0,), dtype=np.float32),
                        dt_ms=np.empty((0,), dtype=np.float32),
                        executed=np.empty((0,), dtype=np.int32),
                        terminals=np.empty((0,), dtype=np.uint8),
                        observations=[],
                    )
                t_sim_total += (time.perf_counter() - t1) * 1000.0

                t2 = time.perf_counter()
                update = tree.append_new_from_wave(RootSearchWave(actions=actions, scores=scores, sim=sim))
                t_update_total += (time.perf_counter() - t2) * 1000.0
                iter_reward += float(np.sum(update.rewards))

                if bool(args.select_every_wave):
                    t3 = time.perf_counter()
                    _ = tree.select_action()
                    t_select_total += (time.perf_counter() - t3) * 1000.0

            for _ in range(0 if str(args.proposal_mode) == "cached_cursor_bulk" else int(args.waves)):
                if str(args.proposal_mode) == "cached_cursor":
                    sync(device)
                    t0 = time.perf_counter()
                    actions, scores = tree.propose_cached_cursor(int(args.top_k))
                    sync(device)
                    t_prop_total += (time.perf_counter() - t0) * 1000.0
                elif str(args.proposal_mode) == "cached":
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
                if actions.size:
                    sim = search.simulate(actions)
                    t_sim_total += (time.perf_counter() - t1) * 1000.0
                else:
                    from batched_branch_sim import BranchStepResult

                    sim = BranchStepResult(
                        rewards=np.empty((0,), dtype=np.float32),
                        dt_ms=np.empty((0,), dtype=np.float32),
                        executed=np.empty((0,), dtype=np.int32),
                        terminals=np.empty((0,), dtype=np.uint8),
                        observations=[],
                    )
                    t_sim_total += (time.perf_counter() - t1) * 1000.0

                t2 = time.perf_counter()
                wave = RootSearchWave(actions=actions, scores=scores, sim=sim)
                if str(args.proposal_mode) == "cached_cursor":
                    update = tree.append_new_from_wave(wave)
                else:
                    update = tree.update_from_wave(wave)
                t_update_total += (time.perf_counter() - t2) * 1000.0
                iter_reward += float(np.sum(update.rewards))

                if bool(args.select_every_wave):
                    t3 = time.perf_counter()
                    _ = tree.select_action()
                    t_select_total += (time.perf_counter() - t3) * 1000.0

            if not bool(args.select_every_wave):
                t3 = time.perf_counter()
                _ = tree.select_action()
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
                selected_actions.append(int(tree.select_action()))
                reward_sums.append(float(iter_reward))
    finally:
        search.close()

    report.update(
        {
            "persistent_neural_propose_total": stats(propose_times),
            "exact_branch_sim_total": stats(sim_times),
            "dense_tree_update_total": stats(update_times),
            "puct_select_total": stats(select_times),
            "combined_iteration": stats(combined_times),
            "mean_unique_actions": float(np.mean(final_sizes)) if final_sizes else 0.0,
            "mean_total_visits": float(np.mean(final_visits)) if final_visits else 0.0,
            "mean_reward_sum": float(np.mean(reward_sums)) if reward_sums else 0.0,
            "last_selected_action": int(selected_actions[-1]) if selected_actions else -1,
            "tree_update_fraction": float(np.mean(update_times) / max(np.mean(combined_times), 1e-12)),
            "selection_fraction": float(np.mean(select_times) / max(np.mean(combined_times), 1e-12)),
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
