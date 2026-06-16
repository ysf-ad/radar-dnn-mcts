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


def make_slot_batch(obs: dict, decisions: int, budget_ms: float):
    from exact_env_mutual import xs_decode_action
    from mutual_features import slot_features
    from perf_fast_planner import physical_action_arrays
    from two_sensor_physical_head_eval import MAXT

    selected_sets = []
    slots = []
    selected = set()
    elapsed = 0.0
    search_count = 0
    track_count = 0
    last = -1
    dwell = np.asarray(obs["t_dwell"], dtype=np.float32)

    # Build deterministic valid scheduling contexts. This does not claim to be
    # the chosen policy path; it gives equivalent slot/mask tensors for latency
    # and batching correctness tests.
    for _ in range(int(decisions)):
        selected_sets.append(set(selected))
        slots.append(slot_features(obs, elapsed, search_count, track_count, last, float(budget_ms)).astype(np.float32))
        actions, _bases, _sensors = physical_action_arrays(obs, selected=selected, max_trackers=MAXT)
        track_action = None
        for action in actions:
            base, _sensor = xs_decode_action(int(action), MAXT)
            if int(base) > 0 and int(base) not in selected:
                track_action = int(action)
                break
        if track_action is None:
            search_count += 1
            elapsed += 10.0
            last = 0
            continue
        base, _sensor = xs_decode_action(track_action, MAXT)
        selected.add(int(base))
        track_count += 1
        dt = float(dwell[int(base) - 1]) if int(base) - 1 < len(dwell) else 10.0
        elapsed += max(1.0, dt)
        last = int(base)
    return np.stack(slots, axis=0), selected_sets


def selected_mask_batch(root_selected: torch.Tensor, selected_sets: list[set[int]]) -> torch.Tensor:
    rows = int(root_selected.shape[1])
    masks = root_selected.expand(len(selected_sets), -1).clone()
    for row, selected in enumerate(selected_sets):
        for base in selected:
            if 0 <= int(base) < rows:
                masks[row, int(base)] = True
    return masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--slot-batches", default="1,4,8,16,32,64")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--out", type=Path, default=Path("perf_lab_batched_slots.json"))
    args = parser.parse_args()

    from exact_env_mutual import attach_env_obs
    from final_radar_campaign import get_obs
    from mutual_features import tokenize
    from perf_fast_planner import FastActionAttentionPlanner
    from realistic_reward_retrain import adapter
    from repaired_campaign_tools import EDFPlanner, build_env, env_preset_cfg
    from two_sensor_physical_head_eval import MAXT, ActionAttentionFactorizedNet

    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1

    eng = build_env(EDFPlanner(MAXT), args.initial_targets, MAXT, args.seed, 200, env_cfg)
    eng.reset(seed=args.seed)
    obs = attach_env_obs(get_obs(eng, 0.0), env_cfg, True, True)
    adapt = adapter()
    model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    planner = FastActionAttentionPlanner(model, env_cfg, device=device, use_amp=bool(args.amp), use_compile=bool(args.compile))

    max_slots = max(int(x) for x in str(args.slot_batches).split(",") if x.strip())
    slots_np, selected_sets = make_slot_batch(obs, max_slots, 200.0)
    root_tok = tokenize(adapt, obs, selected=set(), search_count=0).astype(np.float32)
    with torch.inference_mode():
        root_x = torch.from_numpy(root_tok).to(device, dtype=torch.float32).unsqueeze(0)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp and device.type == "cuda")):
            cls_out, tok_out, root_selected, token_active = planner.model.backbone.encode_tokens(root_x)

    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "amp": bool(args.amp),
        "compile": bool(args.compile),
        "slot_batches": [],
    }
    for batch in [int(x) for x in str(args.slot_batches).split(",") if x.strip()]:
        batch = min(int(batch), int(slots_np.shape[0]))
        slots_t = torch.from_numpy(slots_np[:batch]).to(device, dtype=torch.float32)
        selected_t = selected_mask_batch(root_selected, selected_sets[:batch]).to(device)

        with torch.inference_mode():
            loop_scores = []
            for i in range(batch):
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp and device.type == "cuda")):
                    loop_scores.append(planner._scores_from_encoded(cls_out, tok_out, selected_t[i : i + 1], token_active, slots_t[i : i + 1]))
            loop_ref = torch.cat(loop_scores, dim=0).float().cpu()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp and device.type == "cuda")):
                batch_ref = planner.score_slots_from_encoded(cls_out, tok_out, selected_t, token_active, slots_t).float().cpu()
        max_abs_diff = float((loop_ref - batch_ref).abs().max().item())

        loop_times = []
        batch_times = []
        with torch.inference_mode():
            for i in range(int(args.warmup) + int(args.iters)):
                sync(device)
                t0 = time.perf_counter()
                for row in range(batch):
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp and device.type == "cuda")):
                        _ = planner._scores_from_encoded(cls_out, tok_out, selected_t[row : row + 1], token_active, slots_t[row : row + 1])
                sync(device)
                loop_ms = (time.perf_counter() - t0) * 1000.0

                sync(device)
                t1 = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp and device.type == "cuda")):
                    _ = planner.score_slots_from_encoded(cls_out, tok_out, selected_t, token_active, slots_t)
                sync(device)
                batch_ms = (time.perf_counter() - t1) * 1000.0
                if i >= int(args.warmup):
                    loop_times.append(loop_ms)
                    batch_times.append(batch_ms)

        loop_stat = stats(loop_times)
        batch_stat = stats(batch_times)
        action_count = int(batch) * (MAXT + 1) * 2
        report["slot_batches"].append(
            {
                "slot_contexts": int(batch),
                "max_abs_diff": max_abs_diff,
                "sequential_slots": loop_stat,
                "batched_slots": batch_stat,
                "speedup_mean": float(loop_stat["mean_ms"] / max(batch_stat["mean_ms"], 1e-12)),
                "action_scores_per_second_sequential": float(action_count / max(loop_stat["mean_ms"], 1e-12) * 1000.0),
                "action_scores_per_second_batched": float(action_count / max(batch_stat["mean_ms"], 1e-12) * 1000.0),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    eng.close()


if __name__ == "__main__":
    main()
