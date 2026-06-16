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


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def bench_variant(fn, device: torch.device, iters: int, warmup: int) -> tuple[int, dict[str, float]]:
    times: list[float] = []
    last = -1
    with torch.inference_mode():
        for i in range(int(warmup) + int(iters)):
            sync(device)
            t0 = time.perf_counter()
            last = int(fn())
            sync(device)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if i >= int(warmup):
                times.append(float(dt_ms))
    return last, stats(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--search-score-bias", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("results/perf_lab_gpu_select_variants.json"))
    args = parser.parse_args()

    from final_radar_campaign import get_obs
    from exact_env_mutual import attach_env_obs
    from mutual_features import tokenize
    from perf_fast_planner import FastActionAttentionPlanner
    from repaired_campaign_tools import EDFPlanner, build_env, env_preset_cfg
    from two_sensor_physical_head_eval import MAXT, ActionAttentionFactorizedNet

    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1

    eng = build_env(EDFPlanner(MAXT), int(args.initial_targets), MAXT, int(args.seed), 200, env_cfg)
    eng.reset(seed=int(args.seed))
    obs = get_obs(eng, 0.0)

    model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    planner = FastActionAttentionPlanner(
        model,
        env_cfg,
        device=device,
        search_score_bias=float(args.search_score_bias),
        use_gpu_select=True,
    )
    obs = attach_env_obs(obs, env_cfg, True, True)
    root_tok = tokenize(planner.adapt, obs, selected=set(), search_count=0).astype(np.float32)
    slot = planner._slot_template(obs, 200.0)
    actions_t, flat_indices_t, is_search_t = planner._physical_action_tensors(obs)

    with torch.inference_mode():
        root_x = torch.from_numpy(root_tok).to(device, dtype=torch.float32).unsqueeze(0)
        slot_t = torch.from_numpy(slot).to(device, dtype=torch.float32).unsqueeze(0)
        cls_out, tok_out, selected_t, token_active = planner.model.backbone.encode_tokens(root_x)
        score_t = planner._combined_scores_from_encoded(cls_out, tok_out, selected_t, token_active, slot_t).squeeze(0).float()
        flat = score_t.reshape(-1)
        bias = float(args.search_score_bias)

    def with_bias(vals: torch.Tensor) -> torch.Tensor:
        if bias == 0.0:
            return vals
        return vals + is_search_t.to(vals.dtype) * bias

    variants = {
        "index_select_argmax": lambda: actions_t[torch.argmax(with_bias(flat.index_select(0, flat_indices_t)))].item(),
        "take_argmax": lambda: actions_t[torch.argmax(with_bias(torch.take(flat, flat_indices_t)))].item(),
        "advanced_flat_argmax": lambda: actions_t[torch.argmax(with_bias(flat[flat_indices_t]))].item(),
        "gather_argmax": lambda: actions_t[torch.argmax(with_bias(flat.gather(0, flat_indices_t)))].item(),
        "index_select_max": lambda: actions_t[torch.max(with_bias(flat.index_select(0, flat_indices_t)), dim=0).indices].item(),
        "take_max": lambda: actions_t[torch.max(with_bias(torch.take(flat, flat_indices_t)), dim=0).indices].item(),
        "topk1": lambda: actions_t[torch.topk(with_bias(flat.index_select(0, flat_indices_t)), k=1).indices[0]].item(),
    }

    results = []
    reference_action = None
    for name, fn in variants.items():
        action, timing = bench_variant(fn, device, int(args.iters), int(args.warmup))
        if reference_action is None:
            reference_action = int(action)
        results.append(
            {
                "variant": name,
                "selected_action": int(action),
                "matches_reference": int(action) == int(reference_action),
                "timing": timing,
            }
        )

    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "initial_targets": int(args.initial_targets),
        "rate": float(args.rate),
        "seed": int(args.seed),
        "search_score_bias": float(args.search_score_bias),
        "candidate_count": int(actions_t.numel()),
        "score_shape": list(score_t.shape),
        "iters": int(args.iters),
        "warmup": int(args.warmup),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    eng.close()


if __name__ == "__main__":
    main()
