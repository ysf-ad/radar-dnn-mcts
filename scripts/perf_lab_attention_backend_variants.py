from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
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
        "mean_ms": float(arr.mean()) if arr.size else 0.0,
        "p50_ms": float(np.percentile(arr, 50)) if arr.size else 0.0,
        "p90_ms": float(np.percentile(arr, 90)) if arr.size else 0.0,
        "p99_ms": float(np.percentile(arr, 99)) if arr.size else 0.0,
    }


def _get_sdp_state() -> dict[str, bool]:
    cuda = torch.backends.cuda
    return {
        "flash": bool(cuda.flash_sdp_enabled()),
        "mem_efficient": bool(cuda.mem_efficient_sdp_enabled()),
        "math": bool(cuda.math_sdp_enabled()),
        "cudnn": bool(cuda.cudnn_sdp_enabled()) if hasattr(cuda, "cudnn_sdp_enabled") else False,
    }


def _set_sdp_state(state: dict[str, bool]) -> None:
    cuda = torch.backends.cuda
    cuda.enable_flash_sdp(bool(state.get("flash", False)))
    cuda.enable_mem_efficient_sdp(bool(state.get("mem_efficient", False)))
    cuda.enable_math_sdp(bool(state.get("math", False)))
    if hasattr(cuda, "enable_cudnn_sdp"):
        cuda.enable_cudnn_sdp(bool(state.get("cudnn", False)))


@contextmanager
def sdp_backend(state: dict[str, bool]):
    old = _get_sdp_state()
    _set_sdp_state(state)
    try:
        yield
    finally:
        _set_sdp_state(old)


def time_score(planner, cls, tok, selected, active, slot_t, device: torch.device, iters: int, warmup: int):
    times: list[float] = []
    last = None
    with torch.inference_mode():
        for idx in range(int(warmup) + int(iters)):
            sync(device)
            t0 = time.perf_counter()
            last = planner.score_slots_from_encoded(cls, tok, selected, active, slot_t).float()
            sync(device)
            if idx >= int(warmup):
                times.append((time.perf_counter() - t0) * 1000.0)
    return last, stats(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--initial-targets", type=int, default=60)
    parser.add_argument("--rate", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--out", type=Path, default=Path("results/perf_lab_attention_backend_variants.json"))
    args = parser.parse_args()

    from mutual_features import TOKEN_DIM
    from perf_fast_planner import FastActionAttentionPlanner
    from realistic_reward_retrain import adapter
    from repaired_campaign_tools import env_preset_cfg
    from scripts.perf_lab_multi_env_online_batch import build_envs, pack_root_envs_direct, slot_template_from_packed, tokenize_packed_root_fast
    from two_sensor_physical_head_eval import MAXT, ActionAttentionFactorizedNet

    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1

    env_args = argparse.Namespace(envs=int(args.envs), seed=int(args.seed), initial_targets=int(args.initial_targets), window_ms=200)
    envs = build_envs(env_args, env_cfg)
    ids = list(range(len(envs)))
    debt = [0.0 for _ in ids]
    packed = pack_root_envs_direct(envs, ids, debt, env_cfg, MAXT)
    root_tokens = tokenize_packed_root_fast(adapter(), packed, MAXT, TOKEN_DIM)
    slots = slot_template_from_packed(packed, 200.0)

    model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    planner = FastActionAttentionPlanner(model, env_cfg, device=device, use_amp=False)
    with torch.inference_mode():
        root_t = torch.from_numpy(root_tokens).to(device, dtype=torch.float32)
        slot_t = torch.from_numpy(slots).to(device, dtype=torch.float32)
        cls, tok, selected, active = planner.model.backbone.encode_tokens(root_t)

    variants = {
        "default": _get_sdp_state(),
        "math_only": {"flash": False, "mem_efficient": False, "math": True, "cudnn": False},
        "flash_only": {"flash": True, "mem_efficient": False, "math": False, "cudnn": False},
        "mem_efficient_only": {"flash": False, "mem_efficient": True, "math": False, "cudnn": False},
        "cudnn_only": {"flash": False, "mem_efficient": False, "math": False, "cudnn": True},
        "flash_math": {"flash": True, "mem_efficient": False, "math": True, "cudnn": False},
        "all_no_cudnn": {"flash": True, "mem_efficient": True, "math": True, "cudnn": False},
    }

    results = []
    reference = None
    for name, state in variants.items():
        try:
            with sdp_backend(state):
                out, timing = time_score(planner, cls, tok, selected, active, slot_t, device, int(args.iters), int(args.warmup))
            if reference is None:
                reference = out.detach().clone()
                max_abs_diff = 0.0
            else:
                max_abs_diff = float((reference - out).abs().max().item())
            results.append({"variant": name, "state": state, "timing": timing, "max_abs_diff_vs_default": max_abs_diff})
        except Exception as exc:
            results.append({"variant": name, "state": state, "error": repr(exc)})

    for env in envs:
        try:
            env.close()
        except Exception:
            pass

    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
        "envs": int(args.envs),
        "initial_targets": int(args.initial_targets),
        "rate": float(args.rate),
        "seed": int(args.seed),
        "iters": int(args.iters),
        "warmup": int(args.warmup),
        "score_shape": [int(cls.shape[0]), int(tok.shape[1]), 2],
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
