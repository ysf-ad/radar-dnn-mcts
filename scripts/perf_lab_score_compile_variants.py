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
        "mean_ms": float(arr.mean()) if arr.size else 0.0,
        "p50_ms": float(np.percentile(arr, 50)) if arr.size else 0.0,
        "p90_ms": float(np.percentile(arr, 90)) if arr.size else 0.0,
        "p99_ms": float(np.percentile(arr, 99)) if arr.size else 0.0,
    }


def load_model_checkpoint(model, checkpoint: str | Path | None):
    if checkpoint is None or str(checkpoint).strip() == "":
        return model
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state, strict=True)
    return model


def make_inputs(args, device: torch.device):
    from batched_window_expansion import BatchedWindowExpansionScorer
    from final_radar_campaign import get_obs
    from perf_fast_planner import FastActionAttentionPlanner
    from profile_cached_action_attention_internals import make_prefix_tensors
    from repaired_campaign_tools import EDFPlanner, build_env, env_preset_cfg
    from two_sensor_physical_head_eval import MAXT, ActionAttentionFactorizedNet

    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1
    eng = build_env(EDFPlanner(MAXT), int(args.initial_targets), MAXT, int(args.seed), 200, env_cfg)
    eng.reset(seed=int(args.seed))
    obs = get_obs(eng, 0.0)
    model = load_model_checkpoint(ActionAttentionFactorizedNet(48, 4, 2).eval(), args.checkpoint)
    planner = FastActionAttentionPlanner(model, env_cfg, device=device, use_amp=bool(args.amp))
    scorer = BatchedWindowExpansionScorer(planner, obs, budget_ms=200.0)
    selected_t, slot_t = make_prefix_tensors(scorer, int(args.prefixes))
    cls_out = scorer.cls_out
    tok_out = scorer.tok_out
    token_active = scorer.token_active
    eng.close()
    return planner, cls_out, tok_out, selected_t, token_active, slot_t


class ScoreTraceWrapper(torch.nn.Module):
    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        self.model = planner.model

    def forward(self, cls_out, tok_out, selected_t, token_active, slot_t):
        return self.planner.score_slots_from_encoded(cls_out, tok_out, selected_t, token_active, slot_t).float()


def time_variant(device: torch.device, name: str, fn, iters: int, warmup: int) -> dict[str, object]:
    values: list[float] = []
    last = None
    compile_ms = 0.0
    try:
        with torch.inference_mode():
            for idx in range(int(warmup) + int(iters)):
                sync(device)
                t0 = time.perf_counter()
                last = fn()
                sync(device)
                elapsed = (time.perf_counter() - t0) * 1000.0
                if idx == 0:
                    compile_ms = float(elapsed)
                if idx >= int(warmup):
                    values.append(float(elapsed))
    except Exception as exc:
        return {"name": name, "error": repr(exc), "calls": int(len(values)), "first_call_ms": compile_ms}
    report = stats(values)
    report["name"] = name
    report["calls"] = int(len(values))
    report["first_call_ms"] = compile_ms
    report["checksum"] = float(last.float().sum().detach().cpu()) if last is not None else 0.0
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--matmul-precision", default="", choices=["", "highest", "high", "medium"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--prefixes", type=int, default=64)
    parser.add_argument("--initial-targets", type=int, default=60)
    parser.add_argument("--rate", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("results/perf_lab_score_compile_variants.json"))
    args = parser.parse_args()

    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)
    if str(args.matmul_precision):
        torch.set_float32_matmul_precision(str(args.matmul_precision))
    device = torch.device(args.device)
    planner, cls_out, tok_out, selected_t, token_active, slot_t = make_inputs(args, device)
    for param in planner.model.parameters():
        param.requires_grad_(False)

    def eager_score():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp) and device.type == "cuda"):
            return planner.score_slots_from_encoded(cls_out, tok_out, selected_t, token_active, slot_t).float()

    compiled_reduce = None
    compiled_max = None
    traced_score = None
    trace_error = None
    if hasattr(torch, "compile"):
        compiled_reduce = torch.compile(
            planner.score_slots_from_encoded,
            mode="reduce-overhead",
            fullgraph=False,
        )
        compiled_max = torch.compile(
            planner.score_slots_from_encoded,
            mode="max-autotune",
            fullgraph=False,
        )
    try:
        trace_wrapper = ScoreTraceWrapper(planner).eval().to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp) and device.type == "cuda"):
            traced_score = torch.jit.trace(
                trace_wrapper,
                (cls_out, tok_out, selected_t, token_active, slot_t),
                check_trace=False,
            )
    except Exception as exc:
        trace_error = repr(exc)

    def compiled_reduce_score():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp) and device.type == "cuda"):
            return compiled_reduce(cls_out, tok_out, selected_t, token_active, slot_t).float()

    def compiled_max_score():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp) and device.type == "cuda"):
            return compiled_max(cls_out, tok_out, selected_t, token_active, slot_t).float()

    variants = [time_variant(device, "eager", eager_score, int(args.iters), int(args.warmup))]
    if traced_score is not None:
        def traced_score_call():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp) and device.type == "cuda"):
                return traced_score(cls_out, tok_out, selected_t, token_active, slot_t).float()

        variants.append(time_variant(device, "torchscript_trace", traced_score_call, int(args.iters), int(args.warmup)))
    if compiled_reduce is not None:
        variants.append(time_variant(device, "compile_reduce_overhead", compiled_reduce_score, int(args.iters), int(args.warmup)))
        variants.append(time_variant(device, "compile_max_autotune", compiled_max_score, int(args.iters), int(args.warmup)))

    with torch.inference_mode():
        ref = eager_score()
    equivalence = {}
    if compiled_reduce is not None:
        if traced_score is not None:
            try:
                trace_out = traced_score_call()
                equivalence["trace_max_abs"] = float((ref - trace_out).abs().max().detach().cpu())
                equivalence["trace_allclose"] = bool(torch.allclose(ref, trace_out, atol=1e-5, rtol=1e-5))
            except Exception as exc:
                equivalence["trace_error"] = repr(exc)
        elif trace_error is not None:
            equivalence["trace_error"] = trace_error
        try:
            reduce_out = compiled_reduce_score()
            equivalence["compile_reduce_max_abs"] = float((ref - reduce_out).abs().max().detach().cpu())
            equivalence["compile_reduce_allclose"] = bool(torch.allclose(ref, reduce_out, atol=1e-5, rtol=1e-5))
        except Exception as exc:
            equivalence["compile_reduce_error"] = repr(exc)
        try:
            max_out = compiled_max_score()
            equivalence["compile_max_max_abs"] = float((ref - max_out).abs().max().detach().cpu())
            equivalence["compile_max_allclose"] = bool(torch.allclose(ref, max_out, atol=1e-5, rtol=1e-5))
        except Exception as exc:
            equivalence["compile_max_error"] = repr(exc)

    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
        "amp": bool(args.amp),
        "matmul_precision": str(args.matmul_precision) if str(args.matmul_precision) else None,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "prefixes": int(args.prefixes),
        "score_shape": list(ref.shape),
        "equivalence": equivalence,
        "variants": variants,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
