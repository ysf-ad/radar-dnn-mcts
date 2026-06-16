from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "radar_dnn_mcts"))
sys.path.insert(0, str(ROOT / "scripts"))


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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--active", type=int, default=20)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/profile_score_kernels.json"))
    args = parser.parse_args()

    from perf_lab_score_compile_variants import make_inputs

    torch.manual_seed(123)
    torch.set_num_threads(1)
    if str(args.matmul_precision):
        torch.set_float32_matmul_precision(str(args.matmul_precision))
    device = torch.device(args.device)
    planner, cls_out, tok_out, selected_t, token_active, slot_t = make_inputs(args, device)

    def score_once():
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=bool(args.amp) and device.type == "cuda",
        ):
            return planner.score_slots_from_encoded(cls_out, tok_out, selected_t, token_active, slot_t).float()

    for _ in range(int(args.warmup)):
        _ = score_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(int(args.active)):
            _ = score_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.trace))

    rows = []
    sort_by = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    for item in prof.key_averages(group_by_input_shape=False).table(sort_by=sort_by, row_limit=int(args.top)).splitlines():
        rows.append(item)

    events = []
    for evt in prof.key_averages(group_by_input_shape=False):
        device_time_total = float(getattr(evt, "device_time_total", 0.0) or getattr(evt, "cuda_time_total", 0.0) or 0.0)
        self_device_time_total = float(getattr(evt, "self_device_time_total", 0.0) or getattr(evt, "self_cuda_time_total", 0.0) or 0.0)
        device_memory_usage = int(getattr(evt, "device_memory_usage", 0) or getattr(evt, "cuda_memory_usage", 0) or 0)
        self_device_memory_usage = int(getattr(evt, "self_device_memory_usage", 0) or getattr(evt, "self_cuda_memory_usage", 0) or 0)
        events.append(
            {
                "key": evt.key,
                "count": int(evt.count),
                "cpu_time_total_us": float(evt.cpu_time_total),
                "device_time_total_us": device_time_total,
                "self_cpu_time_total_us": float(evt.self_cpu_time_total),
                "self_device_time_total_us": self_device_time_total,
                "cpu_memory_usage": int(evt.cpu_memory_usage),
                "device_memory_usage": device_memory_usage,
                "self_device_memory_usage": self_device_memory_usage,
            }
        )
    events.sort(key=lambda x: x["device_time_total_us" if device.type == "cuda" else "cpu_time_total_us"], reverse=True)

    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
        "amp": bool(args.amp),
        "matmul_precision": str(args.matmul_precision) if str(args.matmul_precision) else None,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "prefixes": int(args.prefixes),
        "active_iterations": int(args.active),
        "top_table": rows,
        "events": events[: int(args.top)],
        "trace": str(args.trace) if args.trace is not None else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
