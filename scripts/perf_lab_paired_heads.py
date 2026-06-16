from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "radar_dnn_mcts"))


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def _unpack_head(head: nn.Sequential) -> tuple[nn.LayerNorm, nn.Linear, nn.Linear]:
    if len(head) != 4 or not isinstance(head[0], nn.LayerNorm) or not isinstance(head[1], nn.Linear) or not isinstance(head[3], nn.Linear):
        raise TypeError(f"Unsupported head layout: {head}")
    return head[0], head[1], head[3]


def paired_direct(head_a: nn.Sequential, head_b: nn.Sequential, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ln_a, l1_a, l2_a = _unpack_head(head_a)
    ln_b, l1_b, l2_b = _unpack_head(head_b)
    xa = F.layer_norm(x, ln_a.normalized_shape, ln_a.weight, ln_a.bias, ln_a.eps)
    xb = F.layer_norm(x, ln_b.normalized_shape, ln_b.weight, ln_b.bias, ln_b.eps)
    ya = F.linear(F.gelu(F.linear(xa, l1_a.weight, l1_a.bias)), l2_a.weight, l2_a.bias)
    yb = F.linear(F.gelu(F.linear(xb, l1_b.weight, l1_b.bias)), l2_b.weight, l2_b.bias)
    return ya, yb


def paired_stacked(head_a: nn.Sequential, head_b: nn.Sequential, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ln_a, l1_a, l2_a = _unpack_head(head_a)
    ln_b, l1_b, l2_b = _unpack_head(head_b)
    x_pair = torch.stack(
        [
            F.layer_norm(x, ln_a.normalized_shape, ln_a.weight, ln_a.bias, ln_a.eps),
            F.layer_norm(x, ln_b.normalized_shape, ln_b.weight, ln_b.bias, ln_b.eps),
        ],
        dim=0,
    )
    w1 = torch.stack([l1_a.weight, l1_b.weight], dim=0)
    b1 = torch.stack([l1_a.bias, l1_b.bias], dim=0)
    h = torch.einsum("p...i,poi->p...o", x_pair, w1) + b1.reshape(2, *([1] * (x.ndim - 1)), -1)
    h = F.gelu(h)
    w2 = torch.stack([l2_a.weight, l2_b.weight], dim=0)
    b2 = torch.stack([l2_a.bias, l2_b.bias], dim=0)
    y = torch.einsum("p...i,poi->p...o", h, w2) + b2.reshape(2, *([1] * (x.ndim - 1)), -1)
    return y[0], y[1]


def bench(device: torch.device, name: str, fn, iters: int, warmup: int) -> dict:
    values = []
    for idx in range(warmup + iters):
        sync(device)
        t0 = time.perf_counter()
        out = fn()
        sync(device)
        if idx >= warmup:
            values.append((time.perf_counter() - t0) * 1000.0)
        if isinstance(out, tuple):
            _ = out[0]
    return {"name": name, **stats(values)}


def bench_pair(device: torch.device, label: str, head_a: nn.Sequential, head_b: nn.Sequential, x: torch.Tensor, iters: int, warmup: int) -> dict:
    with torch.inference_mode():
        ref_a, ref_b = head_a(x), head_b(x)
        direct_a, direct_b = paired_direct(head_a, head_b, x)
        stacked_a, stacked_b = paired_stacked(head_a, head_b, x)
    diff_direct = max(float((ref_a - direct_a).abs().max().item()), float((ref_b - direct_b).abs().max().item()))
    diff_stacked = max(float((ref_a - stacked_a).abs().max().item()), float((ref_b - stacked_b).abs().max().item()))

    rows = [
        bench(device, "separate_modules", lambda: (head_a(x), head_b(x)), iters, warmup),
        bench(device, "paired_direct_functional", lambda: paired_direct(head_a, head_b, x), iters, warmup),
        bench(device, "paired_stacked_einsum", lambda: paired_stacked(head_a, head_b, x), iters, warmup),
    ]
    return {
        "label": label,
        "shape": list(x.shape),
        "max_abs_diff_direct": diff_direct,
        "max_abs_diff_stacked": diff_stacked,
        "timings": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batches", default="1,8,32,64,128")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("results/perf_lab_paired_heads.json"))
    args = parser.parse_args()

    from two_sensor_physical_head_eval import ActionAttentionFactorizedNet

    torch.manual_seed(123)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    model = ActionAttentionFactorizedNet(48, 4, 2).eval().to(device)
    batches = [int(x) for x in str(args.batches).split(",") if str(x).strip()]
    rows = []
    with torch.inference_mode():
        for batch in batches:
            type_ctx = torch.randn(batch, 2, 144, device=device)
            target_ctx = torch.randn(batch, 101, 2, 192, device=device)
            action_ctx = torch.randn(batch, 202, 48, device=device)
            rows.append(bench_pair(device, f"type_b{batch}", model.type_head, model.type_q_head, type_ctx, int(args.iters), int(args.warmup)))
            rows.append(bench_pair(device, f"target_b{batch}", model.target_head, model.target_q_head, target_ctx, int(args.iters), int(args.warmup)))
            rows.append(
                bench_pair(
                    device,
                    f"action_residual_b{batch}",
                    model.action_policy_residual,
                    model.action_q_residual,
                    action_ctx,
                    int(args.iters),
                    int(args.warmup),
                )
            )

    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "iters": int(args.iters),
        "warmup": int(args.warmup),
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
