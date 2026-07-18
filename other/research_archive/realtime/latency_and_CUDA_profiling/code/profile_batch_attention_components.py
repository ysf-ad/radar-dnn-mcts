"""Warmed component profiler for attention and independent batch decoders."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from canonical_batch_decoder import CudaGraphBatchDecoder, load_model
from mutual_features import SLOT_DIM, TOKEN_DIM


def gpu_time_ms(fn, iterations: int, warmup: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


def cpu_time_ms(fn, iterations: int, warmup: int = 50) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) * 1000.0 / iterations


def profile(name: str, checkpoint: Path, iterations: int) -> list[dict]:
    model = load_model(str(checkpoint), "cuda").half().eval()
    x = torch.randn(1, 101, TOKEN_DIM, device="cuda", dtype=torch.float16)
    x[:, :, 4] = 1.0
    x[:, :, 10] = 1.0
    slot = torch.randn(1, SLOT_DIM, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        eager = lambda: model(x, slot)
        eager_ms = gpu_time_ms(eager, iterations)
        graph = CudaGraphBatchDecoder(model, x, slot)
        replay_ms = gpu_time_ms(graph.graph.replay, iterations)
        copy_replay_ms = gpu_time_ms(lambda: graph(x, slot), iterations)
        outputs = graph(x, slot)
        type_logits, target_logits, valid = outputs
        d2h_ms = cpu_time_ms(
            lambda: (
                type_logits.detach().cpu(),
                target_logits.detach().cpu(),
                valid.detach().cpu(),
                torch.cuda.synchronize(),
            ),
            max(100, iterations // 10),
            warmup=10,
        )

    score = np.random.default_rng(916).normal(size=(20, 100)).astype(np.float32)
    assignment_ms = cpu_time_ms(lambda: linear_sum_assignment(-score), iterations)
    params = sum(parameter.numel() for parameter in model.parameters())
    return [
        {"model": name, "component": "eager_forward", "ms": eager_ms, "parameters": params},
        {"model": name, "component": "cuda_graph_replay", "ms": replay_ms, "parameters": params},
        {"model": name, "component": "device_copy_plus_graph", "ms": copy_replay_ms, "parameters": params},
        {"model": name, "component": "output_device_to_host", "ms": d2h_ms, "parameters": params},
        {"model": name, "component": "hungarian_assignment_cpu", "ms": assignment_ms, "parameters": params},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = profile("attention", args.attention, args.iterations)
    rows.extend(profile("independent", args.independent, args.iterations))
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
