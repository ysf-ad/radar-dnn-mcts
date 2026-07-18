from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from eval_action_attention_muzero_g import (
    CudaGraphARDynamicSOnlyDecode,
    CudaGraphLatentSOnlyWindow,
    LatentG,
    LatentMuZeroPlanner,
    MAXT,
    _DummyPlanner,
    attach_env_obs,
    build_env,
    engine_env_cfg,
    env_cfg_for,
    get_obs,
    load_base_policy_model,
    make_exact_args,
    slot_features,
    tokenize,
)


ROOT = Path(__file__).resolve().parents[4]


def _base_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        windows=100,
        env_mode="current",
        zero_action_rewards=False,
        track_loss_penalty=8.0,
        track_update_reward=0.30,
        target_service_weight=0.0,
        target_service_horizon_ms=3000.0,
        tracked_target_ms_reward_weight=0.0,
        on_time_target_ms_reward_weight=0.0,
        service_pressure_delta_reward_weight=0.0,
        service_pressure_state_penalty_weight=0.0,
        search_frame_state_penalty_weight=0.0,
        search_frame_delta_reward_weight=0.0,
        track_pressure_reward_weight=0.0,
        bounded_service_reward_weight=0.0,
        serviced_target_reward_weight=0.0,
        serviced_target_update_reward_weight=0.0,
        serviced_pressure_delta_reward_weight=0.0,
        discovered_target_reward=0.0,
        sector_staleness_weight=0.0,
        search_frame_overdue_weight=0.20,
        search_frame_desired_ms=3000.0,
        search_frame_deadline_ms=4500.0,
        search_frame_drop_penalty=0.0,
    )


def _load_g(path: str, device: torch.device, ar_history_k: int) -> LatentG:
    ckpt = torch.load(path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    seq_len = int(state["seq_pos"].shape[0]) if isinstance(state, dict) and "seq_pos" in state else 40
    d_model = int(state["sensor_embed"].shape[1]) if isinstance(state, dict) and "sensor_embed" in state else 48
    g = LatentG(d_model=d_model, seq_len=seq_len, ar_history_k=int(ar_history_k)).to(device).eval()
    missing, unexpected = g.load_state_dict(state, strict=False)
    if missing or unexpected:
        print({"g_load_missing": missing, "g_load_unexpected": unexpected}, flush=True)
    for param in g.parameters():
        param.requires_grad_(False)
    return g


def _make_obs(args: argparse.Namespace) -> tuple[dict, dict]:
    exact_args = make_exact_args(_base_args(args))
    exact_args.enable_x_band = False
    exact_args.single_sensor = True
    env_cfg = env_cfg_for(float(args.rate), exact_args)
    env_cfg["enable_x_band"] = 0
    eng = build_env(_DummyPlanner(), int(args.initial), MAXT, int(args.seed), 200, engine_env_cfg(env_cfg))
    try:
        eng.reset(seed=int(args.seed))
        obs = get_obs(eng, 0.0)
    finally:
        eng.close()
    return obs, env_cfg


def _bench(name: str, fn, repeats: int) -> dict[str, float]:
    times = []
    for _ in range(int(repeats)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(1000.0 * (time.perf_counter() - t0))
    arr = np.asarray(times, dtype=np.float64)
    return {
        "name": name,
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def _print_table(rows: list[dict[str, float]]) -> None:
    print("name,mean_ms,median_ms,p90_ms,min_ms,max_ms")
    for row in rows:
        print(
            f"{row['name']},{row['mean_ms']:.4f},{row['median_ms']:.4f},"
            f"{row['p90_ms']:.4f},{row['min_ms']:.4f},{row['max_ms']:.4f}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-state", default=str(ROOT / "CreateValid1" / "results" / "mixed_gate_distill_180_action_attention_step40_state.pt"))
    ap.add_argument("--ar-state", default=str(ROOT / "CreateValid1" / "results" / "ar_histk4_sonly_teacherbias6_refit_v2.pt"))
    ap.add_argument("--muzero-state", default=str(ROOT / "CreateValid1" / "results" / "muzero_gpolicy_sonly_teacherbias6_absorb.pt"))
    ap.add_argument("--variant", default="two_row_action_attention")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--initial", type=int, default=60)
    ap.add_argument("--rate", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=916)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--ar-history-k", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--ar-q-weight", type=float, default=0.0)
    ap.add_argument("--muzero-q-weight", type=float, default=0.0)
    ap.add_argument("--muzero-policy-action-mixer", choices=["full", "tiny", "light", "none"], default="full")
    args = ap.parse_args()

    if not torch.cuda.is_available() or str(args.device) != "cuda":
        raise RuntimeError("This profiler is for CUDA latency paths.")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cuda")

    model_args = SimpleNamespace(lean_base_load=True, base_state=str(args.base_state), variant=str(args.variant))
    model = load_base_policy_model(model_args, device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    ar_g = _load_g(str(args.ar_state), device, int(args.ar_history_k))
    mz_g = _load_g(str(args.muzero_state), device, 0)
    mz_g.policy_action_mixer = str(args.muzero_policy_action_mixer)
    obs, env_cfg = _make_obs(args)

    ar_planner = LatentMuZeroPlanner(
        model,
        ar_g,
        env_cfg,
        policy_weight=1.0,
        q_weight=float(args.ar_q_weight),
        search_score_bias=0.0,
        max_steps=int(args.max_steps),
        cuda_graph_ar_dynamic=True,
        use_ar_seq_policy=True,
        root_seq_factorized_decode=True,
        device=str(device),
    )
    mz_planner = LatentMuZeroPlanner(
        model,
        mz_g,
        env_cfg,
        policy_weight=1.0,
        q_weight=float(args.muzero_q_weight),
        search_score_bias=0.0,
        max_steps=int(args.max_steps),
        cuda_graph_tensor_loop=True,
        cuda_graph_s_only_score=True,
        tensor_loop=True,
        single_sensor_noop_action=True,
        use_g_policy=True,
        device=str(device),
    )

    ar_planner.warmup(obs)
    mz_planner.warmup(obs)
    for _ in range(int(args.warmup)):
        ar_planner.plan(obs)
        mz_planner.plan(obs)
    torch.cuda.synchronize()

    obs_attached = attach_env_obs(dict(obs), env_cfg, True, True)
    slot_np = slot_features(obs_attached, 0.0, 0, 0, -1, 200.0).astype(np.float32)
    x_np = tokenize(ar_planner.adapt, obs_attached, selected=set(), search_count=0).astype(np.float32)
    x = torch.from_numpy(x_np[None]).float().to(device)
    slot = torch.from_numpy(slot_np).float().to(device)
    dwell_np = np.asarray(obs_attached.get("t_dwell", []), dtype=np.float32)
    dwell_full = np.ones((MAXT + 1,), dtype=np.float32) * 10.0
    n_dwell = min(MAXT, int(dwell_np.size))
    if n_dwell > 0:
        dwell_full[1 : n_dwell + 1] = np.maximum(1.0, dwell_np[:n_dwell])
    dwell = torch.from_numpy(dwell_full).to(device)
    row_map = torch.arange(MAXT + 1, dtype=torch.long, device=device)
    const = torch.tensor(ar_planner._slot_constants(obs_attached, 200.0), dtype=torch.float32, device=device)
    service_bonus = torch.from_numpy(mz_planner._service_bonus_np(obs_attached)).to(device)

    ar_graph = ar_planner._graph_ar_dynamic
    if ar_graph is None:
        ar_graph = CudaGraphARDynamicSOnlyDecode(model, ar_g, x, slot, dwell, row_map, max_steps=int(args.max_steps), search_score_bias=0.0)
    mz_graph = mz_planner._graph_sonly_window
    if mz_graph is None:
        mz_graph = CudaGraphLatentSOnlyWindow(mz_planner, x, const, dwell, service_bonus, row_map)

    rows = [
        _bench("AR_full_plan", lambda: ar_planner.plan(obs), int(args.repeats)),
        _bench("MuZero_full_plan", lambda: mz_planner.plan(obs), int(args.repeats)),
        _bench("AR_graph_replay_only", lambda: ar_graph(x, slot, dwell, row_map), int(args.repeats)),
        _bench("MuZero_graph_replay_only", lambda: mz_graph(x, const, dwell, service_bonus, row_map), int(args.repeats)),
        _bench("AR_rows_cpu_decode", lambda: [int(v) for v in ar_graph(x, slot, dwell, row_map).detach().cpu().tolist() if int(v) >= 0], int(args.repeats)),
        _bench("MuZero_rows_cpu_decode", lambda: [int(v) for v in mz_graph(x, const, dwell, service_bonus, row_map).detach().cpu().tolist()], int(args.repeats)),
    ]
    _print_table(rows)


if __name__ == "__main__":
    main()
