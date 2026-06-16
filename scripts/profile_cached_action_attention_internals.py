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


def timed(device, buckets, name, fn):
    sync(device)
    t0 = time.perf_counter()
    out = fn()
    sync(device)
    buckets[name].append((time.perf_counter() - t0) * 1000.0)
    return out


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


def make_prefix_tensors(scorer, count: int):
    from batched_window_expansion import BranchPrefix, prefix_after_action
    from perf_fast_planner import physical_action_arrays
    from two_sensor_physical_head_eval import MAXT

    actions, _bases, _sensors = physical_action_arrays(scorer.obs, selected=set(), max_trackers=MAXT)
    prefixes = [BranchPrefix()]
    for idx, action in enumerate(actions):
        if len(prefixes) >= int(count):
            break
        p1 = prefix_after_action(scorer.obs, BranchPrefix(), int(action))
        prefixes.append(p1)
        for action2 in actions[idx + 1 : idx + 4]:
            if len(prefixes) >= int(count):
                break
            prefixes.append(prefix_after_action(scorer.obs, p1, int(action2)))
    while len(prefixes) < int(count):
        prefixes.append(prefixes[len(prefixes) % max(1, len(prefixes))])
    prefixes = prefixes[: int(count)]
    return scorer._selected_masks(prefixes), scorer._slots(prefixes)


def profile_cached_scores(planner, scorer, selected_t, slot_t, iters: int, warmup: int):
    model = planner.model
    device = planner.device
    amp_enabled = bool(getattr(planner, "use_amp", False)) and torch.device(device).type == "cuda"
    cls_out = scorer.cls_out
    tok_out = scorer.tok_out
    token_active = scorer.token_active
    buckets: dict[str, list[float]] = {
        "slot_projection": [],
        "sensor_coupling": [],
        "type_heads": [],
        "target_context_build": [],
        "target_heads": [],
        "base_score_build": [],
        "action_projection": [],
        "action_self_attention": [],
        "residual_heads": [],
        "mask_and_combine": [],
        "full_cached_score": [],
        "cpu_transfer": [],
    }

    for i in range(int(warmup) + int(iters)):
        record = i >= int(warmup)
        bsz = int(slot_t.shape[0])
        rows = int(tok_out.shape[1])

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            if record:
                slot_emb = timed(device, buckets, "slot_projection", lambda: model.backbone.slot_proj(slot_t))
            else:
                slot_emb = model.backbone.slot_proj(slot_t)

            def sensor_step():
                sensor = model.sensor_embed[None, :, :].expand(bsz, -1, -1)
                cls_s = cls_out.expand(bsz, -1)[:, None, :].expand(-1, 2, -1)
                slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
                sensor_state = model.sensor_state_proj(torch.cat([cls_s, slot_s, sensor], dim=-1))
                coupled_sensor = model.sensor_coupler(sensor_state)
                type_ctx = torch.cat([cls_s, slot_s, coupled_sensor], dim=-1)
                return cls_s, slot_s, coupled_sensor, type_ctx

            if record:
                cls_s, slot_s, coupled_sensor, type_ctx = timed(device, buckets, "sensor_coupling", sensor_step)
            else:
                cls_s, slot_s, coupled_sensor, type_ctx = sensor_step()

            def type_step():
                return model.type_head(type_ctx), model.type_q_head(type_ctx)

            if record:
                type_logits, type_q = timed(device, buckets, "type_heads", type_step)
            else:
                type_logits, type_q = type_step()

            def target_context_step():
                tok_st = tok_out.expand(bsz, -1, -1)[:, :, None, :].expand(-1, -1, 2, -1)
                cls_st = cls_out.expand(bsz, -1)[:, None, None, :].expand(-1, rows, 2, -1)
                slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
                sensor_st = coupled_sensor[:, None, :, :].expand(bsz, rows, -1, -1)
                return torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)

            if record:
                target_ctx = timed(device, buckets, "target_context_build", target_context_step)
            else:
                target_ctx = target_context_step()

            def target_step():
                return model.target_head(target_ctx).squeeze(-1), model.target_q_head(target_ctx).squeeze(-1)

            if record:
                target_logits, target_q = timed(device, buckets, "target_heads", target_step)
            else:
                target_logits, target_q = target_step()

            def base_step():
                base_scores = slot_t.new_full((bsz, rows, 2), -1e9)
                base_q = slot_t.new_zeros((bsz, rows, 2))
                base_scores[:, 0, :] = type_logits[:, :, 0]
                base_q[:, 0, :] = type_q[:, :, 0]
                track_mask = token_active.expand(bsz, -1) & ~selected_t
                track_mask[:, 0] = False
                base_scores[:, 1:, :] = (type_logits[:, None, :, 1] + target_logits)[:, 1:, :]
                base_q[:, 1:, :] = (type_q[:, None, :, 1] + target_q)[:, 1:, :]
                row_is_search = torch.arange(rows, device=slot_t.device)[None, :, None] == 0
                valid = (track_mask[:, :, None] | row_is_search).expand(-1, -1, 2)
                return base_scores, base_q, valid

            if record:
                base_scores, base_q, valid = timed(device, buckets, "base_score_build", base_step)
            else:
                base_scores, base_q, valid = base_step()

            if record:
                action_ctx = timed(device, buckets, "action_projection", lambda: model.action_proj(target_ctx).reshape(bsz, rows * 2, -1))
            else:
                action_ctx = model.action_proj(target_ctx).reshape(bsz, rows * 2, -1)

            if record:
                action_ctx = timed(
                    device,
                    buckets,
                    "action_self_attention",
                    lambda: model.action_coupler(action_ctx, src_key_padding_mask=~valid.reshape(bsz, rows * 2)),
                )
            else:
                action_ctx = model.action_coupler(action_ctx, src_key_padding_mask=~valid.reshape(bsz, rows * 2))

            def residual_step():
                residual = model.action_policy_residual(action_ctx).reshape(bsz, rows, 2)
                q_residual = model.action_q_residual(action_ctx).reshape(bsz, rows, 2)
                return residual, q_residual

            if record:
                residual, q_residual = timed(device, buckets, "residual_heads", residual_step)
            else:
                residual, q_residual = residual_step()

            if record:
                score_t = timed(
                    device,
                    buckets,
                    "mask_and_combine",
                    lambda: (planner.policy_weight * (base_scores + residual).masked_fill(~valid, -1e9))
                    + (planner.q_weight * (base_q + q_residual).masked_fill(~valid, 0.0)),
                )
            else:
                score_t = (planner.policy_weight * (base_scores + residual).masked_fill(~valid, -1e9)) + (
                    planner.q_weight * (base_q + q_residual).masked_fill(~valid, 0.0)
                )

            if record:
                timed(device, buckets, "full_cached_score", lambda: planner.score_slots_from_encoded(cls_out, tok_out, selected_t, token_active, slot_t))
                timed(device, buckets, "cpu_transfer", lambda: score_t.float().cpu().numpy())

    return {key: stats(vals) for key, vals in buckets.items() if vals}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--matmul-precision", default="", choices=["", "highest", "high", "medium"])
    parser.add_argument("--prefix-batches", default="1,4,8,16,32,64")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional ActionAttentionFactorizedNet state dict to profile.")
    parser.add_argument("--out", type=Path, default=Path("profile_cached_action_attention_internals.json"))
    args = parser.parse_args()

    from batched_window_expansion import BatchedWindowExpansionScorer
    from final_radar_campaign import get_obs
    from perf_fast_planner import FastActionAttentionPlanner
    from repaired_campaign_tools import EDFPlanner, build_env, env_preset_cfg
    from two_sensor_physical_head_eval import MAXT, ActionAttentionFactorizedNet

    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)
    if str(args.matmul_precision):
        torch.set_float32_matmul_precision(str(args.matmul_precision))
    device = torch.device(args.device)
    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1

    eng = build_env(EDFPlanner(MAXT), args.initial_targets, MAXT, args.seed, 200, env_cfg)
    eng.reset(seed=args.seed)
    obs = get_obs(eng, 0.0)
    model = load_model_checkpoint(ActionAttentionFactorizedNet(48, 4, 2).eval(), args.checkpoint)
    planner = FastActionAttentionPlanner(model, env_cfg, device=device, use_amp=bool(args.amp), use_compile=bool(args.compile))
    scorer = BatchedWindowExpansionScorer(planner, obs, budget_ms=200.0)

    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "amp": bool(args.amp),
        "compile": bool(args.compile),
        "matmul_precision": str(args.matmul_precision) if str(args.matmul_precision) else None,
        "initial_targets": int(args.initial_targets),
        "rate": float(args.rate),
        "seed": int(args.seed),
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "prefix_batches": [],
    }

    for batch in [int(x) for x in str(args.prefix_batches).split(",") if x.strip()]:
        selected_t, slot_t = make_prefix_tensors(scorer, int(batch))
        result = profile_cached_scores(planner, scorer, selected_t, slot_t, int(args.iters), int(args.warmup))
        full_mean = result.get("full_cached_score", {}).get("mean_ms", 0.0)
        report["prefix_batches"].append(
            {
                "prefixes": int(batch),
                "rows": int(scorer.tok_out.shape[1]),
                "action_tokens": int(scorer.tok_out.shape[1] * 2),
                "full_cached_scores_per_second": float(batch / max(full_mean, 1e-12) * 1000.0),
                "stages": result,
            }
        )

    eng.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
