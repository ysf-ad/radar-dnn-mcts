from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


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


def manual_encoder_layer(layer: nn.TransformerEncoderLayer, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
    if layer.norm_first:
        y = layer.norm1(x)
        attn = layer.self_attn(
            y,
            y,
            y,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=False,
        )[0]
        x = x + layer.dropout1(attn)
        y = layer.norm2(x)
        y = layer.linear2(layer.dropout(layer.activation(layer.linear1(y))))
        return x + layer.dropout2(y)
    attn = layer.self_attn(
        x,
        x,
        x,
        attn_mask=None,
        key_padding_mask=key_padding_mask,
        need_weights=False,
        is_causal=False,
    )[0]
    x = layer.norm1(x + layer.dropout1(attn))
    y = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
    return layer.norm2(x + layer.dropout2(y))


def time_variant(device: torch.device, name: str, fn, iters: int, warmup: int) -> dict[str, float]:
    vals: list[float] = []
    with torch.inference_mode():
        for i in range(int(warmup) + int(iters)):
            sync(device)
            t0 = time.perf_counter()
            out = fn()
            sync(device)
            if i >= int(warmup):
                vals.append((time.perf_counter() - t0) * 1000.0)
    report = stats(vals)
    report["calls"] = int(len(vals))
    report["name"] = name
    report["checksum"] = float(out.float().sum().detach().cpu())
    return report


def build_action_inputs(model, device: torch.device, args):
    from batched_window_expansion import BatchedWindowExpansionScorer
    from final_radar_campaign import get_obs
    from perf_fast_planner import FastActionAttentionPlanner
    from profile_cached_action_attention_internals import make_prefix_tensors
    from repaired_campaign_tools import EDFPlanner, build_env, env_preset_cfg
    from two_sensor_physical_head_eval import MAXT

    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1
    eng = build_env(EDFPlanner(MAXT), int(args.initial_targets), MAXT, int(args.seed), 200, env_cfg)
    eng.reset(seed=int(args.seed))
    obs = get_obs(eng, 0.0)
    planner = FastActionAttentionPlanner(model, env_cfg, device=device, use_amp=bool(args.amp))
    scorer = BatchedWindowExpansionScorer(planner, obs, budget_ms=200.0)
    selected_t, slot_t = make_prefix_tensors(scorer, int(args.prefixes))
    cls_out = scorer.cls_out
    tok_out = scorer.tok_out
    token_active = scorer.token_active
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp) and device.type == "cuda"):
            slot_emb = model.backbone.slot_proj(slot_t)
            bsz = int(slot_t.shape[0])
            rows = int(tok_out.shape[1])
            sensor = model.sensor_embed[None, :, :].expand(bsz, -1, -1)
            cls_s = cls_out.expand(bsz, -1)[:, None, :].expand(-1, 2, -1)
            slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
            sensor_state = model.sensor_state_proj(torch.cat([cls_s, slot_s, sensor], dim=-1))
            coupled_sensor = model.sensor_coupler(sensor_state)
            tok_st = tok_out.expand(bsz, -1, -1)[:, :, None, :].expand(-1, -1, 2, -1)
            cls_st = cls_out.expand(bsz, -1)[:, None, None, :].expand(-1, rows, 2, -1)
            slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
            sensor_st = coupled_sensor[:, None, :, :].expand(bsz, rows, -1, -1)
            target_ctx = torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)
            track_mask = token_active.expand(bsz, -1) & ~selected_t
            track_mask[:, 0] = False
            row_is_search = torch.arange(rows, device=device)[None, :, None] == 0
            valid = (track_mask[:, :, None] | row_is_search).expand(-1, -1, 2)
            action_ctx = model.action_proj(target_ctx).reshape(bsz, rows * 2, -1)
            action_mask = ~valid.reshape(bsz, rows * 2)
    return action_ctx, action_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--prefixes", type=int, default=64)
    parser.add_argument("--initial-targets", type=int, default=60)
    parser.add_argument("--rate", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("results/perf_lab_coupler_variants.json"))
    args = parser.parse_args()

    from two_sensor_physical_head_eval import ActionAttentionFactorizedNet

    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    model = load_model_checkpoint(ActionAttentionFactorizedNet(48, 4, 2).eval(), args.checkpoint).to(device)
    action_ctx, action_mask = build_action_inputs(model, device, args)
    layer = model.action_coupler.layers[0]

    def baseline():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp) and device.type == "cuda"):
            return model.action_coupler(action_ctx, src_key_padding_mask=action_mask)

    def direct_layer():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp) and device.type == "cuda"):
            return layer(action_ctx, src_key_padding_mask=action_mask)

    def manual_layer():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp) and device.type == "cuda"):
            return manual_encoder_layer(layer, action_ctx, action_mask)

    with torch.inference_mode():
        ref = baseline()
        direct = direct_layer()
        manual = manual_layer()
    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "amp": bool(args.amp),
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "input_shape": list(action_ctx.shape),
        "mask_true": int(action_mask.sum().detach().cpu()),
        "equivalence": {
            "direct_max_abs": float((ref.float() - direct.float()).abs().max().detach().cpu()),
            "direct_allclose": bool(torch.allclose(ref.float(), direct.float(), atol=1e-5, rtol=1e-5)),
            "manual_max_abs": float((ref.float() - manual.float()).abs().max().detach().cpu()),
            "manual_allclose": bool(torch.allclose(ref.float(), manual.float(), atol=1e-5, rtol=1e-5)),
        },
        "variants": [
            time_variant(device, "transformer_encoder", baseline, int(args.iters), int(args.warmup)),
            time_variant(device, "direct_layer", direct_layer, int(args.iters), int(args.warmup)),
            time_variant(device, "manual_layer", manual_layer, int(args.iters), int(args.warmup)),
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
