from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from distill_sparse64_sequence_decoder import Sparse64SequenceDecoder, load_sequences
from exact_env_mutual import env_cfg_for
from penalty_window_quota_learner_eval import make_exact_args
from single_sensor_ar_action_attention import CachedSingleSensorActionAttentionAR, load_action_attention_model
from two_sensor_physical_head_eval import PhysicalHeadPlanner


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SEQ_CKPT = ROOT / "CreateValid1" / "results" / "large_sparse64_seqdistill_balance025" / "seqdistill.pt"
DEFAULT_SEQ_DATA = ROOT / "CreateValid1" / "results" / "large_sparse64_seqdistill" / "teacher_sequences.pt"
DEFAULT_ATTENTION_STATE = (
    ROOT
    / "CreateValid1"
    / "results"
    / "single_sensor_fair_exact_action_attention_train_two_row_action_attention_qpolicy_factored_loss.pt"
)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_ms(device: torch.device, fn, repeats: int) -> float:
    for _ in range(10):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(int(repeats)):
        fn()
    sync(device)
    return 1000.0 * (time.perf_counter() - t0) / max(1, int(repeats))


def make_reward_args(windows: int) -> SimpleNamespace:
    return SimpleNamespace(
        windows=int(windows),
        env_mode="pufferlib_service",
        search_frame_overdue_weight=0.5,
        search_frame_drop_penalty=8.0,
        search_frame_state_penalty_weight=2.0,
        search_frame_delta_reward_weight=5.0,
        service_pressure_delta_reward_weight=0.30,
        serviced_pressure_improvement_reward_weight=0.15,
        discovered_target_reward=0.08,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-ckpt", type=Path, default=DEFAULT_SEQ_CKPT)
    ap.add_argument("--seq-data", type=Path, default=DEFAULT_SEQ_DATA)
    ap.add_argument("--attention-state", type=Path, default=DEFAULT_ATTENTION_STATE)
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--out", type=Path, default=ROOT / "CreateValid1" / "results" / "device_lower_bound_profile.csv")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    seqs = load_sequences(args.seq_data)
    seq = seqs[min(max(0, int(args.sample_index)), len(seqs) - 1)]
    root = torch.as_tensor(seq["root_x"], dtype=torch.float32, device=device).unsqueeze(0)
    slots = torch.as_tensor(seq["slots"][: int(args.steps)], dtype=torch.float32, device=device).unsqueeze(0)
    prev_class = torch.as_tensor(seq.get("prev_classes", [0] * int(args.steps))[: int(args.steps)], dtype=torch.long, device=device).unsqueeze(0)
    prev_row = torch.as_tensor(seq.get("prev_rows", [0.0] * int(args.steps))[: int(args.steps)], dtype=torch.float32, device=device).unsqueeze(0)
    labels = [int(x) for x in seq.get("labels", [])[: int(args.steps)]]

    rows = []

    seq_payload = torch.load(args.seq_ckpt, map_location=device, weights_only=False)
    seq_model = Sparse64SequenceDecoder(d_model=int(args.d_model), nhead=int(args.nhead), nlayers=int(args.nlayers)).to(device).eval()
    seq_model.load_state_dict(seq_payload["model"] if isinstance(seq_payload, dict) and "model" in seq_payload else seq_payload, strict=False)

    with torch.inference_mode():
        def seq_full_forward():
            return seq_model(root, slots, prev_class, prev_row)

        def seq_encode_only():
            return seq_model.encode(root)

        cls, tok, _active = seq_model.encode(root)

        def seq_score_loop():
            hidden = None
            for i in range(slots.shape[1]):
                _tl, _tr, hidden = seq_model.score_step(cls, tok, slots[:, i, :], prev_class[:, i], prev_row[:, i], hidden)
            return hidden

        rows.append({"path": "SeqDistill full teacher-forced forward", "ms": elapsed_ms(device, seq_full_forward, args.repeats)})
        rows.append({"path": "SeqDistill root encode only", "ms": elapsed_ms(device, seq_encode_only, args.repeats)})
        rows.append({"path": f"SeqDistill recurrent score loop x{slots.shape[1]}", "ms": elapsed_ms(device, seq_score_loop, args.repeats)})

    exact_args = make_exact_args(make_reward_args(windows=100))
    exact_args.enable_x_band = False
    exact_args.single_sensor = True
    env_cfg = env_cfg_for(3.0, exact_args)
    env_cfg["enable_x_band"] = 0
    attn_model = load_action_attention_model(args.attention_state, str(device), "two_row_action_attention")
    base = PhysicalHeadPlanner(attn_model, "two_row_action_attention", env_cfg, policy_weight=1.0, q_weight=0.5)
    planner = CachedSingleSensorActionAttentionAR(
        base,
        max_steps=32,
        search_floor=0,
        search_cap_frac=1.0,
        env_cfg=env_cfg,
        action_coupler_top_k=int(args.top_k),
        sparse_residuals=bool(int(args.top_k) > 0),
    )

    with torch.inference_mode():
        cls_a, tok_a, root_selected, token_active = attn_model.backbone.encode_tokens(root)

        def attn_encode_only():
            return attn_model.backbone.encode_tokens(root)

        def attn_score_loop():
            selected = set()
            out = None
            for i in range(slots.shape[1]):
                out = planner._scores_from_encoded(cls_a, tok_a, root_selected, token_active, slots[:, i, :], selected)
                if i < len(labels):
                    row = max(0, int(labels[i]) // 2)
                    if row > 0:
                        selected.add(row)
            return out

        rows.append({"path": "ActionAttention root encode only", "ms": elapsed_ms(device, attn_encode_only, args.repeats)})
        rows.append({"path": f"ActionAttention Sparse{int(args.top_k)} score loop x{slots.shape[1]}", "ms": elapsed_ms(device, attn_score_loop, args.repeats)})

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.to_string(index=False), flush=True)
    print(str(out.resolve()), flush=True)


if __name__ == "__main__":
    main()
