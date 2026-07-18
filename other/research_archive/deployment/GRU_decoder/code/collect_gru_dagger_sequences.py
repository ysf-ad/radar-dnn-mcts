from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from eval_action_attention_muzero_g import _DummyPlanner, execute_plan_until_budget_joint_shaped
from exact_env_mutual import MAXT, build_env, engine_env_cfg, env_cfg_for, get_obs, xs_decode_action, xs_s_search_action, xs_s_track_action
from fast_gru_window_decoder_experiment import GRUWindowDecoder, action_to_class
from foundation_mcts_fair_eval import parse_floats, parse_ints, physical_candidates
from mutual_features import slot_features, tokenize
from penalty_window_quota_learner_eval import make_exact_args
from single_sensor_ar_action_attention import CachedSingleSensorActionAttentionAR, load_action_attention_model
from two_sensor_physical_head_eval import PhysicalHeadPlanner


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TEACHER = ROOT / "CreateValid1" / "results" / "single_sensor_fair_exact_action_attention_train_two_row_action_attention_qpolicy_factored_loss.pt"


def load_gru(path: Path) -> GRUWindowDecoder:
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = state.get("config", {}) if isinstance(state, dict) else {}
    action_mixer_mode = str(cfg.get("action_mixer_mode", "none"))
    if "action_mixer_mode" not in cfg and "head_mode" not in cfg:
        action_mixer_mode = "legacy_flat"
    model = GRUWindowDecoder(
        d_model=int(cfg.get("d_model", 48)),
        nhead=int(cfg.get("nhead", 4)),
        enc_layers=int(cfg.get("enc_layers", 2)),
        action_mixer_layers=int(cfg.get("action_mixer_layers", 0)),
        action_mixer_mode=action_mixer_mode,
        head_mode=str(cfg.get("head_mode", "flat")),
    ).eval()
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state, strict=True)
    return model


def choose_row_from_scores(scores: np.ndarray, obs: dict, selected: set[int], elapsed: float) -> int:
    for base in selected:
        if 0 <= int(base) < scores.shape[0]:
            scores[int(base), :] = -1e9
    remaining = max(0.0, 200.0 - float(elapsed))
    if remaining < 10.0:
        scores[0, :] = -1e9
    dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
    if dwell.size:
        n = min(MAXT, dwell.size, scores.shape[0] - 1)
        bad = np.nonzero(dwell[:n] > remaining)[0] + 1
        scores[bad, :] = -1e9
    best_row = 0
    best_score = -np.inf
    for action in physical_candidates(obs, top_k=MAXT):
        base, sensor = xs_decode_action(int(action), MAXT)
        if int(sensor or 0) != 0:
            continue
        base = int(max(0, base))
        if base > 0 and base in selected:
            continue
        val = float(scores[base, 0])
        if val > best_score:
            best_score = val
            best_row = base
    return int(best_row)


def collect_one(student: GRUWindowDecoder, teacher, env_cfg: dict, initial: int, rate: float, seed: int, windows: int, max_seq: int) -> tuple[list[dict], list[dict]]:
    from realistic_reward_retrain import adapter

    eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
    eng.reset(seed=int(seed))
    debt = 0.0
    adapt = adapter()
    sequences: list[dict] = []
    rows: list[dict] = []
    try:
        for window in range(int(windows)):
            if bool(eng.term_buf[0]):
                break
            obs = get_obs(eng, debt)
            root_x = tokenize(adapt, obs, selected=set(), search_count=0).astype(np.float32)
            slots: list[np.ndarray] = []
            labels: list[int] = []
            prev_classes: list[int] = []
            prev_rows: list[float] = []
            selected: set[int] = set()
            elapsed = 0.0
            search_count = 0
            track_count = 0
            last = -1
            prev_action: int | None = None
            student_plan: list[int] = []
            dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
            device = next(student.parameters()).device
            with torch.inference_mode():
                root_t = torch.from_numpy(root_x).float().unsqueeze(0).to(device)
                cls_out, tok_out = student.encode_root(root_t)
                h = student.init(cls_out)
            for _ in range(int(max_seq)):
                if elapsed >= 200.0:
                    break
                # Label the student's current prefix with the strong AR/PQ teacher.
                teacher_row = teacher._choose_row(obs, selected, elapsed, search_count, track_count, last)
                teacher_action = xs_s_search_action(MAXT) if teacher_row <= 0 else xs_s_track_action(int(teacher_row), MAXT)
                slots.append(slot_features(obs, elapsed, search_count, track_count, last, 200.0).astype(np.float32))
                prev_classes.append(0 if prev_action is None else action_to_class(int(prev_action)))
                prev_base = 0 if prev_action is None else max(0, int(xs_decode_action(int(prev_action), MAXT)[0]))
                prev_rows.append(float(prev_base) / float(MAXT))
                labels.append(int(max(0, teacher_row)) * 2)

                # Advance with the student action to expose off-teacher states.
                prev_cls = 0 if prev_action is None else action_to_class(int(prev_action))
                prev_base = 0 if prev_action is None else max(0, int(xs_decode_action(int(prev_action), MAXT)[0]))
                with torch.inference_mode():
                    inp = torch.cat(
                        [
                            student.slot_proj(torch.from_numpy(slots[-1]).float().to(device).unsqueeze(0)),
                            student.prev_class(torch.tensor([prev_cls], dtype=torch.long, device=device)),
                            student.prev_row(torch.tensor([[float(prev_base) / float(MAXT)]], dtype=torch.float32, device=device)),
                        ],
                        dim=-1,
                    )
                    h = student.gru(inp, h)
                    scores = student.score_from_hidden(h, cls_out, tok_out)[0].detach().cpu().numpy()
                row = choose_row_from_scores(scores, obs, selected, elapsed)
                action = xs_s_search_action(MAXT) if row <= 0 else xs_s_track_action(row, MAXT)
                dt = 10.0 if row <= 0 else float(max(1.0, dwell[row - 1] if row - 1 < len(dwell) else 5.0))
                if elapsed + dt > 200.0 and student_plan:
                    break
                student_plan.append(action)
                if row <= 0:
                    search_count += 1
                    last = 0
                else:
                    selected.add(int(row))
                    track_count += 1
                    last = int(row)
                elapsed += dt
                prev_action = action
            reward, spent, debt, executed, searches, _ = execute_plan_until_budget_joint_shaped(
                eng, student_plan, 200.0, debt, "student", int(seed), int(window), env_cfg
            )
            if labels:
                sequences.append(
                    {
                        "root_x": root_x,
                        "slots": np.stack(slots, axis=0).astype(np.float32),
                        "labels": np.asarray(labels, dtype=np.int64),
                        "prev_classes": np.asarray(prev_classes, dtype=np.int64),
                        "prev_rows": np.asarray(prev_rows, dtype=np.float32),
                        "initial": int(initial),
                        "rate": float(rate),
                        "seed": int(seed),
                        "window": int(window),
                        "teacher_reward": float(reward),
                    }
                )
            rows.append(
                {
                    "initial": int(initial),
                    "rate": float(rate),
                    "seed": int(seed),
                    "window": int(window),
                    "student_reward": float(reward),
                    "student_search_fraction": float(searches / max(1, executed)),
                    "executed": int(executed),
                    "spent_ms": float(spent),
                    "seq_len": int(len(labels)),
                }
            )
    finally:
        eng.close()
    return sequences, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--teacher-state", default=str(DEFAULT_TEACHER))
    ap.add_argument("--out", default=str(ROOT / "CreateValid1" / "results" / "gru_dagger_sequences.pt"))
    ap.add_argument("--summary-out", default="")
    ap.add_argument("--initials", default="20,40,60")
    ap.add_argument("--rates", default="2,3,4")
    ap.add_argument("--seeds", default="916")
    ap.add_argument("--windows", type=int, default=100)
    ap.add_argument("--max-seq", type=int, default=24)
    ap.add_argument("--env-mode", default="pufferlib_service")
    ap.add_argument("--track-update-reward", type=float, default=0.30)
    ap.add_argument("--track-loss-penalty", type=float, default=8.0)
    ap.add_argument("--search-frame-overdue-weight", type=float, default=0.5)
    ap.add_argument("--search-frame-drop-penalty", type=float, default=8.0)
    ap.add_argument("--search-frame-state-penalty-weight", type=float, default=2.0)
    ap.add_argument("--search-frame-delta-reward-weight", type=float, default=5.0)
    ap.add_argument("--service-pressure-delta-reward-weight", type=float, default=0.30)
    ap.add_argument("--serviced-pressure-improvement-reward-weight", type=float, default=0.15)
    ap.add_argument("--discovered-target-reward", type=float, default=0.08)
    args = ap.parse_args()
    torch.set_num_threads(1)
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = False
    exact_args.single_sensor = True
    student = load_gru(Path(args.student_model))
    teacher_model = load_action_attention_model(Path(args.teacher_state), "cpu", "two_row_action_attention")
    all_sequences: list[dict] = []
    all_rows: list[dict] = []
    for initial in parse_ints(args.initials):
        for rate in parse_floats(args.rates):
            env_cfg = env_cfg_for(float(rate), exact_args)
            env_cfg["enable_x_band"] = 0
            base = PhysicalHeadPlanner(teacher_model, "two_row_action_attention", env_cfg, policy_weight=1.0, q_weight=0.5, search_score_bias=0.0)
            teacher = CachedSingleSensorActionAttentionAR(base, max_steps=int(args.max_seq), search_floor=0, search_cap_frac=1.0, env_cfg=env_cfg)
            for seed in parse_ints(args.seeds):
                seqs, rows = collect_one(student, teacher, env_cfg, int(initial), float(rate), int(seed), int(args.windows), int(args.max_seq))
                all_sequences.extend(seqs)
                all_rows.extend(rows)
                print({"initial": int(initial), "rate": float(rate), "seed": int(seed), "sequences": len(seqs)}, flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"plan_sequences": all_sequences, "rows": all_rows}, out)
    if args.summary_out:
        import pandas as pd

        pd.DataFrame(all_rows).to_csv(args.summary_out, index=False)
    print({"saved": str(out), "plan_sequences": len(all_sequences)}, flush=True)


if __name__ == "__main__":
    main()
