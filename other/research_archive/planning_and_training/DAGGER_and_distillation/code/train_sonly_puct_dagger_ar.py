from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "CreateValid1" / "experiments" / "code" / "model_code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from current_sonly_exact_puct import ExactSOnlyPuctPlanner  # noqa: E402
from exact_env_mutual import MAXT, attach_env_obs, engine_env_cfg, env_cfg_for, shaped_step_reward, xs_decode_action, xs_s_search_action, xs_s_track_action  # noqa: E402
from final_radar_campaign import get_obs  # noqa: E402
from mutual_features import SLOT_DIM, slot_features, tokenize  # noqa: E402
from penalty_window_quota_learner_eval import make_exact_args  # noqa: E402
from repaired_campaign_tools import build_env, execute_first_valid_action  # noqa: E402
from realistic_reward_retrain import adapter  # noqa: E402
from single_sensor_ar_action_attention import load_action_attention_model  # noqa: E402
from train_action_attention_muzero_g import LatentG, infer_latent_d_model, training_action_pair  # noqa: E402


@dataclass
class DaggerWindow:
    x: np.ndarray
    slots: np.ndarray
    student_pairs: np.ndarray
    target_pairs: np.ndarray
    type_probs: np.ndarray
    row_probs: np.ndarray
    mask: np.ndarray
    meta: dict = field(default_factory=dict)
    rewards: np.ndarray | None = None
    returns: np.ndarray | None = None
    root_q: np.ndarray | None = None
    root_q_mask: np.ndarray | None = None
    state_tokens: np.ndarray | None = None
    next_state_tokens: np.ndarray | None = None


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_float_map(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split(":", 1)
        out[str(key).strip()] = float(value)
    return out


def action_from_row(row: int) -> int:
    return xs_s_search_action(MAXT) if int(row) <= 0 else xs_s_track_action(int(row), MAXT)


def window_service_meta(obs: dict, *, initial: int, rate: float, seed: int, window: int, spent: float, search_count: int, track_count: int) -> dict:
    active = np.asarray(obs.get("active_mask", np.zeros(MAXT, dtype=bool)), dtype=bool)
    deadlines = np.asarray(obs.get("t_deadline", np.full(MAXT, -1.0, dtype=np.float32)), dtype=np.float32)
    desired = np.asarray(obs.get("t_desired", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)
    n = min(MAXT, len(active), len(deadlines), len(desired))
    active = active[:n]
    deadlines = deadlines[:n]
    desired = desired[:n]
    tracked = active & (deadlines >= 0.0)
    active_n = int(active.sum())
    dropped_n = int(np.sum(active & (deadlines < 0.0)))
    active_delays = np.maximum(0.0, -desired[active]) if active_n > 0 else np.zeros(0, dtype=np.float32)
    actions = int(search_count) + int(track_count)
    return {
        "initial": int(initial),
        "rate": float(rate),
        "seed": int(seed),
        "window": int(window),
        "active_targets": active_n,
        "tracked_targets": int(np.sum(tracked)),
        "dropped_targets": dropped_n,
        "drop_pct_active": float(100.0 * dropped_n / active_n) if active_n > 0 else 0.0,
        "mean_delay_active": float(np.mean(active_delays)) if active_delays.size else 0.0,
        "spent_ms": float(spent),
        "search_count": int(search_count),
        "track_count": int(track_count),
        "valid_steps": int(actions),
        "search_fraction": float(search_count / max(1, actions)),
    }


def dynamic_slot_from_root(slot_root: torch.Tensor, elapsed: torch.Tensor, search_count: torch.Tensor, track_count: torch.Tensor, last_search: torch.Tensor) -> torch.Tensor:
    """Mirror ARSOnlyDynamicDecodeGraphModule slot construction."""
    return torch.stack(
        [
            elapsed / 200.0,
            search_count / 20.0,
            track_count / 100.0,
            last_search,
            slot_root[:, 4],
            slot_root[:, 5],
            slot_root[:, 6],
            slot_root[:, 7],
            slot_root[:, 8],
            slot_root[:, 9],
            slot_root[:, 10],
        ],
        dim=1,
    )


def row_duration_from_tokens(x: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """S-band duration used by the dynamic graph: search=10ms, track=t_dwell."""
    rows = rows.clamp(min=0, max=x.shape[1] - 1)
    batch = torch.arange(x.shape[0], device=x.device)
    dwell_ms = (x[batch, rows, 2] * 100.0).clamp(min=1.0)
    return torch.where(rows <= 0, torch.full_like(dwell_ms, 10.0), dwell_ms)


def discounted_returns(rewards: np.ndarray, mask: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        if float(mask[i]) > 0.0:
            running = float(rewards[i]) + float(gamma) * running
            out[i] = float(running)
        else:
            running = 0.0
    return out


def assign_across_episode_returns(windows: list[DaggerWindow], gamma: float) -> None:
    """Backfill return-to-go across 200 ms window boundaries.

    Search discoveries and frame coverage often pay off several windows after
    the action. Resetting G at each window removes exactly that credit.
    """
    episodes: dict[tuple[int, float, int], list[DaggerWindow]] = {}
    for window in windows:
        meta = dict(window.meta or {})
        key = (int(meta.get("initial", 0)), float(meta.get("rate", 0.0)), int(meta.get("seed", 0)))
        episodes.setdefault(key, []).append(window)
    for episode_windows in episodes.values():
        episode_windows.sort(key=lambda item: int((item.meta or {}).get("window", 0)))
        running = 0.0
        for window in reversed(episode_windows):
            rewards = np.asarray(window.rewards, dtype=np.float32)
            mask = np.asarray(window.mask, dtype=np.float32)
            returns = np.zeros_like(rewards, dtype=np.float32)
            for step in range(len(rewards) - 1, -1, -1):
                if float(mask[step]) <= 0.0:
                    continue
                running = float(rewards[step]) + float(gamma) * running
                returns[step] = float(running)
            window.returns = returns


def choose_student_action(model, g: LatentG, state, obs: dict, selected: set[int], remaining: float) -> int:
    h, ar_tok, prev, history, selected_t, token_active, step, slot, search_bias = state
    pos_step = g.seq_pos[min(int(step), int(g.seq_len) - 1)][None, :].expand(1, -1)
    a0 = g.action_emb(prev[:, 0])
    a1 = g.action_emb(prev[:, 1])
    slot_e = g.slot_proj(slot)
    inp = g.ar_input(torch.cat([a0, a1, slot_e, pos_step], dim=-1))
    if history is not None:
        inp = inp + g._ar_history_embedding(history)
    h_next = g.ar_cell(inp, h)
    type_logits, target_logits = g.ar_step_factor_logits(h_next, ar_tok, slot, pos_step, selected_t, token_active)
    search_logit = type_logits[0, 0, 0] + float(search_bias)
    track_logit = type_logits[0, 0, 1]
    target_col = target_logits[0, :, 0].clone()
    target_col[0] = -1e9
    for row in selected:
        if 0 < int(row) < target_col.shape[0]:
            target_col[int(row)] = -1e9
    active = np.asarray(obs.get("active_mask", np.zeros(MAXT, dtype=bool)), dtype=bool)
    deadline = np.asarray(obs.get("t_deadline", np.full(MAXT, -1.0, dtype=np.float32)), dtype=np.float32)
    dwell = np.asarray(obs.get("t_dwell", np.ones(MAXT, dtype=np.float32) * 10.0), dtype=np.float32)
    n = min(MAXT, len(active), len(deadline), len(dwell), target_col.shape[0] - 1)
    for row in range(1, n + 1):
        if (not bool(active[row - 1])) or float(deadline[row - 1]) < 0.0:
            target_col[row] = -1e9
        elif float(dwell[row - 1]) > float(remaining) and selected:
            target_col[row] = -1e9
    if n + 1 < target_col.shape[0]:
        target_col[n + 1 :] = -1e9
    best_track = int(torch.argmax(target_col).detach().cpu())
    has_track = bool(torch.isfinite(target_col[best_track]).detach().cpu()) and float(target_col[best_track].detach().cpu()) > -1e8
    if remaining < 10.0:
        row = best_track if has_track else 0
    elif (not has_track) or float(search_logit.detach().cpu()) >= float(track_logit.detach().cpu()):
        row = 0
    else:
        row = best_track
    return action_from_row(int(row)), h_next


def collect_on_policy(args, model, g: LatentG) -> list[DaggerWindow]:
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = False
    exact_args.single_sensor = True
    exact_args.serviced_pressure_improvement_reward_weight = float(args.serviced_pressure_improvement_reward_weight)
    adapt = adapter()
    rows: list[DaggerWindow] = []
    selfplay_rng = np.random.default_rng(int(args.seed))
    for initial in parse_ints(args.initials):
        for rate in parse_floats(args.rates):
            if len(rows) >= int(args.max_sequences):
                break
            env_cfg_cell = env_cfg_for(float(rate), exact_args)
            env_cfg_cell["enable_x_band"] = 0
            teacher = ExactSOnlyPuctPlanner(
                str(args.base_state),
                str(args.variant),
                env_cfg_cell,
                device=str(args.teacher_device),
                simulations=int(args.teacher_puct_simulations),
                expand_top_k=int(args.teacher_puct_expand_top_k),
                rollout_steps=int(args.teacher_puct_rollout_steps),
                c_puct=float(args.teacher_c_puct),
                discount=float(args.teacher_discount),
                select_mode=str(args.teacher_puct_select),
                policy_weight=float(args.teacher_policy_weight),
                q_weight=float(args.teacher_q_weight),
                search_bias=float(args.teacher_search_bias),
                terminal_service_weight=float(args.teacher_puct_terminal_service_weight),
                terminal_search_frame_weight=float(args.teacher_puct_terminal_search_frame_weight),
                rollout_windows=int(args.teacher_puct_rollout_windows),
                init_child_rollouts=bool(args.teacher_puct_init_child_rollouts),
                leaf_g_state=str(args.teacher_leaf_g_state),
                leaf_value_weight=float(args.teacher_leaf_value_weight),
                leaf_value_top_k=int(args.teacher_leaf_value_top_k),
                direct_child_value_weight=float(args.teacher_direct_child_value_weight),
                prior_uniform_mix=float(args.teacher_puct_prior_uniform_mix),
                root_dirichlet_alpha=float(args.teacher_puct_root_dirichlet_alpha),
                root_dirichlet_fraction=float(args.teacher_puct_root_dirichlet_fraction),
                progressive_widening_c=float(args.teacher_puct_progressive_widening_c),
                progressive_widening_alpha=float(args.teacher_puct_progressive_widening_alpha),
                stratify_root_types=bool(args.teacher_puct_stratify_root_types),
                matched_checkpoint=str(args.teacher_matched_checkpoint),
            )
            for seed in parse_ints(args.seeds):
                if len(rows) >= int(args.max_sequences):
                    break
                eng = build_env(None, int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg_cell))
                debt = 0.0
                cell_start = len(rows)
                for _window in range(int(args.windows)):
                    if len(rows) >= int(args.max_sequences) or bool(eng.term_buf[0]):
                        break
                    if int(args.max_sequences_per_cell) > 0 and len(rows) - cell_start >= int(args.max_sequences_per_cell):
                        break
                    collect_start = int(getattr(args, "collect_start_window", 0))
                    collect_stride = max(1, int(getattr(args, "collect_stride", 1)))
                    record_window = int(_window) >= collect_start and ((int(_window) - collect_start) % collect_stride == 0)
                    obs0 = attach_env_obs(get_obs(eng, debt), env_cfg_cell, True, True)
                    x0_np = tokenize(adapt, obs0, selected=set(), search_count=0).astype(np.float32)
                    slot_root_np = slot_features(obs0, 0.0, 0, 0, -1, 200.0).astype(np.float32)
                    x = torch.from_numpy(x0_np[None]).float().to(args.device)
                    with torch.inference_mode():
                        cls, tok, selected_root, token_active = model.backbone.encode_tokens(x)
                        s_valid = x[:, :, 10] > 0.5
                        s_valid[:, 0] = True
                        token_active = token_active & s_valid
                        h, ar_tok = g._project_state(cls, tok)
                    prev = torch.zeros((1, 2), dtype=torch.long, device=args.device)
                    prev[:, 1] = 1
                    history = None
                    if int(getattr(g, "ar_history_k", 0)) > 0:
                        history = prev[:, None, :].expand(-1, int(g.ar_history_k), -1).clone()
                    selected_t = torch.zeros_like(selected_root)
                    selected: set[int] = set()
                    slots = np.zeros((int(args.seq_len), SLOT_DIM), dtype=np.float32)
                    student_pairs = np.full((int(args.seq_len), 2), -1, dtype=np.int64)
                    target_pairs = np.full((int(args.seq_len), 2), -1, dtype=np.int64)
                    type_probs = np.zeros((int(args.seq_len), 2), dtype=np.float32)
                    row_probs = np.zeros((int(args.seq_len), MAXT + 1), dtype=np.float32)
                    mask = np.zeros((int(args.seq_len),), dtype=np.float32)
                    rewards_np = np.zeros((int(args.seq_len),), dtype=np.float32)
                    root_q_np = np.zeros((int(args.seq_len), MAXT + 1), dtype=np.float32)
                    root_q_mask_np = np.zeros((int(args.seq_len), MAXT + 1), dtype=np.float32)
                    # Float16 keeps matched transition supervision practical at 9-cell scale.
                    state_tokens_np = np.zeros((int(args.seq_len), *x0_np.shape), dtype=np.float16)
                    next_state_tokens_np = np.zeros((int(args.seq_len), *x0_np.shape), dtype=np.float16)
                    spent = 0.0
                    search_count = 0
                    track_count = 0
                    last = -1
                    proposed_search_count = 0
                    fallback_count = 0
                    for step in range(int(args.seq_len)):
                        remaining = 200.0 - float(spent)
                        if remaining <= 0.0 or bool(eng.term_buf[0]):
                            break
                        obs_now = attach_env_obs(get_obs(eng, debt), env_cfg_cell, True, True)
                        x_now_np = tokenize(
                            adapt,
                            obs_now,
                            selected=selected,
                            search_count=search_count,
                        ).astype(np.float32)
                        if bool(args.dynamic_slot_training):
                            slot_np = slot_root_np.copy()
                            slot_np[0] = float(spent) / 200.0
                            slot_np[1] = float(search_count) / 20.0
                            slot_np[2] = float(track_count) / 100.0
                            slot_np[3] = 1.0 if int(last) == 0 else 0.0
                        else:
                            slot_np = slot_features(obs_now, spent, search_count, track_count, last, 200.0).astype(np.float32)
                        slot = torch.from_numpy(slot_np[None]).float().to(args.device)
                        need_teacher = bool(record_window) or bool(getattr(args, "teacher_rollin_all_windows", False))
                        if not need_teacher:
                            teacher_action = None
                            teacher_rows = np.asarray([], dtype=np.int64)
                            teacher_probs = np.asarray([], dtype=np.float32)
                        elif str(args.teacher_target).lower() == "action":
                            teacher_action = int(teacher.choose_action(eng, debt, selected, spent, search_count, track_count, last, remaining))
                            teacher_rows = np.asarray([xs_decode_action(teacher_action, MAXT)[0]], dtype=np.int64)
                            teacher_probs = np.asarray([1.0], dtype=np.float32)
                        else:
                            teacher_action, teacher_rows, teacher_probs, _q_values, _visits = teacher.root_distribution(
                                eng,
                                debt,
                                selected,
                                spent,
                                search_count,
                                track_count,
                                last,
                                remaining,
                                target=str(args.teacher_target),
                                temperature=float(args.teacher_temperature),
                            )
                            sample_until = int(getattr(args, "teacher_sample_until_window", -1))
                            if bool(getattr(args, "teacher_sample_actions", False)) and (
                                sample_until < 0 or int(_window) < sample_until
                            ):
                                probs = np.asarray(teacher_probs, dtype=np.float64)
                                probs = probs / max(1.0e-12, float(probs.sum()))
                                sampled_row = int(selfplay_rng.choice(np.asarray(teacher_rows, dtype=np.int64), p=probs))
                                teacher_action = action_from_row(sampled_row)
                            teacher_action = int(teacher_action)
                        with torch.inference_mode():
                            student_action, h_next = choose_student_action(
                                model,
                                g,
                                (h, ar_tok, prev, history, selected_t, token_active, step, slot, float(args.search_bias)),
                                obs_now,
                                selected,
                                remaining,
                            )
                        proposed_row, _proposed_sensor = xs_decode_action(int(student_action), MAXT)
                        proposed_search_count += int(int(proposed_row) == 0)
                        if teacher_action is None:
                            action_order = [int(student_action)]
                        else:
                            action_order = [int(teacher_action), int(student_action)] if bool(args.execute_teacher_actions) else [int(student_action), int(teacher_action)]
                        reward, dt, executed = execute_first_valid_action(eng, action_order, remaining)
                        if executed is None or dt <= 0.0:
                            break
                        fallback_count += int(int(executed) != int(student_action))
                        student_pair = training_action_pair(int(executed), single_sensor=True)
                        if record_window and teacher_action is not None:
                            slots[step] = slot_np
                            state_tokens_np[step] = x_now_np.astype(np.float16)
                            target_pair = training_action_pair(int(teacher_action), single_sensor=True)
                            student_pairs[step] = student_pair
                            target_pairs[step] = target_pair
                            rprob = np.zeros((MAXT + 1,), dtype=np.float32)
                            for r, p in zip(teacher_rows, teacher_probs):
                                rr = int(max(0, min(MAXT, int(r))))
                                rprob[rr] += float(p)
                            total_prob = float(rprob.sum())
                            if total_prob <= 0.0:
                                rprob[0] = 1.0
                                total_prob = 1.0
                            rprob /= total_prob
                            row_probs[step] = rprob
                            type_probs[step, 0] = float(rprob[0])
                            type_probs[step, 1] = float(rprob[1:].sum())
                            if str(args.teacher_target).lower() != "action":
                                for r, q in zip(teacher_rows, _q_values):
                                    rr = int(max(0, min(MAXT, int(r))))
                                    if np.isfinite(float(q)):
                                        root_q_np[step, rr] = float(q)
                                        root_q_mask_np[step, rr] = 1.0
                            mask[step] = 1.0
                        row, _sensor = xs_decode_action(int(executed), MAXT)
                        next_debt = 0.0 if int(row) == 0 else debt + float(dt)
                        obs_after = attach_env_obs(get_obs(eng, next_debt), env_cfg_cell, True, True)
                        shaped = shaped_step_reward(float(reward), float(dt), obs_now, obs_after, env_cfg_cell, action=int(executed))
                        if record_window and teacher_action is not None:
                            rewards_np[step] = float(shaped)
                        spent += float(dt)
                        debt = float(next_debt)
                        if int(row) == 0:
                            search_count += 1
                        elif int(row) > 0:
                            selected.add(int(row))
                            selected_t[0, int(row)] = True
                            track_count += 1
                        last = int(row)
                        if record_window and teacher_action is not None:
                            obs_next = attach_env_obs(get_obs(eng, debt), env_cfg_cell, True, True)
                            next_state_tokens_np[step] = tokenize(
                                adapt,
                                obs_next,
                                selected=selected,
                                search_count=search_count,
                            ).astype(np.float16)
                        h = h_next
                        prev = torch.from_numpy(student_pair[None]).long().to(args.device)
                        prev[:, 1] = torch.where(prev[:, 1] < 0, torch.ones_like(prev[:, 1]), prev[:, 1])
                        prev = prev.clamp(min=0, max=g.action_emb.num_embeddings - 1)
                        if history is not None:
                            history = g._update_ar_history(history, prev)
                    if mask.any() and record_window:
                        obs_end = attach_env_obs(get_obs(eng, debt), env_cfg_cell, True, True)
                        meta = window_service_meta(
                            obs_end,
                            initial=int(initial),
                            rate=float(rate),
                            seed=int(seed),
                            window=int(_window),
                            spent=float(spent),
                            search_count=int(search_count),
                            track_count=int(track_count),
                        )
                        meta["student_proposed_search_fraction"] = float(proposed_search_count / max(1, int(mask.sum())))
                        meta["student_fallback_fraction"] = float(fallback_count / max(1, int(mask.sum())))
                        returns_np = discounted_returns(rewards_np, mask, float(args.action_value_return_discount))
                        rows.append(
                            DaggerWindow(
                                x0_np,
                                slots,
                                student_pairs,
                                target_pairs,
                                type_probs,
                                row_probs,
                                mask,
                                meta,
                                rewards_np,
                                returns_np,
                                root_q_np,
                                root_q_mask_np,
                                state_tokens_np,
                                next_state_tokens_np,
                            )
                        )
                print({"dagger_sequences": len(rows), "initial": initial, "rate": rate, "seed": seed}, flush=True)
    if str(getattr(args, "return_horizon", "episode")) == "episode":
        assign_across_episode_returns(rows, float(args.action_value_return_discount))
    return rows


def train_on_policy(args, model, g: LatentG, data: list[DaggerWindow]) -> None:
    train_value = float(getattr(args, "action_value_loss_weight", 0.0)) > 0.0 or float(getattr(args, "action_value_listwise_weight", 0.0)) > 0.0
    value_source = str(getattr(args, "action_value_target_source", "returns")).lower()
    value_only = bool(getattr(args, "train_action_value_only", False))
    for name, param in g.named_parameters():
        if value_only:
            param.requires_grad_(train_value and name.startswith("action_value_head"))
        else:
            param.requires_grad_(name.startswith("ar_") or name.startswith("seq_pos") or (train_value and name.startswith("action_value_head")))
    opt = torch.optim.AdamW([p for p in g.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(args.seed))
    load_search_targets = parse_float_map(str(getattr(args, "load_search_targets", "") or ""))
    value_targets_np = []
    if train_value:
        for window in data:
            if value_source == "root_q":
                q = getattr(window, "root_q", None)
                qm = getattr(window, "root_q_mask", None)
                if q is None or qm is None:
                    continue
                q_arr = np.asarray(q, dtype=np.float32)
                q_mask = np.asarray(qm, dtype=bool)
                vals = q_arr[q_mask]
                if vals.size:
                    value_targets_np.append(vals)
            else:
                ret = getattr(window, "returns", None)
                if ret is None:
                    continue
                m = np.asarray(window.mask, dtype=bool)
                r = np.asarray(ret, dtype=np.float32)
                if r.shape[0] >= m.shape[0] and bool(m.any()):
                    value_targets_np.append(r[: m.shape[0]][m])
        if value_targets_np:
            vals = np.concatenate(value_targets_np).astype(np.float32)
            args.action_value_target_mean = float(vals.mean())
            args.action_value_target_std = float(max(float(vals.std()), 1.0e-3))
            print(
                {
                    "action_value_returns": int(vals.size),
                    "action_value_target_mean": float(args.action_value_target_mean),
                    "action_value_target_std": float(args.action_value_target_std),
                },
                flush=True,
            )
        else:
            print({"action_value_loss_weight": float(args.action_value_loss_weight), "active": False, "reason": "no return targets in dataset"}, flush=True)
            train_value = False

    def calibration_target_for(window: DaggerWindow) -> float:
        meta = dict(getattr(window, "meta", {}) or {})
        if "target_search_frac" in meta:
            return float(meta["target_search_frac"])
        initial = meta.get("initial", None)
        if initial is not None:
            key = str(int(initial)) if isinstance(initial, (int, float, np.integer, np.floating)) else str(initial)
            if key in load_search_targets:
                return float(load_search_targets[key])
        return -1.0

    sample_probs = None
    stratified_groups = None
    stratified_group_probs = None
    if bool(getattr(args, "sample_by_meta_weight", False)):
        weights = np.asarray([float(getattr(w, "meta", {}).get("sample_weight", 1.0)) for w in data], dtype=np.float64)
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
        total_weight = float(weights.sum())
        if total_weight > 0.0:
            sample_probs = weights / total_weight
            print(
                {
                    "sample_by_meta_weight": True,
                    "min_weight": float(weights[weights > 0.0].min()) if np.any(weights > 0.0) else 0.0,
                    "mean_weight": float(weights.mean()),
                    "max_weight": float(weights.max()),
                },
                flush=True,
            )
        else:
            print({"sample_by_meta_weight": False, "reason": "no positive finite sample_weight values"}, flush=True)
    stratify_key = str(getattr(args, "stratify_by_meta", "") or "").strip()
    if stratify_key:
        groups: dict[str, list[int]] = {}
        for i, window in enumerate(data):
            meta = getattr(window, "meta", {}) or {}
            value = str(meta.get(stratify_key, "missing"))
            groups.setdefault(value, []).append(i)
        groups = {k: v for k, v in groups.items() if v}
        if len(groups) > 1:
            stratified_groups = []
            for key in sorted(groups.keys()):
                idxs = np.asarray(groups[key], dtype=np.int64)
                probs = None
                if sample_probs is not None:
                    local = sample_probs[idxs].astype(np.float64)
                    local_sum = float(local.sum())
                    if local_sum > 0.0:
                        probs = local / local_sum
                stratified_groups.append((key, idxs, probs))
            stratified_group_probs = np.ones((len(stratified_groups),), dtype=np.float64) / max(1, len(stratified_groups))
            print(
                {
                    "stratify_by_meta": stratify_key,
                    "groups": {k: int(len(v)) for k, v in groups.items()},
                    "uses_meta_weight_within_group": bool(sample_probs is not None),
                },
                flush=True,
            )
        else:
            print({"stratify_by_meta": stratify_key, "active": False, "reason": "fewer than two non-empty groups"}, flush=True)
    rollin = str(getattr(args, "rollin_policy", "teacher")).lower()
    if rollin not in {"teacher", "model"}:
        raise ValueError(f"unsupported --rollin-policy: {rollin}")
    for step in range(int(args.steps)):
        if stratified_groups is not None:
            batch_n = min(int(args.batch_size), len(data))
            group_choices = rng.choice(len(stratified_groups), size=batch_n, replace=True, p=stratified_group_probs)
            picked = []
            for group_i in group_choices:
                _key, idxs, probs = stratified_groups[int(group_i)]
                picked.append(int(rng.choice(idxs, replace=True, p=probs)))
            idx = np.asarray(picked, dtype=np.int64)
        elif sample_probs is None:
            idx = rng.integers(0, len(data), size=min(int(args.batch_size), len(data)))
        else:
            idx = rng.choice(len(data), size=min(int(args.batch_size), len(data)), replace=True, p=sample_probs)
        x = torch.from_numpy(np.stack([data[i].x for i in idx])).float().to(args.device)
        slots = torch.from_numpy(np.stack([data[i].slots for i in idx])).float().to(args.device)
        student_pairs = torch.from_numpy(np.stack([data[i].student_pairs for i in idx])).long().to(args.device)
        target_pairs = torch.from_numpy(np.stack([data[i].target_pairs for i in idx])).long().to(args.device)
        type_probs = torch.from_numpy(np.stack([data[i].type_probs for i in idx])).float().to(args.device)
        row_probs = torch.from_numpy(np.stack([data[i].row_probs for i in idx])).float().to(args.device)
        mask = torch.from_numpy(np.stack([data[i].mask for i in idx])).float().to(args.device)
        returns_np = []
        root_q_np_batch = []
        root_q_mask_np_batch = []
        for i in idx:
            ret = getattr(data[i], "returns", None)
            if ret is None:
                ret = np.zeros_like(data[i].mask, dtype=np.float32)
            returns_np.append(np.asarray(ret, dtype=np.float32))
            rq = getattr(data[i], "root_q", None)
            rqm = getattr(data[i], "root_q_mask", None)
            if rq is None or rqm is None:
                rq = np.zeros((data[i].mask.shape[0], MAXT + 1), dtype=np.float32)
                rqm = np.zeros((data[i].mask.shape[0], MAXT + 1), dtype=np.float32)
            root_q_np_batch.append(np.asarray(rq, dtype=np.float32))
            root_q_mask_np_batch.append(np.asarray(rqm, dtype=np.float32))
        returns_t = torch.from_numpy(np.stack(returns_np)).float().to(args.device)
        root_q_t = torch.from_numpy(np.stack(root_q_np_batch)).float().to(args.device)
        root_q_mask_t = torch.from_numpy(np.stack(root_q_mask_np_batch)).float().to(args.device)
        calib_targets = torch.tensor([calibration_target_for(data[i]) for i in idx], dtype=torch.float32, device=args.device)
        cls, tok, selected_root, token_active = model.backbone.encode_tokens(x)
        s_valid = x[:, :, 10] > 0.5
        s_valid[:, 0] = True
        token_active = token_active & s_valid
        h, ar_tok = g._project_state(cls, tok)
        bsz = x.shape[0]
        prev = torch.zeros((bsz, 2), dtype=torch.long, device=args.device)
        prev[:, 1] = 1
        history = None
        if int(getattr(g, "ar_history_k", 0)) > 0:
            history = prev[:, None, :].expand(-1, int(g.ar_history_k), -1).clone()
        selected_step = torch.zeros_like(selected_root)
        dyn_elapsed = torch.zeros((bsz,), dtype=torch.float32, device=args.device)
        dyn_search_count = torch.zeros((bsz,), dtype=torch.float32, device=args.device)
        dyn_track_count = torch.zeros((bsz,), dtype=torch.float32, device=args.device)
        dyn_last_search = torch.zeros((bsz,), dtype=torch.float32, device=args.device)
        losses = []
        type_acc_terms = []
        pred_search_terms = []
        target_search_terms = []
        search_mask_terms = []
        train_steps = min(int(args.seq_len), int(target_pairs.shape[1]), int(g.seq_len))
        if int(getattr(args, "prefix_loss_steps", 0)) > 0:
            train_steps = min(train_steps, int(args.prefix_loss_steps))
        for t in range(train_steps):
            valid = mask[:, t].bool() & (target_pairs[:, t, 0] >= 0)
            if bool(args.dynamic_slot_training):
                slot_t = dynamic_slot_from_root(slots[:, 0, :], dyn_elapsed, dyn_search_count, dyn_track_count, dyn_last_search)
            else:
                slot_t = slots[:, t, :]
            pos_t = g.seq_pos[t][None, :].expand(bsz, -1)
            a0 = g.action_emb(prev[:, 0])
            a1 = g.action_emb(prev[:, 1])
            slot_e = g.slot_proj(slot_t)
            inp = g.ar_input(torch.cat([a0, a1, slot_e, pos_t], dim=-1))
            if history is not None:
                inp = inp + g._ar_history_embedding(history)
            h = g.ar_cell(inp, h)
            type_logits, target_logits = g.ar_step_factor_logits(h, ar_tok, slot_t, pos_t, selected_step, token_active)
            if bool(valid.any()):
                rows = torch.div(target_pairs[:, t, 0], 2, rounding_mode="floor").clamp(min=0, max=MAXT)
                type_target = (rows > 0).long()
                if not value_only:
                    use_selected_type = str(getattr(args, "type_target_mode", "selected")) == "selected"
                    if str(args.teacher_target).lower() == "action" or use_selected_type:
                        losses.append(float(args.type_weight) * F.cross_entropy(type_logits[:, 0, :][valid], type_target[valid]))
                    else:
                        log_type = F.log_softmax(type_logits[:, 0, :][valid], dim=-1)
                        soft_type = type_probs[:, t, :][valid].clamp_min(0.0)
                        soft_type = soft_type / soft_type.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                        losses.append(float(args.type_weight) * (-(soft_type * log_type).sum(dim=-1)).mean())
                pred_type = torch.argmax(type_logits[:, 0, :], dim=-1)
                type_acc_terms.append((pred_type[valid] == type_target[valid]).float().mean())
                type_prob = F.softmax(type_logits[:, 0, :], dim=-1)
                pred_search = type_prob[:, 0]
                if str(args.teacher_target).lower() == "action" or use_selected_type:
                    target_search = (type_target == 0).float()
                else:
                    target_search = type_probs[:, t, 0].clamp(min=0.0, max=1.0)
                pred_search_terms.append(pred_search)
                target_search_terms.append(target_search)
                search_mask_terms.append(valid.float())
                track = valid & (rows > 0)
                if (not value_only) and bool(track.any()):
                    logits = target_logits[:, 1:, 0][track]
                    target = rows[track] - 1
                    target_logit = logits.gather(1, target[:, None]).squeeze(1)
                    valid_target = torch.isfinite(target_logit) & (target_logit > -1e8)
                    if bool(valid_target.any()):
                        if str(args.teacher_target).lower() == "action":
                            losses.append(float(args.target_weight) * F.cross_entropy(logits[valid_target], target[valid_target]))
                        else:
                            soft_rows = row_probs[:, t, 1:][track][valid_target].clamp_min(0.0)
                            valid_logits = logits[valid_target]
                            finite_rows = torch.isfinite(valid_logits) & (valid_logits > -1e8)
                            soft_rows = soft_rows * finite_rows.float()
                            fallback_target = target[valid_target]
                            empty = soft_rows.sum(dim=-1) <= 1e-6
                            if bool(empty.any()):
                                soft_rows[empty] = 0.0
                                soft_rows[empty, fallback_target[empty].clamp(min=0, max=soft_rows.shape[1] - 1)] = 1.0
                            soft_rows = soft_rows / soft_rows.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                            log_rows = F.log_softmax(valid_logits, dim=-1)
                            losses.append(float(args.target_weight) * (-(soft_rows * log_rows).sum(dim=-1)).mean())
                if train_value and float(args.action_value_loss_weight) > 0.0 and value_source != "root_q":
                    action_pair = target_pairs[:, t, :].clone()
                    action_pair[:, 1] = torch.where(action_pair[:, 1] < 0, torch.ones_like(action_pair[:, 1]), action_pair[:, 1])
                    action_pair = action_pair.clamp(min=0, max=g.action_emb.num_embeddings - 1)
                    pred_v = g.predict_action_value(cls, tok, slot_t, action_pair)
                    mean = float(getattr(args, "action_value_target_mean", 0.0))
                    std = max(1.0e-3, float(getattr(args, "action_value_target_std", 1.0)))
                    target_v = (returns_t[:, t] - mean) / std
                    losses.append(float(args.action_value_loss_weight) * F.smooth_l1_loss(pred_v[valid], target_v[valid]))
                elif train_value and value_source == "root_q":
                    qmask = root_q_mask_t[:, t, :].bool() & valid[:, None]
                    if bool(qmask.any()) and float(args.action_value_loss_weight) > 0.0:
                        batch_idx, row_idx = torch.where(qmask)
                        action_pair = torch.stack([row_idx.clamp(min=0, max=MAXT) * 2, torch.ones_like(row_idx)], dim=1)
                        cls_b = cls[batch_idx]
                        tok_b = tok[batch_idx]
                        slot_b = slot_t[batch_idx]
                        pred_v = g.predict_action_value(cls_b, tok_b, slot_b, action_pair)
                        mean = float(getattr(args, "action_value_target_mean", 0.0))
                        std = max(1.0e-3, float(getattr(args, "action_value_target_std", 1.0)))
                        target_v = (root_q_t[:, t, :][qmask] - mean) / std
                        losses.append(float(args.action_value_loss_weight) * F.smooth_l1_loss(pred_v, target_v))
                    if bool(qmask.any()) and float(args.action_value_listwise_weight) > 0.0:
                        tau = max(1.0e-4, float(args.action_value_listwise_temperature))
                        list_losses = []
                        for b in torch.where(valid)[0]:
                            row_idx = torch.where(root_q_mask_t[b, t, :].bool())[0]
                            if row_idx.numel() < 2:
                                continue
                            action_pair = torch.stack([row_idx.clamp(min=0, max=MAXT) * 2, torch.ones_like(row_idx)], dim=1)
                            cls_b = cls[b : b + 1].expand(row_idx.numel(), -1)
                            tok_b = tok[b : b + 1].expand(row_idx.numel(), -1, -1)
                            slot_b = slot_t[b : b + 1].expand(row_idx.numel(), -1)
                            pred_v = g.predict_action_value(cls_b, tok_b, slot_b, action_pair)
                            q_vals = root_q_t[b, t, row_idx]
                            target_p = F.softmax((q_vals - q_vals.max()) / tau, dim=0)
                            list_losses.append(-(target_p * F.log_softmax(pred_v, dim=0)).sum())
                        if list_losses:
                            losses.append(float(args.action_value_listwise_weight) * torch.stack(list_losses).mean())
            if rollin == "model":
                with torch.no_grad():
                    model_rows = torch.div(student_pairs[:, t, 0], 2, rounding_mode="floor").clamp(min=0, max=MAXT)
                    type_choice = torch.argmax(type_logits[:, 0, :], dim=-1)
                    target_choice = torch.argmax(target_logits[:, 1:, 0], dim=-1) + 1
                    model_rows = torch.where(type_choice == 0, torch.zeros_like(target_choice), target_choice)
                    cur = torch.stack([model_rows.clamp(min=0, max=MAXT) * 2, torch.ones_like(model_rows)], dim=1)
                    cur = torch.where(valid[:, None], cur, student_pairs[:, t, :].clone())
            else:
                cur = student_pairs[:, t, :].clone()
            cur[:, 1] = torch.where(cur[:, 1] < 0, torch.ones_like(cur[:, 1]), cur[:, 1])
            cur = cur.clamp(min=0, max=g.action_emb.num_embeddings - 1)
            chosen_rows = torch.div(cur, 2, rounding_mode="floor").clamp(min=0, max=MAXT)
            active = (mask[:, t:t + 1].bool()) & (chosen_rows > 0)
            if bool(active.any()):
                batch_idx = torch.arange(bsz, device=args.device)[:, None].expand_as(chosen_rows)
                selected_step[batch_idx[active], chosen_rows[active]] = True
            if bool(args.dynamic_slot_training):
                main_rows = chosen_rows[:, 0].clamp(min=0, max=MAXT)
                step_active = mask[:, t].bool()
                dt = row_duration_from_tokens(x, main_rows)
                dyn_elapsed = dyn_elapsed + torch.where(step_active, dt, torch.zeros_like(dt))
                dyn_search_count = dyn_search_count + (step_active & (main_rows == 0)).to(dyn_search_count.dtype)
                dyn_track_count = dyn_track_count + (step_active & (main_rows > 0)).to(dyn_track_count.dtype)
                dyn_last_search = (step_active & (main_rows == 0)).to(dyn_last_search.dtype)
            prev = cur
            if history is not None:
                history = g._update_ar_history(history, prev)
        if float(args.window_type_count_weight) > 0.0 and pred_search_terms:
            pred_search_all = torch.stack(pred_search_terms, dim=1)
            target_search_all = torch.stack(target_search_terms, dim=1)
            search_mask_all = torch.stack(search_mask_terms, dim=1)
            denom = search_mask_all.sum(dim=1).clamp_min(1.0)
            pred_search_frac = (pred_search_all * search_mask_all).sum(dim=1) / denom
            target_search_frac = (target_search_all * search_mask_all).sum(dim=1) / denom
            if not value_only:
                losses.append(float(args.window_type_count_weight) * F.mse_loss(pred_search_frac, target_search_frac))
            if (not value_only) and float(args.load_type_count_weight) > 0.0:
                calib_mask = calib_targets >= 0.0
                if bool(calib_mask.any()):
                    target = calib_targets[calib_mask].clamp(min=0.0, max=1.0)
                    losses.append(float(args.load_type_count_weight) * F.mse_loss(pred_search_frac[calib_mask], target))
        loss = sum(losses) / max(1, len(losses))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in g.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % 50 == 0 or step == int(args.steps) - 1:
            acc = torch.stack(type_acc_terms).mean() if type_acc_terms else loss.new_tensor(0.0)
            print({"step": step, "loss": float(loss.detach().cpu()), "type_acc": float(acc.detach().cpu())}, flush=True)


def save_dataset(path: str | Path, data: list[DaggerWindow], args: argparse.Namespace) -> None:
    if not str(path):
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    service = [getattr(w, "meta", {}) for w in data if getattr(w, "meta", None)]
    service_meta = {}
    if service:
        for key in ("drop_pct_active", "tracked_targets", "mean_delay_active", "search_fraction", "valid_steps"):
            vals = [float(m[key]) for m in service if key in m]
            if vals:
                service_meta[f"mean_{key}"] = float(np.mean(vals))
        service_meta["service_metric_windows"] = int(len(service))
    return_vals = []
    for window in data:
        ret = getattr(window, "returns", None)
        if ret is None:
            continue
        valid = np.asarray(window.mask, dtype=bool)
        vals = np.asarray(ret, dtype=np.float32)[: valid.shape[0]][valid]
        if vals.size:
            return_vals.append(vals)
    if return_vals:
        flat = np.concatenate(return_vals)
        service_meta["return_target_count"] = int(flat.size)
        service_meta["return_target_mean"] = float(flat.mean())
        service_meta["return_target_std"] = float(flat.std())
    root_q_vals = []
    for window in data:
        rq = getattr(window, "root_q", None)
        rqm = getattr(window, "root_q_mask", None)
        if rq is None or rqm is None:
            continue
        vals = np.asarray(rq, dtype=np.float32)[np.asarray(rqm, dtype=bool)]
        if vals.size:
            root_q_vals.append(vals)
    if root_q_vals:
        flat_q = np.concatenate(root_q_vals)
        service_meta["root_q_target_count"] = int(flat_q.size)
        service_meta["root_q_target_mean"] = float(flat_q.mean())
        service_meta["root_q_target_std"] = float(flat_q.std())
    torch.save(
        {
            "windows": data,
            "meta": {
                "initials": str(args.initials),
                "rates": str(args.rates),
                "seeds": str(args.seeds),
                "windows": int(args.windows),
                "collect_start_window": int(args.collect_start_window),
                "collect_stride": int(args.collect_stride),
                "teacher_target": str(args.teacher_target),
                "teacher_puct_simulations": int(args.teacher_puct_simulations),
                "teacher_puct_expand_top_k": int(args.teacher_puct_expand_top_k),
                "teacher_puct_rollout_steps": int(args.teacher_puct_rollout_steps),
                "teacher_puct_rollout_windows": int(args.teacher_puct_rollout_windows),
                **service_meta,
            },
        },
        path,
    )
    print({"saved_dataset": str(path), "windows": len(data)}, flush=True)


def load_dataset(path: str | Path) -> list[DaggerWindow]:
    required = ("x", "slots", "student_pairs", "target_pairs", "type_probs", "row_probs", "mask")
    combined: list[DaggerWindow] = []
    paths = [item.strip() for item in str(path).split(",") if item.strip()]
    for item in paths:
        ckpt = torch.load(item, map_location="cpu", weights_only=False)
        data = ckpt["windows"] if isinstance(ckpt, dict) and "windows" in ckpt else ckpt
        if not isinstance(data, list) or not all(all(hasattr(x, k) for k in required) for x in data):
            raise RuntimeError(f"invalid DAgger dataset: {item}")
        combined.extend(data)
        print({"loaded_dataset": item, "windows": len(data)}, flush=True)
    print({"combined_datasets": len(paths), "combined_windows": len(combined)}, flush=True)
    return combined


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-state", required=True)
    ap.add_argument("--init-g-state", default="", help="Optional AR/g checkpoint. Empty means initialize the AR decoder from scratch.")
    ap.add_argument("--g-d-model", type=int, default=48, help="Latent/AR model width used when --init-g-state is empty.")
    ap.add_argument("--variant", default="two_row_action_attention")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--teacher-device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--initials", default="20,40,60")
    ap.add_argument("--rates", default="2,3,4")
    ap.add_argument("--seeds", default="916")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--max-sequences", type=int, default=180)
    ap.add_argument("--max-sequences-per-cell", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=40)
    ap.add_argument("--ar-history-k", type=int, default=-1, help="Use -1 to infer from checkpoint; old rootseq checkpoints use 0.")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--search-bias", type=float, default=0.0)
    ap.add_argument("--type-weight", type=float, default=1.0)
    ap.add_argument("--target-weight", type=float, default=1.2)
    ap.add_argument(
        "--type-target-mode",
        choices=["selected", "marginal"],
        default="selected",
        help=(
            "Train the factorized search/track head from the planner-selected action type, or from "
            "summed visit mass. Selected avoids track mass growing merely because many track targets exist."
        ),
    )
    ap.add_argument("--action-value-loss-weight", type=float, default=0.0, help="Train action_value_head on discounted return-to-go targets collected from PUCT-executed windows.")
    ap.add_argument("--action-value-target-source", choices=["returns", "root_q"], default="returns", help="Supervise action_value_head from executed discounted returns or PUCT root child Q values.")
    ap.add_argument("--action-value-return-discount", type=float, default=0.997, help="Discount for per-step return-to-go targets saved in new DAgger windows.")
    ap.add_argument("--return-horizon", choices=["episode", "window"], default="episode", help="Backfill return-to-go across the full episode or reset it at each 200 ms window.")
    ap.add_argument("--action-value-listwise-weight", type=float, default=0.0, help="For root_q targets, train action_value_head to rank sibling PUCT candidates by a softmax over root Q.")
    ap.add_argument("--action-value-listwise-temperature", type=float, default=1.0, help="Temperature for root-Q listwise action-value targets.")
    ap.add_argument("--train-action-value-only", action="store_true", help="Freeze the AR policy and train only action_value_head.")
    ap.add_argument("--window-type-count-weight", type=float, default=0.0, help="Auxiliary loss matching total search-vs-track mass over the decoded window.")
    ap.add_argument("--load-type-count-weight", type=float, default=0.0, help="Auxiliary loss matching decoded search fraction to metadata/load-specific search targets.")
    ap.add_argument("--load-search-targets", default="", help="Optional load search targets such as '20:0.24,40:0.14,60:0.13'. Used only when --load-type-count-weight > 0 and metadata target_search_frac is absent.")
    ap.add_argument("--rollin-policy", choices=["teacher", "model"], default="teacher", help="Action history used while unrolling the AR decoder during training.")
    ap.add_argument("--dynamic-slot-training", action="store_true", help="Build AR slot features from decoded elapsed/search/track state, matching deployment dynamic decode.")
    ap.add_argument(
        "--prefix-loss-steps",
        type=int,
        default=0,
        help="Train only the executable prefix used before replanning; zero trains every valid sequence step.",
    )
    ap.add_argument("--sample-by-meta-weight", action="store_true", help="Sample DAgger windows according to DaggerWindow.meta['sample_weight'] when present.")
    ap.add_argument("--stratify-by-meta", default="", help="When set, sample batches uniformly across DaggerWindow.meta groups, e.g. initial.")
    ap.add_argument("--teacher-puct-simulations", type=int, default=4)
    ap.add_argument("--teacher-puct-expand-top-k", type=int, default=8)
    ap.add_argument("--teacher-puct-rollout-steps", type=int, default=2)
    ap.add_argument("--teacher-puct-rollout-windows", type=int, default=1)
    ap.add_argument("--teacher-puct-init-child-rollouts", action="store_true")
    ap.add_argument("--teacher-puct-prior-uniform-mix", type=float, default=0.05)
    ap.add_argument("--teacher-puct-root-dirichlet-alpha", type=float, default=0.3)
    ap.add_argument("--teacher-puct-root-dirichlet-fraction", type=float, default=0.25)
    ap.add_argument("--teacher-puct-progressive-widening-c", type=float, default=2.0)
    ap.add_argument("--teacher-puct-progressive-widening-alpha", type=float, default=0.5)
    ap.add_argument("--teacher-puct-stratify-root-types", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--teacher-matched-checkpoint",
        default="",
        help="Matched AR/batch checkpoint supplying policy priors and action-Q values to PUCT.",
    )
    ap.add_argument("--teacher-leaf-g-state", default="")
    ap.add_argument("--teacher-leaf-value-weight", type=float, default=0.0)
    ap.add_argument("--teacher-leaf-value-top-k", type=int, default=16)
    ap.add_argument("--teacher-direct-child-value-weight", type=float, default=0.0)
    ap.add_argument("--execute-teacher-actions", action="store_true", help="Collect supervised bootstrap data on teacher-executed trajectories instead of DAgger student trajectories.")
    ap.add_argument(
        "--teacher-rollin-all-windows",
        action="store_true",
        help="Use the PUCT teacher for unrecorded warmup/stride windows so sampled states remain on the teacher distribution.",
    )
    ap.add_argument("--collect-start-window", type=int, default=0, help="Execute earlier windows but only add training rows from this window onward.")
    ap.add_argument("--collect-stride", type=int, default=1, help="Record every Nth eligible window after --collect-start-window.")
    ap.add_argument("--teacher-target", choices=["action", "visits", "q_softmax", "prior"], default="action")
    ap.add_argument("--teacher-temperature", type=float, default=0.5)
    ap.add_argument("--teacher-sample-actions", action="store_true", help="Execute actions sampled from the PUCT root distribution during self-play collection.")
    ap.add_argument("--teacher-sample-until-window", type=int, default=-1, help="Sample through this episode window; -1 samples for the full training episode.")
    ap.add_argument("--teacher-c-puct", type=float, default=1.5)
    ap.add_argument("--teacher-discount", type=float, default=0.98)
    ap.add_argument("--teacher-puct-select", default="q")
    ap.add_argument("--teacher-policy-weight", type=float, default=1.0)
    ap.add_argument("--teacher-q-weight", type=float, default=0.5)
    ap.add_argument("--teacher-search-bias", type=float, default=0.0)
    ap.add_argument("--teacher-puct-terminal-service-weight", type=float, default=0.25)
    ap.add_argument("--teacher-puct-terminal-search-frame-weight", type=float, default=8.0)
    ap.add_argument("--env-mode", default="pufferlib_service")
    ap.add_argument("--search-frame-overdue-weight", type=float, default=0.5)
    ap.add_argument("--search-frame-drop-penalty", type=float, default=8.0)
    ap.add_argument("--search-frame-state-penalty-weight", type=float, default=2.0)
    ap.add_argument("--search-frame-delta-reward-weight", type=float, default=5.0)
    ap.add_argument("--service-pressure-delta-reward-weight", type=float, default=0.30)
    ap.add_argument("--serviced-pressure-improvement-reward-weight", type=float, default=0.15)
    ap.add_argument("--discovered-target-reward", type=float, default=0.08)
    ap.add_argument("--tracked-count-delta-reward-weight", type=float, default=0.0)
    ap.add_argument("--tracked-target-ms-reward-weight", type=float, default=0.0)
    ap.add_argument("--save-dataset", default="", help="Optional .pt path for collected PUCT/DAgger windows.")
    ap.add_argument("--load-dataset", default="", help="Optional .pt path to reuse previously collected PUCT/DAgger windows.")
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = load_action_attention_model(Path(args.base_state), args.device, args.variant)
    state = None
    if str(args.init_g_state):
        ckpt = torch.load(str(args.init_g_state), map_location=args.device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    d_model = infer_latent_d_model(state, int(args.g_d_model)) if state is not None else int(args.g_d_model)
    history_k = int(args.ar_history_k)
    if history_k < 0:
        history_k = 4 if state is None or any(str(k).startswith("ar_history_proj.") for k in state.keys()) else 0
    g = LatentG(d_model=d_model, seq_len=int(args.seq_len), ar_history_k=history_k).to(args.device)
    if state is not None:
        missing, unexpected = g.load_state_dict(state, strict=False)
    else:
        missing, unexpected = [], []
    print({"init_g_state": str(args.init_g_state) or "<scratch>", "ar_history_k": history_k, "d_model": d_model, "missing": missing, "unexpected": unexpected}, flush=True)
    data = load_dataset(args.load_dataset) if str(args.load_dataset) else collect_on_policy(args, model, g.eval())
    if not data:
        raise RuntimeError("no DAgger windows collected")
    if str(args.save_dataset) and not str(args.load_dataset):
        save_dataset(args.save_dataset, data, args)
    valid = np.concatenate([d.mask.astype(bool) for d in data])
    targets = np.concatenate([d.target_pairs[:, 0] for d in data])[valid]
    students = np.concatenate([d.student_pairs[:, 0] for d in data])[valid]
    soft_type = np.concatenate([d.type_probs for d in data], axis=0)[valid]
    print(
        {
            "dagger_windows": len(data),
            "mean_len": float(valid.sum() / max(1, len(data))),
            "teacher_search_frac": float((targets == 0).mean()),
            "teacher_soft_search_prob": float(soft_type[:, 0].mean()),
            "student_search_frac": float((students == 0).mean()),
            "teacher_student_same": float((targets == students).mean()),
        },
        flush=True,
    )
    train_on_policy(args, model, g.train(), data)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": g.state_dict(),
            "dagger_windows": len(data),
            "action_value_target_mean": float(getattr(args, "action_value_target_mean", 0.0)),
            "action_value_target_std": float(getattr(args, "action_value_target_std", 1.0)),
            "action_value_target_source": str(args.action_value_target_source),
            "action_value_return_discount": float(args.action_value_return_discount),
            "action_value_listwise_weight": float(args.action_value_listwise_weight),
            "action_value_listwise_temperature": float(args.action_value_listwise_temperature),
            "train_action_value_only": bool(args.train_action_value_only),
        },
        args.out,
    )
    print({"saved": str(args.out)}, flush=True)


if __name__ == "__main__":
    main()
