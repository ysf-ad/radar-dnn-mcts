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

from current_sonly_exact_puct import ExactSOnlyPuctPlanner, parse_floats, parse_ints  # noqa: E402
from exact_env_mutual import (  # noqa: E402
    MAXT,
    attach_env_obs,
    engine_env_cfg,
    env_cfg_for,
    get_obs,
    xs_decode_action,
)
from mutual_features import SLOT_DIM, slot_features, tokenize  # noqa: E402
from penalty_window_quota_learner_eval import make_exact_args  # noqa: E402
from repaired_campaign_tools import build_env, execute_first_valid_action  # noqa: E402
from realistic_reward_retrain import adapter  # noqa: E402
from single_sensor_ar_action_attention import load_action_attention_model  # noqa: E402
from train_sonly_puct_action_attention import forward_sonly_scores, freeze_for_finetune  # noqa: E402


@dataclass
class StepExample:
    x: np.ndarray
    slot: np.ndarray
    selected: np.ndarray
    target_row: int
    teacher_rows: np.ndarray
    teacher_probs: np.ndarray
    root_q: np.ndarray
    root_q_mask: np.ndarray
    meta: dict = field(default_factory=dict)


def action_is_search(action: int) -> bool:
    row, _sensor = xs_decode_action(int(action), MAXT)
    return int(row) == 0


def make_teacher(args, env_cfg_cell: dict) -> ExactSOnlyPuctPlanner:
    return ExactSOnlyPuctPlanner(
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
    )


def collect_steps(args) -> list[StepExample]:
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = False
    exact_args.single_sensor = True
    adapt = adapter()
    out: list[StepExample] = []
    for initial in parse_ints(args.initials):
        for rate in parse_floats(args.rates):
            env_cfg_cell = env_cfg_for(float(rate), exact_args)
            env_cfg_cell["enable_x_band"] = 0
            teacher = make_teacher(args, env_cfg_cell)
            for seed in parse_ints(args.seeds):
                if len(out) >= int(args.max_examples):
                    return out
                eng = build_env(None, int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg_cell))
                debt = 0.0
                for window in range(int(args.windows)):
                    if bool(eng.term_buf[0]) or len(out) >= int(args.max_examples):
                        break
                    elapsed = 0.0
                    search_count = 0
                    track_count = 0
                    last = -1
                    selected: set[int] = set()
                    record = int(window) >= int(args.collect_start_window)
                    record = record and ((int(window) - int(args.collect_start_window)) % max(1, int(args.collect_stride)) == 0)
                    for step in range(int(args.max_steps_per_window)):
                        remaining = 200.0 - float(elapsed)
                        if remaining <= 0.0 or bool(eng.term_buf[0]):
                            break
                        obs_now = attach_env_obs(get_obs(eng, debt), env_cfg_cell, True, True)
                        if record:
                            x_np = tokenize(adapt, obs_now, selected=set(), search_count=search_count).astype(np.float32)
                            slot_np = slot_features(obs_now, elapsed, search_count, track_count, last, 200.0).astype(np.float32)
                            selected_np = np.zeros((MAXT + 1,), dtype=np.bool_)
                            for row in selected:
                                if 1 <= int(row) <= MAXT:
                                    selected_np[int(row)] = True
                            teacher_action, teacher_rows, teacher_probs, q_values, _visits = teacher.root_distribution(
                                eng,
                                debt,
                                selected,
                                elapsed,
                                search_count,
                                track_count,
                                last,
                                remaining,
                                target=str(args.teacher_target),
                                temperature=float(args.teacher_temperature),
                            )
                            row, _sensor = xs_decode_action(int(teacher_action), MAXT)
                            q_full = np.zeros((MAXT + 1,), dtype=np.float32)
                            q_mask = np.zeros((MAXT + 1,), dtype=np.float32)
                            for r, qv in zip(teacher_rows, q_values):
                                if 0 <= int(r) <= MAXT:
                                    q_full[int(r)] = float(qv)
                                    q_mask[int(r)] = 1.0
                            out.append(
                                StepExample(
                                    x=x_np,
                                    slot=slot_np,
                                    selected=selected_np,
                                    target_row=int(row),
                                    teacher_rows=np.asarray(teacher_rows, dtype=np.int64),
                                    teacher_probs=np.asarray(teacher_probs, dtype=np.float32),
                                    root_q=q_full,
                                    root_q_mask=q_mask,
                                    meta={
                                        "initial": int(initial),
                                        "rate": float(rate),
                                        "seed": int(seed),
                                        "window": int(window),
                                        "step": int(step),
                                        "elapsed": float(elapsed),
                                    },
                                )
                            )
                        else:
                            teacher_action = teacher.choose_action(eng, debt, selected, elapsed, search_count, track_count, last, remaining)

                        reward, dt, executed = execute_first_valid_action(eng, [int(teacher_action)], remaining)
                        if executed is None or dt <= 0.0:
                            break
                        exec_row, _sensor = xs_decode_action(int(executed), MAXT)
                        elapsed += float(dt)
                        if int(exec_row) == 0:
                            debt = 0.0
                            search_count += 1
                            last = 0
                        else:
                            debt += float(dt)
                            track_count += 1
                            selected.add(int(exec_row))
                            last = int(exec_row)
    return out


def save_dataset(args) -> None:
    examples = collect_steps(args)
    out_path = Path(args.dataset_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"steps": examples, "args": vars(args)}, str(out_path))
    print(f"saved {len(examples)} post-action examples -> {out_path}")


def load_steps(path: str | Path) -> list[StepExample]:
    import __main__

    __main__.StepExample = StepExample
    try:
        from train_sonly_puct_dagger_ar import DaggerWindow

        __main__.DaggerWindow = DaggerWindow
    except ImportError:
        DaggerWindow = None
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "windows" in ckpt:
        data = []
        for window in ckpt["windows"]:
            valid_steps = np.flatnonzero(np.asarray(window.mask).astype(bool))
            selected_rows: set[int] = set()
            for step in valid_steps:
                flat_action = int(window.target_pairs[int(step), 0])
                row = int(np.clip(flat_action // 2, 0, MAXT))
                type_prob = np.asarray(window.type_probs[int(step)], dtype=np.float32)
                row_prob = np.asarray(window.row_probs[int(step)], dtype=np.float32).copy()
                combined = np.zeros((MAXT + 1,), dtype=np.float32)
                combined[0] = float(type_prob[0])
                track_mass = row_prob[1 : MAXT + 1]
                track_total = float(track_mass.sum())
                if track_total > 0.0:
                    combined[1:] = float(type_prob[1]) * track_mass / track_total
                if float(combined.sum()) <= 0.0:
                    combined[row] = 1.0
                teacher_rows = np.flatnonzero(combined > 0.0).astype(np.int64)
                selected = np.zeros((MAXT + 1,), dtype=np.bool_)
                for selected_row in selected_rows:
                    selected[int(selected_row)] = True
                state_tokens = window.state_tokens if window.state_tokens is not None else None
                x = state_tokens[int(step)] if state_tokens is not None else window.x
                data.append(
                    StepExample(
                        x=np.asarray(x, dtype=np.float32),
                        slot=np.asarray(window.slots[int(step)], dtype=np.float32),
                        selected=selected,
                        target_row=row,
                        teacher_rows=teacher_rows,
                        teacher_probs=combined[teacher_rows],
                        root_q=np.asarray(window.root_q[int(step)], dtype=np.float32),
                        root_q_mask=np.asarray(window.root_q_mask[int(step)], dtype=np.float32),
                        meta={**dict(window.meta or {}), "step": int(step), "source": "matched_puct_window"},
                    )
                )
                if row > 0:
                    selected_rows.add(row)
        return data
    if isinstance(ckpt, dict) and {"tokens", "slot", "base_action", "ret"}.issubset(ckpt):
        # Matched MuZero/PUCT transition view.  Converting it here lets the
        # re-encode policy and Q heads train on exactly the same action and
        # long-horizon return targets as the latent model.
        data = []
        tokens = np.asarray(ckpt["tokens"], dtype=np.float32)
        slots = np.asarray(ckpt["slot"], dtype=np.float32)
        rows = np.asarray(ckpt["base_action"], dtype=np.int64)
        returns = np.asarray(ckpt["ret"], dtype=np.float32)
        for index, (x, slot, row, ret) in enumerate(zip(tokens, slots, rows, returns)):
            row = int(np.clip(row, 0, MAXT))
            root_q = np.zeros((MAXT + 1,), dtype=np.float32)
            root_q_mask = np.zeros((MAXT + 1,), dtype=np.float32)
            root_q[row] = float(ret)
            root_q_mask[row] = 1.0
            data.append(
                StepExample(
                    x=x,
                    slot=slot,
                    selected=np.zeros((MAXT + 1,), dtype=np.bool_),
                    target_row=row,
                    teacher_rows=np.asarray([row], dtype=np.int64),
                    teacher_probs=np.asarray([1.0], dtype=np.float32),
                    root_q=root_q,
                    root_q_mask=root_q_mask,
                    meta={"source": "matched_puct_transition", "index": int(index)},
                )
            )
        return data
    data = ckpt["steps"] if isinstance(ckpt, dict) and "steps" in ckpt else ckpt
    if not isinstance(data, list):
        raise RuntimeError(f"invalid step dataset: {path}")
    return data


def make_batch(data: list[StepExample], idx: np.ndarray, device: str):
    x = torch.from_numpy(np.stack([data[int(i)].x for i in idx])).float().to(device)
    slot = torch.from_numpy(np.stack([data[int(i)].slot for i in idx])).float().to(device)
    selected = torch.from_numpy(np.stack([data[int(i)].selected for i in idx])).bool().to(device)
    target = torch.tensor([int(data[int(i)].target_row) for i in idx], dtype=torch.long, device=device)
    dist = torch.zeros((len(idx), MAXT + 1), dtype=torch.float32, device=device)
    q = torch.from_numpy(np.stack([data[int(i)].root_q for i in idx])).float().to(device)
    qmask = torch.from_numpy(np.stack([data[int(i)].root_q_mask for i in idx])).bool().to(device)
    for b, i in enumerate(idx):
        ex = data[int(i)]
        for row, prob in zip(ex.teacher_rows, ex.teacher_probs):
            if 0 <= int(row) <= MAXT:
                dist[b, int(row)] += float(prob)
    return x, slot, selected, target, dist, q, qmask


def train_model(args) -> None:
    data = load_steps(args.dataset)
    model = load_action_attention_model(Path(args.base_state), args.device, args.variant)
    model.train()
    freeze_for_finetune(model, str(args.freeze))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=1.0e-4)
    rng = np.random.default_rng(int(args.seed))

    for step in range(int(args.steps)):
        idx = rng.integers(0, len(data), size=int(args.batch_size))
        x, slot, selected, target, dist, qtarget, qmask = make_batch(data, idx, args.device)
        scores, qpred, valid = forward_sonly_scores(model, x, slot, selected)
        row_ok = valid.gather(1, target[:, None]).squeeze(1)
        losses = []
        if bool(row_ok.any()) and float(args.hard_target_weight) > 0.0:
            losses.append(float(args.hard_target_weight) * F.cross_entropy(scores[row_ok], target[row_ok]))
        if bool(row_ok.any()) and float(args.policy_loss_weight) > 0.0:
            logp = F.log_softmax(scores[row_ok], dim=1)
            dist_ok = dist[row_ok]
            dist_ok = dist_ok / dist_ok.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
            losses.append(float(args.policy_loss_weight) * (-(dist_ok * logp).sum(dim=1).mean()))
        q_ok = qmask & valid
        if bool(q_ok.any()) and float(args.q_loss_weight) > 0.0:
            losses.append(float(args.q_loss_weight) * F.smooth_l1_loss(qpred[q_ok], qtarget[q_ok]))
        if float(args.q_rank_loss_weight) > 0.0:
            rank_rows = q_ok.sum(dim=1) > 1
            if bool(rank_rows.any()):
                masked_target = qtarget[rank_rows].masked_fill(~q_ok[rank_rows], -torch.inf)
                best_rows = torch.argmax(masked_target, dim=1)
                masked_pred = qpred[rank_rows].masked_fill(~q_ok[rank_rows], -1.0e9)
                losses.append(float(args.q_rank_loss_weight) * F.cross_entropy(masked_pred, best_rows))
        if not losses:
            continue
        loss = sum(losses)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % max(1, int(args.log_every)) == 0 or step + 1 == int(args.steps):
            with torch.no_grad():
                pred = torch.argmax(scores, dim=1)
                acc = (pred[row_ok] == target[row_ok]).float().mean().item() if bool(row_ok.any()) else 0.0
                search = (pred == 0).float().mean().item()
            print(f"step={step:05d} loss={float(loss.detach().cpu()):.4f} acc={acc:.3f} pred_search={search:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path))
    print(f"saved state dict -> {out_path}")


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--base-state", required=True)
    ap.add_argument("--variant", default="two_row_action_attention_qpolicy_factored_loss")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--initials", default="20,40,60")
    ap.add_argument("--rates", default="2,3,4")
    ap.add_argument("--seeds", default="916,917,918")
    ap.add_argument("--windows", type=int, default=24)
    ap.add_argument("--max-steps-per-window", type=int, default=20)
    ap.add_argument("--collect-start-window", type=int, default=0)
    ap.add_argument("--collect-stride", type=int, default=1)
    ap.add_argument("--max-examples", type=int, default=6000)
    ap.add_argument("--env-mode", default="pufferlib_basic")
    ap.add_argument("--teacher-device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--teacher-puct-simulations", type=int, default=8)
    ap.add_argument("--teacher-puct-expand-top-k", type=int, default=8)
    ap.add_argument("--teacher-puct-rollout-steps", type=int, default=2)
    ap.add_argument("--teacher-c-puct", type=float, default=1.5)
    ap.add_argument("--teacher-discount", type=float, default=0.99)
    ap.add_argument("--teacher-puct-select", default="q")
    ap.add_argument("--teacher-policy-weight", type=float, default=1.0)
    ap.add_argument("--teacher-q-weight", type=float, default=0.5)
    ap.add_argument("--teacher-search-bias", type=float, default=0.0)
    ap.add_argument("--teacher-target", choices=["visits", "q_softmax", "prior", "action"], default="q_softmax")
    ap.add_argument("--teacher-temperature", type=float, default=0.75)
    ap.add_argument("--teacher-puct-terminal-service-weight", type=float, default=0.0)
    ap.add_argument("--teacher-puct-terminal-search-frame-weight", type=float, default=0.0)
    ap.add_argument("--teacher-puct-rollout-windows", type=int, default=1)
    ap.add_argument("--teacher-puct-init-child-rollouts", action="store_true")
    ap.add_argument("--teacher-leaf-g-state", default="")
    ap.add_argument("--teacher-leaf-value-weight", type=float, default=0.0)
    ap.add_argument("--teacher-leaf-value-top-k", type=int, default=16)
    ap.add_argument("--teacher-direct-child-value-weight", type=float, default=0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    collect = sub.add_parser("collect")
    add_common_args(collect)
    collect.add_argument("--dataset-out", required=True)

    train = sub.add_parser("train")
    train.add_argument("--base-state", required=True)
    train.add_argument("--dataset", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--variant", default="two_row_action_attention_qpolicy_factored_loss")
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--steps", type=int, default=1000)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=5.0e-5)
    train.add_argument("--seed", type=int, default=123)
    train.add_argument("--freeze", choices=["all", "heads", "residuals"], default="heads")
    train.add_argument("--policy-loss-weight", type=float, default=1.0)
    train.add_argument("--hard-target-weight", type=float, default=1.0)
    train.add_argument("--q-loss-weight", type=float, default=0.05)
    train.add_argument("--q-rank-loss-weight", type=float, default=0.0)
    train.add_argument("--log-every", type=int, default=50)

    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.cmd == "collect":
        save_dataset(args)
    elif args.cmd == "train":
        train_model(args)


if __name__ == "__main__":
    main()
