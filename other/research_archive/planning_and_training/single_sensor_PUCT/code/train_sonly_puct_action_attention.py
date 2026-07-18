from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "CreateValid1" / "experiments" / "code" / "model_code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import train_sonly_puct_dagger_ar as dagger  # noqa: E402
from exact_env_mutual import MAXT  # noqa: E402
from single_sensor_ar_action_attention import load_action_attention_model  # noqa: E402


def load_dataset(path: str | Path):
    import __main__

    # Older DAgger datasets pickle DaggerWindow as __main__.DaggerWindow.
    __main__.DaggerWindow = dagger.DaggerWindow
    combined = []
    for item in [part.strip() for part in str(path).split(",") if part.strip()]:
        ckpt = torch.load(item, map_location="cpu", weights_only=False)
        data = ckpt["windows"] if isinstance(ckpt, dict) and "windows" in ckpt else ckpt
        if not isinstance(data, list):
            raise RuntimeError(f"invalid dataset: {item}")
        combined.extend(data)
        print({"loaded_dataset": item, "windows": len(data)}, flush=True)
    print({"combined_windows": len(combined)}, flush=True)
    return combined


def factorized_policy_loss(
    scores: torch.Tensor,
    rows: torch.Tensor,
    valid_samples: torch.Tensor,
    type_weight: float,
    target_weight: float,
) -> torch.Tensor:
    selected_scores = scores[valid_samples]
    selected_rows = rows[valid_samples]
    type_logits = torch.stack(
        [selected_scores[:, 0], torch.logsumexp(selected_scores[:, 1:], dim=1)],
        dim=1,
    )
    type_targets = (selected_rows > 0).long()
    loss = float(type_weight) * F.cross_entropy(type_logits, type_targets)
    track = selected_rows > 0
    if bool(track.any()):
        loss = loss + float(target_weight) * F.cross_entropy(
            selected_scores[track, 1:],
            selected_rows[track] - 1,
        )
    return loss


def batch_tensors(data, idx, device: str):
    x = torch.from_numpy(np.stack([data[i].x for i in idx])).float().to(device)
    slots = torch.from_numpy(np.stack([data[i].slots for i in idx])).float().to(device)
    pairs = torch.from_numpy(np.stack([data[i].target_pairs for i in idx])).long().to(device)
    mask = torch.from_numpy(np.stack([data[i].mask for i in idx])).float().to(device)
    root_q = []
    root_q_mask = []
    for i in idx:
        rq = getattr(data[i], "root_q", None)
        rqm = getattr(data[i], "root_q_mask", None)
        if rq is None or rqm is None:
            rq = np.zeros((mask.shape[1], MAXT + 1), dtype=np.float32)
            rqm = np.zeros((mask.shape[1], MAXT + 1), dtype=np.float32)
        root_q.append(np.asarray(rq, dtype=np.float32))
        root_q_mask.append(np.asarray(rqm, dtype=np.float32))
    root_q_t = torch.from_numpy(np.stack(root_q)).float().to(device)
    root_q_mask_t = torch.from_numpy(np.stack(root_q_mask)).float().to(device)
    return x, slots, pairs, mask, root_q_t, root_q_mask_t


def infer_selected_before_step(pairs: torch.Tensor, mask: torch.Tensor, step: int) -> torch.Tensor:
    selected = torch.zeros((pairs.shape[0], MAXT + 1), dtype=torch.bool, device=pairs.device)
    if step <= 0:
        return selected
    rows_prev = torch.div(pairs[:, :step, 0].clamp(min=0), 2, rounding_mode="floor").clamp(min=0, max=MAXT)
    active_prev = (mask[:, :step] > 0.0) & (rows_prev > 0)
    if bool(active_prev.any()):
        bidx = torch.arange(pairs.shape[0], device=pairs.device)[:, None].expand_as(rows_prev)
        selected[bidx[active_prev], rows_prev[active_prev]] = True
    return selected


def dynamic_slots_from_actions(x: torch.Tensor, root_slot: torch.Tensor, pairs: torch.Tensor, mask: torch.Tensor, step: int) -> torch.Tensor:
    bsz = x.shape[0]
    elapsed = torch.zeros((bsz,), dtype=torch.float32, device=x.device)
    search_count = torch.zeros((bsz,), dtype=torch.float32, device=x.device)
    track_count = torch.zeros((bsz,), dtype=torch.float32, device=x.device)
    last_search = torch.zeros((bsz,), dtype=torch.float32, device=x.device)
    for t in range(step):
        rows = torch.div(pairs[:, t, 0].clamp(min=0), 2, rounding_mode="floor").clamp(min=0, max=MAXT)
        valid = mask[:, t] > 0.0
        dt = dagger.row_duration_from_tokens(x, rows)
        elapsed = elapsed + torch.where(valid, dt, torch.zeros_like(dt))
        search_count = search_count + (valid & (rows == 0)).to(search_count.dtype)
        track_count = track_count + (valid & (rows > 0)).to(track_count.dtype)
        last_search = (valid & (rows == 0)).to(last_search.dtype)
    return dagger.dynamic_slot_from_root(root_slot, elapsed, search_count, track_count, last_search)


def forward_sonly_scores(model, x: torch.Tensor, slot: torch.Tensor, selected_step: torch.Tensor):
    cls_out, tok_out, root_selected, token_active = model.backbone.encode_tokens(x)
    slot_emb = model.backbone.slot_proj(slot)
    bsz, rows, _ = tok_out.shape

    sensor = model.sensor_embed[None, :, :].expand(bsz, -1, -1)
    cls_s0 = cls_out[:, None, :]
    slot_s0 = slot_emb[:, None, :]

    # Build the same coupled sensor context used by CudaGraphSingleSensorActionAttentionAR.
    cls_s = cls_out[:, None, :].expand(-1, 2, -1)
    slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
    sensor_state = model.sensor_state_proj(torch.cat([cls_s, slot_s, sensor], dim=-1))
    coupled_sensor = model.sensor_coupler(sensor_state)
    coupled_s0 = coupled_sensor[:, 0:1, :]

    type_ctx = torch.cat([cls_s0, slot_s0, coupled_s0], dim=-1)
    type_logits = model.type_head(type_ctx)[:, 0, :]
    type_q = model.type_q_head(type_ctx)[:, 0, :]

    cls_t = cls_out[:, None, :].expand(-1, rows, -1)
    slot_tgt = slot_emb[:, None, :].expand(-1, rows, -1)
    sensor_tgt = coupled_sensor[:, None, 0, :].expand(-1, rows, -1)
    target_ctx = torch.cat([tok_out, cls_t, slot_tgt, sensor_tgt], dim=-1)
    target_logits = model.target_head(target_ctx).squeeze(-1)
    target_q = model.target_q_head(target_ctx).squeeze(-1)

    selected_t = root_selected | selected_step
    track_mask = token_active & ~selected_t
    track_mask[:, 0] = False
    row_is_search = torch.arange(rows, device=x.device)[None, :] == 0
    valid = track_mask | row_is_search

    base_scores = x.new_full((bsz, rows), -1.0e9)
    base_q = x.new_zeros((bsz, rows))
    base_scores[:, 0] = type_logits[:, 0]
    base_q[:, 0] = type_q[:, 0]
    base_scores[:, 1:] = (type_logits[:, None, 1] + target_logits)[:, 1:]
    base_q[:, 1:] = (type_q[:, None, 1] + target_q)[:, 1:]

    action_ctx = model.action_proj(target_ctx)
    action_ctx = model.action_coupler(action_ctx, src_key_padding_mask=~valid)
    residual = model.action_policy_residual(action_ctx).squeeze(-1)
    q_residual = model.action_q_residual(action_ctx).squeeze(-1)
    scores = (base_scores + residual).masked_fill(~valid, -1.0e9)
    q = (base_q + q_residual).masked_fill(~valid, 0.0)
    return scores, q, valid


def freeze_for_finetune(model, mode: str) -> None:
    if mode == "all":
        for p in model.parameters():
            p.requires_grad_(True)
        return
    trainable_keys = {
        "heads": (
            "sensor_state_proj",
            "sensor_coupler",
            "type_head",
            "type_q_head",
            "target_head",
            "target_q_head",
            "action_proj",
            "action_coupler",
            "action_policy_residual",
            "action_q_residual",
        ),
        "residuals": ("action_policy_residual", "action_q_residual"),
    }[mode]
    for name, p in model.named_parameters():
        p.requires_grad_(any(k in name for k in trainable_keys))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-state", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="two_row_action_attention_qpolicy_factored_loss")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5.0e-5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--freeze", choices=["all", "heads", "residuals"], default="heads")
    ap.add_argument("--seq-len", type=int, default=40)
    ap.add_argument("--dynamic-slots", action="store_true")
    ap.add_argument("--policy-loss-weight", type=float, default=1.0)
    ap.add_argument("--factorized-policy-loss", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--type-weight", type=float, default=1.0)
    ap.add_argument("--target-weight", type=float, default=1.0)
    ap.add_argument("--stratify-by-meta", default="initial")
    ap.add_argument("--q-loss-weight", type=float, default=0.1)
    ap.add_argument("--q-listwise-weight", type=float, default=0.1)
    ap.add_argument("--q-listwise-temperature", type=float, default=1.0)
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    data = load_dataset(args.dataset)
    model = load_action_attention_model(Path(args.base_state), args.device, args.variant)
    model.train()
    freeze_for_finetune(model, str(args.freeze))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=1.0e-4)

    rng = np.random.default_rng(int(args.seed))
    groups = None
    stratify_key = str(args.stratify_by_meta).strip()
    if stratify_key:
        grouped: dict[str, list[int]] = {}
        for i, window in enumerate(data):
            value = getattr(window, "meta", {}).get(stratify_key)
            if value is not None:
                grouped.setdefault(str(value), []).append(i)
        if len(grouped) >= 2:
            groups = [np.asarray(indices, dtype=np.int64) for indices in grouped.values()]
            print({"stratify_by_meta": stratify_key, "groups": {key: len(value) for key, value in grouped.items()}}, flush=True)
    for step in range(int(args.steps)):
        if groups is None:
            idx = rng.integers(0, len(data), size=int(args.batch_size))
        else:
            picked = []
            for group_i in rng.integers(0, len(groups), size=int(args.batch_size)):
                group = groups[int(group_i)]
                picked.append(int(group[rng.integers(0, len(group))]))
            idx = np.asarray(picked, dtype=np.int64)
        x, slots, pairs, mask, root_q, root_q_mask = batch_tensors(data, idx, args.device)
        losses = []
        accs = []
        valid_steps = min(int(args.seq_len), pairs.shape[1])
        for t in range(valid_steps):
            valid_t = (mask[:, t] > 0.0) & (pairs[:, t, 0] >= 0)
            if not bool(valid_t.any()):
                continue
            selected_step = infer_selected_before_step(pairs, mask, t)
            slot_t = dynamic_slots_from_actions(x, slots[:, 0, :], pairs, mask, t) if bool(args.dynamic_slots) else slots[:, t, :]
            scores, q, valid_rows = forward_sonly_scores(model, x, slot_t, selected_step)
            rows = torch.div(pairs[:, t, 0].clamp(min=0), 2, rounding_mode="floor").clamp(min=0, max=MAXT)
            row_ok = valid_t & valid_rows.gather(1, rows[:, None]).squeeze(1) & (scores.gather(1, rows[:, None]).squeeze(1) > -1.0e8)
            if not bool(row_ok.any()):
                continue
            if float(args.policy_loss_weight) > 0.0:
                if bool(args.factorized_policy_loss):
                    policy_loss = factorized_policy_loss(
                        scores,
                        rows,
                        row_ok,
                        type_weight=float(args.type_weight),
                        target_weight=float(args.target_weight),
                    )
                else:
                    policy_loss = F.cross_entropy(scores[row_ok], rows[row_ok])
                losses.append(float(args.policy_loss_weight) * policy_loss)
            pred = torch.argmax(scores, dim=1)
            accs.append((pred[row_ok] == rows[row_ok]).float().mean())
            q_targets_t = root_q[:, t, :]
            qmask = (
                root_q_mask[:, t, :].bool()
                & row_ok[:, None]
                & valid_rows
                & torch.isfinite(q_targets_t)
                & (q_targets_t > -1.0e8)
                & (q_targets_t < 1.0e8)
            )
            if bool(qmask.any()) and float(args.q_loss_weight) > 0.0:
                pred_q = q[qmask]
                target_q = q_targets_t[qmask]
                losses.append(float(args.q_loss_weight) * F.smooth_l1_loss(pred_q, target_q))
            if bool(qmask.any()) and float(args.q_listwise_weight) > 0.0:
                list_losses = []
                tau = max(1.0e-4, float(args.q_listwise_temperature))
                for b in torch.where(valid_t)[0]:
                    rows_b = torch.where(qmask[b])[0]
                    if rows_b.numel() < 2:
                        continue
                    q_vals = q_targets_t[b, rows_b]
                    target_p = F.softmax((q_vals - q_vals.max()) / tau, dim=0)
                    list_losses.append(-(target_p * F.log_softmax(q[b, rows_b], dim=0)).sum())
                if list_losses:
                    losses.append(float(args.q_listwise_weight) * torch.stack(list_losses).mean())
        if not losses:
            continue
        loss = sum(losses) / max(1, len(losses))
        if (not torch.isfinite(loss)) or float(loss.detach().cpu()) > 1.0e4:
            raise RuntimeError(f"unstable loss at step {step}: {float(loss.detach().cpu())}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0 or step == int(args.steps) - 1:
            acc = torch.stack(accs).mean() if accs else loss.new_tensor(0.0)
            print({"step": int(step), "loss": float(loss.detach().cpu()), "row_acc": float(acc.detach().cpu())}, flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "variant": str(args.variant),
            "dataset": str(args.dataset),
            "steps": int(args.steps),
            "freeze": str(args.freeze),
            "dynamic_slots": bool(args.dynamic_slots),
        },
        args.out,
    )
    print({"saved": str(args.out)}, flush=True)


if __name__ == "__main__":
    main()
