from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "CreateValid1" / "experiments" / "code" / "model_code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from compare_action_heads_smoke import batch_tensors, usable_targets
from two_sensor_physical_head_eval import make_physical_model


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--teacher-state", required=True)
    ap.add_argument("--teacher-variant", default="two_row_pair_action_attention")
    ap.add_argument("--student-init-state", default="")
    ap.add_argument("--student-variant", default="two_row_pure_pair_action_attention")
    ap.add_argument("--out-state", required=True)
    ap.add_argument("--train-steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model-seed", type=int, default=123)
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--positive-top-k", type=int, default=16)
    ap.add_argument("--per-sensor-top", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--ce-weight", type=float, default=0.25)
    ap.add_argument("--mse-weight", type=float, default=0.10)
    ap.add_argument("--freeze-non-pair-heads", action="store_true")
    ap.add_argument("--cell-balanced-sampling", action="store_true")
    ap.add_argument("--log-every", type=int, default=25)
    return ap.parse_args()


def load_state(model: torch.nn.Module, path: str, strict: bool) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not strict:
        current = model.state_dict()
        skipped = []
        filtered = {}
        for key, value in state.items():
            if key in current and tuple(current[key].shape) == tuple(value.shape):
                filtered[key] = value
            else:
                skipped.append(key)
        state = filtered
    else:
        skipped = []
    missing, unexpected = model.load_state_dict(state, strict=strict)
    print({"loaded": str(path), "strict": bool(strict), "missing": list(missing), "unexpected": list(unexpected)}, flush=True)
    if skipped:
        print({"skipped_shape_mismatch": skipped[:12], "skipped_count": len(skipped)}, flush=True)


def pair_pi_tensor(targets, idx, rows: int, device: torch.device) -> torch.Tensor | None:
    batch = [targets[int(i)] for i in idx]
    if getattr(batch[0], "pair_pi", None) is None:
        return None
    return torch.from_numpy(np.stack([t.pair_pi for t in batch]).astype(np.float32)).to(device)


def candidate_rows_from_batch(
    scores: torch.Tensor,
    valid: torch.Tensor,
    pair_pi: torch.Tensor | None,
    positive_top_k: int,
    per_sensor_top: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = int(scores.shape[1])
    all_pairs: list[list[tuple[int, int]]] = []
    for b in range(int(scores.shape[0])):
        seen: set[tuple[int, int]] = set()
        cand: list[tuple[int, int]] = []
        if pair_pi is not None:
            flat = pair_pi[b].detach().reshape(-1)
            nz = torch.nonzero(flat > 0.0, as_tuple=False).flatten()
            if int(nz.numel()) > 0:
                vals = flat[nz]
                take = min(int(positive_top_k), int(nz.numel()))
                for flat_idx in nz[torch.topk(vals, k=take).indices].tolist():
                    s_row = int(flat_idx) // rows
                    x_row = int(flat_idx) % rows
                    if s_row == x_row and s_row > 0:
                        continue
                    if bool(valid[b, s_row, 0]) and bool(valid[b, x_row, 1]):
                        key = (s_row, x_row)
                        if key not in seen:
                            seen.add(key)
                            cand.append(key)

        for sensor in range(2):
            logits = scores[b, :, sensor].detach().masked_fill(~valid[b, :, sensor], -1e9)
            top = torch.topk(logits, k=min(rows, max(1, int(per_sensor_top)))).indices.tolist()
            if 0 not in top and bool(valid[b, 0, sensor]):
                top.append(0)
            if sensor == 0:
                s_top = [int(v) for v in top]
            else:
                x_top = [int(v) for v in top]

        for s_row in s_top:
            for x_row in x_top:
                if s_row == x_row and s_row > 0:
                    continue
                if bool(valid[b, s_row, 0]) and bool(valid[b, x_row, 1]):
                    key = (int(s_row), int(x_row))
                    if key not in seen:
                        seen.add(key)
                        cand.append(key)
        all_pairs.append(cand or [(0, 0)])

    max_cands = max(len(c) for c in all_pairs)
    pair_rows = torch.zeros((scores.shape[0], max_cands, 2), dtype=torch.long, device=scores.device)
    pair_valid = torch.zeros((scores.shape[0], max_cands), dtype=torch.bool, device=scores.device)
    for b, cand in enumerate(all_pairs):
        pair_rows[b, : len(cand), :] = torch.tensor(cand, dtype=torch.long, device=scores.device)
        pair_valid[b, : len(cand)] = True
    return pair_rows, pair_valid


def distill_loss(
    teacher,
    student,
    x: torch.Tensor,
    slot: torch.Tensor,
    pair_pi: torch.Tensor | None,
    positive_top_k: int,
    per_sensor_top: int,
    temperature: float,
    ce_weight: float,
    mse_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.inference_mode():
        t_scores, t_q, t_action_ctx, t_valid = teacher._base_action_context(x, slot)
        pair_rows, cand_valid = candidate_rows_from_batch(
            t_scores,
            t_valid,
            pair_pi,
            positive_top_k=positive_top_k,
            per_sensor_top=per_sensor_top,
        )
        _ts, _tq, teacher_pair, _teacher_q, teacher_pair_valid = teacher.forward_pair_candidates_from_context(
            t_scores, t_q, t_action_ctx, t_valid, pair_rows
        )
        pair_valid = teacher_pair_valid & cand_valid
        teacher_logits = teacher_pair.masked_fill(~pair_valid, -30.0)
        teacher_prob = F.softmax(teacher_logits / max(1e-6, float(temperature)), dim=1)
    teacher_logits = teacher_logits.detach().clone()
    teacher_prob = teacher_prob.detach().clone()
    pair_valid = pair_valid.detach().clone()

    s_scores, s_q, s_action_ctx, s_valid = student._base_action_context(x, slot)
    _ss, _sq, student_pair, _student_q, student_pair_valid = student.forward_pair_candidates_from_context(
        s_scores, s_q, s_action_ctx, s_valid, pair_rows
    )
    pair_valid = pair_valid & student_pair_valid
    student_logits = student_pair.masked_fill(~pair_valid, -30.0)
    logp = F.log_softmax(student_logits / max(1e-6, float(temperature)), dim=1)
    kl = -(teacher_prob.detach() * logp).sum(dim=1).mean()

    loss = kl
    ce = torch.zeros((), device=x.device)
    if pair_pi is not None and float(ce_weight) > 0.0:
        bidx = torch.arange(x.shape[0], device=x.device)[:, None].expand_as(pair_rows[:, :, 0])
        target = pair_pi[bidx, pair_rows[:, :, 0], pair_rows[:, :, 1]].masked_fill(~pair_valid, 0.0)
        mass = target.sum(dim=1, keepdim=True)
        active = mass.squeeze(1) > 1e-6
        if bool(active.any()):
            target = target[active] / mass[active].clamp_min(1e-6)
            ce = -(target * F.log_softmax(student_logits[active], dim=1)).sum(dim=1).mean()
            loss = loss + float(ce_weight) * ce

    mse = torch.zeros((), device=x.device)
    if float(mse_weight) > 0.0:
        s_center = student_logits - student_logits.masked_fill(~pair_valid, 0.0).sum(dim=1, keepdim=True) / pair_valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        t_center = teacher_logits - teacher_logits.masked_fill(~pair_valid, 0.0).sum(dim=1, keepdim=True) / pair_valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        mse = F.mse_loss(s_center[pair_valid], t_center[pair_valid])
        loss = loss + float(mse_weight) * mse

    with torch.no_grad():
        acc = (student_logits.argmax(dim=1) == teacher_logits.argmax(dim=1)).float().mean()
    return loss, {"kl": float(kl.detach().cpu()), "ce": float(ce.detach().cpu()), "mse": float(mse.detach().cpu()), "teacher_top1_match": float(acc.detach().cpu())}


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.model_seed))
    np.random.seed(int(args.model_seed))
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args = SimpleNamespace(d_model=int(args.d_model), nhead=int(args.nhead), nlayers=int(args.nlayers))

    targets = usable_targets(Path(args.targets))
    teacher = make_physical_model(str(args.teacher_variant), model_args)
    load_state(teacher, args.teacher_state, strict=False)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = make_physical_model(str(args.student_variant), model_args)
    init_state = str(args.student_init_state or args.teacher_state)
    load_state(student, init_state, strict=False)
    student.to(device).train()
    if bool(args.freeze_non_pair_heads):
        for name, p in student.named_parameters():
            p.requires_grad = "pair_" in name.lower()

    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(args.model_seed))

    cell_indices = {}
    if bool(args.cell_balanced_sampling):
        for i, t in enumerate(targets):
            key = (int(getattr(t, "initial", -1)), float(getattr(t, "rate", 0.0)))
            cell_indices.setdefault(key, []).append(int(i))
        cell_indices = {k: np.asarray(v, dtype=np.int64) for k, v in cell_indices.items()}
        cell_keys = list(cell_indices.keys())
    else:
        cell_keys = []

    for step in range(int(args.train_steps)):
        if cell_keys:
            picks = []
            while len(picks) < int(args.batch_size):
                key = cell_keys[int(rng.integers(0, len(cell_keys)))]
                arr = cell_indices[key]
                picks.append(int(arr[int(rng.integers(0, len(arr)))]))
            idx = np.asarray(picks, dtype=np.int64)
        else:
            idx = rng.choice(len(targets), size=min(int(args.batch_size), len(targets)), replace=len(targets) < int(args.batch_size))
        x, slot, _sensor_pi, _sensor_q, _sensor_q_mask = batch_tensors(targets, idx, device)
        pair_pi = pair_pi_tensor(targets, idx, rows=x.shape[1], device=device)
        loss, stats = distill_loss(
            teacher,
            student,
            x,
            slot,
            pair_pi,
            positive_top_k=int(args.positive_top_k),
            per_sensor_top=int(args.per_sensor_top),
            temperature=float(args.temperature),
            ce_weight=float(args.ce_weight),
            mse_weight=float(args.mse_weight),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % max(1, int(args.log_every)) == 0 or step == int(args.train_steps) - 1:
            out = {"step": int(step), "loss": float(loss.detach().cpu()), **stats}
            print(out, flush=True)

    student.eval()
    out_path = Path(args.out_state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), out_path)
    print({"saved_student": str(out_path), "variant": str(args.student_variant)}, flush=True)


if __name__ == "__main__":
    main()
