from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazero_orthodox import joint_sensor_log_probs, load_targets
from mutual_foundation import MAXT, MutualRadarNet


class TwoRowFactorizedNet(nn.Module):
    def __init__(self, d_model: int = 48, nhead: int = 4, nlayers: int = 2, head_arch: str = "branch_context"):
        super().__init__()
        self.backbone = MutualRadarNet(d_model=d_model, nhead=nhead, nlayers=nlayers, head_arch=head_arch)
        self.sensor_embed = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.type_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.target_head = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.type_q_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.target_q_head = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.value_head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward_scores(self, tokens: torch.Tensor, slot: torch.Tensor):
        cls_out, tok_out, selected, token_active = self.backbone.encode_tokens(tokens)
        slot_emb = self.backbone.slot_proj(slot)
        bsz, rows, d_model = tok_out.shape

        sensor = self.sensor_embed[None, :, :].expand(bsz, -1, -1)
        cls_s = cls_out[:, None, :].expand(-1, 2, -1)
        slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
        type_ctx = torch.cat([cls_s, slot_s, sensor], dim=-1)
        type_logits = self.type_head(type_ctx)
        type_q = self.type_q_head(type_ctx)

        tok_st = tok_out[:, :, None, :].expand(-1, -1, 2, -1)
        cls_st = cls_out[:, None, None, :].expand(-1, rows, 2, -1)
        slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
        sensor_st = self.sensor_embed[None, None, :, :].expand(bsz, rows, -1, -1)
        target_ctx = torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)
        target_logits = self.target_head(target_ctx).squeeze(-1)
        target_q = self.target_q_head(target_ctx).squeeze(-1)

        scores = tokens.new_full((bsz, rows, 2), -1e9)
        q = tokens.new_zeros((bsz, rows, 2))
        scores[:, 0, :] = type_logits[:, :, 0]
        q[:, 0, :] = type_q[:, :, 0]
        track_mask = token_active & ~selected
        track_mask[:, 0] = False
        scores[:, 1:, :] = (type_logits[:, None, :, 1] + target_logits)[:, 1:, :]
        q[:, 1:, :] = (type_q[:, None, :, 1] + target_q)[:, 1:, :]
        row_is_search = torch.arange(rows, device=tokens.device)[None, :, None] == 0
        scores = scores.masked_fill((~track_mask[:, :, None]) & (~row_is_search), -1e9)
        return scores, q

    def forward_value(self, tokens: torch.Tensor, slot: torch.Tensor):
        cls_out, _tok_out, _selected, _token_active = self.backbone.encode_tokens(tokens)
        slot_emb = self.backbone.slot_proj(slot)
        return self.value_head(torch.cat([cls_out, slot_emb], dim=-1)).squeeze(-1)


def usable_targets(path: Path):
    targets = load_targets(path)
    out = []
    for t in targets:
        if getattr(t, "sensor_pi", None) is None:
            continue
        if float(np.asarray(t.sensor_pi).sum()) <= 0.0:
            continue
        out.append(t)
    if not out:
        raise RuntimeError(f"no usable sensor_pi targets in {path}")
    return out


def batch_tensors(targets, idx, device):
    batch = [targets[int(i)] for i in idx]
    x = torch.from_numpy(np.stack([t.x for t in batch]).astype(np.float32)).to(device)
    slot = torch.from_numpy(np.stack([t.slot for t in batch]).astype(np.float32)).to(device)
    sensor_pi = torch.from_numpy(np.stack([t.sensor_pi for t in batch]).astype(np.float32)).to(device)
    mass = sensor_pi.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    sensor_pi = sensor_pi / mass
    sensor_q = None
    sensor_q_mask = None
    if getattr(batch[0], "sensor_q", None) is not None and getattr(batch[0], "sensor_q_mask", None) is not None:
        sensor_q = torch.from_numpy(np.stack([t.sensor_q for t in batch]).astype(np.float32)).to(device)
        sensor_q_mask = torch.from_numpy(np.stack([t.sensor_q_mask for t in batch]).astype(np.float32)).to(device)
    return x, slot, sensor_pi, sensor_q, sensor_q_mask


def top_metrics(log_probs, sensor_pi):
    pred = log_probs.reshape(log_probs.shape[0], -1).argmax(dim=1)
    target = sensor_pi.reshape(sensor_pi.shape[0], -1).argmax(dim=1)
    top1 = (pred == target).float().mean()
    pred_base = pred // 2
    target_base = target // 2
    search_acc = ((pred_base == 0) == (target_base == 0)).float().mean()
    return float(top1.detach().cpu()), float(search_acc.detach().cpu())


def factorized_step(model, x, slot, sensor_pi, sensor_q, sensor_q_mask, value_scale):
    type_logit, track_logits, _value, type_q, track_q, sensor_logits, pred_sensor_q = model.forward_with_sensor(x, slot)
    log_probs = joint_sensor_log_probs(type_logit, track_logits, sensor_logits)
    policy_loss = -(sensor_pi * log_probs).sum(dim=(1, 2)).mean()
    q_loss = torch.zeros((), device=x.device)
    if sensor_q is not None and sensor_q_mask is not None and bool((sensor_q_mask > 0.5).any()):
        target_q = sensor_q / float(value_scale)
        branch_q = torch.where(torch.arange(MAXT + 1, device=x.device)[None, :] == 0, type_q[:, 1:2], type_q[:, 0:1])
        pred_q = branch_q[:, :, None] + track_q[:, :, None] + pred_sensor_q
        q_loss = F.smooth_l1_loss(pred_q[sensor_q_mask > 0.5], target_q[sensor_q_mask > 0.5])
    return policy_loss, q_loss, log_probs


def flat_step(model, x, slot, sensor_pi, sensor_q, sensor_q_mask, value_scale):
    logits, pred_q, _value = model.forward_physical_flat(x, slot)
    log_probs = F.log_softmax(logits.reshape(logits.shape[0], -1), dim=1).reshape_as(logits)
    policy_loss = -(sensor_pi * log_probs).sum(dim=(1, 2)).mean()
    q_loss = torch.zeros((), device=x.device)
    if sensor_q is not None and sensor_q_mask is not None and bool((sensor_q_mask > 0.5).any()):
        target_q = sensor_q / float(value_scale)
        q_loss = F.smooth_l1_loss(pred_q[sensor_q_mask > 0.5], target_q[sensor_q_mask > 0.5])
    return policy_loss, q_loss, log_probs


def two_row_factorized_step(model, x, slot, sensor_pi, sensor_q, sensor_q_mask, value_scale):
    scores, pred_q = model.forward_scores(x, slot)
    log_probs = F.log_softmax(scores.reshape(scores.shape[0], -1), dim=1).reshape_as(scores)
    policy_loss = -(sensor_pi * log_probs).sum(dim=(1, 2)).mean()
    q_loss = torch.zeros((), device=x.device)
    if sensor_q is not None and sensor_q_mask is not None and bool((sensor_q_mask > 0.5).any()):
        target_q = sensor_q / float(value_scale)
        q_loss = F.smooth_l1_loss(pred_q[sensor_q_mask > 0.5], target_q[sensor_q_mask > 0.5])
    return policy_loss, q_loss, log_probs


def variant_step(name: str, model, x, slot, sensor_pi, sensor_q, sensor_q_mask, value_scale):
    if name == "flat":
        return flat_step(model, x, slot, sensor_pi, sensor_q, sensor_q_mask, value_scale)
    if name == "two_row_factorized":
        return two_row_factorized_step(model, x, slot, sensor_pi, sensor_q, sensor_q_mask, value_scale)
    return factorized_step(model, x, slot, sensor_pi, sensor_q, sensor_q_mask, value_scale)


def eval_variant(model, name: str, targets, args, device, value_scale: float, max_batches: int = 16):
    model.eval()
    rng = np.random.default_rng(int(args.seed) + 1009)
    rows = []
    with torch.inference_mode():
        for _ in range(max(1, int(max_batches))):
            idx = rng.integers(0, len(targets), size=min(int(args.batch_size), len(targets)))
            x, slot, sensor_pi, sensor_q, sensor_q_mask = batch_tensors(targets, idx, device)
            policy_loss, q_loss, log_probs = variant_step(name, model, x, slot, sensor_pi, sensor_q, sensor_q_mask, value_scale)
            top1, search_acc = top_metrics(log_probs, sensor_pi)
            rows.append(
                {
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "q_loss": float(q_loss.detach().cpu()),
                    "top1": float(top1),
                    "search_acc": float(search_acc),
                }
            )
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def train_variant(name: str, train_targets, val_targets, args, device):
    if name == "two_row_factorized":
        model = TwoRowFactorizedNet(
            d_model=int(args.d_model),
            nhead=int(args.nhead),
            nlayers=int(args.nlayers),
            head_arch=str(args.head_arch),
        ).to(device)
    else:
        model = MutualRadarNet(d_model=int(args.d_model), nhead=int(args.nhead), nlayers=int(args.nlayers), head_arch=str(args.head_arch)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = np.random.default_rng(int(args.seed))
    abs_q = []
    for t in train_targets:
        if getattr(t, "sensor_q", None) is not None and getattr(t, "sensor_q_mask", None) is not None:
            mask = np.asarray(t.sensor_q_mask) > 0.5
            if np.any(mask):
                abs_q.extend(np.abs(np.asarray(t.sensor_q)[mask]).tolist())
    value_scale = max(1.0, float(np.percentile(abs_q, 90))) if abs_q else 10.0
    rows = []
    for step in range(int(args.steps)):
        idx = rng.integers(0, len(train_targets), size=int(args.batch_size))
        x, slot, sensor_pi, sensor_q, sensor_q_mask = batch_tensors(train_targets, idx, device)
        model.train()
        policy_loss, q_loss, log_probs = variant_step(name, model, x, slot, sensor_pi, sensor_q, sensor_q_mask, value_scale)
        loss = policy_loss + float(args.q_loss_weight) * q_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, int(args.log_every)) == 0 or step == int(args.steps) - 1:
            top1, search_acc = top_metrics(log_probs.detach(), sensor_pi)
            val = eval_variant(model, name, val_targets, args, device, value_scale, max_batches=int(args.val_batches))
            row = {
                "variant": name,
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "q_loss": float(q_loss.detach().cpu()),
                "top1": top1,
                "search_acc": search_acc,
                "val_policy_loss": val["policy_loss"],
                "val_q_loss": val["q_loss"],
                "val_top1": val["top1"],
                "val_search_acc": val["search_acc"],
                "value_scale": float(value_scale),
            }
            print(row, flush=True)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="CreateValid1/results/refstate_prefix_causalq_edfest_901_904_targets.pt")
    ap.add_argument("--out", default="CreateValid1/results/action_head_compare_smoke.csv")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--head-arch", default="branch_context")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--q-loss-weight", type=float, default=0.25)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.set_num_threads(1)
    device = torch.device(args.device if str(args.device) == "cpu" or torch.cuda.is_available() else "cpu")
    targets = usable_targets(Path(args.targets))
    rng = np.random.default_rng(int(args.seed) + 77)
    order = rng.permutation(len(targets))
    n_val = max(1, int(round(len(targets) * float(args.val_frac))))
    val_idx = set(int(i) for i in order[:n_val])
    train_targets = [t for i, t in enumerate(targets) if i not in val_idx]
    val_targets = [t for i, t in enumerate(targets) if i in val_idx]
    print({"targets": len(targets), "train": len(train_targets), "val": len(val_targets), "path": str(args.targets)}, flush=True)
    frames = [
        train_variant("factorized", train_targets, val_targets, args, device),
        train_variant("two_row_factorized", train_targets, val_targets, args, device),
        train_variant("flat", train_targets, val_targets, args, device),
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
