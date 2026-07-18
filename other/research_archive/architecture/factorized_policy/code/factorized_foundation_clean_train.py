"""Clean factorized foundation model for radar scheduling.

This intentionally removes mixed planners from the training target.  The model
is one shared transformer encoder with factorized heads:

1. type head: search vs track
2. track head: which target if tracking
3. value/Q heads: auxiliary return prediction, kept for later MCTS use

The direct online planner is autoregressive over a 200 ms window, so search can
appear multiple times. That fixes the one-search-token limitation of the older
one-shot PolicyQ sorting model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from alpharadar_foundation_train import AlphaRadarNet
from final_radar_campaign import MAXT, build_env, get_obs, run_fixed, seedall, summarize_window_df
from hierarchical_sequence_transformer import slot_features, tokenize
from load_adaptive_train_eval import ComboPlanner, load_model, make_env
from realistic_reward_retrain import adapter
from repaired_campaign_tools import EDFPlanner, ESTPlanner, SEARCH_DWELL_MS
from strict_window_report import execute_plan_until_budget


ROOT = Path(r"C:\Users\yousi\Downloads\Model1 1\CreateValid1\experiments\code")
RES = ROOT / "CreateValid1" / "results"
OUT = RES / "load_adaptive_20260508" / "factorized_foundation_clean_20260513"
VISIBLE = Path(r"C:\Users\yousi\Downloads\radar_figures_visible")
OUT.mkdir(parents=True, exist_ok=True)
VISIBLE.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)


class EDFSearchEvery:
    """Single deterministic curriculum teacher: EDF plus fixed sparse search."""

    def __init__(self, every=5):
        self.every = int(every)

    def plan(self, obs, budget_ms=200):
        active = np.asarray(obs["active_mask"]).astype(bool)
        dead = np.asarray(obs["t_deadline"])
        valid = (np.where(active & (dead >= 0.0))[0] + 1).tolist()
        valid.sort(key=lambda a: float(dead[a - 1]))
        out = []
        for i, a in enumerate(valid):
            if i > 0 and i % self.every == 0:
                out.append(0)
            out.append(int(a))
        out.extend([0] * 20)
        return out or [0]


class FactorizedFoundationPlanner:
    def __init__(self, model: AlphaRadarNet, threshold=0.5, q_alpha=0.0, type_q_beta=0.0):
        self.model = model.eval()
        self.threshold = float(threshold)
        self.q_alpha = float(q_alpha)
        self.type_q_beta = float(type_q_beta)
        self.adapt = adapter()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def plan(self, obs, budget_ms=200):
        selected = set()
        plan = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
        while elapsed < float(budget_ms) and len(plan) < 64:
            x = tokenize(self.adapt, obs, selected=selected, search_count=search_count)
            s = slot_features(obs, elapsed, search_count, track_count, last, float(budget_ms))
            with torch.inference_mode():
                type_logit, track_logits, _, type_q, track_q = self.model(
                    torch.from_numpy(x).float().unsqueeze(0).to(self.device),
                    torch.from_numpy(s).float().unsqueeze(0).to(self.device),
                )
            p_search = float(torch.sigmoid(type_logit[0] + self.type_q_beta * type_q[0, 1]).detach().cpu())
            track_scores = track_logits[0].detach().cpu().numpy()
            if self.q_alpha:
                track_scores = track_scores + self.q_alpha * track_q[0].detach().cpu().numpy()
            best_track = int(np.argmax(track_scores))
            has_track = np.isfinite(track_scores[best_track]) and track_scores[best_track] > -1e8
            if (not has_track) or p_search >= self.threshold:
                a = 0
                elapsed += SEARCH_DWELL_MS
                search_count += 1
            else:
                a = best_track
                selected.add(a)
                elapsed += max(1.0, float(dwell[a - 1]) if 1 <= a <= len(dwell) else SEARCH_DWELL_MS)
                track_count += 1
            plan.append(int(a))
            last = int(a)
        return plan or [0]


def collect_dataset(episodes=140, windows=120, every=5, overwrite=False):
    path = OUT / f"factorized_clean_e{episodes}_w{windows}_every{every}.npz"
    meta_path = OUT / f"factorized_clean_e{episodes}_w{windows}_every{every}_meta.csv"
    if path.exists() and not overwrite:
        z = np.load(path)
        return {k: z[k] for k in z.files}

    adapt = adapter()
    inits = [10, 15, 20, 30, 40, 50, 60, 75, 100]
    rates = [0.0, 1.0, 2.0, 5.0]
    xs, slots, y_type, y_track, y_ret, y_q_type, y_q_track, meta = [], [], [], [], [], [], [], []
    t0 = time.perf_counter()

    for ep in range(episodes):
        seed = 140000 + ep
        seedall(seed)
        init = inits[ep % len(inits)]
        rate = rates[(ep // len(inits)) % len(rates)]
        env = make_env(rate)
        teacher = EDFSearchEvery(every)
        eng = build_env(teacher, init, MAXT, seed, 200, env)
        eng.reset(seed=seed)
        debt = 0.0
        pending = []
        total = 0.0

        for w in range(windows):
            if eng.term_buf[0]:
                break
            obs = get_obs(eng, debt)
            plan = teacher.plan(obs, 200)
            elapsed = 0.0
            search_count = 0
            track_count = 0
            last = -1
            selected = set()
            dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
            step_indices = []
            for a in plan:
                if elapsed >= 200.0:
                    break
                a = int(a)
                if a != 0:
                    idx = a - 1
                    if idx < 0 or idx >= len(obs["active_mask"]) or not obs["active_mask"][idx] or obs["t_deadline"][idx] < 0:
                        continue
                xs.append(tokenize(adapt, obs, selected=selected, search_count=search_count))
                slots.append(slot_features(obs, elapsed, search_count, track_count, last, 200.0))
                y_type.append(1.0 if a == 0 else 0.0)
                y_track.append(max(0, a))
                y_q_type.append(1 if a == 0 else 0)
                y_q_track.append(max(0, a))
                step_indices.append(len(y_ret))
                y_ret.append(0.0)
                if a == 0:
                    elapsed += SEARCH_DWELL_MS
                    search_count += 1
                else:
                    selected.add(a)
                    elapsed += max(1.0, float(dwell[a - 1]))
                    track_count += 1
                last = a

            reward, spent, debt, executed, _, _ = execute_plan_until_budget(eng, plan, 200.0, debt, "factorized_clean_teacher", seed, w)
            total += float(reward)
            # Same-window return is the supervised Q target for each action in
            # that planned window; this is auxiliary, policy heads drive direct inference.
            for idx in step_indices:
                y_ret[idx] = float(reward)
            if executed <= 0 or spent <= 0:
                break

        eng.close()
        meta.append({"episode": ep, "init": init, "rate": rate, "samples": len(y_type), "return": total})
        if (ep + 1) % 10 == 0:
            print(f"collect {ep+1}/{episodes} samples={len(y_type)} elapsed={time.perf_counter()-t0:.1f}s", flush=True)

    data = {
        "x": np.asarray(xs, dtype=np.float32),
        "slot": np.asarray(slots, dtype=np.float32),
        "y_type": np.asarray(y_type, dtype=np.float32),
        "y_track": np.asarray(y_track, dtype=np.int64),
        "ret": np.asarray(y_ret, dtype=np.float32),
        "q_type_action": np.asarray(y_q_type, dtype=np.int64),
        "q_track_action": np.asarray(y_q_track, dtype=np.int64),
    }
    np.savez_compressed(path, **data)
    pd.DataFrame(meta).to_csv(meta_path, index=False)
    return data


def train_model(data, epochs=14, batch=512, overwrite=False):
    ckpt = OUT / "factorized_foundation_clean.pt"
    meta_path = OUT / "factorized_foundation_clean_meta.json"
    if ckpt.exists() and not overwrite:
        model = AlphaRadarNet()
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
        return model

    model = AlphaRadarNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    x = torch.from_numpy(data["x"]).to(DEVICE)
    slot = torch.from_numpy(data["slot"]).to(DEVICE)
    yt = torch.from_numpy(data["y_type"]).to(DEVICE)
    ya = torch.from_numpy(data["y_track"]).to(DEVICE)
    ret_scale = float(max(1.0, np.percentile(np.abs(data["ret"]), 90)))
    ret = torch.from_numpy(data["ret"] / ret_scale).to(DEVICE)
    q_type_action = torch.from_numpy(data["q_type_action"]).to(DEVICE)
    q_track_action = torch.from_numpy(data["q_track_action"]).to(DEVICE)
    n = x.shape[0]
    rows = []

    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=DEVICE)
        losses, type_accs, track_accs, search_rates = [], [], [], []
        for st in range(0, n, batch):
            idx = perm[st : st + batch]
            type_logit, track_logits, value, type_q, track_q = model(x[idx], slot[idx])
            loss_type = F.binary_cross_entropy_with_logits(type_logit, yt[idx])
            track_rows = yt[idx] < 0.5
            loss_track = F.cross_entropy(track_logits[track_rows], ya[idx][track_rows]) if bool(torch.any(track_rows)) else torch.zeros((), device=DEVICE)
            loss_value = F.smooth_l1_loss(value, ret[idx])
            loss_type_q = F.smooth_l1_loss(type_q[torch.arange(idx.numel(), device=DEVICE), q_type_action[idx]], ret[idx])
            q_rows = q_track_action[idx] > 0
            loss_track_q = F.smooth_l1_loss(track_q[q_rows, q_track_action[idx][q_rows]], ret[idx][q_rows]) if bool(torch.any(q_rows)) else torch.zeros((), device=DEVICE)
            loss = loss_type + loss_track + 0.15 * loss_value + 0.1 * loss_type_q + 0.15 * loss_track_q
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.inference_mode():
                pred_type = (torch.sigmoid(type_logit) > 0.5).float()
                type_accs.append(float((pred_type == yt[idx]).float().mean().detach().cpu()))
                if bool(torch.any(track_rows)):
                    track_accs.append(float((track_logits[track_rows].argmax(dim=1) == ya[idx][track_rows]).float().mean().detach().cpu()))
                search_rates.append(float(pred_type.mean().detach().cpu()))
            losses.append(float(loss.detach().cpu()))
        row = {
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "type_acc": float(np.mean(type_accs)),
            "track_acc": float(np.mean(track_accs)),
            "argmax_search": float(np.mean(search_rates)),
            "ret_scale": ret_scale,
        }
        rows.append(row)
        print("train_factorized", row, flush=True)

    torch.save(model.cpu().state_dict(), ckpt)
    pd.DataFrame(rows).to_csv(OUT / "factorized_foundation_clean_train_log.csv", index=False)
    meta_path.write_text(json.dumps({"ret_scale": ret_scale, "teacher": "single deterministic EDFSearchEvery(5)", "heads": "type, track, value, type_q, track_q"}, indent=2), encoding="utf-8")
    model.eval()
    return model


def evaluate(model, windows=500, seeds=(50, 60, 70)):
    old_model = load_model(RES / "realistic_reward_retrain_20260508" / "realistic_return_q_policy.pt")
    scenarios = [
        ("main_50_rate2_100s", 50, 2.0),
        ("stress_100_rate5_100s", 100, 5.0),
        ("light_15_rate1_100s", 15, 1.0),
    ]
    thresholds = [0.35, 0.45, 0.55, 0.65, 0.75]
    rows, traces = [], []
    for label, init, rate in scenarios:
        env = make_env(rate)
        specs = {f"Factorized_t{t:g}": (lambda t=t: FactorizedFoundationPlanner(model, threshold=t)) for t in thresholds}
        specs.update(
            {
                "Old_MixedPolicyQ_a0.5": lambda: ComboPlanner(old_model, 0.5),
                "Old_MixedPolicyQ_a1.5": lambda: ComboPlanner(old_model, 1.5),
                "EDF": lambda: EDFPlanner(MAXT),
                "EST": lambda: ESTPlanner(MAXT),
            }
        )
        for seed in seeds:
            for name, fac in specs.items():
                seedall(seed)
                w, _ = run_fixed(fac(), name, init, MAXT, seed, windows, 200, env)
                s = summarize_window_df(w, "fixed")
                s.update(scenario=label, planner=name, seed=seed, initial_targets=init, rate=rate, final_cumulative_reward=float(w["cumulative_reward"].iloc[-1]))
                rows.append(s)
                xdf = w.copy()
                xdf["scenario"] = label
                xdf["seed"] = seed
                traces.append(xdf)
                print(label, name, seed, f"r={s['reward_per_200ms_eq']:.3f}", f"cum={s['final_cumulative_reward']:.1f}", f"lat={s['planning_ms_per_200ms_eq']:.2f}", flush=True)

    raw = pd.DataFrame(rows)
    win = pd.concat(traces, ignore_index=True)
    raw.to_csv(OUT / "factorized_foundation_eval_raw.csv", index=False)
    win.to_csv(OUT / "factorized_foundation_eval_windows.csv", index=False)
    summary = (
        raw.groupby(["scenario", "planner"])
        .agg(
            cumulative=("final_cumulative_reward", "mean"),
            reward=("reward_per_200ms_eq", "mean"),
            drop=("mean_drop_pct_active", "mean"),
            delay=("mean_delay_active", "mean"),
            search=("search_fraction", "mean"),
            latency=("planning_ms_per_200ms_eq", "mean"),
        )
        .reset_index()
        .sort_values(["scenario", "reward"], ascending=[True, False])
    )
    summary.to_csv(OUT / "factorized_foundation_eval_summary.csv", index=False)

    for label, _, _ in scenarios:
        top = summary[summary.scenario.eq(label)].head(7)["planner"].tolist()
        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        for name in top:
            sub = win[(win.scenario == label) & (win.planner == name)]
            mean = sub.groupby("window")["cumulative_reward"].mean()
            ax.plot((mean.index + 1) * 0.2, mean.values, label=name, linewidth=2)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative reward")
        ax.set_title(label.replace("_", " "))
        path = OUT / f"{label}_factorized_foundation_cumulative.png"
        visible = VISIBLE / path.name
        fig.savefig(path, dpi=180)
        fig.savefig(visible, dpi=180)
        plt.close(fig)

    print("\nSUMMARY")
    print(summary.to_string(index=False))
    return summary


def main():
    data = collect_dataset(overwrite=True)
    print("dataset", {k: v.shape for k, v in data.items()}, "search_frac", float(data["y_type"].mean()), flush=True)
    model = train_model(data, overwrite=True)
    summary = evaluate(model)
    print("OUT", OUT)
    print("VISIBLE", VISIBLE)


if __name__ == "__main__":
    main()
