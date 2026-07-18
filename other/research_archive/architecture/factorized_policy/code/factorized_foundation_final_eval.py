from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from alpharadar_foundation_train import AlphaRadarNet
from final_radar_campaign import MAXT, run_fixed, seedall, summarize_window_df
from hierarchical_sequence_transformer import slot_features, tokenize
from load_adaptive_train_eval import ComboPlanner, load_model, make_env
from realistic_reward_retrain import adapter
from repaired_campaign_tools import EDFPlanner, ESTPlanner, SEARCH_DWELL_MS


ROOT = Path(r"C:\Users\yousi\Downloads\Model1 1\CreateValid1\experiments\code")
RES = ROOT / "CreateValid1" / "results"
OUT = RES / "load_adaptive_20260508" / "factorized_foundation_clean_20260513"
VISIBLE = Path(r"C:\Users\yousi\Downloads\radar_figures_visible")
VISIBLE.mkdir(parents=True, exist_ok=True)


class CachedSmallFactorizedPlanner:
    """Fast cached deployment for the factorized foundation model.

    One transformer encoder pass per 200 ms window; repeated cheap factorized
    type/track heads generate the sequence. This preserves the two-head
    semantics while bringing latency back to the old PolicyQ range.
    """

    def __init__(self, model, threshold=0.55):
        self.model = model.eval()
        self.threshold = float(threshold)
        self.adapt = adapter()

    def plan(self, obs, budget_ms=200):
        selected = set()
        plan = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
        x = tokenize(self.adapt, obs, selected=None, search_count=0)
        tokens = torch.from_numpy(x).float().unsqueeze(0)
        with torch.inference_mode():
            token_active = tokens[:, :, 4] > 0.5
            token_active[:, 0] = True
            emb = self.model.token_proj(tokens)
            cls = self.model.cls_token[None, None, :].expand(tokens.shape[0], 1, -1)
            emb = torch.cat([cls, emb], dim=1)
            cls_valid = torch.ones((1, 1), dtype=torch.bool)
            out = self.model.encoder(emb, src_key_padding_mask=~torch.cat([cls_valid, token_active], dim=1))
            cls_out = out[:, 0, :]
            tok_out = out[:, 1:, :]
            cls_rep = cls_out.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
            while elapsed < float(budget_ms) and len(plan) < 64:
                slot = torch.from_numpy(slot_features(obs, elapsed, search_count, track_count, last, float(budget_ms))).float().unsqueeze(0)
                slot_emb = self.model.slot_proj(slot)
                type_logit = self.model.type_head(torch.cat([cls_out, slot_emb], dim=-1)).squeeze(-1)
                p_search = float(torch.sigmoid(type_logit[0]))
                slot_rep = slot_emb.unsqueeze(1).expand(-1, tok_out.shape[1], -1)
                logits = self.model.track_head(torch.cat([tok_out, cls_rep, slot_rep], dim=-1)).squeeze(-1)[0].numpy()
                mask = token_active[0].numpy().astype(bool)
                mask[0] = False
                for a in selected:
                    if 0 <= a < len(mask):
                        mask[a] = False
                logits = np.where(mask, logits, -1e9)
                best = int(np.argmax(logits))
                has_track = np.isfinite(logits[best]) and logits[best] > -1e8
                if (not has_track) or p_search >= self.threshold:
                    a = 0
                    elapsed += SEARCH_DWELL_MS
                    search_count += 1
                else:
                    a = best
                    selected.add(a)
                    elapsed += max(1.0, float(dwell[a - 1]) if 1 <= a <= len(dwell) else SEARCH_DWELL_MS)
                    track_count += 1
                plan.append(int(a))
                last = int(a)
        return plan or [0]


def load_factorized():
    # Epoch 10 is the selected checkpoint: it had the best held-out reward/latency
    # tradeoff before the small model started overfitting the sparse-search label.
    ckpt = OUT / "factorized_foundation_clean_small.pt"
    model = AlphaRadarNet(d_model=48, nhead=4, nlayers=1)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model


def main():
    model = load_factorized()
    old = load_model(RES / "realistic_reward_retrain_20260508" / "realistic_return_q_policy.pt")
    thresholds = [0.35, 0.45, 0.55, 0.65, 0.75]
    scenarios = [
        ("main_50_rate2_100s", 50, 2.0),
        ("stress_100_rate5_100s", 100, 5.0),
        ("light_15_rate1_100s", 15, 1.0),
    ]
    rows, traces = [], []
    for label, init, rate in scenarios:
        env = make_env(rate)
        specs = {f"FactorizedFoundation_t{t:g}": (lambda t=t: CachedSmallFactorizedPlanner(model, t)) for t in thresholds}
        specs.update(
            {
                "Old_MixedPolicyQ_a0.5": lambda: ComboPlanner(old, 0.5),
                "Old_MixedPolicyQ_a1.5": lambda: ComboPlanner(old, 1.5),
                "EDF": lambda: EDFPlanner(MAXT),
                "EST": lambda: ESTPlanner(MAXT),
            }
        )
        for seed in [50, 60, 70]:
            for name, fac in specs.items():
                seedall(seed)
                w, _ = run_fixed(fac(), name, init, MAXT, seed, 500, 200, env)
                s = summarize_window_df(w, "fixed")
                s.update(
                    scenario=label,
                    planner=name,
                    seed=seed,
                    initial_targets=init,
                    rate=rate,
                    final_cumulative_reward=float(w["cumulative_reward"].iloc[-1]),
                )
                rows.append(s)
                x = w.copy()
                x["scenario"] = label
                x["seed"] = seed
                traces.append(x)
                print(label, name, seed, f"r={s['reward_per_200ms_eq']:.3f}", f"lat={s['planning_ms_per_200ms_eq']:.2f}", flush=True)

    raw = pd.DataFrame(rows)
    win = pd.concat(traces, ignore_index=True)
    raw.to_csv(OUT / "factorized_foundation_final_eval_raw.csv", index=False)
    win.to_csv(OUT / "factorized_foundation_final_eval_windows.csv", index=False)
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
    summary.to_csv(OUT / "factorized_foundation_final_eval_summary.csv", index=False)
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
        p = OUT / f"{label}_factorized_foundation_final_cumulative.png"
        v = VISIBLE / p.name
        fig.savefig(p, dpi=180)
        fig.savefig(v, dpi=180)
        plt.close(fig)
    print(summary.to_string(index=False))
    print("OUT", OUT)
    print("VISIBLE", VISIBLE)


if __name__ == "__main__":
    main()
