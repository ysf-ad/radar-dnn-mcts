from __future__ import annotations

import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from balanced_q_retrain import BASE_Q, make_env, make_q_planner
from final_radar_campaign import MAXT, run_fixed, seedall, summarize_window_df
from mutual_features import slot_features, tokenize
from mutual_foundation import MutualRadarDirectPlanner
from realistic_reward_retrain import adapter
from refresh_method_suite import load_mutual
from repaired_campaign_tools import EDFPlanner, ESTPlanner, load_student_model
from sequence_decoder_experiment import ParallelSequenceDecoder, SequenceDirectPlanner


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CreateValid1" / "results" / "original_factorized_direct_compare"
OUT.mkdir(parents=True, exist_ok=True)
CLEAN = Path(r"C:\Users\yousi\Downloads\radar_outputs")
CLEAN.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)


def single_env(rate: float):
    env = make_env(rate, refresh=1, strict=False)
    env["enable_x_band"] = 0
    return env


def load_qdistill():
    model = ParallelSequenceDecoder(seq_len=32).to(DEVICE)
    model.load_state_dict(
        torch.load(
            ROOT / "CreateValid1/results/single_sensor_qdistill_batch/qdistill_batch_decoder.pt",
            map_location=DEVICE,
        )
    )
    return model.eval()


def load_original_pvq():
    d = ROOT / "CreateValid1/results/mutual_improvement/refresh_pvq_factorized"
    return load_mutual(d).to(DEVICE).eval()


def make_teacher(env):
    q_model = load_student_model(str(BASE_Q), MAXT, "cpu", q_head_use_tanh=False)
    return make_q_planner(q_model, env, rollouts=16, topk=8, search_prior_scale=2.0, q_weight=1.0, sim_h=600.0)


class OriginalPVQBatchHeadPlanner:
    """Use the trained original factorized heads in one batched pass.

    This intentionally does not train/distill a sequence head. It reuses the
    original policy/Q heads for many synthetic window slots at once. Because the
    state is not updated between slots, this is a diagnostic for "can the
    original factorized head be used directly as a 0-rollout batch decoder?"
    """

    def __init__(self, model, threshold: float = 0.75, seq_len: int = 32, mode: str = "q"):
        self.model = model.eval()
        self.adapt = adapter()
        self.threshold = float(threshold)
        self.seq_len = int(seq_len)
        self.mode = str(mode)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def warmup(self, obs, budget_ms=200):
        _ = self.plan(obs, budget_ms)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def plan(self, obs, budget_ms=200):
        x = tokenize(self.adapt, obs, selected=set(), search_count=0)
        slots = []
        last_action = -1
        for i in range(self.seq_len):
            frac = i / max(1, self.seq_len - 1)
            slots.append(slot_features(obs, frac * float(budget_ms), i, i, last_action, float(budget_ms)))
            last_action = 0
        with torch.inference_mode():
            tokens = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            cls, tok, selected, active = self.model.encode_tokens(tokens)
            slot_t = torch.from_numpy(__import__("numpy").stack(slots).astype("float32")).to(self.device)
            cls = cls.expand(self.seq_len, -1)
            tok = tok.expand(self.seq_len, -1, -1)
            selected = selected.expand(self.seq_len, -1)
            active = active.expand(self.seq_len, -1)
            type_logit, track_logits, _, type_q, track_q = self.model.forward_heads(cls, tok, selected, active, slot_t)
            if self.mode == "q":
                best_track = torch.argmax(type_q[:, 0:1] + track_q, dim=-1)
                choose_search = type_q[:, 1] >= (type_q[:, 0] + track_q[torch.arange(self.seq_len, device=self.device), best_track])
            else:
                p_search = torch.sigmoid(type_logit)
                best_track = torch.argmax(track_logits, dim=-1)
                choose_search = p_search >= self.threshold
            actions = torch.where(choose_search, torch.zeros_like(best_track), best_track).detach().cpu().numpy().astype(int)
            scores_np = track_logits.detach().cpu().numpy()
        used = set()
        fixed = []
        for i, a in enumerate(actions.tolist()):
            if a <= 0:
                fixed.append(0)
                continue
            if a not in used:
                used.add(int(a))
                fixed.append(int(a))
                continue
            row = scores_np[i].copy()
            row[0] = -1e9
            for u in used:
                if 0 <= u < row.shape[0]:
                    row[u] = -1e9
            repl = int(__import__("numpy").argmax(row))
            fixed.append(repl if row[repl] > -1e8 else 0)
            if fixed[-1] > 0:
                used.add(fixed[-1])
        return fixed


def evaluate():
    raw_path = OUT / "original_direct_raw.csv"
    win_path = OUT / "original_direct_windows.csv"
    if raw_path.exists() and win_path.exists():
        return pd.read_csv(raw_path), pd.read_csv(win_path)

    qdistill = load_qdistill()
    pvq = load_original_pvq()
    rows, wins = [], []
    cells = [(15, 0.0), (15, 2.0), (50, 0.0), (50, 2.0), (75, 2.0), (100, 2.0)]
    for init, rate in cells:
        env = single_env(rate)
        methods = {
            "QDistillBatch_t0.75": lambda qdistill=qdistill: SequenceDirectPlanner(qdistill, threshold=0.75, mode="branch", allow_retrack=False),
            "OriginalPVQ_direct_t0.30": lambda pvq=pvq: MutualRadarDirectPlanner(
                pvq,
                alpha=1.0,
                beta=1.0,
                threshold=0.30,
                direct_mode="branch",
                simulate_state=True,
                search_refresh_tracked=True,
            ),
            "OriginalPVQ_direct_t0.50": lambda pvq=pvq: MutualRadarDirectPlanner(
                pvq,
                alpha=1.0,
                beta=1.0,
                threshold=0.50,
                direct_mode="branch",
                simulate_state=True,
                search_refresh_tracked=True,
            ),
            "OriginalPVQ_direct_q": lambda pvq=pvq: MutualRadarDirectPlanner(
                pvq,
                alpha=1.0,
                beta=1.0,
                threshold=0.0,
                direct_mode="q",
                simulate_state=True,
                search_refresh_tracked=True,
            ),
            "OriginalPVQ_batchhead_q": lambda pvq=pvq: OriginalPVQBatchHeadPlanner(pvq, mode="q"),
            "OriginalPVQ_batchhead_t0.75": lambda pvq=pvq: OriginalPVQBatchHeadPlanner(pvq, threshold=0.75, mode="branch"),
            "LegacyQ_Teacher_r16": lambda env=env: make_teacher(env),
            "EDF": lambda: EDFPlanner(MAXT),
            "EST": lambda: ESTPlanner(MAXT),
        }
        for name, factory in methods.items():
            seedall(93)
            t0 = time.perf_counter()
            planner = factory()
            w, _ = run_fixed(planner, name, init, MAXT, 93, 60, 200, env)
            s = summarize_window_df(w, "fixed")
            s.update(planner=name, initial_targets=init, rate=rate, seed=93, wall_s=time.perf_counter() - t0)
            rows.append(s)
            ww = w.copy()
            ww["planner"] = name
            ww["initial_targets"] = init
            ww["rate"] = rate
            ww["seed"] = 93
            wins.append(ww)
            print(
                "eval",
                init,
                rate,
                name,
                round(s["reward_per_200ms_eq"], 3),
                round(s["mean_drop_pct_active"], 2),
                round(s["mean_delay_active"], 1),
                round(s["search_fraction"], 2),
                round(s["planning_ms_per_200ms_eq"], 2),
                flush=True,
            )
    raw = pd.DataFrame(rows)
    win = pd.concat(wins, ignore_index=True)
    raw.to_csv(raw_path, index=False)
    win.to_csv(win_path, index=False)
    return raw, win


def make_filtered_frontier_plot():
    src = ROOT / "CreateValid1/results/distill_teacher_window_frontier/window_frontier_summary.csv"
    fr = pd.read_csv(src)
    student = fr[fr["planner"].str.startswith("QDistill")].copy()
    # Keep only competitive settings so pathological 33/50ms execution does not
    # dominate the visual range.
    student = student[(student["reward_per_200ms"] >= 1.45) & (student["drop_pct"] <= 3.0)]
    out_csv = OUT / "window_frontier_competitive_only.csv"
    student.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    for name, sub in student.groupby("planner"):
        sub = sub.sort_values("latency_ms_per_200")
        ax.plot(sub["latency_ms_per_200"], sub["reward_per_200ms"], marker="o", label=name)
        for _, r in sub.iterrows():
            ax.annotate(r["config"], (r["latency_ms_per_200"], r["reward_per_200ms"]), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_title("Window Frontier, Competitive Settings Only")
    ax.set_xlabel("planning latency per executed 200ms (ms, log)")
    ax.set_ylabel("reward per executed 200ms")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUT / "window_frontier_competitive_only.png"
    fig.savefig(out)
    plt.close(fig)
    return out, out_csv


def plot(raw: pd.DataFrame, win: pd.DataFrame):
    summary = (
        raw.groupby("planner")
        .agg(
            reward_per_200ms=("reward_per_200ms_eq", "mean"),
            final_cumulative=("final_cumulative_reward", "mean"),
            drop_pct=("mean_drop_pct_active", "mean"),
            avg_delay_ms=("mean_delay_active", "mean"),
            tracked=("mean_tracked_targets", "mean"),
            active=("mean_active_targets", "mean"),
            search_fraction=("search_fraction", "mean"),
            latency_ms_per_200=("planning_ms_per_200ms_eq", "mean"),
        )
        .reset_index()
        .sort_values("reward_per_200ms", ascending=False)
    )
    summary.to_csv(OUT / "original_direct_summary.csv", index=False)

    piv = win.pivot_table(
        index="elapsed_ms",
        columns=["planner", "initial_targets", "rate", "seed"],
        values="cumulative_reward",
        aggfunc="last",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    for name in summary["planner"]:
        cols = [c for c in piv.columns if c[0] == name]
        if cols:
            ax.plot(piv.index / 1000.0, piv[cols].mean(axis=1), label=name, linewidth=2 if "QDistill" in name or "OriginalPVQ" in name else 1.3)
    ax.set_title("Original Factorized PVQ Direct vs Distilled Batch")
    ax.set_xlabel("episode time (s)")
    ax.set_ylabel("mean cumulative reward")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    cum = OUT / "original_direct_cumulative.png"
    fig.savefig(cum)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    ax.scatter(summary["latency_ms_per_200"], summary["reward_per_200ms"], s=70)
    for _, r in summary.iterrows():
        ax.annotate(r["planner"], (r["latency_ms_per_200"], r["reward_per_200ms"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_title("Original Direct vs Distilled Batch: Reward-Latency")
    ax.set_xlabel("latency per 200ms (ms, log)")
    ax.set_ylabel("reward per 200ms")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    lat = OUT / "original_direct_reward_latency.png"
    fig.savefig(lat)
    plt.close(fig)

    front_plot, front_csv = make_filtered_frontier_plot()
    for src in [OUT / "original_direct_summary.csv", cum, lat, front_plot, front_csv]:
        shutil.copy2(src, CLEAN / src.name)
    return summary, cum, lat, front_plot


def main():
    raw, win = evaluate()
    summary, cum, lat, front = plot(raw, win)
    print(summary.to_string(index=False))
    print("OUT", OUT.resolve())
    print("PLOTS", cum.resolve(), lat.resolve(), front.resolve())
    print("CLEAN", CLEAN.resolve())


if __name__ == "__main__":
    main()
