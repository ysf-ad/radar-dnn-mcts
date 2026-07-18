from __future__ import annotations

import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from balanced_q_retrain import make_env
from final_radar_campaign import MAXT, build_env, get_obs, run_fixed, seedall, summarize_window_df
from mutual_features import tokenize
from realistic_reward_retrain import adapter
from refresh_method_suite import make_base_q, make_legacy_policy
from repaired_campaign_tools import EDFPlanner, ESTPlanner, infer_elapsed_ms
from sensor_batch_decoder_suite import SensorBatchPlanner, SensorFactorizedBatchDecoder
from sequence_decoder_experiment import ParallelSequenceDecoder, SequenceDirectPlanner
from strict_window_report import SEARCH_DWELL_MS


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CreateValid1" / "results" / "xs_curriculum_batch"
OUT.mkdir(parents=True, exist_ok=True)
CLEAN = Path(r"C:\Users\yousi\Downloads\radar_outputs")
CLEAN.mkdir(parents=True, exist_ok=True)
EXT = ROOT / "CreateValid1" / "results" / "extended_refresh_suite"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)


def xband_env(rate: float):
    env = make_env(rate, refresh=1, strict=False)
    env["enable_x_band"] = 1
    return env


def load_base_decoder():
    m = ParallelSequenceDecoder(seq_len=32).to(DEVICE)
    m.load_state_dict(torch.load(EXT / "batch_factorized_decoder.pt", map_location=DEVICE))
    return m.eval()


def step_action(eng, action: int, debt: float):
    from pufferlib.ocean.radarxs import binding

    before = get_obs(eng, debt)
    if action != 0:
        idx = int(action) - 1
        if idx < 0 or idx >= len(before["active_mask"]) or not before["active_mask"][idx] or before["t_deadline"][idx] < 0:
            return False, 0.0, debt
    eng.act_buf[0] = int(action)
    binding.vec_step(eng.env)
    after = get_obs(eng, debt)
    if action == 0:
        return True, SEARCH_DWELL_MS, 0.0
    dt = infer_elapsed_ms(before, after)
    return True, float(dt), debt + max(float(dt), 0.0)


def collect_data(seq_len: int = 32):
    path = OUT / "xs_curriculum_teacher_data.npz"
    if path.exists():
        z = np.load(path)
        return z["x"], z["sensor0"], z["y"], z["ysensor"]

    base = load_base_decoder()
    adapt = adapter()
    xs, s0s, ys, sys = [], [], [], []
    # Include light, medium, and saturated loads. No max-load scalar is exposed;
    # active/tracked counts are already present in the observation.
    initials = [5, 10, 15, 30, 50, 75, 100]
    rates = [0.0, 1.0, 2.0, 5.0, 10.0]
    seeds = [200, 201, 202, 203]
    for init in initials:
        for rate in rates:
            env = xband_env(rate)
            # The teacher threshold is intentionally load-dependent during data
            # generation; the student has to infer the regime from observation.
            threshold = 0.85 if init < 75 else 0.75
            teacher = SequenceDirectPlanner(base, threshold=threshold, mode="branch", allow_retrack=False)
            for seed in seeds:
                seedall(seed)
                eng = build_env(teacher, init, MAXT, seed, 200, env)
                eng.reset(seed=seed)
                debt = 0.0
                for w in range(6):
                    obs0 = get_obs(eng, debt)
                    plan = teacher.plan(obs0, budget_ms=200)
                    y = np.full((seq_len,), -100, dtype=np.int64)
                    sy = np.full((seq_len,), -100, dtype=np.int64)
                    spent = 0.0
                    slot = 0
                    for a in plan:
                        if slot >= seq_len or spent >= 200.0 or eng.term_buf[0]:
                            break
                        before = get_obs(eng, debt)
                        ok, dt, debt = step_action(eng, int(a), debt)
                        if not ok:
                            continue
                        y[slot] = int(a)
                        sy[slot] = int(before.get("sensor_id", 0))
                        spent += dt
                        slot += 1
                    xs.append(tokenize(adapt, obs0, selected=set(), search_count=0))
                    s0s.append(int(obs0.get("sensor_id", 0)))
                    ys.append(y)
                    sys.append(sy)
                    if eng.term_buf[0]:
                        break
                eng.close()
                print("curriculum_data", len(xs), flush=True)
    x = np.stack(xs).astype(np.float32)
    s0 = np.asarray(s0s, dtype=np.int64)
    y = np.stack(ys).astype(np.int64)
    sy = np.stack(sys).astype(np.int64)
    np.savez_compressed(path, x=x, sensor0=s0, y=y, ysensor=sy)
    return x, s0, y, sy


def train():
    ckpt = OUT / "xs_curriculum_sensor_batch.pt"
    if ckpt.exists():
        m = SensorFactorizedBatchDecoder().to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        return m.eval()
    x, s0, y, sy = collect_data()
    model = SensorFactorizedBatchDecoder().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    x_t = torch.from_numpy(x).to(DEVICE)
    s0_t = torch.from_numpy(s0).to(DEVICE)
    y_t = torch.from_numpy(y).to(DEVICE)
    sy_t = torch.from_numpy(sy).to(DEVICE)
    log = []
    for step in range(700):
        idx = torch.randint(0, x_t.shape[0], (min(128, x_t.shape[0]),), device=DEVICE)
        tl, tr, sl = model(x_t[idx], s0_t[idx])
        yb = y_t[idx]
        syb = sy_t[idx]
        valid = yb >= 0
        type_loss = F.binary_cross_entropy_with_logits(tl[valid], (yb[valid] == 0).float())
        sensor_loss = F.cross_entropy(sl[valid], syb[valid])
        track_valid = valid & (yb > 0)
        track_loss = F.cross_entropy(tr[track_valid], yb[track_valid]) if bool(track_valid.any()) else torch.zeros((), device=DEVICE)
        loss = type_loss + track_loss + 0.25 * sensor_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 699:
            row = dict(step=step, loss=float(loss.detach().cpu()), type_loss=float(type_loss.detach().cpu()), track_loss=float(track_loss.detach().cpu()), sensor_loss=float(sensor_loss.detach().cpu()))
            log.append(row)
            print("curriculum_train", row, flush=True)
    pd.DataFrame(log).to_csv(OUT / "xs_curriculum_train_log.csv", index=False)
    torch.save(model.state_dict(), ckpt)
    return model.eval()


def evaluate(model):
    raw_path = OUT / "xs_curriculum_eval_raw.csv"
    win_path = OUT / "xs_curriculum_eval_windows.csv"
    if raw_path.exists() and win_path.exists():
        return pd.read_csv(raw_path), pd.read_csv(win_path)
    base = load_base_decoder()
    rows, wins = [], []
    cells = [(15, 0.0), (15, 2.0), (50, 2.0), (75, 2.0), (75, 5.0), (100, 2.0), (100, 5.0), (100, 10.0)]
    for init, rate in cells:
        env = xband_env(rate)
        methods = {
            "EDF": lambda: EDFPlanner(MAXT),
            "EST": lambda: ESTPlanner(MAXT),
            "LegacyQ_r8": lambda env=env: make_base_q(env, 8),
            "LegacyPolicy_r8": lambda env=env: make_legacy_policy(env, 8, 16),
            "BatchDecoder_t0.75": lambda: SequenceDirectPlanner(base, threshold=0.75, mode="branch", allow_retrack=False),
            "BatchDecoder_t0.85": lambda: SequenceDirectPlanner(base, threshold=0.85, mode="branch", allow_retrack=False),
            "XSCurriculum_t0.65": lambda: SensorBatchPlanner(model, threshold=0.65, x_dwell_cap=None, allow_retrack=False),
            "XSCurriculum_t0.75": lambda: SensorBatchPlanner(model, threshold=0.75, x_dwell_cap=None, allow_retrack=False),
            "XSCurriculum_t0.85": lambda: SensorBatchPlanner(model, threshold=0.85, x_dwell_cap=None, allow_retrack=False),
        }
        for name, factory in methods.items():
            seedall(61)
            t0 = time.perf_counter()
            w, _ = run_fixed(factory(), name, init, MAXT, 61, 60, 200, env)
            s = summarize_window_df(w, "fixed")
            s.update(planner=name, initial_targets=init, rate=rate, seed=61, wall_s=time.perf_counter() - t0)
            rows.append(s)
            ww = w.copy()
            ww["planner"] = name
            ww["initial_targets"] = init
            ww["rate"] = rate
            ww["seed"] = 61
            wins.append(ww)
            print("curriculum_eval", init, rate, name, round(s["reward_per_200ms_eq"], 3), round(s["mean_drop_pct_active"], 2), round(s["mean_delay_active"], 1), round(s["search_fraction"], 2), round(s["planning_ms_per_200ms_eq"], 2), flush=True)
    raw = pd.DataFrame(rows)
    win = pd.concat(wins, ignore_index=True)
    raw.to_csv(raw_path, index=False)
    win.to_csv(win_path, index=False)
    return raw, win


def plot(raw: pd.DataFrame, win: pd.DataFrame):
    summary = raw.groupby("planner").agg(
        reward_per_200ms=("reward_per_200ms_eq", "mean"),
        drop_pct=("mean_drop_pct_active", "mean"),
        avg_delay_ms=("mean_delay_active", "mean"),
        tracked=("mean_tracked_targets", "mean"),
        active=("mean_active_targets", "mean"),
        search_fraction=("search_fraction", "mean"),
        latency_ms_per_200=("planning_ms_per_200ms_eq", "mean"),
    ).reset_index().sort_values("reward_per_200ms", ascending=False)
    summary.to_csv(OUT / "xs_curriculum_eval_summary.csv", index=False)
    keep = summary.planner.tolist()
    piv = win.pivot_table(index="elapsed_ms", columns=["planner", "initial_targets", "rate", "seed"], values="cumulative_reward", aggfunc="last").sort_index()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    for name in keep:
        cols = [c for c in piv.columns if c[0] == name]
        if cols:
            ax.plot(piv.index / 1000.0, piv[cols].mean(axis=1), label=name, linewidth=2.2 if name.startswith("XS") or name.startswith("Batch") else 1.5)
    ax.set_title("X/S High-Load Curriculum: Cumulative Reward")
    ax.set_xlabel("episode time (s)")
    ax.set_ylabel("mean cumulative reward")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = OUT / "xs_curriculum_cumulative.png"
    fig.savefig(p)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=180, sharex=True)
    panel_cells = [(50, 2.0), (75, 5.0), (100, 2.0), (100, 10.0)]
    for ax, (init, rate) in zip(axes.flat, panel_cells):
        sub = win[(win.initial_targets == init) & (win.rate == rate)]
        for name in keep[:7]:
            s = sub[sub.planner == name].sort_values("elapsed_ms")
            if not s.empty:
                ax.plot(s.elapsed_ms / 1000.0, s.cumulative_reward, label=name, linewidth=1.5)
        ax.set_title(f"init={init}, rate={rate}/s")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=6, ncol=2)
    for ax in axes[:, 0]:
        ax.set_ylabel("cumulative reward")
    for ax in axes[-1, :]:
        ax.set_xlabel("episode time (s)")
    fig.tight_layout()
    pp = OUT / "xs_curriculum_cumulative_highload_by_cell.png"
    fig.savefig(pp)
    plt.close(fig)
    for src in [p, pp, OUT / "xs_curriculum_eval_summary.csv"]:
        shutil.copy2(src, CLEAN / src.name)
    return summary, p, pp


def main():
    model = train()
    raw, win = evaluate(model)
    summary, p, pp = plot(raw, win)
    print(summary.to_string(index=False))
    print("OUT", OUT.resolve())
    print("PLOT", p.resolve())
    print("PANEL", pp.resolve())
    print("CLEAN", (CLEAN / p.name).resolve())


if __name__ == "__main__":
    main()
