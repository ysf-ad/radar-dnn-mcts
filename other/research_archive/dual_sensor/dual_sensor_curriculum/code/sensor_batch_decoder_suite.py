from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from balanced_q_retrain import make_env
from final_radar_campaign import MAXT, build_env, get_obs, run_fixed, seedall, summarize_window_df
from mutual_features import TOKEN_DIM, tokenize
from realistic_reward_retrain import adapter
from refresh_method_suite import make_base_q, make_legacy_policy
from repaired_campaign_tools import EDFPlanner, ESTPlanner, infer_elapsed_ms
from sequence_decoder_experiment import ParallelSequenceDecoder, SequenceDirectPlanner
from strict_window_report import SEARCH_DWELL_MS


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CreateValid1" / "results" / "sensor_batch_decoder_suite"
OUT.mkdir(parents=True, exist_ok=True)
CLEAN = Path(r"C:\Users\yousi\Downloads\radar_outputs")
CLEAN.mkdir(parents=True, exist_ok=True)
EXT = ROOT / "CreateValid1" / "results" / "extended_refresh_suite"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)


class SensorFactorizedBatchDecoder(nn.Module):
    """One-pass factorized decoder with an auxiliary sensor head.

    The C environment still accepts one action id: 0=search, 1..N=track.
    The extra sensor head therefore does not force S/X; it conditions and
    regularizes the decoder so the sequence can learn the current free sensor
    context and avoid X-band-invalid targets when a soft dwell mask is used.
    """

    def __init__(self, seq_len: int = 32, token_dim: int = TOKEN_DIM, d_model: int = 112, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.seq_len = int(seq_len)
        self.token_proj = nn.Linear(token_dim, d_model)
        self.sensor_embed = nn.Embedding(2, d_model)
        self.cls_token = nn.Parameter(torch.randn(d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.05,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=nlayers, enable_nested_tensor=False, mask_check=False)
        self.pos = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        self.type_head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.sensor_head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        self.track_head = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, tokens: torch.Tensor, sensor_id: torch.Tensor):
        token_active = tokens[:, :, 4] > 0.5
        token_active[:, 0] = True
        sid = sensor_id.long().clamp(0, 1)

        emb = self.token_proj(tokens)
        cls = self.cls_token[None, None, :].expand(tokens.shape[0], 1, -1) + self.sensor_embed(sid)[:, None, :]
        emb = torch.cat([cls, emb], dim=1)
        valid = torch.cat([torch.ones((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device), token_active], dim=1)
        out = self.encoder(emb, src_key_padding_mask=~valid)
        cls_out = out[:, 0, :]
        tok_out = out[:, 1:, :]

        pos = self.pos[None, :, :].expand(tokens.shape[0], -1, -1)
        cls_seq = cls_out[:, None, :].expand(-1, self.seq_len, -1)
        seq_feat = torch.cat([cls_seq, pos], dim=-1)
        type_logits = self.type_head(seq_feat).squeeze(-1)
        sensor_logits = self.sensor_head(seq_feat)

        tok = tok_out[:, None, :, :].expand(-1, self.seq_len, -1, -1)
        cls_rep = cls_out[:, None, None, :].expand(-1, self.seq_len, tok_out.shape[1], -1)
        pos_rep = pos[:, :, None, :].expand(-1, self.seq_len, tok_out.shape[1], -1)
        track_logits = self.track_head(torch.cat([tok, cls_rep, pos_rep], dim=-1)).squeeze(-1)
        track_mask = token_active.clone()
        track_mask[:, 0] = False
        track_logits = track_logits.masked_fill(~track_mask[:, None, :], -1e9)
        return type_logits, track_logits, sensor_logits


class SensorBatchPlanner:
    def __init__(
        self,
        model: SensorFactorizedBatchDecoder,
        threshold: float = 0.5,
        x_dwell_cap: float | None = None,
        allow_retrack: bool = False,
    ):
        self.model = model.eval()
        self.threshold = float(threshold)
        self.x_dwell_cap = None if x_dwell_cap is None else float(x_dwell_cap)
        self.allow_retrack = bool(allow_retrack)
        self.adapt = adapter()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def warmup(self, obs, budget_ms=200):
        _ = self.plan(obs, budget_ms)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def plan(self, obs, budget_ms=200):
        tokens_np = tokenize(self.adapt, obs, selected=set(), search_count=0)
        sensor_np = np.asarray([int(obs.get("sensor_id", 0))], dtype=np.int64)
        with torch.inference_mode():
            tokens = torch.from_numpy(tokens_np).float()[None, :, :].to(self.device)
            sid = torch.from_numpy(sensor_np).to(self.device)
            type_logits, track_logits, sensor_logits = self.model(tokens, sid)
            p_search = torch.sigmoid(type_logits[0])
            scores = track_logits[0].clone()
            pred_sensor = torch.argmax(sensor_logits[0], dim=-1)
            if self.x_dwell_cap is not None:
                dwell = torch.as_tensor(obs["t_dwell"], device=self.device, dtype=scores.dtype)
                x_invalid = torch.zeros((scores.shape[1],), dtype=torch.bool, device=self.device)
                x_invalid[1 : 1 + dwell.numel()] = dwell > self.x_dwell_cap
                # Token 0 is search. Target action a maps to token a.
                rows = pred_sensor == 1
                scores[rows[:, None] & x_invalid[None, :]] = -1e9
            best_track = torch.argmax(scores, dim=-1)
            choose_search = p_search >= self.threshold
            actions = torch.where(choose_search, torch.zeros_like(best_track), best_track).cpu().numpy().astype(int)
            scores_np = scores.detach().cpu().numpy()

        if not self.allow_retrack:
            used = set()
            for i, a in enumerate(actions.tolist()):
                if a <= 0:
                    continue
                if a not in used:
                    used.add(int(a))
                    continue
                row = scores_np[i].copy()
                row[0] = -1e9
                for u in used:
                    if 0 <= u < row.shape[0]:
                        row[u] = -1e9
                repl = int(np.argmax(row))
                actions[i] = repl if row[repl] > -1e8 else 0
                if actions[i] > 0:
                    used.add(int(actions[i]))
        return [int(a) for a in actions]


def xband_env(rate: float):
    env = make_env(rate, refresh=1, strict=False)
    env["enable_x_band"] = 1
    return env


def teacher_factory(name: str, env_cfg) -> Callable[[], object]:
    if name == "EST":
        return lambda: ESTPlanner(MAXT)
    if name == "EDF":
        return lambda: EDFPlanner(MAXT)
    if name == "LegacyPolicy":
        return lambda: make_legacy_policy(env_cfg, 8, 16)
    raise ValueError(name)


def step_labelled_action(eng, action: int, search_debt_ms: float):
    from pufferlib.ocean.radarxs import binding

    obs_before = get_obs(eng, search_debt_ms)
    if action != 0:
        idx = int(action) - 1
        if idx < 0 or idx >= len(obs_before["active_mask"]) or not obs_before["active_mask"][idx] or obs_before["t_deadline"][idx] < 0:
            return False, 0.0, search_debt_ms
    eng.act_buf[0] = int(action)
    binding.vec_step(eng.env)
    obs_after = get_obs(eng, search_debt_ms)
    if action == 0:
        dt = SEARCH_DWELL_MS
        search_debt_ms = 0.0
    else:
        dt = infer_elapsed_ms(obs_before, obs_after)
        search_debt_ms += max(float(dt), 0.0)
    return True, float(dt), search_debt_ms


def collect_teacher_data(teacher_name: str, seq_len: int = 32):
    path = OUT / f"sensor_teacher_{teacher_name}_seq{seq_len}.npz"
    if path.exists():
        z = np.load(path)
        return z["x"], z["sensor0"], z["y"], z["ysensor"]

    adapt = adapter()
    xs, sensor0, ys, ysensor = [], [], [], []
    for init in [5, 15, 30, 50, 75]:
        for rate in [0.0, 1.0, 2.0, 5.0]:
            env = xband_env(rate)
            for seed in [100, 101, 102]:
                seedall(seed)
                teacher = teacher_factory(teacher_name, env)()
                eng = build_env(teacher, init, MAXT, seed, 200, env)
                eng.reset(seed=seed)
                debt = 0.0
                for w in range(8):
                    if eng.term_buf[0]:
                        break
                    obs0 = get_obs(eng, debt)
                    plan = teacher.plan(obs0, budget_ms=200)
                    y = np.full((seq_len,), -100, dtype=np.int64)
                    sy = np.full((seq_len,), -100, dtype=np.int64)
                    spent = 0.0
                    slot = 0
                    for action in plan:
                        if slot >= seq_len or spent >= 200.0 or eng.term_buf[0]:
                            break
                        before = get_obs(eng, debt)
                        ok, dt, debt_next = step_labelled_action(eng, int(action), debt)
                        if not ok:
                            continue
                        y[slot] = int(action)
                        sy[slot] = int(before.get("sensor_id", 0))
                        debt = debt_next
                        spent += dt
                        slot += 1
                    xs.append(tokenize(adapt, obs0, selected=set(), search_count=0))
                    sensor0.append(int(obs0.get("sensor_id", 0)))
                    ys.append(y)
                    ysensor.append(sy)
                eng.close()
                print("sensor_data", teacher_name, len(xs), flush=True)
    x = np.stack(xs).astype(np.float32)
    s0 = np.asarray(sensor0, dtype=np.int64)
    y = np.stack(ys).astype(np.int64)
    sy = np.stack(ysensor).astype(np.int64)
    np.savez_compressed(path, x=x, sensor0=s0, y=y, ysensor=sy)
    return x, s0, y, sy


def train_sensor_decoder(teacher_name: str):
    ckpt = OUT / f"sensor_batch_{teacher_name}.pt"
    if ckpt.exists():
        model = SensorFactorizedBatchDecoder().to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        return model.eval()

    x, s0, y, sy = collect_teacher_data(teacher_name)
    model = SensorFactorizedBatchDecoder().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    x_t = torch.from_numpy(x).to(DEVICE)
    s0_t = torch.from_numpy(s0).to(DEVICE)
    y_t = torch.from_numpy(y).to(DEVICE)
    sy_t = torch.from_numpy(sy).to(DEVICE)
    log = []
    for step in range(450):
        idx = torch.randint(0, x_t.shape[0], (min(96, x_t.shape[0]),), device=DEVICE)
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
        if step % 75 == 0 or step == 449:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "type_loss": float(type_loss.detach().cpu()),
                "track_loss": float(track_loss.detach().cpu()),
                "sensor_loss": float(sensor_loss.detach().cpu()),
            }
            print("sensor_train", teacher_name, row, flush=True)
            log.append(row)
    pd.DataFrame(log).to_csv(OUT / f"sensor_batch_{teacher_name}_train_log.csv", index=False)
    torch.save(model.state_dict(), ckpt)
    return model.eval()


def load_base_batch_decoder():
    ckpt = EXT / "batch_factorized_decoder.pt"
    model = ParallelSequenceDecoder(seq_len=32).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return model.eval()


def evaluate():
    raw_path = OUT / "xs_factorized_eval_raw.csv"
    win_path = OUT / "xs_factorized_eval_windows.csv"
    if raw_path.exists() and win_path.exists():
        return pd.read_csv(raw_path), pd.read_csv(win_path)

    base_batch = load_base_batch_decoder()
    sensor_models = {name: train_sensor_decoder(name) for name in ["EDF", "EST", "LegacyPolicy"]}
    rows, wins = [], []
    for init, rate in [(15, 0.0), (15, 2.0), (50, 0.0), (50, 2.0)]:
        env = xband_env(rate)
        methods: dict[str, Callable[[], object]] = {
            "EDF": lambda env=env: EDFPlanner(MAXT),
            "EST": lambda env=env: ESTPlanner(MAXT),
            "LegacyPolicy_r8": lambda env=env: make_legacy_policy(env, 8, 16),
            "LegacyQ_r8": lambda env=env: make_base_q(env, 8),
            "BatchDecoder_t0.50": lambda: SequenceDirectPlanner(base_batch, threshold=0.5, mode="branch", allow_retrack=False),
        }
        for teacher_name, model in sensor_models.items():
            for thresh in [0.35, 0.50]:
                methods[f"XSBatch_{teacher_name}_t{thresh:.2f}"] = (
                    lambda model=model, thresh=thresh: SensorBatchPlanner(model, threshold=thresh, x_dwell_cap=None, allow_retrack=False)
                )
            methods[f"XSBatch_{teacher_name}_t0.50_xmask120"] = (
                lambda model=model: SensorBatchPlanner(model, threshold=0.50, x_dwell_cap=120.0, allow_retrack=False)
            )
        for name, factory in methods.items():
            seedall(50)
            t0 = time.perf_counter()
            w, _ = run_fixed(factory(), name, init, MAXT, 50, 50, 200, env)
            s = summarize_window_df(w, "fixed")
            s.update(planner=name, initial_targets=init, rate=rate, seed=50, wall_s=time.perf_counter() - t0)
            rows.append(s)
            ww = w.copy()
            ww["planner"] = name
            ww["initial_targets"] = init
            ww["rate"] = rate
            wins.append(ww)
            print("xs_eval", init, rate, name, round(s["reward_per_200ms_eq"], 3), round(s["planning_ms_per_200ms_eq"], 2), flush=True)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            pd.concat(wins, ignore_index=True).to_csv(win_path, index=False)
    return pd.DataFrame(rows), pd.concat(wins, ignore_index=True)


def plot_outputs(raw: pd.DataFrame, wins: pd.DataFrame):
    summary = raw.groupby("planner").agg(
        reward_per_200ms=("reward_per_200ms_eq", "mean"),
        final_cumulative=("final_cumulative_reward", "mean"),
        drop_pct=("mean_drop_pct_active", "mean"),
        avg_delay_ms=("mean_delay_active", "mean"),
        tracked=("mean_tracked_targets", "mean"),
        active=("mean_active_targets", "mean"),
        search_fraction=("search_fraction", "mean"),
        latency_ms_per_200=("planning_ms_per_200ms_eq", "mean"),
    ).reset_index().sort_values("reward_per_200ms", ascending=False)
    summary.to_csv(OUT / "xs_factorized_eval_summary.csv", index=False)
    keep = ["EST", "EDF", "LegacyPolicy_r8", "LegacyQ_r8", "BatchDecoder_t0.50"] + summary[summary.planner.str.startswith("XSBatch")].head(4)["planner"].tolist()

    piv = wins[wins.planner.isin(keep)].pivot_table(
        index="elapsed_ms",
        columns=["planner", "initial_targets", "rate", "seed"],
        values="cumulative_reward",
        aggfunc="last",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    for name in keep:
        cols = [c for c in piv.columns if c[0] == name]
        if cols:
            ax.plot(piv.index / 1000.0, piv[cols].mean(axis=1), label=name, linewidth=2.0 if name.startswith("XSBatch") else 1.6)
    ax.set_title("X/S Refresh-On Cumulative Reward: Sensor-Aware Factorized Batch Decoder")
    ax.set_xlabel("episode time (s)")
    ax.set_ylabel("mean cumulative reward")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = OUT / "xs_factorized_cumulative.png"
    fig.savefig(p)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=180, sharex=True)
    for ax, (init, rate) in zip(axes.flat, [(15, 0.0), (15, 2.0), (50, 0.0), (50, 2.0)]):
        sub = wins[(wins.initial_targets == init) & (wins.rate == rate) & (wins.planner.isin(keep))]
        for name in keep:
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
    pp = OUT / "xs_factorized_cumulative_by_load.png"
    fig.savefig(pp)
    plt.close(fig)

    for src in [p, pp, OUT / "xs_factorized_eval_summary.csv"]:
        shutil.copy2(src, CLEAN / src.name)
    return summary, p, pp


def main():
    raw, wins = evaluate()
    summary, p, pp = plot_outputs(raw, wins)
    print(summary.to_string(index=False))
    print("OUT", OUT.resolve())
    print("PLOT", p.resolve())
    print("PANEL", pp.resolve())
    print("CLEAN", (CLEAN / p.name).resolve())


if __name__ == "__main__":
    main()
