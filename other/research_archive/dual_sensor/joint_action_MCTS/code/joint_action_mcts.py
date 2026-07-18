from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from alphazero_orthodox import base_exact_args, load_state_into, parse_floats, parse_ints
from exact_env_mutual import (
    EDFPlanner,
    ESTPlanner,
    ExactEnvMCTS,
    ExactReplaySimulator,
    MAXT,
    attach_env_obs,
    build_env,
    env_cfg_for,
    get_obs,
    load_model,
    run_fixed,
    summarize_window_df,
    xs_decode_action,
)


@dataclass
class JointNode:
    seq: Tuple[int, ...]
    macro: Tuple[int, ...] = ()
    prior: float = 1.0
    parent: Optional["JointNode"] = None
    visits: int = 0
    total_value: float = 0.0
    edge_reward: float = 0.0
    edge_dt: float = 0.0
    raw_value: float = 0.0
    expanded: bool = False
    children: List["JointNode"] = field(default_factory=list)

    @property
    def mean_value(self) -> float:
        return self.total_value / max(1, self.visits)


class JointActionMCTS:
    """PUCT over macro-actions where one edge may schedule S and X together.

    The environment still executes atomic explicit S/X commands. The search
    object, however, chooses a joint macro `(s_action, x_action)` from one root
    state and scores the pair as one decision. This is the smallest faithful
    bridge toward AlphaZero-style two-sensor policy improvement.
    """

    def __init__(
        self,
        model,
        sim: ExactReplaySimulator,
        prefix: Sequence[int],
        exact_args,
        rollouts: int = 8,
        c_puct: float = 1.25,
        expand_top_k: int = 12,
        macro_depth: int = 4,
        per_sensor_top_k: int = 6,
        q_weight: float = 0.25,
        macro_q_weight: float = 0.10,
    ):
        self.model = model
        self.sim = sim
        self.prefix = tuple(int(a) for a in prefix)
        self.rollouts = int(rollouts)
        self.c_puct = float(c_puct)
        self.expand_top_k = int(expand_top_k)
        self.macro_depth = int(macro_depth)
        self.per_sensor_top_k = int(per_sensor_top_k)
        self.q_weight = float(q_weight)
        self.macro_q_weight = float(macro_q_weight)
        self.base = ExactEnvMCTS(
            model,
            sim,
            self.prefix,
            rollouts=0,
            c_puct=c_puct,
            expand_top_k=max(expand_top_k, per_sensor_top_k * 2),
            horizon_windows=max(1, macro_depth),
            rollout_policy="pq",
            prior_mode=str(exact_args.prior_mode),
            q_scale=float(exact_args.q_scale),
            prior_uniform_mix=float(exact_args.prior_uniform_mix),
            leaf_value_mix=float(exact_args.leaf_value_mix),
            head_mode=str(exact_args.head_mode),
            sensor_action_mode=str(exact_args.sensor_action_mode),
            disable_x_search=bool(exact_args.disable_x_search),
            canonical_search_only=bool(exact_args.canonical_search_only),
            prior_search_bias=float(exact_args.prior_search_bias),
        )

    def state(self, seq: Sequence[int]):
        return self.sim.replay([*self.prefix, *[int(a) for a in seq]])

    def _action_score(self, priors: np.ndarray, q: np.ndarray, action: int) -> float:
        p = max(1e-12, float(self.base._prior_for_action(priors, int(action))))
        qv = float(self.base._q_for_action(q, int(action))) / max(1e-6, float(self.base.q_scale))
        return math.log(p) + self.q_weight * qv

    def _best_order(self, seq: Tuple[int, ...], a: int, b: int) -> Tuple[int, ...]:
        """Use the simulator to choose the better atomic ordering for a joint pair."""
        first = tuple([int(a), int(b)])
        second = tuple([int(b), int(a)])
        s0 = self.state(seq)
        s1 = self.state((*seq, *first))
        s2 = self.state((*seq, *second))
        r1 = float(s1.reward - s0.reward)
        r2 = float(s2.reward - s0.reward)
        dt1 = float(s1.dt_ms - s0.dt_ms)
        dt2 = float(s2.dt_ms - s0.dt_ms)
        if dt2 <= 0.0:
            return first
        if dt1 <= 0.0:
            return second
        return second if r2 > r1 + 1e-9 else first

    def candidates(self, seq: Tuple[int, ...]) -> List[Tuple[Tuple[int, ...], float, float]]:
        st = self.state(seq)
        valid = [int(a) for a in self.base.valid_actions(st.obs)]
        if not valid:
            return []
        priors, _value, q, _x, _slot = self.base._net(st.obs, seq)
        ranked = sorted(valid, key=lambda a: self._action_score(priors, q, a), reverse=True)
        s_actions: List[int] = []
        x_actions: List[int] = []
        other: List[int] = []
        for action in ranked:
            base, sensor = xs_decode_action(int(action), MAXT)
            if sensor == 0:
                s_actions.append(int(action))
            elif sensor == 1:
                x_actions.append(int(action))
            else:
                other.append(int(action))
        s_actions = s_actions[: self.per_sensor_top_k]
        x_actions = x_actions[: self.per_sensor_top_k]
        macros: Dict[Tuple[int, ...], Tuple[float, float]] = {}

        def macro_q(macro: Tuple[int, ...]) -> float:
            vals = [float(self.base._q_for_action(q, int(a))) for a in macro]
            return float(sum(vals) / max(1, len(vals)))

        def add(macro: Tuple[int, ...], logp: float) -> None:
            if not macro:
                return
            key = tuple(int(a) for a in macro)
            qv = macro_q(key)
            old = macros.get(key)
            if old is None or float(logp) > old[0]:
                macros[key] = (float(logp), float(qv))

        for action in [*s_actions, *x_actions, *other[: self.per_sensor_top_k]]:
            add((int(action),), self._action_score(priors, q, int(action)))
        for sa in s_actions:
            s_base, _ = xs_decode_action(int(sa), MAXT)
            for xa in x_actions:
                x_base, _ = xs_decode_action(int(xa), MAXT)
                if int(s_base) > 0 and int(s_base) == int(x_base):
                    continue
                # Deterministic macro order keeps candidate generation cheap.
                # Both actions were valid at the same root state; if the second
                # becomes invalid after the first, replay skips it and the edge
                # value reflects that.
                macro = (int(sa), int(xa))
                add(macro, self._action_score(priors, q, int(sa)) + self._action_score(priors, q, int(xa)))
        if not macros:
            return []
        items = sorted(macros.items(), key=lambda kv: kv[1][0], reverse=True)[: self.expand_top_k]
        logits = np.asarray([v[0] for _, v in items], dtype=np.float64)
        logits -= float(np.max(logits))
        probs = np.exp(np.clip(logits, -60.0, 60.0))
        probs /= max(float(probs.sum()), 1e-12)
        return [(macro, float(p), float(v[1])) for (macro, v), p in zip(items, probs.tolist())]

    def expand(self, node: JointNode) -> None:
        st = self.state(node.seq)
        if st.terminal:
            node.expanded = True
            return
        priors, value, _q, _x, _slot = self.base._net(st.obs, node.seq)
        node.raw_value = float(value)
        children: List[JointNode] = []
        parent_state = self.state(node.seq)
        for macro, prior, macro_q in self.candidates(node.seq):
            child_seq = tuple([*node.seq, *macro])
            child_state = self.state(child_seq)
            dt = float(child_state.dt_ms - parent_state.dt_ms)
            if dt <= 0.0:
                continue
            child = JointNode(
                seq=child_seq,
                macro=tuple(macro),
                prior=float(prior),
                parent=node,
                edge_reward=float(child_state.reward - parent_state.reward),
                edge_dt=dt,
                total_value=float(self.macro_q_weight) * float(macro_q),
            )
            children.append(child)
        node.children = children
        node.expanded = True

    def select(self, node: JointNode) -> JointNode:
        total = math.sqrt(max(1, node.visits))
        best = None
        best_score = -float("inf")
        for child in node.children:
            q = child.edge_reward + child.mean_value
            u = self.c_puct * child.prior * total / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best = child
        return best if best is not None else node.children[0]

    def rollout_value(self, node: JointNode) -> float:
        seq = tuple(node.seq)
        start = self.state(seq)
        total = 0.0
        for _ in range(max(0, self.macro_depth - len(seq))):
            cands = self.candidates(seq)
            if not cands:
                break
            macro = cands[0][0]
            nxt = self.state((*seq, *macro))
            total += float(nxt.reward - self.state(seq).reward)
            seq = tuple([*seq, *macro])
            if nxt.terminal:
                break
        leaf = self.state(seq)
        if not leaf.terminal:
            _p, value, _q, _x, _slot = self.base._net(leaf.obs, seq)
            total += float(value)
        return total

    def backprop(self, node: JointNode, value: float) -> None:
        cur = node
        v = float(value)
        while cur is not None:
            cur.visits += 1
            cur.total_value += v
            v += cur.edge_reward
            cur = cur.parent

    def run(self) -> JointNode:
        root = JointNode(seq=())
        self.expand(root)
        for _ in range(self.rollouts):
            node = root
            depth = 0
            while node.expanded and node.children and depth < self.macro_depth:
                node = self.select(node)
                depth += 1
            if not node.expanded:
                self.expand(node)
            value = self.rollout_value(node)
            self.backprop(node, value)
        return root


class JointMCTSPlanner:
    def __init__(self, model, exact_args, rollouts: int, expand_top_k: int, macro_depth: int, per_sensor_top_k: int):
        self.model = model
        self.exact_args = exact_args
        self.rollouts = int(rollouts)
        self.expand_top_k = int(expand_top_k)
        self.macro_depth = int(macro_depth)
        self.per_sensor_top_k = int(per_sensor_top_k)
        self.sim: Optional[SnapshotSimulator] = None

    def plan(self, obs, budget_ms=200):
        if self.sim is None:
            return []
        mcts = JointActionMCTS(
            self.model,
            self.sim,
            self.sim.history,
            self.exact_args,
            rollouts=self.rollouts,
            expand_top_k=self.expand_top_k,
            macro_depth=self.macro_depth,
            per_sensor_top_k=self.per_sensor_top_k,
        )
        root = mcts.run()
        if not root.children:
            return []
        best = max(root.children, key=lambda c: (c.visits, c.edge_reward + c.mean_value, c.prior))
        return list(best.macro)


def run_joint_episode(
    model,
    exact_args,
    initial: int,
    rate: float,
    seed: int,
    windows: int,
    rollouts: int,
    expand_top_k: int,
    macro_depth: int,
    per_sensor_top_k: int,
    macro_q_weight: float,
) -> pd.DataFrame:
    env_cfg = env_cfg_for(float(rate), exact_args)
    sim = ExactReplaySimulator(int(initial), int(seed), env_cfg, MAXT)
    history: List[int] = []
    rows: List[dict] = []
    cumulative = 0.0
    for window in range(int(windows)):
        window_reward = 0.0
        window_ms = 0.0
        window_actions: List[int] = []
        while window_ms < 200.0:
            mcts = JointActionMCTS(
                model,
                sim,
                history,
                exact_args,
                rollouts=rollouts,
                expand_top_k=expand_top_k,
                macro_depth=macro_depth,
                per_sensor_top_k=per_sensor_top_k,
                macro_q_weight=macro_q_weight,
            )
            root = mcts.run()
            if not root.children:
                break
            best = max(root.children, key=lambda c: (c.visits, c.edge_reward + c.mean_value, c.prior))
            if not best.macro:
                break
            before = sim.replay(history)
            after = sim.replay([*history, *best.macro])
            dr = float(after.reward - before.reward)
            dt = float(after.dt_ms - before.dt_ms)
            if dt <= 0.0:
                break
            remaining = 200.0 - window_ms
            if dt > remaining + 1e-6:
                # Fall back to the first atomic action if the full macro would
                # overrun the window. This keeps the runner honest about budget.
                macro = (int(best.macro[0]),)
                after = sim.replay([*history, *macro])
                dr = float(after.reward - before.reward)
                dt = float(after.dt_ms - before.dt_ms)
                if dt <= 0.0 or dt > remaining + 1e-6:
                    break
            else:
                macro = tuple(int(a) for a in best.macro)
            history.extend(macro)
            window_actions.extend(macro)
            window_reward += dr
            window_ms += dt
            if after.terminal:
                break
        cumulative += float(window_reward)
        st = sim.replay(history)
        obs = st.obs
        active = np.asarray(obs["active_mask"]).astype(bool)
        deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
        tracked = active & (deadline >= 0.0)
        dropped = active & (deadline < 0.0)
        rows.append(
            {
                "window": int(window + 1),
                "window_reward": float(window_reward),
                "cumulative_reward": float(cumulative),
                "window_ms_used": float(window_ms),
                "search_fraction": float(np.mean([xs_decode_action(a, MAXT)[0] == 0 for a in window_actions])) if window_actions else 0.0,
                "tracked_targets": float(np.sum(tracked)),
                "drop_pct_active": float(100.0 * np.sum(dropped) / max(1, int(np.sum(active)))),
            }
        )
        if st.terminal:
            break
    return pd.DataFrame(rows)


def summarize_df(df: pd.DataFrame) -> dict:
    return {
        "reward": float(df["window_reward"].mean()) if "window_reward" in df and not df.empty else 0.0,
        "total_reward": float(df["window_reward"].sum()) if "window_reward" in df else 0.0,
        "search": float(df["search_fraction"].mean()) if "search_fraction" in df and not df.empty else 0.0,
        "windows_completed": int(len(df)),
        "mean_tracked_targets": float(df["tracked_targets"].mean()) if "tracked_targets" in df and not df.empty else 0.0,
        "mean_drop_pct_active": float(df["drop_pct_active"].mean()) if "drop_pct_active" in df and not df.empty else 0.0,
        "final_cumulative_reward": float(df["cumulative_reward"].iloc[-1]) if "cumulative_reward" in df and not df.empty else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", default=str(Path("CreateValid1/results/joint_action_mcts_smoke.csv")))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--initials", default="60")
    ap.add_argument("--rates", default="3")
    ap.add_argument("--seeds", default="916")
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--expand-top-k", type=int, default=10)
    ap.add_argument("--macro-depth", type=int, default=4)
    ap.add_argument("--per-sensor-top-k", type=int, default=5)
    ap.add_argument("--macro-q-weight", type=float, default=0.10)
    ap.add_argument("--env-mode", default="mcts_sched_v1")
    ap.add_argument("--track-loss-penalty", type=float, default=8.0)
    ap.add_argument("--target-service-weight", type=float, default=10.0)
    ap.add_argument("--target-service-horizon-ms", type=float, default=3000.0)
    ap.add_argument("--sector-staleness-weight", type=float, default=0.01)
    ap.add_argument("--search-frame-overdue-weight", type=float, default=0.2)
    ap.add_argument("--search-frame-drop-penalty", type=float, default=8.0)
    args = ap.parse_args()

    exact_args = base_exact_args(
        SimpleNamespace(
            ckpt=args.ckpt,
            device=args.device,
            head_arch="branch_context",
            windows=args.windows,
            max_targets_per_episode=1000000,
            rollouts=args.rollouts,
            c_puct=1.25,
            expand_top_k=args.expand_top_k,
            horizon_windows=max(1, args.macro_depth),
            prior_mode="physical_flat",
            prior_uniform_mix=0.03,
            root_dirichlet_alpha=0.3,
            root_dirichlet_frac=0.0,
            prior_search_bias=0.0,
            prior_q_beta=0.0,
            leaf_value_mix=1.0,
            head_mode="pq",
            q_utility_weight=0.0,
            q_utility_normalize=False,
            q_scale=100.0,
            self_play_sample_tau=0.0,
            sensor_action_mode="explicit_head",
            disable_x_search=False,
            canonical_search_only=False,
            env_mode=args.env_mode,
            enable_x_band=True,
            single_sensor=False,
            zero_action_rewards=False,
            track_loss_penalty=args.track_loss_penalty,
            target_service_weight=args.target_service_weight,
            target_service_horizon_ms=args.target_service_horizon_ms,
            sector_staleness_weight=args.sector_staleness_weight,
            search_frame_overdue_weight=args.search_frame_overdue_weight,
            search_frame_drop_penalty=args.search_frame_drop_penalty,
            use_arrival_feature=False,
            use_grid_feature=False,
            gamma=0.99,
        )
    )
    device = torch.device(args.device)
    model = load_model(exact_args).to(device)
    load_state_into(model, args.state, device)
    model.eval()

    rows = []
    for seed in parse_ints(args.seeds):
        for init in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                env_cfg = env_cfg_for(float(rate), exact_args)
                for name, planner in [("EDF", EDFPlanner(MAXT)), ("EST", ESTPlanner(MAXT))]:
                    t0 = time.perf_counter()
                    df, _ = run_fixed(planner, name, int(init), MAXT, int(seed), int(args.windows), 200, env_cfg)
                    row = {"method": name, "initial": init, "rate": rate, "seed": seed, **summarize_df(df)}
                    row["wall_seconds"] = time.perf_counter() - t0
                    rows.append(row)
                    print(row, flush=True)
                t0 = time.perf_counter()
                df = run_joint_episode(
                    model,
                    exact_args,
                    int(init),
                    float(rate),
                    int(seed),
                    int(args.windows),
                    int(args.rollouts),
                    int(args.expand_top_k),
                    int(args.macro_depth),
                    int(args.per_sensor_top_k),
                    float(args.macro_q_weight),
                )
                row = {"method": "joint_mcts", "initial": init, "rate": rate, "seed": seed, **summarize_df(df)}
                row["wall_seconds"] = time.perf_counter() - t0
                rows.append(row)
                print(row, flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).sort_values("reward", ascending=False).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
