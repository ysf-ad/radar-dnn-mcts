from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "CreateValid1" / "experiments" / "code" / "model_code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from current_sonly_exact_puct import parse_floats, parse_ints  # noqa: E402
from canonical_scheduler_contract import add_canonical_reward_args  # noqa: E402
from eval_action_attention_muzero_g import run_receding_eval, summarize  # noqa: E402
from exact_env_mutual import (  # noqa: E402
    EDFPlanner,
    ESTPlanner,
    MAXT,
    attach_env_obs,
    engine_env_cfg,
    env_cfg_for,
    xs_decode_action,
    xs_s_search_action,
    xs_s_track_action,
)
from mutual_features import slot_features, tokenize  # noqa: E402
from penalty_window_quota_learner_eval import make_exact_args  # noqa: E402
from realistic_reward_retrain import adapter  # noqa: E402
from repaired_campaign_tools import execute_first_valid_action  # noqa: E402
from single_sensor_ar_action_attention import load_action_attention_model  # noqa: E402
from train_sonly_puct_action_attention import forward_sonly_scores  # noqa: E402


def action_from_row(row: int) -> int:
    return xs_s_search_action(MAXT) if int(row) <= 0 else xs_s_track_action(int(row), MAXT)


class SOnlyReencodeGraphModule(torch.nn.Module):
    def __init__(self, model, policy_weight: float, q_weight: float):
        super().__init__()
        self.model = model
        self.policy_weight = float(policy_weight)
        self.q_weight = float(q_weight)

    def forward(self, x, slot, selected):
        scores, q, valid = forward_sonly_scores(self.model, x, slot, selected)
        utility = (self.policy_weight * scores + self.q_weight * q).masked_fill(~valid, -1.0e9)
        return torch.argmax(utility, dim=1)


class CudaGraphSOnlyReencode:
    def __init__(self, model, device: torch.device, policy_weight: float, q_weight: float):
        self.device = device
        self.module = SOnlyReencodeGraphModule(model, policy_weight, q_weight).eval()
        self.x = torch.zeros((1, MAXT + 1, 13), dtype=torch.float32, device=device)
        self.slot = torch.zeros((1, 11), dtype=torch.float32, device=device)
        self.selected = torch.zeros((1, MAXT + 1), dtype=torch.bool, device=device)
        with torch.inference_mode():
            for _ in range(8):
                self.out = self.module(self.x, self.slot, self.selected)
            torch.cuda.synchronize(device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self.module(self.x, self.slot, self.selected)
            self.graph.replay()
            torch.cuda.synchronize(device)

    def __call__(self, x, slot, selected):
        with torch.inference_mode():
            self.x.copy_(x, non_blocking=True)
            self.slot.copy_(slot, non_blocking=True)
            self.selected.copy_(selected, non_blocking=True)
            self.graph.replay()
            return self.out

    def copy_from_cpu(self, x, slot, selected):
        with torch.inference_mode():
            self.x.copy_(x, non_blocking=True)
            self.slot.copy_(slot, non_blocking=True)
            self.selected.copy_(selected, non_blocking=True)
            self.graph.replay()
            return self.out


class SOnlyReencodeActionAttentionPlanner:
    def __init__(
        self,
        model_path: str | Path,
        variant: str,
        env_cfg: dict,
        *,
        device: str,
        policy_weight: float,
        q_weight: float,
        cuda_graph: bool = True,
        selected_token_context: bool = True,
    ) -> None:
        self.model = load_action_attention_model(Path(model_path), device, variant).to(device).eval()
        self.env_cfg = dict(env_cfg)
        self.device = torch.device(device)
        self.policy_weight = float(policy_weight)
        self.q_weight = float(q_weight)
        self.selected_token_context = bool(selected_token_context)
        self.adapt = adapter()
        self.selected: set[int] = set()
        self.elapsed = 0.0
        self.search_count = 0
        self.track_count = 0
        self.last = -1
        self._window_marker = None
        self._graph = (
            CudaGraphSOnlyReencode(self.model, self.device, self.policy_weight, self.q_weight)
            if bool(cuda_graph) and self.device.type == "cuda"
            else None
        )

    def plan(self, obs: dict, budget_ms: float = 200.0):
        # run_receding_eval calls plan once per executed action. Detect the
        # start of a new window from the full remaining budget.
        if float(budget_ms) >= 199.0:
            self.selected = set()
            self.elapsed = 0.0
            self.search_count = 0
            self.track_count = 0
            self.last = -1
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        token_selected = self.selected if self.selected_token_context else set()
        x_np = tokenize(self.adapt, obs, selected=token_selected, search_count=self.search_count).astype(np.float32)
        slot_np = slot_features(obs, self.elapsed, self.search_count, self.track_count, self.last, 200.0).astype(np.float32)
        selected_np = np.zeros((MAXT + 1,), dtype=np.bool_)
        for row in self.selected:
            if 1 <= int(row) <= MAXT:
                selected_np[int(row)] = True
        with torch.inference_mode():
            x_cpu = torch.from_numpy(x_np[None]).float()
            slot_cpu = torch.from_numpy(slot_np[None]).float()
            selected_cpu = torch.from_numpy(selected_np[None]).bool()
            if self._graph is not None:
                row = int(self._graph.copy_from_cpu(x_cpu, slot_cpu, selected_cpu).detach().cpu()[0])
            else:
                x = x_cpu.to(self.device)
                slot = slot_cpu.to(self.device)
                selected = selected_cpu.to(self.device)
                scores, q, valid = forward_sonly_scores(self.model, x, slot, selected)
                utility = self.policy_weight * scores + self.q_weight * q
                row = int(torch.argmax(utility[0]).detach().cpu())
                if not bool(valid[0, row].detach().cpu()):
                    row = 0
        # Maintain the planner-side within-window context. The environment will
        # still validate the action; this is only for next-step features/masks.
        if row <= 0:
            self.search_count += 1
            self.last = 0
            self.elapsed += 10.0
        else:
            dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
            dt = float(dwell[row - 1]) if row - 1 < len(dwell) else 10.0
            self.elapsed += max(1.0, dt)
            self.selected.add(row)
            self.track_count += 1
            self.last = row
        return [action_from_row(row)]


class OneStepDirectAdapter:
    def __init__(self, planner):
        self.planner = planner
        self._queue: list[int] = []
        self._selected: set[int] = set()

    def plan(self, obs: dict, budget_ms: float = 200.0):
        # EDF/EST return a complete window plan. Preserve that sequence across
        # receding-evaluator calls instead of recomputing and repeatedly taking
        # its first action, which can cause invalid repeats and early underfill.
        if float(budget_ms) >= 199.0:
            self._queue.clear()
            self._selected.clear()
        if not self._queue:
            plan = self.planner.choose(None, 0.0, obs) if hasattr(self.planner, "choose") else self.planner.plan(obs, budget_ms)
            if isinstance(plan, (list, tuple)):
                native_actions = [int(action) for action in plan]
            elif plan is not None:
                native_actions = [int(plan)]
            else:
                native_actions = []
            self._queue.extend(
                xs_s_search_action(MAXT) if action <= 0 else xs_s_track_action(action, MAXT)
                for action in native_actions
            )
        active = np.asarray(obs.get("active_mask", []), dtype=np.bool_)
        dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
        while self._queue:
            action = int(self._queue.pop(0))
            row, sensor = xs_decode_action(action, MAXT)
            if int(sensor or 0) != 0:
                continue
            if int(row) <= 0:
                if float(budget_ms) >= 10.0:
                    return [action]
                continue
            index = int(row) - 1
            if int(row) in self._selected or index < 0 or index >= len(active) or not bool(active[index]):
                continue
            dt = float(dwell[index]) if index < len(dwell) else 10.0
            if dt <= float(budget_ms) + 1.0e-6:
                self._selected.add(int(row))
                return [action]
        return [xs_s_search_action(MAXT)] if float(budget_ms) >= 10.0 else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-state", required=True)
    ap.add_argument("--variant", default="two_row_action_attention_qpolicy_factored_loss")
    ap.add_argument("--out", required=True)
    ap.add_argument("--initials", default="20,40,60")
    ap.add_argument("--rates", default="2,3,4")
    ap.add_argument("--seeds", default="916")
    ap.add_argument("--windows", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--q-weight", type=float, default=0.0)
    ap.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--selected-token-context", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-baselines", action="store_true")
    add_canonical_reward_args(ap)
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    exact_args = make_exact_args(args)
    all_windows = []
    summaries = []
    for initial in parse_ints(args.initials):
        for rate in parse_floats(args.rates):
            env_cfg = env_cfg_for(float(rate), exact_args)
            env_cfg["enable_x_band"] = 0
            for seed in parse_ints(args.seeds):
                planner = SOnlyReencodeActionAttentionPlanner(
                    args.base_state,
                    args.variant,
                    env_cfg,
                    device=args.device,
                    policy_weight=args.policy_weight,
                    q_weight=args.q_weight,
                    cuda_graph=bool(args.cuda_graph),
                    selected_token_context=bool(args.selected_token_context),
                )
                df, _actions = run_receding_eval(planner, "S-only full-reencode action-attention", int(initial), int(seed), int(args.windows), env_cfg)
                df["initial"] = int(initial)
                df["rate"] = float(rate)
                df["seed"] = int(seed)
                df["policy_weight"] = float(args.policy_weight)
                df["q_weight"] = float(args.q_weight)
                df["selected_token_context"] = bool(args.selected_token_context)
                df["checkpoint"] = str(Path(args.base_state).resolve())
                df["variant"] = str(args.variant)
                all_windows.append(df)
                row = {
                    "planner": "S-only full-reencode action-attention",
                    "initial": int(initial),
                    "rate": float(rate),
                    "seed": int(seed),
                    "policy_weight": float(args.policy_weight),
                    "q_weight": float(args.q_weight),
                    "selected_token_context": bool(args.selected_token_context),
                    "checkpoint": str(Path(args.base_state).resolve()),
                    "variant": str(args.variant),
                    **summarize(df),
                }
                summaries.append(row)
                print(row, flush=True)
                if bool(args.include_baselines):
                    for name, base_planner in [("EDF", EDFPlanner(MAXT)), ("EST", ESTPlanner(MAXT))]:
                        bdf, _ = run_receding_eval(OneStepDirectAdapter(base_planner), name, int(initial), int(seed), int(args.windows), env_cfg)
                        bdf["initial"] = int(initial)
                        bdf["rate"] = float(rate)
                        bdf["seed"] = int(seed)
                        all_windows.append(bdf)
                        brow = {"planner": name, "initial": int(initial), "rate": float(rate), "seed": int(seed), **summarize(bdf)}
                        summaries.append(brow)
                        print(brow, flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_windows, ignore_index=True).to_csv(out, index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print({"windows": str(out), "summary": str(out.with_name(out.stem + "_summary.csv"))})


if __name__ == "__main__":
    main()
