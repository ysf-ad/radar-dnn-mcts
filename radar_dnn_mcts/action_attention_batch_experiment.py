from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compare_action_heads_smoke import usable_targets
from exact_env_mutual import (
    MAXT,
    attach_env_obs,
    env_cfg_for,
    xs_decode_action,
    xs_s_search_action,
)
from final_radar_campaign import get_obs, run_fixed, summarize_window_df
from foundation_mcts_fair_eval import parse_floats, parse_ints, physical_candidates
from mutual_features import slot_features, tokenize
from penalty_window_quota_learner_eval import make_exact_args
from repaired_campaign_tools import EDFPlanner, ESTPlanner
from realistic_reward_retrain import adapter
from two_sensor_physical_head_eval import ActionAttentionFactorizedNet, PhysicalHeadPlanner, train_head


class CachedActionAttentionPlanner:
    """One-window planner that reuses the root token encoding.

    This is not the final architecture; it is a latency/reward probe for the
    core question: can the winning action-attention head produce good direct
    plans without exact rerank if we avoid re-encoding the same root state?
    """

    def __init__(
        self,
        model: ActionAttentionFactorizedNet,
        env_cfg: dict,
        policy_weight: float = 1.0,
        q_weight: float = 1.0,
        search_score_bias: float = 0.0,
        stateless_selected: bool = True,
    ):
        self.model = model.eval()
        self.env_cfg = dict(env_cfg)
        self.policy_weight = float(policy_weight)
        self.q_weight = float(q_weight)
        self.search_score_bias = float(search_score_bias)
        self.stateless_selected = bool(stateless_selected)
        self.adapt = adapter()

    def _scores_from_encoded(self, cls_out, tok_out, selected_t, token_active, slot_t):
        model = self.model
        slot_emb = model.backbone.slot_proj(slot_t)
        bsz, rows, _ = tok_out.shape

        sensor = model.sensor_embed[None, :, :].expand(bsz, -1, -1)
        cls_s = cls_out[:, None, :].expand(-1, 2, -1)
        slot_s = slot_emb[:, None, :].expand(-1, 2, -1)
        sensor_state = model.sensor_state_proj(torch.cat([cls_s, slot_s, sensor], dim=-1))
        coupled_sensor = model.sensor_coupler(sensor_state)
        type_ctx = torch.cat([cls_s, slot_s, coupled_sensor], dim=-1)
        type_logits = model.type_head(type_ctx)
        type_q = model.type_q_head(type_ctx)

        tok_st = tok_out[:, :, None, :].expand(-1, -1, 2, -1)
        cls_st = cls_out[:, None, None, :].expand(-1, rows, 2, -1)
        slot_st = slot_emb[:, None, None, :].expand(-1, rows, 2, -1)
        sensor_st = coupled_sensor[:, None, :, :].expand(bsz, rows, -1, -1)
        target_ctx = torch.cat([tok_st, cls_st, slot_st, sensor_st], dim=-1)
        target_logits = model.target_head(target_ctx).squeeze(-1)
        target_q = model.target_q_head(target_ctx).squeeze(-1)

        base_scores = slot_t.new_full((bsz, rows, 2), -1e9)
        base_q = slot_t.new_zeros((bsz, rows, 2))
        base_scores[:, 0, :] = type_logits[:, :, 0]
        base_q[:, 0, :] = type_q[:, :, 0]
        track_mask = token_active & ~selected_t
        track_mask[:, 0] = False
        base_scores[:, 1:, :] = (type_logits[:, None, :, 1] + target_logits)[:, 1:, :]
        base_q[:, 1:, :] = (type_q[:, None, :, 1] + target_q)[:, 1:, :]

        row_is_search = torch.arange(rows, device=slot_t.device)[None, :, None] == 0
        valid = (track_mask[:, :, None] | row_is_search).expand(-1, -1, 2)
        action_ctx = model.action_proj(target_ctx).reshape(bsz, rows * 2, -1)
        action_ctx = model.action_coupler(action_ctx, src_key_padding_mask=~valid.reshape(bsz, rows * 2))
        residual = model.action_policy_residual(action_ctx).reshape(bsz, rows, 2)
        q_residual = model.action_q_residual(action_ctx).reshape(bsz, rows, 2)
        scores = (base_scores + residual).masked_fill(~valid, -1e9)
        q = (base_q + q_residual).masked_fill(~valid, 0.0)
        return self.policy_weight * scores + self.q_weight * q

    def plan(self, obs, budget_ms=200):
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        root_tok = tokenize(self.adapt, obs, selected=set(), search_count=0).astype(np.float32)
        with torch.inference_mode():
            root_x = torch.from_numpy(root_tok).float().unsqueeze(0)
            cls_out, tok_out, root_selected, token_active = self.model.backbone.encode_tokens(root_x)

        selected: set[int] = set()
        plan: list[int] = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        while elapsed < float(budget_ms) and len(plan) < 64:
            slot = slot_features(obs, elapsed, search_count, track_count, last, float(budget_ms)).astype(np.float32)
            with torch.inference_mode():
                slot_t = torch.from_numpy(slot).float().unsqueeze(0)
                selected_t = root_selected.clone()
                if not self.stateless_selected:
                    for base in selected:
                        if 0 <= int(base) < selected_t.shape[1]:
                            selected_t[0, int(base)] = True
                score = self._scores_from_encoded(cls_out, tok_out, selected_t, token_active, slot_t).squeeze(0).cpu().numpy()
            score = np.asarray(score, dtype=np.float32).copy()
            if self.search_score_bias != 0.0:
                score[0, :] += self.search_score_bias
            if self.stateless_selected:
                for base in selected:
                    if 0 <= int(base) < score.shape[0]:
                        score[int(base), :] = -1e9

            best_action = None
            best_score = -np.inf
            for action in physical_candidates(obs, top_k=MAXT):
                base, sensor = xs_decode_action(int(action), MAXT)
                if int(base) < 0 or int(base) in selected:
                    continue
                sidx = 0 if sensor is None else int(sensor)
                val = float(score[int(base), sidx])
                if val > best_score:
                    best_action, best_score = int(action), val
            if best_action is None:
                break
            plan.append(best_action)
            base, _sensor = xs_decode_action(best_action, MAXT)
            if int(base) == 0:
                search_count += 1
                dt = 10.0
            else:
                selected.add(int(base))
                track_count += 1
                dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
                dt = float(dwell[int(base) - 1]) if int(base) - 1 < len(dwell) else 10.0
            elapsed += max(1.0, float(dt))
            last = int(base)
        return plan if plan else [xs_s_search_action(MAXT)]


def eval_planners(planner_factories: dict, args, exact_args) -> pd.DataFrame:
    rows = []
    for seed in parse_ints(args.eval_seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                env_cfg = env_cfg_for(float(rate), exact_args)
                env_cfg["enable_x_band"] = 1
                planners = {
                    "EDF": EDFPlanner(MAXT),
                    "EST": ESTPlanner(MAXT),
                    **{name: factory(env_cfg) for name, factory in planner_factories.items()},
                }
                for name, planner in planners.items():
                    t0 = time.perf_counter()
                    w, _ = run_fixed(planner, name, int(initial), MAXT, int(seed), int(args.eval_windows), 200, env_cfg)
                    seconds = time.perf_counter() - t0
                    s = summarize_window_df(w, "fixed")
                    row = {
                        "method": name,
                        "initial": int(initial),
                        "rate": float(rate),
                        "seed": int(seed),
                        "reward": float(s.get("reward_per_200ms_eq", np.nan)),
                        "search": float(s.get("search_fraction", np.nan)),
                        "latency_ms_window": float(1000.0 * seconds / max(1, int(args.eval_windows))),
                    }
                    rows.append(row)
                    print(row, flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-out", default="CreateValid1/results/gated_accepted_fullgrid_after_60r4_late_targets.pt")
    ap.add_argument("--out", default="CreateValid1/results/action_attention_batch_experiment.csv")
    ap.add_argument("--initials", default="20,40,60")
    ap.add_argument("--rates", default="2,3,4")
    ap.add_argument("--eval-seeds", default="916")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--eval-windows", type=int, default=20)
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--train-steps", type=int, default=90)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--q-loss-weight", type=float, default=0.25)
    ap.add_argument("--value-loss-weight", type=float, default=0.25)
    ap.add_argument("--log-every", type=int, default=45)
    ap.add_argument("--model-seed", type=int, default=123)
    ap.add_argument("--cell-balanced-sampling", action="store_true")
    ap.add_argument("--policy-score-weight", type=float, default=1.0)
    ap.add_argument("--q-score-weight", type=float, default=1.0)
    ap.add_argument("--search-score-bias", type=float, default=0.0)
    ap.add_argument("--search-score-biases", default="")
    args = ap.parse_args()

    torch.manual_seed(int(args.model_seed))
    np.random.seed(int(args.model_seed))
    torch.set_num_threads(1)
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = True
    exact_args.single_sensor = False

    targets = usable_targets(Path(args.targets_out))
    model = train_head("two_row_action_attention_qpolicy_factored_loss", targets, args, torch.device("cpu"))

    bias_values = parse_floats(str(args.search_score_biases)) if str(args.search_score_biases).strip() else [float(args.search_score_bias)]

    factories = {}
    for bias in bias_values:
        suffix = f"{float(bias):+g}".replace("+", "p").replace("-", "m").replace(".", "p")

        def sequential(env_cfg, bias=bias):
            return PhysicalHeadPlanner(
                model,
                "two_row_action_attention_qpolicy_factored_loss",
                env_cfg,
                policy_weight=float(args.policy_score_weight),
                q_weight=float(args.q_score_weight),
                search_score_bias=float(bias),
            )

        def cached_masked(env_cfg, bias=bias):
            return CachedActionAttentionPlanner(
                model,
                env_cfg,
                policy_weight=float(args.policy_score_weight),
                q_weight=float(args.q_score_weight),
                search_score_bias=float(bias),
                stateless_selected=True,
            )

        def cached_selected(env_cfg, bias=bias):
            return CachedActionAttentionPlanner(
                model,
                env_cfg,
                policy_weight=float(args.policy_score_weight),
                q_weight=float(args.q_score_weight),
                search_score_bias=float(bias),
                stateless_selected=False,
            )

        factories[f"AA_direct_sequential_sb{suffix}"] = sequential
        factories[f"AA_cached_root_masked_sb{suffix}"] = cached_masked
        factories[f"AA_cached_root_selected_sb{suffix}"] = cached_selected

    raw = eval_planners(factories, args, exact_args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out, index=False)
    summary = (
        raw.groupby("method")
        .agg(reward=("reward", "mean"), search=("search", "mean"), latency_ms_window=("latency_ms_window", "mean"), n=("reward", "size"))
        .reset_index()
        .sort_values("reward", ascending=False)
    )
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
