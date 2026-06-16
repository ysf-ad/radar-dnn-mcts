from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "radar_dnn_mcts"))


class Timer:
    def __init__(self):
        self.values = defaultdict(list)

    def time(self, key):
        timer = self

        class Ctx:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, exc_type, exc, tb):
                timer.values[key].append((time.perf_counter() - self.t0) * 1000.0)

        return Ctx()

    def summary(self):
        out = {}
        for key, vals in self.values.items():
            arr = np.asarray(vals, dtype=np.float64)
            out[key] = {
                "calls": int(arr.size),
                "total_ms": float(arr.sum()),
                "mean_ms": float(arr.mean()),
                "p50_ms": float(np.percentile(arr, 50)),
                "p90_ms": float(np.percentile(arr, 90)),
            }
        return dict(sorted(out.items(), key=lambda kv: kv[1]["total_ms"], reverse=True))


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--decisions", type=int, default=20)
    parser.add_argument("--initial-targets", type=int, default=40)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--out", type=Path, default=Path("profile_action_attention_steps.json"))
    args = parser.parse_args()

    from exact_env_mutual import attach_env_obs, xs_decode_action, xs_s_search_action
    from final_radar_campaign import get_obs
    from mutual_features import slot_features, tokenize
    from perf_fast_planner import select_best_action
    from realistic_reward_retrain import adapter
    from repaired_campaign_tools import EDFPlanner, build_env, env_preset_cfg
    from two_sensor_physical_head_eval import MAXT, ActionAttentionFactorizedNet, physical_candidates

    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)
    device = torch.device(args.device)

    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1
    eng = build_env(EDFPlanner(MAXT), args.initial_targets, MAXT, args.seed, 200, env_cfg)
    eng.reset(seed=args.seed)
    obs0 = get_obs(eng, 0.0)
    adapt = adapter()

    baseline_model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    fast_model = ActionAttentionFactorizedNet(48, 4, 2).eval().to(device)
    fast_model.load_state_dict(baseline_model.state_dict())

    def baseline_profile():
        timer = Timer()
        obs = attach_env_obs(obs0, env_cfg, True, True)
        selected = set()
        plan = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        for _ in range(int(args.decisions)):
            with timer.time("baseline_tokenize"):
                tok = tokenize(adapt, obs, selected=selected, search_count=int(search_count))
            with timer.time("baseline_slot_features"):
                slot = slot_features(obs, elapsed, search_count, track_count, last, 200.0)
            with timer.time("baseline_model_forward_scores"):
                with torch.inference_mode():
                    x = torch.from_numpy(tok).float().unsqueeze(0)
                    s = torch.from_numpy(slot).float().unsqueeze(0)
                    scores, q = baseline_model.forward_scores(x, s)
                    score = (scores + q).squeeze(0).cpu().numpy()
            score = np.asarray(score, dtype=np.float32).copy()
            with timer.time("baseline_physical_candidates"):
                cands = physical_candidates(obs, top_k=MAXT)
            with timer.time("baseline_python_candidate_select"):
                best_action = None
                best_score = -np.inf
                for action in cands:
                    base, sensor = xs_decode_action(int(action), MAXT)
                    if int(base) < 0:
                        continue
                    sidx = 0 if sensor is None else int(sensor)
                    val = float(score[int(base), sidx])
                    if val > best_score:
                        best_action, best_score = int(action), val
            if best_action is None:
                break
            plan.append(best_action)
            base, _ = xs_decode_action(best_action, MAXT)
            if int(base) == 0:
                search_count += 1
                dt = 10.0
            else:
                selected.add(int(base))
                track_count += 1
                dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
                dt = float(dwell[int(base) - 1]) if int(base) - 1 < len(dwell) else 10.0
            elapsed += max(1.0, dt)
            last = int(base)
        return plan if plan else [xs_s_search_action(MAXT)], timer.summary()

    def fast_profile():
        timer = Timer()
        obs = attach_env_obs(obs0, env_cfg, True, True)
        selected = set()
        plan = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        with timer.time("fast_root_tokenize"):
            root_tok = tokenize(adapt, obs, selected=set(), search_count=0)
        with timer.time("fast_root_encode_once"):
            with torch.inference_mode():
                root_x = torch.from_numpy(root_tok).to(device, dtype=torch.float32).unsqueeze(0)
                sync(device)
                t0 = time.perf_counter()
                cls_out, tok_out, selected_t, token_active = fast_model.backbone.encode_tokens(root_x)
                sync(device)
                timer.values["fast_root_encode_once_cuda_sync"].append((time.perf_counter() - t0) * 1000.0)
        from perf_fast_planner import FastActionAttentionPlanner

        helper = FastActionAttentionPlanner(fast_model, env_cfg, device=device)
        for _ in range(int(args.decisions)):
            with timer.time("fast_slot_features"):
                slot = slot_features(obs, elapsed, search_count, track_count, last, 200.0)
            with timer.time("fast_action_score_from_encoded"):
                with torch.inference_mode():
                    slot_t = torch.from_numpy(slot).to(device, dtype=torch.float32).unsqueeze(0)
                    sync(device)
                    t0 = time.perf_counter()
                    score_t = helper._combined_scores_from_encoded(cls_out, tok_out, selected_t, token_active, slot_t)
                    sync(device)
                    timer.values["fast_action_score_from_encoded_cuda_sync"].append((time.perf_counter() - t0) * 1000.0)
                    score = score_t.squeeze(0).float().cpu().numpy()
            with timer.time("fast_vectorized_candidate_select"):
                best_action = select_best_action(score, obs, selected=selected, max_trackers=MAXT)
            if best_action is None:
                break
            plan.append(best_action)
            base, _ = xs_decode_action(best_action, MAXT)
            if int(base) == 0:
                search_count += 1
                dt = 10.0
            else:
                selected.add(int(base))
                if 0 <= int(base) < selected_t.shape[1]:
                    selected_t[0, int(base)] = True
                track_count += 1
                dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
                dt = float(dwell[int(base) - 1]) if int(base) - 1 < len(dwell) else 10.0
            elapsed += max(1.0, dt)
            last = int(base)
        return plan if plan else [xs_s_search_action(MAXT)], timer.summary()

    baseline_plan, baseline = baseline_profile()
    fast_plan, fast = fast_profile()
    report = {
        "device": str(device),
        "decisions": int(args.decisions),
        "baseline_plan": [int(x) for x in baseline_plan],
        "fast_plan": [int(x) for x in fast_plan],
        "plans_match": [int(x) for x in baseline_plan] == [int(x) for x in fast_plan],
        "baseline": baseline,
        "fast": fast,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    eng.close()


if __name__ == "__main__":
    main()
