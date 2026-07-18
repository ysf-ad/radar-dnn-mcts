"""Phase 0: EDF Imitation Learning for PolicyOnlyTransformer.

Runs EDF planner on RadarXS episodes, collects (adapted_obs, EDF_action)
pairs, and trains the policy head via cross-entropy.  The resulting checkpoint
is used as a warm-start for Phase 1 iterative RL so MCTS has a meaningful
prior from day one.

Usage:
    python train_edf_imitation.py \\
        --out-dir CreateValid1/results/projectA_edf_pretrain_v1 \\
        --task-batch 20,20,50,100 \\
        --episodes 300 \\
        --epochs 50 \\
        --max-trackers 100

Then pass --base-ckpt <out-dir>/best.pt to train_radarxs_iterative_rl.py.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Ensure the local pufferlib tree is importable when running from its directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pufferlib.ocean.radarxs import binding, engine
from pufferlib.ocean.radarxs.models.edf import EDFPlanner
from pufferlib.ocean.radarxs.models.transformer_mcts_policy import (
    PolicyOnlyMCTSPlanner,
    PolicyOnlyTransformer,
)
from train_radarxs_iterative_rl import (
    ReplayBuffer,
    env_preset_cfg,
    get_obs,
    infer_elapsed_ms,
    parse_int_list,
)


def collect_edf_episode(adapter, edf, args, initial_targets: int, seed: int):
    """Run one episode with EDF and yield (adapted_obs, one_hot_label) pairs.

    Only produces examples for steps where EDF's chosen action is valid for the
    neural policy (search=0, or a *tracked* target).  Untracked targets are
    masked out in PolicyOnlyTransformer, so training on them would be noise.
    """
    eng = engine.RadarEngine(
        planner=edf,
        initial_targets=initial_targets,
        max_trackers=args.max_trackers,
        seed=seed,
        window_ms=args.window_ms,
        **args.env_cfg,
    )
    eng.reset(seed=seed)

    examples = []
    total_reward = 0.0

    for _ in range(args.episode_windows):
        remaining_ms = float(args.window_ms)
        while remaining_ms > 0 and not eng.term_buf[0]:
            obs = get_obs(eng)
            plan = edf.plan(obs, budget_ms=int(max(1.0, remaining_ms)))
            if not plan:
                break

            action = plan[0]

            # Filter: only teach on actions the policy can actually take.
            active = obs["active_mask"].astype(bool)
            t_deadline = obs["t_deadline"]
            tracked = active & (t_deadline > 0)  # same inference as adapt_obs

            is_search = action == 0
            is_tracked_target = (
                action > 0
                and (action - 1) < len(tracked)
                and tracked[action - 1]
            )

            if is_search or is_tracked_target:
                adapted = adapter.adapt_obs(obs)[0]  # (num_tasks, 8)
                label = np.zeros(args.max_trackers + 1, dtype=np.float32)
                label[action] = 1.0
                examples.append((adapted, label))

            # Advance environment with the EDF plan.
            obs_before = obs
            eng.act_buf[0] = int(action)
            binding.vec_step(eng.env)
            obs_after = get_obs(eng)
            dt = infer_elapsed_ms(obs_before, obs_after)
            total_reward += float(eng.rew_buf[0])
            remaining_ms -= max(dt, 1.0)

        if eng.term_buf[0]:
            break

    eng.close()
    return examples, total_reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-ckpt", default="", help="Optional warm-start checkpoint")
    ap.add_argument(
        "--env-preset",
        choices=["original_like", "repaired_stress"],
        default="original_like",
    )
    ap.add_argument(
        "--task-batch",
        default="20,20,50,100",
        help="Comma-separated task sizes, cycled across episodes",
    )
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=300,
                    help="Number of EDF episodes to collect demonstrations from")
    ap.add_argument("--episode-windows", type=int, default=20)
    ap.add_argument("--window-ms", type=int, default=200)
    ap.add_argument("--max-trackers", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=50,
                    help="Training epochs over the collected demonstration data")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    args.task_batch = parse_int_list(args.task_batch)
    args.env_cfg = env_preset_cfg(args.env_preset)

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    print(f"Device: {device}", flush=True)

    model = PolicyOnlyTransformer(num_tasks=args.max_trackers + 1).to(device)

    if args.base_ckpt and os.path.exists(args.base_ckpt):
        state = torch.load(args.base_ckpt, map_location=device)
        ms = model.state_dict()
        compat = {k: v for k, v in state.items() if k in ms and ms[k].shape == v.shape}
        ms.update(compat)
        model.load_state_dict(ms)
        print(f"Loaded base checkpoint: {args.base_ckpt}", flush=True)

    # Adapter: reuse PolicyOnlyMCTSPlanner.adapt_obs (no model needed for this).
    adapter = PolicyOnlyMCTSPlanner(max_trackers=args.max_trackers, model=None)

    edf = EDFPlanner(max_trackers=args.max_trackers)

    # ------------------------------------------------------------------ #
    # Phase A: Collect EDF demonstrations
    # ------------------------------------------------------------------ #
    print(f"Collecting {args.episodes} EDF episodes...", flush=True)
    all_examples = []
    t0 = time.time()
    for ep in range(args.episodes):
        task = args.task_batch[ep % len(args.task_batch)]
        seed = args.seed_base + ep
        examples, ep_reward = collect_edf_episode(adapter, edf, args, task, seed)
        all_examples.extend(examples)
        if (ep + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(
                f"  ep {ep+1}/{args.episodes}  task={task}  "
                f"examples_so_far={len(all_examples)}  ep_reward={ep_reward:.2f}  "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    print(
        f"Collected {len(all_examples)} examples from {args.episodes} episodes "
        f"in {time.time()-t0:.1f}s",
        flush=True,
    )

    if len(all_examples) == 0:
        print("ERROR: no examples collected — check EDF/env configuration.", flush=True)
        return

    obs_arr = np.array([e[0] for e in all_examples], dtype=np.float32)
    lbl_arr = np.array([e[1] for e in all_examples], dtype=np.float32)
    n = len(obs_arr)
    print(f"Dataset: {n} examples, obs shape={obs_arr.shape}", flush=True)

    # ------------------------------------------------------------------ #
    # Phase B: Supervised training
    # ------------------------------------------------------------------ #
    print(f"Training {args.epochs} epochs (lr={args.lr}, batch={args.batch_size})...", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    best_path = os.path.join(args.out_dir, "best.pt")
    last_path = os.path.join(args.out_dir, "last.pt")

    t1 = time.time()
    for epoch in range(args.epochs):
        idx = np.random.permutation(n)
        epoch_losses = []
        model.train()
        for start in range(0, n, args.batch_size):
            batch_idx = idx[start : start + args.batch_size]
            obs_t = torch.from_numpy(obs_arr[batch_idx]).to(device)
            lbl_t = torch.from_numpy(lbl_arr[batch_idx]).to(device)
            logits = model(obs_t)
            logp = F.log_softmax(logits, dim=1)
            loss = -(lbl_t * logp).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        scheduler.step()

        avg_loss = float(np.mean(epoch_losses))
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_path)

        if (epoch + 1) % args.log_every == 0:
            print(
                f"  epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  "
                f"best={best_loss:.4f}  elapsed={time.time()-t1:.1f}s",
                flush=True,
            )

    torch.save(model.state_dict(), last_path)
    print(
        f"\nDone. Best loss={best_loss:.4f}. Saved {best_path} and {last_path}",
        flush=True,
    )
    print(
        f"\nNext step — Phase 1 iterative RL:\n"
        f"  python train_radarxs_iterative_rl.py \\\n"
        f"    --base-ckpt {best_path} \\\n"
        f"    --out-dir <rl_out_dir> \\\n"
        f"    --task-batch 20,20,50,100 --iterations 30 --epochs-per-iter 30",
        flush=True,
    )


if __name__ == "__main__":
    main()
