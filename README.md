# radar-dnn-mcts

Research code for radar scheduling with transformer policy/Q models and MCTS-style planning.

This repository contains the core simulator, model, planner, training, and evaluation code used for the radar DNN/MCTS experiments.

## Main Components

- `pufferlib/ocean/radarxs/`
  - Radar C simulator.
  - Python binding source.
  - Environment wrapper.
  - EDF, EST, and MCTS planner implementations.

- `radar_dnn_mcts/`
  - Transformer model code.
  - Flat policy head experiments.
  - Factorized type/target policy head experiments.
  - Action-attention factorized policy/Q model.
  - Dual-sensor sequential and joint-action planning utilities.
  - Training and evaluation scripts used for the main ablations.

- `scripts/`
  - Binding build script.
  - Import smoke test.
  - Baseline cross-test helper.
  - Small training/evaluation entry points.

- `configs/`
  - Small reproducibility configs for smoke and 9-cell evaluation runs.

- `tests/`
  - Lightweight import/smoke tests.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\build_binding.py
```

On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/build_binding.py
```

## Smoke Test

```powershell
python scripts\smoke_import.py
```

Expected output:

```text
smoke_import ok
model=ActionAttentionFactorizedNet
```

## Baseline Cross-Test

The baseline cross-test compares EDF/EST behavior between this repo and another source tree.

```powershell
python scripts\cross_test_baselines.py `
  --code-root "C:\path\to\other\model_code" `
  --out baseline_check.csv `
  --windows 100 `
  --seed 916
```

## Common Entry Points

Build the C binding:

```powershell
python scripts\build_binding.py
```

Run import smoke test:

```powershell
python scripts\smoke_import.py
```

Run a small action-attention PQ smoke experiment:

```powershell
python scripts\train_action_attention_pq_smoke.py
```

Run the 9-cell evaluation entry point:

```powershell
python scripts\eval_9cell.py --checkpoint path\to\model.pt
```

Run the action-attention performance lab:

```powershell
python scripts\perf_lab_action_attention.py --device cuda --forward-batches 1,8,32,128
python scripts\profile_action_attention_steps.py --device cuda
python scripts\perf_lab_batched_roots.py --device cuda --batch-sizes 1,8,32,128
python scripts\perf_lab_batched_root_tables.py --device cuda --batch-sizes 1,8,32,128
python scripts\profile_root_table_steps.py --device cuda --batch-size 32
python scripts\perf_lab_batched_branch_sim.py --branch-sizes 1,8,32,128
python scripts\perf_lab_multi_root_branch_sim.py --root-counts 1,4,8,16,32 --branches-per-root 8
python scripts\profile_online_pipeline.py --device cpu --windows 20 --planners edf,physical,fast
python scripts\perf_lab_batched_slots.py --device cuda --slot-batches 1,4,8,16,32,64
python scripts\perf_lab_batched_window_expansion.py --device cuda --prefix-batches 1,4,8,16,32,64
python scripts\profile_cached_action_attention_internals.py --device cuda --prefix-batches 1,4,8,16,32,64
python scripts\perf_lab_batched_beam_planner.py --device cuda --beam-widths 1,4,8,16 --max-depth 24
python scripts\perf_lab_neural_exact_wave.py --device cuda --wave-sizes 1,4,8,16,32
python scripts\perf_lab_persistent_neural_exact_wave.py --device cuda --wave-sizes 1,4,8,16,32
python scripts\perf_lab_persistent_dense_root_tree.py --device cuda --waves 8 --top-k 32
python scripts\perf_lab_persistent_dense_root_tree.py --device cuda --waves 8 --top-k 32 --proposal-mode cached
python scripts\perf_lab_persistent_dense_root_tree.py --device cuda --waves 8 --top-k 32 --proposal-mode cached_cursor
python scripts\perf_lab_multi_env_online_batch.py --device cuda --envs 64 --windows 20 --initial-targets 60 --rate 4 --amp --fast-env-step --direct-root-pack --cached-action-table --gpu-action-template --gpu-valid-mask --batch-env-step
```
