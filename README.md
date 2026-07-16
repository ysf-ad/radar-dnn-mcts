# radar-dnn-mcts

Research code for learned radar scheduling with transformer policy/Q models and PUCT-style planning.

## Main Components

- `pufferlib/ocean/radarxs/`: radar C simulator, Python binding, and EDF/EST baselines.
- `radar_dnn_mcts/env/`: reward, action, feature, and shadow-transition contracts.
- `radar_dnn_mcts/models/`: shared state encoder, action attention, policy/Q heads, latent dynamics, autoregressive decoder, batch decoder, and boundary predictor.
- `radar_dnn_mcts/schedulers/`: full re-encode, MuZero latent, autoregressive, batched, and asynchronous deployment paths.
- `radar_dnn_mcts/search/`: PUCT implementation.
- `radar_dnn_mcts/training/`: replay records and policy/Q training losses.
- `radar_dnn_mcts/evaluation/`: common benchmark, service metrics, latency reporting, and plot generation.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts\build_binding.py
```

## Test

```powershell
python -m pytest
python scripts\smoke_import.py
```

## Evaluate

Run the nine-cell, 100-window benchmark:

```powershell
python scripts\evaluate.py --config configs\paper_9cell.yaml --checkpoint path\to\model.pt
```

Run a short baseline smoke test:

```powershell
python scripts\evaluate.py --config configs\smoke.yaml --methods edf,est
```

The evaluator writes per-window traces, per-cell summaries, an aggregate summary, and a six-panel plot suite under the requested output directory.

## Training

Collect PUCT targets and train the shared policy/Q model:

```powershell
python scripts\collect_puct_targets.py --out data\puct_targets.npz
python scripts\train.py --data path\to\puct_targets.npz --out checkpoints\model.pt
```

The target dataset contains state tokens, window context, planner-improved policy targets, and action-Q targets and masks.
PUCT uses the predicted policy and action-Q by default. Set
`--learned-q-weight 0` when collecting a policy-only ablation.

The sequence and latent-dynamics stages use explicit trajectory datasets:

```powershell
python scripts\train_sequence.py --data data\windows.npz --checkpoint checkpoints\model.pt --out checkpoints\model_sequence.pt
python scripts\train_dynamics.py --data data\transitions.npz --checkpoint checkpoints\model_sequence.pt --out checkpoints\model_full.pt
python scripts\train_boundary.py --data data\boundaries.npz --checkpoint checkpoints\model_full.pt --out checkpoints\model_async.pt
```

## Reproducibility Checks

Compare simulator state trajectories with a reference experiment checkout:

```powershell
python scripts\cross_test_environment.py --other-root path\to\experiment\model_code
```

Validate an archived result table from its per-window trace:

```powershell
python scripts\validate_reference.py --windows path\to\canonical_windows.csv --expected path\to\canonical_summary.csv
```

See `docs/ENVIRONMENT.md` for the simulator and reward contract, and
`docs/VALIDATION.md` for the distinction between fresh benchmarks and archived
result validation.
