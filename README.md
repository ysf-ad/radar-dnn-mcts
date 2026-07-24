# radar-dnn-mcts
<!-- Minimal operator guide; architecture details live in docs/. -->

Research code for learned radar scheduling with transformer policy/Q models and PUCT-style planning.

## Main Components

- `pufferlib/ocean/radarxs/`: radar C simulator, Python binding, and EDF/EST baselines.
- `radar_dnn_mcts/env/`: reward, action, feature, and shadow-transition contracts.
- `radar_dnn_mcts/models/`: shared state encoder, action attention, policy/value heads, latent dynamics, autoregressive decoder, batch decoder, and boundary predictor.
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

Run the reproducible nine-cell comparison:

```powershell
python -m scripts.build_binding
python -m scripts.evaluate --config configs\repro_9cell.yaml --checkpoint path\to\model.pt --methods reencode,muzero,ar,batch,edf,est --device cuda
```

Run a short baseline smoke test:

```powershell
python scripts\evaluate.py --config configs\smoke.yaml --methods edf,est
```

The evaluator writes per-window traces, per-cell summaries, an aggregate summary, and a six-panel plot suite under the requested output directory.

## Training

Collect PUCT targets and train the shared policy/Q model:

```powershell
python -m scripts.collect_puct_targets --teacher edf --out data\edf.npz
python -m scripts.train --data data\edf.npz --out checkpoints\edf.pt

python -m scripts.collect_puct_targets --checkpoint checkpoints\edf.pt --out data\puct.npz
python -m scripts.train --checkpoint checkpoints\edf.pt --data data\puct.npz --out checkpoints\puct.pt
```

The collector stores grouped 200 ms trajectories with PUCT visit targets,
executed rewards, and episode returns. Train a sequence decoder from the same data:

```powershell
python -m scripts.train_sequence --data data\puct.npz --checkpoint checkpoints\puct.pt --out checkpoints\model_sequence.pt
```

Automate repeated collection and training:

```powershell
python scripts\self_play.py --checkpoint checkpoints\model_sequence.pt --out-dir runs\ar --search-model ar --learner ar
python scripts\self_play.py --checkpoint checkpoints\model_sequence.pt --out-dir runs\muzero --search-model core --learner muzero
```

`--learner` accepts `core`, `ar`, `muzero`, or `batch`. Recurrent MuZero training
follows the MuZero paper and the MIT-licensed
[MuZero General](https://github.com/werner-duvaud/muzero-general) trainer;
radar-specific state and action modules retain the local interfaces.

The asynchronous boundary stage retains its dedicated transition dataset:

```powershell
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
