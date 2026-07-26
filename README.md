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

## Training

Each command runs collection, replay training, validation, and checkpoint
promotion.

**Full re-encode**

```powershell
python scripts\self_play.py --checkpoint checkpoints\initial.pt --out-dir runs\reencode --train-config configs\training_9cell.yaml --validation-config configs\promotion_full_horizon.yaml --search-model core --learner core --iterations 30 --rollouts 64 --epochs 5 --batch-size 256 --return-horizon-windows 5 --device cuda
```

**MuZero**

```powershell
python scripts\self_play.py --checkpoint checkpoints\initial.pt --out-dir runs\muzero --train-config configs\training_9cell.yaml --validation-config configs\promotion_full_horizon.yaml --search-model muzero --learner muzero --iterations 30 --rollouts 64 --epochs 5 --batch-size 256 --unroll-steps 10 --return-horizon-windows 5 --device cuda
```

**Autoregressive**

```powershell
python scripts\self_play.py --checkpoint checkpoints\initial.pt --out-dir runs\ar --train-config configs\training_9cell.yaml --validation-config configs\promotion_full_horizon.yaml --search-model ar --learner ar --iterations 30 --rollouts 64 --epochs 5 --batch-size 256 --return-horizon-windows 5 --device cuda
```

**Batch distillation**

```powershell
python scripts\self_play.py --checkpoint checkpoints\initial.pt --out-dir runs\batch --train-config configs\training_9cell.yaml --validation-config configs\promotion_full_horizon.yaml --search-model core --learner batch --iterations 30 --rollouts 64 --epochs 5 --batch-size 256 --return-horizon-windows 5 --device cuda
```

Each run keeps a bounded replay dataset, writes a training history and manifest,
and promotes a candidate only when it improves over both the incumbent and EDF
on the fixed validation grid. `--return-horizon-windows 0` uses the rest of the
episode; the default uses five windows.

The asynchronous boundary stage retains its dedicated transition dataset:

```powershell
python scripts\train_boundary.py --data data\boundaries.npz --checkpoint checkpoints\model_full.pt --out checkpoints\model_async.pt
```

Its latency-aware planning time is
`window_ms - max_latency_ms - buffer_ms`, clipped to the window.

## Evaluate

Evaluate each trained method on the final held-out grid.

**Full re-encode**

```powershell
python scripts\evaluate.py --checkpoint runs\reencode\incumbent.pt --config configs\final_presentation_9cell.yaml --methods reencode,edf,est --reencode-rollouts 1 --device cuda --out results\reencode
```

**MuZero**

```powershell
python scripts\evaluate.py --checkpoint runs\muzero\incumbent.pt --config configs\final_presentation_9cell.yaml --methods muzero,edf,est --muzero-rollouts 1 --device cuda --out results\muzero
```

**Autoregressive**

```powershell
python scripts\evaluate.py --checkpoint runs\ar\incumbent.pt --config configs\final_presentation_9cell.yaml --methods ar,edf,est --ar-rollouts 1 --device cuda --out results\ar
```

**Batch**

```powershell
python scripts\evaluate.py --checkpoint runs\batch\incumbent.pt --config configs\final_presentation_9cell.yaml --methods batch,edf,est --device cuda --out results\batch
```

For a quick end-to-end check:

```powershell
python scripts\evaluate.py --config configs\smoke.yaml --checkpoint checkpoints\initial.pt --methods reencode,muzero,ar,batch,edf,est --reencode-rollouts 1 --muzero-rollouts 1 --ar-rollouts 1 --device cpu
```

Evaluation writes window traces, cell summaries, aggregate summaries, latency,
observation counts, and neural-call counts.

## Documentation

See `docs/ENVIRONMENT.md` for the simulator and reward contract,
`docs/VALIDATION.md` for benchmark validation, and `docs/CODEPATH.md` for the
state, search, collection, training, evaluation, and boundary-execution paths.
