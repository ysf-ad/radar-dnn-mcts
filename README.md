# radar-dnn-mcts

Research code for radar scheduling with transformer policy/Q models and MCTS-style planning.

This repository is a cleaned research snapshot of the radar scheduler work. It keeps the radar-specific C simulator, Python environment wrapper, heuristic baselines, transformer heads, MCTS utilities, training loops, and evaluation scripts. It intentionally excludes large checkpoints, generated plots, old one-off experiment folders, and unrelated PufferLib environments.

## What Is Included

- Fast radar scheduling simulator in C, exposed through a Python extension.
- Radar environment wrapper and action encoding utilities.
- Heuristic baselines: EDF and EST.
- Transformer foundation encoder.
- Flat action policy head.
- Factorized type/target policy head.
- Action-attention factorized policy/Q model.
- Sequential and joint dual-sensor planning utilities.
- Clean ablation entry points for P vs PQ, flat vs factorized, and action-attention variants.

## What Is Not Included

- Large trained checkpoints.
- Generated result tables and plots.
- Exact rerank as the main research method.
- Full PufferLib. Only a minimal `pufferlib/ocean/radarxs` compatibility path is retained because the existing simulator imports use that namespace.

## Repository Layout

```text
radar-dnn-mcts/
  pufferlib/ocean/radarxs/      # Minimal radar C simulator + Python binding wrapper
  radar_dnn_mcts/               # Research model, training, planning, and eval code
  scripts/                      # Thin command-line entry points
  configs/                      # Small reproducibility configs
  tests/                        # Smoke tests
  checkpoints/                  # Local only; ignored by git
  results/                      # Local only; ignored by git
```

## Install

Create an environment, install Python dependencies, then build the radar C extension.

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

This verifies that the radar binding, environment wrapper, heuristics, and main action-attention model can be imported.

## Main Model

The main research model is:

```text
Action-attention factorized PQ
```

Conceptually:

```text
target/context tokens -> shared transformer encoder
encoded targets + CLS + learned sensor embeddings -> action token builder
candidate action tokens -> action self-attention
type head + target head -> policy prior
Q head -> downstream return estimate
```

The dual-sensor action is represented as:

```text
a = (j, k, i)
j in {S, X}                 sensor
k in {search, track}        action type
i = 0 for search, i > 0 for target track
```

The factorized policy prior is:

```text
P(a | s) = P(j, k, i | s)
         = P(k | s, j) P(i | s, j, k)
```

For slides, the simplified version is:

```text
P(action) = P(type) * P(target | type)
```

with one policy stream per sensor.

## Training Path

The recommended research path is:

1. Generate planner targets from the radar simulator.
2. Bootstrap the policy/Q heads from heuristic or planner-improved targets.
3. Train the transformer model with policy and Q losses.
4. Evaluate on the 9-cell load grid:

```text
initial targets: 20, 40, 60
arrival rates:   2, 3, 4
```

The code also keeps side-model ablations for:

- Policy-only `P`.
- Policy + Q `PQ`.
- Flat action head.
- Factorized type/target head.
- Action-attention factorized head.
- Sequential vs joint dual-sensor planning.

## Notes on PufferLib

Earlier development used a PufferLib-style environment layout. PufferLib is not central to the ML method. The useful part was the radar-specific C simulator and Python binding pattern. This repo keeps only the minimum compatibility namespace needed by existing code:

```text
pufferlib/ocean/radarxs
```

The released research code should be understood as a radar scheduling project, not as a general PufferLib project.

## Typical Commands

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

Run a 9-cell evaluation once a checkpoint/target file is available:

```powershell
python scripts\eval_9cell.py --checkpoint checkpoints\model.pt
```

The exact training/evaluation commands depend on which target file or checkpoint you place under `checkpoints/`.
