# Performance Lab

This repo includes an initial performance lab for the main action-attention factorized PQ path.

## Scripts

```powershell
python scripts\perf_lab_action_attention.py --device cpu
python scripts\perf_lab_action_attention.py --device cuda --forward-batches 1,8,32,128
python scripts\profile_action_attention_steps.py --device cuda
python scripts\perf_lab_batched_roots.py --device cuda --batch-sizes 1,8,32,128
python scripts\perf_lab_batched_branch_sim.py --branch-sizes 1,8,32,128
```

## First Findings

The model already evaluates all target rows and both sensor streams in one dense tensor pass:

```text
scores: [batch, rows, sensors]
Q:      [batch, rows, sensors]
```

The next parallelism layer is therefore not individual action scoring; it is batching across:

- windows,
- root states,
- rollout branches,
- MCTS simulations,
- candidate joint actions.

## Measured Direction

On the tested machine, single-window sequential planning does not saturate the GPU because each decision is small and pays Python/control-flow overhead. Batched `forward_scores` does scale on GPU:

```text
batch=1:    ~28k action scores/sec
batch=8:   ~252k action scores/sec
batch=32:  ~978k action scores/sec
batch=128: ~3.95M action scores/sec
```

This says the GPU path becomes useful when we batch many states/actions, not when we send one tiny sequential decision at a time.

## Batched Root Scoring

`BatchedActionAttentionScorer` batches many independent radar states/root windows:

```text
observations[] -> stacked tokens/slots -> one model.forward_scores call
               -> per-state valid action masks
               -> best action or top-K root proposals
```

Measured CUDA root-scoring speedups against looping over states one by one:

```text
batch=1:   ~1.0x
batch=8:   ~4.4x
batch=32:  ~7.5x
batch=128: ~6.4x
```

Root action choices matched the old per-state scorer in these tests.

## Current Optimization

`FastActionAttentionPlanner` reuses the root target encoding inside a scheduling window:

```text
root tokens -> transformer encode once
per decision -> update slot features and selected mask
             -> action attention / heads
             -> vectorized candidate selection
```

This removes repeated target/context transformer encoding from the inner loop.

## Step-Level Profile

The first profiler separates:

- tokenization,
- root transformer encoding,
- action scoring from cached encodings,
- slot feature construction,
- candidate generation,
- candidate selection.

The dominant cost remains neural scoring and sequential planning control flow. Python candidate selection is small by comparison.

Example CPU step profile for an 8-decision window:

```text
baseline model forward_scores: ~7.7 ms/decision
fast cached action scoring:    ~4.8 ms/decision
root encode once:              ~1.8 ms/window
candidate selection:           ~0.2 ms/decision
```

So the useful targets are still model scoring, batching, and tree/control-flow structure. Hand-optimizing candidate selection is not the first lever.

## Batched Branch Simulation

`BatchedRootBranchSimulator` uses the C binding's vector environment API to evaluate many one-step root branches from the same snapshot:

```text
root snapshot -> vec_restore_all(batch envs)
              -> one candidate action per env
              -> vec_step_validated(batch envs)
```

This is the simulator-side counterpart to dense root proposals. It preserves the exact C environment transition while removing the Python loop over snapshot restore/step calls.

Measured speedups against a scalar restore/step loop:

```text
branches=1:  ~0.94x
branches=8:  ~1.42x
branches=32: ~1.40x
branches=58: ~1.46x
```

The executed actions, elapsed times, and rewards matched the scalar path in the benchmark. The speedup is real but modest because the C binding still loops internally over vector envs. The larger win should come from combining this with batched neural scoring and dense tree tensors so MCTS expansion/evaluation happens in grouped batches instead of Python node-by-node control flow.

## MCTX Takeaway

MCTX is useful as a design reference because its search tree is dense and batched:

```text
Tree tensors carry a batch dimension.
Search uses vectorized/JIT loops.
Recurrent evaluation is called over batches.
```

Our current radar MCTS/planning stack is still Python-control-flow heavy. The next major optimization should be a batched tree-state representation for radar planning, where rollouts and root states are grouped into tensor batches before model evaluation.

`DenseRootSearchState` is the first step in that direction. It stores top-K root actions as dense arrays:

```text
actions:      [batch, top_k]
prior_scores: [batch, top_k]
visits:       [batch, top_k]
value_sums:   [batch, top_k]
valid:        [batch, top_k]
```

That layout mirrors the important MCTX idea: a batch dimension over independent searches and dense action dimensions inside each tree.

## Next Work

- Batch multiple environment windows during evaluation.
- Expand `DenseRootSearchState` beyond the root into full batched tree tensors.
- Replace Python node objects with dense tree tensors.
- Use `BatchedRootBranchSimulator` for exact one-step branch expansion.
- Keep deeper simulator state transitions as the remaining hard part; either extend the C vector stepping path for deeper batched rollouts or build a PyTorch/JAX-compatible approximate rollout model.
