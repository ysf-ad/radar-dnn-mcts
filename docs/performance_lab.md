# Performance Lab

This repo includes an initial performance lab for the main action-attention factorized PQ path.

## Scripts

```powershell
python scripts\perf_lab_action_attention.py --device cpu
python scripts\perf_lab_action_attention.py --device cuda --forward-batches 1,8,32,128
python scripts\profile_action_attention_steps.py --device cuda
python scripts\perf_lab_batched_roots.py --device cuda --batch-sizes 1,8,32,128
python scripts\perf_lab_batched_branch_sim.py --branch-sizes 1,8,32,128
python scripts\profile_online_pipeline.py --device cpu --windows 20 --planners edf,physical,fast
python scripts\perf_lab_batched_slots.py --device cuda --slot-batches 1,4,8,16,32,64
python scripts\perf_lab_batched_window_expansion.py --device cuda --prefix-batches 1,4,8,16,32,64
python scripts\perf_lab_batched_beam_planner.py --device cuda --beam-widths 1,4,8,16 --max-depth 24
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

## End-to-End Online Profile

`profile_online_pipeline.py` times the full online evaluation loop:

```text
observation read -> planner.plan -> execute scheduled plan -> sample metrics
```

It also records the top `cProfile` cumulative functions for each planner. On a 20-window CPU profile with 40 initial targets and rate 3:

```text
EDF planner.plan:                ~1.5 ms/window
original action-attention plan: ~162.4 ms/window
cached fast action-attention:   ~103.8 ms/window
simulator execution:              ~4.4-4.6 ms/window for learned planners
```

The cached planner is about 1.56x faster than the original online action-attention planner on this run, while producing the same actions for the random-weight latency comparison. The remaining learned-planner cost is dominated by repeated action-attention/head scoring inside each scheduling window:

```text
original forward_scores:      ~2.62 s cumulative over 400 decisions
fast _scores_from_encoded:    ~1.76 s cumulative over 400 decisions
transformer/action couplers:  dominant torch kernels in both paths
```

This confirms the main performance direction:

- cache root target/context encoding, which is already implemented;
- batch independent windows/root states/rollout branches before model scoring;
- move MCTS/search from Python node loops into dense batched tree tensors;
- consider lower-level kernels only after the dense batched formulation is in place.

## Batched Slot Scoring

After the root target/context transformer is cached, the remaining learned-planner loop repeatedly changes only:

```text
slot/context vector
selected-target mask
```

`score_slots_from_encoded` scores many such contexts against the same cached root encoding:

```text
cached root encoding + [slot_1, ..., slot_B] + [mask_1, ..., mask_B]
    -> one batched action-attention/head pass
    -> scores [B, rows, sensors]
```

This is directly relevant to batched rollout branches and dense MCTS expansion, where many candidate states share a root or near-root target encoding.

Measured equivalence:

```text
max absolute score difference vs sequential loop: <= 5e-7
```

Measured CPU speedups:

```text
B=4:  ~1.57x
B=8:  ~1.75x
B=16: ~1.84x
B=32: ~1.76x
B=64: ~1.72x
```

Measured CUDA speedups:

```text
B=4:  ~3.9x
B=8:  ~7.7x
B=16: ~14.8x
B=32: ~23.0x
B=64: ~16.5x
```

This is the strongest evidence so far that the GPU path needs larger search batches, not tiny one-decision calls. The best next implementation step is to make MCTS/root expansion produce batches of slot/mask contexts and call this batched scorer once per expansion wave.

## Batched Expansion Waves

`BatchedWindowExpansionScorer` turns partial within-window plans into a batched next-action expansion problem:

```text
BranchPrefix = (actions, selected targets, elapsed ms, search count, track count, last action)

many BranchPrefix objects
    -> one slot/mask batch
    -> one action-attention/head pass
    -> best next action for every prefix
```

This is the practical bridge from microbenchmark to search implementation. It keeps the root target/context encoding fixed and evaluates an expansion wave of partial branches in one call.

Measured CUDA speedups against scoring each prefix independently:

```text
prefixes=4:  ~3.2x
prefixes=8:  ~5.3x
prefixes=16: ~7.6x
prefixes=32: ~8.1x
prefixes=64: ~7.6x
```

The benchmark checks that the selected next actions match the sequential prefix loop. Score differences stayed within normal floating-point noise:

```text
max absolute score difference: <= 2.1e-7
```

Compared with pure slot scoring, expansion waves include prefix metadata, selected-target masks, candidate construction, and best-action selection, so this is the more realistic speedup to expect when wiring batched scoring into MCTS/root expansion.

## Batched Beam Planner

`BatchedBeamWindowPlanner` is the first usable planner built on the expansion-wave scorer. It maintains a beam of partial window plans and expands the beam in batched waves:

```text
frontier prefixes -> batched score_prefixes/expand_prefixes
                  -> keep top beam_width prefixes by cumulative model score
                  -> repeat until the window budget or max depth is reached
```

The compatibility mode `beam_width=1, branch_top_k=1` matches the current greedy fast planner on the benchmarked root state.

Measured CUDA latency on the same 40-target, rate-3 profile state:

```text
fast cached greedy planner:  ~96.7 ms/window
beam=1, top_k=1:            ~104.4 ms/window  (~1.08x fast latency)
beam=4, top_k=2:            ~128.0 ms/window  (~1.32x fast latency)
beam=8, top_k=2:            ~168.7 ms/window  (~1.74x fast latency)
beam=16, top_k=2:           ~239.9 ms/window  (~2.48x fast latency)
```

This is not a replacement for the greedy fast planner when no extra search is needed. Its purpose is different: it provides a concrete batched-search path where additional beam/MCTS work is grouped into expansion waves instead of serial model calls. The remaining optimization target is to reduce Python prefix/candidate overhead and connect these expansion waves to exact branch simulation.

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
