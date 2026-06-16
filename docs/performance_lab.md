# Performance Lab

This repo includes an initial performance lab for the main action-attention factorized PQ path.

## Scripts

```powershell
python scripts\perf_lab_action_attention.py --device cpu
python scripts\perf_lab_action_attention.py --device cuda --forward-batches 1,8,32,128
python scripts\profile_action_attention_steps.py --device cuda
python scripts\perf_lab_batched_roots.py --device cuda --batch-sizes 1,8,32,128
python scripts\perf_lab_batched_root_tables.py --device cuda --batch-sizes 1,8,32,128
python scripts\perf_lab_batched_branch_sim.py --branch-sizes 1,8,32,128
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

## Batched Root Action Tables

Root search needs more than the single best action: it needs a sorted table of
valid root actions and scores that can be consumed by the cached cursor search.
`BatchedActionAttentionScorer.all_root_action_tables` now builds those tables
for many root observations in one model pass:

```text
observations[]
    -> batched token/slot tensors
    -> one action-attention policy/Q pass
    -> per-root valid physical action gather
    -> sorted padded action tables
```

This is the multi-root counterpart to the single-root cached action table.
Measured CUDA fp32 speedups against looping over roots one at a time:

```text
batch=1:    ~1.0x, ~130 root tables/sec
batch=8:    ~4.7x, ~618 root tables/sec
batch=32:   ~7.6x, ~970 root tables/sec
batch=128:  ~6.5x, ~935 root tables/sec
```

The batched fp32 tables matched the per-root action order and scores in the
benchmark. AMP was slower and changed some action rankings, so root action table
construction should stay fp32 unless a tolerance-aware/ranking-stable mixed
precision path is added later.

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

The branch simulator now also has a lower-overhead C path:

```text
vec_restore_n(root snapshot, active_branch_count)
vec_step_validated_into(dt_buffer, executed_buffer, active_branch_count)
```

This avoids restoring inactive vector slots, avoids Python lists for `dt`/`executed`, and lets root-search callers skip branch observation dictionaries when they only need rewards and execution metadata.

Measured branch-sim correctness matched scalar execution for executed actions, dwell time, and reward. The isolated branch benchmark showed modest direct speedups over the legacy batched API:

```text
branches=1:  fast vs legacy ~1.22x
branches=8:  fast vs legacy ~1.01x
branches=32: fast vs legacy ~1.05x
branches=58: fast vs legacy ~1.00x
```

The larger practical win appears in cached root search, where later waves may have fewer active actions and observation dictionaries are unnecessary.

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

## Neural + Exact Root Wave

`perf_lab_neural_exact_wave.py` connects the batched neural root expansion to exact C branch simulation:

```text
root observation
    -> encode root / build expansion scorer
    -> neural top-K root action proposals
    -> vec_restore_all(root snapshot)
    -> vec_step_validated(one action per env)
```

This is the first combined P/Q proposal plus exact simulator branch-evaluation benchmark. It times the three components separately:

```text
root encode/setup
neural expansion
exact C branch simulation
```

Measured CUDA split:

```text
wave=1:   combined ~13.6 ms, exact sim ~1.5%
wave=4:   combined ~12.8 ms, exact sim ~3.1%
wave=8:   combined ~13.9 ms, exact sim ~5.1%
wave=16:  combined ~14.7 ms, exact sim ~8.7%
wave=32:  combined ~15.5 ms, exact sim ~15.7%
```

For one-step root waves, exact C branch simulation is no longer the dominant cost. The bottleneck is still neural setup plus expansion, especially repeated root encoding/scorer construction. The next major implementation target is a persistent dense tree/search state that reuses encoded root state across multiple expansion waves and batches deeper expansions.

## Persistent Root Wave

`perf_lab_persistent_neural_exact_wave.py` measures the same neural proposal plus exact branch simulation path after constructing the root scorer once:

```text
one-time root encode/scorer setup
then repeated:
    neural top-K proposal from persistent scorer
    exact batched C branch simulation
```

This isolates steady-state search-wave cost after root encoding is cached. Measured CUDA steady-state:

```text
wave=1:   combined ~5.4 ms, exact sim ~3.2%
wave=4:   combined ~5.2 ms, exact sim ~7.3%
wave=8:   combined ~6.0 ms, exact sim ~12.6%
wave=16:  combined ~6.3 ms, exact sim ~19.5%
wave=32:  combined ~7.7 ms, exact sim ~31.5%
```

Compared with rebuilding the scorer every wave, persistent root state roughly halves the 32-action combined wave time:

```text
rebuild scorer every wave, wave=32: ~15.5 ms
persistent scorer, wave=32:         ~7.7 ms
```

This confirms that caching encoded root/search state is not optional for a fast planner. After caching, exact C branch simulation becomes a meaningful share of large waves, so the next optimization should combine persistent dense tree state with batched C branch stepping rather than repeatedly constructing Python scorer objects.

The reusable primitive is `PersistentRootSearch`:

```text
PersistentRootSearch
    owns root snapshot
    owns vector branch simulator
    owns cached root action-attention scorer

search_wave(top_k)
    -> propose top-K actions with cached neural state
    -> simulate actions exactly in one vector C step
```

This is still root-scoped, but it is now a real code path future MCTS work can call instead of a one-off benchmark script.

## Persistent Dense Root Tree

`PersistentDenseRootTree` adds a dense root-state layer around `PersistentRootSearch`:

```text
persistent neural proposer + exact vector C simulation
    -> dense arrays for actions, priors, visits, value sums
    -> PUCT-style root selection
```

This tests whether MCTS bookkeeping is a meaningful bottleneck once neural proposal and exact branch simulation are already batched. Measured CUDA profile for 8 waves of 32 actions from one cached root:

```text
neural proposal total:     ~56.1 ms
exact branch sim total:    ~23.4 ms
dense tree update total:    ~0.8 ms
PUCT selection total:       ~1.4 ms
combined iteration:        ~82.3 ms
unique root actions:        32
total visits:              256
```

The dense tree update is below 1% of the total, and PUCT selection is about 1.7%. This means the next real optimization target is not Python visit/value bookkeeping. It is:

- reducing the repeated neural proposal cost inside search waves;
- reducing exact branch simulation cost for larger waves;
- batching deeper branches so the model sees larger tensor batches per call.

AMP was slower in this benchmark:

```text
fp32 combined iteration: ~82.3 ms
amp combined iteration: ~111.4 ms
```

So mixed precision should not be assumed helpful for this specific small-batch action-attention path.

## Cached All-Action Root Proposals

The root action-attention model already scores every valid root action in one dense tensor pass. Repeated root expansion waves should not recompute that same root proposal table. `PersistentRootSearch.root_action_table()` now caches the sorted root action table:

```text
fixed root observation
    -> one cached action-attention score table over all valid root actions
    -> repeated root waves draw unsimulated actions from the sorted table
    -> exact C branch simulation evaluates those unique actions
```

This is a direct speed optimization of the same proposal model. It does not add a new heuristic or change the model scores.

Measured CUDA A/B for 8 waves of 32 root actions:

```text
recompute neural proposal every wave:
    combined iteration:        ~65.5 ms
    neural proposal total:     ~43.7 ms
    exact branch sim total:    ~19.5 ms
    total visits:              256 duplicate wave visits

cached all-action root table:
    combined iteration:         ~6.3 ms
    proposal lookup total:      ~0.24 ms
    exact branch sim total:     ~4.5 ms
    total visits:                58 unique valid root actions
```

This is about a 10x reduction for repeated root-wave evaluation. The important architectural conclusion is that root search should be:

```text
score all root actions once -> cache sorted proposal table -> exact-sim unique candidates
```

not:

```text
rerun action attention for every root wave
```

The remaining latency after this change is mostly exact branch simulation plus small PUCT/select overhead. For deeper MCTS, the analogous optimization is to cache action tables per expanded node and batch only genuinely new node/action evaluations.

After adding active-count C stepping, skipping branch observations, and live-slice PUCT selection, the same cached root-tree benchmark improved again:

```text
cached root table + fast branch step + live PUCT:
    combined iteration:         ~2.0 ms
    proposal lookup total:      ~0.23 ms
    exact branch sim total:     ~0.75 ms
    dense tree update total:    ~0.21 ms
    PUCT selection total:       ~0.60 ms
    total visits:                58 unique valid root actions
```

Compared with the original repeated neural proposal benchmark:

```text
recompute proposal/root waves: ~65.5 ms
cached + fast branch root:      ~2.0 ms
```

That is roughly a 32x speedup for this root-wave workload.

The root table can also be consumed with a cursor instead of rebuilding an
`exclude` set each wave:

```text
cached sorted root action table
    -> wave 0 consumes actions [0:top_k]
    -> wave 1 consumes actions [top_k:2*top_k]
    -> ...
```

Because cursor waves contain unseen actions by construction, dense tree updates can bulk-append array slices instead of running the generic duplicate-aware update loop.

Measured CUDA profile with cursor proposals and bulk appends:

```text
cursor + bulk append + final selection:
    combined iteration:         ~1.31 ms
    proposal lookup total:      ~0.14 ms
    exact branch sim total:     ~0.75 ms
    dense tree update total:    ~0.11 ms
    final PUCT selection:       ~0.09 ms
    total visits:                58 unique valid root actions

cursor + bulk append + select every wave:
    combined iteration:         ~1.93 ms
    PUCT selection total:       ~0.65 ms
```

The final-selection number is the root-expansion throughput lower bound for this workload. The select-every-wave number is closer to a tree-policy loop that needs a selection after each expansion.

## Cached Action-Attention Internals

`profile_cached_action_attention_internals.py` splits the cached action-attention scoring path into stage timings. On CUDA, the full cached score pass stays around 4.2-4.5 ms across prefix batches from 1 to 64, so throughput improves primarily by batching more prefixes per call:

```text
prefixes=1:   ~241 cached score batches/sec
prefixes=32: ~7626 cached score batches/sec
prefixes=64: ~14568 cached score batches/sec
```

Typical per-call stage costs at this model size:

```text
sensor coupling:       ~0.85-0.90 ms
action self-attention: ~0.75-1.04 ms
target/type/residual heads combined: ~1.5 ms
score/mask assembly:   ~0.3-0.6 ms
CPU transfer:          ~0.08-0.14 ms
```

`torch.compile` did not improve this profile on the tested Windows/CUDA setup; it was slightly slower. This points toward algorithmic batching/cache reuse before custom kernels.

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
