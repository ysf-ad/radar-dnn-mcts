# Performance Lab

This repo includes an initial performance lab for the main action-attention factorized PQ path.

## Scripts

```powershell
python scripts\perf_lab_action_attention.py --device cpu
python scripts\perf_lab_action_attention.py --device cuda --forward-batches 1,8,32,128
python scripts\profile_action_attention_steps.py --device cuda
python scripts\perf_lab_batched_roots.py --device cuda --batch-sizes 1,8,32,128
python scripts\perf_lab_batched_root_tables.py --device cuda --batch-sizes 1,8,32,128
python scripts\profile_root_table_steps.py --device cuda --batch-size 32
python scripts\perf_lab_batched_branch_sim.py --branch-sizes 1,8,32,128
python scripts\perf_lab_multi_root_branch_sim.py --root-counts 1,4,8,16,32 --branches-per-root 8
python scripts\profile_online_pipeline.py --device cpu --windows 20 --planners edf,physical,fast
python scripts\profile_branch_sim_steps.py --batch-sizes 1,8,16,32,64,128,256
python scripts\perf_lab_batched_slots.py --device cuda --slot-batches 1,4,8,16,32,64
python scripts\perf_lab_batched_window_expansion.py --device cuda --prefix-batches 1,4,8,16,32,64
python scripts\profile_cached_action_attention_internals.py --device cuda --prefix-batches 1,4,8,16,32,64
python scripts\perf_lab_paired_heads.py --device cuda --batches 1,8,32,64,128
python scripts\perf_lab_batched_beam_planner.py --device cuda --beam-widths 1,4,8,16 --max-depth 24
python scripts\perf_lab_neural_exact_wave.py --device cuda --wave-sizes 1,4,8,16,32
python scripts\perf_lab_persistent_neural_exact_wave.py --device cuda --wave-sizes 1,4,8,16,32
python scripts\perf_lab_persistent_dense_root_tree.py --device cuda --waves 8 --top-k 32
python scripts\perf_lab_persistent_dense_root_tree.py --device cuda --waves 8 --top-k 32 --proposal-mode cached
python scripts\perf_lab_persistent_dense_root_tree.py --device cuda --waves 8 --top-k 32 --proposal-mode cached_cursor
python scripts\perf_lab_persistent_dense_root_tree.py --device cuda --waves 8 --top-k 32 --proposal-mode cached_cursor_bulk
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

The root-table path now also has a vectorized physical action table builder:

```text
observations[]
    -> batched active/deadline/range/free-sensor masks
    -> dense candidate action table [batch, 2 + 2 * max_targets]
    -> batched score gather
    -> per-root sorted valid actions
```

The vectorized path matched the legacy batched path for action order, scores,
and counts. Measured CUDA fp32 timings:

```text
batch=1:    legacy batched ~3.90 ms, vectorized ~3.82 ms, ~1.02x
batch=8:    legacy batched ~5.72 ms, vectorized ~6.03 ms, ~0.95x
batch=32:   legacy batched ~12.79 ms, vectorized ~12.24 ms, ~1.04x
batch=128:  legacy batched ~47.42 ms, vectorized ~44.10 ms, ~1.08x
```

This is useful but not a major win by itself. A stage profile at batch 32 showed
where the remaining root-table time goes:

```text
tokenize_stack:                    ~5.55 ms, ~41.0%
model_forward_and_cpu_transfer:     ~3.83 ms, ~28.4%
slot_features_stack:                ~1.81 ms, ~13.4%
legacy physical action arrays:      ~1.08 ms,  ~8.0%
vectorized physical action table:   ~0.29 ms,  ~2.2%
vectorized gather/sort:             ~0.39 ms,  ~2.9%
host_to_device:                     ~0.25 ms,  ~1.8%
attach_env_obs:                     ~0.02 ms,  ~0.2%
```

So the next larger opportunity is to reduce repeated token/slot feature
construction or cache more state, not to custom-kernel the physical action mask.

That next feature-construction pass is now implemented for the batched scorer.
`tokenize_batch` and `slot_features_batch` reproduce the legacy per-observation
feature definitions exactly, but construct the dense `[batch, rows, features]`
arrays with batched NumPy operations:

```text
observations[]
    -> batched t_desired/deadline/dwell/active/tracked/range arrays
    -> batched token tensor [batch, max_targets + 1, token_dim]
    -> batched slot tensor [batch, slot_dim]
    -> one action-attention policy/Q pass
```

Equivalence checks on real simulator observations matched the legacy functions:

```text
token max abs diff: 0.0
slot max abs diff:  0.0
```

Updated CUDA batch-32 stage profile:

```text
legacy_tokenize_stack:            ~5.22 ms
batched_tokenize:                 ~1.76 ms  (~3.0x faster)
legacy_slot_features_stack:       ~1.78 ms
batched_slot_features:            ~1.05 ms  (~1.7x faster)
model_forward_and_cpu_transfer:   ~4.03 ms
```

Updated end-to-end root table timings after batched features:

```text
batch=8:    vectorized root tables ~4.92 ms,  ~6.2x vs loop
batch=32:   vectorized root tables ~7.14 ms, ~17.1x vs loop
batch=128:  vectorized root tables ~23.83 ms, ~20.8x vs loop
```

Compared with the prior vectorized-candidate-only run:

```text
batch=32:   ~12.24 ms -> ~7.14 ms
batch=128:  ~44.10 ms -> ~23.83 ms
```

This confirms feature construction was a real bottleneck. The next larger
target is the remaining model forward/action-attention stack and avoiding CPU
round-trips when the caller can consume tensors directly.

The root-table scorer now also has a Torch-native gather/sort path:

```text
score tensor [batch, rows, sensors] stays on CUDA
    -> gather valid physical candidate scores on CUDA
    -> sort candidate scores with torch.sort
    -> transfer only compact sorted root tables back to CPU
```

This avoids copying the entire dense score table to NumPy before action-table
construction. It is not universally faster because small batches pay extra
tensor-construction and sort overhead, but it helps once the batch is large
enough:

```text
batch=8:    torch path slower than vectorized NumPy
batch=32:   roughly parity / slightly slower
batch=128:  torch path faster, ~1.03-1.13x vs vectorized NumPy
```

`all_root_action_tables_fast` uses the measured rule:

```text
if CUDA and batch >= 96:
    use Torch gather/sort
else:
    use vectorized NumPy gather/sort
```

Short validation run:

```text
batch=8:    fast ~4.88 ms,  ~7.2x vs per-root loop
batch=32:   fast ~8.35 ms, ~16.8x vs per-root loop
batch=128:  fast ~22.05 ms, ~25.2x vs per-root loop
```

This means large multi-root evaluations can now keep the high-volume score
gather/sort work on the GPU, while small online batches stay on the lower
overhead CPU/NumPy path.

## Combined Policy/Q Inference

For online planning and root-table construction, callers usually need only:

```text
combined_score = policy_weight * policy_score + q_weight * q_score
```

The earlier inference path built two full dense tensors first:

```text
policy_scores [batch, rows, sensors]
q_scores      [batch, rows, sensors]
mask both
combine
```

The optimized inference path computes the same weighted score directly. It still
evaluates all action policy and Q heads, but it avoids allocating and masking two
separate full score tables before combining them:

```text
type/target policy + type/target Q + action residual policy/Q
    -> one combined dense score table
    -> one final validity mask
```

Equivalence checks on cached-prefix scoring and root scoring matched exactly:

```text
cached max abs diff: 0.0
root max abs diff:   0.0
```

Updated cached action-attention profile on CUDA:

```text
prefixes=1:   full cached score ~2.24 ms,   ~446 score batches/sec
prefixes=8:   full cached score ~2.40 ms,  ~3335 score batches/sec
prefixes=32:  full cached score ~2.50 ms, ~12787 score batches/sec
prefixes=64:  full cached score ~3.54 ms, ~18092 score batches/sec
```

The remaining model-side costs are the transformer-style pieces and head MLPs:

```text
sensor coupling:       ~0.61 ms
action self-attention: ~0.46 ms at batch 1, ~1.16 ms at batch 64
target/type/residual heads: still a large cumulative share
```

Updated root-table timings after combined-score inference:

```text
batch=32:   torch root tables ~7.48 ms,  ~16.2x vs per-root loop
batch=128:  torch root tables ~18.21 ms, ~26.0x vs per-root loop
```

The next likely model-side optimization is to reduce the action-attention
TransformerEncoder overhead. Options include a lean custom action-mixing block
for the fixed 2-sensor action grid, or compiling/fusing the small MLP heads into
larger shared projections. Custom CUDA kernels are only justified after that,
because the current bottleneck is still high-level transformer/head dispatch
rather than a single obvious scalar kernel.

For the single-window greedy online planner, an attempted GPU scalar action
selector was measured and rejected. It avoided copying the small `[rows, 2]`
score table back to CPU, but it introduced per-decision tensor construction and a
scalar synchronization. The measured default is therefore:

```text
cached combined score on CUDA
    -> copy one small [rows, 2] score table to CPU
    -> vectorized NumPy physical candidate selection
```

Online step profile on CUDA after this correction:

```text
plans_match baseline: true
fast_action_score_from_encoded:       ~3.33 ms mean, ~2.20 ms p50
fast_vectorized_candidate_select:     ~0.13 ms mean
failed GPU scalar candidate selector: ~1.42 ms mean in the rejected test
```

This reinforces the current rule: keep large batched action-table gather/sort on
GPU, but keep tiny single-window greedy candidate selection on CPU.

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

## Multi-Root Branch Simulation

`BatchedMultiRootBranchSimulator` extends the exact C branch simulator from:

```text
one root snapshot -> many branch actions
```

to:

```text
many root snapshots + many branch actions
    -> flatten `(root_id, action)` pairs
    -> restore each vector slot from its own root snapshot
    -> one validated vector step
```

The C binding now includes `vec_restore_many`, which restores a list of snapshot dictionaries into the first `N` vector slots. This gives search code one primitive for exact branch evaluation across many independent roots/windows.

Correctness matched the per-root loop for executed action, dwell time, and reward. Measured CPU-side C simulator throughput with 8 branches per root:

```text
roots=1, branches=8:     ~0.97x vs per-root loop
roots=4, branches=32:    ~1.11x
roots=8, branches=64:    ~1.16x
roots=16, branches=128:  ~1.05x
roots=32, branches=256:  ~1.13x
```

With 32 branches per root the path is near parity:

```text
roots=4, branches=128:    ~1.02x
roots=8, branches=256:    ~1.03x
roots=16, branches=512:   ~1.02x
roots=32, branches=1024:  ~0.97x
```

So multi-root flattening is useful architecturally, but it is not the next large speed source by itself. The exact simulator is already fairly efficient after the previous active-count and buffer-output changes. Larger wins should come from batching neural state/action-table construction and reducing Python tree policy overhead around the simulator.

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

The cursor path can be pushed further when the waves are independent root
branches. Instead of simulating eight `top_k` cursor waves separately, the
`cached_cursor_bulk` mode requests `waves * top_k` actions from the cached
root table and simulates them in one vectorized C call:

```text
waves=8, top_k=16:
    cached cursor waves: ~0.80 ms, 58 unique actions
    bulk cursor:         ~0.43 ms, 58 unique actions

waves=8, top_k=32:
    cached cursor waves: ~0.63 ms, 58 unique actions
    bulk cursor:         ~0.50 ms, 58 unique actions
```

Rewards and selected actions matched in these paired runs. After bulk cursor
expansion, the remaining cost is mostly the vectorized exact branch simulator
rather than neural proposal or dense tree bookkeeping.

## Fused C Restore/Step

`profile_branch_sim_steps.py` splits branch simulation into root restore,
action-buffer assignment, validated C stepping, and result copies. The profile
showed validated stepping at roughly 90% of branch simulation time and restore
at roughly 7-9%:

```text
batch=32:  validated step ~0.180 ms, restore ~0.014 ms
batch=128: validated step ~0.727 ms, restore ~0.060 ms
batch=256: validated step ~1.421 ms, restore ~0.134 ms
```

The binding now exposes `vec_restore_step_validated_into`, which restores the
root snapshot, assigns the action, validates/steps, and writes `dt`/`executed`
inside one C call. `BatchedRootBranchSimulator.step_actions` uses this fused
entrypoint when available, with the older restore-then-step path as fallback.

Branch benchmark equivalence checks passed for executed actions, dwell times,
and rewards. The speedup is modest but real:

```text
58-root-action batch:
    old fast vector step:  ~0.356-0.365 ms
    fused restore/step:    ~0.347-0.351 ms

bulk dense root expansion, waves=8, top_k=32:
    before fused C path:   ~0.50 ms combined iteration
    after fused C path:    ~0.42 ms combined iteration
```

At this point the exact simulator is the dominant remaining non-neural cost.
Deeper gains require parallelizing the C loop itself or replacing exact branch
simulation with a learned/vectorized rollout model.

The fused restore/step loop is now parallelized with OpenMP when the branch
batch has at least 32 actions. The binding setup enables OpenMP for the local C
extension build, and the fused path extracts snapshot bytes before entering the
parallel region so no Python API calls occur inside the loop.

Equivalence checks again passed for executed actions, dwell times, and rewards.
The branch simulator speedup is substantial at realistic root batch sizes:

```text
58-root-action branch batch:
    previous fused serial path: ~0.347-0.351 ms
    OpenMP fused path:          ~0.100-0.116 ms

bulk dense root expansion, waves=8, top_k=32:
    fused serial C path:        ~0.42 ms combined iteration
    OpenMP fused C path:        ~0.28 ms combined iteration
    exact branch sim portion:   ~0.083 ms
```

This shifts the root-search bottleneck away from branch stepping; PUCT
selection and dense tree update are now visible fractions of the remaining
sub-millisecond root expansion path.

## Cached Dense-Tree Selection State

After OpenMP branch stepping, PUCT selection became a visible part of the bulk
root path. `PersistentDenseRootTree` now maintains:

```text
_total_visits:       updated incrementally instead of summing visits
_prior_live_cache:   softmax over live prior scores, invalidated on tree update
_q_live_cache:       live Q values, updated when visits/rewards change
```

This avoids recomputing total visits, prior softmax, and `value_sum / visits`
inside every selection call.

Measured on the same `waves=8, top_k=32, cached_cursor_bulk` root expansion:

```text
OpenMP branch stepping, uncached selector:
    combined iteration: ~0.28 ms
    PUCT selection:     ~0.089 ms

cached prior only:
    combined iteration: ~0.24 ms
    PUCT selection:     ~0.073 ms

cached prior + Q:
    combined iteration: ~0.19 ms
    PUCT selection:     ~0.046 ms
```

The selected action, unique action count, visit count, and reward sum matched
the uncached selector runs.

## Cached Action-Attention Internals

`profile_cached_action_attention_internals.py` splits the cached action-attention scoring path into stage timings. After the combined policy/Q scoring path, a full cached score pass is roughly 2.2-2.6 ms on CUDA across prefix batches from 1 to 64:

```text
prefixes=1:    ~2.22 ms/call,   ~450 prefixes/sec
prefixes=32:   ~2.18 ms/call, ~14706 prefixes/sec
prefixes=64:   ~2.61 ms/call, ~24475 prefixes/sec
```

Typical per-call stage costs at this model size:

```text
sensor coupling:       ~0.49-0.54 ms
action self-attention: ~0.46-0.92 ms
target/type/residual heads combined: ~0.77-0.93 ms
score/mask assembly:   ~0.17-0.22 ms
CPU transfer:          ~0.06-0.08 ms
```

`torch.compile` did not improve this profile on the tested Windows/CUDA setup; it was slightly slower. This points toward algorithmic batching/cache reuse before custom kernels.

## Full Stack Profile

`profile_online_pipeline.py` profiles real online windows and records `cProfile` output. A 30-window CUDA run on the 40-target, rate-3 cell gave:

```text
EDF planner:                  ~0.47 ms/window planning, ~3.98 ms/window execution
old physical learned planner: ~88.36 ms/window planning, ~2.20 ms/window execution
fast action-attention planner:~68.38 ms/window planning, ~2.05 ms/window execution
```

For the fast planner, the cumulative profiler is dominated by repeated `_combined_scores_from_encoded` calls. Environment stepping and result execution are small relative to learned planning latency.

The batch-root stage profile at batch 64 shows a different bottleneck mix:

```text
legacy per-state tokenization: ~13.06 ms/batch
model forward + CPU transfer:  ~4.58 ms/batch
legacy per-state slot features:~4.26 ms/batch
batched tokenization:          ~3.94 ms/batch
batched slot features:         ~2.70 ms/batch
```

This confirms the split strategy:

- scalar online planning is bottlenecked by repeated neural score calls;
- batched root/MCTS work is bottlenecked by feature construction unless it uses the batched builders;
- the best optimization target is still batched search/tree expansion, not more scalar planner micro-optimization.

`summarize_perf_profiles.py` converts the online, root-table, and cached-internals JSON files into a compact Markdown report for slide/debug use.

## Cached Search-Row Mask

The score builders need a fixed mask identifying row 0 as the search action. Rebuilding that mask with `torch.arange(rows) == 0` inside every score call creates a small repeated CUDA allocation. `FastActionAttentionPlanner` and `BatchedActionAttentionScorer` now cache that mask per `(rows, device)`.

A same-process A/B on one cached root score call showed exact equality and a small latency improvement:

```text
fresh mask allocation:  mean ~2.263 ms, p90 ~2.755 ms
cached mask tensor:     mean ~2.247 ms, p90 ~2.568 ms
max abs diff:           0.0
```

This is a marginal scalar-path optimization, not the main performance lever. It is kept because it removes fixed allocation overhead without changing model semantics.

## Paired Policy/Q Head Experiment

`perf_lab_paired_heads.py` benchmarks whether the policy and Q MLP heads should be evaluated as paired functional calls instead of separate `nn.Sequential` modules. The experiment checks three paths:

```text
separate modules:       current runtime path
paired direct:          same weights, functional LayerNorm/Linear/GELU/Linear
paired stacked/einsum:  stack policy/Q as a leading dimension and batch matmuls
```

The paired-direct path is algebraically exact and often wins for tiny type/residual head microbenchmarks. The stacked/einsum variant is generally slower. When wired into the full cached scorer, however, the paired-direct runtime path made end-to-end cached scoring slower:

```text
current cached score, prefixes=1:       ~2.22 ms
paired-head cached score, prefixes=1:   ~2.57 ms
current cached score, prefixes=32:      ~2.18 ms
paired-head cached score, prefixes=32:  ~2.37 ms
```

Conclusion: do not fuse these heads in the current PyTorch runtime. The microbenchmark benefit does not survive the full action-attention scorer. The larger optimization remains batching the search/tree work so each neural call handles more prefixes.

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
