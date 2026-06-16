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
python scripts\perf_lab_attention_backend_variants.py --device cuda --envs 64
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

Cursor-bulk expansion can also skip the Python action-to-index map. The cursor
walks a sorted root action table once, so actions are unique by construction and
no duplicate-aware exclusion is needed. The default benchmark now disables map
maintenance for `cached_cursor_bulk`, while duplicate-aware modes keep it.

```text
cached_cursor_bulk with action map:
    dense tree update:  ~0.035 ms
    combined iteration: ~0.265 ms

cached_cursor_bulk without action map:
    dense tree update:  ~0.022 ms
    combined iteration: ~0.238 ms
```

Selection and reward outputs matched. This is a root-only specialization; the
generic duplicate-aware tree update still maintains the map.

Cursor proposals now also avoid copying cached root-table slices. The generic
`propose_cached` path still returns concrete arrays because exclusion may
materialize a filtered table, but cursor expansion uses `propose_cached_view`,
which returns a contiguous view over the already-sorted cached action table.

```text
cursor-bulk without action map, copied proposal slices:
    combined iteration: ~0.238 ms

cursor-bulk without action map, no-copy proposal view:
    combined iteration: ~0.213 ms
```

The output action/reward invariants matched.

PUCT selection now also uses an in-place scratch buffer for live scores instead
of allocating a temporary score vector on every call:

```text
cached Q/prior + no-copy cursor proposal:
    PUCT selection with temporary scores: ~0.064 ms
    PUCT selection with scratch scores:   ~0.049 ms
```

The selected action and reward outputs matched. At this scale the full
iteration is close to timing noise, but avoiding the allocation keeps the hot
selection path predictable.

The visit denominator is also cached as `_visit_inv_cache = 1 / (1 + visits)`
when visits change. Selection now multiplies by this cached reciprocal instead
of dividing by the live integer visit array:

```text
scratch scores + live visit divide: ~0.049 ms PUCT selection
scratch scores + cached visit inv:  ~0.042 ms PUCT selection
```

The selected action and reward outputs matched.

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

## Root Expansion Step Profile

`perf_lab_root_expansion_compare.py` is the current one-command profile for the
root expansion hot path. It compares the older proposal/update modes against
the fastest batched cursor path and reports the four major step costs:

```text
mode                         total   propose   sim     update   select   unique
recompute                    46.914  44.973    0.629   0.865    0.066    32
cached_unique                 1.204   0.223    0.426   0.277    0.052    58
cached_cursor                 0.666   0.078    0.331   0.086    0.035    58
cached_cursor_bulk_with_map   0.237   0.016    0.089   0.035    0.047    58
cached_cursor_bulk_best       0.217   0.017    0.082   0.023    0.046    58
```

Configuration:

```text
device=cuda, initial_targets=40, arrival_rate=3, seed=916,
waves=8, top_k=32, warmup=8, iterations=40
```

This profile makes the current optimization picture clear:

- recomputing neural proposals every wave is the original bottleneck;
- cached root action tables remove almost all repeated neural work;
- cursor expansion removes duplicate filtering and repeated top-K selection;
- bulk cursor expansion exposes enough simulator work for the OpenMP fused C
  stepper to matter;
- the best path is now dominated by exact branch simulation and PUCT selection,
  each under 0.1 ms at this root workload.

The `cached_cursor_bulk_best` path is a root-only specialization. It disables
the Python action-to-index map because the cursor walks a sorted cached root
action table exactly once, so duplicates are impossible in that benchmarked
path. Duplicate-aware modes still keep the map.

## Batched Root Scoring Throughput

`BatchedActionAttentionScorer.best_actions_torch(...)` keeps batched score
gather and `argmax` on the GPU and returns only the selected action ids. This is
the throughput-oriented counterpart to the low-latency CUDA Graph planner:
single online windows are sequential, but independent root states and rollout
branches can be grouped into one large model call.

`perf_lab_batched_roots.py` compares sequential root decisions against batched
scoring on the same observation family:

```text
device=cuda, initial_targets=40, arrival_rate=3, seed=916

batch  sequential   batched GPU-select   speedup   GPU-select throughput
1        3.744 ms       4.082 ms          0.92x       245 states/sec
8       29.318 ms       5.161 ms          5.68x     1,550 states/sec
32     112.302 ms       7.080 ms         15.86x     4,520 states/sec
128    586.699 ms      47.953 ms         12.23x     2,669 states/sec
```

All batched GPU-selected actions matched the sequential reference. The useful
operating range starts once there are enough independent roots to amortize
tokenization, model launch, and action-table construction. This is the path to
using the GPU for evaluation sweeps, batched root expansion, and MCTS rollout
frontiers.

`BatchedActionAttentionScorer.prepare_root_batch(...)` caches the CPU-side
inputs for repeated scoring of the same root batch:

```text
attached observations
token tensor
slot/context tensor
physical action ids
flat action indices
valid action mask
```

`best_actions_prepared_torch(...)` then only transfers the prepared arrays,
runs the model, gathers candidate scores, and returns best action ids. This is
useful when the same root/frontier batch is scored repeatedly, as in benchmark
sweeps, cached root expansion, or repeated rollout-frontier evaluation.

Prepared-path timing:

```text
batch 32:
    full batched GPU-select:  ~7.08 ms,  ~4,520 states/sec
    prepared GPU-select:      ~3.56 ms,  ~8,995 states/sec
    sequential speedup:       ~30.8x

batch 128:
    full batched GPU-select:  ~27.82 ms mean, ~19.93 ms p50
    prepared GPU-select:      ~17.96 ms mean,  ~8.57 ms p50
    sequential speedup:       ~24.9x mean
```

The prepared path also has an optional fixed-shape CUDA Graph replay:

```text
script: scripts/perf_lab_batched_roots.py --cuda-graph
device=cuda, initial_targets=40, arrival_rate=3, seed=916,
warmup=8, iterations=30

batch 32:
    prepared GPU-select:      ~3.63 ms,  ~8,813 states/sec
    prepared CUDA graph:      ~1.71 ms, ~18,686 states/sec
    sequential speedup:       ~66.1x
    selected actions match:   true

batch 128:
    prepared GPU-select:      ~17.85 ms mean, ~9.02 ms p50
    prepared CUDA graph:      ~16.61 ms mean, ~8.50 ms p50
    sequential speedup:       ~27.6x mean
    selected actions match:   true
```

The graph path captures model forward plus GPU gather/argmax. It still copies
the prepared token, slot, action-index, and validity arrays into static GPU
buffers before replay. That is why batch 32 benefits strongly from lower launch
overhead, while batch 128 is closer: larger transfer/model work dominates.

For repeated scoring of the exact same prepared frontier, the fastest path is
to keep the prepared batch resident on the GPU with
`prepared_to_device(...)` / `best_actions_prepared_device_graph(...)`. This
captures the same model-forward and selection graph, but avoids repeated H2D
copies of the prepared arrays:

```text
script: scripts/perf_lab_batched_roots.py --cuda-graph
device=cuda, initial_targets=40, arrival_rate=3, seed=916,
warmup=8, iterations=30

batch 32:
    copied prepared CUDA graph:       ~1.73 ms, ~18,534 states/sec
    device-resident CUDA graph:       ~1.58 ms, ~20,209 states/sec
    sequential speedup:               ~71.6x
    selected actions match:           true

batch 128:
    copied prepared CUDA graph:       ~13.93 ms mean, ~8.38 ms p50
    device-resident CUDA graph:       ~12.96 ms mean, ~7.25 ms p50
    sequential speedup:               ~35.3x mean
    selected actions match:           true
```

This is the right interface for MCTS/frontier code that revisits a fixed batch
of candidate roots or leaves. For changing observations, the normal prepared
path is still needed because the tokens, slots, validity masks, and action
tables must be rebuilt.

## Batched Prefix/Frontier Expansion

`BatchedWindowExpansionScorer` batches many partial within-window prefixes under
one encoded root state. The original path already scored prefixes in one model
call, but copied the full dense `[prefixes, targets, sensors]` score table back
to CPU and selected valid actions in Python.

The prefix scorer now has two top-1-oriented alternatives:

- `score_prefixes_gpu_select(...)`: keeps valid-action gather/argmax on GPU, but
  still rebuilds prefix masks, slots, and action tables each call.
- `prepare_prefixes_device(...)` plus
  `score_prepared_prefixes_device_graph(...)`: keeps fixed prefix masks, slots,
  and action tables resident on GPU and replays model scoring plus selection
  through a CUDA Graph.

Measured on a hard root state:

```text
script: scripts/perf_lab_batched_window_expansion.py
device=cuda, initial_targets=60, arrival_rate=4, seed=916,
warmup=10, iterations=50

prefixes  current full table   prepared device graph   speedup vs sequential
1         ~2.64 ms             ~0.49 ms                ~6.0x
8         ~4.07 ms             ~0.69 ms                ~33.3x
32        ~8.45 ms             ~1.89 ms                ~47.9x
64        ~14.40 ms            ~3.12 ms                ~55.6x
128       ~26.70 ms            ~5.26 ms                ~67.0x

selected actions match: true
max score difference:   < 2e-7
```

The plain GPU-select path was not faster than the existing full-table path for
most prefix counts because it still paid Python/Numpy preparation and tensor
construction costs. The prepared device graph path is the useful one for dense
MCTS/frontier batches that can be prepared once and replayed many times.

The physical action table for prefix batches is now vectorized: the scorer
caches one root action table and applies a per-prefix selected-target validity
mask instead of rebuilding `physical_action_arrays(...)` for every prefix. This
keeps top-1 action selection equivalent while reducing CPU preparation for large
prefix batches:

```text
prefixes  full-table path   GPU top-1 with vectorized table
32        ~8.26 ms          ~7.38 ms
64        ~18.59 ms         ~14.98 ms
128       ~35.29 ms         ~25.72 ms

selected actions match: true
max score difference:   < 2e-7
```

The prefix scorer also moved selected-mask construction and slot-feature
construction to batched array paths. With vectorized masks, batched slots, and
the cached root action table, the same hard-root prefix benchmark improved
further:

```text
prefixes  full-table path   GPU top-1 path   prepared device graph
32        ~6.03 ms          ~4.78 ms         ~1.60 ms
64        ~9.63 ms          ~7.13 ms         ~3.16 ms
128       ~17.65 ms         ~11.48 ms        ~5.73 ms

selected actions match: true
max score difference:   < 2e-7
```

For a fixed root observation, only four slot fields vary across prefixes:
elapsed time, search count, track count, and whether the last action was search.
The scorer now caches a slot template and overwrites just those columns. This
removes repeated observation-derived slot work:

```text
prefixes  full-table path   GPU top-1 path   prepared device graph
32        ~4.63 ms          ~3.74 ms         ~1.61 ms
64        ~8.08 ms          ~5.21 ms         ~2.79 ms
128       ~13.42 ms         ~7.89 ms         ~4.98 ms

selected actions match: true
max score difference:   < 2e-7
```

The default full-table path also now uses the cached root action table for its
top-1 CPU selection. `score_prefixes(...)` gathers valid candidate scores with
one NumPy operation, and `expand_prefixes(top_k=1)` consumes those top-1 results
directly instead of rebuilding physical action arrays per prefix. The measured
prefix benchmark remains noisy at large batch sizes, but the live top-1 beam
planner stays in the same improved range while preserving plans:

```text
beam width 1, full-table top-1:       ~60.25 ms/window
beam width 4, full-table top-1:       ~59.27 ms/window
beam width 8, full-table top-1:       ~63.28 ms/window
plans match fast planner:             true
```

The prefix scorer also caches root action/base/sensor/flat-index tensors on the
GPU. This removes repeated device construction for the experimental device
top-1 path and slightly tightens the default live beam timing:

```text
beam width 1, full-table top-1:       ~54.37 ms/window
beam width 1, device top-1:           ~65.18 ms/window
beam width 8, full-table top-1:       ~53.70 ms/window
beam width 8, device top-1:           ~63.59 ms/window
plans match fast planner:             true
```

Prefix selected-mask construction and action-validity construction are now
fused into one pass over each prefix batch. This avoids walking
`prefix.selected` twice for every dynamic frontier. The live top-1 beam path
improved again:

```text
beam width 1, full-table top-1:       ~51.20 ms/window
beam width 4, full-table top-1:       ~49.57 ms/window
beam width 8, full-table top-1:       ~52.34 ms/window
plans match fast planner:             true
```

Live beam-planner A/B confirms that distinction. With a dynamic frontier, each
depth creates a new prefix batch, so graph capture cannot be reused and the
device-selection setup cost is paid every depth:

```text
script: scripts/perf_lab_batched_beam_planner.py
device=cuda, initial_targets=60, arrival_rate=4, seed=916,
branch_top_k=1, max_depth=24

fast cached planner:                 ~49.66 ms/window
beam width 1, full-table top-1:       ~58.01 ms/window
beam width 1, device top-1:           ~74.13 ms/window
beam width 8, full-table top-1:       ~61.50 ms/window
beam width 8, device top-1:           ~72.36 ms/window

plans match fast planner: true
```

After vectorizing prefix action tables, the isolated large-prefix GPU-selection
path improves, but the dynamic beam result is still not better:

```text
beam width 1, full-table top-1:       ~74.44 ms/window
beam width 1, device top-1:           ~79.27 ms/window
beam width 8, full-table top-1:       ~64.97 ms/window
beam width 8, device top-1:           ~80.21 ms/window
plans match fast planner:             true
```

After vectorized masks/slots, the normal full-table beam path improves, but the
dynamic device top-1 path still does not win:

```text
beam width 1, full-table top-1:       ~59.35 ms/window
beam width 1, device top-1:           ~72.29 ms/window
beam width 8, full-table top-1:       ~59.27 ms/window
beam width 8, device top-1:           ~78.31 ms/window
plans match fast planner:             true
```

After the slot-template optimization, the same live beam case remains fastest on
the full-table path:

```text
beam width 1, full-table top-1:       ~60.74 ms/window
beam width 1, device top-1:           ~73.61 ms/window
beam width 8, full-table top-1:       ~60.33 ms/window
beam width 8, device top-1:           ~73.60 ms/window
plans match fast planner:             true
```

So the online beam planner should keep using the full-table path unless the
frontier can be prepared and replayed. The device top-1 path is available as an
explicit switch for experiments, but it is not the default dynamic-beam path.

The staged profiler makes the bottleneck explicit. For batch 128, the full path
spent about `5.8 ms` in tokenization and `3.6-4.0 ms` in slot-feature
construction before the GPU model forward. The prepared path removes that
repeated CPU work; staged prepared scoring was about `5.7 ms` per call after a
one-time `~10.3 ms` prepare step.

## CUDA Graph Planner Replay

The online fast planner still had a different bottleneck from root expansion:
one 200 ms scheduling window makes many small sequential score calls. Each call
scores all candidate actions in parallel, but the calls are tiny GPU workloads,
so CUDA launch overhead is a major cost.

`FastActionAttentionPlanner(..., use_cuda_graph=True)` now captures the
fixed-shape per-decision score graph once per tensor shape and replays it inside
the sequential planning loop. The captured graph has static inputs for:

```text
encoded CLS/state tensor
encoded target tokens
selected-target mask
active-token mask
slot/window context vector
```

Each window copies the newly encoded state into those static buffers; each
decision copies the current slot vector and selected mask, then replays the
captured action-attention policy/Q scorer. The model math and selected action
are unchanged.

Clean A/B on the same root observation:

```text
script: scripts/perf_lab_cuda_graph_planner.py
device=cuda, initial_targets=40, arrival_rate=3, seed=916,
warmup=5, iterations=30

regular fast planner:             58.404 ms / plan
GPU-select fast planner:          50.509 ms / plan
CUDA graph fast planner:          10.093 ms / plan
CUDA graph + GPU-select planner:   9.171 ms / plan
best speedup:                      6.37x
plans match:                       true
```

The internal online profiler shows why this helps:

```text
regular per-decision score forward: ~2.5-2.6 ms
CUDA graph replay score forward:    ~0.38-0.39 ms
```

GPU-select is a smaller complementary optimization. Instead of copying the
full `[target_rows, sensors]` score table back to CPU and running NumPy
selection, the planner gathers valid physical actions and runs `argmax` on the
GPU, transferring only the selected action id. On its own it is neutral because
the uncaptured model forward dominates latency, but after CUDA Graph replay it
removes enough CPU/D2H work to improve the graph path.

The selector uses a precomputed flat index table:

```text
flat_index = target_row * 2 + sensor_id
candidate_score = score.reshape(-1)[flat_index]
```

This avoids two-dimensional advanced indexing in the hot loop. The internal
profile improved GPU-selection time from about `0.209 ms/decision` to
`0.185 ms/decision`.

`perf_lab_gpu_select_variants.py` compares the standard PyTorch ways to gather
candidate scores and pick the best action. On the 58-candidate table used by
the stress-cell root state, all tested variants selected the same action. The
fastest path was:

```text
vals = torch.take(score.reshape(-1), flat_index)
action = actions[torch.max(vals, dim=0).indices]
```

That replaces the earlier `index_select + argmax` selector.

The CUDA Graph replay path also reuses its static selected-mask input as the
live selected-mask tensor for the planning loop. That removes a per-decision
GPU mask copy before graph replay. The slot vector is likewise wrapped as a CPU
tensor once per plan call and reused for all graph replays. Internal timing for
the graph replay stage improved from about `0.345 ms/decision` to
`0.326 ms/decision`.

The action bookkeeping path now also precomputes an action-id to target-base
lookup table and the dwell array once per plan call. This removes repeated
`xs_decode_action(...)` calls and repeated `np.asarray(obs["t_dwell"])` wrapping
inside the decision loop. It is a small cleanup, but the profiled bookkeeping
stage dropped from roughly `0.022 ms/decision` to `0.021 ms/decision`.

`FastActionAttentionPlanner.warmup(obs, budget_ms)` now runs one untimed plan
with internal profiling disabled. For CUDA Graph mode, this pays kernel startup
and graph capture before the online control loop. In `profile_online_pipeline`,
the warmup call cost about `209 ms` once, while measured-window
`cuda_graph_prepare` dropped to about `0.086 ms/window` and root encoding stayed
near `1.5 ms/window`.

The fast planner also now builds a per-window slot-feature template. The
observation-dependent slot terms are constant throughout a planning call, so the
loop only updates:

```text
elapsed / budget
search_count / 20
track_count / 100
last_action_is_search
```

This is numerically equivalent to calling `slot_features(...)` every decision
for a fixed observation. In the internal profiler, slot construction dropped
from roughly `0.18 ms/decision` to `0.024 ms/decision`.

The graph path is optional because capture has a one-time cost and depends on
CUDA availability. It is most useful for repeated fixed-shape online planning,
where the graph cache is reused across windows.

End-to-end profiling of the current fast path shows the full control-loop
shape:

```text
script: scripts/profile_online_pipeline.py
planner: fast_graph_gpu_select
device=cuda, initial_targets=60, arrival_rate=4, seed=916,
windows=30, window_ms=200

planner plan:                14.71 ms/window
environment execution:        2.08 ms/window
state metrics:                0.12 ms/window
observation extraction:       0.05 ms/window
one-time warmup/capture:    220.05 ms
```

Planner-internal timing for the same run:

```text
CUDA graph score replay:      0.312 ms/decision
GPU action selection:         0.139 ms/decision
root transformer encode:      1.43 ms/window
root tokenization:            0.43 ms/window
slot feature update:          0.022 ms/decision
bookkeeping:                  0.021 ms/decision
```

The main online bottleneck is therefore still repeated per-decision score
replay plus GPU action selection. The simulator, observation extraction, and
window execution are not the limiting path in this configuration. For batched
root/frontier workloads, the bottleneck shifts to CPU preprocessing unless the
prepared-batch path is used.

## Beam Prefix Graph Profile

The batched beam path now has an optional profiler:

```text
python scripts/perf_lab_batched_beam_planner.py --device cuda --cuda-graph --profile
```

This exposes wall-clock timing plus internal beam stages:

- `beam_scorer_init`: root observation attach, tokenization, and transformer encoding
- `beam_expand_prefixes`: total next-prefix expansion cost
- `scorer_selected_valid`: selected-target mask and valid-action mask construction
- `scorer_slots`: per-prefix slot/context feature construction
- `scorer_score_slots`: neural policy/Q score replay
- `scorer_score_d2h`: score table transfer back to CPU
- `scorer_cpu_select`: valid-action argmax
- `expand_child_build_top1/topk`: child prefix construction
- `beam_filter_sort_prune`: beam sorting and pruning

The profile exposed a graph-cache bug in dynamic prefix scoring. The prefix
CUDA Graph cache was scoped to each per-window scorer, so every timed plan
recaptured graphs and produced large `scorer_score_slots` p99 spikes. Moving the
prefix graph cache to the long-lived planner and copying the root encoding into
static graph buffers fixed this.

Measured on CUDA with `initial_targets=60`, `arrival_rate=4`, `seed=916`,
`max_depth=24`, and `branch_top_k=1`:

```text
Before reusable prefix graph cache:
beam width 1: 107.95 ms/window
beam width 4: 100.51 ms/window
scorer_score_slots p50: ~0.30 ms, p99: ~80-88 ms

After reusable prefix graph cache:
fast one-pass planner: 14.25 ms/window
beam width 1:          19.96 ms/window
beam width 4:          21.42 ms/window
beam width 8:          23.23 ms/window
beam width 16:         22.88 ms/window
scorer_score_slots p50: ~0.34 ms, p99: ~0.52-0.56 ms
```

The next latency opportunity is no longer CUDA Graph capture. It is the repeated
per-depth prefix expansion loop: even after graph reuse, a 20-action window
still performs roughly 160 score/selection substeps in the profiled benchmark.
The one-pass fast planner remains the lower-latency deployment path; beam search
is now close enough to be useful for offline/teacher generation and small online
lookahead experiments.

## Clean Online CUDA Graph Timing

The internal profiler is useful for attribution, but it synchronizes around
every stage. Clean wall-clock timing without per-stage synchronization is much
lower. On the same stress root state (`initial_targets=60`, `arrival_rate=4`,
`seed=916`, `iters=40`, `warmup=8`):

```text
base fast planner:              45.61 ms/window
GPU select only:                44.29 ms/window
CUDA graph:                     10.42 ms/window
CUDA graph + GPU select:         8.58 ms/window
```

All four variants selected the same 20-action plan. The graph+GPU-select path is
therefore the current lowest-latency one-pass deployment path, at about `5.3x`
faster than the uncaptured fast planner on this benchmark.

## Action Selection Kernel Notes

The GPU selector microbenchmark compares native PyTorch ways to gather candidate
action scores and pick the best action from the dense `[target, sensor]` score
table. On the same 60-target/rate-4 state with 82 valid candidates:

```text
index_select + max: 0.083 ms
take + max:         0.083 ms
topk(1):            0.088 ms
gather + argmax:    0.090 ms
index_select+argmax 0.090 ms
```

With a nonzero search bias, all variants clustered around `0.131-0.133 ms`.
This means the current PyTorch selector is already close to the best available
native path.

I also tested moving `take -> bias -> argmax -> action` inside the captured CUDA
Graph. On this Windows/CUDA/PyTorch stack, that operation fails during graph
capture with:

```text
RuntimeError: CUDA error: operation failed due to a previous error during capture
```

A fused action-select CUDA kernel could still remove kernel-launch overhead and
return the action more directly, but the practical upper bound is modest: the
clean graph+GPU-select planner is already `8-9 ms/window`, and the remaining
selector work is partly the unavoidable CPU synchronization needed to update the
next selected-target mask and window bookkeeping.

## Multi-Environment Online Batching

Single-window planning is sequential because each chosen action changes elapsed
time, selected targets, search/track counts, and the next valid action set. The
next useful parallelization level is therefore independent windows/environments:
batch one decision depth across many environments, step each simulator, then
batch the next decision depth.

`perf_lab_multi_env_online_batch.py` compares three paths:

- `serial_fast_graph_gpu_select`: current low-latency one-pass planner per env
- `batched_multi_env_reencode`: batch every decision depth but re-tokenize and
  re-encode roots each depth
- `batched_multi_env_cached_root`: encode all environment roots once per window,
  then batch only changing selected masks and slot/context features each depth

The cached-root path is the useful one. It preserves the within-window decision
dependency while using the GPU across independent envs. On CUDA with
`initial_targets=60`, `arrival_rate=4`, `seed=916`, and 20 decisions/window:

```text
envs  windows  serial env-w/s  cached env-w/s  speedup  reward delta
4     3          87.55           55.76          0.64x    0.0
8     5          49.63           76.02          1.53x    0.0
16    5          56.83          133.42          2.35x    0.0
32    5          97.49          221.24          2.27x    0.0
64    3          96.57          305.22          3.16x    0.0
```

The small-batch case loses because serial CUDA Graph replay is extremely fast
for one environment. Once the batch reaches roughly 8+ environments, root-cache
batching wins and the per-env neural planning cost drops sharply:

```text
8 envs:   0.483 ms/env-action
16 envs:  0.227 ms/env-action
32 envs:  0.103 ms/env-action
64 envs:  0.053 ms/env-action
```

The naive re-encode path is included as a negative control. It matches reward,
but spends about `6-10 ms` per batched decision round because it repeats the
root transformer encode every depth. The cached-root path is the correct
parallelization: encode target/context tokens once per environment window, then
parallelize policy/Q scoring for all active environments at each decision depth.

### Cached Multi-Env CUDA Graph Attempt

`perf_lab_multi_env_online_batch.py` also includes
`batched_multi_env_cached_root_graph`, which captures the score-only part of a
cached-root multi-env decision round. The graph deliberately does not include
`gather/argmax` action selection because PyTorch's selection path fails CUDA
Graph capture on this stack.

The graph path improves the inner score/replay round but loses end-to-end due
per-window graph capture:

```text
envs  cached round ms  graph round ms  graph build ms/window  cached env-w/s  graph env-w/s
4       2.73             0.96            72.35                  56.97          37.13
16      4.10             1.66            76.88                 126.79         104.93
64      3.62             2.47            78.05                 295.34         236.00
```

So the conclusion is precise: CUDA Graph replay helps the batched score kernel,
but rebuilding a graph for every root window is too expensive. It would become
useful only if fixed-shape graph capture can be reused across many windows with
static root buffers, or for offline rollouts where many repeated decision rounds
share one prepared root batch.

That reusable version is now implemented for the multi-env cached-root graph
path. The graph cache is keyed by tensor shape and dtype; root encodings,
selected masks, active masks, and slot tensors are copied into static graph
buffers before replay. A direct score equivalence check matched the raw scorer
exactly:

```text
max_abs_diff: 0.0
allclose:     true
```

On short runs the first capture still hurts wall-clock throughput, but once the
capture is amortized the graph path wins. With direct root packing and fast env
stepping:

```text
64 envs, 60 targets, rate 4, 20 windows
raw cached root:       ~680.1 env-windows/s
reusable score graph:  ~870.8 env-windows/s
reward delta:          0.0
```

Per-decision planning rounds improved at the same setting:

```text
raw cached root:       ~3.39 ms/round
reusable score graph:  ~2.17 ms/round
```

This makes reusable CUDA Graph replay useful for longer batched eval/training
runs. For very short smoke tests, `--skip-graph` remains useful because one graph
capture can dominate only a few windows.

The reusable graph path now also profiles its own stages and reuses fixed-shape
device buffers for the dense physical action table (`actions`, flattened score
indices, and validity mask). This removes repeated GPU allocation for candidate
tables in the full-batch replay case. The 16-env profiled smoke showed:

```text
graph_score_replay:          ~0.64 ms
graph_decision_select_device:~0.28 ms
graph_action_tensor_prep_h2d:~0.10 ms
graph_slot_h2d:              ~0.06 ms
```

On the 64-env, 20-window run, the long-run graph throughput improved further:

```text
reusable score graph before action buffers: ~870.8 env-windows/s
reusable score graph with action buffers:   ~909.4 env-windows/s
reward delta:                               0.0
```

A CPU-slot shortcut was tested and rejected: copying CPU slot tensors directly
inside graph replay moved the transfer into the replay stage and reduced the
64-env long-run throughput, so the graph path keeps the explicit `graph_slot_h2d`
stage.

The selector was then simplified to the minimum valid-action path. Because the
dense candidate table always includes at least one valid search action and model
scores are finite for valid actions, the selector no longer needs per-row
`isfinite`, `any`, and fallback `where` checks:

```text
candidate_scores = gather(score, flat_indices)
candidate_scores = mask_invalid(candidate_scores, -inf)
idx = argmax(candidate_scores)
action = gather(actions, idx)
```

This is still pure PyTorch, but it removes several small kernels from every
decision depth. The 64-env profiled run showed:

```text
raw cached decision_select_device:   ~0.56 ms -> ~0.13 ms
graph decision_select_device:        ~0.60 ms -> ~0.14 ms
reward delta:                         0.0
```

The 64-env, 20-window long run with reusable graph, action buffers, and minimal
selector measured:

```text
cached root graph throughput: ~911.3 env-windows/s
planning round:               ~2.00 ms
planning ms per env-action:   ~0.0313 ms
reward delta:                 0.0
```

At this point a custom CUDA kernel for selection is unlikely to be the next
large win; the selector is around `0.14 ms` in the profiled graph path, while
`graph_score_replay` remains much larger. A fused selection kernel can still be
revisited later, but the next high-leverage target is model replay itself.

The graph path now also has a reusable root-encoder graph. Root token tensors
are copied into a static graph input and `backbone.encode_tokens(...)` is
replayed by tensor shape. A direct equivalence check matched the raw encoder
exactly:

```text
cls max_abs_diff: 0.0
tok max_abs_diff: 0.0
selected equal:   true
active equal:     true
```

This is a short-run tradeoff because the first root graph capture is expensive.
On the 64-env, 5-window profiled run, capture dominated the graph path and hurt
wall time. On the longer 64-env, 20-window run, capture was amortized:

```text
root raw encode p50:        ~1.23 ms
root graph encode p50:      ~0.77 ms
cached root graph throughput: ~914.7 env-windows/s
planning round:               ~1.92 ms
planning ms per env-action:   ~0.0300 ms
reward delta:                 0.0
```

The improvement over score-only graph replay is modest, but it keeps moving
fixed-shape model work into reusable graph capture. For short smoke tests,
`--skip-graph` or raw cached-root timing remains easier to interpret.

The same benchmark now supports whole-path Python profiling with
`--profile-cpu-top N` in addition to synchronized stage timers. The profiled
command was:

```bash
python scripts/perf_lab_multi_env_online_batch.py --device cuda --envs 64 --windows 20 --initial-targets 60 --rate 4 --amp --fast-env-step --direct-root-pack --profile-stages --profile-cpu-top 20 --out results/perf_lab_multi_env_profile_full_steps.json
```

Because `--profile-stages` synchronizes around every timed block, this run is
for attribution rather than headline throughput. Reward still matched the
serial reference exactly. The optimized graph path stage profile was:

```text
graph_score_replay:            mean 1.90 ms, p50 1.58 ms
graph_env_step_batch:          mean 1.10 ms, p50 1.08 ms
graph_root_pack_direct:        mean 0.82 ms, p50 0.79 ms
graph_root_tokenize_batch:     mean 0.63 ms, p50 0.61 ms
graph_physical_action_table:   mean 0.39 ms, p50 0.38 ms
graph_root_slot_template:      mean 0.29 ms, p50 0.27 ms
graph_decision_select_device:  mean 0.17 ms, p50 0.16 ms
graph_action_tensor_prep_h2d:  mean 0.16 ms, p50 0.15 ms
graph_slot_h2d:                mean 0.12 ms, p50 0.09 ms
graph_slot_context_update:     mean 0.08 ms, p50 0.08 ms
```

The matching Python cumulative profile for the graph path confirms that most
remaining non-model time is in environment stepping and physical action table
construction:

```text
sync / cuda synchronize:             profiling overhead from synchronized timers
step_envs:                           ~425 ms cumulative
execute_known_valid_action_fast:     ~275 ms cumulative
binding.vec_step:                    ~229 ms cumulative
physical_action_table_from_packed:   ~142 ms cumulative
```

Current opportunity order:

1. Reduce graph score replay latency further, likely by simplifying the cached
   action-attention score graph or moving more of the static gather/mask work
   into capture.
2. Batch or vectorize environment stepping across envs; we still call
   `binding.vec_step` once per executed action.
3. Move direct root packing/tokenization closer to the C buffers or a reusable
   tensor buffer.
4. Replace Python physical action-table construction with a preallocated or C
   backed path.

An experimental selected-index C batch step was added behind
`--batch-env-step`. It creates a non-owning vector view over the existing
one-env handles and calls `vec_step_selected_validated_into(...)` once per
decision depth instead of calling `vec_step(...)` once per env. The important
correctness result is clean:

```text
16 envs x 5 windows: reward delta graph minus serial = 0.0
64 envs x 20 windows: reward delta graph minus serial = 0.0
```

The synchronized stage profile shows the env-step bucket can drop from roughly
`1.10 ms` to `0.39 ms` at 64 envs, but the no-profile headline throughput was
not a consistent win on this machine:

```text
no batch env step:  ~829.6 env-windows/s
batch env step:     ~810.5 env-windows/s
```

So this remains an opt-in experiment. It proves the selected-step API is
semantically usable, but the current online path is still dominated by graph
score replay and Python/tensor staging enough that the OpenMP selected-step
call does not obviously improve end-to-end throughput.

An inference-only paired policy/Q head path was also tested behind
`--paired-heads`. It packs same-architecture policy and Q MLP pairs into cached
block-diagonal projections for type, target, and action-residual heads.

Important correction: the first paired-head and direct-coupler benchmark only
affected the older `_scores_from_encoded` helper. The online graph path calls
`_combined_scores_from_encoded`, so the flags did not affect the real hot path
until the scorer was fixed. After applying the flags to the combined scorer,
the direct cached-score equivalence check showed:

```text
direct couplers: max_abs_diff 0.0, allclose true
paired heads:    max_abs_diff 0.00048828125, allclose false under fp16 AMP
```

Because paired heads are not bit-close under the strict AMP check, they remain
an experimental path rather than a recommended speed path.

The corrected direct-coupler path is exact but did not improve the full
64-env, 20-window graph benchmark:

```text
gpu action template, no direct couplers: ~944.8 env-windows/s
gpu action template, direct couplers:    ~907.7 env-windows/s
reward delta:                            0.0
```

This suggests the outer `TransformerEncoder` wrapper is not the bottleneck once
CUDA graphs are active. The action self-attention kernels and surrounding
tensor staging remain the higher-value targets.

Two additional graph-path switches were tested:

```text
--direct-couplers
--cached-action-table
```

`--direct-couplers` bypasses the outer `TransformerEncoder` wrapper for the
one-layer sensor/action couplers and calls the contained
`TransformerEncoderLayer` directly. A direct cached-score equivalence check was
exact:

```text
score shape:       (1, 101, 2)
max_abs_diff:      0.0
allclose:          true
```

`--cached-action-table` moves per-window physical action sorting/layout out of
the per-decision loop. The action order is fixed from the root state, and each
decision only masks targets already selected in the current window. This was
also reward-equivalent:

```text
16 envs x 5 windows: reward delta graph minus serial = 0.0
64 envs x 20 windows: reward delta graph minus serial = 0.0
```

The profiled physical table stage improved substantially:

```text
old graph_physical_action_table:       ~0.34-0.39 ms per decision
cached graph_physical_action_table:    ~0.06 ms per decision
one-time graph_physical_action_template: ~0.25 ms per window
```

The cached action table path now also has `--gpu-action-template`, which keeps
the fixed per-window action IDs and score indices resident on GPU. It can be
paired with `--gpu-valid-mask`, which derives the per-decision validity mask on
GPU from the fixed action base indices and the current selected-target mask.
This avoids rebuilding and uploading the changing validity mask from CPU each
decision.

The smoke and 64-env runs preserved reward parity:

```text
16 envs x 5 windows:  reward delta graph minus serial = 0.0
64 envs x 20 windows: reward delta graph minus serial = 0.0
```

The best measured command before C env-step batching was:

```bash
python scripts/perf_lab_multi_env_online_batch.py --device cuda --envs 64 --windows 20 --initial-targets 60 --rate 4 --amp --fast-env-step --direct-root-pack --cached-action-table --gpu-action-template --gpu-valid-mask
```

Result:

```text
graph throughput:          ~954.0 env-windows/s
planning ms/env-action:    ~0.0307 ms
reward delta:              0.0
```

Precomputing the GPU validity-mask gather indices and search-action mask inside
the action template gave:

```text
graph throughput:          ~955.9 env-windows/s
planning ms/env-action:    ~0.0305 ms
reward delta:              0.0
```

The original `--batch-env-step` path used the validated C batch function. It was
reward-equivalent and changed the graph path from 25600 scalar env-step calls
to 400 batch env-step calls, but it was slower overall:

```text
--batch-env-step graph throughput:       ~926.4 env-windows/s
--batch-env-step planning ms/env-action: ~0.0326 ms
reward delta:                            0.0
```

That exposed the mismatch: the scalar fast path was already operating on
known-valid model-selected actions, while the batch path was repeating wrapper
validation. A new `vec_step_selected_known_valid_into` C function now batches
the same known-valid transition used by the scalar fast path.

The new recommended command is:

```bash
python scripts/perf_lab_multi_env_online_batch.py --device cuda --envs 64 --windows 20 --initial-targets 60 --rate 4 --amp --fast-env-step --direct-root-pack --cached-action-table --gpu-action-template --gpu-valid-mask --batch-env-step
```

Two clean 64-env, 20-window runs preserved reward parity:

```text
run 1 graph throughput:       ~1051.2 env-windows/s
run 2 graph throughput:       ~1057.2 env-windows/s
planning ms/env-action:       ~0.0309-0.0311 ms
reward delta graph-serial:    0.0
batch env-step calls:         400
scalar env-step calls:        0
```

This is the strongest end-to-end result so far in the current clean repo. The
synchronized profile is useful for bottleneck ranking, but it inserts many CUDA
synchronizations and should not be used as the headline latency number. On the
64-env, 10-window profiled graph path, the largest per-decision stages were:

```text
graph_score_replay:             ~1.84 ms mean
graph_root_pack_direct:         ~1.38 ms mean
graph_root_tokenize_batch:      ~1.18 ms mean
graph_env_step_batch:           ~0.70 ms mean
graph_action_template_h2d:      ~0.53 ms mean, once per root/window
graph_root_slot_template:       ~0.51 ms mean, once per root/window
graph_physical_action_template: ~0.50 ms mean, once per root/window
graph_decision_select_device:   ~0.26 ms mean
graph_action_tensor_prep_h2d:   ~0.20 ms mean
```

The next high-value targets are model replay and CPU root packing/tokenization.
The action-template, validity-mask, and known-valid batch-step optimizations
mainly remove repeated tensor staging and Python/C call overhead around
selection and environment stepping.

An internal scorer profile also tested `torch.compile --mode reduce-overhead`
on the current AMP cached-score shape. It did not improve the full cached score:

```text
no compile full cached score: ~2.30 ms
compile full cached score:    ~2.51 ms
```

So `torch.compile` is not part of the recommended path on this stack.

The online benchmark and internal cached-score profiler now accept
`--checkpoint` for trained `ActionAttentionFactorizedNet` state dicts. This is
important because the default performance lab uses a freshly initialized model,
while trained action-attention checkpoints have nonzero action residual heads.

Example trained-model profiler command:

```bash
python scripts/profile_cached_action_attention_internals.py --device cuda --amp --prefix-batches 64 --iters 80 --warmup 20 --initial-targets 60 --rate 4 --checkpoint ../CreateValid1/results/critic_bootstrap_medium_eval_two_row_action_attention_qpolicy_factored_loss.pt
```

On that checkpoint, the 64-prefix cached-score profile was similar to the
default initialized model:

```text
full_cached_score:      ~2.36 ms
action_self_attention:  ~0.88 ms
sensor_coupling:        ~0.46 ms
target_heads:           ~0.32 ms
base_score_build:       ~0.33 ms
residual_heads:         ~0.30 ms
```

The trained checkpoint also exposed a correctness issue in the benchmark
harness: the selected-target mask returned from an inference-mode encoder was
later mutated as actions were selected. The benchmark now clones that mask after
root encoding so it remains mutable.

Partial-live CUDA graph replay was tested for trained checkpoints because many
envs finish early, reducing the live batch size. It removed raw score rounds but
was slower overall because many distinct live-batch shapes triggered graph
capture and produced high p99 stalls. The optimization was rejected; partial
live batches continue to use the eager raw scoring path.

`perf_lab_coupler_variants.py` isolates the one-layer sensor/action couplers.
For the action coupler shape used by the main model (`64 x 202 x 48`), a manual
single-layer path that calls the layer's `self_attn`, norms, and feed-forward
modules directly was exact against `TransformerEncoder`:

```bash
python scripts/perf_lab_coupler_variants.py --device cuda --amp --prefixes 64 --checkpoint ../CreateValid1/results/critic_bootstrap_medium_eval_two_row_action_attention_qpolicy_factored_loss.pt
```

Measured action-coupler-only timings on the trained checkpoint:

```text
TransformerEncoder: ~1.43 ms
direct layer:       ~1.35 ms
manual layer:       ~1.32 ms
max abs diff:       0.0
```

The online benchmark exposes this as `--manual-couplers`. It is exact in the
full combined scorer, but the end-to-end gain is small and noisy because CUDA
graph build/replay, environment stepping, and root packing also contribute. It
is kept as an opt-in lab switch rather than replacing the recommended command.

`--padded-live-graph` keeps using the fixed full-batch score CUDA graph after
some environments in the batch finish their 200 ms window. The previous graph
path replayed only while every root in the batch was live; partial live batches
fell back to raw eager scoring:

```text
full live batch:    replay fixed score graph
partial live batch: raw score forward
```

The padded path instead builds a full slot batch, replays the same graph, then
gathers the live rows:

```text
full slot batch -> fixed score graph -> live-row gather -> action selection
```

This does extra GPU work for inactive rows, but it avoids many slow partial
eager forwards and preserves fixed-shape graph replay. On the trained action
attention checkpoint:

```bash
python scripts/perf_lab_multi_env_online_batch.py --device cuda --envs 64 --windows 20 --initial-targets 60 --rate 4 --amp --fast-env-step --direct-root-pack --cached-action-table --gpu-action-template --gpu-valid-mask --batch-env-step --checkpoint ../CreateValid1/results/critic_bootstrap_medium_eval_two_row_action_attention_qpolicy_factored_loss.pt
python scripts/perf_lab_multi_env_online_batch.py --device cuda --envs 64 --windows 20 --initial-targets 60 --rate 4 --amp --fast-env-step --direct-root-pack --cached-action-table --gpu-action-template --gpu-valid-mask --batch-env-step --padded-live-graph --checkpoint ../CreateValid1/results/critic_bootstrap_medium_eval_two_row_action_attention_qpolicy_factored_loss.pt
```

Same-condition clean A/B:

```text
base graph path:        ~383.9 env-windows/s
padded live graph path: ~731.6 env-windows/s
raw score rounds:       221 -> 0
reward delta:           0.0
executed actions delta: 0
```

A synchronized 64-env, 10-window profile confirmed that the raw partial-score
bucket disappeared:

```text
graph_score_replay:        ~1.86 ms mean
graph_padded_score_replay: ~2.06 ms mean
raw score forward:         0 calls
```

The padded branch also reuses a fixed GPU slot buffer and caches live-position
tensors for repeated active-set patterns. With this change, the recommended
multi-env graph benchmark includes `--padded-live-graph`.

The follow-up GPU-valid-mask change extends the cached action template to
partial live batches too. Instead of rebuilding the physical action table on CPU
after roots finish, the graph path indexes the full GPU action/gather template
down to live rows and computes validity from `selected_t_all` on GPU:

```text
actions/flat/gather template [root, action]
    -> live row index_select
    -> selected-target gather
    -> valid action mask
```

This keeps partial live decisions on the device for both scoring and action
masking. Same trained-checkpoint 64-env, 20-window run:

```text
padded live graph before partial GPU validity: ~731.6 env-windows/s
padded live graph with partial GPU validity:   ~898.0 env-windows/s
raw score rounds:                              0
reward delta:                                  0.0
executed actions delta:                        0
```

The graph path now also maintains the per-window slot/context matrix
incrementally. Only the first four slot features change inside a scheduling
window (`elapsed`, `search_count`, `track_count`, and `last_action_is_search`);
the other slot features come from the root state. Updating those four columns
after each executed action avoids rebuilding slot rows from Python lists every
decision:

```text
slot_template.copy()
    -> update columns 0..3 after each action
    -> score graph consumes current slot rows
```

This is a small cleanup, not a headline speedup. In the synchronized 64-env
profile, `graph_slot_context_update` dropped from roughly `0.12 ms` to
`0.05 ms` per decision, while the clean 64-env, 20-window headline stayed around
`900 env-windows/s` with identical reward and executed action count.

Direct root packing now uses a dense C aux-array path for sensor busy timers,
X-band availability, and target ranges. The previous direct pack path still
called `vec_aux` once per environment, which rebuilt Python dictionaries and
range lists. The new binding returns dense NumPy arrays in one call:

```text
vec_aux_arrays(vec)
    -> s_band_busy_ms [env]
    -> x_band_busy_ms [env]
    -> enable_x_band [env]
    -> target_range [env, target]
```

The Python root packer reuses the existing batch-step vector handle when the
whole environment batch is still live, and falls back to a temporary vector view
or old `vec_aux` path when needed. A direct comparison against the old per-env
aux dict path matched exactly:

```text
sample shape: 4 x 100 target ranges
max abs diff: 0.0
```

On the trained-checkpoint 64-env, 20-window clean benchmark, the recommended
graph path improved from roughly `902` to `929 env-windows/s`, again with
identical reward and executed action count.

`perf_lab_attention_backend_variants.py` tests PyTorch SDPA backend toggles for
the current cached score shape. On this stack (`torch 2.7.1+cu118`, 64 envs,
101 target rows), all tested backends were bit-exact versus default. The longer
run showed only noise-level differences:

```text
default:            ~2.318 ms cached score
mem_efficient_only: ~2.299 ms cached score
math_only:          ~2.327 ms cached score
flash_only:         ~2.329 ms cached score
cudnn_only:         ~2.342 ms cached score
```

The best case was less than 1% faster than default, so no attention backend is
forced. The model path should stay on PyTorch defaults unless a future runtime
or model shape changes this result.

Mixed precision was also tested on the current direct-pack/fast-step path and
rejected on this Windows/CUDA/PyTorch stack:

```text
64 envs, 60 targets, rate 4, 3 windows
fp32 cached root: ~757.2 env-windows/s
AMP cached root:  ~544.1 env-windows/s

profiled decision_score_forward:
fp32: ~3.2 ms
AMP:  ~5.5 ms
```

So the current default should stay fp32 unless the runtime/model changes.

### Cached Multi-Env Stage Profile

`perf_lab_multi_env_online_batch.py --profile-stages --skip-graph` splits the
winning cached-root path into synchronized stages. This is not the clean
throughput mode, but it shows where the next optimization work should go.

Profiled at `initial_targets=60`, `arrival_rate=4`, `seed=916`:

```text
stage                         32 envs mean ms   64 envs mean ms
env_step_batch                  2.21             4.35
decision_score_forward          2.33             2.58
root_tokenize_batch             1.66             3.14
slot_features_batch             1.07             2.03
root_obs_attach                 0.95             2.01
root_h2d_encode                 1.51             1.42
decision_tensor_prep_h2d        0.39             0.49
physical_action_table_batch     0.28             0.46
decision_select_device          0.31             0.34
decision_action_d2h             0.06             0.06
```

This changes the optimization picture. At high batch counts, neural score
forward is no longer the only bottleneck. Simulator stepping, root observation
attachment, root tokenization, and slot feature construction are now comparable
or larger. The next major speedup should therefore batch/vectorize simulator
execution and feature construction rather than only shaving more microseconds
from policy/Q scoring.

### Slot Template Optimization

Inside a cached-root window, most slot features are root-state constants:
active target count, tracked target count, workload, deadline pressure, busy
times, and arrival/load features. Only four fields change at each decision:

```text
elapsed / budget
search_count / 20
track_count / 100
last_action_is_search
```

`perf_lab_multi_env_online_batch.py` now computes a per-window slot template
once and updates only those four columns for live environments at each decision
depth. This preserves the exact chosen actions/reward while avoiding repeated
full `slot_features_batch(...)` calls.

64-env profiled result:

```text
old slot_features_batch:  2.03 ms/decision round
new slot_context_update:  0.046 ms/decision round
root_slot_template:       2.02 ms/window
```

Clean 64-env throughput improved from about `305 env-windows/s` to
`368 env-windows/s` on the same 60-target/rate-4 benchmark, with zero reward
delta versus serial execution. The remaining large stages are now simulator
stepping, root tokenization, and policy/Q score forward.

### Root Tokenizer Fast Path

The cached-root benchmark only tokenizes root windows: no targets are selected
yet and root `search_count` is zero. `perf_lab_multi_env_online_batch.py` now
uses a root-only tokenizer for this path. It preserves the same root tensor as
the general tokenizer, including the root-token normalization and optional grid
feature override.

Validation check on an 8-env batch:

```text
tokenize_batch shape:          (8, 101, 13)
tokenize_root_batch_fast shape:(8, 101, 13)
max_abs_diff:                  0.0
allclose:                      True
```

Corrected 64-env profile:

```text
root_tokenize_batch: 2.89 ms/window batch
clean cached throughput: 366.86 env-windows/s
reward delta vs serial:  0.0
```

The improvement is smaller than the slot-template change because most tokenizer
time is real array work: stacking target arrays, computing sector urgency from
the grid, and filling the token tensor. The next tokenization speedup should
therefore come from a broader packed observation representation rather than more
root-only branches.

### Packed Root Observations

The cached-root benchmark now builds a compact `PackedRootObs` once per
environment-window batch and derives both root tokens and slot templates from
that packed array structure. This keeps exact equivalence with the general
list-of-dict feature path while avoiding repeated dictionary lookups and array
stacking in separate token/slot functions.

Validation on an 8-env root batch:

```text
token_max_abs_diff: 0.0  allclose=True
slot_max_abs_diff:  0.0  allclose=True
```

64-env profiled stage changes:

```text
root_tokenize_batch: 2.89 ms -> 0.54 ms
root_slot_template:  1.96 ms -> 0.22 ms
root_pack_observations:          0.61 ms
```

Clean 64-env cached throughput is now about `381 env-windows/s`, versus roughly
`368 env-windows/s` after slot-template caching alone and `305 env-windows/s`
before the cached feature work. Reward remains identical to serial execution.

The physical action table now also consumes the same packed root arrays instead
of rebuilding active/deadline/range/free-sensor stacks from observation
dictionaries at every decision depth. Direct equivalence checks matched the
legacy `physical_action_table_batch` output for action ids, target base ids,
sensor ids, and validity masks.

64-env profiled result after the packed physical table:

```text
physical_action_table_batch: 0.46 ms -> 0.31 ms
clean cached throughput:    ~384.6 env-windows/s
reward delta vs serial:     0.0
```

The improvement is real but modest. The current high-batch bottlenecks are now:

```text
env_step_batch:         ~4.26 ms
decision_score_forward: ~3.11 ms
root_obs_attach:        ~1.90 ms
root_h2d_encode:        ~1.54 ms
```

An additional `cProfile` run confirms the same hot areas after import overhead:
GPU action selection, tensor transfers, transformer/linear layers, and
observation extraction dominate runtime. The next larger wins should therefore
come from reducing simulator/observation overhead and keeping more of the
batched state representation resident, not from further scalar action-table
micro-optimizations.

### Fast Validated Env Step

The generic `execute_first_valid_action` path is deliberately defensive: for
each action it reads a full observation before execution, revalidates target and
sensor constraints, steps the C environment, reads another full observation, and
infers elapsed time from observation deltas. In the packed cached-root planner,
that validation has already happened in the dense physical action table. The
benchmark now has an opt-in `--fast-env-step` path:

```text
validated candidate action
    -> one C environment step
    -> elapsed time from search dwell or target dwell
    -> no per-action before/after observation rereads
```

This is not a replacement for the general executor; it is valid for this
benchmark path because invalid actions are masked before selection. A/B runs
matched total reward and executed action count against the defensive executor.

Measured result on CUDA with `64` envs, `60` initial targets, arrival rate `4`,
and `3` windows:

```text
cached root, defensive env step: ~384.6 env-windows/s
cached root, fast env step:      ~738.1 env-windows/s
reward delta vs serial:          0.0
```

The synchronized 64-env profile shows the simulator/control stage is no longer
dominant:

```text
env_step_batch:         ~4.26 ms -> ~0.85 ms
decision_score_forward: now the largest stage, ~3.41 ms
root_obs_attach:        ~1.85 ms
root_h2d_encode:        ~1.85 ms
```

This changes the next optimization target back toward the neural score path and
root observation/encoding residency. The high-batch planner is now mostly
limited by batched action-attention inference and root-state refresh, not by
per-action simulator bookkeeping.

### Direct Root Packing

The cached-root path also has an opt-in `--direct-root-pack` mode. Instead of:

```text
C obs buffer -> get_obs_from_buf dict -> attach_env_obs dict -> PackedRootObs arrays
```

it parses the C observation buffers directly into `PackedRootObs`:

```text
C obs buffers [env, raw_obs]
    -> grid tensor
    -> tracker tensor [env, target, desired/deadline/dwell/priority/az/el]
    -> aux busy/range arrays
    -> PackedRootObs
```

Equivalence checks matched the dictionary path exactly for:

```text
packed arrays
token tensors
slot templates
physical action ids/base ids/sensor ids/valid masks
```

Measured with `--fast-env-step`, CUDA, `64` envs, `60` initial targets, arrival
rate `4`, and `3` windows:

```text
fast env step + dictionary root pack: ~738.1 env-windows/s
fast env step + direct root pack:     ~757.2 env-windows/s
reward delta vs serial:               0.0
```

The stage profile now has a single root packing stage:

```text
root_obs_attach + root_pack_observations: removed
root_pack_direct:                         ~0.61 ms
```

This confirms the packed state representation is the right direction, but the
remaining large costs have moved back to neural scoring and root encoding. The
next high-leverage step is to reduce `decision_score_forward` or avoid
re-encoding unchanged root state across repeated eval/training batches.

### Graph Replay and Batched Online Path

The current fastest benchmark path combines:

```text
direct packed root observations
    + cached root encodings
    + cached/GPU physical action templates
    + GPU valid masks
    + padded live CUDA Graph replay
    + batched C environment stepping
```

Recommended command:

```powershell
python scripts\perf_lab_multi_env_online_batch.py --device cuda --envs 64 --windows 20 --initial-targets 60 --rate 4 --amp --fast-env-step --direct-root-pack --cached-action-table --gpu-action-template --gpu-valid-mask --batch-env-step --padded-live-graph --checkpoint ..\CreateValid1\results\critic_bootstrap_medium_eval_two_row_action_attention_qpolicy_factored_loss.pt --out results\perf_lab_64x20.json
```

The best clean 64-env/20-window run after dense auxiliary root arrays measured:

```text
graph throughput:          ~928.6 env-windows/s
planning ms/env-action:    ~0.0326
reward delta vs cached:     0.0
executed actions matched:   24449
```

A later run with zero-search-bias guards measured `~892.4 env-windows/s` with
the same cached-root and graph reward/action count. This is within observed
run-to-run noise and confirms the guard is behavior-preserving.

The zero-search-bias guard skips a default no-op score update:

```text
if search_score_bias != 0:
    score[:, search_row, :] += search_score_bias
```

It is small, but it removes an avoidable GPU add kernel from the default
configuration and keeps benchmark, reusable fast planner, branch expansion, and
profiling helpers consistent.

The GPU action-template upload also avoids a duplicated valid-mask transfer in
the default benchmark path. The template valid mask is uploaded once and the
mutable per-decision valid buffer is allocated with `empty_like` on device:

```text
template_valid_t = H2D(valid_mask)
valid_t = empty_like(template_valid_t)
```

A 64-env/20-window run with this cleanup matched cached-root reward/action
counts exactly and measured `~887.9 env-windows/s`, within the same run-to-run
band as the previous `~892.4 env-windows/s` run.

Two score-body alternatives were profiled but not promoted:

```text
direct/manual coupler calls: exact, but not faster than the current graph path
paired-head execution: faster in places, but changed decisions/reward
```

Paired-head execution therefore remains an experimental approximation, not an
online replacement for the exact policy/Q head path.

The attention backend lab now accepts the trained checkpoint and AMP settings,
and the online multi-env benchmark has a matching `--sdp-backend` switch. On
the isolated score-body microbenchmark, disabling cuDNN SDP was slightly faster:

```text
default SDP:      ~2.394 ms/score batch
all_no_cudnn SDP: ~2.349 ms/score batch
```

However, the full online graph smoke run with `--sdp-backend all_no_cudnn`
was slower than the default while preserving reward/actions. This means SDP
backend tuning should stay an explicit benchmark knob for now; the default
runtime path should keep PyTorch's default backend selection.

The score-body compile lab checks whether `torch.compile` can fuse the cached
policy/Q score path:

```powershell
python scripts\perf_lab_score_compile_variants.py --device cuda --amp --prefixes 64 --checkpoint ..\CreateValid1\results\critic_bootstrap_medium_eval_two_row_action_attention_qpolicy_factored_loss.pt
```

On the current Windows environment, Inductor cannot compile the score body
because Triton is unavailable. The script records that as a benchmark result
instead of crashing. This means whole-score fusion via `torch.compile` remains
blocked by the local runtime, not by model logic.

The compile lab also includes a TorchScript trace variant for fixed-shape cached
score calls. In AMP mode it was not exact (`max_abs` around `8e-3` to `1.6e-2`)
and did not reliably beat eager execution in the scripted benchmark, so it is
also rejected as an online planner path.

`profile_score_kernels.py` adds a kernel-level Torch profiler view of the cached
score body. On the trained AMP checkpoint at the 64-prefix shape, the top CUDA
costs are:

```text
scaled dot-product attention: ~8.18 ms / 8 active calls
linear/addmm stack:           ~4.80 ms / 8 active calls
safe softmax:                 ~3.96 ms / 8 active calls
layer norm:                   ~3.33 ms / 8 active calls
autocast copy/to kernels:     ~2.58-2.92 ms / 8 active calls
bmm:                          ~2.34 ms / 8 active calls
```

This confirms the remaining bottleneck is not a single Python loop. It is a
mix of real action self-attention work, many small MLP/LayerNorm launches, and
autocast conversion kernels. A custom fused inference module would need to
attack those groups directly.

`perf_lab_cuda_env_check.py` records whether this machine can actually build or
generate fused GPU code. Current result:

```text
GPU:             NVIDIA GeForce RTX 3080 Ti
PyTorch:         2.7.1+cu118
torch CUDA:      11.8
NVCC:            12.9 available
MSVC cl.exe:     not on PATH
Triton:          not installed
```

So there are two practical blockers for deeper fusion on this workstation:
`torch.compile`/Inductor GPU fusion cannot run without Triton, and custom
PyTorch CUDA extensions on Windows generally need MSVC `cl.exe` plus a tested
CUDA-toolkit/PyTorch ABI combination. The current repo therefore keeps custom
kernel work behind environment validation rather than adding unbuildable code.

`profile_batched_scorer_stages.py` now also compares CPU-prepared, per-call
host-to-device prepared, device-resident prepared, device-resident tensor-return,
and CUDA-graph replay paths. On the RTX 3080 Ti AMP smoke
(`--batch-sizes 8,32,64 --iters 10 --warmup 3`), the repeated score/select call
times were:

```text
batch  prepared CPU->GPU  device resident  device tensor return  device graph
8      5.71 ms            5.56 ms          5.47 ms               5.50 ms
32     6.03 ms            5.95 ms          5.81 ms               5.64 ms
64     6.35 ms            5.50 ms          5.39 ms               5.75 ms
```

The reliable optimization is keeping root tensors/action tables resident on the
GPU and, when a downstream batched search can consume it, returning the selected
action tensor without an immediate device-to-host synchronization. CUDA Graph
replay is shape-compatible and action-compatible in this probe, but not
consistently faster, so it should stay behind the measured opt-in path.

The online cached-root graph path still has one unavoidable host boundary today:
the C simulator step consumes NumPy action ids. `--pinned-action-d2h` adds a
preallocated pinned CPU transfer buffer for that boundary. On a paired
32-env/8-window CUDA AMP run with the trained action-attention checkpoint and
the full graph/cached-action-table/batched-env-step path:

```text
mode             env-windows/s  plan ms/env-action  action D2H mean
regular D2H      396.46         0.06581             0.06998 ms
pinned D2H       397.95         0.06443             0.06850 ms
```

This is intentionally a small opt-in optimization. It does not change reward or
executed action count in the paired run; it just removes repeated CPU transfer
allocation and stabilizes the GPU-to-simulator boundary.

The device selection path also now masks invalid gathered action scores
in-place before `argmax`. This preserves the same selected actions because the
gathered candidate-score tensor is throwaway. In the 32-env cached-root graph
path, the measured `graph_decision_select_device` stage dropped from about
`0.232 ms` to `0.188 ms`; reward and executed action count remained
`-531.6933881170116` and `4916`.

The cached-root graph path now also reuses a stable CPU tensor view over the
per-window slot matrix instead of rebuilding `torch.from_numpy(current_slots)`
inside the decision loop. This is exact because the tensor shares the same
NumPy backing store updated by the simulator bookkeeping. In the same 32-env
graph probe, `graph_slot_h2d` moved from about `0.0819 ms` to `0.0763 ms`; the
end-to-end timing was noisy, so this is tracked as a local allocation cleanup,
not a claimed whole-pipeline speedup.

Preallocating persistent GPU buffers for the per-window action template was
tested and rejected. It preserved reward/actions, but `graph_action_template_h2d`
became worse in the 32-env graph probe, increasing to about `1.06 ms` versus
the prior roughly `0.53 ms`. The direct `.to(device)` uploads are faster here
than several explicit slice `copy_` operations plus helper tensor updates.

Lower-precision model conversion was checked as a way to reduce autocast copies.
The fast planner now uses a dtype-safe invalid-action sentinel, preserving
`-1e9` for the current float32/AMP path while using the finite FP16 minimum only
when a score tensor is actually half precision. That makes FP16 score-body
experiments runnable instead of failing on mask overflow.

FP16 still is not promoted as the online path: in a 64-prefix score probe it
matched the sampled argmax but not full scores, and it was not faster in that
run. BF16 runs faster in the score microbenchmark, but changes scores and argmax
decisions substantially. Neither is an exact online replacement.

The same labs now also expose `--matmul-precision`. In the isolated score-body
profile, `--matmul-precision high` improved the mean cached score call:

```text
default precision: ~2.39 ms
high precision:    ~2.19 ms
```

But the online smoke run changed decisions/reward. It is therefore not an exact
runtime optimization for the current trained policy; keep it as an experimental
speed/accuracy knob only.

A no-copy CUDA Graph state prototype was also tested. The idea was to let the
captured score graph own the mutable selected-target and slot tensors, then
replay without copying full selected/slot state into static graph inputs every
round. The 16-env/5-window smoke run diverged from cached-root reward/actions,
so this path was rejected and not kept. Any future version needs a stronger
state-synchronization proof before promotion.

Current synchronized 64-env/10-window graph-stage profile:

```text
graph_score_replay:          ~1.90 ms
graph_padded_score_replay:   ~1.90 ms
graph_env_step_batch:        ~1.36 ms
graph_root_tokenize_batch:   ~1.22 ms
graph_root_pack_direct:      ~0.68 ms
graph_action_template_h2d:   ~0.68 ms
graph_physical_action_template: ~0.54 ms
graph_root_slot_template:    ~0.53 ms
graph_action_tensor_prep_h2d: ~0.28 ms
graph_decision_select_device: ~0.27 ms
```

This puts the next optimization opportunities in three buckets:

- Reduce/fuse the action-attention score replay body.
- Keep more root/action template data resident on device across rounds.
- Reduce batched environment stepping and tokenization cost.

## Next Work

- Promote cached-root multi-environment batching from benchmark script to a reusable evaluator/training data path.
- Continue reducing the fixed-shape score graph body; padded live replay removes partial eager forwards, so the graph score kernels are again the main neural target.
- Reduce root observation attachment and root encoding overhead for multi-env runs.
- Replace list-of-dict observations with a packed batched observation structure for multi-env runs.
- Expand `DenseRootSearchState` beyond the root into full batched tree tensors.
- Replace Python node objects with dense tree tensors.
- Use `BatchedRootBranchSimulator` for exact one-step branch expansion.
- For online latency, prefer one-pass/low-depth planning unless a beam branch changes decisions enough to justify the extra ~6-9 ms/window.
- Keep deeper simulator state transitions as the remaining hard part; either extend the C vector stepping path for deeper batched rollouts or build a PyTorch/JAX-compatible approximate rollout model.
