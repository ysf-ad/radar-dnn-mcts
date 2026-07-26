# Codepath

This page traces the common path from a radar observation to training and
evaluation. The schedulers share the same environment, action numbering,
feature builder, checkpoint modules, and benchmark runner.

## State and execution

1. `pufferlib/ocean/radarxs/engine.py` owns the C radar simulator and exposes
   the current observation through `get_obs_from_buf`.
2. `radar_dnn_mcts/evaluation/benchmark.py` requests that observation once at
   each window boundary and passes it to the selected scheduler.
3. `radar_dnn_mcts/env/features.py` converts the observation into target tokens
   and global context for the neural models.
4. The scheduler returns an ordered list of actions. The engine executes that
   list and reports the actual rewards, durations, and next observation.

## Search methods

`radar_dnn_mcts/search/puct.py` contains the shared PUCT selection, expansion,
backup, and visit-target extraction. Raw backed-up returns are used directly;
there is no tree min--max normalization.

- **Full re-encode:** `FullWindowPUCTScheduler` creates a
  `RadarWindowSearchState`. Each edge applies the deterministic transition in
  `env/transition.py`, computes the aligned reward in `env/reward.py`, and the
  evaluator encodes the resulting shadow observation again.
- **Autoregressive:** `AutoregressivePUCTScheduler` uses the same PUCT tree and
  shadow state, while `RadarAREvaluator` encodes the root once and evaluates
  later prefixes with the AR decoder.
- **MuZero:** `MuZeroPUCTScheduler` enters `search/puct_dynamics.py`.
  The representation network \(h\) encodes the root once, dynamics \(g\)
  advances each traversed edge, and prediction \(f\) evaluates expanded latent
  nodes.
- **Batch:** `BatchScheduler` decodes all schedule positions in one forward
  pass and masks invalid output rows. It is trained from collected teacher
  trajectories rather than running a tree internally.

The re-encode, AR, and MuZero evaluators all expose compatible policy and scalar
value outputs to their search adapters.

## Collection and returns

`scripts/collect_puct_targets.py` runs one simulator episode. For each planned
window, `WindowTrajectoryCollector` stores the pre-encoder observation
features, PUCT visit distribution, chosen action, and action mask. After the
engine executes the plan, the collector attaches the actual simulator reward
and duration to the matching record.

Return targets are then backfilled across either a fixed number of future
windows or the complete remaining episode. `arrays()` groups the variable
action sequences into padded window arrays and records masks so padding never
contributes to a loss.

## Connected training

`scripts/self_play.py` is the top-level loop:

1. collect fresh trajectories over `configs/training_9cell.yaml`;
2. merge them into a bounded replay dataset;
3. call the existing core, AR/batch, or MuZero trainer;
4. evaluate the candidate and incumbent on
   `configs/promotion_full_horizon.yaml`;
5. promote only a candidate that improves over both the incumbent and EDF.

The trainers rerun the neural networks on sampled executed trajectories.
For MuZero, \(h\) processes the first observation, then saved actions are
passed through \(g\) and \(f\) for the recurrent training steps. Policy, value,
and reward losses backpropagate jointly through that computation graph.

## Evaluation and boundary prediction

`scripts/evaluate.py` constructs the requested planners and hands each to the
common `BenchmarkRunner`.

`schedulers/async_boundary.py` supports planning while the current window is
still executing. For method latency \(L_{\max}\), window length \(B\), and
safety buffer \(b\), planning starts at

\[
t_{\mathrm{plan}} = \max(0,\min(B, B-L_{\max}-b)).
\]

The boundary predictor receives the current encoded state, actions remaining
before the boundary, and remaining execution time; its predicted boundary
embedding initializes planning for the following window.
