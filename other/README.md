# Historical Research Archive

This directory preserves selected implementations from the development history of `radar-dnn-mcts`.

## Contents

- `legacy/` contains the original main program, original `MCTS_NEW` implementation, transformer MCTS variants, and pretrained transformer experiments.
- `research_archive/architecture/` contains flat, factorized, action-attention, pair-head, and physical-action formulations.
- `research_archive/planning_and_training/` contains P/PQ, AlphaZero/PUCT, MCTS, self-play, bootstrap, distillation, and Q-head experiments.
- `research_archive/deployment/` contains full re-encoding, MuZero, latent-loop, autoregressive, batch, sequence, GRU, sparse, and windowed schedulers.
- `research_archive/dual_sensor/` contains the historical atomic, joint, pair-head, work-conserving, busy-aware, asynchronous, and curriculum experiments.
- `research_archive/realtime/` contains boundary-state prediction, asynchronous planning, dead-time evaluation, and latency profiling.
- `research_archive/evaluation/` contains the code used for canonical evaluations, ablations, service metrics, latency frontiers, reward studies, and dual-sensor results.

## Provenance

The files were selected from the larger local research workspace, primarily `CreateValid1/experiments/code`, `CreateValid1/experiments/code/model_code`, and their result directories. Existing filenames are retained inside each experiment capsule.

Only source code is archived. Results, plots, checkpoints, raw traces, caches, virtual environments, build products, third-party repositories, and repetitive smoke runs are excluded.

The archive is historical and is not imported by the maintained package or used by its tests.
