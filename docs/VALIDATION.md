# Validation

The clean repository uses two validation levels.

1. Unit tests check feature schemas, masks, model outputs, PUCT, recurrent transitions, and all scheduler modes.
2. `scripts/validate_reference.py` regenerates aggregate metrics from an external archived window trace and compares them with its archived summary.

The repository does not require generated experiment artifacts to run its unit tests.

## Reward Provenance

Fresh runs use the checked-in C source and `RewardConfig` as one versioned contract. The archived July paper trace was produced by an intermediate runtime/reward contract: its state trajectories match the reference simulator, but its reward column must be interpreted with the archived summary that accompanies it. `validate_reference.py` checks that pair without silently relabeling it as a fresh rerun.

Published aggregate values are established by recomputing them from their
archived per-window trace.
