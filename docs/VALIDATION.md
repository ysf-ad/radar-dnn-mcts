# Validation

The clean repository uses three validation levels.

1. Unit tests check feature schemas, masks, model outputs, PUCT, recurrent transitions, and all scheduler modes.
2. `scripts/cross_test_environment.py` compares deterministic EDF and EST state trajectories with a reference experiment checkout. Reward parity is optional because the clean repository uses a revised, versioned reward contract.
3. `scripts/validate_reference.py` regenerates aggregate metrics from an external archived window trace and compares them with its archived summary.

The repository does not require generated experiment artifacts to run its unit tests or simulator cross-test.

## Reward Provenance

Fresh runs use the checked-in C source and `RewardConfig` as one versioned contract. The archived July paper trace was produced by an intermediate runtime/reward contract: its state trajectories match the reference simulator, but its reward column must be interpreted with the archived summary that accompanies it. `validate_reference.py` checks that pair without silently relabeling it as a fresh rerun.

The two valid claims are therefore separate:

- simulator and service-state parity are established by the cross-test;
- published aggregate values are established by recomputing them from their archived per-window trace.
