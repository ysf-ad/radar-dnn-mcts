# Validation

This clean repository was cross-tested against the original experiment tree.

## Baseline Cross-Test

Command shape:

```powershell
python scripts\build_binding.py
python scripts\cross_test_baselines.py --code-root "<old model_code path>" --out results\cross_test_old_baselines.csv --windows 100 --seed 916
python scripts\cross_test_baselines.py --code-root "<clean repo path>" --out results\cross_test_clean_baselines.csv --windows 100 --seed 916
```

Configuration:

```text
initial targets: 20, 40, 60
arrival rates: 2, 3, 4
methods: EDF, EST
windows: 100
seed: 916
```

Result:

```text
reward_per_window:        max absolute diff = 0
final_cumulative_reward:  max absolute diff = 0
tracked_targets:          max absolute diff = 0
active_targets:           max absolute diff = 0
search_fraction:          max absolute diff = 0
```

No cell had a nonzero old-vs-clean difference above `1e-9`.

## Model Cross-Test

The main `ActionAttentionFactorizedNet` was instantiated in both trees with the same random seed and evaluated on the same deterministic dummy token/slot batch.

Result:

```text
scores shape: [2, 8, 2]
Q shape:      [2, 8, 2]
policy scores: exact match
Q outputs:     exact match
```

This confirms that the clean repo preserves the baseline environment behavior and the main model implementation.
