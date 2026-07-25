# MuZero-General reference snapshots

Both reference directories are complete snapshots of:

- Repository: `https://github.com/werner-duvaud/muzero-general`
- Commit: `0825bd544fc172a2e2dcc96d43711123222c4a2f`
- License: MIT

The original source, games, training pipeline, replay buffer, documentation,
requirements, and notebook are retained in each directory. The changes below
are the complete differences from that pinned snapshot.

## Stepwise snapshot

Changed upstream file: `muzero_general_stepwise/self_play.py`

1. Ray import has a local fallback so the MCTS can run without installing Ray.
2. `models` supports package-relative import.
3. Recurrent node expansion calls `model.legal_actions(hidden_state)` instead
   of expanding the fixed game action space. This is required because radar
   deadlines and selected targets change along a latent prefix.

Added files:

- `__init__.py`: exposes the radar scheduler.
- `radar_adapter.py`: maps the local radar `h`, `f`, and `g` modules to
  MuZero-General's `initial_inference` and `recurrent_inference` interfaces.
- `RADAR_CHANGES.md`: local copy of the stepwise change record.

Unchanged behavior includes MuZero-General's stop-at-first-new-leaf
simulations, UCB equation, min-max normalization, random exact-tie handling,
mean-value backup, categorical scalar support conversion, and maximum-root-
visit action choice.

## Windowed snapshot

The same three radar compatibility changes are applied to
`muzero_general_windowed/self_play.py`.

One search behavior is additionally changed:

1. After expanding a new leaf, the simulation continues selecting and
   expanding until `model.legal_actions(hidden_state)` is empty at the radar
   window boundary.

Added files:

- `__init__.py`: exposes the windowed radar scheduler.
- `radar_adapter.py`: invokes the modified vendored MCTS and extracts one
  complete schedule after all rollouts.
- `RADAR_CHANGES.md`: local copy of the windowed change record.

All modifications in `self_play.py` are marked `RADAR ADAPTER CHANGE` or
`WINDOWED CHANGE`.

## Comparison

```powershell
python scripts\compare_muzero_implementations.py `
  --checkpoint path\to\checkpoint.pt `
  --methods ours,stepwise,windowed
```

The comparison reports reward, latency, predicted dynamics calls, and radar
observations per window. Search-count parameters have different meanings, so
the measured dynamics calls should be used for compute matching.
