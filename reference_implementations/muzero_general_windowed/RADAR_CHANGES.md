# Radar and window changes from MuZero-General `0825bd5`

The repository snapshot is otherwise unchanged.

## Modified `self_play.py`

1. Added an optional local Ray fallback.
2. Added package-relative `models` import support.
3. Replaced recurrent expansion over `config.action_space` with
   `model.legal_actions(hidden_state)`.
4. Continued each simulation after a new expansion until the radar window
   boundary instead of stopping at the first new leaf.

## Added adapter

`radar_adapter.py` supplies the radar inference interface and extracts one
complete schedule after all full-window rollouts.
