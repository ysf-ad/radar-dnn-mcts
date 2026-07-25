# Radar changes from MuZero-General `0825bd5`

The repository snapshot is otherwise unchanged.

## Modified `self_play.py`

1. Added an optional local Ray fallback.
2. Added package-relative `models` import support.
3. Replaced recurrent expansion over `config.action_space` with
   `model.legal_actions(hidden_state)`.

## Added adapter

`radar_adapter.py` translates:

- radar encoding `h` to `initial_inference`;
- latent dynamics and prediction `g` then `f` to `recurrent_inference`;
- scalar radar reward/value outputs to MuZero-General categorical supports;
- dynamic radar rows, deadlines, and selected targets to legal actions.

The vendored MCTS algorithm itself remains stepwise and stops each simulation
after its first new leaf expansion.
