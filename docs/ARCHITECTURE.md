# Architecture

The package separates shared learned components from deployment behavior.

## Shared Components

- `StateEncoder` encodes the search row, visible target rows, and window context.
- `ActionTokenMixer` builds one token for each valid search or track candidate and applies candidate self-attention.
- `RadarSchedulerModel` returns a factorized policy and complete-action Q values.
- `LatentDynamics` predicts the next latent state and immediate reward after a selected action.

PUCT uses the policy as its prior and action-Q as both an additional selection
term and a policy-weighted estimate at newly expanded leaves. The Q weight
defaults to one and can be set to zero for the policy-only ablation.

## Deployment Modes

- **Full re-encode** updates a shadow observation and runs the state encoder after every action.
- **MuZero** runs the state encoder once and updates the latent state through learned dynamics.
- **Autoregressive** runs the state encoder once and conditions each decision on the decoded action prefix.
- **Batch** scores all schedule positions in one parallel decoder pass and applies validity and budget masks afterward.

All modes use the same action convention: row zero is search, and positive rows track visible active targets. Invalid, dropped, and already selected targets are masked before selection.

## Asynchronous Planning

`BoundaryPredictor` estimates the next window's latent state from a midpoint state, the actions still executing, and the remaining time. `AsynchronousBoundaryScheduler` decodes the next schedule from that estimate while the current schedule continues to execute.
