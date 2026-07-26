# Environment Contract

The repository includes the radar simulator source and Python extension under
`pufferlib/ocean/radarxs`. The namespace is retained because the C extension and
historical simulator import paths use it; the project does not depend on the
full PufferLib training stack.

## State and Dynamics

The clean simulator preserves the reference target motion, Poisson arrivals,
search grid, dwell execution, tracker deadlines, and EDF/EST planners. Search is
row zero and positive rows identify visible active targets. Hidden, inactive,
and dropped targets are excluded from learned action masks.

## Reward Contract

Training and fresh evaluation share `radar_dnn_mcts.env.config.RewardConfig`.
The default contract:

- disables target priority and hidden-target penalties;
- assigns no direct reward merely for issuing search or track actions;
- uses normalized linear target tardiness in clean service mode;
- treats sector staleness as a time-scaled cost rate;
- applies an explicit drop penalty and dense pressure-improvement shaping.

These reward changes do not alter target arrivals or state transitions.

## Binding Surface

The checked-in binding exposes the environment operations required by training,
evaluation, snapshotting, and deterministic replay. Experimental OpenMP and
bulk snapshot helpers from the development tree are intentionally omitted; they
were performance prototypes rather than part of the research interface.
