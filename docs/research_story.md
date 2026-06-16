# Research Story

## 1. Baselines

The starting point is a radar scheduling environment with two classical heuristics:

- EDF: prioritize the target closest to its deadline.
- EST: prioritize search/track service pressure.

These baselines are fast and interpretable, but they are local rules.

## 2. Transformer Policy Prior

The first neural model uses a transformer encoder over target/context tokens and predicts a policy prior over candidate actions.

This gives a learned proposal model:

```text
P_theta(a | s)
```

The limitation is that policy probability alone does not necessarily rank actions by downstream return.

## 3. From P to PQ

The Q head adds value awareness:

```text
Q_theta(s, a)
```

MCTS or planner scoring can then combine:

```text
tree value + policy prior + learned Q
```

This changes the model from a proposal model into a value-aware scheduler.

## 4. Flat vs Factorized Policy

A flat head treats every full action as one class. The factorized head decomposes the action into type and target:

```text
P(action) = P(type) P(target | type)
```

This better matches the structure of radar actions:

- search is a type decision;
- track is a type decision plus target selection.

## 5. Dual Sensor Formulation

For S/X sensors, each candidate action includes a sensor identity:

```text
a = (j, k, i), j in {S, X}
```

The model has sensor-conditioned policy and Q streams.

## 6. Action Attention

The shared transformer encodes the state and target tokens. The action token builder then forms candidate action tokens by combining:

- encoded target/search token;
- global CLS/context embedding;
- learned sensor embedding.

Action self-attention lets candidate actions exchange context before final policy/Q scoring.

## 7. Main Architecture

The final research architecture is:

```text
shared transformer encoder
-> action token builder
-> action self-attention
-> factorized type/target policy heads
-> Q head
```

The main method is action-attention factorized PQ.
