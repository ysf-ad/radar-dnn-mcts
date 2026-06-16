# Equations

## Action

```text
a = (j, k, i)
```

where:

```text
j in {S, X}
k in {search, track}
i = 0 for search
i > 0 for a track target
```

## Factorized Policy

All probabilities are conditioned on the current state unless written explicitly.

```text
P_theta(a) = P_theta(k) P_theta(i | k)
```

For the dual-sensor version:

```text
P_theta(a^j) = P_theta(k^j) P_theta(i^j | k^j),  j in {S, X}
```

## Policy Loss

```text
L_P = L_P^S + L_P^X
```

```text
L_P^j = L_type^j + L_target^j
```

```text
L_type = - sum_k pi_type(k) log P_theta(k)
```

```text
L_target = - I{k = track} sum_i pi_target(i) log P_theta(i | k)
```

## Q Loss

```text
L_Q = L_Q^S + L_Q^X
```

```text
L_Q^j = sum_{i in A_j(s_t)} (Q_theta^j(i) - G_i^j)^2
```

The implementation often uses smooth L1/Huber loss rather than pure squared error.

## Combined Training Loss

```text
L_total = L_P + lambda_Q L_Q
```

Additional auxiliary losses may be enabled in ablations, but the slide-level story should focus on policy and Q.
