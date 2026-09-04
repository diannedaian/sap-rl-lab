# Baseline results

These numbers are executable sanity checks, not claims that the game is solved.
They use catalog `turtle-v0.46-tier1`, the default seeded curriculum opponent,
and episode seeds 0 through 99.

| Policy | Mean return | Mean wins | Ten-win runs | Truncations |
|---|---:|---:|---:|---:|
| Uniform random legal action | -4.19 | 0.46 | 0 / 100 | 13 / 100 |
| Spend-gold heuristic | 9.95 | 10.00 | 100 / 100 | 0 / 100 |

The separation proves that decisions matter and that the action mask, rewards,
and terminal states permit a competent policy. It also proves the default
opponent is too weak for a final benchmark. The next evaluation milestone is a
frozen held-out snapshot league where scripted play does not win automatically.

Reproduce the checks with:

```bash
python -m sap_rl_lab.cli random --episodes 100
python -m sap_rl_lab.cli greedy --episodes 100
```

A 2,048-step Maskable PPO run has also been completed as an integration test:
the model trained, checkpointed, reloaded, and evaluated. Its 0/20 success rate
is expected at that tiny budget and is intentionally not presented as a learned
result.

## Held-out scripted league

The more useful curriculum check builds 202 training snapshots from seeds 0–19
and 204 evaluation snapshots from seeds 10000–10019. Policies are then tested
for 100 episodes on seeds 20000–20099 against only the evaluation snapshots.

| Policy | Mean return | Mean wins | Ten-win runs | Truncations |
|---|---:|---:|---:|---:|
| Uniform random legal action | -4.76 | 0.04 | 0 / 100 | 9 / 100 |
| Spend-gold heuristic | 2.40 | 6.29 | 51 / 100 | 0 / 100 |

This is a healthier learning target: it retains a clear gap between random and
competent play without letting the heuristic win automatically. The same
league-loading path has been exercised by a Maskable PPO train/save/load/eval
smoke run.
