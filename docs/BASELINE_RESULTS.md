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

## First full Maskable PPO experiment

A single Maskable PPO policy was trained for 500,000 requested decisions with
eight subprocess environments and seed 17. PPO completed its final rollout at
503,808 decisions. The training and evaluation leagues were generated from
disjoint seed ranges, and the learned policy was evaluated deterministically on
500 episode seeds beginning at 20000.

For a fair comparison, all three policies below used that experiment's exact
held-out league and episode seeds.

| Policy | Mean return | Mean wins | Ten-win runs | Truncations |
|---|---:|---:|---:|---:|
| Uniform random legal action | -4.832 | 0.038 | 0 / 500 | 36 / 500 |
| Spend-gold heuristic | 2.978 | 6.58 | 278 / 500 | 0 / 500 |
| Maskable PPO | **3.826** | **7.23** | **295 / 500** | 17 / 500 |

PPO won 3,615 of 6,022 individual battles (60.03%), drew 722, and lost 1,685.
Its training monitor contains 8,888 completed episodes: mean return rose from
-4.79 over the first 100 to +2.59 over the last 100, while mean episode length
rose from 49.54 to 78.88 actions.

This is evidence that the agent learned a strategy which generalizes beyond its
training snapshots. It is not yet evidence that PPO reliably beats the
heuristic: the 59.0% versus 55.6% success-rate difference comes from one
training seed, and the corresponding binomial confidence intervals overlap.
The next defensible experiment is an identical multi-seed run with aggregate
confidence intervals.

That follow-up is now recorded in [the second experiment](ROUND2.md): three
continuation seeds reached 86.9–91.7% success on a fresh test pool, compared with
54.5% for the original policy evaluated on that same pool. The report includes
the seeding fix, validation selection, and an optional reward-shaping experiment
that reduced cutoffs but did not improve average success.
