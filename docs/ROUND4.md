# Round 4 — swap penalty versus success-only efficiency bonus

## Result — completed September 5, 2026

**The swap-cost candidate is the better candidate in this paired run.** It
passes the predeclared cutoff screen; the success-bonus candidate does not.

| Policy | 10-win rate | Cutoffs | Mean swaps / episode | Mean actions in wins |
| --- | ---: | ---: | ---: | ---: |
| Frozen parent, no additional training | 79.33% | 6.63% (199/3,000) | 48.36 | 114.86 |
| Swap cost −0.005 | **92.73%** | **0.20% (6/3,000)** | **10.66** | **72.20** |
| Success-only efficiency bonus | 86.17% | 5.03% (151/3,000) | 32.91 | 96.00 |

Swap cost beats the bonus by 6.57 percentage points in success and 4.83 points
in cutoff reduction. The episode-paired 95% interval for its success advantage
is +5.13 to +8.03 points, **conditional on these two fixed policies**, not
training-seed uncertainty. Its cutoff rate is 0.20% in each of the three
opponent families. The remaining six cutoffs still contain repeated swaps.

Both candidates trained for 1,003,520 steps. Validation selected the final
swap-cost checkpoint and the bonus checkpoint at 500,000 steps. Training,
validation, pool generation and final evaluation took about six minutes on CPU.
No training remains running. No default reward or historical replay was replaced.

All 9,000 test rows passed reward-accounting and provenance checks; 45 saved-policy
episodes were independently rerun and matched. The Python suite passes 62 tests.

## Fixed comparison

Exactly two full training runs, continuing the same Round 3 `mixed-seed307`
checkpoint with the same continuation seed (409). Both use corrected Fish
rules, fresh identical optimizers, eight environments, and 1,000,000 requested
steps. PPO rounds each budget up to 1,003,520 steps.

| Arm | Training reward change |
| --- | --- |
| Swap cost | Subtract 0.005 on every swap, including useful swaps |
| Success bonus | Only at 10-win termination, add `max(0, 1 − 0.005 × total decisions)` |

The bonus counts every decision, including end-turn. A 100-decision victory
adds 0.5; a victory taking 200 or more decisions adds zero. Losses and cutoffs
receive no bonus. The coefficient was fixed before training, not tuned on results.

Both policies observe the cumulative episode action count because the bonus
depends on that history. One zero-weight input column is inserted into the
actor and critic, preserving the old policy's initial outputs; both arms start
with identical expanded weights. Historical models keep their original inputs.

The legal mask, gamma=1, and original truncation/bootstrap handling remain
unchanged. This experiment does not also test terminal forfeiting.

## Evaluation safeguards

- New, disjoint train/validation/test pools for greedy, stats, and summon
  opponents, generated under the same corrected rules.
- Checkpoints selected by equal-family raw validation success, then raw return;
  both choices frozen before final test evaluation.
- Final evaluation removes all costs and bonuses: 1,000 episodes per family
  for each candidate and the unchanged parent, 9,000 episodes total.
- Judge raw win rate and cutoff rate first; also report swaps and action count
  **within successful episodes**, so losing quickly cannot look efficient.
- Screening target: below 1% macro cutoffs with no more than a 1-percentage-point
  success drop relative to the frozen parent. This is exploratory, not a formal
  statistical non-inferiority claim.

## Scope

This is one paired continuation seed, not a multi-seed result. The unchanged
parent is an evaluation reference, not a third training arm. Without an
unshaped continuation control, improvements over that parent cannot be
attributed solely to reward shaping rather than additional training. Results
do not establish generalization to new opponent families or live SAP.

Artifacts: `runs/round4-two-objectives-v1/` includes the frozen protocol, source
archive, pool/model hashes, full training logs, validation histories, and tests.

Reproduce with a new output directory:

```sh
python -m sap_rl_lab.round4 --output runs/round4-two-objectives-v2
python scripts/audit_round4.py runs/round4-two-objectives-v2
```
