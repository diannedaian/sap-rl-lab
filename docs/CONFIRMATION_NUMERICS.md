# Confirmation statistic boundary audit

## Resolved before the next frozen batch

Confirmation v2 was stopped for an unrelated **simulator rounding** discrepancy,
not this reporting bug. After both live workers exited, `expanded_gate.screen`
was repaired using exact rational counts and unchanged macro definitions and
thresholds. Both strict xfail markers were removed; all 24 existing/boundary/
independent-reference tests pass. No held-out scores were opened or relabeled.
The old source remains in the aborted batch's source ZIP. The v2 audit command
below is historical planning, **not runnable completion evidence for that aborted
batch**, which has no frozen selection or test evaluations. New batches use the
fixed screen from their own source archive.

## Historical discovery and planned audit

2026-09-07. Two **postprocessing-only** false-rejection bugs were reproduced in
the frozen `expanded_gate.screen` helper. They do not affect environment dynamics,
PPO updates, reward signals, or validation checkpoint selection.

- Candidate/control both win 813/938/949 of their three 1,000-episode families:
  the exact macro is 2700/3000 = 90%, but the float mean is
  0.8999999999999999 and the current helper rejects it.
- Candidate wins 900/902/900 and control 910/912/910: exactly 30 extra control
  successes among 3,000 episodes, i.e. a one-percentage-point difference, which
  the declared rule permits. Binary float arithmetic can nevertheless reject it.

Strict expected-failure regressions preserve both examples in
`tests/test_expanded_gate_boundaries.py`; they are known issues, **not passing
tests**. Confirmation v2's entire source manifest is frozen while its six jobs
run. Do not edit that source mid-run or discard unaffected training to fix a
reporting helper. No held-out scores were opened to discover these synthetic cases.

Before final sign-off, replace numerical pass/fail comparisons with exact
rational rates formed from integer counts (averaging per-family rates, preserving
the macro definition). Keep floats only for display. Remove the expected-failure
markers after the actual fix, rerun boundary and existing rejection tests, and
independently recompute any affected saved summaries from full episode rows.
Do not change the 90% / <1% / <=1-percentage-point thresholds or add a permissive
epsilon. Preserve the original frozen source and outputs for reproduction.

An executable independent exact-count reference is now implemented outside the
frozen source in `scripts/exact_confirmation_counts.py`. Ten focused tests cover
inclusive 90% / one-point boundaries, one extra failure, strict 9-versus-10 per
thousand reliability, structural rejection and unweighted family means with
unequal sample sizes. The two expected failures remain attached to the old
frozen helper until integration; they are not silently made green by testing
only the new reference.

After the original six-run confirmation has finished all stages, audit its
already-existing raw results with a new output filename:

```sh
.venv/bin/python scripts/exact_confirmation_counts.py \
  --confirmation runs/expanded-confirmation-v2 \
  --output runs/expanded-confirmation-v2/exact-count-summary.json
```

The audit verifies frozen source/pools and selection provenance, preserves the
old float decisions beside exact ones, records input/script hashes, and never
trains or generates new test trajectories. It does not certify the overall goal.
It has not been run on actual held-out results yet; those remain unopened.
