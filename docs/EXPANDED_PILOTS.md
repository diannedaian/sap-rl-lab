# Expanded pilots — ongoing, not a final training report

## Current pilot v8: per-ability rounding repair

All three arms completed their equal budgets and archived-source reloads reproduced
every selected validation row (180 per model). They share the same initial weights.

| Arm | Ten-win success | Success without forcing | Episodes with forcing | Raw return |
| --- | --- | --- | --- | --- |
| No shaping | 90.56% | 90.56% | 0% | 6.8000 |
| Swap cost 0.005 | 92.78% | 92.22% | 1.11% | 7.4500 |
| All-action cost 0.005 | 92.78% | 92.78% | 0% | 7.2722 |

All selected checkpoints have zero external truncations. Action cost wins the
predeclared unassisted-success criterion, despite swap cost's higher raw return.
This is a single-seed validation choice, not proof of a universal reward benefit.
It is also **not evidence that the selected cost policy uses fewer actions than
control**: mean actions per episode are97.17(control),114.95(swap),101.57(action),
and aggregate actions per battle are7.03,8.53,7.54 respectively. The treatment's
selection is about unassisted success, not demonstrated action efficiency.
Learned policies are approximate and their trajectories differ; a cost in the
objective does not guarantee a lower observed action count at a fixed budget.
Control/swap select the final update; action cost selects the pre-final update at
the same 4,194,304-step evaluation boundary. Exact selected filenames and hashes
are retained, so this distinction is not hidden.

Fresh confirmation v3 is now prepared with seeds1811/1907/2027, paired action-cost
and control, 8,388,608 decisions each, fresh held-out region7,000,000. Its first
two workers are live; later seeds queue. See `EXPANDED_CONFIRMATION.md`.

Fresh same-seed 1709, equal 4,194,304-step control/swap-cost/action-cost arms.
At most two local CPU workers; swap-cost ran after control. Catalog digest
`2d71a64f5136f0a24fffb7d0ccc91ff9567d9cb7fae645119d932525642e23fe`.
Dodo and Badger now floor percentage results; Crab explicitly retains ceiling.
The source also includes repaired exact-count confirmation comparisons.
See [observed evidence](PERCENT_ROUNDING_AUDIT.md). Full suite 577 passed;
`expanded-stress-v7` reproduces 6,000 battles twice and 400 full episodes;
`expanded-audit-legacy-v5` again matches all 300 old-release rows exactly.

No v8 held-out results exist. Older pilot and confirmation-v2 scores below are
historical old-rule evidence, never a substitute for repaired-rule confirmation.
All v2 artifacts are retained with an abort record; its seed1811 completed both
budgets, seed1907 was stopped partial, seed2027 never ran, tests unopened.

## Pilot v1: 131,072 steps per arm, one paired fresh seed

Both local CPU arms completed in about one minute each (concurrent workers).
Their initial parameter fingerprints match exactly. New scripted training and
validation pools cover 28–30 species per family, including legitimately obtained
Tier-3 pets; no synthetic high-tier opponents or old eight-pet test scores.

Neither arm reached a ten-win episode in the 180-episode validation suite.
The selected control averaged 0.84 wins/game; selected swap-cost averaged 0.58.
Both needed forced shop endings in 100% of validation episodes. This is a failed
short learning pilot, not a reliable delivery or evidence that a coefficient is
universally worse.

Inspected ten smallest-seed validation trajectories per selected arm (20 total),
with reconstructed legal-action probabilities and critic values. Both repeated
shop states extensively. Swap-only penalties left repeated freeze/unfreeze
actions unpenalized. There were no successful model episodes available for a
success/failure comparison; the scripted baselines do complete games.

Evidence: `runs/expanded-pilot-v1/` includes protocol/source archive, disjoint
opponent pools, manifests, complete validation records, selected/final models,
and `diagnostics-control/` / `diagnostics-swap/`. No held-out test was opened.

## Next bounded iteration: v2

Three equal-budget, same-initial-seed, from-scratch arms, **1,048,576 steps each**:

- Control: no action shaping.
- Swap-only cost: 0.005, all other settings identical to control.
- All-action cost: 0.005, no swap-specific cost, otherwise identical to control.

This separates more training from a targeted or general action penalty. The
all-action penalty can also discourage useful purchases/positioning; raw and
unassisted validation success, not shaped training reward, determine selection.
At most two workers run concurrently. This remains a single-seed development
comparison, NOT the later independent-seed stability confirmation.

Real-client parity work is still open in [the case ledger](REAL_GAME_CASES.md).
The active goal is not complete; larger pilot budgets do not waive those gates.

### v2 execution failure and numerical repair

The control completed 1,048,576 decisions. Both penalty arms stopped early in
SB3-Contrib's categorical probability validation (action-cost at 821,248
decisions). These are incomplete training runs, not equal-budget evidence
against the reward choices. Their checkpoints and logs remain intact.

The installed Contrib 2.7.1 retains a cached `probs` tensor while reinitializing
the distribution with new logits. This is the upstream issue fixed by
[PR #326](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/326).
We backport its cache clearing in an opt-in expanded-policy subclass, without
disabling numerical validation or changing frozen eight-pet policies. Tests
reproduce the stale-cache failure, preserve NaN/empty-mask rejection, exercise
sharp logits and repeated masks, and verify unchanged normal probabilities.

Next is a fresh v3 three-arm rerun with the same numerical fix in every arm,
the same seed and 1,048,576-decision budget. No held-out test has been opened.

## v3: repaired, equal-budget million-step comparison

All three arms completed, with identical initial parameter fingerprints and
verified frozen source/pool hashes. Validation-selected results, 60 episodes
per family (180 total), one training seed:

| Reward | Ten-win success | Success without forcing | Episodes needing forcing | Mean wins |
| --- | ---: | ---: | ---: | ---: |
| None | 0.56% | 0.56% | 44.44% | 2.72 |
| Swap cost 0.005 | 0.56% | 0.56% | 43.33% | 3.10 |
| Every-action cost 0.005 | 8.89% | 7.78% | 14.44% | 4.55 |

All externally truncated rates were zero. The forcing target is still missed
by a wide margin; this is not a stable model. These figures support further
investigation of the general cost at this budget/seed, not universal superiority.

Inspected the first 20 stats-family validation trajectories of the action-cost
candidate, including one ten-win success and three forced failures. Reversible
swaps/freezes remain: seed 2300007 reaches action 30 with zero gold and assigns
~0.91 probability to another swap; seed 2300015 forces through a freeze action
with ~0.59 probability. Success seed 2300011 also repeats states (36 times), so
success alone does not establish efficient play. The files include exact states,
top legal probabilities and reconstructed critic values; no original PPO buffer
values are inferred. Selling remains very rare in this pilot's validation,
potentially limiting replacement of early teams; this is a diagnosis hypothesis.

## v4: bounded longer-learning check (stopped on rule audit)

Fresh same-seed starts, **4,194,304 decisions per arm**, same three reward choices,
same source and pool content as v3, validation every 1,048,576 decisions. This
first tests whether the still-rising learning curve improves with more training,
before changing architecture or adding demonstrations. All arms get the same
budget; at most two training workers run concurrently. No held-out test is open.

**Stopped intentionally, exit 143**, before editing simulator sources. The audit
found an omitted official 0.29 rule: random buffs prioritize non-maxed pets.
Control had logged 905,216 decisions, action-cost 1,865,728; swap-cost had not
started. Logs/checkpoints/source archive remain intact (`aborted.json`). This
unequal interrupted batch is not a result comparison. Earlier v1–v3 results
describe their archived simulator, not the repaired one.

The repair is enabled by hashed v4 ability data for Ant, Fish, Otter and Beaver.
It prioritizes pets below displayed 50/50, retaining distinct targets and a
fallback when everyone is maxed. It does not invent per-stat targeting beyond
the official wording. Old checkpoint hashes reject the changed catalog.
Fresh **v5** started the planned four-million-step three-arm experiment.

## v5: stopped while completing shared-rule audit

The follow-up audit found two more material omissions: merge/level-up trigger
counters, and official 0.36 perk-gain food notifications. Both active workers
were stopped before source edits (exit 143). Control last logged 475,136 steps;
action-cost 430,080; swap-cost had not started. `aborted.json` records this and
all artifacts remain intact. These partial runs are not comparable results.
The next source freeze must follow focused shared-rule tests and replay stress,
not silently resume these checkpoints under changed dynamics.

## v6: revised-rules, bounded longer-learning pilot

After the shared-rule follow-up, **499 tests passed**, plus 6,000 twice-reproduced
mixed battles and 400 exact legal full-episode replays (`expanded-stress-v4`).
The relevant official patch-history primitives were re-read through the public
Steam news API: stock/frozen slots, merge XP, lethal Hurt/After Attack, non-maxed
random buffs, identical perks and perk-as-food. This is not a blanket client
parity certificate; the case ledger retains outstanding edge-case uncertainty.

Fresh v6 repeats 4,194,304 decisions per arm, seed 1709, three reward arms, at
most two concurrent CPU workers. Revised sources, catalog and newly generated
opponent pools are frozen together. Earlier-rule scores are historical only.
Validation uses 60 episodes per family; no held-out challenge or confirmation
test has been opened, and no stable-delivery claim is made.

### v6 intermediate diagnostic (1,048,576 decisions; not final selection)

At the first periodic validation, action-cost has 23.33% ten-win success,
20.00% unassisted success and 19.44% forced episodes. Control has 0.56% success
and 77.78% forcing. Revised rules change trajectories, so do not attribute the
difference from v3 to one particular rule repair. Both v6 runs continue.

An immutable copy of the action checkpoint and its manifest is retained in
`action-cost-at-1048576/`. The first 30 stats-family validation trajectories are
in `diagnostics-action-1048576/`: three successes, five forced failures. Successful
seeds 2300001/2300016/2300026 still have 31/21/7 repeated states respectively.
This sample is not a new test or an unbiased success-rate estimate.

Seed 2300025, turn 5, is a particularly clear loop: after spending its gold,
actions 4–29 repeatedly swap slots 3/4, returning to the same team/shop every
two actions. At action 29, swap probability is 0.650; forcing happens to win the
battle and returns +0.995 after the step cost. The reconstructed critic value
for the same arrangement rises from -0.590 at action 4 to -0.574 at action 6:
the two-step bootstrapped residual is about +0.006 despite paying -0.010 for
the cycle. This is evidence of imperfect credit assignment, not proof of a
single cause. A fixed looping policy can genuinely have higher value closer
to the deadline because fewer future waste costs remain; do not confuse its
value with the optimal value. END_TURN exploration and longer-horizon advantage
estimation are plausible follow-up controls **if the full learning curve stalls**.

Of the five forced transitions, three win their battle (+0.995), two lose
(-1.005); none terminates the episode merely because of the shop budget. The
forced mechanism is operating as designed. At this checkpoint selling is still
only one action per 60-episode family, so useful early-team replacement remains
another learning weakness to inspect after the longer run.

### v6 first two arms complete; swap arm still running

Both control and action-cost completed 4,194,304 decisions, with identical
initial parameters and verified source/pool hashes. Control's selected checkpoint
is at 3,145,728: 82.78% success, 81.67% unassisted success, 1.11% forcing.
Its final update scores higher raw success (88.33%) but forcing worsens to
23.89%, so it is correctly **not** selected under the frozen rule.

Action-cost's selected checkpoint is the periodic evaluation at 4,194,304,
before the final update: **92.78% ten-win/unassisted success, zero forcing in
180 validation episodes**. Per-family success: stats 90%, summon 98.33%, tempo
90%. Final-update success is 91.11%; it does not replace the selected model.
Elapsed runtime: control 1,038 seconds, action-cost 1,096 seconds. These are still
single-seed validation results, not a completed three-arm comparison or proof
of <1% true failure probability.

The selected action model has 120 reconstructed stats/tempo validation replays
(`diagnostics-action-final-stats/`, `diagnostics-action-final-tempo/`). All 12
failures end through ordinary battles/life exhaustion, not action-limit forcing.
Residual repeated states remain: stats averages 23.93 per episode, max 82.
High success and no observed forcing therefore do **not** mean minimal-action
play or that all pointless swaps are gone. Both successful and failed examples
were inspected; no unexplained simulator termination appeared.

Across those 120 episodes the policy buys 26 species, including eight Tier-3
species, but purchases remain dominated by early pets and it never sells.
Full Tier-3 mechanics being executable is not evidence that the policy has
mastered all ten Tier-3 pets or learned good late-team replacement. These are
important scope/strategy limitations for the later full-Turtle expansion.

### v6 completed comparison and next decision

All three arms now completed the same 4,194,304-decision budget with the same
initial parameters and frozen simulator/pools. Selected validation checkpoints:

| Arm | Ten-win success | Unassisted ten-win success | Forced episodes |
| --- | ---: | ---: | ---: |
| Control | 82.78% | 81.67% | 1.11% |
| Swap cost 0.005 | 89.44% | 87.78% | 2.22% |
| Every-action cost 0.005 | 92.78% | 92.78% | 0.00% |

All external truncations were zero. Swap's selected checkpoint is its final
update; elapsed time was 959 seconds. This remains one training seed and only
60 validation episodes per family. The completed comparison selects **general
action cost 0.005 for independent-seed confirmation**, not as a universal winner.

`expanded-confirmation-v1` uses fresh seeds 1811/1907/2027, paired with unshaped
controls, 8,388,608 decisions per run. The larger fixed budget follows the still
improving action-cost curve, and both arms receive it. The held-out seed region
starts at 5,000,000; no test or learned-challenge score has been opened.

### v7: explicit provisional long-battle draw rule

Confirmation v1 stopped during seed 1811 after a legal 1/50 Swan mirror exposed
the old 200-attack exception. Neither partial run finished; no held-out scores
were opened. The abort record preserves counts and checkpoints, not a comparison.

The development catalog now fingerprints a 30-exchange draw limit, a **provisional
community-supported interpretation**, not a certified v0.46 client constant.
Natural outcomes on the last allowed exchange take precedence; its event queue
fully resolves before declaring a survivor draw. Evaluation counts these draws
separately from forced shop endings. See [the unresolved source issue](LONG_BATTLE_RULE.md).

543 tests pass. `expanded-stress-v6` includes 25% low-attack/high-health synthetic
battles, reproduces all 6,000 battles twice and replays 400 legal episodes exactly.
800 synthetic battles reach the limit; this is deliberately adversarial sampling,
not a natural play rate. The released model again matches all 300 old episode
rows exactly in `expanded-audit-legacy-v4`.

`expanded-pilot-v7` repeats the three arms, seed 1709, 4,194,304 decisions each,
with newly generated pools and frozen source. Control and general action-cost
started first; swap-cost follows in one freed slot. At most two CPU workers.
Catalog digest: `82e7400641521961e492324a9705831c5b8e6a3311bbf424cb52ddec80ecc1e5`.
It is an experimental pilot while original-client cutoff evidence remains open.
All three arms subsequently completed. Selected validation metrics exactly
match the v6 table above, but were measured afresh under v7 and all three saved
models reloaded every selected validation row exactly. Each fixed model's
30/40/50-exchange sensitivity check changed no validation rows. This selects
action cost 0.005 for `expanded-confirmation-v2`, not a universal winner or a
stable-release claim. The new six-run batch uses the previously planned
8,388,608 decisions per run and held-out seed region starting at 6,000,000.

Current v7 stats/tempo reconstructions contain 120 episodes: 108 successes,
12 ordinary battle failures, no forcing or truncation. All twelve failure
endings were inspected; none had Tier-3 pets in its final team. Across the 120
episodes there were no sales and a mean 23.43 repeated shop states. Last-state
critic values were positive even on these failed trajectories, but a single
realized loss is not the expected value target; this is a diagnostic observation,
not proof of one universal failure cause. Full logs retain action probabilities,
rewards, state and value estimates in `expanded-v7-replays-{stats,tempo}-v1`.

### Reproduce a historical pilot after rules change

Do not override a saved model's catalog digest to make it load under a newer
environment. For **trusted local project artifacts only**, this helper verifies
the archived source/model/pool hashes, loads the old source in an isolated
temporary directory, and compares every selected validation row:

```sh
.venv/bin/python scripts/reload_archived_pilot.py \
  --run runs/expanded-pilot-v6 --arm action_cost \
  --output runs/my-v6-archived-reload
```

Use a new output path. It never trains, opens held-out tests or replaces current
source. The current installed dependencies are reused, not recreated; drift is
detected through exact output comparison. Source and model pickles execute code,
so matching hashes do not make unknown third-party archives safe.

Verified on v6 action-cost: all 180 selected validation trajectories match exactly
in `expanded-v6-archived-reload-v1`, even though current v7 has a different catalog
fingerprint. Seven extraction tests cover content integrity, overwrite refusal,
unsafe paths, symlinks, mismatched hashes and unexpected extra members.
