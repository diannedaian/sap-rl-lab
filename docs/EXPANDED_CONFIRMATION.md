# Expanded confirmation protocol

## Current batch: v3

Prepared after all three repaired-rule v8 pilots completed and exactly reloaded
their selected validation rows. Action cost0.005 is selected by unassisted
validation success:92.78%, versus swap92.22% and control90.56%. This single-seed
choice does not establish a general benefit. The rules and numerical thresholds
are unchanged from v8.

- Six fresh runs: seeds1811/1907/2027, each paired action-cost/control.
- Predeclared budget:8,388,608 decisions each; at most two CPU training workers.
  All three paired seeds completed their full budgets.
- New test generation region7,000,000; test episodes7,100,000; learned opponent
  generation7,200,000; challenge episodes7,300,000. All nine scripted/learned
  evaluations completed after all models and delivery choice were frozen/reloaded.
- Catalog:`2d71a64f5136f0a24fffb7d0ccc91ff9567d9cb7fae645119d932525642e23fe`.
- Source archive:`78db9add1ca353959e2f1b63f389385fb996ed50a43cf1118a150c553520b38d`.
- Protocol:`e318a5cbb7b1c0c7fc58071f907486db02fb8fba401cc35df24cb2f9db83d108`.

The six-model freeze and six exact validation reloads are complete. Delivery is
`action_cost-seed1907`, selected exclusively by validation before any test.
All three candidate seeds passed the original numerical screen; independent
rational-count recomputation agrees. Final32delivery and3forced-candidate
reconstructions have been reviewed. See [delivery results](EXPANDED_DELIVERY.md)
for scores, limitations and reproduction. Frozen source/seeds/budgets are unchanged.

A bounded post-training runner completed successfully in
`runs/expanded-confirmation-v3/post-training/`. It never starts training. Once
all six completion artifacts exist, the existing freeze stage verifies complete
budgets and provenance; **all six exact reloads** must then finish before any
evaluation starts. Inference stages use at most two separate CPU processes.
The wait budget is7,200seconds; abort markers, timeout or stage errors stop the
runner without retries, new seeds, altered gates or a goal-completion claim.

`scripts/finish_expanded_confirmation.py` has eight focused passing tests for
the stage plan, actual execution barriers, complete/aborted input states,
duplicate execution rejection, timeout without training restart and freeze
failure preventing later stages. Runner/protocol hashes, stage logs and final
success/failure markers are written once below the batch's `post-training`
directory. This orchestrates already-tested stages; it is not an independent
statistical or game-parity verifier. Final replay/rule/delivery review remains
manual even when its numerical screen passes.

### First completed pair, validation only

Seed1811 control/action both select the final update at8,388,608 decisions
(`validation_8388608_eval009.json`). On600validation episodes, control has
94.17%ten-win success,94.00%unassisted success and0.33%episodes with forcing;
action cost has96.17%success/unassisted and0%forcing. Both have0%truncation.
Raw returns are7.5783/7.9133; elapsed training times2068.4/2098.9seconds.
At this first-pair checkpoint no held-out evaluation or delivery selection had occurred.

Saved model SHA256:

- Control:`c28a7999675e127745995285fb2b161b0146f3c00fca8ecf1efb5e293210f931`.
- Action:`59a69860ef5140f277d5663352aa68e712e6715f967bdd79c01aa84685490590`.

The new seed1907 pair has matching initial weights:
`4bc53386fb97b04599dcbd7ac8c4a8b57c6d632b88f6a841dac5d615ae5eef5b`.
Exact reloads and final tests remain behind the six-run completion barrier.

### Second completed pair, validation only

Seed1907 control/action both select the final update at8,388,608 decisions
(`validation_8388608_eval009.json`). Validation ten-win/unassisted success is
96.50%/96.83%, with zero forcing and truncation in both arms. Raw returns are
7.9300/8.0533. These are development-validation scores, not the final held-out
gate. No changes to seeds, budgets or rules are made from this intermediate result.

### Third completed pair, validation only

Seed2027 control selects the6,291,456-decision checkpoint:89.00%ten wins,
88.00%unassisted success,1.67%forcing, no truncation. Action cost selects the
7,340,032-decision checkpoint:95.17%success/unassisted, no forcing or truncation.
Both actually trained the full8,388,608-decision budget; earlier selected
checkpoints do not mean shorter training. Raw selected returns are6.6883/7.8400.
The completed full test suite is615passing tests, with Ruff/diff checks clean.

## Historical stopped batch: v2

Stopped after seed1811 completed both full budgets, while seed1907 was partial.
Directly observed Dodo/Badger rounding contradicted the frozen ceiling rule;
see [numeric rule audit](PERCENT_ROUNDING_AUDIT.md). Supervisor and both workers
were stopped before source changes. Seed2027 did not run. There is no selection,
held-out evaluation, or numerical stability result for this batch. Artifacts and
an abort record remain intact. The paragraph below records its original plan.

The repaired-rule v7 pilot completed all three equal-budget arms and exact
validation reloads. Action cost 0.005 again leads this single-seed pilot and is
selected for the new `runs/expanded-confirmation-v2` batch. The unchanged planned
budget is 8,388,608 decisions per run, seeds 1811/1907/2027, treatment and control,
at most two live CPU training workers. Seed 1811 is running; later pairs are queued.

This batch uses the explicitly provisional 30-exchange draw rule with catalog
digest `82e7400641521961e492324a9705831c5b8e6a3311bbf424cb52ddec80ecc1e5`.
It cannot by itself settle the original-client rule uncertainty. A new split
region starts at 6,000,000 (tests 6,100,000; learned generation 6,200,000; learned
challenge 6,300,000). No held-out scores have been opened. Training, source/pool
hashes, selected checkpoints and test evaluation remain separate stages.

## Historical aborted batch: v1

The full v6 pilot comparison is complete. **Action cost 0.005** is selected for
confirmation against the unshaped control. `runs/expanded-confirmation-v1` is
prepared, but was stopped during its first paired seed (1811) after an independent
audit exposed a legal long-battle simulator exception. The control stopped at
3,889,152 logged decisions and action cost at 3,825,664; these unequal partial
budgets are not a final comparison. All artifacts are preserved with an abort
record. No held-out evaluation has started; the learned-policy challenge contains
4,157 reachable snapshots. Resolve the rule gap before preparing a new batch.

Planned confirmation budget, fixed before opening any held-out scores:
**8,388,608 decisions per run**, six runs, two CPU workers at most. The action
pilot was still improving from 3.15M to 4.19M, so the larger bounded budget gives
new seeds room to learn before judging reliability. Both treatment and control
receive this budget. This is not a claim that more steps always help: validation
keeps earlier checkpoints when later updates regress.

Use `python -m sap_rl_lab.expanded_confirmation` for explicit stages:

1. `prepare`: requires a completed, source-matching three-arm pilot and an
   explicitly selected treatment (`action_cost` or `swap_cost`). Reuse its
   training/validation pools; create new scripted test pools and a separate
   learned-opponent pool from the pilot control's actual reachable teams.
2. `run`: six fresh starts, seeds **1811, 1907, 2027**, each paired with an
   unshaped control. At most two CPU training workers concurrently. All arms
   receive the same predeclared decision budget; validation is 200 episodes per
   family. An explicit larger budget can be chosen before preparation, never
   silently extended after seeing test scores.
3. `freeze`: require all runs complete and paired initial weights/contracts.
   Fix all six validation-selected checkpoints and the delivery candidate.
   Selection uses unassisted success, then raw return, then smaller seed for
   cross-seed ties. Test results never enter selection.
4. `reload`: each frozen checkpoint must reproduce every saved validation
   episode row exactly before its test may open.
5. `evaluate`: 1,000 new episodes per scripted family, plus 1,000 against the
   separate learned control pool. Evaluate all six models and all three
   transparent heuristic baselines. Each named evaluation is write-once;
   unfinished runs leave a started marker, not a false completion record.
6. `summarize`: recompute the numerical gate from full paired episode rows.
   Report learned-challenge scores separately. Never equate a passing numerical
   screen to official-client parity or automatic goal completion.

Default fresh region: scripted test generation starts at 5,000,000, test episode
seeds at 5,100,000, learned-pool generation at 5,200,000, and learned-challenge
episodes at 5,300,000. Test pools may be generated before model selection, but
their scores stay unopened until selection is frozen. If subsequent iteration
uses observed test results, reserve a new region and disclose the old test as
development evidence.

The learned opponent is deliberately a policy that candidates have never
trained against. It is **not** a full self-play league, and it was trained on
the same scripted-family distribution. This challenge measures transfer to a
new learned policy, not arbitrary real-player or full-Turtle generalization.

The module and its tests are separate post-training tooling, not imported by
the frozen v6 training jobs. It becomes part of the next source archive when
confirmation is actually prepared. See [the completion contract](EXPANDED_GOAL.md)
for all remaining rule, replay and reporting requirements.
