# Expanded sandbox: completion contract and acceptance audit

Started 2026-09-07. The current thirty-pet engineering milestone passed its
predeclared training/replay acceptance on2026-09-07; see
[the concise delivery report](EXPANDED_DELIVERY.md). Historical progress below
is retained, including stopped batches and earlier incomplete checks.

## User objective and scope

Finish the expanded simulator, check real game cases, run small training/smoke
tests, iterate controlled experiments if performance is inadequate, and stop
only at a verified stable delivery point. Do not ask routine design questions.

Preserve the agreed thirty-pet curriculum: all Turtle Tier 1–3 pets have real
abilities; normal shops cap at Tier 2 and full Tier 3 supplies upgrades/summons.
This does not mean normal Tier-3 shops with unimplemented Tier-4 reward fallbacks.
The eventual full-pack goal remains sixty pets. Any later shop-tier expansion
must carry its real next-tier dependencies, not silently downgrade them.

## Completion evidence required

1. Source/version ledger plus observed real-client or recorded-game cases for
   high-risk event interactions, with mismatches fixed and regression fixtures.
   Specification tests alone do not prove official parity; no claim of literal
   bug freedom is possible. Unresolved material mismatches prevent completion.
2. All enabled abilities and shared primitives tested, deterministic replays,
   full-episode and mixed-battle stress checks; old releases remain reproducible.
3. Versioned model/environment contract, fresh expanded opponent pools, disjoint
   train/validation/test inputs, bounded smoke training and timing measurements.
4. Validation-selected candidates from multiple independent training seeds;
   controlled equal-budget comparisons for any claimed reward/training benefit.
   Tests are held out until selection is frozen; subsequent iterations use a new
   held-out split. No result-driven seed replacement.
5. Delivery reliability: each of three independent seeds has <1% externally
   truncated episodes AND <1% episodes requiring forced shop endings, on at least
   1,000 held-out episodes per opponent family. Report battle-level forcing too.
   Require >=90% macro ten-win success on the frozen scripted suite, and no
   more than a one-percentage-point macro success loss versus the paired
   equal-budget control. These are engineering thresholds, not significance tests.
6. A separate unseen learned-opponent challenge, with per-family results and
   baseline comparisons; do not equate scripted-suite success to full-game skill.
7. Inspect successful and failed/forced replays, verify reloads reproduce scores,
   archive source/config/data hashes, write a concise final report and runnable
   reproduction commands. No hidden simulator exceptions or unexplained loops.

The old eight-pet Round 5 did NOT satisfy its strict reliability target; it stopped
as an educational milestone with limitations. That historical exception is not
permission to call the expanded goal stable while its reliability checks fail.

## Execution boundaries

- Start with local CPU; profile before choosing more hardware. Finite versioned
  batches, at most two concurrent local training workers initially.
- No paid resources, group-directory changes, credential storage, unattended Duo,
  destructive cleanup or new external publication. Existing cluster boundaries
  still apply if cluster use becomes necessary.
- A failed or ordinary first model triggers diagnosis and another bounded,
  controlled iteration, not a smaller redefinition of completion.

## Progress

### Final acceptance evidence, 2026-09-07

The original seven requirements above were checked against current files and
actual completed processes, not inferred from an intended plan:

| Requirement | Evidence inspected and result |
| --- | --- |
| Versioned rules and real cases | All30executable abilities retained; rule/source ledgers distinguish official patch notes, community version rows, real video components and specification tests. Upgrade/Cricket–Ox/start-turn/lethal-trigger/rounding/long-battle cases inspected and regression-tested; eight key late-added screenshot hashes reverified. No known remaining material contradiction was found. Exact target-client certification is NOT claimed; documented unobserved edge cases remain specification choices. |
| Tests, stress and legacy | `delivery-tests.xml`:615passing, no failures/skips; Ruff/diff checks clean. Current-source stress:6,000battles twice and400complete replays;200reachable states/3,458legal branches twice. All30species in battle stress. Old release300rows exact. Stress/mask source manifests equal the current frozen30-file manifest. |
| Contracts, pools, smoke/profile | Saved effective model/environment contracts and package versions; previous bounded smoke/profile plus full runs. Ten pool hashes verified. Distinct source-label sets train180/validation180/test600/challenge300 are pairwise disjoint; evaluation seed regions are distinct. This does not assert every naturally repeated team state is unique. |
| Independent, paired, pre-test selection | Six completed8,388,608budgets; paired initial hashes equal within each seed and distinct across three seeds. Validation-only delivery remains action_cost-seed1907. Freeze and ALL six exact600-row reloads preceded evaluation through the tested barrier runner. |
| Original numerical gate | All three candidates passed every original check, including strict per-family<1%forcing/truncation and>=90%macro success. Candidate test forcing2/9,000, truncation0/9,000; maximum family forcing0.1%. Exact rational-count recheck agrees with the frozen screen. |
| Separate challenge and baselines | All six models plus three baselines completed both suites (36,000episodes). Learned-challenge results disclosed separately, including delivery40%versus paired control44%; no post-test model switch and no claim of universal shaping improvement. |
| Replays and reproducible delivery |32delivery replays (20failures/12successes) plus all3candidate forced cases matched and reviewed. Remaining repetitions/value errors are explicitly documented, and all forced transitions have correct nonterminal semantics. Archived-source delivery reload600rows exact. Source ZIP/config/data/model hashes and runnable commands retained; no unexplained simulator exception, cleanup or new publication. |

Training supervisor464 and post-training supervisor503 both completed normally;
all replay/reload/test sessions also returned success. No further training is
queued by these supervisors. The automatic numerical and reconstruction artifacts
retain their original `goal_complete:false` / `manual_inspection_done:false` fields:
they intentionally do not substitute for this separate final manual audit.

This accepts the agreed cap-two, Tier3-reward thirty-pet **sandbox milestone**,
not full60-pet completion or the separate official-client-parity certification
gate in the roadmap/rule ledger. Those claims remain explicitly unmade. Literal
bug-freedom cannot be proven; evidence, tested behaviors and unobserved version
boundaries are kept distinct rather than relabelling tests as official footage.

### Historical progress (newest first)

- All six confirmation-v3 budgets completed; training supervisor464 is terminal.
  Full615-test suite passed in5.75seconds after the fifth training run finished,
  keeping actual training concurrency within two; Ruff/diff checks clean.
  All six validation reload artifacts exist. Validation-only delivery selection
  is action_cost-seed1907; first held-out evaluations have now started through
  post-training supervisor503. No numerical gate or final replay claim yet.
  Third seed control/action validation89.00%/95.17%; action has zero forcing/truncation.
  The delivery reconstruction helper now has13passing tests including simulated
  end-to-end success, incomplete-stage/model-change rejection and mismatch stopping.

- Confirmation-v3 now has four completed equal-budget runs. Seed1907 control/action
  selected final-update validation: 96.50%/96.83% ten wins, both zero forcing and
  truncation. The last seed2027 pair is live. Frozen source and all opponent pools
  still match; zero held-out-start markers at this check. Current non-training
  suite: 606 passed, five training tests deferred (611 total); Ruff/diff checks clean.
  A separate bounded delivery-replay helper is ready, with nine passing sample/
  comparison tests. It requires completed frozen evaluations and cannot train or
  select another model. Actual delivery reconstructions/manual inspection remain
  pending; see `REPLAY_INSPECTION.md` for the command.

- Additional mask-soundness audit passed on200reachable states,50from each
  random/stats/summon/tempo rollout family. All3,458advertised legal branches
  were executed on two independent copies, checking matching states/rewards/RNG,
  invariants and an unchanged original. All nine action kinds covered;28species
  appeared in originating teams. This is not official action-set completeness
  or full thirty-species mastery. Evidence:`expanded-mask-branches-v1`; source
  and helper hashes verified afterward. Runtime sources remain frozen.

- Current non-training tests597passed/five training tests deferred (602total).
  Two new tests cover the chosen Sheep-death/repeated-hit boundary; recent
  advanced Elephant footage did NOT isolate that rule and is not labelled a
  parity fixture. Existing v8 replay audit also establishes a policy-independent
  critic error:1,382/18,283values exceed remaining possible wins, with zero bonus.
  This is a diagnostic, not proof of historical loop causality or a new gate.
  Repeat on the final selected model. See `REPLAY_INSPECTION.md`.

- Confirmation-v3 seed1811 pair completed equal8,388,608 budgets, both choosing
  final-update600-row validation. Control/action ten-win94.17%/96.17%,
  unassisted94.00%/96.17%, forcing0.33%/0%, both0%truncation. These are validation
  only, not stable-delivery evidence. Seed1907 workers are now live with paired
  initialSHA`4bc53386fb97b04599dcbd7ac8c4a8b57c6d632b88f6a841dac5d615ae5eef5b`;
  seed2027 queues. Post-training runner waits for all six before selection,
  exact reloads and held-outs. Zero held-out-start markers at the latest audit.

- Current verification is595passed/five actual-training tests deselected,
  with600tests collected in total. New Crab L3 footage establishes19×75%=14.25→15,
  closing the ceiling-versus-nearest ambiguity left by6×25%=1.5→2. Its numeric
  regression passes with no source change; three archived key frame hashes
  verified in`expanded-rule-video-audit-v4`. The frozen source and all ten pools
  still match. Seed1811 control/action both exceed8Mdecisions, near their fixed
  budget endpoints; no held-out test started. Complete600-test suite still pending.

- Confirmation-v3 post-training work is now connected by a bounded write-once
  runner, with eight focused passing tests. It waits at most7,200seconds for all
  six completed budgets, then freezes choices, finishes ALL six exact reloads,
  evaluates all six models plus three baselines on the scripted/learned suites,
  and summarizes. At most two inference subprocesses; no training/retries/seed
  replacement/gate relaxation. Failure leaves a record and no completion claim.
  Runner session20613 is supervised independently of the two training queues.
  At launch and the latest audit, zero held-out-start markers exist. Training
  control/action seed1811 have both passed5Mdecisions; remaining seeds still queue.
  Final goal audit and selected-model replay inspection remain required.

- Recorded-case progress during confirmation-v3: newly inspected post-nerf
  Camel/Elephant lethal exchange from Ninjin Plays. Camel23/13 and Elephant20/3
  simultaneously reach−5/−18; dead Elephant still deals two1damage hits, then
  dead Camel gives its rear friend+2/+4. Two component regressions pass; passive
  supported recipients and observed net damage avoid inventing Garlic/Bison/
  Blowfish implementations. Five key archived screenshot hashes verified.
  Old Haps footage explicitly excluded from current numeric expectations.
  See `REAL_GAME_CASES.md` and `test_recorded_lethal_exchange.py`.

- Current non-training verification:586passed, five actual-training tests
  deselected while both confirmation workers are live. The five exclusions
  include legacy fresh-pair/continuation/shop-contract training tests, not only
  the two tests named smoke; this corrects the earlier incomplete exclusion
  description. Last full suite remains589passed before these two new regressions.
  Ruff, whitespace, frozen source/all ten pool hashes and captured frame hashes
  pass. Full591-test run remains to be performed when a worker slot is free.

- v8 now completed all three equal budgets; every model exactly reloaded180rows.
  All share initial policySHA`b85c0d5e8ad7d43401abb0a507dfb38c40a52bb91ab44fb39a43a003e550cefc`.
  Action/swap both92.78%tenwins, but unassisted92.78%/92.22% and forcing0%/1.11%;
  action cost is the validation-selected treatment. Fresh confirmation-v3 is
  prepared with the unchanged8,388,608 per-run budget, three paired seeds,
  test region7,000,000. Seed1811 control/action workers are live; later pairs
  queued under a two-worker supervisor. Source/pools verified unchanged; all
  held-out evaluations remain unopened. No stable-release claim yet.

- Current full suite is **589 passing tests**, including six exact-prefix tests
  for paired alternative-action inspection; Ruff and whitespace checks pass.
  Frozen v8 sources and pools remain unchanged. Control/action pilots completed
  and each archived model exactly reproduced its selected 180 validation rows;
  swap remains live. The action model's 180 replays contain 167 successes and
  13 battle failures, no forcing/truncation. Inspected all failure endings against
  twelve successes; two bounded paired probes confirm local inefficiencies,
  not global causal claims. See `REPLAY_INSPECTION.md`. No held-outs opened.

- During v8, all non-training tests pass: 581 passed, two training-smoke tests
  deliberately deselected to respect the two-live-worker bound. The last full
  suite before the six new reader tests was 577 passed. Ruff and whitespace
  checks are clean; frozen v8 source and all six pool hashes remain unchanged.

- Independent archived-source checks reproduced both completed confirmation-v2
  seed1811 models: 600 validation rows each, no training or held-out evaluation.
  A replay postprocessor outside frozen source now has six passing focused tests.
  It measures identical-pet swaps, repeated states, species/purchases/sales and
  observed discounted returns without treating truncation as a full return.
  Historical v7's 120 existing replays contain 79 identical swaps across seven
  episodes and zero sales; not new-rule delivery evidence. See `REPLAY_INSPECTION.md`.

- Per-ability rounding repair is now verified by the full 577-test suite, Ruff,
  6,000 twice-reproduced battles / 400 exact episode replays (`expanded-stress-v7`),
  and all 300 archived legacy rows (`expanded-audit-legacy-v5`). Fresh pilot v8
  starts three equal 4,194,304-step arms with at most two local CPU workers;
  control/action-cost live and swap-cost queued. Catalog digest is
  `2d71a64f5136f0a24fffb7d0ccc91ff9567d9cb7fae645119d932525642e23fe`.
  No held-out evaluation is open; goal remains active.

- New observed percentage cases exposed Dodo/Badger ceiling errors. Confirmation
  v2 was stopped before changing source: paired seed1811 complete, unequal partial
  seed1907 preserved, seed2027 never started, no held-outs opened. Per-ability
  catalog parameters now use Dodo/Badger floor and Crab ceiling, with observed
  chain/tooltip/damage/health fixtures. The separate gate float bugs were also
  repaired after the workers exited; all 24 gate tests pass with no xfails.
  Next: full regression/stress, fresh same-budget pilot, then independent seeds.

- A synthetic postprocessing audit found two false-negative boundary comparisons
  in the frozen numerical screen: exact 90% and exact one-percentage-point loss
  can be rejected by float means. Strict expected-failure regressions preserve
  the cases; they are not counted as passes. They affect no training or checkpoint
  selection. Keep the running source frozen, then fix exact-count comparisons
  before final sign-off; see `CONFIRMATION_NUMERICS.md`. No tests were opened.

- Long-battle evidence advanced: public headless YouTube playback allowed actual
  frame review of Skoottie's 2022 test-server clip. Thirty exchanges, final
  summons/buff, and Draw with survivors were observed. Detector menu false
  positives and one merged real attack were explicitly resolved by visual review;
  a playback failure was covered by a separate ending pass. Evidence is archived
  locally with a timestamped component fixture, and its focused regression passes.
  This is historical evidence, not an exact v0.46 or natural-last-hit parity claim.
  No frozen training source or catalog changed. Full suite last passed at 552;
  the additional component test passes separately (25 focused tests in this run).

- v7 subsequently completed all three arms; swap-cost selected ten-win success
  89.44%, forcing 2.22%, exact 180-row reload. Its 30/40/50 limit sensitivity also
  changed no validation rows. Fresh confirmation v2 is prepared and its first
  paired seed is live, with the same prespecified 8,388,608 budget and a new
  held-out seed region starting at 6,000,000. All six runs will complete before
  freezing selection and opening tests. Current source remains frozen.
- Reconstructed v7 action-cost stats/tempo replays: 120 episodes, 108 successes,
  12 ordinary battle failures, no forcing or truncation. Mean repeated states
  23.43 per episode; zero sales. This is not final delivery-model replay sign-off.

- Latest continuation: full suite is 552 passing tests, Ruff and whitespace checks
  clean. v7 control and action-cost completed their equal 4,194,304-step budgets;
  selected validation scores are 82.78% / 92.78% ten wins and 1.11% / 0% forcing.
  Both saved models exactly reloaded all 180 selected rows. Counterfactual limits
  40 and 50 changed no rows for either fixed model; this is conditional validation
  sensitivity, not official-rule verification. Swap-cost remains running, and no
  held-out scores have been opened. Original-client long-battle parity remains open.

- Entry state re-inspected: v4 implementation, 339 prior tests, 30 real abilities,
  development gate active. No expanded training yet.
- Real-client access started through the developer's official itch.io game.
- Goal remains active; all completion requirements are presently unproven.
- Continued audit fixed token levels, Honey summon placement, actual pre-battle
  snapshot capture and gold-observation aliasing. Full suite: 352 tests passed.
- Real-game provenance and outstanding target-client checks: [case ledger](REAL_GAME_CASES.md).
- Explicit experimental training/save/reload smoke test passed. Expanded model
  contracts fingerprint catalog content/order and restore the development opt-in.
- `runs/expanded-pilot-v1/` completed, but both short-run models looped extensively.
- v2 exposed an upstream categorical-probability cache bug; both interrupted
  arms are retained. The opt-in expanded policy backports the upstream repair;
  validation remains enabled and frozen legacy policies are unchanged.
- v3 completed three equal-budget million-step arms. General action cost is the
  most promising at this seed/budget, but 14.44% of validation episodes still
  need forcing and only 8.89% reach ten wins. Not stable; not a test-set claim.
- v4's longer comparison was intentionally stopped when the official audit
  exposed missing non-maxed random-buff priority. Logs/checkpoints remain intact;
  fresh v5 will use repaired rules. See [pilot evidence](EXPANDED_PILOTS.md).
- Three observed video components now have regression fixtures: historical
  Cricket/Ox, recent Giraffe/Worm start-turn and Ant linked upgrade rewards.
- `runs/expanded-stress-v2/`: 6,000 synthetic mixed battles reproduced twice,
  400 legal full episodes exactly replayed. v1 stopped on a harness method-name
  error, not a simulator failure; its incomplete directory remains intact.
- Expanded learned-opponent snapshot generation now captures actual battle
  boundaries, including forced endings, with tests. No candidate challenge
  evaluation or final independent-seed confirmation has occurred yet.
- v5 was intentionally stopped before correcting two further shared rules:
  target-inherited/level-up-reset counters and perk-gain food notifications.
  Both are catalog-fingerprinted; focused tests pass. Complete rule review and
  regressions before the next frozen pilot. The stability goal remains unmet.
- Follow-up full suite: 499 passed. `expanded-stress-v4` reproduced 6,000 mixed
  battles twice and replayed 400 legal episodes exactly. v6 begins a fresh
  four-million-step three-arm pilot with repaired rules and fresh pools.
- Legacy check `expanded-audit-legacy-v3` again matched all 300 archived episode
  rows exactly (95/90/83 ten-win successes by family); release model unchanged.
- Added a separate numerical stopping-screen helper (`expanded_gate.py`) with
  12 tests: it recomputes rates from raw rows and rejects insufficient, unpaired
  or duplicate-seed evidence. One bad seed/family cannot hide behind a macro
  average. Passing this helper explicitly does not certify provenance, parity,
  learned-opponent generalization or goal completion. Full suite now 511 passed;
  this helper is not imported by or part of the frozen v6 training source set.
- v6 control/action-cost are still running toward 4,194,304 decisions each;
  swap-cost remains queued until a local training slot is free. The first
  million-step action-cost checkpoint reaches 23.33% validation ten-win success,
  but 19.44% forcing remains. Thirty trajectories were inspected and archived;
  see the pilot log. No final confirmation or learned-challenge test is open.
- v6 control and action-cost completed their full 4,194,304-step budgets.
  Validation selected action-cost at 92.78% ten-win success with zero forcing
  in 180 episodes; control's selected model is 82.78% success / 1.11% forcing.
  Both saved models exactly reproduced all 180 selected validation rows on
  reload. Swap-cost is now the only live training worker and has not completed.
- Independent-seed confirmation tooling and nine tests are implemented; full
  suite is 520 passing tests. It freezes all choices before test, requires exact
  reload checks and evaluates a separate unseen learned-policy pool. It has not
  been dispatched; the full pilot comparison must finish first.
- Inspected 120 selected action-model validation trajectories, including all
  12 stats/tempo failures. They are ordinary battle losses, not shop cutoffs.
  Residual redundant moves and almost no team replacement remain limitations;
  high success is not evidence of mastery of all 30 pets.
- The Mac is currently locked, so additional browser frame-by-frame checks are
  unavailable without user interaction. Do not wake the user or accept game TOS;
  headless training, rule/source review and existing captured-case regressions
  continue. This is not a whole-goal blocker while useful work remains.
- Added two tests for independent permanent/temporary merge maxima and expiry,
  supported by the descriptive Experience reference; no simulator change was
  needed. Full suite is 522 passing tests, formatting/whitespace checks clean.
  The proposed confirmation budget is fixed at 8,388,608 decisions per run
  before seeing held-out scores; independent seeds and both arms keep equal budgets.
- v6 swap-cost completed: selected success 89.44%, unassisted success 87.78%,
  forcing 2.22%. It also reproduced all 180 selected validation rows on reload.
  The full same-seed, equal-budget pilot now favors general action cost 0.005
  for confirmation, not as a universal claim.
- `expanded-confirmation-v1` is prepared with ten distinct frozen pool hashes:
  three training, three validation, three new scripted tests and one unseen
  learned-policy challenge (4,157 reachable snapshots from the pilot control).
  No held-out scores have been opened. The first paired seed 1811 is now
  training in two CPU workers; seeds 1907/2027 remain queued. Each run receives
  8,388,608 decisions. All current source files are frozen for this batch.
- A targeted low-attack/high-health case exposed a known simulator exception:
  five 1/50 Swans on each side exceed the 200-attack technical guard. Randomized
  stress had reached only 35 attacks and did not cover this boundary. The queue
  and both first-seed workers were stopped; exit codes 143 and partial counts
  are recorded in `expanded-confirmation-v1/aborted.json`. No test scores opened.
  A strict expected-failure regression preserves the problem without hiding it
  in a passing-test total. Public firsthand reports support a long-battle draw,
  but conflict on exact count (30 versus 40); official Steam notes inspected so
  far do not settle it. No arbitrary new cutoff has been implemented. The goal
  remains active; rule-source verification and a versioned repair are next.
- Continued source checks found multiple 30-attack community descriptions but
  no original-client constant, and browser access is still locked. Implemented
  a catalog-fingerprinted **provisional** 30-exchange draw mechanism rather than
  treating the old technical exception as a game loss. Boundary tests cover
  natural last-exchange outcomes, last-exchange summons, non-attack ability
  events, scoreboard/episode continuation, distinct evaluation metrics, invalid
  limits and unchanged legacy fingerprints. Original-client threshold/order
  verification remains an explicit open requirement, not completed by unit tests.
- Full suite: 543 passed. New stress includes adversarial low-attack/high-health
  stats: 6,000 battles reproduced twice, 400 exact full replays, 800 limit draws.
  The old delivery model again matches all 300 archived rows exactly. v7 is now
  training three equal-budget 4,194,304-decision arms on newly frozen source/pools;
  two live CPU workers and one queued arm. No held-out tests have been opened.
- Added a trusted-local historical replay helper outside the frozen training
  source set. It verifies source ZIP membership/digests, isolates old code in a
  temporary directory, remaps validation paths and compares every saved row.
  v6 action-cost exactly reproduced all 180 selected validation rows under its
  archived rules. Seven focused extraction tests pass. Current v7 source/pool
  hashes remain unchanged, and its first two initial parameter hashes match.
- The same archived-source helper also reproduces all 180 v6 control validation
  rows. A separate, validation-only 30/40/50 battle-limit sensitivity script and
  two pairing/full-row tests are ready for completed v7 models. It refuses to
  choose a rule or open test data; it is not a replacement for original-client
  evidence. It has not run yet because no v7 arm is complete.
