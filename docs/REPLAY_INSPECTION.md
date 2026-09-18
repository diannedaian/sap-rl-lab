# Reproducible replay inspection

This is diagnostic evidence, not an additional benchmark or a causal explanation
of a policy's entire training history. Keep the checkpoint and opponent pool fixed.

## Two distinct steps

1. `sap_rl_lab.expanded_diagnostics` reconstructs saved-policy episodes, including
   pre-action state, selected action, top-five probabilities, critic value,
   objective reward, forced-battle flags and termination/truncation.
2. `scripts/summarize_policy_replays.py` reads those **existing** files. It counts
   swaps of fully identical pet records, consecutive same-slot swaps, repeated
   states, purchases including merges, sales and species present in the team.
   It also computes the observed discounted remaining shaped reward using the
   saved run's gamma and preserves input/model/manifest/script hashes.

The second step neither trains nor generates new trajectories. Output files are
write-once. It rejects changed model hashes, mismatched episode metadata and
incomplete trajectories. Six focused tests cover these episode computations,
non-identical pets, discounting, truncation, forcing, purchase/merge/sale slots.

Example on already-generated historical development replays:

```sh
.venv/bin/python scripts/summarize_policy_replays.py \
  --directory runs/expanded-v7-replays-stats-v1 \
  --output runs/expanded-v7-replays-stats-v1/behavior-audit.json
```

That output already exists; choose a new output path only when another audit is
actually required. New-rule models need their own reconstruction directories.

## Final confirmation reconstruction and review

After the post-training pipeline completes, use the validation-frozen delivery
checkpoint, not the best test-scoring seed. The following helper selects the
earliest five failed and three successful episodes in each scripted/challenge
family, plus up to five forced/truncated examples, deduplicated. This is an
outcome-stratified diagnostic sample, **not** an estimate of population rates.

```sh
MPLCONFIGDIR=/private/tmp/sap-rl-mpl-cache .venv/bin/python \
  scripts/inspect_confirmation_delivery.py \
  --confirmation runs/expanded-confirmation-v3 \
  --output runs/expanded-confirmation-v3-delivery-replays-v1
```

The output path must be new. The helper requires completed frozen evaluations,
verifies source/pool/model/selection provenance, reuses the existing inspector,
and compares each reconstructed game summary with its held-out evaluation row.
Differences fail explicitly rather than silently treating a different trajectory
as the original. No model training or selection occurs. Thirteen focused tests
cover sample quotas, ordering, deduplication, mismatched game summaries and the
end-to-end orchestration with a simulated inspector. Incomplete evaluations or
changed checkpoints prevent reconstruction; a differing replay stops later cases
and leaves a failure record. The actual delivery run completed all32cases exactly.
Successful reconstruction is not manual inspection:
its completion artifact deliberately leaves `manual_inspection_done` false.

The subsequent review is recorded separately in
`runs/expanded-confirmation-v3-delivery-replays-v1/manual-review.md`: all20failure
and12success endings, probabilities, values and battle traces were checked.
All3,174recorded decisions were scanned:457consecutive same-slot swap pairs
(overlapping), one identical-Flamingo swap, and61critic estimates above the
conservative remaining-win upper bound. These outcome-selected counts are not
population rates or a historical causal attribution.

All three forced candidate episodes were also reconstructed and exactly matched
under `runs/expanded-confirmation-v3-forcing-replays-v1`: learned seed7300025 for
training seed1811, and scripted stats7100186/tempo7100060 for training seed2027.
Repetitive swaps/freezes consumed their budgets; each forced battle correctly
continued to a new shop with both ending flags false. The delivery model itself
has zero forcing/truncation in its complete4,000held-out episodes. Inefficient
actions and imperfect critic predictions remain disclosed limitations, not
hidden simulator exceptions or proof that every failed match was avoidable.

## Historical v7 inspection, not v8 delivery evidence

The existing stats/tempo samples contain 120 episodes: 108 successes and 12 normal
battle failures, zero forced shop endings or external truncations. The new reader
finds **79 swaps of identical pet records across seven episodes** and 2,296
consecutive same-slot swap pairs (overlapping pairs are counted). There were no
sales. Team states contain 26 species; purchases/merges include eight Tier-3
species, but this does not establish competent use or full thirty-pet mastery.

All twelve failures' last decision has a positive critic value but a realized
terminal shaped reward of −1.005. These are conditional-on-failure examples:
one unfavorable realization is **not** proof that the expected value is wrong.
Inspect successes alongside failures and use paired alternative-action rollouts
before attributing failures to a particular action/value mechanism.

## Current-rule v8 pilot inspection

The saved action-cost checkpoint reproduced all 180 validation episodes, with
167 ten-win successes and 13 ordinary battle failures, no forced shop endings or
external truncations. These are validation results, not held-out confirmation.
Files are in `runs/expanded-v8-replays-action-{stats,summon,tempo}-v1`.

The audit found zero identical-record swaps but 2,734 consecutive same-slot swap
pairs, and 37 sales. Twenty-nine ordinary species plus Zombie Cricket appeared;
Dolphin did not. All thirteen failures' terminal decisions were inspected beside
the first four successful seeds in each family (twelve successes). Successful
episodes also include long swap sequences and some unspent-gold endings. Thus
inefficiency is not confined to failures, and zero forcing is not perfect play.
Nor does shaping guarantee fewer actions than the selected control: v8 control
averages97.17 decisions/game and7.03/battle; action cost averages101.57 and7.54.
The equal-budget pilot selected action cost for higher unassisted success, not
for lower action counts. Swap cost averages114.95/game and8.53/battle.

Two preselected development states were then reconstructed exactly and compared
with 64 paired future-RNG continuations per alternative. The checkpoint and
training sources were unchanged. `scripts/inspect_replay_alternatives.py` reuses
the existing bounded rollout helper; six tests check prefix reconstruction and
reject changed states, wrong seeds, invalid indices and prematurely ended games.

| State and intervention | Mean shaped future return | Interpretation |
| --- | --- | --- |
| Tempo seed2300001, step41: end turn vs Apple on slot3, with 3 gold remaining | 0.0266 vs 0.4777; paired difference +0.4510, SE 0.1515 | Buying the Apple did better in these continuations; neither branch reached ten wins in any of the 64 samples. |
| Stats seed2300040, step73: freeze slot1 vs end turn with zero gold | −0.9762 vs −0.9262; every paired difference +0.0500 | Ending immediately avoided ten paid freeze/unfreeze decisions; the tested raw outcomes were unchanged. |

The policy preferred end-turn over Apple by 53.69% vs 46.17% in the first state;
it preferred the freeze action with 65.52% probability in the second. This is
evidence of locally imperfect decisions, not proof that shaping globally harms
play or that a different action would rescue these entire episodes. The critic's
first-state value (4.1195) also exceeded both sampled continuation means, but the
saved critic is trained with sampled-policy targets whereas these probes follow
deterministic policy actions; this is not a clean global value-calibration test.

The two `alternative-*.json` reports retain full returns, states, probabilities
and input/helper hashes. Future randomness is paired but action-dependent draw
consumption can diverge. These development probes do not alter candidate selection,
the confirmation budget or the held-out thresholds. Final delivery still needs
its own multi-seed, held-out results and selected-model replay inspection.

### A policy-independent value bound

A separate read-only audit of the same180v8 action-cost replays verified the
checkpoint hashes, gamma1, zero success bonus, and every recorded shaped reward
equal to its game reward minus0.005. Under these rules, the only positive game
reward is+1 for a win, and the episode ends at10wins. Consequently **any policy's
expected remaining return is at most `10 - state.wins`**. Ignoring action costs
makes this a conservative upper bound, not an estimate from lucky/unlucky outcomes.

The saved critic exceeds that upper bound by more than1e-6 at1,382of18,283recorded decisions,
across119of180episodes. At9wins,629of1,307decisions exceed1. Example:
summon seed2300039, step109, `swap_adjacent:2,3`, predicts2.3542 despite an upper
bound of1. These are repeated-state observations, not independent statistical
samples or a new held-out benchmark. The bound applies to sampled or deterministic
policies, unlike the earlier paired-continuation comparison.

This establishes some impossible value predictions from the approximate critic.
It does **not** prove that those predictions caused the actor's historical loops:
these are saved-checkpoint predictions, not the original training advantages.
Do not silently clip values or change training in response. Repeat this bound
check on the final frozen delivery model alongside successful/failed replays;
retain it as a diagnostic, not a newly invented numerical stopping gate.

## Incentives and interpretation

The unshaped control uses gamma=1 and zero cost for an ordinary swap. A reversible
detour with the same eventual game outcomes therefore has the same raw objective
until it hits a resource/action limit. Action cost 0.005 makes the detour expensive
in the training objective, but an approximate learned policy need not eliminate
every inefficient action. This explains an incentive difference, not a verified
causal reconstruction of any individual failure's training history.

- Identical-record swaps leave the pet arrangement unchanged; the shop-action
  counter and reward may still change. They are not the same as every swap of two
  pets of the same species, which can differ in stats, XP, perks or counters.
- Consecutive same-slot swaps are a useful search filter, not a standalone proof
  that every such pair caused a loss. These pairs can overlap in a longer loop.
- A forced shop ending enters battle; it is not an external episode truncation.
- At external truncation, the remaining recorded reward is only a partial return.
  The reader reports no critic-error claim and invents no bootstrap value.
- Saved-policy values are reconstructions, not historical PPO rollout values.
- Final inspection should cover roughly 10–20 failures plus successes and any
  forcing cases under the final frozen rules, not merely reuse this old sample.
