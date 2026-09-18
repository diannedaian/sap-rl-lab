# Roadmap: all 60 Turtle Pack pets

**Ultimate goal: a version-pinned, testable simulator and RL agent for the entire
Turtle Pack, including its foods, perks, tokens, shop progression, and interacting
abilities.** Eight pets were Phase 1, not the project's endpoint.

The user reaffirmed this goal on 2026-09-06. Round 5 closes the eight-pet experiment;
it does not cancel the full-pack goal. This document plans the next phase, but
does not authorize unbounded training or silently change old rules.

## Completed full-pack stage — 2026-09-18

All 60 pets and normal Tier 1–6 shops are implemented in the version-pinned
sandbox. Three validation-selected models reached 82.27% mean ten-win success
on the held-out learned pools and 81.87% on a second episode-seed confirmation.
See [project wrap-up](PROJECT_WRAPUP.md), [model card](MODEL_CARD.md), and the
[frozen delivery report](FULLPACK_DELIVERY.md). This completes the educational
project, not certification of official-client parity. Adjacent-only movement
and the other documented simulator differences remain explicit limitations.
No further training or rule changes are required for this release.

## Historical stage — 2026-09-14

The user approved the next bounded curriculum after reviewing the completed
BC/KL experiments. **v6 now implements 50 pets: normal Tier 1–4 shops plus all
ten Tier 5 level-up reward dependencies.** It is an experimental, versioned
simulator, not a full 60-pet release or certified official-client parity.
See the [new rules and limitations](TIER4_RULES.md),
[frozen training recipe](TIER4_PLAN.md), and [current launch status](TIER4_STATUS.md).
The old 40-pet weights and results remain intact; the previous mean-80% gate
was not passed and is not retroactively reclassified. New input dimensions
require fresh BC initialization, followed by the same small-MLP Maskable PPO
and KL-brake recipe. This is not a weight-transfer experiment.

The milestones below describe earlier stages; their historical counts and
then-pending work do not supersede the latest status above.

Implementation has started: see [the Tier 1–2 development milestone](TIER12.md).
The separate v4 development catalog implements **30 pets**: normal shops capped
at Tier 2, complete Tier 3 for level-up rewards and Spider summons, associated
foods/perks, and shared shop/battle events. See [rules and evidence](TIER12_RULES.md).
It is specification-tested and available for explicitly experimental training,
with multiple observed video-component fixtures. The thirty-pet curriculum passed
its predeclared training reliability gates; it is not certified full-client parity
or a full-pack release. See [delivery results](EXPANDED_DELIVERY.md).

The next **v5 forty-pet curriculum** is now integrated: normal shops through
Tier 3, all ten actual Tier 4 level-up dependencies, and the Tier-3 food pool.
Its pilot, six fresh opponent generators and three paired scripted/mixed training
runs are complete, with 30,000 independently held-out test episodes and 42 exact
diagnostic replays. Two pairs improved on learned opponents; both arms of the
third pair collapsed to empty-team losses. This is a completed experiment, **not
a stable-training or full-pack-strength claim**. See the
[frozen experiment](MIDGAME_CONFIRMATION.md), [current report](MIDGAME_TRAINING_REPORT.md)
and [versioned rule evidence](MIDGAME_RULES.md). Normal Tier 4 shops still require
Tier 5 level-up dependencies; the full-pack goal is not reduced to these 40 pets.

The subsequent bounded stability investigation is complete, with five fresh
paired training seeds and a separate registered checkpoint-selector comparison.
Neither fixed entropy 0.01 nor reliability-first filtering confirmed a robust,
performance-preserving recipe. See the [short report](STABILITY_REPORT.md).
The proposed reward-cost ablation has not started and requires a new bounded
budget; normal Tier 4 shops and full-pack training remain future work. This does
not replace the previous delivery or change the full Turtle Pack objective.

## Where we are

Implemented: eight rollable tier-1 pets, a small food curriculum, deterministic
shop/battle engine, 71 masked shop actions, PPO, frozen scripted opponent pools,
six-run confirmation, and a replay inspector. See [Round 5](ROUND5.md).
The separate thirty-pet curriculum has fresh scripted opponent pools, bounded
pilots and completed three-pair independent-seed confirmation. Its learned-opponent
challenge remains substantially harder; full-client parity is still separate.

Not yet full-pack parity: complete roster/food data, complete reward pools,
verified shop/merge edge cases, all trigger types, perks/damage modifiers, or
verified priority ordering. The development curriculum has thirty implemented
abilities, not thirty officially certified mechanics. Fish is fixed; the parity limitations
in [DATA_ACCURACY](DATA_ACCURACY.md) remain open.

## Expansion gates

| Stage | Deliverable | Gate before larger RL training |
| --- | --- | --- |
| 1. Freeze the target | One official game version; inventory of all 60 pets, foods, perks and tokens, with evidence/status per item | Unknown or unsupported mechanics explicitly disabled, never silently approximated |
| 2. Complete early game | All tier-1 and tier-2 content, correct turns 1–4 shop rules, economy and level-up behavior | Every enabled ability tested at all levels; official-game parity fixtures for ordering and interactions |
| 3. Midgame | Tier 3–4; hurt/attack/knockout chains, scaling, positioning, food/perk combinations | Shared event system, deterministic replay, invariant tests, and new opponent pools pass |
| 4. Late game | Tier 5–6; copying/repetition, richer summons and damage modifiers | Explicit resolution limits and tested multi-ability chains; no unexplained simulator exceptions |
| 5. Full Turtle agent | All 60 enabled pets plus related content, realistic progression and a frozen full-pack benchmark | Held-out opponents, several training seeds, failure replays, runtime and reliability reports |

Add content in small mechanic-based batches **within** those stages. A pet that
reuses a tested buff primitive is cheaper than a pet introducing a new trigger.
Reaching 60 catalog names alone is not completion.

## How the implementation should scale

**Engine first.** Separate target selection, trigger scheduling, effect application,
and priority/tie-breaking into reusable primitives. Preserve the pure engine's
independence from Torch/Gymnasium. Capture golden event traces for simultaneous
faints, summoned pets, dead targets, perk consumption and copied abilities.
Preserve the archived eight-pet engine and catalogs for historical replay.

**State before network size.** The legacy shop stores item IDs; the opt-in
development shop now observes mutable pet stats and linked reward choices. Extend
versioned, append-only IDs for all target pets/foods/perks, slot-presence masks and
explicit per-pet counters. Represent every field affecting future transitions.
Expanding or reordering the one-hot vocabulary can shift old IDs and invalidate
weights: never assume an old checkpoint loads correctly just because shapes fit.

**Keep a small PPO baseline.** First expand the observation and train a new baseline
on the expanded rules. Shared per-pet encoders with identity/ability features and
position embeddings are a sensible later comparison; a transformer is optional,
not the entry ticket. Team order matters, so do not erase position by pooling.
Species count alone need not multiply action IDs: slot-based actions already
generalize across species. Revisit the codec only for larger shops or genuinely
new action types. Version the observation/action contract with each model.

**One curriculum boundary at a time.** Train early-game tiers first, then unlock
later tiers while retaining some earlier scenarios. Compare warm-starting with a
fresh model at a boundary instead of assuming transfer helps. Do not compare raw
win rates across different rosters/opponent pools as though difficulty were fixed.

**Scale opponents as well.** Keep interpretable scripted opponents, but add frozen
snapshots from several policy generations and strategies. Refresh only the
training pool on a declared schedule; keep validation/test pools separate. A
snapshot league is sufficient for asynchronous arena play; simultaneous live
self-play is not necessary. Evaluate by opponent family, not only one average.

**Profile before using more hardware.** Measure environment steps/sec, PPO update
time and evaluation time at each content stage. Use CPU workers if simulation
dominates; consider GPU batched inference/updates only when the network warrants
it. More pets do not automatically make this a GPU-heavy task. Compare serial and
subprocess execution on the same workload before choosing a worker count.
See [SB3's vector-environment documentation](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html).

## Handle looping as a separate environment-design decision

Swap cost 0.005 remains an eight-pet finding, **not an automatic full-pack default**.
Round 5 still showed both swapping and freeze/unfreeze loops. Do not indefinitely
add a new penalty for each discovered loop or declare positional changes useless.

Before expanded training, specify whether a shop-decision limit is an external
collection cutoff, a game forfeit, or a forced transition to battle. These are
different tasks with different value targets. True termination and external
truncation must not be conflated; see [Gymnasium's time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
If changing the task, give it a new environment version and report the change.
Check exact no-op/reversal masking against the full observable state and
declared rules, not just two pets' names.

## Concrete next implementation slice

1. Build the versioned 60-pet coverage ledger, using official patch notes and
   controlled game fixtures. Record uncertain facts instead of guessing.
2. Audit early-game shop state, slot progression, food availability and level-up
   effects; extend the engine/state contract with tests.
3. Tier 1–2 and the Tier-3 dependency pool are now implemented. Validate the
   event and shop edge cases listed in the v4 ledger before unlocking more tiers.
4. Diagnostics, frozen reward comparison and paired independent-seed confirmation
   have completed on fresh expanded pools under the
   [predeclared completion contract](EXPANDED_GOAL.md). Preserve this benchmark;
   later expansion needs new next-tier dependencies and broader opponent pools,
   not more tuning on this now-observed test set. See the limitations in the
   [delivery report](EXPANDED_DELIVERY.md).

The first practical milestone is **a correct tier-1/2 game**, not another slightly
higher eight-pet score. Screen automation, game art, research-scale populations,
and cluster orchestration are not prerequisites for 60 pets.
