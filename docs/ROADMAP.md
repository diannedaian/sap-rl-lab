# Roadmap: all 60 Turtle Pack pets

**Ultimate goal: a version-pinned, testable simulator and RL agent for the entire
Turtle Pack, including its foods, perks, tokens, shop progression, and interacting
abilities.** Eight pets were Phase 1, not the project's endpoint.

The user reaffirmed this goal on 2026-09-06. Round 5 closes the eight-pet experiment;
it does not cancel the full-pack goal. This document plans the next phase, but
does not authorize unbounded training or silently change old rules.

## Where we are

Implemented: eight rollable tier-1 pets, a small food curriculum, deterministic
shop/battle engine, 71 masked shop actions, PPO, frozen scripted opponent pools,
six-run confirmation, and a private replay inspector. See [Round 5](ROUND5.md).

Not yet full-pack parity: complete roster/food data, real shop-slot progression,
all experience/level-up rewards, persistent shop-pet stats, all trigger types,
perks/damage modifiers, or verified priority ordering. Do not equate eight
implemented pets with complete tier 1. Fish is fixed; the parity limitations
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

**State before network size.** The current shop stores item IDs, not mutable pet
stats; shop-buffing pets therefore need richer state and observations. Introduce
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
3. Add the missing tier-1 and tier-2 pets in a few mechanic-based batches.
4. Run simulator diagnostics and scripted baselines; then one bounded paired
   PPO smoke comparison on fresh pools. Set budget and acceptance criteria
   before running, rather than launching another hyperparameter sweep.

The first practical milestone is **a correct tier-1/2 game**, not another slightly
higher eight-pet score. Screen automation, game art, research-scale populations,
and cluster orchestration are not prerequisites for 60 pets.
