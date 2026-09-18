# Thirty-pet curriculum: implementation and evidence

Checked 2026-09-07. Catalog `turtle-v0.46-tier12-curriculum-v4`; target game
version 0.46. **Executable, specification-tested development environment, not
certified official-client parity.** The released eight-pet rules stay frozen.

## Agreed boundary

- Normal shops: Tier 1 on turns 1–2, Tier 1–2 thereafter. Never roll Tier 3.
- Level-up offers: exactly the **current shop tier + 1**, two linked paid choices.
  Thus offers are Tier 2 initially and Tier 3 thereafter, even for a Tier-3 owner.
- Spider: complete ten-pet Tier-3 pool, with actual abilities at its summon level.
- Shop capacity still follows actual turn: 3/4/5 normal pet slots at turns 1/5/9,
  1/2 normal food slots at turns 1/5, plus two stock spaces. The tier cap does not
  freeze capacity. This is a curriculum, not normal full-pack progression.
- Thirty shop actions force one ordinary battle; ten wins/life exhaustion end a
  game. The separate whole-game turn limit remains an external truncation.
- Four tokens, five normally rollable foods, generated Bread Crumbs/Better Apple/
  Best Apple, and Honey/Meat Bone/Melon are included. No Tier-3 normal food pool.
- Default scripted opponents use the same normal tier cap, but do not model
  learned level-up strategies. New frozen expanded training/validation pools
  now exist in the [pilot series](EXPANDED_PILOTS.md); held-out confirmation has
  passed the declared training reliability gates, with limitations in the
  [delivery report](EXPANDED_DELIVERY.md). Old eight-pet benchmarks are not
  expanded-game benchmarks. This does not close the official-client parity gates.

## Ability coverage

Every listed pet has executable abilities, with level-1/2/3 tests (Fish's ability
uses the departing level at its two possible upgrades). Numbers separated by
slashes are level-specific. Position 0 is the front of the team.

| Tier | Pet | Implemented rule |
| --- | --- | --- |
| 1 | Ant | Faint: one distinct living friend gains +1/+1, +2/+2, +3/+3 |
| 1 | Cricket | Faint: summon a 1/1, 2/2, 3/3 no-ability Cricket token |
| 1 | Fish | Level 1→2 / 2→3: two friends gain +1/+1 / +2/+2 |
| 1 | Horse | Friend summoned: +1/2/3 attack until next turn |
| 1 | Mosquito | Start battle: 1 damage to 1/2/3 distinct enemies |
| 1 | Otter | Buy: 1/2/3 distinct friends gain +1 health; base stats **1/4** |
| 1 | Pig | Sell: +1/2/3 extra gold, in addition to normal sale value |
| 1 | Pigeon | Sell: stock 1/2/3 free Bread Crumbs |
| 1 | Duck | Sell: all current shop pets gain +1/2/3 health |
| 1 | Beaver | Sell: two distinct friends gain +1/2/3 attack |
| 2 | Snail | End turn after a loss: three nearest friends ahead gain +1/2/3 attack |
| 2 | Crab | Start battle: **gain**, not replace, 25/50/75% of healthiest other friend's health |
| 2 | Swan | Start turn: gain 1/2/3 gold |
| 2 | Rat | Faint: 1/2/3 Dirty Rats at opponent's front; no summon notifications |
| 2 | Hedgehog | Faint: deal 2/4/6 damage to all other pets on both teams |
| 2 | Peacock | Hurt, including lethal damage: gain 3/6/9 attack |
| 2 | Flamingo | Faint: two nearest friends behind gain +1/+1, +2/+2, +3/+3 |
| 2 | Worm | Start turn: stock a 2-gold Apple / Better Apple / Best Apple |
| 2 | Kangaroo | Friend immediately ahead attacks: gain +1/+1, +2/+2, +3/+3 |
| 2 | Spider | Faint: random Tier 3 with 2/2, 4/4, 6/6 stats and matching ability level |
| 3 | Dodo | Start battle: nearest friend ahead gains 50/100/150% of own attack |
| 3 | Badger | Faint: adjacent pets take 50/100/150% of own attack |
| 3 | Dolphin | Start battle: 4 damage to lowest-health enemy, 1/2/3 shots |
| 3 | Giraffe | **Start turn**: nearest 1/2/3 friends ahead gain +1/+1 |
| 3 | Elephant | After attack: nearest friend behind takes 1 damage, 1/2/3 hits |
| 3 | Camel | Hurt, including lethal: nearest friend behind gains +1/+2, +2/+4, +3/+6 |
| 3 | Rabbit | Food eaten: eater gains 1/2/3 health, three uses per turn |
| 3 | Ox | Friend immediately ahead faints: +1 attack and Melon, 1/2/3 uses per turn |
| 3 | Dog | Friend summoned: gain +2/+1, +4/+2, +6/+3 until next turn |
| 3 | Sheep | Faint: two Rams with 2/2, 4/4, 6/6 stats |

Apple is permanent +1/+1; Cupcake is temporary +3/+3. Meat Bone adds **3 attack
damage**, not 3 base attack. Melon blocks 20 damage once; fully blocked damage is
not Hurt. Sleeping Pill costs 1, directly faints the eater (no Hurt), and runs the
shared faint/summon system in the shop. Rabbit cannot revive the pilled pet.

## Sources and version control

The developer's [official 0.27 announcement](https://steamcommunity.com/games/1714040/announcements/detail/3698065064748065722)
establishes changed merge experience, shop rewards/stock spaces and lethal Hurt /
After Attack behavior. It is historical evidence, not proof of all 0.46 rules.
The [v3 ledger](TIER12_SOURCES.md) records shop evidence and unresolved details.

All 30 pet records were checked against the **16 March 2026 historical rows** on
Grounded SAP profiles, not blindly copied from today's ability descriptions.
These are community-maintained records, not an official authoritative export.
Profile URLs have the form `https://groundedsap.co.uk/PetProfile.aspx?ID=N`:

- Tier 1 IDs: Ant 0, Cricket 17, Fish 32, Horse 39, Mosquito 47, Otter 51,
  Pig 59, Pigeon 559, Duck 26, Beaver 3.
- Tier 2 IDs: Snail 72, Crab 16, Swan 76, Rat 57, Hedgehog 37, Peacock 54,
  Flamingo 29, Worm 82, Kangaroo 40, Spider 74.
- Tier 3 IDs: Dodo 21, Badger 2, Dolphin 23, Giraffe 33, Elephant 28, Camel 10,
  Rabbit 60, Ox 52, Dog 22, Sheep 68.

Examples: [Otter](https://groundedsap.co.uk/PetProfile.aspx?ID=51),
[Crab](https://groundedsap.co.uk/PetProfile.aspx?ID=16),
[Giraffe](https://groundedsap.co.uk/PetProfile.aspx?ID=33).
The Otter audit found 1/4 instead of the old sandbox's 1/3; only v4 is corrected.
Food/perk evidence: [foods](https://groundedsap.co.uk/Foods.aspx),
[perks](https://groundedsap.co.uk/Perks.aspx),
[Meat Bone history](https://superautopets.wiki.gg/wiki/Meat_Bone).
The [Rat reference](https://superautopets.wiki.gg/wiki/Rat) specifies the important
no-summon-trigger exception for enemy Dirty Rats.

## Shared event model and remaining parity gates

Long battles use a catalog-fingerprinted, provisional 30-exchange draw limit.
The last exchange's abilities resolve first, and natural outcomes take precedence.
This is separate from shop forcing and whole-episode truncation; original-client
count/order verification remains open in [the draw-rule ledger](LONG_BATTLE_RULE.md).

`events.py` is shared by battles and shop-side death effects. Battles use copies;
shop effects persist. Pets are tracked by identity, simultaneous attack damage
is applied before effects, and summons occur after their owner disappears.
Per-turn Rabbit/Ox uses are observable and carry from shop to battle. A merge
inherits the destination's spent uses; a genuine level increase refreshes the
new-level quota. Next turn also resets uses. Newly gained Melon counts as food
for Rabbit; replacing an identical held Melon does not. Purchased perk food
counts exactly once even when identical. Evidence and remaining limits are in
the [case ledger](REAL_GAME_CASES.md). Temporary and permanent stats have separate durations.

The implemented ordering follows the relevant subset of this
[community event-order guide](https://groundedsap.co.uk/Article.aspx?ID=11):
start abilities first; after-attack before friend-attacks; then Hurt, summoned,
faint, friend-ahead-faints and after-faint. Equal-priority events use current
attack then seeded random ties. Already queued start abilities can fire after a
lethal snipe; dead targets cannot be revived by buffs. This interpretation is
also described in this [community discussion](https://www.reddit.com/r/superautopets/comments/1758uy5/).
It is **not yet a captured target-client fixture**.

Before formal expanded training, obtain controlled client traces for:

1. Simultaneous faints, dead starters/targets, changing attack priorities,
   repeated Elephant/Dolphin hits and summon positioning/capacity.
2. Fractional rounding (Dodo/Badger floor, Crab ceiling; see
   [observed numeric audit](PERCENT_ROUNDING_AUDIT.md)), temporary-health damage absorption,
   merge temporary-stat/perk precedence and per-turn-counter merging.
3. Crowded/frozen shops, linked reward choices, unusual pre-levelled purchases
   and purchase positioning. The action API currently appends purchases; moving
   an existing team member is a separate action, an explicit sandbox abstraction.

Tests confirm the chosen specification and guard regressions. They **cannot by
themselves prove the specification matches the official client**. Development
Gym use still requires `allow_development=True`; training defaults remain v2.

## Reproduce local diagnostics (no learning updates)

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m sap_rl_lab.cli inspect --ruleset tier12-curriculum
.venv/bin/python -m sap_rl_lab.cli random --ruleset tier12-curriculum --episodes 100
.venv/bin/python -m sap_rl_lab.cli greedy --ruleset tier12-curriculum --episodes 100
```

On 2026-09-07, the suite passed **339 tests**, including 1,000 generated mixed
30-pet battles and 20 complete deterministic replay checks. An additional 100
random + 100 spend-gold episodes completed without simulator exceptions or
whole-game truncations. The unmodified spend-gold heuristic won 73/100 ten-win
episodes against the basic generated opponents; random won 0/100. These numbers
are diagnostics, **not RL results, parity evidence, or a held-out benchmark**.

Legacy protection: the delivery model reproduced all 300 frozen evaluation rows
exactly (`runs/tier12-curriculum-v4-legacy-regression-v1/`); the five Round-3
teaching replays reproduced all 325 recorded actions. Ruff and whitespace checks
passed. No formal expanded training, cluster job, website deployment or GitHub
push was performed in this implementation batch.

Subsequent overnight audit/training is tracked separately in
[the active goal](EXPANDED_GOAL.md), [real-game cases](REAL_GAME_CASES.md) and
[pilot results](EXPANDED_PILOTS.md). The 339-test/no-training paragraph above
describes the original implementation batch, not the current training status.

Additional current-rule mask audit: `scripts/audit_mask_branches.py` sampled200
reachable states from random/stats/summon/tempo play and executed every advertised
legal action twice on independent copies. All3,458branches matched states,
transition rewards and RNG states without changing the original. Nine action
kinds were covered, including four team-merge branches. Originating teams held
28species; this is not an all-species or official action-set completeness proof.
`runs/expanded-mask-branches-v1` retains source/helper hashes and the result digest.

```sh
.venv/bin/python scripts/audit_mask_branches.py \
  --output runs/expanded-mask-branches-new --states-per-family 50
```

Use a new output directory; this diagnostic never trains or opens held-out pools.
