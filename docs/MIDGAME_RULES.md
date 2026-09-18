# Midgame v5: rules expansion and training gate

Target game version: **0.46**. Simulator revision: **v5**, separate from the
frozen v4 experiment. Audit started 2026-09-07.

## Scope and current status

The agreed next curriculum enables normal **Tier 3 shops**, with all ten actual
Tier 4 pets as exact next-tier level-up rewards, and their food/perk/token and
event dependencies. This is 40 pets, not full Turtle parity. Normal Tier 4 shops
would require the next Tier 5 reward pool; they are not silently substituted with
Tier 4 rewards. Full 60-pet Turtle remains the eventual project endpoint.

**Experimental integration complete; bounded pilot is now permitted.** All 40
pets and the full Tier-3 food pool are integrated with the separate v5 runtime,
shop configuration, observable state and snapshot contract. Explicit development
opt-in is required. This is specification-tested code, not certified client parity.
The original staging guards were removed only after battle/shop integration;
the pilot launcher checks the current rules against the completed stress audit.

## Versioned pet specification

Stats and all three ability levels were checked by directly reading the dated
**16 March 2026** archive rows of the linked community profiles. These are
community-maintained records, not official-client fixtures. In particular, cached
search snippets and older tips can contradict the dated cards. The community
[0.46 history](https://superautopets.wiki.gg/wiki/Version_0.46) dates the release
17 March and confirms the Bison change; the archive uses the preceding date.

| Pet | Base stats | Executable specification / remaining work | Source |
| --- | --- | --- | --- |
| Skunk | 3/5 | Highest-health enemy loses 33/66/99%, rounded-up removal, minimum 1 HP; not damage, no Hurt or armor consumption. Component tested; rounding/armor still needs target-client fixture. | [Profile 70](https://groundedsap.co.uk/PetProfile.aspx?ID=70), [community Skunk](https://superautopets.fandom.com/wiki/Skunk) |
| Hippo | 4/6 | Knockout grants +3/+3, +6/+6, +9/+9; at most three per battle. Attribution, lethal-owner handling and trigger chains implemented and tested; target-client ordering fixture pending. | [Profile 38](https://groundedsap.co.uk/PetProfile.aspx?ID=38) |
| Bison | 4/4 | Another level-3 friend enables +2/+2, +4/+4, +6/+6 at end turn; one Bison per team. Component tested. Which Bison wins follows existing attack-priority queue; target-client fixture still needed. | [Profile 5](https://groundedsap.co.uk/PetProfile.aspx?ID=5), [0.46 history](https://superautopets.wiki.gg/wiki/Bison) |
| Blowfish | 3/6 | Hurt deals 3/6/9 to one random enemy, including lethal Hurt; direct Pill faint does not count. Component tested with inherited damage queue. | [Profile 7](https://groundedsap.co.uk/PetProfile.aspx?ID=7) |
| Turtle | 2/5 | Faint gives Melon to the next 1/2/3 eligible friends behind, skipping identical perks; new perk counts as food. Component tested including shop faint and Rabbit. | [Profile 80](https://groundedsap.co.uk/PetProfile.aspx?ID=80); inherited official 0.24/0.36 perk rules in v4 ledger |
| Squirrel | 3/5 | Start turn discounts currently stocked food by 1/2/3, floor zero; frozen food can accumulate discounts, fresh subsequent rolls do not inherit them. Components and full shop lifecycle tested. | [Profile 75](https://groundedsap.co.uk/PetProfile.aspx?ID=75) |
| Penguin | 2/3 | **Start**, not end, of turn: two distinct level-2+ friends gain +1/+1, +2/+2, +3/+3. Component tested with inherited non-maxed preference. | [Profile 56](https://groundedsap.co.uk/PetProfile.aspx?ID=56) |
| Deer | 2/2 | Faint summons a level-matched 5/3, 10/6, 15/9 Bus with Chili. Perk-on-summon and simultaneous splash selection implemented and tested. | [Profile 20](https://groundedsap.co.uk/PetProfile.aspx?ID=20), [Bus 85](https://groundedsap.co.uk/PetProfile.aspx?ID=85) |
| Whale | 3/7 | Start battle swallows the nearest friend ahead and releases it at Whale's level on faint. Saved displayed stats, fresh-pet semantics and nested interactions implemented and tested; target-client fixture pending. Do not use obsolete base-stat multiplication. | [Profile 81](https://groundedsap.co.uk/PetProfile.aspx?ID=81), dated 2024 wording change |
| Parrot | 4/2 | End turn copies nearest friend's ability at its own level until next turn; copied end-turn ability is not retriggered immediately. Copy chains follow event order. Serialization, snapshots and reset integrated and tested; target-client fixture pending. | [Profile 53](https://groundedsap.co.uk/PetProfile.aspx?ID=53), [community explanation](https://superautopets.fandom.com/wiki/Parrot) |

The public official Steam news API was also checked, with and without the feed
filter. Its returned text did not establish the current 0.46 changes; older
official announcements cannot certify absence of subsequent changes. Keep this
evidence limitation explicit rather than calling community data official.

## Completed integration and remaining parity limitations

- Implemented Hippo knockout attribution and lethal-owner handling. Knockouts
  resolve after Hurt/faint chains, so a dead Hippo is not healed back to life.
  See [community operation order](https://www.groundedsap.co.uk/Article.aspx?ID=11).
- Implemented Parrot end-turn copying, attack-ordered copy chains, own-level
  copied abilities, snapshots and pre-start-turn reset. It does not immediately
  retrigger a copied end-turn ability.
- Implemented Whale swallow/faint/release. Released pets retain displayed
  attack/health and take Whale's level, but are otherwise new: no perk, copied
  ability, spent quota or nested swallow memory. Original faint effects occur.
  [Current Whale explanation](https://superautopets.wiki.gg/wiki/Whale).
- Implemented Deer Bus Chili with both second-position targets selected before
  simultaneous damage. Summoned pets cannot retroactively become splash targets.
- Implemented the **version-0.46** Tier 3 food pool: Garlic, Salad Bowl, Cake.
  Garlic's reduced-damage floor is two (0.40); a one-damage hit stays one rather
  than being amplified. This floor needs a target-client edge-case fixture.
  [Garlic history](https://superautopets.wiki.gg/wiki/Garlic).
  Salad selects distinct non-maxed-preferred recipients and sends food events
  for those actual recipients; only one canonical purchase action is exposed.
  Cake also belongs to this target: its community history records the sell-value
  ability and return to Turtle in **0.44**, before 0.46. Its persistent sale
  bonus is represented in state, snapshots, observations, sales and merges.
  Merge retains the maximum accumulated bonus, not the sum; that merge edge
  remains provisional pending a target-client fixture.
  [Current food reference](https://groundedsap.co.uk/Foods.aspx),
  [Cake version history](https://superautopets.wiki.gg/wiki/Cake),
  [Chili behavior](https://superautopets.wiki.gg/wiki/Chili).
- Implemented the v5 contract: normal shop cap 3, reward cap 4, 139 slot actions,
  1,699 observation values. Additional perk bits, sale bonus, copied identity and
  non-clipping gold encoding are observable. Old v4 dimensions and digest remain
  unchanged. A tiny 32-decision PPO save/reload and learned-pool test passed;
  it checks plumbing, not learning quality.

## Training recipe to preserve after that gate

Small Maskable PPO; game reward minus 0.005 per action; initially 30-action forced
battle budget, rechecked for new content; 50/50 scripted and historical learned
opponents from the **new** environment. Validation/test generators stay separate.
Use bounded pilots then fixed budgets, multiple seeds and scripted controls.
The v5 checkpoint rule, frozen before pilot scores: highest macro ten-win rate,
then highest weakest-family ten-win rate, then lowest worst-family forcing,
then highest raw return; exact ties keep the earlier checkpoint. This replaces
v4's unassisted-win-first rule without changing any old result. No hard 1% gate.
Preserve early checkpoints and test only validation-selected models. Report
ten-win performance per opponent family, forcing/repetition and truncation,
without retroactively changing the old experiment's failed 1% gate.

Current verification: **712 tests passed**, including 88 new midgame tests and
the old suite; lint and whitespace checks passed. In `runs/midgame-stress-v1`,
6,000 synthetic battles reproduced twice exactly and 160 legally reached whole
episodes replayed exactly, with state/observation/mask invariants checked. Every
rollable species appeared in the synthetic audit. Synthetic stress is not a
reachable training pool and does not certify official-game behavior.

The fixed **1,048,576-decision** fresh PPO pilot completed and its selected model
exactly reproduced all 180 validation episodes. Mean wins rose from 0.05 to 4.06;
ten-win rate reached 5.6%, still below scripted baselines. See the
[pilot review and frozen formal protocol](MIDGAME_CONFIRMATION.md). The expanded
test suite including orchestration checks now has **718 passing tests**. No
held-out test was opened by the pilot. Formal mixed training and independent
evaluation use new pools and fresh seeds, not more tuning on pilot validation.

Final verification (2026-09-08): **727 tests passed**. Formal training and all
30,000 held-out episodes completed; 42 diagnostic replays and 400 archived-source
delivery validation rows reproduced exactly. Two seed pairs benefited from mixed
opponents; the third pair failed in both arms. This does not remove the client
parity limitations above. See the [final training report](MIDGAME_TRAINING_REPORT.md).
