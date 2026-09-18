# Real-game case audit (in progress)

## Percentage components added on 2026-09-07

Direct browser-played recordings now supply five additional numeric observations:
Dodo's chained 20.5→20 / 12.5→12 transfers, its level-3 13.5→13 tooltip, Badger's
23-attack faint dealing 11 to an unprotected friend, and Crab's 6-health ×25%
granting +2, and a later level3 Crab gaining15 from19health (14.25→15), which
distinguishes ceiling from nearest rounding. Dodo/Badger now floor; Crab retains ceiling.
The final Crab component regression passes without any source change. All are component-only
fixtures, not full replays of unsupported pets or proof of an exact v0.46 build.
See [sources, timestamps and limitations](PERCENT_ROUNDING_AUDIT.md).

This ledger distinguishes **observed gameplay**, **community explanations**, and
our own **simulator specification tests**. None should be relabelled as another.
The target remains version 0.46; older footage only supports unchanged component
mechanics, not a full target-version battle fixture.

## Access and provenance, 2026-09-07

- Opened the developer's [official browser game](https://teamwood.itch.io/super-auto-pets).
  It loads but requires a new Terms of Service acceptance before play. No terms
  were accepted on the user's behalf. Target-client controlled tests remain open.
- Opened the public Honey–Spider Reddit case `u02irz`; the browser presented a
  CAPTCHA. It was not solved or bypassed. This clip has **not** been inspected.

## Cases inspected / being inspected

### Elephant → Camel → rear friend

[Haps' actual gameplay](https://www.youtube.com/watch?v=V-IVNFlpp0k), inspected in
the browser at 0:00, 0:06, 0:11 and 0:16. This is historical footage, not 0.46.
The starting team shows level-2 Elephant 8/10, level-2 Camel 4/8 and the next
friend (Crocodile) 8/4. The first battle visibly applies the Elephant/Camel chain
and increases the rear friend's health; at 0:16 its health reads 28. The clip
contains pets outside the enabled thirty and does not establish an exact full
battle fixture for our catalog. Captions obscure some intermediate stats.

Supported component observation: damage to Camel feeds the **next friend behind**,
not Elephant or the whole team. Our all-level Camel/Elephant tests cover that
component with supported pets. **Not yet verified by this footage:** repeated-hit
interleaving on lethal hits, target changes after death, or exact 0.46 parity.

A second frame review on 2026-09-07 confirms why this cannot be a current numeric
fixture: at12.014s the rear Crocodile shows24/20 (from8/4), and at15.005s it shows
32/28 while Camel reaches0health. The visible buff animation is+4/+4 for level2,
not our target version's+2/+4. At9.005s Camel already has6health while Crocodile
still shows8/4; this suggests grouped damage/animation in that historical client,
but does not establish current event timing. No runtime rules were changed from
these old observations. Screenshots are archived under
`runs/expanded-rule-video-audit-v3/sap-native-frames-BPrheO`.

### Post-nerf simultaneous lethal Camel / Elephant exchange

[Ninjin Plays' ranked Turtle recording](https://www.youtube.com/watch?v=_bvUkiIKkDY&t=510s)
is titled "Camel Still Good in 2024!"; the public listing says two years ago.
The exact build/date is not certified. It is newer post-nerf component evidence,
not a full v0.46 arena/economy fixture.

At510.610s, level2 Camel23/13 stands ahead of Bison8/15; opposing level2
Elephant20/3 stands ahead of Blowfish3/4. At510.911s they deal visible net damage
18/21, leaving Camel−5 and Elephant−18. The dead Elephant still fires two
one-damage hits behind, leaving Blowfish3/2. At512.519–513.005s, the dead Camel
gives Bison+2/+4 to10/19 before faint removal. Sub-second review distinguishes
this final buff from the earlier nonlethal buff at510.610s.

Two new regressions in `test_recorded_lethal_exchange.py` reproduce these
components and the Elephant-after-attack-before-Camel-Hurt order. Bison/Blowfish
are replaced with passive supported recipients, and observed **net** attack
damage is supplied directly: Garlic and the recipients' own unsupported abilities
are not being implemented or claimed as a full replay. Both regressions pass
without changing the frozen training source. Exact frames, timestamps and five
key screenshot hashes are retained in `camel_elephant_lethal_2024.json` and
`runs/expanded-rule-video-audit-v3/`.

This narrows the lethal queued-trigger evidence gap. It does not verify every
repeated-hit retargeting, death/summon interleaving or target-version edge case.

### Repeated-hit target death: remaining boundary

A bounded review of [Grounded SAP's recent Elephant recording](https://www.youtube.com/watch?v=cOU6ZeACVgk)
at120/240/280/300/320/340/360seconds did not isolate the desired case. It contains
advanced copied abilities, Mushroom and out-of-curriculum pets/perks; the inspected
setups do not cleanly establish where a remaining Elephant hit goes after its
first target dies. It is **not** added as an observed parity fixture, and no
runtime change was inferred from it.

`test_repeat_hit_faint_boundary.py` separately protects the existing sandbox
specification: a level3 Elephant kills a1- or2-health Sheep, directs remaining
hits at the next living friend, and resolves the Sheep's two Rams afterward.
Those Rams do not absorb the earlier hits. These are deliberately labelled
specification tests, not evidence that a target client was observed doing so.

### Cricket → same-level Zombie Cricket

[SuperAutoGaming's token run](https://www.youtube.com/watch?v=bnet27L5aFg), published
2022-11-09, **visually inspected at 1:20–1:25, including frame-stepping at 1:21,
1:22 and 1:22.5**. A level-2 3/4 Cricket is pilled; the level-1 1/3 Ox immediately
behind gains Melon and becomes 2/3 before the level-2 2/2 Zombie Cricket appears.
The player then moves Ant/Ox; that later arrangement is not pill output.
The transcript became available after the full page loaded and helped locate
the event, but these observations were checked on the actual video frames.
This supports both token-level preservation and Ox-before-summon ordering.
It is now a limited historical component fixture in
`tests/fixtures/real_game/cricket_ox_pill_2022.json`, exercised by
`tests/test_real_game_components.py`; not a complete 0.46 game fixture.
Independent descriptive evidence explicitly says both stats and level scale in
the [Cricket reference](https://superautopets.wiki.gg/wiki/Cricket). A player's
[firsthand achievement account](https://www.reddit.com/r/superautopets/comments/s9jvdh/)
also describes leveling the original then pilling it at the end.

### Recent Giraffe / Worm start-of-turn case

[micahwzowski's Turtle run](https://www.youtube.com/watch?v=tYDluMIOgjI&t=143s),
published **2026-08-23**, visually inspected at 2:23, 2:28, 2:43 and frame-stepped
through 2:44–2:45. Giraffe stands behind a 6/6 level-2 Ant on turn 4. Battle entry
still shows 6/6, so it does not buff at end of turn. At turn 5, Worm 2/5 triggers
first and stocks a visibly 2-gold Apple; Giraffe 1/2 then buffs Ant to 7/7.
Both displayed tooltips explicitly say **Start of turn**. This is a recent
observed component fixture, not merely an ability table or transcript.

The client build number is not visible, and the normal Tier-3 shop contains
out-of-curriculum foods. The fixture asserts only the supported team, trigger
order and generated Apple, not complete shop or battle parity. Stored in
`tests/fixtures/real_game/giraffe_worm_start_2026.json`.

### Recent Ant upgrade / linked paid choices

The same recent video also supplies a **linked upgrade reward** component:
2:03–2:06 shows two 3-gold shop Ant merges, front Ant 3/3 → 4/4 → level-2 5/5,
then linked Tier-3 Giraffe/Dolphin offers on turn 4. At 2:18 the Giraffe tooltip
shows price 3; dragging it previews a red X over Dolphin. By 2:23 purchasing it
uses all three gold and removes both offers while Pig remains. Regression fixture
`ant_upgrade_choices_2026.json` fixes the observed random offers, not client RNG,
and excludes UI compaction/insertion layout from its assertions.

## Corrections found during audit (tested, not client-certified)

1. **Cricket and Sheep tokens lost their level.** Their v4 summon data now retains
   owner level, affecting shop sale value and future merges, not just appearance.
   The [Sheep reference](https://superautopets.fandom.com/wiki/Sheep) describes
   level-2 Rams from level-2 Sheep; its obsolete XP-merging advice was NOT adopted.
2. **Honey's Bee was behind the owner's summon.** The Bee is created after that
   summon but inserted ahead of it, enabling Spider → Ox adjacency and Spider →
   Dog summon buffs. Evidence: [Spider reference](https://superautopets.wiki.gg/wiki/Spider)
   and a separate [historical guide](https://steamcommunity.com/sharedfiles/filedetails/?id=2725758944).
   Regression tests cover both Ox and Dog; direct current-client fixture pending.
3. **Expanded opponent capture missed end-turn skills/forced battles.** Capture
   now occurs at the actual pre-battle boundary after Snail, not merely when a
   policy selects END_TURN. This is an engineering correction, not a game source.
4. **Gold observations hid Swan-generated funds above 20.** Expanded observations
   now use a conservative curriculum-specific bound; 20, 21 and 25 are distinct.
5. **Random buffs did not prefer non-maxed teammates.** The developer's
   [official 0.29 announcement](https://store.steampowered.com/news/posts/?enddate=1696248072&feed=steam_community_announcements)
   explicitly introduced this priority. Ant/Fish/Otter/Beaver now share an
   opt-in, catalog-fingerprinted selector, tested against full-cap teammates,
   distinct targets, empty teams and all-maxed fallback. The conservative
   interpretation is displayed 50/50, not a per-stat optimization (e.g. preferring
   low health for Otter). The announcement does not specify that finer case;
   a target-client single-stat/temporary-cap check remains a limitation.

This discovery stopped the two active v4 pilot jobs before rule edits. They are
not silently resumed with different dynamics. New expanded replays now include
the catalog fingerprint and reject mismatched rules before replaying.

No bug-free or official-parity claim is made. Unresolved case work remains part
of the active completion goal while explicitly experimental pilot training runs.

### Trigger-counter and perk-food follow-up

Two further shared-rule omissions stopped v5 before changing its source:

- **Merge counters:** Rabbit/Ox inherited the maximum spent count, and upgrading
  did not refresh charges. The implemented correction retains the destination's
  spent count unless the result exceeds both input levels, which resets it.
  Evidence is explicitly community/firsthand, not official code:
  [Grounded's Alpaca tips](https://www.groundedsap.co.uk/PetProfile.aspx?ID=194)
  (2023-03-01) document both level-up refresh and dragging an exhausted high-level
  pet onto a fresh lower-level target. [Hamster tips](https://groundedsap.co.uk/PetProfile.aspx?ID=110)
  independently describe level-up reset. The earlier
  [firsthand trigger guide](https://www.reddit.com/r/superautopets/comments/113ogmv/)
  gives destination-used-count examples. New tests cover every XP pair in both
  directions, genuine upgrades, quota exhaustion, and legacy opt-out semantics.
  A target-client Rabbit/Ox merge clip is still not captured. Do not generalize
  this into unlimited respawn refresh: official 0.37 changed Mushroom/Pteranodon
  to remember triggers, outside this curriculum.
- **Perks count as food:** the developer's
  [official 0.36 announcement](https://steamcommunity.com/games/1714040/announcements/detail/4617966511040089938)
  adds food-eaten notifications for gaining perks. Ox gaining Melon now notifies
  Rabbit in shops and battles. Identical ability-given perks are skipped under
  [official 0.24](https://store.steampowered.com/news/posts/?enddate=1673431427&feed=steam_community_announcements);
  purchasing identical perk food still counts as eating once, without a double
  notification. Tests cover replacing a different perk, exhausted/dead Rabbits,
  side isolation, repeated consumed/regained Melon, and battle-copy isolation.
  This does not certify whether an inherited merge perk produces a food event;
  merge inheritance is currently not treated as a separate food gain.

Both changes are opt-in through hashed v4 ability data; old model contracts
reject the new catalog. v5 partial runs are preserved with explicit stop reasons,
not resumed or compared as completed equal-budget experiments.

### Temporary merge cross-check (no rule change)

The [community Experience reference](https://superautopets.wiki.gg/wiki/Experience)
describes temporary stats being applied after the permanent merge, keeping the
higher temporary amount when both pets have it. This supports the implemented
separate maxima, rather than merging displayed totals and accidentally converting
temporary gains into permanent ones. Added both-direction tests with deliberately
different permanent/temporary maxima and next-turn expiry. The page's old summed
XP paragraph conflicts with official 0.27 and was **not adopted**. This is a
descriptive cross-check, not a new observed current-client fixture.

### Long-battle failure and provisional correction

An independent 1/50 Swan mirror case exposed a 200-attack safety exception and
stopped confirmation v1. A versioned draw mechanism now uses a provisional
30-exchange threshold, with dedicated edge and accounting tests. Community
reports disagree on the exact cutoff; there is no new observed-client fixture
for it. See [sources, counterevidence and remaining verification](LONG_BATTLE_RULE.md).
The subsequent v7 pilot is explicitly experimental, not final parity sign-off.

Follow-up: an isolated headless browser can inspect public YouTube gameplay
without unlocking the desktop or using the user's profile. Reddit still returns
a network-security block and was not bypassed. Skoottie's
[2022-03-29 test-server recording](https://www.youtube.com/watch?v=Z2xB3Sgo6Uk&t=563s)
was reviewed through the full relevant battle: **30 front-pet exchanges**, last
exchange's summons/buff resolved, then Draw with survivors at 10:13. The count
excludes two menu-text detector false positives and includes a separately checked
attack initially merged with the menu. Raw screenshots and metric traces are
retained locally; no video was downloaded. See the detailed method and limitations
in the linked ledger and `long_battle_tapir_2022.json`.

This supplies a fourth observed **component**, not full thirty-pet client parity.
It does not settle exact v0.46 provenance or natural outcome priority on the
last exchange. A limited count/queue-order regression on supported pets passes;
it is explicitly not a simulation of the recorded Tapir teams.
