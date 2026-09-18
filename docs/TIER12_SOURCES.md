# Tier 1–2 implementation ledger

Historical **v3** milestone. The 2026-09-07 **v4** implementation completes the
30-pet Tier-2-capped curriculum; see [current rules and evidence](TIER12_RULES.md).
The missing-content items below describe v3, not the latest v4 state.

Target: **0.46**. Checked: **2026-09-06**. Development catalog:
`turtle-v0.46-tier12-dev-v3`. The `v3` suffix is our simulator revision, not an
official game version. No new training has been performed on this catalog.

## Evidence and status

| Item | Implemented specification | Evidence | Remaining verification |
| --- | --- | --- | --- |
| Duck | Tier 1, 2/2; sell adds 1/2/3 health to current shop pets at levels 1/2/3. Frozen buffs persist; fresh rolls do not inherit them. | [Historical profile](https://groundedsap.co.uk/PetProfile.aspx?ID=26), March 16, 2026 row; June 23, 2025 change from 2/3 to 2/2 | Community-maintained data; target-client fixture still required |
| Beaver | Tier 1, 3/2; sell buffs up to two distinct remaining friends by 1/2/3 attack. Not health. | [Official 0.27 announcement](https://steamcommunity.com/games/1714040/announcements/detail/3698065064748065722); [historical profile](https://groundedsap.co.uk/PetProfile.aspx?ID=3), March 16, 2026 row | Target-client fixture still required |
| Merge experience | Internal copy count becomes `min(6, max(a, b) + 1)`; thresholds remain 3 and 6. Team-to-team merge costs no gold and does not trigger a buy. | Official 0.27 announcement | Perk precedence, temporary-stat combination and unusual pre-levelled purchases remain provisional pending client fixtures |
| Level-up choices | Two linked, paid shop offers from exactly the next shop tier, capped at tier 6; purchasing or shop-merging one removes its partner. Combining two level-2 pets grants no shop reward. | Official 0.27 announcement for double rewards and exception; [community basics](https://superautopets.wiki.gg/wiki/The_Basics) for linkage | Duplicate-choice sampling, multiple rewards, freezing/rolling and ordering relative to buy/level-up abilities need client fixtures |
| Shop capacity | Normal pet slots 3 → 4 → 5 at turns 1/5/9; food slots 1 → 2 at turn 5; two extra stock spaces | Official 0.27 announcement for extra spaces and food timing; [slot calculation discussion](https://github.com/bencoveney/super-auto-pets-db/issues/31) | Exact crowded-shop layout and frozen-item ordering need client fixtures |
| Stocking | Pet/food opposite edges; empty spaces used before unfrozen replacements; frozen items protected. Stock a batch without overwriting its earlier members. | Official 0.27 announcement | Which unfrozen items the client replaces first needs a crowded-shop fixture |

Official announcement text was read through the public [Steam news API](https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/?appid=1714040&count=100&maxlength=0&feeds=steam_community_announcements).
Older official notes establish changes, not proof that later versions never changed.
Community evidence is marked as such; these tests do not certify live-game parity.

## Implemented and protected

- The old two catalog files, default rules, 71-action vocabulary and old observation
  shape are unchanged. New optional serialization fields are omitted when absent.
- The development shop uses a fixed nine-row observation (unused rows are empty),
  persistent pet stats and observable links between reward choices. There are 139
  action IDs, including the new directed team-merge actions.
- `AutoBattler(load_catalog_by_id("turtle-v0.46-tier12-dev-v3"))` opts into the
  development contract. `GameConfig.turtle_development()` supplies its configuration.
- The Gym adapter rejects this unfinished catalog unless `allow_development=True`
  is explicitly passed for diagnostics. The training entrypoint still uses the
  frozen eight-pet catalog; no expanded training command is advertised.
- Only Tier 1 is currently marked implemented. Reaching a Tier-2 roll or reward
  raises `ContentNotReady`, restores the pre-action game/RNG state, and stops the
  diagnostic. Missing content is not an RL penalty or a low-tier fallback.
- `tests/test_tier12_development.py` covers all new pet levels, target uniqueness,
  frozen and capped buffs, purchase/merge inheritance, team-merge identity, linked
  choices, capacity boundaries, atomic failures, observation links and replays.
  Higher-tier reward tests use explicitly named synthetic pets, not fake SAP pets.

Verification on 2026-09-06: **149 Python tests passed** (55 new development tests;
14 pre-existing dependency deprecation warnings). Ruff and whitespace checks
passed. The frozen delivery model reproduced 300 archived evaluation rows exactly
(`runs/tier12-dev-v3-legacy-regression-v1/`); five Round-3 teaching replays reproduced
all 325 recorded actions. These are regression checks, not new learning results.

## Next work before training

1. Start/end-shop event lifecycle for Swan, Worm and Snail. Snail's old health-buff
   description must not replace the later positional attack-buff ability.
2. Remaining Tier-2 battle events and foods; shop-side faint resolution for Pill.
3. Tier-3 reward and Spider summon dependencies. A list of names without abilities
   is not an implemented pool. Later normal shop tiers also need an explicit
   curriculum boundary before a Tier-1/2 training environment can run whole games.
4. Remaining core parity gaps: turn-3 life recovery, shop-to-shop stacking,
   purchase positioning, battle event priorities, crowded-shop behavior and real
   client fixtures. Current development rules are intentionally not called a
   complete Tier-1/2 simulator.

The next experiment starts only after the chosen content boundary and these
relevant dependencies are validated. Existing eight-pet scores remain historical
results, not evidence about the expanded environment.
