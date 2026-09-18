# Turtle Pack data accuracy policy

The target is the 60-pet Turtle Pack, but game data changes over time. “Turtle
Pack compatible” must always name a game version.

## Final delivered scope — 2026-09-18

The delivered model uses the separate full-pack runtime: all 60 pets, normal
Tier 1–6 shops, 18 ordinary foods, and seven battle tokens. See
[full-pack rules](FULLPACK_RULES.md) and [model limitations](MODEL_CARD.md).
This is a version-pinned v0.46-inspired sandbox, not certified official-client
parity. The older catalogs below remain available for historical experiments;
they do not describe the final model's scope.

## Historical default catalog

- Catalog: `turtle-v0.46-tier1-rules-v2`
- Checked: 2026-09-05
- Pet scope: eight currently rollable tier-1 pets, plus battle tokens
- Food scope: a deliberately small learning curriculum, not current-pack parity

The current Pigeon 3/2 stats and its Bread Crumbs mechanic were checked against
the community wiki. Other tier-1 records were checked against the current pets
listing. Community pages can be wrong; these records remain provisional until
validated through controlled in-game examples or a maintained authoritative
data export.

Rules v2 corrects Fish's level-up indexing using Team Wood Games' official
ability announcement. The old catalog remains frozen for historical replays.
See the [eight-pet level audit](REPLAY_LAB.md) for source-backed rules,
regression coverage, and the remaining parity limitations. This version suffix
identifies a sandbox rules revision, not a new official game release.

## Rules for adding content

1. Record the source URL, source revision/date, and target game version.
2. Add a direct unit test for each level of every new ability.
3. Add interaction tests when trigger ordering can change an outcome.
4. Add at least one recorded parity fixture from the target game.
5. Never silently change an existing catalog. Create a new catalog version.
6. Do not enable a pet in training if its engine primitive is unimplemented.

The wiki is a useful factual reference, but this repository does not copy its
articles or art. Game icons and screenshots are intentionally excluded.

## Tier 1–2 work in progress

The eight-pet milestone omits Duck and Beaver: complete Turtle Tier 1 has ten
ordinary pets, and Tier 2 adds ten more. See [the coverage and implementation
record](TIER12.md). The explicitly gated v4 curriculum implements all thirty
Tier 1–3 pets, with normal shops capped at Tier 2 and Tier 3 reserved for rewards
and summons. The default training catalog is unchanged. Otter's historical
1/4 stats are corrected only in v4, not retroactively in released catalogs.
Most new tests are specification tests. Four limited observed-video component
fixtures now cover Cricket/Ox, Giraffe/Worm, linked Ant upgrade offers and a
historical thirty-exchange survivor draw; they
are not complete official-game episode fixtures. See [case provenance](REAL_GAME_CASES.md).
Unreleased v4 draft revisions are distinguished by the catalog SHA-256 plus
their archived source, not the development ID alone. Models and new expanded
replays reject a changed catalog fingerprint; published eight-pet catalogs are
not edited. Earlier pilot scores refer only to their archived draft revision.
The
[current source/status ledger](TIER12_RULES.md) separates implemented behavior
from provisional details and the remaining client-fixture gates.
