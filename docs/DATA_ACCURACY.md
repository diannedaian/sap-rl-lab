# Turtle Pack data accuracy policy

The target is the 60-pet Turtle Pack, but game data changes over time. “Turtle
Pack compatible” must always name a game version.

## Current catalog

- Catalog: `turtle-v0.46-tier1`
- Checked: 2026-09-04
- Pet scope: eight currently rollable tier-1 pets, plus battle tokens
- Food scope: a deliberately small learning curriculum, not current-pack parity

The current Pigeon 3/2 stats and its Bread Crumbs mechanic were checked against
the community wiki. Other tier-1 records were checked against the current pets
listing. Community pages can be wrong; these records remain provisional until
validated through controlled in-game examples or a maintained authoritative
data export.

## Rules for adding content

1. Record the source URL, source revision/date, and target game version.
2. Add a direct unit test for each level of every new ability.
3. Add interaction tests when trigger ordering can change an outcome.
4. Add at least one recorded parity fixture from the target game.
5. Never silently change an existing catalog. Create a new catalog version.
6. Do not enable a pet in training if its engine primitive is unimplemented.

The wiki is a useful factual reference, but this repository does not copy its
articles or art. Game icons and screenshots are intentionally excluded.

