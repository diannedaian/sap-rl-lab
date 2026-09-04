# Roadmap to the 60-pet Turtle Pack

## Foundation — implemented

- Pure-Python engine and fixed slot state
- Versioned and validated catalog
- Seeded randomness
- Typed fixed action codec and masks
- Buy, merge, feed, sell, roll, freeze, reorder, end turn
- Incremental battle rewards and correct truncation distinction
- Faint, summon, random buff/damage, buy/sell, and summon triggers
- Human-readable battle traces
- Versioned JSON episode recording and deterministic replay verification
- Random and spend-gold baselines
- Serializable round-indexed opponent snapshots and weighted provider mixtures
- Gymnasium and Maskable PPO entrypoints
- Core mechanics and invariant tests

## Next mechanics milestone

- Verify the exact current Turtle shop distribution and food roster
- Add per-turn and per-level shop-slot rules
- Complete experience/level-up shop rewards
- Add damage modifiers and the current Turtle perks
- Implement hurt, knockout, before-attack, after-attack, start/end-turn triggers
- Add deterministic priority ordering parity fixtures
- Populate separate training and held-out round-indexed snapshot pools
- Calibrate opponents so random, scripted, and PPO policies are meaningfully
  separated

## Expansion sequence

1. Exact tier 1 and foods available on turns 1–2
2. Tier 2 plus food/perk interactions
3. Tier 3, including hurt chains and positional support
4. Tier 4, including shop scaling and targeted abilities
5. Tier 5, including stronger summon and economy engines
6. Tier 6, copying/repetition and late-game abilities
7. Full 60-pet parity suite

Each phase has a gate: no new content enters RL training until its mechanics,
observations, action masks, and replay traces pass tests.

## Training milestones

- MLP Maskable PPO beats random on held-out seeds
- Five-seed result against frozen scripted opponents
- Checkpoint league with round-indexed snapshots
- Entity encoder comparison
- Full Turtle curriculum and ablation report

## Deferred intentionally

- Screen-reading and mouse control of the commercial game
- Art assets
- Research-scale population-based training
- Cloud/GPU orchestration

Those do not help validate the sandbox today and can remain separate from the
core repository.
