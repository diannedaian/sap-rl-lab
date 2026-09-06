# Replay Lab: delivery and mechanics audit

[Open the owner-private viewer](https://sap-rl-replay-lab.dcao2028.chatgpt.site).

## Current delivery

The default collection is now **Round 5 · Confirmation**, with corrected Fish
rules, all six final-study policies, residual failures, and ten-win successes.
The original Round 3 collection remains available without changing its data.
See [the final result](ROUND5.md) and [viewer/model instructions](DELIVERY.md).
The new reward displays separate raw game reward from the swap-shaped objective
that the critic learned. The earlier inspection and mechanics audit below are
preserved as historical context, not the latest experiment summary.

## Initial Round 3 inspection

The viewer contains **12 cutoffs + 6 successes**, two failures and one success
per Round 3 model. Start with **Watch endings**, then compare probabilities,
raw reward, and critic values. **From start** includes shop and battle playback.
The successful comparison is illustrative, not a controlled counterfactual.

All 18 regenerated trajectories match their original evaluation rows. Export
uses the hash-verified archived engine, not today's changed rules. Values are
selected-checkpoint reconstructions, not the training optimizer's original buffer.

Initial evidence, **not a causal conclusion**:

- 11/12 selected cutoffs end with a swap; one ends with a merge.
- 11/12 have positive next-state critic values; one has a negative value.
- With gamma=1, the reconstructed cutoff reward is `-1 + V(next)`.
  It is positive in 10/12 examples. This does not mean the TD error or advantage
  is positive: those also depend on the preceding value estimate and trajectory.
- Next investigation: inspect the full final-turn sequence, then test one
  predeclared cutoff-handling change against an unchanged control. Do not infer
  the original training-time cause from these checkpoint diagnostics alone.

## Existing tools checked first

| Project | Useful existing capability | Why this viewer is separate |
| --- | --- | --- |
| [sapai](https://github.com/manny405/sapai) | Battle history and `graph_battle` diagrams | Different engine; no our-policy/critic/cutoff panel |
| [super-ml-pets](https://github.com/andreped/super-ml-pets) | Training plots and live-game screen automation | Not an offline shop-and-battle diagnostic viewer |
| [Super Auto Sim](https://www.mattkeeter.com/projects/super/) | First-turn simulation/analysis | Different scope; not complete RL episodes |

No external simulator or copyrighted game art was copied. The viewer plays
states observed from our archived engine; it does not claim to be a live client
or let you branch a replay into a new game.

## Fish fix and all-eight-pet level audit

The new default is `turtle-v0.46-tier1-rules-v2`. Historical
`turtle-v0.46-tier1` remains explicitly replayable and unchanged.
Fish used the **destination** level's ability entry after merging. The fix uses
the **departing** level for the level-up trigger only: 1→2 gives two friends
+1/+1; 2→3 gives two friends +2/+2. It excludes Fish itself and caps stats at 50.
This follows the ability entries in the developer's
[official 0.27 announcement](https://steamcommunity.com/games/1714040/announcements/detail/3698065064748065722).

| Pet | Level-1/2/3 and boundary coverage added |
| --- | --- |
| Fish | Both level-ups, non-level merges, few targets, no self-buff, stat cap |
| Otter | Buy targets 1/2/3; merge buy-trigger keeps current-level behavior |
| Ant | Faint buff +1/+1, +2/+2, +3/+3 in the actual battle resolver |
| Cricket | Faint summons 1/1, 2/2, 3/3 tokens |
| Horse | Summon buffs +1/+2/+3 attack; temporary shop buff expires after battle |
| Mosquito | Start-of-battle damage hits 1/2/3 targets |
| Pig | Sell bonus scales with current level |
| Pigeon | Sell stocks 1/2/3 free Bread Crumbs |

Fish is the only implemented pet with a level-up trigger, so the departing-level
fix does not shift buy, sell, faint, summon, or battle-start abilities down a level.
The official 0.27 source directly supports Fish, Ant and Otter's ability values.
The other rows test our declared sandbox contract, **not certified live-game
parity**. In particular, current Pigeon 3/2 stats remain community-sourced.
Otter's merge-trigger ordering and simultaneous-faint/dead-target interactions
still need controlled in-game parity fixtures before making a stronger claim.

## Run and reproduce

```sh
python -m http.server 8765 --bind 127.0.0.1 --directory viewer
node --test viewer/tests/model.test.mjs
node viewer/build.mjs
python -m pytest
```

To regenerate, use a **new, nonexistent** output directory and the original
RL environment with its archived checkpoints and leagues available:

```sh
python scripts/export_replay_viewer.py runs/round3-full-v1 runs/replay-export-v2
```

The initial viewer delivery did not train or change rewards. The subsequent
user-requested Round 4 and Round 5 experiments are documented separately.

Initial validation: 53 Python tests and 23 viewer/data tests passed. Browser checks covered
start/step/play/pause/end, shared speed, replay selection, pet details, all-action
probabilities, keyboard scrubbing, battle frames, and both cutoff and true-terminal
endings. Desktop panels and a 360-pixel stacked layout showed no horizontal
overflow. Browser error/warning logs were empty during the final checks.
