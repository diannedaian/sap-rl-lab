# Architecture

## Dependency direction

```text
versioned catalog
      ↓
domain objects ← action codec
      ↓
pure game engine ← opponent provider
      ↓
Gymnasium adapter
      ↓
Maskable PPO / future algorithms
```

The arrows only point downward. The simulator never imports Gymnasium, NumPy,
PyTorch, or Stable Baselines. Consequently, a library upgrade cannot silently
change battle mechanics.

## Modules

| Module | Responsibility |
|---|---|
| `catalog.py` | Load and validate versioned pet, food, trigger, and effect data. |
| `domain.py` | Serializable state with no behavior hidden in neural-network code. |
| `actions.py` | Stable action IDs and human-readable descriptions. |
| `engine.py` | Validate and apply shop actions; resolve battles; record traces. |
| `env.py` | Convert complete state into bounded `float32` observations. |
| `baselines.py` | Random and scripted policies used as executable controls. |
| `opponents.py` | Serializable round-indexed snapshot leagues and provider mixtures. |
| `replay.py` | Versioned JSON episode recording and deterministic drift checks. |
| `training.py` | Train and evaluate Maskable PPO without owning game rules. |

## Engine invariants

1. An invalid action is rejected before mutation. At the Gym boundary it
   becomes a documented `-1` truncation so arbitrary policies still receive a
   valid transition.
2. Randomness belongs to one seeded engine instance.
3. Shop slots retain identity after a purchase; an empty slot is explicit.
4. Shopping actions return zero task reward.
5. A battle returns only the outcome caused by that transition.
6. Time limits are truncations, not natural terminal states.
7. Training observations include every persistent field that changes legal
   actions or future transitions.
8. Battle calculations happen on copies, so battle-only changes do not leak
   into the persistent team.
9. A saved episode can be rerun action-for-action and must reproduce every
   transition, battle trace, and final state.

## Scaling from 8 to 60 pets

Pet count should increase through two mechanisms:

- **Data entries** describe identity, tier, base stats, and parameters by level.
- **Effect primitives** implement reusable mechanics such as damage, buff,
  summon, stock, copy, transform, apply perk, and target selection.

Adding the tenth pet that says “buff two random friends” should add catalog
data, not a tenth bespoke class. A new engine primitive is justified only when
the mechanic changes the state in a genuinely new way.

The catalog is immutable after loading. Every training run records its catalog
ID, game version, seed, environment configuration, and model configuration.
This prevents a balance update from making two experiments incomparable.

## Actions

The older environment allocated almost 150 action IDs to complete team
permutations. Here, reordering uses four adjacent swaps. Any permutation remains
reachable, but the policy learns one reusable relation: whether two neighbors
should exchange positions.

The fixed action vocabulary is:

```text
end turn                         1
roll                             1
buy pet from shop                S
feed shop food to team pet       S × T
merge shop pet into team pet     S × T
sell team pet                    T
swap adjacent team pets          T - 1
toggle frozen shop item          S
```

For `S = T = 5`, this is 71 actions. A later parametric-action policy can encode
and score only the legal `(kind, source, target)` candidates while preserving
the same domain-level `Action` object.

## Observations

The Gymnasium adapter emits a dictionary with:

- `global`: turn, gold, lives, wins, action budget, previous outcome, and tier;
- `team`: pet identity, stats, level/experience, temporary stats, perk, position;
- `shop`: item identity/type, price, frozen flag, and tier.

The explicit representation is intentionally easy to inspect. A future entity
encoder can share weights between slots without changing game state or tests.

## Opponents

An `OpponentProvider` is injected into the engine. The initial provider is a
transparent seeded generator. `SnapshotLeague` provides a serializable,
round-indexed pool today, and `OpponentMixture` combines it with scripts or
other leagues:

```text
scripted bots + recent policy + historical checkpoints + held-out league
```

Arena matchmaking is naturally snapshot-based, so this achieves self-play
without forcing the shop phase into a simultaneous multi-agent API.
