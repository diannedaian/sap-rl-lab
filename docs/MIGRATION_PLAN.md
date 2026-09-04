# Migration plan from the earlier projects

The goal is not to cosmetically update imports. It is to retain ideas that made
the old projects useful while replacing choices that invalidate experiments.

## Keep

| Source idea | How it appears here |
|---|---|
| `sapai`: shop and battle are separate concerns | Pure engine methods and an isolated battle resolver. |
| `sapai`: explicit trigger phases | Named triggers and reusable effect specifications. |
| `sapai`: state can be copied and inspected | Dataclasses, dictionary snapshots, and battle traces. |
| `sapai`: mechanics need direct tests | Interaction and invariant tests precede RL tests. |
| `sapai-gym`: standard environment interface | Current Gymnasium API. |
| `sapai-gym`: invalid-action masking | Stable mask aligned with a fixed codec. |
| `sapai-gym`: replaceable opponents | Injected opponent-provider protocol. |
| `super-ml-pets`: Maskable PPO baseline | Optional, reproducible training entrypoint. |
| `super-ml-pets`: compare with scripts | Random and spend-gold controls. |

## Replace

| Earlier choice | Problem | Replacement |
|---|---|---|
| `wins / 10` returned after every action | Repeats old rewards and rewards longer shopping sequences. | Zero on shop actions; `-1/0/+1` at battle resolution. |
| Flat observation missing level, experience, previous result, and frozen state | Distinct game states look identical to the policy. | Complete structured observation with declared `float32` type. |
| `uint8` space containing fractional values | Observation violates its own contract. | Bounded `float32` arrays plus environment validation. |
| Old four-value Gym API | Incorrect timeout bootstrapping and stale tooling. | Gymnasium terminated/truncated API. |
| Ignored reset seed and global NumPy RNG | Results cannot be replayed or compared reliably. | Per-environment seeded RNG. |
| Catch every exception and retry an action | Can hide partial mutation and train on corrupted transitions. | Pre-validation, atomic mutation, fail-fast exceptions. |
| 213 actions dominated by full permutations | Wastes capacity and duplicates related choices. | Typed atomic actions and adjacent swaps. |
| Freeze absent | Removes a central shop decision. | Explicit toggle with fixed slot identity. |
| One raw-stat opponent | Encourages narrow exploitation. | Scripted diversity followed by checkpoint league. |
| Evaluation on the training setup | Measures memorization rather than generalization. | Held-out seeds and opponent pools with episode metrics. |
| Hard-coded UI coordinates and screenshot templates | Fragile and unrelated to simulator learning. | Deferred until the policy and simulator are independently valid. |
| Hard-coded, unversioned balance data | Live-game changes silently redefine the task. | Immutable, version-labeled catalogs and parity fixtures. |

## Migration sequence

1. **Correct substrate:** domain model, fixed slots, actions, RNG, traces, tests.
2. **Learning interface:** complete observations, incremental rewards, masks,
   proper termination, environment checker.
3. **Curriculum slice:** current tier-1 pets plus simple foods and three baselines.
4. **Opponent league:** generate round-indexed snapshots from scripts and policy
   checkpoints; keep a frozen evaluation league.
5. **Turtle expansion:** add one tier at a time, requiring parity fixtures for
   every ability and combination before enabling it in training.
6. **Representation upgrade:** shared entity encoder and candidate-action scorer
   only after the MLP baseline exposes a measurable limitation.
7. **Optional live-game bridge:** separate repository/module, never required by
   training or evaluation.

