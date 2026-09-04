# RL training guide

This file is both the training plan and the explanation of why each choice was
made.

## 1. What the agent is learning

At time step `t`, the agent observes `s_t`, selects legal action `a_t`, and the
engine produces `s_(t+1)` and reward `r_t`.

Most shop actions have immediate reward zero. Their value comes from increasing
the probability of later battle wins. The critic learns an estimate:

```text
V(s) ≈ expected future battle reward from state s
```

PPO compares the return actually observed with that estimate. If an action led
to a better-than-expected future, PPO increases its probability; if worse, it
decreases it. The clipped update prevents one noisy batch from changing the
policy too aggressively.

## 2. Why Maskable PPO first

- The action space is discrete and state-dependent.
- Many actions are structurally impossible in a given state.
- PPO collects fresh experience, which is conceptually easy to inspect.
- Multiple independent environments reduce wall-clock time and correlation.
- A well-supported baseline lets us study the environment instead of debugging
  a novel algorithm.

Action masks encode rules, not strategy. “Cannot buy with two gold” belongs in
the mask. “Buying this Ant is probably bad” must be learned.

## 3. Reward design

The baseline reward is:

| Transition | Reward |
|---|---:|
| Shop action | 0 |
| Battle win | +1 |
| Draw | 0 |
| Battle loss | -1 |
| Exceed shop action limit | -1 and truncate |

This is intentionally close to the actual objective. We do not reward raw team
stats: doing so teaches the agent that stats are the goal and can suppress
summon, economy, perk, and positioning strategies.

`gamma=1.0` is the initial setting because shopping sequences have variable
length. Discounting each micro-action would otherwise favor ending a turn early
even when another purchase is beneficial. We can later compare `0.99`, `0.995`,
and `1.0` as a controlled experiment.

## 4. Training curriculum

### Stage A — prove the environment

Run thousands of random and scripted episodes. Verify:

- no illegal state is produced;
- seeded episodes replay exactly;
- battle outcome distributions are plausible;
- random < greedy against every basic opponent;
- no policy can obtain reward without ending a turn.

### Stage B — first PPO signal

Use the tier-1 catalog and a mixture of weak seeded opponents. Train roughly
100,000 steps as a smoke test. Success means the policy beats random and its
learning curve rises; it does not mean the game is solved.

### Stage C — robust baseline

Train five seeds for 1–5 million steps each. Evaluate every checkpoint against
held-out random, greedy, summon-heavy, and stat-heavy opponent snapshots.

### Stage D — league training

Every fixed number of updates:

1. Roll out the current policy and save teams by round.
2. Add them to a bounded historical snapshot pool.
3. Sample training opponents from scripts, recent policy snapshots, and older
   checkpoints.
4. Never add evaluation opponents to the training pool.

Historical opponents prevent the latest policy from forgetting how to beat an
older strategy.

### Stage E — expand mechanics

Enable another tier only after the current tier passes mechanics and parity
tests. This is curriculum learning for both the agent and the humans developing
the simulator.

## 5. Evaluation protocol

Each reported experiment should contain:

- catalog ID and git commit;
- algorithm configuration;
- training seeds and number of environment steps;
- opponent mixture used for training;
- held-out evaluation set;
- mean final wins and battle win/draw/loss rates;
- ten-win success rate;
- confidence interval across independent seeds;
- at least five human-readable episode replays.

Training reward is a debugging plot, not the final result.

## 6. Experiments that teach RL

Run these one change at a time:

1. **Reward bug:** incremental battle reward versus repeated cumulative wins.
2. **Masking:** legal-action masking versus penalties for invalid actions.
3. **Observability:** complete state versus intentionally removing experience.
4. **Curriculum:** static greedy opponent versus a mixed opponent pool.
5. **Generalization:** train on fixed seeds, test on unseen seeds.
6. **Representation:** flat MLP versus a shared per-item encoder.
7. **Self-play:** latest-only opponent versus a historical league.

Each experiment isolates a core RL lesson: reward hacking, exploration,
Markov state, distribution shift, overfitting, inductive bias, or forgetting.

## 7. When to use a better model

For eight pets, a standard MLP is desirable because it is easy to debug. At 60
pets, use a shared entity encoder:

```text
pet identity embedding + stats + level + perk + position
                         ↓
              shared slot encoder
                         ↓
         attention over team and shop entities
                         ↓
         value head + legal-action scoring head
```

This lets knowledge about an Otter transfer across shop positions. It is an
architectural improvement, not a reason to skip environment correctness.

## 8. Compute expectations

The pure engine is CPU work. Initial PPO runs should work on a laptop; a GPU may
not help much while simulation is the bottleneck. Larger leagues and attention
models may justify a GPU later. Profile environment steps per second before
renting hardware.

