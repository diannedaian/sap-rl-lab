# Five shop decisions to work through

Policy: `mixed-seed211`, selected using validation only. These five fixed development seeds are not benchmark results.

Try choosing before opening each answer. Slot numbers start at zero, matching the action traces. Team slot 0 fights first. `merge:shop,team` buys a copy from the shop into an existing pet; `buy_food:shop,team` feeds that team slot.

An action probability is the policy's preference, not the chance of winning. The critic predicts future net reward, not win probability. Both may be wrong.
The family label below describes the scenario's opponent pool; it is not an input given to the policy, and the next opponent team is not revealed.

Rules reminder: buying a pet appends it to the back; rolling costs one gold; swapping and freezing cost no gold but consume decisions. A turn allows 30 decisions. Shop moves earn zero immediate reward, battles earn +1/0/-1 for win/draw/loss, and reaching the shop limit cuts the episode short with -1. Temporary buffs add to the displayed base stats for the next battle only.

These examples use the frozen Round 3 simulator, not verified live-game parity. In particular, Fish's level-up indexing needs a separate correction and parity test; see [the Round 3 report](../ROUND3.md).

## Situation 1

Turn 1 · gold 4 · wins 0 · lives 5 · shop decisions used 2/30 · opponent family greedy

| Team slot | Pet | Attack / health | XP | Perk | Temporary buff |
|---|---|---|---|---|---|
| 0 | fish | 2 / 4 | 1 | — | +0 / +0 |
| 1 | otter | 1 / 3 | 1 | — | +0 / +0 |

| Shop slot | Item | Cost | Frozen |
|---|---|---|---|
| 2 | pigeon | 3 | False |
| 3 | pig | 3 | False |
| 4 | apple | 3 | False |

Before revealing: What would you do? What later battle benefit do you expect? Which alternative would you test?

<details>
<summary>Policy answer and evidence</summary>

| Action | Policy probability |
|---|---:|
| `roll` | 57.0% |
| `buy_pet:2` | 33.3% |
| `swap_adjacent:0,1` | 3.6% |
| `buy_pet:3` | 2.9% |
| `end_turn` | 1.8% |

Critic value: **7.70**. The recorded continuation earned **8.0**. Their difference is **+0.30**.

That difference illustrates a full-return advantage estimate. Actual PPO training uses GAE (lambda 0.95), rollout bootstrapping, and a stochastic policy—not exactly this deterministic full-episode calculation.

For each action below, force that one move and then follow the policy over 32 sampled futures. These are noisy one-step-deviation estimates, not optimal action values. Matched initial random seeds can diverge after different actions consume randomness.

| Forced first action | Mean future reward | Standard error |
|---|---:|---:|
| `roll` | 9.06 | 0.19 |
| `buy_pet:2` | 7.66 | 0.63 |
| `buy_pet:3` | 8.44 | 0.39 |

Does the most probable action also look best in these rollouts? What would additional samples clarify?

[Complete episode replay](episode-1.json)

</details>

## Situation 2

Turn 1 · gold 0 · wins 0 · lives 5 · shop decisions used 5/30 · opponent family stats

| Team slot | Pet | Attack / health | XP | Perk | Temporary buff |
|---|---|---|---|---|---|
| 0 | fish | 2 / 4 | 1 | — | +0 / +0 |
| 1 | cricket | 1 / 3 | 1 | — | +0 / +0 |
| 2 | otter | 1 / 3 | 1 | — | +0 / +0 |

| Shop slot | Item | Cost | Frozen |
|---|---|---|---|
| 0 | pigeon | 3 | False |
| 1 | horse | 3 | False |
| 4 | honey | 3 | False |

Before revealing: What would you do? What later battle benefit do you expect? Which alternative would you test?

<details>
<summary>Policy answer and evidence</summary>

| Action | Policy probability |
|---|---:|
| `end_turn` | 99.7% |
| `swap_adjacent:0,1` | 0.2% |
| `swap_adjacent:1,2` | 0.0% |
| `toggle_freeze:4` | 0.0% |
| `toggle_freeze:0` | 0.0% |

Critic value: **7.57**. The recorded continuation earned **6.0**. Their difference is **-1.57**.

That difference illustrates a full-return advantage estimate. Actual PPO training uses GAE (lambda 0.95), rollout bootstrapping, and a stochastic policy—not exactly this deterministic full-episode calculation.

For each action below, force that one move and then follow the policy over 32 sampled futures. These are noisy one-step-deviation estimates, not optimal action values. Matched initial random seeds can diverge after different actions consume randomness.

| Forced first action | Mean future reward | Standard error |
|---|---:|---:|
| `end_turn` | 7.66 | 0.50 |
| `swap_adjacent:0,1` | 7.66 | 0.50 |

Does the most probable action also look best in these rollouts? What would additional samples clarify?

[Complete episode replay](episode-2.json)

</details>

## Situation 3

Turn 2 · gold 0 · wins 1 · lives 5 · shop decisions used 4/30 · opponent family summon

| Team slot | Pet | Attack / health | XP | Perk | Temporary buff |
|---|---|---|---|---|---|
| 0 | ant | 2 / 3 | 1 | — | +0 / +0 |
| 1 | mosquito | 3 / 3 | 2 | — | +0 / +0 |
| 2 | fish | 3 / 4 | 2 | — | +0 / +0 |
| 3 | otter | 1 / 3 | 1 | — | +0 / +0 |

| Shop slot | Item | Cost | Frozen |
|---|---|---|---|
| 0 | horse | 3 | False |
| 1 | mosquito | 3 | False |
| 2 | cricket | 3 | False |
| 3 | ant | 3 | False |
| 4 | apple | 3 | False |

Before revealing: What would you do? What later battle benefit do you expect? Which alternative would you test?

<details>
<summary>Policy answer and evidence</summary>

| Action | Policy probability |
|---|---:|
| `end_turn` | 94.6% |
| `swap_adjacent:2,3` | 4.2% |
| `swap_adjacent:1,2` | 1.0% |
| `toggle_freeze:3` | 0.1% |
| `toggle_freeze:4` | 0.1% |

Critic value: **7.69**. The recorded continuation earned **9.0**. Their difference is **+1.31**.

That difference illustrates a full-return advantage estimate. Actual PPO training uses GAE (lambda 0.95), rollout bootstrapping, and a stochastic policy—not exactly this deterministic full-episode calculation.

For each action below, force that one move and then follow the policy over 32 sampled futures. These are noisy one-step-deviation estimates, not optimal action values. Matched initial random seeds can diverge after different actions consume randomness.

| Forced first action | Mean future reward | Standard error |
|---|---:|---:|
| `end_turn` | 6.78 | 0.43 |
| `swap_adjacent:2,3` | 6.41 | 0.57 |

Does the most probable action also look best in these rollouts? What would additional samples clarify?

[Complete episode replay](episode-3.json)

</details>

## Situation 4

Turn 3 · gold 10 · wins 2 · lives 5 · shop decisions used 0/30 · opponent family greedy

| Team slot | Pet | Attack / health | XP | Perk | Temporary buff |
|---|---|---|---|---|---|
| 0 | ant | 2 / 3 | 1 | — | +0 / +0 |
| 1 | mosquito | 4 / 4 | 3 | — | +0 / +0 |
| 2 | otter | 1 / 3 | 1 | — | +0 / +0 |
| 3 | cricket | 1 / 3 | 1 | — | +0 / +0 |

| Shop slot | Item | Cost | Frozen |
|---|---|---|---|
| 0 | horse | 3 | False |
| 1 | horse | 3 | False |
| 2 | mosquito | 3 | False |
| 3 | otter | 3 | False |
| 4 | honey | 3 | False |

Before revealing: What would you do? What later battle benefit do you expect? Which alternative would you test?

<details>
<summary>Policy answer and evidence</summary>

| Action | Policy probability |
|---|---:|
| `merge:3,2` | 99.1% |
| `buy_pet:3` | 0.8% |
| `merge:2,1` | 0.0% |
| `roll` | 0.0% |
| `swap_adjacent:2,3` | 0.0% |

Critic value: **5.79**. The recorded continuation earned **8.0**. Their difference is **+2.21**.

That difference illustrates a full-return advantage estimate. Actual PPO training uses GAE (lambda 0.95), rollout bootstrapping, and a stochastic policy—not exactly this deterministic full-episode calculation.

For each action below, force that one move and then follow the policy over 32 sampled futures. These are noisy one-step-deviation estimates, not optimal action values. Matched initial random seeds can diverge after different actions consume randomness.

| Forced first action | Mean future reward | Standard error |
|---|---:|---:|
| `merge:3,2` | 6.69 | 0.20 |
| `buy_pet:3` | 4.28 | 0.63 |
| `merge:2,1` | 6.16 | 0.30 |

Does the most probable action also look best in these rollouts? What would additional samples clarify?

[Complete episode replay](episode-4.json)

</details>

## Situation 5

Turn 5 · gold 7 · wins 3 · lives 4 · shop decisions used 1/30 · opponent family stats

| Team slot | Pet | Attack / health | XP | Perk | Temporary buff |
|---|---|---|---|---|---|
| 0 | fish | 5 / 7 | 3 | — | +0 / +0 |
| 1 | otter | 3 / 5 | 1 | — | +0 / +0 |
| 2 | fish | 2 / 3 | 1 | — | +0 / +0 |
| 3 | ant | 6 / 6 | 3 | — | +0 / +0 |
| 4 | cricket | 4 / 6 | 4 | — | +0 / +0 |

| Shop slot | Item | Cost | Frozen |
|---|---|---|---|
| 0 | mosquito | 3 | False |
| 1 | fish | 3 | False |
| 2 | pig | 3 | False |
| 4 | honey | 3 | False |

Before revealing: What would you do? What later battle benefit do you expect? Which alternative would you test?

<details>
<summary>Policy answer and evidence</summary>

| Action | Policy probability |
|---|---:|
| `merge:1,0` | 89.0% |
| `merge:1,2` | 10.6% |
| `roll` | 0.2% |
| `buy_food:4,4` | 0.1% |
| `swap_adjacent:1,2` | 0.0% |

Critic value: **4.87**. The recorded continuation earned **6.0**. Their difference is **+1.13**.

That difference illustrates a full-return advantage estimate. Actual PPO training uses GAE (lambda 0.95), rollout bootstrapping, and a stochastic policy—not exactly this deterministic full-episode calculation.

For each action below, force that one move and then follow the policy over 32 sampled futures. These are noisy one-step-deviation estimates, not optimal action values. Matched initial random seeds can diverge after different actions consume randomness.

| Forced first action | Mean future reward | Standard error |
|---|---:|---:|
| `merge:1,0` | 6.03 | 0.16 |
| `merge:1,2` | 6.19 | 0.16 |

Does the most probable action also look best in these rollouts? What would additional samples clarify?

[Complete episode replay](episode-5.json)

</details>

## How to interpret your answers

- A forced move is followed by the existing policy, which can undo that move. Similar returns do not prove that positioning never matters.
- A high action probability can coexist with similar or better sampled returns for another action. PPO has learned a preference, not a proof of optimality.
- The uncertainty shown is a standard error for each mean, not a 95% interval for the difference. With only 32 futures, small gaps need more evidence.
- Try explaining one zero-reward shop move in terms of later rewards, then explain why the critic can be wrong about its continuation.
