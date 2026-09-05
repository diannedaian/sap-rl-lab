# Second experiment: inefficient shop loops

## Evidence and hypothesis

The saved first policy was replayed locally on exactly the original 500 seeds
and held-out pool. All summary metrics matched the original GPU evaluation.
Detailed action logging showed 10,491 adjacent swaps out of 38,111 decisions
(27.5%). All 17 truncated episodes hit the 30-action shop limit. Failure traces
repeatedly swapped the same adjacent pair, sometimes even identical pets.

The original Gym adapter marks those failures as truncations. Maskable PPO
bootstraps its value estimate at a truncation. That is suitable for an external
time interruption, but can assign optimistic value beyond an agent-caused
safety cutoff. Also, the existing -1 cutoff penalty can be cheaper than playing
out several losing battles. These are mechanisms to test, not proof that the
policy intentionally exploits the cutoff.

The proposed **efficient** training objective introduces two explicit changes:

- Every decision costs 0.005 reward, making long zero-reward loops costly.
- Hitting a safety limit forfeits remaining lives and ends the training
  episode without value bootstrapping. The existing -1 shop-limit penalty is
  replaced, rather than charged twice. At the turn limit the final battle
  reward is retained before the remaining-life penalty.

This shaping changes the optimization objective; it is not guaranteed to
preserve every optimal strategy. All candidate policies are therefore evaluated
using the original game rewards, full legal-action masks, and original cutoffs.
No strategic action is removed. Evaluation reports raw returns and 10-win
success, plus swap counts, unspent gold, and failure reasons.

Sources: [Gymnasium's termination and truncation explanation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
and [Maskable PPO's rollout implementation](https://sb3-contrib.readthedocs.io/en/master/_modules/sb3_contrib/ppo_mask/ppo_mask.html).

## Comparison protocol

`sap_rl_lab.experiments` writes its full protocol and file hashes before training.
It runs one experiment at a time in a new directory and refuses to overwrite
existing results.

Both arms start from the same first-run policy weights with fresh optimizer
state, use the same training league and PPO settings, and receive the same
additional decision budget. **Control** continues the original objective.
**Efficient** enables the two changes above. This separates the combined
intervention from simply training longer. It does not isolate the contributions
of the two changes individually.

The full protocol uses continuation seeds 17, 23, and 41, with 1 million
additional decisions per arm and seed. These share a pretrained initialization;
they are not independent experiments from scratch.

- Training league: exact first-run pool.
- Validation league: 100 scripted episodes beginning at seed 30000.
- Validation episodes: seeds 40000–40199 every 100,000 decisions and at the end.
- Test league: 100 scripted episodes beginning at seed 50000.
- Final test episodes: seeds 60000–60999.

Checkpoint selection uses validation success, then validation return, with the
earliest checkpoint winning exact ties. The untouched initial policy is also a
candidate. The test pool is evaluated only after all training and selection.
Original PPO, greedy, and random policies use the same final test protocol.

Every evaluation includes per-episode rows for paired comparisons and the
opponent-pool hash. Bootstrap intervals quantify episode-sampling uncertainty
for fixed policy pairs; they do not establish uncertainty over independent
training runs. Report each continuation seed and the mean without treating
shared test episodes as independent training replicates.

## Running

With the first-run artifacts in `BASE_RUN`, use a fresh `OUTPUT` path:

```bash
python -m sap_rl_lab.experiments --base-run BASE_RUN --output OUTPUT
```

For a short validation-only pilot:

```bash
python -m sap_rl_lab.experiments --base-run BASE_RUN --output OUTPUT \
  --timesteps 200000 --seeds 17 --pilot
```

`cluster/round2.sbatch` runs the full comparison in one bounded Slurm allocation.
Pass an existing personal Python environment as `SAP_RL_PYTHON` and the first-run
directory as `SAP_RL_BASE_RUN`. No dependency installation or shared software
changes are required. Use a fresh versioned checkout for the source and supply
site-specific routing privately at submission.
