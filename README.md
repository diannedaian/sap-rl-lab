# SAP RL Lab

An educational reinforcement-learning sandbox for a small, testable slice of
the **Super Auto Pets Turtle Pack**, designed to scale to the complete 60-pet
pack without inheriting the fragile parts of older community projects.

The project currently contains a playable pure-Python shop/battle loop, eight
rollable tier-1 pets, two curriculum foods, one generated food, legal-action
masks, seeded opponents, scripted baselines, a Gymnasium adapter, and a
Maskable PPO training entrypoint.

Phase 1, the eight-pet learning study, is complete, with residual looping documented.
The ultimate goal remains **all 60 Turtle Pack pets and their associated mechanics**;
see the [gated expansion roadmap](docs/ROADMAP.md). The next milestone is a correct
tier-1/2 game, not another eight-pet hyperparameter sweep.
Start with [the one-page conclusion](docs/ROUND5.md),
[delivery and reproduction instructions](docs/DELIVERY.md), or the
[private replay viewer](https://sap-rl-replay-lab.dcao2028.chatgpt.site/).
Models and complete experiment records are available in the
[public release](https://github.com/diannedaian/sap-rl-lab/releases/tag/v0.1.0-eight-pet);
see [download and restore instructions](docs/ARTIFACTS.md). Viewer source and all
38 selected replays are public in `viewer/` and can be run locally without an account.

> This is not yet a frame-perfect reproduction of the live game. The catalog
> records the target game version and verification date. Current tier-1 pet
> mechanics are modeled; Apple and Honey are deliberately retained as simple
> curriculum foods while the current Turtle food roster is being validated.

## Why this exists

The project is meant to make RL understandable. Every learned result should be
answerable in terms of five concrete objects:

1. **Observation:** what information did the agent receive?
2. **Action:** what decisions could it make?
3. **Transition:** how did the simulator change the state?
4. **Reward:** exactly what behavior did we incentivize?
5. **Opponent distribution:** what did “good play” mean during training?

Read [the RL guide](docs/RL_GUIDE.md) before changing the algorithm. Most bad
results in this domain come from the environment or evaluation, not PPO.

## Quick start

The game engine has no dependencies; running the tests requires pytest:

```bash
PYTHONPATH=src python -m sap_rl_lab.cli inspect
PYTHONPATH=src python -m sap_rl_lab.cli random --episodes 100
PYTHONPATH=src python -m sap_rl_lab.cli greedy --episodes 100
PYTHONPATH=src python -m pytest
```

For RL training, use Python 3.9+ in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[rl,dev]"
python -m sap_rl_lab.training train --timesteps 100000 --environments 4
python -m sap_rl_lab.training evaluate runs/ppo/final_model.zip --episodes 100
```

Build separate scripted training and evaluation leagues, then opt into one:

```bash
sap-rl build-league --episodes 100 --seed 0 --output data/leagues/train.json
sap-rl build-league --episodes 100 --seed 10000 --output data/leagues/eval.json
python -m sap_rl_lab.training train --opponent-league data/leagues/train.json
python -m sap_rl_lab.training evaluate runs/ppo/final_model.zip \
  --opponent-league data/leagues/eval.json
```

The first run is a smoke test, not evidence of a strong agent. A meaningful
claim requires multiple training seeds and held-out opponent pools.

## Current scope

Implemented rollable pets:

- Ant, Cricket, Fish, Horse
- Mosquito, Otter, Pig, Pigeon

Implemented tokens/perks:

- Zombie Cricket, Bee, Honey, Bread Crumbs

Implemented shop decisions:

- Buy, merge, feed, sell, roll, freeze, adjacent swap, end turn

Training infrastructure also includes serializable, round-indexed opponent
snapshots and weighted mixtures, so a checkpoint league can be introduced
without changing the engine API.

The fixed action vocabulary contains 71 actions. Only legal actions are exposed
through the mask, and an illegal action raises before changing state.

Complete episodes can be saved as versioned JSON with action names, rewards,
battle traces, configuration, and final state. Replaying the file verifies that
the same seed and action sequence still produce byte-for-byte equivalent data.

## Design documents

- [Replay viewer, failure inspection, and Fish mechanics audit](docs/REPLAY_LAB.md)
- [Architecture](docs/ARCHITECTURE.md)
- [What was retained or replaced](docs/MIGRATION_PLAN.md)
- [RL training guide](docs/RL_GUIDE.md)
- [Reproducible baselines and first PPO result](docs/BASELINE_RESULTS.md)
- [Second experiment: shop loops and reproducible continuation](docs/ROUND2.md)
- [Third experiment: opponent diversity from scratch](docs/ROUND3.md)
- [Fourth experiment: swap penalty versus success-only efficiency bonus](docs/ROUND4.md)
- [Final confirmation: three paired seeds and equal training budgets](docs/ROUND5.md)
- [Frozen model, replay controls, and reproduction](docs/DELIVERY.md)
- [Five shop decisions: choose before revealing the policy](docs/round3-exercises/EXERCISES.md)
- [Turtle Pack accuracy policy](docs/DATA_ACCURACY.md)
- [Roadmap to all 60 pets](docs/ROADMAP.md)
- [Portable Slurm workflow](cluster/README.md)

## Attribution

The architecture was informed by the MIT-licensed `sapai`, `sapai-gym`, and
`super-ml-pets` projects. No source or assets are vendored. See
[third-party notices](THIRD_PARTY_NOTICES.md).
