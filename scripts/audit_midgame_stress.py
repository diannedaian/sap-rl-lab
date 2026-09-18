"""Bounded v5 stress and exact replay audit, not a client-parity certificate."""

import argparse
import random
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from sap_rl_lab.baselines import scripted_policy
from sap_rl_lab.catalog import catalog_digest, load_catalog_by_id
from sap_rl_lab.domain import GameConfig
from sap_rl_lab.engine import AutoBattler, make_pet
from sap_rl_lab.env import SapAutoBattlerEnv
from sap_rl_lab.evaluation import file_digest
from sap_rl_lab.events import attack, health
from sap_rl_lab.experiments import save_json
from sap_rl_lab.midgame_events import midgame_battle_runtime
from sap_rl_lab.replay import ReplayRecorder, verify_replay
from sap_rl_lab.round3 import source_archive


def audit(output, battles=6000, episodes=40):
    output.mkdir(parents=True, exist_ok=False)
    catalog = load_catalog_by_id("turtle-v0.46-midgame-v5")
    hashes = source_archive(output)
    species, effects, endings, actions = Counter(), Counter(), Counter(), Counter()
    max_events = max_attacks = draw_limits = 0
    for seed in range(14_000_000, 14_000_000 + battles):
        rng = random.Random(seed)
        teams = []
        for _ in range(2):
            team = []
            for _ in range(rng.randint(1, 5)):
                pet = make_pet(catalog, rng.choice(catalog.rollable_pet_ids))
                pet.attack, pet.health = rng.randint(1, 50), rng.randint(1, 50)
                if seed % 4 == 0:
                    pet.attack, pet.health = rng.randint(1, 2), rng.randint(40, 50)
                pet.experience = rng.choice((1, 3, 6))
                pet.perk = rng.choice(
                    (None, "honey", "meat_bone", "melon", "garlic", "chili", "cake")
                )
                if pet.spec_id == "parrot":
                    pet.copied_ability = rng.choice(catalog.rollable_pet_ids)
                species[pet.spec_id] += 1
                team.append(pet)
            teams.append(team)
        before = deepcopy(teams)
        results = []
        for _ in range(2):
            runtime = midgame_battle_runtime(*teams, catalog, random.Random(seed + 10000))
            outcome, exchanges = runtime.battle()
            assert teams == before
            assert all(len(team) <= 5 for team in runtime.teams)
            assert all(
                0 < health(p) <= 50 and 0 <= attack(p) <= 50 for team in runtime.teams for p in team
            )
            max_events = max(max_events, runtime.events_processed)
            max_attacks = max(max_attacks, exchanges)
            results.append(
                (
                    outcome,
                    exchanges,
                    runtime.trace,
                    [[asdict(p) for p in team] for team in runtime.teams],
                )
            )
        assert results[0] == results[1]
        draw_limits += int(runtime.attack_limit_reached)
        effects.update(s.split(": ", 1)[-1] for s in runtime.trace if " L" in s)
    for family in ("random", "midgame_stats", "midgame_summon", "midgame_tempo"):
        policy = scripted_policy(family)
        for index in range(episodes):
            seed = 14_100_000 + index
            engine = AutoBattler(catalog, GameConfig.turtle_midgame())
            recorder = ReplayRecorder(engine, seed)
            env = SapAutoBattlerEnv(catalog, allow_development=True)
            env.engine = engine  # Observe the exact audited state, not a second episode.
            rng = random.Random(seed + 500000)
            while not (engine.state.terminated or engine.state.truncated):
                legal = engine.legal_action_ids()
                assert set(legal) == {
                    i for i, allowed in enumerate(engine.action_mask()) if allowed
                }
                action = engine.codec.encode(policy.choose(engine, rng))
                assert action in legal
                transition = recorder.step(action)
                actions[engine.codec.decode(action).kind.value] += 1
                assert len(engine.state.team) <= 5 and len(engine.state.shop) == 9
                assert engine.state.gold >= 0 and engine.unlocked_tier <= 3
                assert all(0 < health(p) <= 50 and 0 <= attack(p) <= 50 for p in engine.state.team)
                assert env.observation_space.contains(env._observation())
                assert (
                    len(recorder.steps)
                    <= engine.config.max_turns * engine.config.max_actions_per_turn
                )
            replay = recorder.finish()
            verify_replay(replay)
            endings[transition.info.get("reason", "ordinary")] += 1
            if index < 3:
                replay.save(output / f"{family}-{seed}.json")
        print(f"Verified {family}: {episodes} whole-episode replays", flush=True)
    assert set(species) == set(catalog.rollable_pet_ids)
    save_json(
        output / "summary.json",
        {
            "catalog_id": catalog.catalog_id,
            "catalog_sha256": catalog_digest(catalog),
            "source_files_sha256": hashes,
            "audit_script_sha256": file_digest(__file__),
            "battles_twice_reproduced": battles,
            "episodes_exactly_replayed": episodes * 4,
            "species_counts": dict(species),
            "effect_counts": dict(effects),
            "endings": dict(endings),
            "action_counts": dict(actions),
            "max_battle_events": max_events,
            "max_battle_attacks": max_attacks,
            "battle_attack_limit_draws": draw_limits,
            "limitations": "Synthetic battles are stress cases, not reachable training opponents. "
            "Episode replays use actual capped-Tier-3 shops. Not official-client parity.",
        },
    )
    print(
        f"Passed {battles} twice-reproduced battles; {episodes * 4} exact episode replays",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--battles", type=int, default=6000)
    parser.add_argument("--episodes-per-family", type=int, default=40)
    args = parser.parse_args()
    if min(args.battles, args.episodes_per_family) < 1:
        parser.error("counts must be positive")
    audit(Path(args.output), args.battles, args.episodes_per_family)
