"""Finite v6 stress gate: deterministic battles and exact reachable episode replays."""

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sap_rl_lab.baselines import scripted_policy
from sap_rl_lab.catalog import catalog_digest, load_catalog_by_id
from sap_rl_lab.engine import make_pet, resolve_battle
from sap_rl_lab.env import SapAutoBattlerEnv
from sap_rl_lab.expanded_confirmation import save_new
from sap_rl_lab.round3 import source_archive


def audit(output, battles=6000, episodes=40):
    output.mkdir(parents=True, exist_ok=False)
    sources = source_archive(output)
    catalog = load_catalog_by_id("turtle-v0.46-tier4-v6")
    rng = random.Random(51000000)
    fingerprints, seen = hashlib.sha256(), set()
    case = None
    try:
        for index in range(battles):
            teams = []
            for _ in range(2):
                team = []
                for _ in range(rng.randrange(1, 6)):
                    species = rng.choice(list(catalog.pets))
                    p = make_pet(catalog, species)
                    p.attack, p.health = rng.randint(1, 50), rng.randint(1, 50)
                    p.experience = rng.choice([1, 3, 6])
                    p.perk = rng.choice(
                        [None, "honey", "melon", "garlic", "chili", "peanut", "bread"]
                    )
                    if species == "parrot":
                        p.copied_ability = rng.choice(catalog.rollable_pet_ids)
                    team.append(p)
                    seen.add(species)
                teams.append(team)
            case = {"index": index, "teams": [[asdict(p) for p in t] for t in teams]}
            a = resolve_battle(*teams, catalog, random.Random(index))
            b = resolve_battle(*teams, catalog, random.Random(index))
            assert a == b and [[asdict(p) for p in t] for t in teams] == case["teams"]
            fingerprints.update(json.dumps(asdict(a), sort_keys=True).encode())
        rows, actions, species_seen = [], 0, set()
        for family_index, family in enumerate(
            ("tier4_stats", "tier4_summon", "tier4_tempo", "random")
        ):
            policy = None if family == "random" else scripted_policy(family)
            env = SapAutoBattlerEnv(catalog=catalog, allow_development=True, action_cost=0.005)
            replay = SapAutoBattlerEnv(catalog=catalog, allow_development=True, action_cost=0.005)
            try:
                for i in range(episodes):
                    seed = 52000000 + family_index * 10000 + i
                    obs, _ = env.reset(seed=seed)
                    obs2, _ = replay.reset(seed=seed)
                    policy_rng = random.Random(seed + 1000003)
                    trace = hashlib.sha256()
                    length = 0
                    while True:
                        case = {
                            "family": family,
                            "seed": seed,
                            "step": length,
                            "state": env.engine.state.to_dict(),
                        }
                        assert env.observation_space.contains(obs)
                        for key in obs:
                            np.testing.assert_array_equal(obs[key], obs2[key])
                        mask = env.action_masks()
                        np.testing.assert_array_equal(mask, replay.action_masks())
                        action = (
                            int(policy_rng.choice(np.flatnonzero(mask)))
                            if policy is None
                            else env.engine.codec.encode(policy.choose(env.engine, policy_rng))
                        )
                        assert mask[action]
                        obs, reward, done, cut, info = env.step(action)
                        obs2, r2, d2, c2, i2 = replay.step(action)
                        assert (reward, done, cut, info) == (r2, d2, c2, i2)
                        assert env.engine.state.to_dict() == replay.engine.state.to_dict()
                        trace.update(
                            json.dumps(
                                [action, reward, env.engine.state.to_dict()], sort_keys=True
                            ).encode()
                        )
                        species_seen.update(p.spec_id for p in env.engine.state.team)
                        length += 1
                        actions += 1
                        if done or cut:
                            assert not cut, f"Unexpected cutoff: {info}"
                            break
                    rows.append(
                        {
                            "family": family,
                            "seed": seed,
                            "actions": length,
                            "wins": env.engine.state.wins,
                            "sha256": trace.hexdigest(),
                        }
                    )
            finally:
                env.close()
                replay.close()
        assert set(catalog.rollable_pet_ids) <= seen
        save_new(
            output / "summary.json",
            {
                "catalog_id": catalog.catalog_id,
                "catalog_sha256": catalog_digest(catalog),
                "source_files_sha256": sources,
                "battles_twice_reproduced": battles,
                "battle_trace_sha256": fingerprints.hexdigest(),
                "episodes_exactly_replayed": len(rows),
                "episode_actions": actions,
                "synthetic_species": sorted(seen),
                "reachable_species": sorted(species_seen),
                "episodes": rows,
                "official_client_parity": False,
            },
        )
        print(
            f"PASS: {battles} battles reproduced twice; "
            f"{len(rows)} exact episodes / {actions} actions",
            flush=True,
        )
    except Exception as exc:
        save_new(output / "failure.json", {"case": case, "error": repr(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--battles", type=int, default=6000)
    parser.add_argument("--episodes", type=int, default=40)
    args = parser.parse_args()
    audit(args.output, args.battles, args.episodes)
