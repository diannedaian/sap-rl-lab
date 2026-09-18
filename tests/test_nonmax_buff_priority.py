"""Official 0.29 random-buff priority, opt-in to the revised v4 catalog only."""

import random
from dataclasses import replace

import pytest

from sap_rl_lab.catalog import catalog_digest, load_catalog_by_id
from sap_rl_lab.engine import make_pet
from sap_rl_lab.events import EventRuntime


@pytest.mark.parametrize(
    "species,trigger,count",
    [
        ("ant", "faint", 1),
        ("fish", "level_up", 2),
        ("otter", "buy", 1),
        ("beaver", "sell", 2),
    ],
)
@pytest.mark.parametrize("seed", range(10))
def test_random_buff_prefers_nonmaxed_distinct_friends(species, trigger, count, seed):
    catalog = load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")
    owner = make_pet(catalog, species)
    maxed = [make_pet(catalog, "fish") for _ in range(2)]
    for pet in maxed:
        pet.attack = pet.health = 50
    low = [make_pet(catalog, "fish") for _ in range(2)]
    team = [owner, *maxed, *low]
    runtime = EventRuntime(catalog, random.Random(seed), [team, []], combat=False)
    runtime.emit(0, owner, trigger)
    runtime.drain()
    assert sum((p.attack, p.health) != (2, 3) for p in low) == count
    assert all((p.attack, p.health) == (50, 50) for p in maxed)


def test_priority_handles_all_maxed_fewer_nonmaxed_and_no_friends():
    catalog = load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")
    for size in (0, 1, 4):
        owner = make_pet(catalog, "otter")
        owner.experience = 6
        friends = [make_pet(catalog, "fish") for _ in range(size)]
        for pet in friends:
            pet.attack = pet.health = 50
        if size == 4:
            friends[-1].health = 49
        runtime = EventRuntime(catalog, random.Random(7), [[owner, *friends], []], combat=False)
        runtime.emit(0, owner, "buy")
        runtime.drain()
        assert all(p.health == 50 for p in friends)


def test_priority_changes_catalog_fingerprint_not_legacy_data():
    catalog = load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")
    old_pets = {}
    for key, pet in catalog.pets.items():
        abilities = tuple(
            replace(a, params={k: v for k, v in a.params.items() if k != "prefer_nonmaxed"})
            for a in pet.abilities
        )
        old_pets[key] = replace(pet, abilities=abilities)
    assert catalog_digest(catalog) != catalog_digest(replace(catalog, pets=old_pets))
    legacy = load_catalog_by_id("turtle-v0.46-tier1-rules-v2")
    assert not any(
        a.params.get("prefer_nonmaxed") for p in legacy.pets.values() for a in p.abilities
    )


def test_new_expanded_replay_fingerprints_catalog_and_rejects_drift(tmp_path):
    from sap_rl_lab.engine import AutoBattler
    from sap_rl_lab.replay import EpisodeReplay, ReplayRecorder, verify_replay

    catalog = load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")
    replay = ReplayRecorder(AutoBattler(catalog), seed=12).finish()
    assert replay.catalog_sha256 == catalog_digest(catalog)
    path = tmp_path / "replay.json"
    replay.save(path)
    verify_replay(EpisodeReplay.load(path))
    with pytest.raises(ValueError, match="catalog content changed"):
        verify_replay(replace(replay, catalog_sha256="0" * 64))
