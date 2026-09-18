"""Chosen sandbox ordering, NOT an additional observed official-client fixture."""

import random

import pytest

from sap_rl_lab.catalog import load_catalog_by_id
from sap_rl_lab.domain import Pet
from sap_rl_lab.events import EventRuntime


@pytest.mark.parametrize("sheep_health", [1, 2])
def test_elephant_retargets_living_rear_before_delayed_sheep_summons(sheep_health):
    elephant = Pet("elephant", 3, 7, experience=6)
    sheep = Pet("sheep", 2, sheep_health)
    friend = Pet("swan", 1, 10)
    runtime = EventRuntime(
        load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4"),
        random.Random(9),
        [[elephant, sheep, friend], []],
        combat=True,
    )
    runtime.emit(0, elephant, "after_attack")
    runtime.drain()
    assert sheep.health == 0
    assert friend.health == 10 - (3 - sheep_health)
    assert [p.spec_id for p in runtime.teams[0]] == ["elephant", "ram", "ram", "swan"]
    assert all(p.health == 2 for p in runtime.teams[0][1:3])
    damage_indices = [i for i, line in enumerate(runtime.trace) if "takes 1 damage" in line]
    summon_indices = [i for i, line in enumerate(runtime.trace) if "summons" in line]
    assert len(damage_indices) == 3 and len(summon_indices) == 2
    assert max(damage_indices) < runtime.trace.index("you:sheep faints") < min(summon_indices)
