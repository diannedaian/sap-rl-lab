"""Observed lethal triggers, isolated from unsupported armor and recipient skills."""

import json
import random
from pathlib import Path

from sap_rl_lab.catalog import load_catalog_by_id
from sap_rl_lab.domain import Pet
from sap_rl_lab.events import EventRuntime


def observed_exchange():
    case = json.loads(
        (Path(__file__).parent / "fixtures/real_game/camel_elephant_lethal_2024.json").read_text()
    )
    left, right = case["friendly"], case["enemy"]
    camel, elephant = Pet(**left["owner"]), Pet(**right["owner"])
    # Swan has no relevant in-battle triggered ability. We transfer only the
    # measured damage/buff component, not Bison/Blowfish behavior or Garlic.
    friend = Pet("swan", **left["recipient_before"])
    enemy_friend = Pet("swan", **right["recipient_before"])
    runtime = EventRuntime(
        load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4"),
        random.Random(9),
        [[camel, friend], [elephant, enemy_friend]],
        combat=True,
    )
    runtime.emit(0, camel, "after_attack")
    runtime.emit(1, elephant, "after_attack")
    runtime.damage_batch(
        [(0, camel, left["net_attack_damage"]), (1, elephant, right["net_attack_damage"])]
    )
    assert camel.health == left["owner_health_after"]
    assert elephant.health == right["owner_health_after"]
    # Neither ability's effect has been drained yet.
    assert friend.health == left["recipient_before"]["health"]
    assert enemy_friend.health == right["recipient_before"]["health"]
    runtime.drain()
    return case, runtime, camel, elephant, friend, enemy_friend


def test_recorded_lethal_camel_still_gives_level_two_buff():
    case, runtime, camel, _, friend, _ = observed_exchange()
    assert camel.health == -5
    expected = case["friendly"]["recipient_after"]
    assert (friend.attack, friend.health) == (expected["attack"], expected["health"])
    assert runtime.teams[0] == [friend]


def test_recorded_dead_elephant_hits_twice_before_camel_hurt_and_removal():
    case, runtime, _, elephant, _, friend = observed_exchange()
    assert elephant.health == -18
    expected = case["enemy"]["recipient_after"]
    assert (friend.attack, friend.health) == (expected["attack"], expected["health"])
    assert runtime.teams[1] == [friend]
    elephant_trigger = runtime.trace.index("after_attack opponent:elephant L2: damage_behind")
    camel_trigger = runtime.trace.index("hurt you:camel L2: buff_position")
    assert runtime.trace[elephant_trigger + 1 : camel_trigger] == [
        "opponent:swan takes 1 damage",
        "opponent:swan takes 1 damage",
    ]
    assert elephant_trigger < camel_trigger < runtime.trace.index("you:camel faints")
    assert camel_trigger < runtime.trace.index("opponent:elephant faints")
