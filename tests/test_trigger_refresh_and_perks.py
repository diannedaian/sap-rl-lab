"""Source-backed merge counters and 0.36 perk-as-food specification cases.

These are constructed cases, not captures of the official client's RNG.
"""

import random
from dataclasses import replace

import pytest

from sap_rl_lab.actions import Action, ActionKind
from sap_rl_lab.catalog import catalog_digest, load_catalog_by_id
from sap_rl_lab.domain import GameConfig, ShopItem
from sap_rl_lab.engine import AutoBattler, make_pet
from sap_rl_lab.events import EventRuntime, attack, battle_runtime, health
from sap_rl_lab.shop import merged_pet


def setup():
    catalog = load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")
    engine = AutoBattler(catalog, GameConfig.turtle_curriculum())
    engine.reset(101)
    return catalog, engine


@pytest.mark.parametrize("species", ["ox", "rabbit"])
@pytest.mark.parametrize("target_xp", range(1, 6))
@pytest.mark.parametrize("incoming_xp", range(1, 7))
def test_merge_counter_inherits_destination_or_resets_at_actual_level_up(
    species, target_xp, incoming_xp
):
    catalog, engine = setup()
    target, incoming = (make_pet(catalog, species) for _ in range(2))
    target.experience, incoming.experience = target_xp, incoming_xp
    target.ability_uses, incoming.ability_uses = 0, 1
    old_level = max(target.level, incoming.level)
    engine.state.team = [incoming, target]
    engine.step(Action(ActionKind.MERGE_TEAM, 0, 1))
    result = engine.state.team[0]
    assert result.ability_uses == 0  # The spent source never poisons a fresh target.
    assert result.experience == min(6, max(target_xp, incoming_xp) + 1)

    target.ability_uses, incoming.ability_uses = 1, 0
    engine.state.team = [target, incoming]
    engine.step(Action(ActionKind.MERGE_TEAM, 1, 0))
    result = engine.state.team[0]
    assert result.ability_uses == (0 if result.level > old_level else 1)


@pytest.mark.parametrize(
    "species,trigger", [("rabbit", "friendly_ate_food"), ("ox", "friend_ahead_faints")]
)
@pytest.mark.parametrize("xp,new_level", [(2, 2), (5, 3)])
def test_buy_merge_level_up_refreshes_full_new_quota(species, trigger, xp, new_level):
    catalog, engine = setup()
    target, incoming = (make_pet(catalog, species) for _ in range(2))
    target.experience, target.ability_uses = xp, 3 if species == "rabbit" else (1 if xp == 2 else 2)
    engine.state.team = [target]
    engine.state.shop[0] = ShopItem("pet", species, 3, pet=incoming)
    engine.step(Action(ActionKind.MERGE, 0, 0))
    target = engine.state.team[0]
    assert target.level == new_level and target.ability_uses == 0
    runtime = EventRuntime(catalog, random.Random(1), [engine.state.team, []], combat=False)
    for _ in range(5):
        runtime.emit(0, target, trigger, target)
        runtime.drain()
    quota = 3 if species == "rabbit" else new_level
    assert target.ability_uses == quota


def test_legacy_merge_primitive_keeps_old_counter_semantics_and_data_changes_hash():
    catalog, _ = setup()
    target, incoming = (make_pet(catalog, "ox") for _ in range(2))
    incoming.ability_uses = 1
    assert merged_pet(target, incoming).ability_uses == 1
    assert merged_pet(target, incoming, refresh_triggers=True).ability_uses == 0
    ability = catalog.pets["ox"].abilities[0]
    revised = replace(ability, params={**ability.params, "perk_counts_as_food": False})
    old = replace(
        catalog, pets={**catalog.pets, "ox": replace(catalog.pets["ox"], abilities=(revised,))}
    )
    assert catalog_digest(catalog) != catalog_digest(old)


@pytest.mark.parametrize("combat", [False, True])
@pytest.mark.parametrize("initial_perk", [None, "honey", "meat_bone", "melon"])
@pytest.mark.parametrize("rabbit_level", [1, 2, 3])
def test_ox_new_melon_counts_as_one_food_on_own_side(combat, initial_perk, rabbit_level):
    catalog, _ = setup()
    front, ox, rabbit, enemy_rabbit = (
        make_pet(catalog, name) for name in ("fish", "ox", "rabbit", "rabbit")
    )
    ox.perk = initial_perk
    rabbit.experience = (1, 3, 6)[rabbit_level - 1]
    runtime = EventRuntime(
        catalog, random.Random(1), [[front, ox, rabbit], [enemy_rabbit]], combat=combat
    )
    runtime.faint(0, front)
    runtime.drain()
    new_perk = initial_perk != "melon"
    assert ox.perk == "melon" and ox.attack == 2 and ox.ability_uses == 1
    assert ox.health == 3 + rabbit_level * new_perk
    assert rabbit.ability_uses == int(new_perk)
    assert enemy_rabbit.ability_uses == 0


def test_consuming_then_regaining_melon_triggers_again_but_respects_remaining_food_uses():
    catalog, _ = setup()
    ox, rabbit = (make_pet(catalog, name) for name in ("ox", "rabbit"))
    ox.experience, rabbit.ability_uses = 6, 2
    runtime = EventRuntime(catalog, random.Random(1), [[ox, rabbit], []], combat=True)
    for expected_uses, expected_health in [(3, 4), (3, 4), (3, 4)]:
        runtime.emit(0, ox, "friend_ahead_faints")
        runtime.drain()
        assert ox.perk == "melon" and ox.health == expected_health
        assert rabbit.ability_uses == expected_uses
        runtime.damage_batch([(0, ox, 1)])
        runtime.drain()
        assert ox.perk is None and ox.health == expected_health
    assert ox.ability_uses == 3


@pytest.mark.parametrize("food_name", ["honey", "meat_bone"])
def test_buy_perk_food_notifies_exactly_once_even_with_identical_perk(food_name):
    catalog, engine = setup()
    rabbit = make_pet(catalog, "rabbit")
    engine.state.team = [rabbit]
    for count in (1, 2):
        engine.state.shop[0] = ShopItem("food", food_name, 3)
        engine.step(Action(ActionKind.BUY_FOOD, 0, 0))
        assert rabbit.ability_uses == count and rabbit.health == 2 + count


@pytest.mark.parametrize("dead_species", ["ox", "rabbit"])
def test_simultaneous_dead_ox_or_rabbit_cannot_gain_food_health(dead_species):
    catalog, _ = setup()
    front, ox, rabbit = (make_pet(catalog, name) for name in ("fish", "ox", "rabbit"))
    dead = ox if dead_species == "ox" else rabbit
    dead.health = 0
    runtime = EventRuntime(catalog, random.Random(1), [[front, ox, rabbit], []], combat=True)
    runtime.faint(0, front)
    runtime.drain()
    assert all(p is not dead for p in runtime.teams[0])
    assert rabbit.ability_uses == 0
    if dead_species == "rabbit":
        assert ox.health == 3 and ox.perk == "melon"


def test_shop_food_uses_carry_to_battle_without_mutating_original_team():
    catalog, _ = setup()
    front, ox, rabbit = (make_pet(catalog, name) for name in ("fish", "ox", "rabbit"))
    rabbit.ability_uses = 3
    runtime = battle_runtime([front, ox, rabbit], [], catalog, random.Random(1))
    runtime.faint(0, runtime.teams[0][0])
    runtime.drain()
    assert runtime.teams[0][0].health == 3
    assert ox.perk is None and ox.ability_uses == 0
    assert rabbit.ability_uses == 3


@pytest.mark.parametrize("reverse", [False, True])
def test_merge_keeps_separate_max_permanent_and_temporary_stats_then_expires(reverse):
    catalog, engine = setup()
    first, second = (make_pet(catalog, "duck") for _ in range(2))
    first.attack, first.health, first.temporary_attack, first.temporary_health = 10, 4, 10, 6
    second.attack, second.health, second.temporary_attack, second.temporary_health = 15, 9, 1, 2
    engine.state.team = [first, second]
    engine.step(Action(ActionKind.MERGE_TEAM, int(reverse), int(not reverse)))
    result = engine.state.team[0]
    assert (result.attack, result.health) == (16, 10)
    assert (result.temporary_attack, result.temporary_health) == (10, 6)
    assert (attack(result), health(result)) == (26, 16)
    engine.opponent_provider = lambda *args: []
    engine.step(Action(ActionKind.END_TURN))
    assert (attack(result), health(result)) == (16, 10)
    assert result.temporary_attack == result.temporary_health == 0
