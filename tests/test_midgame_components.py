"""Staged midgame specification tests, not official-client parity fixtures."""

import random

import pytest

from sap_rl_lab.catalog import catalog_digest, load_catalog, load_catalog_by_id
from sap_rl_lab.domain import GameConfig, GameState, ShopItem
from sap_rl_lab.engine import AutoBattler, make_pet, resolve_battle
from sap_rl_lab.midgame_events import MidgameEventRuntime


def catalog():
    return load_catalog("turtle_v0_46_midgame_v5.json")


def pet(name, level=1, **values):
    result = make_pet(catalog(), name)
    result.experience = (1, 3, 6)[level - 1]
    for key, value in values.items():
        setattr(result, key, value)
    return result


def runtime(team, enemies=None, *, combat=False, shop=None, seed=13):
    state = GameState(team=team, shop=shop or [])
    return MidgameEventRuntime(
        catalog(),
        random.Random(seed),
        [team, enemies or []],
        combat=combat,
        state=None if combat else state,
    )


def test_midgame_catalog_is_complete_roster_with_explicit_new_contract():
    c = catalog()
    assert len(c.rollable_pet_ids) == 40
    assert {p.id for p in c.pets.values() if p.tier == 4} == set(
        "skunk hippo bison blowfish turtle squirrel penguin deer whale parrot".split()
    )
    assert 4 in c.implemented_shop_tiers
    assert AutoBattler(c).config == GameConfig.turtle_midgame()
    with pytest.raises(ValueError, match="contract"):
        AutoBattler(c, GameConfig.turtle_curriculum())
    assert resolve_battle([], [], c, random.Random(1)).attacks == 0
    assert catalog_digest(load_catalog_by_id("turtle-v0.46-tier12-curriculum-v4")) == (
        "2d71a64f5136f0a24fffb7d0ccc91ff9567d9cb7fae645119d932525642e23fe"
    )


@pytest.mark.parametrize("level", [1, 2, 3])
def test_bison_condition_all_levels_and_permanent_buff(level):
    owner, friend = pet("bison", level), pet("fish", 2)
    r = runtime([owner, friend])
    r.phase("end_turn")
    assert (owner.attack, owner.health) == (4, 4)
    friend.experience = 6
    r.phase("end_turn")
    assert (owner.attack, owner.health) == (4 + 2 * level, 4 + 2 * level)
    assert owner.temporary_attack == owner.temporary_health == 0
    r.phase("end_turn")
    assert (owner.attack, owner.health) == (4 + 4 * level, 4 + 4 * level)


def test_bison_cannot_count_itself_and_only_one_per_team_phase():
    lonely = pet("bison", 3)
    runtime([lonely]).phase("end_turn")
    assert (lonely.attack, lonely.health) == (4, 4)
    weak, strong, friend = pet("bison"), pet("bison", attack=10), pet("fish", 3)
    r = runtime([weak, strong, friend])
    r.phase("end_turn")
    assert (weak.attack, weak.health) == (4, 4)
    assert (strong.attack, strong.health) == (12, 6)
    # Opposing team has an independent allowance in the same queue.
    enemy = pet("bison")
    r.teams[1] = [enemy, pet("fish", 3)]
    r.phase("end_turn")
    assert (enemy.attack, enemy.health) == (6, 6)


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("seed", range(5))
def test_penguin_only_start_turn_distinct_eligible_nonmaxed_friends(level, seed):
    owner = pet("penguin", level)
    low = pet("fish", 1)
    high = [pet("fish", 2), pet("fish", 3)]
    maxed = pet("fish", 3, attack=50, health=50)
    r = runtime([owner, low, *high, maxed], seed=seed)
    r.phase("end_turn")
    assert all((p.attack, p.health) == (2, 3) for p in high)
    r.phase("start_turn")
    assert all((p.attack, p.health) == (2 + level, 3 + level) for p in high)
    assert (low.attack, low.health) == (2, 3)
    assert (owner.attack, owner.health) == (2, 3)
    assert (maxed.attack, maxed.health) == (50, 50)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_squirrel_existing_foods_only_discount_floor_and_frozen_stacking(level):
    foods = [ShopItem("food", "apple", 3), ShopItem("food", "sleeping_pill", 1, True)]
    offered_pet = ShopItem("pet", "fish", 3, pet=pet("fish"))
    r = runtime([pet("squirrel", level)], shop=[*foods, offered_pet, None])
    r.phase("start_turn")
    assert [p.cost for p in foods] == [3 - level, 0]
    assert offered_pet.cost == 3 and foods[1].frozen
    # Stocking new food later in the turn does not inherit the earlier discount.
    new_food = ShopItem("food", "apple", 3)
    r.state.shop.append(new_food)
    assert new_food.cost == 3
    r.phase("start_turn")
    assert foods[0].cost == max(0, 3 - 2 * level)
    assert new_food.cost == 3 - level


@pytest.mark.parametrize("level", [1, 2, 3])
def test_turtle_skips_identical_perks_and_grants_permanent_shop_melon(level):
    owner = pet("turtle", level)
    already = pet("fish", perk="melon")
    targets = [pet("fish") for _ in range(3)]
    r = runtime([owner, already, *targets])
    r.faint(0, owner)
    r.drain()
    assert already.perk == "melon"
    assert [p.perk for p in targets] == ["melon"] * level + [None] * (3 - level)
    assert r.teams[0] == [already, *targets]


def test_turtle_perk_counts_as_eating_once_and_does_not_feed_skipped_friend():
    owner, already, eater, rabbit = (
        pet("turtle"),
        pet("fish", perk="melon"),
        pet("fish"),
        pet("rabbit"),
    )
    r = runtime([owner, already, eater, rabbit])
    r.faint(0, owner)
    r.drain()
    assert already.health == 3
    assert eater.perk == "melon" and eater.health == 4


@pytest.mark.parametrize("level,remaining", [(1, 18), (2, 9), (3, 1)])
@pytest.mark.parametrize("perk", [None, "melon", "garlic"])
def test_skunk_removes_health_without_damage_hurt_or_armor_consumption(level, remaining, perk):
    target = pet("blowfish", health=27, perk=perk)
    other = pet("fish", health=26)
    r = runtime([pet("skunk", level)], [target, other], combat=True)
    r.phase("start_battle")
    assert target.health == remaining and target.perk == perk
    assert other.health == 26
    assert r.teams[0][0].health == 5  # No Blowfish Hurt shot.
    assert not any("takes" in line for line in r.trace)


def test_skunk_never_kills_and_handles_no_enemies():
    for hp in (1, 2, 3, 50):
        target = pet("fish", health=hp)
        r = runtime([pet("skunk", 3)], [target], combat=True)
        r.phase("start_battle")
        assert target.health == 1 and r.teams[1] == [target]
    runtime([pet("skunk")], combat=True).phase("start_battle")


@pytest.mark.parametrize("level", [1, 2, 3])
def test_blowfish_lethal_hurt_fires_but_pill_does_not(level):
    owner, enemy = pet("blowfish", level, health=1), pet("fish", health=40)
    r = runtime([owner], [enemy], combat=True)
    r.damage_batch([(0, owner, 1)])
    r.drain()
    assert enemy.health == 40 - 3 * level
    assert not r.teams[0]
    owner = pet("blowfish", level)
    r = runtime([owner])
    r.faint(0, owner)
    r.drain()
    assert not any("hurt" in line for line in r.trace)
