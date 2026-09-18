"""v6 specification tests, not mislabelled official-client recordings."""

import random
from dataclasses import asdict

import numpy as np
import pytest

from sap_rl_lab.actions import Action, ActionKind
from sap_rl_lab.catalog import catalog_digest, load_catalog_by_id
from sap_rl_lab.domain import GameConfig, GameState, ShopItem
from sap_rl_lab.engine import AutoBattler, make_pet, resolve_battle
from sap_rl_lab.env import SapAutoBattlerEnv
from sap_rl_lab.events import health
from sap_rl_lab.opponents import PetSnapshot
from sap_rl_lab.tier4_events import Tier4EventRuntime

CATALOG_ID = "turtle-v0.46-tier4-v6"


def catalog():
    return load_catalog_by_id(CATALOG_ID)


def pet(name, level=1, **kwargs):
    p = make_pet(catalog(), name)
    p.experience = (1, 3, 6)[level - 1]
    for key, value in kwargs.items():
        setattr(p, key, value)
    return p


def runtime(team, enemies=None, *, combat=False, shop=None):
    state = GameState(team=team, shop=shop or [None] * 9, turn=7)
    return Tier4EventRuntime(
        catalog(),
        random.Random(42),
        [team, enemies or []],
        combat=combat,
        state=None if combat else state,
    )


def test_catalog_contract_append_only_and_old_digest():
    c, old = catalog(), load_catalog_by_id("turtle-v0.46-midgame-v5")
    assert len(c.rollable_pet_ids) == 50
    assert list(c.pets)[: len(old.pets)] == list(old.pets)
    assert all(c.pets[k] == v for k, v in old.pets.items())
    assert all(c.foods[k] == v for k, v in old.foods.items())
    assert c.implemented_shop_tiers == (1, 2, 3, 4, 5)
    assert catalog_digest(old) == "54d325038aaf9abaa41cc5c0ba3abe4b72b5fe76bccd656f1f81c80b032115f4"
    assert AutoBattler(c).config == GameConfig.turtle_tier4()
    with pytest.raises(ValueError, match="contract"):
        AutoBattler(c, GameConfig.turtle_midgame())


@pytest.mark.parametrize("level", [1, 2, 3])
def test_armadillo_buffs_every_living_pet_both_sides_and_cap(level):
    owner, friend, enemy, dead = (
        pet("armadillo", level),
        pet("fish"),
        pet("fish", health=48),
        pet("ant", health=0),
    )
    r = runtime([owner, friend], [enemy, dead], combat=True)
    r.phase("start_battle")
    assert owner.health == 8 + 8 * level and friend.health == 3 + 8 * level
    assert enemy.health == 50 and dead.health <= 0


@pytest.mark.parametrize("level", [1, 2, 3])
def test_croc_projectiles_retarget_and_melon_blocks_one(level):
    owner = pet("crocodile", level)
    enemies = [pet("fish", health=30), pet("fish", health=5, perk="melon")]
    r = runtime([owner], enemies, combat=True)
    r.phase("start_battle")
    assert enemies[-1].perk is None if level == 1 else len(r.alive(1)) == 1
    assert r.alive(1)[0].health == (22 if level == 3 else 30)
    assert sum("damage_last_enemy" in s for s in r.trace) == level


@pytest.mark.parametrize("level", [1, 2, 3])
def test_monkey_targets_front_pet_including_self(level):
    owner, friend = pet("monkey", level), pet("fish")
    runtime([friend, owner]).phase("end_turn")
    assert (friend.attack, friend.health) == (2 + 2 * level, 3 + 2 * level)
    runtime([owner]).phase("end_turn")
    assert (owner.attack, owner.health) == (1 + 2 * level, 2 + 2 * level)


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize(
    "target_id,multiplier", [("fish", 2), ("bee", 2), ("chick", 2), ("hippo", 1)]
)
def test_rhino_damage_and_tier1_token_bonus(level, target_id, multiplier):
    owner, first, target = pet("rhino", level), pet("ant", health=1), pet(target_id, health=50)
    r = runtime([owner], [first, target], combat=True)
    r.damage_batch([(1, first, 10)], sources=[(0, owner)])
    r.drain()
    assert target.health == 50 - 4 * level * multiplier


def test_rhino_chain_and_no_double_knockout():
    owner = pet("rhino")
    enemies = [pet("ant", health=1), pet("chick"), pet("chick"), pet("hippo", health=50)]
    r = runtime([owner], enemies, combat=True)
    r.damage_batch([(1, enemies[0], 10)], sources=[(0, owner)])
    r.drain()
    assert len(r.alive(1)) == 1 and r.alive(1)[0].health == 46
    assert sum("rhino_snipe" in s for s in r.trace) == 3


@pytest.mark.parametrize("level", [1, 2, 3])
def test_rooster_chicks_levels_current_attack_and_honey(level):
    owner = pet("rooster", level, attack=7, temporary_attack=2, perk="honey")
    r = runtime([owner])
    r.faint(0, owner)
    r.drain()
    assert [p.spec_id for p in r.teams[0]] == ["bee"] + ["chick"] * level
    assert all((p.attack, p.health, p.level) == (5, 1, 1) for p in r.teams[0][1:])


def test_rooster_full_team_cannot_overfill_or_apply_shop_buff_to_chicks():
    owner = pet("rooster", 3)
    r = runtime([owner] + [pet("fish") for _ in range(4)])
    r.state.shop_attack_bonus = 20
    r.faint(0, owner)
    r.drain()
    assert len(r.teams[0]) == 5 and r.teams[0][0].attack == 3


@pytest.mark.parametrize("level", [1, 2, 3])
def test_scorpion_only_gains_peanut_when_summoned(level):
    owner, rabbit = pet("scorpion", level), pet("rabbit")
    r = runtime([owner, rabbit])
    r.emit(0, owner, "buy", owner)
    r.drain()
    assert owner.perk is None
    r.notify_summon(0, owner)
    r.drain()
    assert owner.perk == "peanut" and owner.health == 4
    assert PetSnapshot.from_pet(owner).to_pet() == owner


@pytest.mark.parametrize(
    "attack_value,melon,dead", [(1, False, True), (20, True, False), (21, True, True)]
)
def test_peanut_only_kills_if_attack_penetrates_melon(attack_value, melon, dead):
    owner = pet("scorpion", attack=attack_value, health=50, perk="peanut")
    enemy = pet("hippo", attack=1, health=50, perk="melon" if melon else None)
    r = runtime([owner], [enemy], combat=True)
    r.exchange_damage(owner, enemy, attack_value, 1)
    r.drain()
    assert (health(enemy) <= 0) == dead
    assert sum("hurt opponent:hippo" in s for s in r.trace) == 0


def test_peanut_not_on_ability_damage_and_both_attacks_resolve():
    source, target = pet("crocodile", perk="peanut"), pet("fish", attack=50, health=50)
    r = runtime([source], [target], combat=True)
    r.phase("start_battle")
    assert target.health == 42
    r.exchange_damage(source, target, source.attack, target.attack)
    r.drain()
    assert not r.alive(0) and not r.alive(1)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_cow_replaces_all_food_including_frozen_and_milk_stats(level):
    cow = pet("cow", level)
    r = runtime([cow], shop=[ShopItem("food", "pear", 3, True)] + [None] * 8)
    r.emit(0, cow, "buy", cow)
    r.drain()
    milk = [i for i in r.state.shop if i]
    assert len(milk) == 2 and all(i.cost == 0 and not i.frozen for i in milk)
    f = catalog().foods[milk[0].item_id]
    assert (f.params["attack"], f.params["health"]) == (level, 2 * level)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_seal_only_own_food_distinct_friends_and_turkey_summons(level):
    owner = pet("seal", level)
    friends = [pet("fish") for _ in range(4)]
    r = runtime([owner, *friends])
    r.notify_food_eaten(0, friends[0])
    r.drain()
    assert all(p.attack == 2 for p in friends)
    r.notify_food_eaten(0, owner)
    r.drain()
    assert sorted(p.attack for p in friends) == [2] + [2 + level] * 3
    assert owner.attack == 3
    turkey, newcomer = pet("turkey", level), pet("fish")
    r = runtime([turkey, newcomer])
    r.notify_summon(0, newcomer)
    r.drain()
    assert (newcomer.attack, newcomer.health) == (2 + 3 * level, 3 + level)
    assert newcomer.temporary_attack == 0


@pytest.mark.parametrize("level", [1, 2, 3])
def test_shark_each_other_faint_once_no_self_revive(level):
    owner, a, b = pet("shark", level), pet("fish"), pet("fish")
    r = runtime([a, b, owner])
    r.damage_batch([(0, a, 99), (0, b, 99)])
    r.drain()
    assert (owner.attack, owner.health) == (2 + 4 * level, 2 + 4 * level)
    r.queue_faints()
    r.drain()
    assert owner.attack == 2 + 4 * level
    r.faint(0, owner)
    r.drain()
    assert not r.alive(0)


def test_bread_temporary_health_resets_each_round_and_perk_stays():
    e = AutoBattler(catalog(), opponent_provider=lambda *args: [])
    e.reset(seed=123)
    e.state.team = [pet("fish", perk="bread")]
    seen = []

    def capture(*args):
        seen.append(e.state.team[0].temporary_health)
        return []

    e.opponent_provider = capture
    for _ in range(2):
        e.step(Action(ActionKind.END_TURN))
        assert e.state.team[0].temporary_health == 0 and e.state.team[0].perk == "bread"
    assert seen == [7, 7]


def test_canned_food_empty_team_current_and_future_rewards_not_eaten():
    e = AutoBattler(catalog())
    e.reset(seed=123)
    offered = pet("fish")
    e.state.shop = [ShopItem("pet", "fish", 3, pet=offered), ShopItem("food", "canned_food", 3)] + [
        None
    ] * 7
    move = Action(ActionKind.BUY_FOOD, 1, 0)
    assert move in e.legal_actions()
    e.step(move)
    assert (offered.attack, offered.health) == (3, 4)
    assert (e.state.shop_attack_bonus, e.state.shop_health_bonus) == (1, 1)
    e.state.turn = 7
    e._roll_shop(free=True)
    for item in e.state.shop:
        if item and item.kind == "pet":
            spec = catalog().pets[item.item_id]
            assert (item.pet.attack, item.pet.health) == (spec.attack + 1, spec.health + 1)
    e._stock_level_up_choices()
    rewards = [i for i in e.state.shop if i and i.choice_group]
    assert rewards and all(catalog().pets[i.item_id].tier == 5 for i in rewards)
    assert all(i.pet.attack == catalog().pets[i.item_id].attack + 1 for i in rewards)
    assert GameState(**e.state.to_dict()).shop_attack_bonus == 1


def test_v6_observes_new_perks_and_shop_bonus_and_old_contract_unchanged():
    env = SapAutoBattlerEnv(catalog=catalog(), allow_development=True)
    obs, _ = env.reset(seed=1)
    assert {k: v.shape for k, v in obs.items()} == {
        "global": (10,),
        "team": (5, 130),
        "shop": (9, 159),
    }
    env.engine.state.team = [pet("scorpion", perk="peanut"), pet("fish", perk="bread")]
    env.engine.state.shop_attack_bonus = 3
    obs = env._observation()
    assert env.observation_space.contains(obs)
    np.testing.assert_equal(obs["team"][:2, -2:], [[1, 0], [0, 1]])
    assert obs["global"][-2] == np.float32(3 / 50)
    assert env.action_space.n == 139
    old = SapAutoBattlerEnv(
        catalog=load_catalog_by_id("turtle-v0.46-midgame-v5"), allow_development=True
    )
    assert sum(np.prod(s.shape) for s in old.observation_space.spaces.values()) == 1699
    assert "shop_attack_bonus" not in GameState().to_dict()
    assert asdict(env.engine.config)["max_shop_tier"] == 4


def test_battle_copies_never_change_permanent_team():
    team = [pet("rooster", 3), pet("shark", 2), pet("turkey", 3), pet("armadillo")]
    before = [asdict(p) for p in team]
    first = resolve_battle(
        team, [pet("rhino", 3), pet("crocodile", 3)], catalog(), random.Random(34)
    )
    second = resolve_battle(
        team, [pet("rhino", 3), pet("crocodile", 3)], catalog(), random.Random(34)
    )
    assert first == second and [asdict(p) for p in team] == before
